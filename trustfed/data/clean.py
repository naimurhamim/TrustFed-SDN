"""Ordered, auditable dataset cleaning pipeline for TrustFed-SDN (Step 01).

Applies the cleaning rules from `docs/DATASETS.md` Section 5 in strict order and
records row and column counts before and after every rule so `results/01_data/cleaning_log.csv`
is 100% reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_MAP_PATH = REPO_ROOT / "configs" / "data" / "label_map.yaml"

# Identifier and leakage-prone columns dropped in Rule 3 (per docs/DATASETS.md Section 5
# and docs/USER_PREFERENCES.md decision to drop port numbers).
IDENTIFIER_AND_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "flow id",
        "src ip",
        "source ip",
        "dst ip",
        "destination ip",
        "timestamp",
        "src port",
        "source port",
        "dst port",
        "destination port",
        "unnamed: 0",
    }
)


@dataclass
class CleaningStepRecord:
    """Audit log row for a single cleaning rule execution."""

    dataset: str
    rule_id: str
    rule_name: str
    rows_before: int
    rows_after: int
    rows_removed: int
    cols_before: int
    cols_after: int
    details: str


def load_label_map(path: Path | None = None) -> dict[str, Any]:
    """Load canonical label mapping YAML configuration."""
    target = path or DEFAULT_LABEL_MAP_PATH
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def map_raw_label_to_canonical(
    raw_label: str,
    dataset_name: str,
    label_map_cfg: dict[str, Any],
) -> str | None:
    """Normalize a single raw label string to its canonical family name."""
    ds_cfg = label_map_cfg["datasets"][dataset_name.lower()]
    mapping: dict[str, str] = ds_cfg["raw_to_canonical"]
    cleaned = str(raw_label).strip()
    if cleaned in mapping:
        return mapping[cleaned]

    # Case-insensitive match
    lower_map = {k.strip().lower(): v for k, v in mapping.items()}
    if cleaned.lower() in lower_map:
        return lower_map[cleaned.lower()]

    # Substring fallback (e.g., corrupted Windows-1252 en-dash in CICIDS2017 "Web Attack  XSS")
    for sub, target in ds_cfg.get("substring_fallbacks", {}).items():
        if sub.lower() in cleaned.lower():
            return target
    return None


def clean_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    label_map_path: Path | None = None,
    drop_ports: bool = True,
) -> tuple[pd.DataFrame, list[CleaningStepRecord], dict[str, Any]]:
    """Run all ordered cleaning rules on `df` and return `(cleaned_df, log_records, metadata)`."""
    ds_key = dataset_name.strip().lower()
    label_cfg = load_label_map(label_map_path)
    records: list[CleaningStepRecord] = []
    work = df.copy()

    # ------------------------------------------------------------------
    # Rule 1: Strip whitespace from column names & resolve duplicate headers
    # ------------------------------------------------------------------
    r1_rows_before, r1_cols_before = len(work), work.shape[1]
    stripped_cols = [str(c).strip() for c in work.columns]
    work.columns = stripped_cols

    dropped_dup_cols: list[str] = []
    keep_col_indices: list[int] = []
    seen_names: set[str] = set()
    for idx, col_name in enumerate(work.columns):
        # Pandas renames duplicate CSV headers as `<name>.1`
        base_name = col_name[:-2].strip() if col_name.endswith(".1") else col_name
        if base_name in seen_names:
            dropped_dup_cols.append(col_name)
            continue
        seen_names.add(base_name)
        keep_col_indices.append(idx)

    if len(keep_col_indices) < work.shape[1]:
        work = work.iloc[:, keep_col_indices].copy()

    records.append(
        CleaningStepRecord(
            dataset=ds_key,
            rule_id="R1",
            rule_name="strip_column_whitespace_and_resolve_duplicate_headers",
            rows_before=r1_rows_before,
            rows_after=len(work),
            rows_removed=0,
            cols_before=r1_cols_before,
            cols_after=work.shape[1],
            details=(
                f"Dropped duplicate header columns: {dropped_dup_cols}"
                if dropped_dup_cols
                else "Stripped column whitespace; no duplicate headers"
            ),
        )
    )

    # Identify the label column
    label_col = next((c for c in work.columns if c.lower() == "label"), work.columns[-1])
    if label_col != "Label":
        work = work.rename(columns={label_col: "Label"})

    # ------------------------------------------------------------------
    # Rule 2: Remove repeated header rows embedded inside CSV files
    # ------------------------------------------------------------------
    r2_rows_before = len(work)
    header_mask = work["Label"].astype(str).str.strip().str.lower() == "label"
    repeated_headers_count = int(header_mask.sum())
    if repeated_headers_count > 0:
        work = work.loc[~header_mask].reset_index(drop=True)

    records.append(
        CleaningStepRecord(
            dataset=ds_key,
            rule_id="R2",
            rule_name="remove_repeated_header_rows",
            rows_before=r2_rows_before,
            rows_after=len(work),
            rows_removed=repeated_headers_count,
            cols_before=work.shape[1],
            cols_after=work.shape[1],
            details=f"Removed {repeated_headers_count} embedded header rows",
        )
    )

    # ------------------------------------------------------------------
    # Rule 3: Drop identifier and leakage-prone columns (Flow ID, IPs, Timestamp, Ports)
    # ------------------------------------------------------------------
    r3_cols_before = work.shape[1]
    drop_set = set(IDENTIFIER_AND_LEAKAGE_COLUMNS)
    if not drop_ports:
        drop_set -= {"src port", "source port", "dst port", "destination port"}

    cols_to_drop = [c for c in work.columns if c.lower() in drop_set and c != "Label"]
    if cols_to_drop:
        work = work.drop(columns=cols_to_drop)

    records.append(
        CleaningStepRecord(
            dataset=ds_key,
            rule_id="R3",
            rule_name="drop_identifier_and_leakage_columns",
            rows_before=len(work),
            rows_after=len(work),
            rows_removed=0,
            cols_before=r3_cols_before,
            cols_after=work.shape[1],
            details=f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}",
        )
    )

    # ------------------------------------------------------------------
    # Rule 4: Replace infinite values with NaN and drop rows with missing values
    # ------------------------------------------------------------------
    r4_rows_before = len(work)
    feature_cols = [c for c in work.columns if c != "Label"]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.to_numeric(work[col], errors="coerce")

    feat_matrix = work[feature_cols].to_numpy(dtype=np.float64, copy=False)
    inf_cells_count = int(np.isinf(feat_matrix).sum())
    if inf_cells_count > 0:
        work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan)

    nan_row_mask = work[feature_cols].isna().any(axis=1) | work["Label"].isna()
    nan_rows_removed = int(nan_row_mask.sum())
    if nan_rows_removed > 0:
        work = work.loc[~nan_row_mask].reset_index(drop=True)

    records.append(
        CleaningStepRecord(
            dataset=ds_key,
            rule_id="R4",
            rule_name="replace_infinite_and_drop_missing_rows",
            rows_before=r4_rows_before,
            rows_after=len(work),
            rows_removed=nan_rows_removed,
            cols_before=work.shape[1],
            cols_after=work.shape[1],
            details=(
                f"Replaced {inf_cells_count} infinite cells with NaN; "
                f"dropped {nan_rows_removed} rows containing NaN"
            ),
        )
    )

    # ------------------------------------------------------------------
    # Rule 5: Normalize labels and build binary + multi-class targets
    # ------------------------------------------------------------------
    r5_rows_before = len(work)
    canonical_labels = [
        map_raw_label_to_canonical(val, ds_key, label_cfg) for val in work["Label"].astype(str)
    ]
    work["label_multi"] = canonical_labels
    unmapped_mask = work["label_multi"].isna()
    unmapped_count = int(unmapped_mask.sum())
    if unmapped_count > 0:
        work = work.loc[~unmapped_mask].reset_index(drop=True)

    ds_cfg = label_cfg["datasets"][ds_key]
    class_order: list[str] = list(ds_cfg["class_order"])
    # Ensure any canonical label present in data is in class_order
    for lbl in sorted(work["label_multi"].unique()):
        if lbl not in class_order:
            class_order.append(lbl)

    class_to_id = {name: idx for idx, name in enumerate(class_order)}
    work["y_multi"] = work["label_multi"].map(class_to_id).astype(np.int64)
    work["y_binary"] = np.where(work["label_multi"] == "Benign", 0, 1).astype(np.int64)
    work = work.drop(columns=["Label"])

    records.append(
        CleaningStepRecord(
            dataset=ds_key,
            rule_id="R5",
            rule_name="normalize_and_map_labels",
            rows_before=r5_rows_before,
            rows_after=len(work),
            rows_removed=unmapped_count,
            cols_before=len(feature_cols) + 1,
            cols_after=len(feature_cols) + 3,
            details=f"Mapped to {len(class_order)} canonical classes: {class_order}",
        )
    )

    # ------------------------------------------------------------------
    # Rule 6: Remove exact duplicate rows & deduplicate feature vectors for zero split leakage
    # ------------------------------------------------------------------
    r6_rows_before = len(work)
    exact_dups = int(work.duplicated(subset=[*feature_cols, "label_multi"]).sum())
    work = work.drop_duplicates(subset=feature_cols, keep="first").reset_index(drop=True)
    r6_removed = r6_rows_before - len(work)

    records.append(
        CleaningStepRecord(
            dataset=ds_key,
            rule_id="R6",
            rule_name="remove_duplicate_feature_rows",
            rows_before=r6_rows_before,
            rows_after=len(work),
            rows_removed=r6_removed,
            cols_before=work.shape[1],
            cols_after=work.shape[1],
            details=(
                f"Removed {r6_removed} duplicate feature rows "
                f"(including {exact_dups} exact feature+label duplicates)"
            ),
        )
    )

    meta = {
        "dataset": ds_key,
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "class_order": class_order,
        "class_to_id": class_to_id,
        "rows_raw": r1_rows_before,
        "rows_cleaned": len(work),
        "class_counts_cleaned": {
            cls: int((work["label_multi"] == cls).sum()) for cls in class_order
        },
        "binary_counts_cleaned": {
            "Benign": int((work["y_binary"] == 0).sum()),
            "Attack": int((work["y_binary"] == 1).sum()),
        },
    }
    return work, records, meta


def save_cleaning_log(
    records: list[CleaningStepRecord],
    out_csv: Path,
    append: bool = False,
) -> pd.DataFrame:
    """Write cleaning step records to CSV (`results/01_data/cleaning_log.csv`)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame([asdict(r) for r in records])
    if append and out_csv.exists():
        prev = pd.read_csv(out_csv)
        prev = prev[prev["dataset"] != df_log["dataset"].iloc[0]]
        df_log = pd.concat([prev, df_log], axis=0, ignore_index=True)
    df_log.to_csv(out_csv, index=False)
    return df_log

"""Dataset loading and raw schema/quality profiling for TrustFed-SDN.

Supports:
- `insdn`: UCD ASEADOS InSDN (2020) SDN intrusion dataset (`Normal_data.csv`,
  `OVS.csv`, `metasploitable-2.csv`).
- `aseados_sdn_iot`: UCD ASEADOS-SDN-IoT (2026) SDN-IoT dataset (`ASEADOS_SDN_IoT.csv`).
- `cicids2017`: UNB CICIDS2017 `MachineLearningCSV` benchmark (8 CSV files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"

SUPPORTED_DATASETS: tuple[str, ...] = ("insdn", "aseados_sdn_iot", "cicids2017")


def list_dataset_csv_files(
    dataset_name: str,
    raw_root: Path | None = None,
) -> list[Path]:
    """Return sorted CSV file paths for a given dataset name."""
    key = dataset_name.strip().lower()
    if key not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Expected one of {SUPPORTED_DATASETS}."
        )
    base = (raw_root or DEFAULT_RAW_ROOT) / key
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {base}")

    csv_files = sorted(p for p in base.rglob("*.csv") if p.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {base}")
    return csv_files


def read_single_csv(
    csv_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Read a single CSV file with UTF-8/Latin-1 fallback and low_memory=False."""
    try:
        return pd.read_csv(
            csv_path,
            nrows=max_rows,
            low_memory=False,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            csv_path,
            nrows=max_rows,
            low_memory=False,
            encoding="latin-1",
        )


def load_raw_dataset(
    dataset_name: str,
    raw_root: Path | None = None,
    max_rows_per_file: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate all raw CSV files for a dataset without mutating values."""
    csv_files = list_dataset_csv_files(dataset_name, raw_root=raw_root)
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = read_single_csv(path, max_rows=max_rows_per_file)
        frames.append(df)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return pd.concat(frames, axis=0, ignore_index=True)


def profile_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    source_files: list[str] | None = None,
) -> dict[str, Any]:
    """Compute schema, class counts, NaN/Inf counts, and duplicate counts for a raw DataFrame."""
    raw_columns = [str(c) for c in df.columns]
    stripped_columns = [str(c).strip() for c in df.columns]

    # Detect columns with leading/trailing whitespace and pandas-renamed duplicate columns (.1)
    whitespace_columns = [
        raw for raw, s in zip(raw_columns, stripped_columns, strict=True) if raw != s
    ]
    duplicate_header_cols = [
        c for c in stripped_columns if c.endswith(".1") or stripped_columns.count(c) > 1
    ]

    # Locate Label column
    label_col = next(
        (raw for raw, s in zip(raw_columns, stripped_columns, strict=True) if s.lower() == "label"),
        raw_columns[-1],
    )

    label_series = df[label_col].astype(str)
    # Detect embedded repeated header rows where the label cell equals "Label"
    repeated_header_rows = int((label_series.str.strip().str.lower() == "label").sum())

    raw_label_counts = {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).items()}
    stripped_label_counts = {
        str(k): int(v) for k, v in label_series.str.strip().value_counts(dropna=False).items()
    }

    missing_cells = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())

    # Count numeric infinite values
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        num_vals = num_df.to_numpy(dtype=np.float64, copy=False)
        inf_mask = np.isinf(num_vals)
        infinite_cells = int(inf_mask.sum())
        rows_with_infinite = int(inf_mask.any(axis=1).sum())
    else:
        infinite_cells = 0
        rows_with_infinite = 0

    exact_duplicate_rows = int(df.duplicated().sum())

    return {
        "dataset": dataset_name,
        "source_files": source_files or [],
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "label_column": label_col.strip(),
        "columns_with_whitespace_count": len(whitespace_columns),
        "duplicate_column_headers": duplicate_header_cols,
        "repeated_header_rows_inside_csv": repeated_header_rows,
        "missing_cells": missing_cells,
        "rows_with_missing": rows_with_missing,
        "infinite_cells": infinite_cells,
        "rows_with_infinite": rows_with_infinite,
        "exact_duplicate_rows": exact_duplicate_rows,
        "raw_label_counts": raw_label_counts,
        "stripped_label_counts": stripped_label_counts,
        "columns": stripped_columns,
    }


def profile_raw_dataset(
    dataset_name: str,
    raw_root: Path | None = None,
    max_rows_per_file: int | None = None,
) -> dict[str, Any]:
    """Load a raw dataset and return its complete schema and data-quality profile."""
    files = list_dataset_csv_files(dataset_name, raw_root=raw_root)
    per_file_rows: dict[str, int] = {}
    frames: list[pd.DataFrame] = []
    for path in files:
        df_part = read_single_csv(path, max_rows=max_rows_per_file)
        per_file_rows[path.name] = int(len(df_part))
        frames.append(df_part)

    full_df = pd.concat(frames, axis=0, ignore_index=True) if len(frames) > 1 else frames[0]
    prof = profile_dataframe(
        full_df,
        dataset_name=dataset_name,
        source_files=[p.name for p in files],
    )
    prof["per_file_rows"] = per_file_rows
    return prof

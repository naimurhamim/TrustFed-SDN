"""Step 01 end-to-end dataset profiling, cleaning, feature selection, partitioning,
and scaling script for TrustFed-SDN.

Saves:
- `results/01_data/profile_<dataset>.json`
- `results/01_data/cleaning_log.csv`
- `configs/data/features_<dataset>.yaml`
- `results/01_data/feature_ranking_<dataset>.csv` & `.png`
- `results/01_data/class_distributions_<dataset>_n3.png` & `_n10.png`
- `results/01_data/summary_<dataset>.json` & `results/01_data/summary.json`
- `data/processed/<dataset>/{splits_and_scaled.npz, partitions.npz, metadata.json}`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trustfed.data.clean import (  # noqa: E402
    CleaningStepRecord,
    clean_dataset,
    load_label_map,
    save_cleaning_log,
)
from trustfed.data.load import (  # noqa: E402
    SUPPORTED_DATASETS,
    list_dataset_csv_files,
    load_raw_dataset,
    profile_dataframe,
)
from trustfed.data.partition import (  # noqa: E402
    fit_and_apply_scalers,
    partition_dirichlet,
    partition_iid,
    partition_natural_non_iid,
    plot_partition_class_distributions,
    save_feature_selection_artifacts,
    select_topk_features_per_class,
    stratified_train_val_test_split,
    verify_zero_split_overlap,
)


def process_single_dataset(
    dataset_name: str,
    raw_root: Path,
    results_dir: Path,
    processed_root: Path,
    configs_data_dir: Path,
    k_per_class: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], list[CleaningStepRecord]]:
    """Run full Step 01 pipeline for one dataset and save all artifacts."""
    t0 = time.perf_counter()
    ds_key = dataset_name.strip().lower()
    print(f"\n[Step 01] Loading raw dataset '{ds_key}'...")
    csv_files = list_dataset_csv_files(ds_key, raw_root=raw_root)
    raw_df = load_raw_dataset(ds_key, raw_root=raw_root)

    print(f"[Step 01] Profiling raw '{ds_key}' ({len(raw_df):,} rows, {raw_df.shape[1]} cols)...")
    raw_profile = profile_dataframe(
        raw_df,
        dataset_name=ds_key,
        source_files=[p.name for p in csv_files],
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    profile_path = results_dir / f"profile_{ds_key}.json"
    profile_path.write_text(json.dumps(raw_profile, indent=2) + "\n", encoding="utf-8")

    print(f"[Step 01] Cleaning '{ds_key}' via ordered rules R1..R6...")
    cleaned_df, clean_records, clean_meta = clean_dataset(raw_df, dataset_name=ds_key)
    del raw_df

    print(
        f"[Step 01] Cleaned '{ds_key}': {clean_meta['rows_raw']:,} -> "
        f"{clean_meta['rows_cleaned']:,} rows ({clean_meta['num_features']} numeric features)."
    )

    print(f"[Step 01] Performing stratified 70/15/15 split (seed={seed}) BEFORE any fitting...")
    splits = stratified_train_val_test_split(
        cleaned_df,
        feature_cols=clean_meta["feature_columns"],
        class_order=clean_meta["class_order"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
    )
    del cleaned_df

    print(f"[Step 01] Verifying zero row-hash leakage across train/val/test for '{ds_key}'...")
    leakage_check = verify_zero_split_overlap(splits.X_train, splits.X_val, splits.X_test)
    if (
        leakage_check["train_val_overlap"] != 0
        or leakage_check["train_test_overlap"] != 0
        or leakage_check["val_test_overlap"] != 0
    ):
        raise RuntimeError(f"Split hash overlap detected in {ds_key}: {leakage_check}")

    print(f"[Step 01] Selecting top-{k_per_class} features per attack class on X_train ONLY...")
    fs_res = select_topk_features_per_class(
        splits.X_train,
        splits.y_mul_train,
        feature_names=splits.feature_names,
        class_order=splits.class_order,
        k_per_class=k_per_class,
        seed=seed,
    )
    feat_yaml = configs_data_dir / f"features_{ds_key}.yaml"
    feat_plot = results_dir / f"feature_ranking_{ds_key}.png"
    save_feature_selection_artifacts(ds_key, fs_res, feat_yaml, feat_plot)
    pd.DataFrame(fs_res["feature_ranking"]).to_csv(
        results_dir / f"feature_ranking_{ds_key}.csv",
        index=False,
    )

    sel_idx = fs_res["selected_feature_indices"]
    X_train_sel = splits.X_train[:, sel_idx]
    X_val_sel = splits.X_val[:, sel_idx]
    X_test_sel = splits.X_test[:, sel_idx]

    label_cfg = load_label_map()
    specialty_map = label_cfg["datasets"][ds_key]["tenant_natural_specialty"]

    print(f"[Step 01] Building natural non-IID, Dirichlet, and IID partitions for '{ds_key}'...")
    nat_train = partition_natural_non_iid(
        splits.y_mul_train,
        class_order=splits.class_order,
        specialty_map=specialty_map,
        primary_share=0.70,
        seed=seed,
    )
    nat_val = partition_natural_non_iid(
        splits.y_mul_val,
        class_order=splits.class_order,
        specialty_map=specialty_map,
        primary_share=0.70,
        seed=seed + 10,
    )

    regimes_n3 = {
        "natural_n3": nat_train,
        "dirichlet_a0.1_n3": partition_dirichlet(splits.y_mul_train, 3, 0.1, seed=seed + 1),
        "dirichlet_a0.5_n3": partition_dirichlet(splits.y_mul_train, 3, 0.5, seed=seed + 2),
        "dirichlet_a1.0_n3": partition_dirichlet(splits.y_mul_train, 3, 1.0, seed=seed + 3),
        "iid_n3": partition_iid(splits.y_mul_train, 3, seed=seed + 4),
    }
    regimes_n5 = {
        "dirichlet_a0.1_n5": partition_dirichlet(splits.y_mul_train, 5, 0.1, seed=seed + 5),
        "dirichlet_a0.5_n5": partition_dirichlet(splits.y_mul_train, 5, 0.5, seed=seed + 6),
        "dirichlet_a1.0_n5": partition_dirichlet(splits.y_mul_train, 5, 1.0, seed=seed + 7),
        "iid_n5": partition_iid(splits.y_mul_train, 5, seed=seed + 8),
    }
    regimes_n10 = {
        "dirichlet_a0.1_n10": partition_dirichlet(splits.y_mul_train, 10, 0.1, seed=seed + 9),
        "dirichlet_a0.5_n10": partition_dirichlet(splits.y_mul_train, 10, 0.5, seed=seed + 10),
        "dirichlet_a1.0_n10": partition_dirichlet(splits.y_mul_train, 10, 1.0, seed=seed + 11),
        "iid_n10": partition_iid(splits.y_mul_train, 10, seed=seed + 12),
    }

    counts_n3 = plot_partition_class_distributions(
        ds_key,
        regimes_n3,
        splits.y_mul_train,
        splits.class_order,
        results_dir / f"class_distributions_{ds_key}_n3.png",
    )
    counts_n10 = plot_partition_class_distributions(
        ds_key,
        regimes_n10,
        splits.y_mul_train,
        splits.class_order,
        results_dir / f"class_distributions_{ds_key}_n10.png",
    )

    print(
        f"[Step 01] Fitting per-tenant local StandardScalers (and global ablation) on '{ds_key}'..."
    )
    scaled_bundle = fit_and_apply_scalers(
        X_train_sel,
        X_val_sel,
        X_test_sel,
        tenant_train_indices=nat_train,
        tenant_val_indices=nat_val,
    )

    out_proc_dir = processed_root / ds_key
    out_proc_dir.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {
        "X_train_raw": X_train_sel.astype(np.float32),
        "y_bin_train": splits.y_bin_train,
        "y_mul_train": splits.y_mul_train,
        "X_val_raw": X_val_sel.astype(np.float32),
        "y_bin_val": splits.y_bin_val,
        "y_mul_val": splits.y_mul_val,
        "X_test_raw": X_test_sel.astype(np.float32),
        "y_bin_test": splits.y_bin_test,
        "y_mul_test": splits.y_mul_test,
        "X_train_global_scaled": scaled_bundle["global_ablation"]["X_train_scaled"],
        "X_val_global_scaled": scaled_bundle["global_ablation"]["X_val_scaled"],
        "X_test_global_scaled": scaled_bundle["global_ablation"]["X_test_scaled"],
        "global_scaler_mean": scaled_bundle["global_ablation"]["mean"],
        "global_scaler_scale": scaled_bundle["global_ablation"]["scale"],
    }
    for t_name, t_data in scaled_bundle["local_per_tenant"].items():
        npz_payload[f"{t_name}_X_train_local_scaled"] = t_data["X_train_scaled"]
        npz_payload[f"{t_name}_X_val_local_scaled"] = t_data["X_val_scaled"]
        npz_payload[f"{t_name}_scaler_mean"] = t_data["mean"]
        npz_payload[f"{t_name}_scaler_scale"] = t_data["scale"]

    np.savez_compressed(out_proc_dir / "splits_and_scaled.npz", **npz_payload)

    part_npz: dict[str, np.ndarray] = {}
    for reg_dict in (regimes_n3, regimes_n5, regimes_n10):
        for reg_name, t_map in reg_dict.items():
            for t_name, idx_arr in t_map.items():
                part_npz[f"{reg_name}__{t_name}"] = idx_arr
    for t_name, idx_arr in nat_val.items():
        part_npz[f"natural_val_n3__{t_name}"] = idx_arr
    np.savez_compressed(out_proc_dir / "partitions.npz", **part_npz)

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "dataset": ds_key,
        "processed_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "elapsed_seconds": elapsed,
        "raw_rows": clean_meta["rows_raw"],
        "cleaned_rows": clean_meta["rows_cleaned"],
        "candidate_features_count": clean_meta["num_features"],
        "selected_features_count": fs_res["num_selected_features"],
        "selected_features": fs_res["selected_features"],
        "class_order": splits.class_order,
        "class_counts_cleaned": clean_meta["class_counts_cleaned"],
        "binary_counts_cleaned": clean_meta["binary_counts_cleaned"],
        "split_sizes": {
            "train": int(len(splits.X_train)),
            "val": int(len(splits.X_val)),
            "test": int(len(splits.X_test)),
        },
        "leakage_hash_check": leakage_check,
        "natural_n3_tenant_train_counts": counts_n3["natural_n3"],
        "natural_n3_tenant_totals": {t: int(len(idx)) for t, idx in nat_train.items()},
        "partition_counts_n3": counts_n3,
        "partition_counts_n10": counts_n10,
        "scaler_verification": {
            "local_scalers_fitted_on": "tenant_train_only",
            "global_ablation_fitted_on": "pooled_train_only",
            "tenant_train_samples_seen": {
                t: int(d["n_samples_seen"]) for t, d in scaled_bundle["local_per_tenant"].items()
            },
        },
    }

    (out_proc_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / f"summary_{ds_key}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[Step 01] Finished '{ds_key}' in {elapsed:.1f}s.")
    return summary, clean_records


def main() -> int:
    """CLI entry point for Step 01 data processing."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run Step 01 dataset profiling and processing.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SUPPORTED_DATASETS),
        help="Dataset names to process (default: insdn aseados_sdn_iot cicids2017).",
    )
    parser.add_argument("--k-per-class", type=int, default=5, help="Top-k features per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    raw_root = REPO_ROOT / "data" / "raw"
    results_dir = REPO_ROOT / "results" / "01_data"
    processed_root = REPO_ROOT / "data" / "processed"
    configs_data_dir = REPO_ROOT / "configs" / "data"

    all_summaries: dict[str, Any] = {}
    all_clean_records: list[CleaningStepRecord] = []

    for ds_name in args.datasets:
        summary, clean_recs = process_single_dataset(
            ds_name,
            raw_root=raw_root,
            results_dir=results_dir,
            processed_root=processed_root,
            configs_data_dir=configs_data_dir,
            k_per_class=args.k_per_class,
            seed=args.seed,
        )
        all_summaries[ds_name] = summary
        all_clean_records.extend(clean_recs)
        save_cleaning_log(
            clean_recs,
            results_dir / "cleaning_log.csv",
            append=True,
        )

    for existing_file in sorted(results_dir.glob("summary_*.json")):
        ds_id = existing_file.stem.removeprefix("summary_")
        if ds_id not in all_summaries:
            all_summaries[ds_id] = json.loads(existing_file.read_text(encoding="utf-8"))

    master_summary = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "k_per_class": args.k_per_class,
        "datasets": all_summaries,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(master_summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\n[Step 01] All datasets processed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

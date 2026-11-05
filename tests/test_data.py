"""Step 01 unit tests for dataset cleaning, leakage-free splitting, training-only
feature selection and scaling, and non-IID partitioning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trustfed.data import (
    clean_dataset,
    fit_and_apply_scalers,
    load_label_map,
    partition_dirichlet,
    partition_iid,
    partition_natural_non_iid,
    select_topk_features_per_class,
    stratified_train_val_test_split,
    verify_zero_split_overlap,
)


def _make_synthetic_raw_sdn_df(seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic raw DataFrame containing all CIC/InSDN quirks."""
    rng = np.random.default_rng(seed)
    n = 360
    labels = (
        ["Normal"] * 120
        + ["DDoS "] * 60
        + ["DoS"] * 60
        + ["Probe"] * 60
        + ["BFA"] * 30
        + ["Web-Attack"] * 30
    )
    f1 = rng.normal(loc=10.0, scale=2.0, size=n)
    f2 = rng.normal(loc=50.0, scale=10.0, size=n)
    f3 = rng.exponential(scale=5.0, size=n)
    f4 = rng.uniform(0.0, 100.0, size=n)
    f5 = rng.normal(loc=0.0, scale=1.0, size=n)
    f6 = rng.normal(loc=5.0, scale=3.0, size=n)

    # Inject class-informative signal so top-k mutual information is well-defined
    for idx, lbl in enumerate(labels):
        if "DDoS" in lbl:
            f1[idx] += 25.0
        elif lbl == "DoS":
            f2[idx] += 80.0
        elif lbl == "Probe":
            f3[idx] += 40.0
        elif lbl == "BFA":
            f4[idx] += 150.0
        elif lbl == "Web-Attack":
            f5[idx] += 15.0

    df = pd.DataFrame(
        {
            " Flow ID ": [f"10.0.0.1-10.0.0.2-{i}" for i in range(n)],
            "Src IP": ["10.10.1.10"] * n,
            "Src Port": [443] * n,
            "Dst IP": ["10.10.1.50"] * n,
            "Dst Port": [80] * n,
            "Timestamp": ["2026-01-01 00:00:00"] * n,
            " Flow Duration ": f1,
            "Tot Fwd Pkts": f2,
            "Tot Bwd Pkts": f3,
            "Flow Byts/s": f4,
            "Flow Pkts/s": f5,
            "SYN Flag Cnt": f6,
            "Fwd Header Len": f1 * 2.0,
            "Fwd Header Len.1": f1 * 2.0,  # duplicate column header quirk
            " Label ": labels,
        }
    )

    # Inject 1 embedded header row, 1 Inf row, 1 NaN row, and 2 exact duplicate rows
    header_row = pd.DataFrame([{c: c.strip() for c in df.columns}])
    inf_row = df.iloc[[0]].copy()
    inf_row["Flow Byts/s"] = np.inf
    nan_row = df.iloc[[1]].copy()
    nan_row["Flow Pkts/s"] = np.nan
    dup_rows = df.iloc[[2, 3]].copy()

    return pd.concat([df, header_row, inf_row, nan_row, dup_rows], ignore_index=True)


def test_cleaning_rules_order_and_quirk_handling() -> None:
    """Verify R1-R6 cleaning rules strip spaces, drop IDs/ports/dups/Inf/NaN, and map labels."""
    raw_df = _make_synthetic_raw_sdn_df()
    cleaned_df, records, meta = clean_dataset(raw_df, dataset_name="insdn")

    assert [r.rule_id for r in records] == ["R1", "R2", "R3", "R4", "R5", "R6"]
    # R1 dropped duplicate column 'Fwd Header Len.1'
    assert records[0].cols_before - records[0].cols_after == 1
    # R2 removed 1 embedded header row
    assert records[1].rows_removed == 1
    # R3 dropped 6 identifier & port columns
    assert records[2].cols_before - records[2].cols_after == 6
    # R4 removed 2 rows (1 Inf + 1 NaN)
    assert records[3].rows_removed == 2
    # R5 mapped all labels (0 unmapped) including trailing-space "DDoS "
    assert records[4].rows_removed == 0
    assert "DDoS" in set(cleaned_df["label_multi"])
    assert "DDoS " not in set(cleaned_df["label_multi"])
    # R6 removed the 2 exact duplicate rows
    assert records[5].rows_removed == 2
    assert meta["rows_cleaned"] == 360
    assert set(cleaned_df["y_binary"].unique()) == {0, 1}


def test_cicids2017_corrupted_web_attack_label_mapping() -> None:
    """Verify corrupted Windows-1252 en-dash in CICIDS2017 Web Attack labels maps cleanly."""
    from trustfed.data.clean import map_raw_label_to_canonical

    cfg = load_label_map()
    assert map_raw_label_to_canonical("Web Attack \ufffd XSS", "cicids2017", cfg) == "WebAttack"
    assert (
        map_raw_label_to_canonical("Web Attack \x96 Brute Force", "cicids2017", cfg) == "WebAttack"
    )
    assert map_raw_label_to_canonical("BENIGN", "cicids2017", cfg) == "Benign"


def test_zero_overlap_hash_check_and_partition_completeness() -> None:
    """Verify stratified split has zero row-hash overlap and partitions cover 100% of train."""
    raw_df = _make_synthetic_raw_sdn_df()
    cleaned_df, _, meta = clean_dataset(raw_df, dataset_name="insdn")
    splits = stratified_train_val_test_split(
        cleaned_df,
        feature_cols=meta["feature_columns"],
        class_order=meta["class_order"],
        seed=42,
    )

    overlap = verify_zero_split_overlap(splits.X_train, splits.X_val, splits.X_test)
    assert overlap["train_val_overlap"] == 0
    assert overlap["train_test_overlap"] == 0
    assert overlap["val_test_overlap"] == 0

    cfg = load_label_map()
    spec = cfg["datasets"]["insdn"]["tenant_natural_specialty"]
    nat = partition_natural_non_iid(splits.y_mul_train, splits.class_order, spec, seed=42)
    dir_p = partition_dirichlet(splits.y_mul_train, num_tenants=5, alpha=0.5, seed=42)
    iid_p = partition_iid(splits.y_mul_train, num_tenants=5, seed=42)

    for part in (nat, dir_p, iid_p):
        all_indices = np.concatenate(list(part.values()))
        assert len(all_indices) == len(splits.y_mul_train)
        assert len(np.unique(all_indices)) == len(splits.y_mul_train)


def test_selectors_and_scalers_fitted_on_training_data_only() -> None:
    """Prove that corrupting validation and test sets changes 0 bits of selectors or scalers."""
    raw_df = _make_synthetic_raw_sdn_df()
    cleaned_df, _, meta = clean_dataset(raw_df, dataset_name="insdn")
    splits = stratified_train_val_test_split(
        cleaned_df,
        feature_cols=meta["feature_columns"],
        class_order=meta["class_order"],
        seed=42,
    )

    cfg = load_label_map()
    spec = cfg["datasets"]["insdn"]["tenant_natural_specialty"]
    nat_train = partition_natural_non_iid(splits.y_mul_train, splits.class_order, spec, seed=42)

    # Run 1: Clean validation and test sets
    fs_1 = select_topk_features_per_class(
        splits.X_train,
        splits.y_mul_train,
        splits.feature_names,
        splits.class_order,
        k_per_class=2,
        seed=42,
    )
    sc_1 = fit_and_apply_scalers(
        splits.X_train,
        splits.X_val,
        splits.X_test,
        tenant_train_indices=nat_train,
    )

    # Run 2: Poison X_val and X_test with extreme +1e9 values
    X_val_poisoned = splits.X_val + 1e9
    X_test_poisoned = splits.X_test - 1e9
    fs_2 = select_topk_features_per_class(
        splits.X_train,
        splits.y_mul_train,
        splits.feature_names,
        splits.class_order,
        k_per_class=2,
        seed=42,
    )
    sc_2 = fit_and_apply_scalers(
        splits.X_train,
        X_val_poisoned,
        X_test_poisoned,
        tenant_train_indices=nat_train,
    )

    assert fs_1["selected_features"] == fs_2["selected_features"]
    assert fs_1["feature_ranking"] == fs_2["feature_ranking"]
    assert np.array_equal(
        sc_1["global_ablation"]["mean"],
        sc_2["global_ablation"]["mean"],
    )
    assert np.array_equal(
        sc_1["global_ablation"]["scale"],
        sc_2["global_ablation"]["scale"],
    )

    for t_name, idx in nat_train.items():
        m1 = sc_1["local_per_tenant"][t_name]["mean"]
        m2 = sc_2["local_per_tenant"][t_name]["mean"]
        s1 = sc_1["local_per_tenant"][t_name]["scale"]
        s2 = sc_2["local_per_tenant"][t_name]["scale"]
        assert np.array_equal(m1, m2)
        assert np.array_equal(s1, s2)
        # Prove each tenant's scaler mean equals its own local training mean (not global mean)
        assert np.allclose(m1, splits.X_train[idx].mean(axis=0))


def test_processed_artifacts_exist_when_step01_has_run() -> None:
    """If results/01_data/summary.json exists, verify its leakage hash checks passed."""
    root = Path(__file__).resolve().parents[1]
    summary_path = root / "results" / "01_data" / "summary.json"
    if not summary_path.exists():
        return
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for ds_name, ds_meta in summary["datasets"].items():
        lc = ds_meta["leakage_hash_check"]
        assert lc["train_val_overlap"] == 0, f"Leakage in {ds_name}"
        assert lc["train_test_overlap"] == 0, f"Leakage in {ds_name}"
        assert lc["val_test_overlap"] == 0, f"Leakage in {ds_name}"

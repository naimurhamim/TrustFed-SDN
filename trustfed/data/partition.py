"""Leakage-free splitting, training-only feature selection, non-IID partitioning,
and per-tenant local scaling for TrustFed-SDN (Step 01).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class SplitBundle:
    """Global stratified train/validation/test arrays before feature selection or scaling."""

    feature_names: list[str]
    class_order: list[str]
    X_train: np.ndarray
    y_bin_train: np.ndarray
    y_mul_train: np.ndarray
    X_val: np.ndarray
    y_bin_val: np.ndarray
    y_mul_val: np.ndarray
    X_test: np.ndarray
    y_bin_test: np.ndarray
    y_mul_test: np.ndarray


def compute_row_hashes(X: np.ndarray) -> set[bytes]:
    """Compute a set of SHA-256 digests over contiguous float64 row bytes of `X`."""
    arr = np.ascontiguousarray(X, dtype=np.float64)
    return {hashlib.sha256(arr[i].tobytes()).digest() for i in range(arr.shape[0])}


def verify_zero_split_overlap(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, int]:
    """Verify zero feature-row hash overlap across train, validation, and test sets."""
    h_train = compute_row_hashes(X_train)
    h_val = compute_row_hashes(X_val)
    h_test = compute_row_hashes(X_test)
    return {
        "train_unique_hashes": len(h_train),
        "val_unique_hashes": len(h_val),
        "test_unique_hashes": len(h_test),
        "train_val_overlap": len(h_train & h_val),
        "train_test_overlap": len(h_train & h_test),
        "val_test_overlap": len(h_val & h_test),
    }


def stratified_train_val_test_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    class_order: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitBundle:
    """Split a cleaned DataFrame into stratified train/val/test sets BEFORE any fitting."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    X = df[feature_cols].to_numpy(dtype=np.float64, copy=True)
    y_bin = df["y_binary"].to_numpy(dtype=np.int64, copy=True)
    y_mul = df["y_multi"].to_numpy(dtype=np.int64, copy=True)

    # Guard for ultra-rare classes (< 3 samples): keep all classes stratifiable
    unique_cls, counts = np.unique(y_mul, return_counts=True)
    rare_classes = set(unique_cls[counts < 3].tolist())
    strat_target = y_mul.copy()
    if rare_classes:
        for rc in rare_classes:
            strat_target[strat_target == rc] = -1

    idx_all = np.arange(len(df))
    idx_train, idx_temp = train_test_split(
        idx_all,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=strat_target if len(np.unique(strat_target)) > 1 else None,
    )

    strat_temp = strat_target[idx_temp]
    u_temp, c_temp = np.unique(strat_temp, return_counts=True)
    for rc in u_temp[c_temp < 2]:
        strat_temp[strat_temp == rc] = -1

    rel_test = test_ratio / (val_ratio + test_ratio)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=rel_test,
        random_state=seed + 1,
        stratify=strat_temp if len(np.unique(strat_temp)) > 1 else None,
    )

    return SplitBundle(
        feature_names=list(feature_cols),
        class_order=list(class_order),
        X_train=X[idx_train],
        y_bin_train=y_bin[idx_train],
        y_mul_train=y_mul[idx_train],
        X_val=X[idx_val],
        y_bin_val=y_bin[idx_val],
        y_mul_val=y_mul[idx_val],
        X_test=X[idx_test],
        y_bin_test=y_bin[idx_test],
        y_mul_test=y_mul[idx_test],
    )


def select_topk_features_per_class(
    X_train: np.ndarray,
    y_mul_train: np.ndarray,
    feature_names: list[str],
    class_order: list[str],
    k_per_class: int = 5,
    max_mi_samples: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    """Select top-k features per attack class using mutual information on training data ONLY."""
    rng = np.random.default_rng(seed)
    n_train, n_features = X_train.shape

    # Drop constant features on X_train first
    train_std = np.std(X_train, axis=0)
    non_const_mask = train_std > 1e-12
    valid_indices = np.where(non_const_mask)[0]

    # Build a deterministic training-only subsample for fast k-NN mutual information
    # while keeping 100% of minority-class training examples.
    if n_train > max_mi_samples:
        selected_rows: list[np.ndarray] = []
        classes, counts = np.unique(y_mul_train, return_counts=True)
        per_class_cap = max(500, max_mi_samples // max(len(classes), 1))
        for cls_id, cnt in zip(classes, counts, strict=True):
            cls_idx = np.where(y_mul_train == cls_id)[0]
            if cnt <= per_class_cap:
                selected_rows.append(cls_idx)
            else:
                chosen = rng.choice(cls_idx, size=per_class_cap, replace=False)
                selected_rows.append(np.sort(chosen))
        sub_idx = np.concatenate(selected_rows)
        X_sub = X_train[sub_idx][:, valid_indices]
        y_sub = y_mul_train[sub_idx]
    else:
        X_sub = X_train[:, valid_indices]
        y_sub = y_mul_train

    per_class_top_features: dict[str, list[dict[str, Any]]] = {}
    aggregate_scores = np.zeros(n_features, dtype=np.float64)
    union_indices: list[int] = []

    for cls_id, cls_name in enumerate(class_order):
        if cls_name == "Benign":
            continue
        binary_ovr = (y_sub == cls_id).astype(np.int64)
        if binary_ovr.sum() < 5:
            continue

        mi_vals = mutual_info_classif(
            X_sub,
            binary_ovr,
            discrete_features=False,
            n_neighbors=3,
            random_state=seed + cls_id,
        )
        full_scores = np.zeros(n_features, dtype=np.float64)
        full_scores[valid_indices] = mi_vals
        aggregate_scores += full_scores

        ranked_idx = np.argsort(-full_scores, kind="mergesort")[:k_per_class]
        per_class_top_features[cls_name] = [
            {
                "feature": feature_names[int(idx)],
                "index": int(idx),
                "mi_score": round(float(full_scores[int(idx)]), 6),
            }
            for idx in ranked_idx
        ]
        for idx in ranked_idx:
            if int(idx) not in union_indices:
                union_indices.append(int(idx))

    # Fallback if a dataset has only 1 attack class and union < k_per_class
    if not union_indices:
        mi_multi = mutual_info_classif(
            X_sub,
            y_sub,
            discrete_features=False,
            n_neighbors=3,
            random_state=seed,
        )
        full_scores = np.zeros(n_features, dtype=np.float64)
        full_scores[valid_indices] = mi_multi
        aggregate_scores = full_scores
        union_indices = [int(i) for i in np.argsort(-full_scores)[: max(k_per_class, 10)]]

    # Sort selected union features by aggregate mutual information descending
    union_indices_sorted = sorted(
        union_indices,
        key=lambda idx: (-float(aggregate_scores[idx]), idx),
    )
    selected_feature_names = [feature_names[idx] for idx in union_indices_sorted]

    ranking_all = [
        {
            "rank": rank + 1,
            "feature": feature_names[int(idx)],
            "index": int(idx),
            "aggregate_mi_score": round(float(aggregate_scores[int(idx)]), 6),
            "selected_in_topk_union": int(idx) in union_indices_sorted,
        }
        for rank, idx in enumerate(np.argsort(-aggregate_scores, kind="mergesort"))
    ]

    return {
        "method": "mutual_info_one_vs_rest_per_attack_class_train_only",
        "fitted_on_split": "train_only",
        "k_per_class": k_per_class,
        "train_rows_used_total": int(n_train),
        "mi_subsample_rows": int(X_sub.shape[0]),
        "num_candidate_features": int(n_features),
        "num_selected_features": len(selected_feature_names),
        "selected_feature_indices": union_indices_sorted,
        "selected_features": selected_feature_names,
        "per_class_top_k": per_class_top_features,
        "feature_ranking": ranking_all,
    }


def save_feature_selection_artifacts(
    dataset_name: str,
    selection_result: dict[str, Any],
    yaml_path: Path,
    plot_path: Path,
) -> None:
    """Save selected feature list YAML (`configs/data/features_*.yaml`) and ranking bar plot."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "method": selection_result["method"],
        "fitted_on_split": selection_result["fitted_on_split"],
        "k_per_class": selection_result["k_per_class"],
        "num_candidate_features": selection_result["num_candidate_features"],
        "num_selected_features": selection_result["num_selected_features"],
        "selected_features": selection_result["selected_features"],
        "per_class_top_k": selection_result["per_class_top_k"],
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    top_items = selection_result["feature_ranking"][
        : min(25, len(selection_result["feature_ranking"]))
    ]
    names = [item["feature"] for item in reversed(top_items)]
    scores = [item["aggregate_mi_score"] for item in reversed(top_items)]
    colors = [
        "#1f77b4" if item["selected_in_topk_union"] else "#aec7e8" for item in reversed(top_items)
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=150)
    ax.barh(names, scores, color=colors)
    ax.set_xlabel("Aggregate One-vs-Rest Mutual Information (Train Only)")
    n_sel = selection_result["num_selected_features"]
    k_cls = selection_result["k_per_class"]
    ax.set_title(
        f"{dataset_name.upper()} — Top Feature Ranking "
        f"({n_sel} Selected via Top-{k_cls}/Class Union)"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def partition_natural_non_iid(
    y_mul: np.ndarray,
    class_order: list[str],
    specialty_map: dict[str, list[str]],
    primary_share: float = 0.70,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Partition sample indices across 3 domain tenants (Finance, Healthcare, Education)."""
    rng = np.random.default_rng(seed)
    tenant_names = list(specialty_map.keys())
    n_tenants = len(tenant_names)
    tenant_buckets: dict[str, list[np.ndarray]] = {t: [] for t in tenant_names}

    # Map each attack class to its specialist tenant (if any)
    class_to_specialist: dict[str, str] = {}
    for t_name, fam_list in specialty_map.items():
        for fam in fam_list:
            class_to_specialist[fam] = t_name

    for cls_id, cls_name in enumerate(class_order):
        cls_indices = np.where(y_mul == cls_id)[0].copy()
        if len(cls_indices) == 0:
            continue
        rng.shuffle(cls_indices)

        if cls_name == "Benign" or cls_name not in class_to_specialist:
            # Distribute Benign (or unassigned classes) evenly across all 3 tenants
            splits = np.array_split(cls_indices, n_tenants)
            for t_name, chunk in zip(tenant_names, splits, strict=True):
                tenant_buckets[t_name].append(chunk)
        else:
            specialist = class_to_specialist[cls_name]
            others = [t for t in tenant_names if t != specialist]
            n_primary = int(round(len(cls_indices) * primary_share))
            primary_chunk = cls_indices[:n_primary]
            remainder = cls_indices[n_primary:]
            tenant_buckets[specialist].append(primary_chunk)
            other_splits = np.array_split(remainder, len(others))
            for o_name, chunk in zip(others, other_splits, strict=True):
                tenant_buckets[o_name].append(chunk)

    result: dict[str, np.ndarray] = {}
    for t_name in tenant_names:
        merged = (
            np.concatenate(tenant_buckets[t_name])
            if tenant_buckets[t_name]
            else np.empty(0, dtype=np.int64)
        )
        rng.shuffle(merged)
        result[t_name] = merged.astype(np.int64)
    return result


def partition_dirichlet(
    y_mul: np.ndarray,
    num_tenants: int,
    alpha: float,
    min_samples_per_tenant: int = 16,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Partition sample indices across `num_tenants` using a class-wise Dirichlet(alpha) draw."""
    if alpha <= 0:
        raise ValueError("Dirichlet concentration alpha must be > 0")
    rng = np.random.default_rng(seed)
    classes = np.unique(y_mul)
    tenant_names = [f"tenant_{i + 1:02d}" for i in range(num_tenants)]

    for _attempt in range(25):
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_tenants)]
        for cls_id in classes:
            cls_idx = np.where(y_mul == cls_id)[0].copy()
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_tenants, alpha, dtype=np.float64))
            cut_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            splits = np.split(cls_idx, cut_points)
            for t_idx, chunk in enumerate(splits):
                buckets[t_idx].append(chunk)

        merged = [np.concatenate(b) if b else np.empty(0, dtype=np.int64) for b in buckets]
        if min(len(m) for m in merged) >= min(
            min_samples_per_tenant, len(y_mul) // max(num_tenants * 2, 1)
        ):
            for m in merged:
                rng.shuffle(m)
            return {name: m.astype(np.int64) for name, m in zip(tenant_names, merged, strict=True)}

    # Deterministic fallback: ensure every tenant has at least min_samples_per_tenant
    all_idx = np.arange(len(y_mul))
    rng.shuffle(all_idx)
    splits = np.array_split(all_idx, num_tenants)
    return {name: s.astype(np.int64) for name, s in zip(tenant_names, splits, strict=True)}


def partition_iid(
    y_mul: np.ndarray,
    num_tenants: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified IID partition across `num_tenants`."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y_mul)
    tenant_names = [f"tenant_{i + 1:02d}" for i in range(num_tenants)]
    buckets: list[list[np.ndarray]] = [[] for _ in range(num_tenants)]

    for cls_id in classes:
        cls_idx = np.where(y_mul == cls_id)[0].copy()
        rng.shuffle(cls_idx)
        splits = np.array_split(cls_idx, num_tenants)
        for t_idx, chunk in enumerate(splits):
            buckets[t_idx].append(chunk)

    result: dict[str, np.ndarray] = {}
    for name, b in zip(tenant_names, buckets, strict=True):
        merged = np.concatenate(b) if b else np.empty(0, dtype=np.int64)
        rng.shuffle(merged)
        result[name] = merged.astype(np.int64)
    return result


def fit_and_apply_scalers(
    X_train_full: np.ndarray,
    X_val_full: np.ndarray,
    X_test_full: np.ndarray,
    tenant_train_indices: dict[str, np.ndarray],
    tenant_val_indices: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Fit privacy-preserving per-tenant local StandardScalers (on each tenant's X_train ONLY)
    plus the global-scaler ablation (fitted on pooled X_train ONLY).
    """
    local_scalers: dict[str, dict[str, Any]] = {}
    for t_name, tr_idx in tenant_train_indices.items():
        scaler = StandardScaler()
        X_t_train = X_train_full[tr_idx]
        X_t_train_scaled = scaler.fit_transform(X_t_train).astype(np.float32)

        entry: dict[str, Any] = {
            "scaler": scaler,
            "mean": scaler.mean_.copy(),
            "scale": scaler.scale_.copy(),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "X_train_scaled": X_t_train_scaled,
        }
        if tenant_val_indices and t_name in tenant_val_indices:
            X_t_val = X_val_full[tenant_val_indices[t_name]]
            entry["X_val_scaled"] = scaler.transform(X_t_val).astype(np.float32)
        local_scalers[t_name] = entry

    global_scaler = StandardScaler()
    X_train_global_scaled = global_scaler.fit_transform(X_train_full).astype(np.float32)
    X_val_global_scaled = global_scaler.transform(X_val_full).astype(np.float32)
    X_test_global_scaled = global_scaler.transform(X_test_full).astype(np.float32)

    return {
        "local_per_tenant": local_scalers,
        "global_ablation": {
            "scaler": global_scaler,
            "mean": global_scaler.mean_.copy(),
            "scale": global_scaler.scale_.copy(),
            "n_samples_seen": int(global_scaler.n_samples_seen_),
            "X_train_scaled": X_train_global_scaled,
            "X_val_scaled": X_val_global_scaled,
            "X_test_scaled": X_test_global_scaled,
        },
    }


def plot_partition_class_distributions(
    dataset_name: str,
    partitions: dict[str, dict[str, np.ndarray]],
    y_mul_train: np.ndarray,
    class_order: list[str],
    out_path: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    """Plot per-tenant class distributions for every partition regime and return count table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_regimes = len(partitions)
    fig, axes = plt.subplots(1, n_regimes, figsize=(4.6 * n_regimes, 5.2), dpi=150, sharey=False)
    if n_regimes == 1:
        axes = [axes]

    summary_counts: dict[str, dict[str, dict[str, int]]] = {}
    cmap = plt.get_cmap("tab10")

    for ax, (regime_name, tenant_map) in zip(axes, partitions.items(), strict=True):
        tenant_names = list(tenant_map.keys())
        summary_counts[regime_name] = {}
        bottom = np.zeros(len(tenant_names), dtype=np.float64)

        for cls_id, cls_name in enumerate(class_order):
            vals = []
            for t_name in tenant_names:
                t_idx = tenant_map[t_name]
                cnt = int((y_mul_train[t_idx] == cls_id).sum())
                summary_counts[regime_name].setdefault(t_name, {})[cls_name] = cnt
                vals.append(cnt)
            vals_arr = np.array(vals, dtype=np.float64)
            ax.bar(
                tenant_names,
                vals_arr,
                bottom=bottom,
                label=cls_name,
                color=cmap(cls_id % 10),
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += vals_arr

        ax.set_title(f"{regime_name}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Training Flows" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(class_order), 5), fontsize=8.5)
    fig.suptitle(
        f"{dataset_name.upper()} — Per-Tenant Training Class Distributions",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path)
    plt.close(fig)
    return summary_counts

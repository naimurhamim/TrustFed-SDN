"""Step 03 — Federated Learning Simulation Runner and FedAvg Benchmark (`sim/run_experiment.py`).

Usage:
  1. Run a single experiment config:
     `python sim/run_experiment.py --config configs/experiments/fedavg_insdn_natural_n3.yaml`
  2. Run the full Step 03 FedAvg benchmark sweep (all partitions, N in {3, 5, 10}, 5 seeds):
     `python sim/run_experiment.py --sweep-step03`
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trustfed.aggregation import build_aggregator as _build_agg_registry  # noqa: E402
from trustfed.attacks import build_attack, compute_backdoor_asr  # noqa: E402
from trustfed.data.clean import load_label_map  # noqa: E402
from trustfed.data.partition import (  # noqa: E402
    partition_dirichlet,
    partition_iid,
    partition_natural_non_iid,
)
from trustfed.fl import (  # noqa: E402
    Aggregator,
    ClientUpdate,
    FedAvgAggregator,
    FLClient,
    FLServer,
)
from trustfed.metrics import summarize_seed_metrics  # noqa: E402
from trustfed.models import (  # noqa: E402
    SequenceMode,
    build_model,
    get_default_device,
    set_all_seeds,
)
from trustfed.privacy import (  # noqa: E402
    DPConfig,
    DPFLClient,
    allocate_budgets,
    compute_sensitivity_scores,
    evaluate_membership_inference,
)
from trustfed.verify import (  # noqa: E402
    DEFAULT_SCALE_BITS,
    create_signed_submission,
    generate_tenant_keys,
    measure_quantization_error,
    server_secure_aggregate,
    tamper_dropped_tenant,
    tamper_modified_aggregate,
    tamper_replayed_update,
    tamper_substituted_update,
    verify_aggregate,
)

SEEDS_DEFAULT: tuple[int, ...] = (0, 1, 2, 3, 4)


def build_aggregator(name: str, **kwargs: Any) -> Aggregator:
    """Instantiate an `Aggregator` by name (`fedavg`, `static_weight`, `trimmed_mean_*`, `median`, `clustering`)."""
    return _build_agg_registry(name, **kwargs)


def load_dataset_artifacts(
    dataset_name: str,
    processed_root: Path | None = None,
    results_01_dir: Path | None = None,
) -> dict[str, Any]:
    """Load `splits_and_scaled.npz`, `partitions.npz`, and `metadata.json`."""
    proc_root = processed_root or (REPO_ROOT / "data" / "processed")
    r01_root = results_01_dir or (REPO_ROOT / "results" / "01_data")

    ds_dir = proc_root / dataset_name
    npz_path = ds_dir / "splits_and_scaled.npz"
    part_path = ds_dir / "partitions.npz"
    meta_path = ds_dir / "metadata.json"
    sum_path = meta_path if meta_path.exists() else (r01_root / f"summary_{dataset_name}.json")

    if not npz_path.exists() or not part_path.exists():
        raise FileNotFoundError(
            f"Processed dataset '{dataset_name}' not found at {ds_dir}. "
            "Run `python scripts/profile_dataset.py` first."
        )

    splits_npz = dict(np.load(npz_path))
    part_npz = dict(np.load(part_path))
    summary_01 = json.loads(sum_path.read_text(encoding="utf-8"))
    label_cfg = load_label_map()
    specialty_map = label_cfg["datasets"][dataset_name]["tenant_natural_specialty"]
    class_order: list[str] = summary_01["class_order"]

    return {
        "dataset": dataset_name,
        "splits": splits_npz,
        "partitions": part_npz,
        "summary_01": summary_01,
        "class_order": class_order,
        "specialty_map": specialty_map,
    }


def resolve_tenant_partitions(
    bundle: dict[str, Any],
    partition_type: str,
    num_tenants: int = 3,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return `(train_tenant_indices, test_tenant_indices)` for the requested partition regime."""
    ptype = partition_type.strip().lower()
    sp = bundle["splits"]
    part_npz = bundle["partitions"]
    y_mul_train = sp["y_mul_train"]
    y_mul_test = sp["y_mul_test"]

    if ptype == "natural":
        train_map_3: dict[str, np.ndarray] = {}
        prefix = "natural_n3__"
        for k, v in part_npz.items():
            if k.startswith(prefix):
                train_map_3[k[len(prefix) :]] = v.astype(np.int64)
        if not train_map_3:
            train_map_3 = partition_natural_non_iid(
                y_mul_train,
                class_order=bundle["class_order"],
                specialty_map=bundle["specialty_map"],
                primary_share=0.70,
                seed=seed,
            )
        test_map_3 = partition_natural_non_iid(
            y_mul_test,
            class_order=bundle["class_order"],
            specialty_map=bundle["specialty_map"],
            primary_share=0.70,
            seed=seed + 99,
        )
        if int(num_tenants) == 3:
            return train_map_3, test_map_3
        if int(num_tenants) == 10:
            # Subdivide the 3 domain specialties into 10 domain-specialized tenants (3 Finance, 4 Healthcare, 3 Education)
            sub_counts = [3, 4, 3]
            train_10: dict[str, np.ndarray] = {}
            test_10: dict[str, np.ndarray] = {}
            rng_tr = np.random.default_rng(seed + 313)
            rng_te = np.random.default_rng(seed + 419)
            t_idx_counter = 1
            for (_, tr_arr), (_, te_arr), n_sub in zip(
                sorted(train_map_3.items()),
                sorted(test_map_3.items()),
                sub_counts,
                strict=True,
            ):
                tr_perm = rng_tr.permutation(tr_arr)
                te_perm = rng_te.permutation(te_arr)
                tr_chunks = np.array_split(tr_perm, n_sub)
                te_chunks = np.array_split(te_perm, n_sub)
                for c_tr, c_te in zip(tr_chunks, te_chunks, strict=True):
                    t_key = f"tenant_{t_idx_counter:02d}"
                    train_10[t_key] = np.sort(c_tr.astype(np.int64))
                    test_10[t_key] = np.sort(c_te.astype(np.int64))
                    t_idx_counter += 1
            return train_10, test_10
        raise ValueError(
            f"Natural domain-specialty partition supports num_tenants in {{3, 10}}, got {num_tenants}."
        )

    if ptype == "dirichlet":
        alpha_str = f"{float(dirichlet_alpha):.1f}"
        prefix = f"dirichlet_a{alpha_str}_n{int(num_tenants)}__"
        train_map = {}
        for k, v in part_npz.items():
            if k.startswith(prefix):
                train_map[k[len(prefix) :]] = v.astype(np.int64)
        if not train_map:
            train_map = partition_dirichlet(
                y_mul_train,
                num_tenants=int(num_tenants),
                alpha=float(dirichlet_alpha),
                seed=seed,
            )
        test_map = partition_dirichlet(
            y_mul_test,
            num_tenants=int(num_tenants),
            alpha=float(dirichlet_alpha),
            seed=seed + 99,
        )
        return train_map, test_map

    if ptype == "iid":
        prefix = f"iid_n{int(num_tenants)}__"
        train_map = {}
        for k, v in part_npz.items():
            if k.startswith(prefix):
                train_map[k[len(prefix) :]] = v.astype(np.int64)
        if not train_map:
            train_map = partition_iid(
                y_mul_train,
                num_tenants=int(num_tenants),
                seed=seed,
            )
        test_map = partition_iid(
            y_mul_test,
            num_tenants=int(num_tenants),
            seed=seed + 99,
        )
        return train_map, test_map

    raise ValueError(
        f"Unknown partition_type '{partition_type}'. Expected 'natural', 'dirichlet', or 'iid'."
    )


def subsample_tenant_indices_stratified(
    tenant_indices: dict[str, np.ndarray],
    y_mul_train: np.ndarray,
    max_total_samples: int | None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified per-tenant training cap that preserves rare attack classes and tenant ratios."""
    if max_total_samples is None or max_total_samples <= 0:
        return tenant_indices
    total_current = sum(len(idx) for idx in tenant_indices.values())
    if total_current <= max_total_samples:
        return tenant_indices

    rng = np.random.default_rng(seed)
    ratio = float(max_total_samples) / float(total_current)
    subsampled: dict[str, np.ndarray] = {}

    for t_idx, (t_name, indices) in enumerate(tenant_indices.items()):
        target_n = max(64, int(round(len(indices) * ratio)))
        if len(indices) <= target_n:
            subsampled[t_name] = indices
            continue

        y_local = y_mul_train[indices]
        classes, counts = np.unique(y_local, return_counts=True)
        keep_local: list[np.ndarray] = []
        rare_cutoff = max(250, target_n // max(len(classes) * 2, 2))
        rem_budget = target_n
        large_classes: list[int] = []

        for cls_id, cnt in zip(classes, counts, strict=True):
            c_pos = np.flatnonzero(y_local == cls_id)
            if cnt <= rare_cutoff:
                keep_local.append(indices[c_pos])
                rem_budget -= len(c_pos)
            else:
                large_classes.append(int(cls_id))

        rem_budget = max(len(large_classes) * 16, rem_budget)
        large_total = sum(int(np.sum(y_local == c)) for c in large_classes)
        for cls_id in large_classes:
            c_pos = np.flatnonzero(y_local == cls_id)
            share = max(16, int(round(rem_budget * (len(c_pos) / max(1, large_total)))))
            share = min(share, len(c_pos))
            chosen = rng.choice(indices[c_pos], size=share, replace=False)
            keep_local.append(np.sort(chosen))

        merged = np.concatenate(keep_local) if keep_local else indices[:target_n]
        rng_t = np.random.default_rng(seed + t_idx + 1)
        rng_t.shuffle(merged)
        subsampled[t_name] = merged.astype(np.int64)

    return subsampled


def run_single_seed_fl(
    bundle: dict[str, Any],
    *,
    task: str = "binary",
    partition_type: str = "natural",
    num_tenants: int = 3,
    dirichlet_alpha: float = 0.5,
    rounds: int = 30,
    local_epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    model_cfg: dict[str, Any] | None = None,
    aggregator_name: str = "fedavg",
    aggregator_kwargs: dict[str, Any] | None = None,
    attack_name: str = "none",
    poison_fraction: float = 0.0,
    attack_kwargs: dict[str, Any] | None = None,
    scaler_mode: str = "global",
    max_train_samples: int | None = None,
    eval_every: int = 1,
    eval_per_tenant_every: int = 5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run one multi-round FL simulation for a single random seed and return full results."""
    if device is None:
        device = get_default_device(prefer_gpu=True)

    mcfg = model_cfg or {}
    model_type = str(mcfg.get("type", "gru"))
    hidden_dim = int(mcfg.get("hidden_dim", 64))
    num_layers = int(mcfg.get("num_layers", 2))
    dropout = float(mcfg.get("dropout", 0.1))
    seq_mode: SequenceMode = mcfg.get("seq_mode", "single_step")
    num_chunks = int(mcfg.get("num_chunks", 1))

    sp = bundle["splits"]
    is_binary = task.strip().lower() == "binary"
    y_train_all = sp["y_bin_train"] if is_binary else sp["y_mul_train"]
    y_val_all = sp["y_bin_val"] if is_binary else sp["y_mul_val"]
    y_test_all = sp["y_bin_test"] if is_binary else sp["y_mul_test"]
    class_names = ["Benign", "Attack"] if is_binary else list(bundle["class_order"])
    num_classes = len(class_names)

    train_map_raw, test_map = resolve_tenant_partitions(
        bundle,
        partition_type=partition_type,
        num_tenants=num_tenants,
        dirichlet_alpha=dirichlet_alpha,
        seed=42,
    )
    train_map = subsample_tenant_indices_stratified(
        train_map_raw,
        sp["y_mul_train"],
        max_total_samples=max_train_samples,
        seed=42 + seed,
    )

    if scaler_mode == "local_per_tenant":
        X_val_global = sp["X_val_global_scaled"]
        X_test_global = sp["X_test_global_scaled"]
    else:
        X_train_global = sp["X_train_global_scaled"]
        X_val_global = sp["X_val_global_scaled"]
        X_test_global = sp["X_test_global_scaled"]

    input_dim = int(sp["X_train_global_scaled"].shape[1])

    # Seed global model initialization deterministically
    set_all_seeds(seed)
    global_model = build_model(
        model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        seq_mode=seq_mode,
        num_chunks=num_chunks,
    ).to(device)
    init_flat = global_model.get_flat_params()

    clients: list[FLClient] = []
    tenant_names = sorted(train_map.keys())
    n_total_clients = len(tenant_names)
    n_malicious = int(np.round(float(poison_fraction) * n_total_clients))
    n_malicious = max(0, min(n_total_clients, n_malicious))
    if str(attack_name).strip().lower() in ("", "none", "clean", "honest"):
        n_malicious = 0

    rng_mal = np.random.default_rng(int(seed) * 1009 + 17)
    malicious_indices = (
        set(rng_mal.choice(n_total_clients, size=n_malicious, replace=False).tolist())
        if n_malicious > 0
        else set()
    )
    atk_cfg = dict(attack_kwargs or {})

    for idx, t_name in enumerate(tenant_names):
        tr_indices = train_map[t_name]
        te_indices = test_map.get(t_name, np.empty(0, dtype=np.int64))

        if scaler_mode == "local_per_tenant":
            local_scaler = StandardScaler()
            X_t_train = local_scaler.fit_transform(sp["X_train_raw"][tr_indices]).astype(np.float32)
            X_t_test = (
                local_scaler.transform(sp["X_test_raw"][te_indices]).astype(np.float32)
                if len(te_indices) > 0
                else None
            )
        else:
            X_t_train = X_train_global[tr_indices]
            X_t_test = X_test_global[te_indices] if len(te_indices) > 0 else None

        y_t_train = y_train_all[tr_indices]
        y_t_test = y_test_all[te_indices] if len(te_indices) > 0 else None

        c_model = build_model(
            model_type,
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            seq_mode=seq_mode,
            num_chunks=num_chunks,
        ).to(device)
        c_model.set_flat_params(init_flat)

        client_attack = build_attack(attack_name, **atk_cfg) if idx in malicious_indices else None

        clients.append(
            FLClient(
                client_id=t_name,
                client_index=idx,
                model=c_model,
                X_train=X_t_train,
                y_train=y_t_train,
                X_test=X_t_test,
                y_test=y_t_test,
                device=device,
                use_class_weights=not is_binary,
                attack=client_attack,
            )
        )

    agg_cfg = dict(aggregator_kwargs or {})
    if aggregator_name.strip().lower() in ("static_weight", "static", "static_weights"):
        agg_cfg.setdefault("val_model", global_model)
        agg_cfg.setdefault("X_val", X_val_global)
        agg_cfg.setdefault("y_val", y_val_all)
    aggregator = build_aggregator(aggregator_name, **agg_cfg)
    server = FLServer(
        global_model=global_model,
        clients=clients,
        aggregator=aggregator,
        X_test=X_test_global,
        y_test=y_test_all,
        X_val=X_val_global,
        y_val=y_val_all,
        class_names=class_names,
        device=device,
    )

    fit_res = server.fit(
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        eval_every=eval_every,
        eval_per_tenant_every=eval_per_tenant_every,
    )

    # Always compute Backdoor Attack Success Rate (ASR) on held-out test attack samples
    trig_feats = tuple(int(c) for c in atk_cfg.get("trigger_features", (3, 6, 8)))
    trig_val = float(atk_cfg.get("trigger_value", 4.5))
    asr_stats = compute_backdoor_asr(
        server.global_model,
        X_test_global,
        y_test_all,
        trigger_features=trig_feats,
        trigger_value=trig_val,
        target_class=0,
        device=device,
    )
    fit_res["final_metrics"]["backdoor_asr"] = float(asr_stats["backdoor_asr"])
    fit_res["final_metrics"]["clean_evasion_rate"] = float(asr_stats["clean_evasion_rate"])
    fit_res["final_metrics"]["asr_gain"] = float(asr_stats["asr_gain"])
    fit_res["final_metrics"]["mean_honest_false_rejection_rate"] = float(
        fit_res["mean_honest_false_rejection_rate"]
    )
    fit_res["final_metrics"]["mean_malicious_rejection_rate"] = float(
        fit_res["mean_malicious_rejection_rate"]
    )

    fit_res["dataset"] = bundle["dataset"]
    fit_res["task"] = task
    fit_res["partition_type"] = partition_type
    fit_res["num_tenants"] = int(num_tenants)
    fit_res["dirichlet_alpha"] = float(dirichlet_alpha) if partition_type == "dirichlet" else None
    fit_res["scaler_mode"] = scaler_mode
    fit_res["attack_name"] = str(attack_name)
    fit_res["poison_fraction"] = float(poison_fraction)
    fit_res["num_malicious_clients"] = int(n_malicious)
    fit_res["global_final_flat_params"] = server.get_global_flat_params().cpu()
    return fit_res


def summarize_multi_seed_fl(
    seed_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate final metrics, system overhead, and round curves across seeds."""
    if not seed_runs:
        raise ValueError("seed_runs cannot be empty.")

    final_metrics_list: list[dict[str, Any]] = []
    for sr in seed_runs:
        fm = dict(sr["final_metrics"])
        fm["total_wall_time_sec"] = sr["total_wall_time_sec"]
        fm["mean_round_local_train_time_sec"] = sr["mean_round_local_train_time_sec"]
        fm["mean_round_aggregation_time_sec"] = sr["mean_round_aggregation_time_sec"]
        fm["bytes_per_round"] = float(sr["bytes_per_round"])
        fm["total_comm_bytes"] = float(sr["total_comm_bytes"])
        fm["round_to_95pct_f1"] = float(sr["round_to_95pct_f1"])
        final_metrics_list.append(fm)

    metric_keys = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_attack",
        "attack_detection_rate",
        "false_positive_rate",
        "roc_auc",
        "tenant_f1_macro_mean",
        "tenant_f1_macro_std",
        "tenant_f1_macro_min",
        "backdoor_asr",
        "clean_evasion_rate",
        "asr_gain",
        "mean_honest_false_rejection_rate",
        "mean_malicious_rejection_rate",
        "mean_round_local_train_time_sec",
        "mean_round_aggregation_time_sec",
        "bytes_per_round",
        "total_comm_bytes",
        "round_to_95pct_f1",
    )
    summary_stats = summarize_seed_metrics(final_metrics_list, scalar_keys=metric_keys)

    # Round-by-round mean & std curves for convergence plots
    rounds_map: dict[int, list[dict[str, Any]]] = {}
    for sr in seed_runs:
        for h in sr["history"]:
            r_idx = int(h["round"])
            rounds_map.setdefault(r_idx, []).append(h)

    round_curves: list[dict[str, Any]] = []
    for r_idx in sorted(rounds_map.keys()):
        items = rounds_map[r_idx]
        losses = [float(x["train_loss_weighted"]) for x in items]
        entry: dict[str, Any] = {
            "round": r_idx,
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_std": float(np.std(losses, ddof=0)),
            "local_train_time_sec_mean": float(
                np.mean([float(x["local_train_time_sec"]) for x in items])
            ),
            "aggregation_time_sec_mean": float(
                np.mean([float(x["aggregation_time_sec"]) for x in items])
            ),
            "total_comm_bytes": int(items[0]["total_comm_bytes"]),
            "cumulative_comm_bytes": int(items[0]["cumulative_comm_bytes"]),
        }
        f1s = [float(x["f1_macro"]) for x in items if "f1_macro" in x]
        accs = [float(x["accuracy"]) for x in items if "accuracy" in x]
        adrs = [float(x["attack_detection_rate"]) for x in items if "attack_detection_rate" in x]
        fprs = [float(x["false_positive_rate"]) for x in items if "false_positive_rate" in x]
        tf1s = [float(x["tenant_f1_macro_mean"]) for x in items if "tenant_f1_macro_mean" in x]
        if f1s:
            entry["f1_macro_mean"] = float(np.mean(f1s))
            entry["f1_macro_std"] = float(np.std(f1s, ddof=0))
            entry["accuracy_mean"] = float(np.mean(accs))
            entry["accuracy_std"] = float(np.std(accs, ddof=0))
            entry["adr_mean"] = float(np.mean(adrs))
            entry["fpr_mean"] = float(np.mean(fprs))
        if tf1s:
            entry["tenant_f1_macro_mean"] = float(np.mean(tf1s))
            entry["tenant_f1_macro_std"] = float(np.std(tf1s, ddof=0))
        round_curves.append(entry)

    first = seed_runs[0]
    return {
        "dataset": first["dataset"],
        "task": first["task"],
        "partition_type": first["partition_type"],
        "num_tenants": int(first["num_tenants"]),
        "dirichlet_alpha": first["dirichlet_alpha"],
        "scaler_mode": first["scaler_mode"],
        "aggregator": first["aggregator"],
        "attack_name": first.get("attack_name", "none"),
        "poison_fraction": float(first.get("poison_fraction", 0.0)),
        "num_malicious_clients": int(first.get("num_malicious_clients", 0)),
        "rounds": int(first["rounds"]),
        "local_epochs": int(first["local_epochs"]),
        "batch_size": int(first["batch_size"]),
        "lr": float(first["lr"]),
        "param_count": int(first["param_count"]),
        "num_seeds": len(seed_runs),
        "seeds": [int(sr["seed"]) for sr in seed_runs],
        "metrics": summary_stats,
        "round_curves": round_curves,
    }


def run_experiment_from_config(
    config: dict[str, Any],
    *,
    processed_root: Path | None = None,
    results_01_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute a single FL experiment configuration (across configured seeds) and write outputs."""
    if config.get("mode") == "step03_sweep":
        out_dir = REPO_ROOT / str(config.get("output_dir", "results/03_fedavg"))
        seeds = tuple(int(s) for s in config.get("seeds", SEEDS_DEFAULT))
        return run_step03_full_sweep(
            out_dir=out_dir,
            seeds=seeds,
            rounds=int(config.get("rounds", 30)),
            local_epochs=int(config.get("local_epochs", 3)),
            batch_size=int(config.get("batch_size", 128)),
            lr=float(config.get("lr", 1e-3)),
            max_train_samples=int(config.get("max_train_samples", 24000)),
            device=device,
        )

    if config.get("mode") == "step05_sweep":
        out_dir = REPO_ROOT / str(config.get("output_dir", "results/05_dp"))
        seeds = tuple(int(s) for s in config.get("seeds", SEEDS_DEFAULT))
        return run_step05_full_sweep(
            out_dir=out_dir,
            seeds=seeds,
            rounds=int(config.get("rounds", 12)),
            local_epochs=int(config.get("local_epochs", 1)),
            expected_batch_size=int(config.get("expected_batch_size", 512)),
            device=device,
        )

    if config.get("mode") == "step06_sweep":
        out_dir = REPO_ROOT / str(config.get("output_dir", "results/06_verify"))
        seeds = tuple(int(s) for s in config.get("seeds", SEEDS_DEFAULT))
        return run_step06_full_sweep(
            out_dir=out_dir,
            seeds=seeds,
            rounds=int(config.get("rounds", 30)),
            local_epochs=int(config.get("local_epochs", 3)),
            batch_size=int(config.get("batch_size", 128)),
            lr=float(config.get("lr", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            max_train_samples=int(config.get("max_train_samples", 24000)),
            device=device,
        )

    dataset_name = str(config.get("dataset", "insdn"))
    task = str(config.get("task", "binary"))
    partition_type = str(config.get("partition_type", "natural"))
    num_tenants = int(config.get("num_tenants", 3))
    dirichlet_alpha = float(config.get("dirichlet_alpha", 0.5))
    rounds = int(config.get("rounds", 30))
    local_epochs = int(config.get("local_epochs", 3))
    batch_size = int(config.get("batch_size", 128))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    seeds = [int(s) for s in config.get("seeds", list(SEEDS_DEFAULT))]
    model_cfg = dict(
        config.get(
            "model",
            {
                "type": "gru",
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.1,
                "seq_mode": "single_step",
                "num_chunks": 1,
            },
        )
    )
    aggregator_name = str(config.get("aggregator", "fedavg"))
    aggregator_kwargs = config.get("aggregator_kwargs")
    attack_name = str(config.get("attack", config.get("attack_name", "none")))
    poison_fraction = float(config.get("poison_fraction", 0.0))
    attack_kwargs = config.get("attack_kwargs")
    scaler_mode = str(config.get("scaler_mode", "global"))
    max_train_samples = config.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
    eval_every = int(config.get("eval_every", 1))
    eval_per_tenant_every = int(config.get("eval_per_tenant_every", 5))

    bundle = load_dataset_artifacts(
        dataset_name,
        processed_root=processed_root,
        results_01_dir=results_01_dir,
    )

    seed_runs: list[dict[str, Any]] = []
    per_round_rows: list[dict[str, Any]] = []

    for seed in seeds:
        res = run_single_seed_fl(
            bundle,
            task=task,
            partition_type=partition_type,
            num_tenants=num_tenants,
            dirichlet_alpha=dirichlet_alpha,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            model_cfg=model_cfg,
            aggregator_name=aggregator_name,
            aggregator_kwargs=aggregator_kwargs,
            attack_name=attack_name,
            poison_fraction=poison_fraction,
            attack_kwargs=attack_kwargs,
            scaler_mode=scaler_mode,
            max_train_samples=max_train_samples,
            eval_every=eval_every,
            eval_per_tenant_every=eval_per_tenant_every,
            device=device,
        )
        seed_runs.append(res)
        for h in res["history"]:
            row = {
                "dataset": dataset_name,
                "task": task,
                "partition_type": partition_type,
                "num_tenants": num_tenants,
                "dirichlet_alpha": dirichlet_alpha if partition_type == "dirichlet" else np.nan,
                "seed": seed,
                **{k: v for k, v in h.items() if k != "per_tenant_f1_macro"},
            }
            per_round_rows.append(row)

    summary = summarize_multi_seed_fl(seed_runs)
    summary["config"] = config

    output_dir_val = config.get("output_dir")
    if output_dir_val:
        out_path = Path(output_dir_val)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.mkdir(parents=True, exist_ok=True)
        exp_id = str(
            config.get(
                "experiment_id",
                f"fedavg_{dataset_name}_{task}_{partition_type}_n{num_tenants}",
            )
        )
        pd.DataFrame(per_round_rows).to_csv(
            out_path / f"per_round_{exp_id}.csv",
            index=False,
        )
        summary_file = out_path / f"summary_{exp_id}.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["seed_runs"] = seed_runs
    return summary


def extract_step02_reference(
    baselines_02: dict[str, Any],
    ds_name: str,
    task_key: str,
) -> dict[str, float]:
    """Extract Centralized GRU upper bound and Local-Only GRU lower bound from Step 02 summary."""
    b02 = baselines_02["datasets"][ds_name]
    cent_metrics = b02["centralized"][task_key]["gru"]["metrics"]
    cent_f1_mean = float(cent_metrics["f1_macro"]["mean"])
    cent_f1_std = float(cent_metrics["f1_macro"]["std"])
    cent_acc_mean = float(cent_metrics["accuracy"]["mean"])

    loc_map = b02["local_only"][task_key]
    loc_f1s = [float(v["metrics"]["global_f1_macro"]["mean"]) for v in loc_map.values()]
    loc_accs = [float(v["metrics"]["global_accuracy"]["mean"]) for v in loc_map.values()]

    return {
        "cent_f1_mean": cent_f1_mean,
        "cent_f1_std": cent_f1_std,
        "cent_acc_mean": cent_acc_mean,
        "loc_f1_mean": float(np.mean(loc_f1s)),
        "loc_f1_std": float(np.std(loc_f1s, ddof=0)),
        "loc_acc_mean": float(np.mean(loc_accs)),
    }


def plot_convergence_curves(
    sweep_results: dict[str, Any],
    baselines_02: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot round-by-round Macro-F1 and Train Loss curves (`1..30`) across partition regimes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=160)

    datasets = [("insdn", "InSDN (SDN Native)"), ("aseados_sdn_iot", "ASEADOS-SDN-IoT (2026)")]
    regimes = [
        ("iid_n3", "FedAvg — IID (N=3)", "#1f77b4", "-"),
        ("natural_n3", "FedAvg — Natural Non-IID (N=3)", "#d62728", "-"),
        ("dirichlet_a1.0_n3", "FedAvg — Dirichlet α=1.0 (N=3)", "#2ca02c", "--"),
        ("dirichlet_a0.5_n3", "FedAvg — Dirichlet α=0.5 (N=3)", "#ff7f0e", "-."),
        ("dirichlet_a0.1_n3", "FedAvg — Dirichlet α=0.1 (N=3)", "#9467bd", ":"),
    ]

    for col_idx, (ds_key, ds_title) in enumerate(datasets):
        ax_f1 = axes[0, col_idx]
        ax_mult = axes[1, col_idx]

        # Plot Binary Task in Row 0, Multi-Class Task in Row 1
        for row_idx, (task_key, task_label, ax) in enumerate(
            [
                ("binary", "Binary Detection Macro-F1 (%)", ax_f1),
                ("multiclass", "Multi-Class Macro-F1 (%)", ax_mult),
            ]
        ):
            _ = row_idx
            ds_res = sweep_results["datasets"].get(ds_key, {}).get(task_key, {})

            # Reference lines from Step 02
            ref02 = extract_step02_reference(baselines_02, ds_key, task_key)
            cent_f1 = ref02["cent_f1_mean"] * 100.0
            loc_mean_f1 = ref02["loc_f1_mean"] * 100.0

            ax.axhline(
                cent_f1,
                color="#111111",
                linestyle="--",
                linewidth=1.5,
                label=f"Centralized Upper Bound ({cent_f1:.2f}%)",
            )
            ax.axhline(
                loc_mean_f1,
                color="#7f7f7f",
                linestyle=":",
                linewidth=1.5,
                label=f"Local-Only Mean Lower Bound ({loc_mean_f1:.2f}%)",
            )

            for reg_key, reg_label, color, lstyle in regimes:
                if reg_key not in ds_res:
                    continue
                curves = ds_res[reg_key]["round_curves"]
                rds = [int(c["round"]) for c in curves if "f1_macro_mean" in c]
                means = [float(c["f1_macro_mean"]) * 100.0 for c in curves if "f1_macro_mean" in c]
                stds = [float(c["f1_macro_std"]) * 100.0 for c in curves if "f1_macro_mean" in c]
                ax.plot(
                    rds,
                    means,
                    label=f"{reg_label} ({means[-1]:.2f}%)",
                    color=color,
                    linestyle=lstyle,
                    linewidth=2.0,
                    marker="o",
                    markersize=3.5,
                )
                ax.fill_between(
                    rds,
                    np.asarray(means) - np.asarray(stds),
                    np.asarray(means) + np.asarray(stds),
                    color=color,
                    alpha=0.12,
                )

            ax.set_title(f"{ds_title} — {task_label}", fontsize=10.5, fontweight="bold")
            ax.set_xlabel("Federated Communication Round (t = 1..30)")
            ax.set_ylabel("Global Test Macro-F1 (%)")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    fig.suptitle(
        "Step 03 — FedAvg Convergence Curves vs. Centralized Upper Bound "
        "& Local-Only Lower Bound (5 Seeds)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_centralized_vs_fedavg_vs_local_only(
    sweep_results: dict[str, Any],
    baselines_02: dict[str, Any],
    out_path: Path,
) -> None:
    """Bar chart comparing Centralized, FedAvg (IID, Natural, Dirichlet), and Local-Only."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=160)

    datasets = ["insdn", "aseados_sdn_iot", "cicids2017"]
    ds_labels = [
        "InSDN\n(SDN Native)",
        "ASEADOS-SDN-IoT\n(2026 Modern)",
        "CICIDS2017\n(General IDS)",
    ]

    for idx, (task_key, task_title) in enumerate(
        [
            ("binary", "Binary Threat Detection — Global Test Macro-F1 (%)"),
            ("multiclass", "Multi-Class Attack Classification — Global Test Macro-F1 (%)"),
        ]
    ):
        ax = axes[idx]
        x = np.arange(len(datasets))
        width = 0.15

        cent_means, cent_stds = [], []
        iid_means, iid_stds = [], []
        nat_means, nat_stds = [], []
        dir05_means, dir05_stds = [], []
        loc_means, loc_stds = [], []

        for ds in datasets:
            ref02 = extract_step02_reference(baselines_02, ds, task_key)
            cent_means.append(ref02["cent_f1_mean"] * 100.0)
            cent_stds.append(ref02["cent_f1_std"] * 100.0)
            loc_means.append(ref02["loc_f1_mean"] * 100.0)
            loc_stds.append(ref02["loc_f1_std"] * 100.0)

            ds_f = sweep_results["datasets"][ds][task_key]
            iid_key = "iid_n3" if "iid_n3" in ds_f else "iid_n10"
            dir_key = "dirichlet_a0.5_n3" if "dirichlet_a0.5_n3" in ds_f else "dirichlet_a0.5_n10"

            iid_means.append(float(ds_f[iid_key]["metrics"]["f1_macro"]["mean"]) * 100.0)
            iid_stds.append(float(ds_f[iid_key]["metrics"]["f1_macro"]["std"]) * 100.0)

            nat_means.append(float(ds_f["natural_n3"]["metrics"]["f1_macro"]["mean"]) * 100.0)
            nat_stds.append(float(ds_f["natural_n3"]["metrics"]["f1_macro"]["std"]) * 100.0)

            dir05_means.append(float(ds_f[dir_key]["metrics"]["f1_macro"]["mean"]) * 100.0)
            dir05_stds.append(float(ds_f[dir_key]["metrics"]["f1_macro"]["std"]) * 100.0)

        bars = [
            (cent_means, cent_stds, "Centralized Pooled (Upper Bound)", "#1b4f72", -2 * width),
            (iid_means, iid_stds, "FedAvg — IID", "#2874a6", -width),
            (nat_means, nat_stds, "FedAvg — Natural Non-IID (N=3)", "#e67e22", 0.0),
            (dir05_means, dir05_stds, "FedAvg — Dirichlet α=0.5", "#27ae60", width),
            (loc_means, loc_stds, "Local-Only Mean (Lower Bound)", "#95a5a6", 2 * width),
        ]

        for means, stds, label, color, offset in bars:
            rects = ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.6,
            )
            for rect in rects:
                h = rect.get_height()
                ax.annotate(
                    f"{h:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, fontsize=9.5)
        ax.set_ylabel("Global Test Macro-F1 (%)")
        ax.set_title(task_title, fontsize=10.5, fontweight="bold")
        if task_key == "binary":
            ax.set_ylim(74.0, 103.5)
        else:
            ax.set_ylim(25.0, 75.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Step 03 — Centralized vs. FedAvg vs. Local-Only Comparison "
        "Across All 3 Datasets (GRU, 5 Seeds)",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def plot_non_iid_and_scalability(
    sweep_results: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot FedAvg across Dirichlet alpha in {0.1, 0.5, 1.0} & IID for N in {3, 5, 10}."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=160)

    regimes_order = [
        ("dirichlet_a0.1", "Dirichlet\nα = 0.1 (Extreme Non-IID)"),
        ("dirichlet_a0.5", "Dirichlet\nα = 0.5 (Moderate Non-IID)"),
        ("dirichlet_a1.0", "Dirichlet\nα = 1.0 (Mild Non-IID)"),
        ("iid", "Stratified\nIID Control"),
    ]
    tenant_counts = [
        (3, "N = 3 Tenants", "#1f77b4", "o"),
        (5, "N = 5 Tenants", "#ff7f0e", "s"),
        (10, "N = 10 Tenants", "#2ca02c", "^"),
    ]

    for idx, (task_key, task_title) in enumerate(
        [
            ("binary", "InSDN Binary Macro-F1 (%) vs. Non-IID & Tenant Count N"),
            ("multiclass", "InSDN Multi-Class Macro-F1 (%) vs. Non-IID & Tenant Count N"),
        ]
    ):
        ax = axes[idx]
        ds_f = sweep_results["datasets"]["insdn"][task_key]
        x = np.arange(len(regimes_order))

        for n_val, n_label, color, marker in tenant_counts:
            means, stds = [], []
            for prefix, _ in regimes_order:
                rkey = f"{prefix}_n{n_val}"
                m = float(ds_f[rkey]["metrics"]["f1_macro"]["mean"]) * 100.0
                s = float(ds_f[rkey]["metrics"]["f1_macro"]["std"]) * 100.0
                means.append(m)
                stds.append(s)

            ax.errorbar(
                x,
                means,
                yerr=stds,
                label=n_label,
                color=color,
                marker=marker,
                markersize=6.5,
                linewidth=2.0,
                capsize=4,
            )

        nat_m = float(ds_f["natural_n3"]["metrics"]["f1_macro"]["mean"]) * 100.0
        ax.axhline(
            nat_m,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"Natural Domain Non-IID (N=3): {nat_m:.2f}%",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in regimes_order], fontsize=8.8)
        ax.set_ylabel("Global Test Macro-F1 (%)")
        ax.set_title(task_title, fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.suptitle(
        "Step 03 — Effect of Data Heterogeneity (Dirichlet α) "
        "and Tenant Scalability (N ∈ {3, 5, 10}) on FedAvg",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def compute_paired_wilcoxon(vals_a: list[float], vals_b: list[float]) -> float:
    """Compute two-sided Wilcoxon signed-rank p-value across paired seed runs."""
    arr_a = np.asarray(vals_a, dtype=np.float64)
    arr_b = np.asarray(vals_b, dtype=np.float64)
    if len(arr_a) != len(arr_b) or len(arr_a) < 5 or np.allclose(arr_a, arr_b):
        return 1.0
    try:
        stat = wilcoxon(arr_a, arr_b, zero_method="zsplit", alternative="two-sided")
        return float(stat.pvalue)
    except Exception:
        return 1.0


def run_step03_full_sweep(
    *,
    out_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS_DEFAULT,
    rounds: int = 30,
    local_epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_train_samples: int = 8000,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute the complete Step 03 5-seed FedAvg benchmark and save all artifacts."""
    if device is None:
        device = get_default_device(prefer_gpu=True)
    target_dir = out_dir or (REPO_ROOT / "results" / "03_fedavg")
    target_dir.mkdir(parents=True, exist_ok=True)

    baselines_02_path = REPO_ROOT / "results" / "02_baselines" / "summary.json"
    baselines_02 = json.loads(baselines_02_path.read_text(encoding="utf-8"))

    # Regimes per dataset
    insdn_regimes: list[tuple[str, str, int, float]] = [
        ("natural_n3", "natural", 3, 0.0),
        ("dirichlet_a0.1_n3", "dirichlet", 3, 0.1),
        ("dirichlet_a0.5_n3", "dirichlet", 3, 0.5),
        ("dirichlet_a1.0_n3", "dirichlet", 3, 1.0),
        ("iid_n3", "iid", 3, 0.0),
        ("dirichlet_a0.1_n5", "dirichlet", 5, 0.1),
        ("dirichlet_a0.5_n5", "dirichlet", 5, 0.5),
        ("dirichlet_a1.0_n5", "dirichlet", 5, 1.0),
        ("iid_n5", "iid", 5, 0.0),
        ("dirichlet_a0.1_n10", "dirichlet", 10, 0.1),
        ("dirichlet_a0.5_n10", "dirichlet", 10, 0.5),
        ("dirichlet_a1.0_n10", "dirichlet", 10, 1.0),
        ("iid_n10", "iid", 10, 0.0),
    ]
    aseados_regimes: list[tuple[str, str, int, float]] = [
        ("natural_n3", "natural", 3, 0.0),
        ("dirichlet_a0.1_n3", "dirichlet", 3, 0.1),
        ("dirichlet_a0.5_n3", "dirichlet", 3, 0.5),
        ("dirichlet_a1.0_n3", "dirichlet", 3, 1.0),
        ("iid_n3", "iid", 3, 0.0),
        ("dirichlet_a0.5_n5", "dirichlet", 5, 0.5),
        ("iid_n5", "iid", 5, 0.0),
        ("dirichlet_a0.1_n10", "dirichlet", 10, 0.1),
        ("dirichlet_a0.5_n10", "dirichlet", 10, 0.5),
        ("dirichlet_a1.0_n10", "dirichlet", 10, 1.0),
        ("iid_n10", "iid", 10, 0.0),
    ]
    cicids_regimes: list[tuple[str, str, int, float]] = [
        ("natural_n3", "natural", 3, 0.0),
        ("dirichlet_a0.5_n10", "dirichlet", 10, 0.5),
        ("iid_n10", "iid", 10, 0.0),
    ]

    dataset_plan = [
        ("insdn", insdn_regimes),
        ("aseados_sdn_iot", aseados_regimes),
        ("cicids2017", cicids_regimes),
    ]

    model_cfg_by_ds: dict[str, dict[str, Any]] = {
        "insdn": {
            "type": "gru",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "seq_mode": "single_step",
            "num_chunks": 1,
        },
        "aseados_sdn_iot": {
            "type": "gru",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "seq_mode": "single_step",
            "num_chunks": 1,
        },
        "cicids2017": {
            "type": "gru",
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "seq_mode": "feature_chunks",
            "num_chunks": 4,
        },
    }

    all_per_round_rows: list[dict[str, Any]] = []
    all_per_seed_rows: list[dict[str, Any]] = []
    sweep_summary: dict[str, Any] = {
        "step": "03_fedavg",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "fl_defaults": {
            "aggregator": "fedavg",
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "optimizer": "Adam",
            "lr": float(lr),
            "weight_decay": 1e-4,
            "client_participation": "100% (all tenants every round)",
            "seeds": list(seeds),
            "max_train_samples_stratified": int(max_train_samples),
        },
        "datasets": {},
        "comparisons_vs_step02": {},
        "scaler_ablation": {},
    }

    ckpt_path = target_dir / "summary_checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            sweep_summary["datasets"] = ckpt.get("datasets", {})
            sweep_summary["comparisons_vs_step02"] = ckpt.get("comparisons_vs_step02", {})
            sweep_summary["scaler_ablation"] = ckpt.get("scaler_ablation", {})
        except Exception:
            pass

    t_sweep_start = time.perf_counter()

    for ds_name, regimes in dataset_plan:
        existing_ds = sweep_summary["datasets"].get(ds_name, {})
        if (
            len(existing_ds.get("binary", {})) == len(regimes)
            and len(existing_ds.get("multiclass", {})) == len(regimes)
            and ds_name in sweep_summary["comparisons_vs_step02"]
        ):
            print(
                f"[Step 03] Resuming from checkpoint: '{ds_name}' already 100% completed "
                f"({len(regimes)} binary + {len(regimes)} multiclass regimes); skipping.",
                flush=True,
            )
            for task_key in ("binary", "multiclass"):
                for reg_key, ptype, n_ten, alpha in regimes:
                    reg_sum = existing_ds[task_key][reg_key]
                    m = reg_sum["metrics"]
                    for s_idx, f1_val in enumerate(reg_sum.get("seed_f1_macros", [])):
                        all_per_seed_rows.append(
                            {
                                "dataset": ds_name,
                                "task": task_key,
                                "regime": reg_key,
                                "partition_type": ptype,
                                "num_tenants": n_ten,
                                "dirichlet_alpha": alpha if ptype == "dirichlet" else np.nan,
                                "seed": s_idx,
                                "rounds": rounds,
                                "local_epochs": local_epochs,
                                "accuracy": m["accuracy"]["mean"],
                                "precision_macro": m["precision_macro"]["mean"],
                                "recall_macro": m["recall_macro"]["mean"],
                                "f1_macro": f1_val,
                                "f1_attack": m["f1_attack"]["mean"],
                                "attack_detection_rate": m["attack_detection_rate"]["mean"],
                                "false_positive_rate": m["false_positive_rate"]["mean"],
                                "roc_auc": m["roc_auc"]["mean"],
                                "tenant_f1_macro_mean": m.get("tenant_f1_macro_mean", {}).get(
                                    "mean"
                                ),
                                "tenant_f1_macro_min": m.get("tenant_f1_macro_min", {}).get("mean"),
                                "bytes_per_round": int(m["bytes_per_round"]["mean"]),
                                "total_comm_bytes": int(m["total_comm_bytes"]["mean"]),
                                "mean_round_local_train_time_sec": m[
                                    "mean_round_local_train_time_sec"
                                ]["mean"],
                                "mean_round_aggregation_time_sec": m[
                                    "mean_round_aggregation_time_sec"
                                ]["mean"],
                                "total_wall_time_sec": m["mean_round_local_train_time_sec"]["mean"]
                                * rounds,
                                "round_to_95pct_f1": int(m["round_to_95pct_f1"]["mean"]),
                            }
                        )
                    for c in reg_sum.get("round_curves", []):
                        all_per_round_rows.append(
                            {
                                "dataset": ds_name,
                                "task": task_key,
                                "regime": reg_key,
                                "partition_type": ptype,
                                "num_tenants": n_ten,
                                "dirichlet_alpha": alpha if ptype == "dirichlet" else np.nan,
                                "seed": -1,
                                "round": c["round"],
                                "train_loss_weighted": c.get("train_loss_mean"),
                                "local_train_time_sec": c.get("local_train_time_sec_mean"),
                                "aggregation_time_sec": c.get("aggregation_time_sec_mean"),
                                "total_comm_bytes": c.get("total_comm_bytes"),
                                "cumulative_comm_bytes": c.get("cumulative_comm_bytes"),
                                "f1_macro": c.get("f1_macro_mean"),
                                "accuracy": c.get("accuracy_mean"),
                                "attack_detection_rate": c.get("adr_mean"),
                                "false_positive_rate": c.get("fpr_mean"),
                            }
                        )
            continue

        bundle = load_dataset_artifacts(ds_name)
        mcfg = model_cfg_by_ds[ds_name]
        sweep_summary["datasets"].setdefault(ds_name, {})
        sweep_summary["datasets"][ds_name].setdefault("binary", {})
        sweep_summary["datasets"][ds_name].setdefault("multiclass", {})

        for task_key in ("binary", "multiclass"):
            for reg_key, ptype, n_ten, alpha in regimes:
                if reg_key in sweep_summary["datasets"][ds_name][task_key]:
                    print(
                        f"[Step 03] Resuming from checkpoint: {ds_name} | {task_key} | {reg_key} "
                        "already completed; skipping.",
                        flush=True,
                    )
                    continue
                t_reg = time.perf_counter()
                # Evaluate every 3 rounds for N=3 curves, every 5 rounds for N=5,10
                ev_every = 3 if n_ten == 3 else 5
                seed_runs: list[dict[str, Any]] = []

                for s in seeds:
                    sr = run_single_seed_fl(
                        bundle,
                        task=task_key,
                        partition_type=ptype,
                        num_tenants=n_ten,
                        dirichlet_alpha=alpha,
                        rounds=rounds,
                        local_epochs=local_epochs,
                        batch_size=batch_size,
                        lr=lr,
                        seed=s,
                        model_cfg=mcfg,
                        aggregator_name="fedavg",
                        scaler_mode="global",
                        max_train_samples=max_train_samples,
                        eval_every=ev_every,
                        eval_per_tenant_every=10,
                        device=device,
                    )
                    seed_runs.append(sr)

                    fm = sr["final_metrics"]
                    all_per_seed_rows.append(
                        {
                            "dataset": ds_name,
                            "task": task_key,
                            "regime": reg_key,
                            "partition_type": ptype,
                            "num_tenants": n_ten,
                            "dirichlet_alpha": alpha if ptype == "dirichlet" else np.nan,
                            "seed": s,
                            "rounds": rounds,
                            "local_epochs": local_epochs,
                            "accuracy": fm.get("accuracy"),
                            "precision_macro": fm.get("precision_macro"),
                            "recall_macro": fm.get("recall_macro"),
                            "f1_macro": fm.get("f1_macro"),
                            "f1_attack": fm.get("f1_attack"),
                            "attack_detection_rate": fm.get("attack_detection_rate"),
                            "false_positive_rate": fm.get("false_positive_rate"),
                            "roc_auc": fm.get("roc_auc"),
                            "tenant_f1_macro_mean": fm.get("tenant_f1_macro_mean"),
                            "tenant_f1_macro_min": fm.get("tenant_f1_macro_min"),
                            "bytes_per_round": sr["bytes_per_round"],
                            "total_comm_bytes": sr["total_comm_bytes"],
                            "mean_round_local_train_time_sec": sr[
                                "mean_round_local_train_time_sec"
                            ],
                            "mean_round_aggregation_time_sec": sr[
                                "mean_round_aggregation_time_sec"
                            ],
                            "total_wall_time_sec": sr["total_wall_time_sec"],
                            "round_to_95pct_f1": sr["round_to_95pct_f1"],
                        }
                    )

                    for h in sr["history"]:
                        all_per_round_rows.append(
                            {
                                "dataset": ds_name,
                                "task": task_key,
                                "regime": reg_key,
                                "partition_type": ptype,
                                "num_tenants": n_ten,
                                "dirichlet_alpha": alpha if ptype == "dirichlet" else np.nan,
                                "seed": s,
                                **{k: v for k, v in h.items() if k != "per_tenant_f1_macro"},
                            }
                        )

                reg_summary = summarize_multi_seed_fl(seed_runs)
                reg_summary["seed_f1_macros"] = [
                    float(sr["final_metrics"]["f1_macro"]) for sr in seed_runs
                ]
                sweep_summary["datasets"][ds_name][task_key][reg_key] = reg_summary
                (target_dir / "summary_checkpoint.json").write_text(
                    json.dumps(sweep_summary, indent=2),
                    encoding="utf-8",
                )
                f1_m = reg_summary["metrics"]["f1_macro"]["mean"] * 100.0
                f1_s = reg_summary["metrics"]["f1_macro"]["std"] * 100.0
                dt = time.perf_counter() - t_reg
                print(
                    f"[Step 03] {ds_name:16s} | {task_key:10s} | {reg_key:18s} "
                    f"-> Macro-F1 = {f1_m:6.2f} ± {f1_s:4.2f}% ({dt:.1f}s)",
                    flush=True,
                )

        # Compute explicit comparisons against Step 02 Centralized & Local-Only baselines
        sweep_summary["comparisons_vs_step02"][ds_name] = {}
        for task_key in ("binary", "multiclass"):
            ref02 = extract_step02_reference(baselines_02, ds_name, task_key)
            cent_f1 = ref02["cent_f1_mean"]
            cent_acc = ref02["cent_acc_mean"]
            loc_mean_f1 = ref02["loc_f1_mean"]
            loc_mean_acc = ref02["loc_acc_mean"]

            ds_task_res = sweep_summary["datasets"][ds_name][task_key]
            iid_ref_key = "iid_n3" if "iid_n3" in ds_task_res else "iid_n10"
            iid_f1 = float(ds_task_res[iid_ref_key]["metrics"]["f1_macro"]["mean"])
            iid_acc = float(ds_task_res[iid_ref_key]["metrics"]["accuracy"]["mean"])

            nat_f1 = float(ds_task_res["natural_n3"]["metrics"]["f1_macro"]["mean"])
            nat_acc = float(ds_task_res["natural_n3"]["metrics"]["accuracy"]["mean"])

            dir05_key = (
                "dirichlet_a0.5_n3" if "dirichlet_a0.5_n3" in ds_task_res else "dirichlet_a0.5_n10"
            )
            dir05_f1 = float(ds_task_res[dir05_key]["metrics"]["f1_macro"]["mean"])
            dir01_key = (
                "dirichlet_a0.1_n3" if "dirichlet_a0.1_n3" in ds_task_res else "dirichlet_a0.5_n10"
            )
            dir01_f1 = float(ds_task_res[dir01_key]["metrics"]["f1_macro"]["mean"])

            p_iid_vs_nat = compute_paired_wilcoxon(
                ds_task_res[iid_ref_key]["seed_f1_macros"],
                ds_task_res["natural_n3"]["seed_f1_macros"],
            )
            p_iid_vs_dir01 = compute_paired_wilcoxon(
                ds_task_res[iid_ref_key]["seed_f1_macros"],
                ds_task_res[dir01_key]["seed_f1_macros"],
            )

            sweep_summary["comparisons_vs_step02"][ds_name][task_key] = {
                "centralized_f1_macro": cent_f1,
                "centralized_accuracy": cent_acc,
                "fedavg_iid_f1_macro": iid_f1,
                "fedavg_iid_accuracy": iid_acc,
                "fedavg_natural_n3_f1_macro": nat_f1,
                "fedavg_natural_n3_accuracy": nat_acc,
                "fedavg_dirichlet_a0.5_f1_macro": dir05_f1,
                "fedavg_dirichlet_a0.1_f1_macro": dir01_f1,
                "local_only_mean_f1_macro": loc_mean_f1,
                "local_only_mean_accuracy": loc_mean_acc,
                "loss_to_federation_iid_f1_pts": float((cent_f1 - iid_f1) * 100.0),
                "loss_to_federation_iid_acc_pts": float((cent_acc - iid_acc) * 100.0),
                "loss_to_non_iid_natural_vs_iid_f1_pts": float((iid_f1 - nat_f1) * 100.0),
                "loss_to_non_iid_natural_vs_iid_acc_pts": float((iid_acc - nat_acc) * 100.0),
                "loss_to_non_iid_dir0.5_vs_iid_f1_pts": float((iid_f1 - dir05_f1) * 100.0),
                "loss_to_non_iid_dir0.1_vs_iid_f1_pts": float((iid_f1 - dir01_f1) * 100.0),
                "total_loss_centralized_to_natural_f1_pts": float((cent_f1 - nat_f1) * 100.0),
                "fedavg_natural_gain_over_local_only_f1_pts": float((nat_f1 - loc_mean_f1) * 100.0),
                "wilcoxon_p_iid_vs_natural": p_iid_vs_nat,
                "wilcoxon_p_iid_vs_dirichlet_a0.1": p_iid_vs_dir01,
            }
        (target_dir / "summary_checkpoint.json").write_text(
            json.dumps(sweep_summary, indent=2),
            encoding="utf-8",
        )

    # Scaler ablation on InSDN natural_n3 (local_per_tenant vs global/federated-moments)
    insdn_bundle = load_dataset_artifacts("insdn")
    local_scaler_runs = [
        run_single_seed_fl(
            insdn_bundle,
            task="binary",
            partition_type="natural",
            num_tenants=3,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            seed=s,
            model_cfg=model_cfg_by_ds["insdn"],
            scaler_mode="local_per_tenant",
            max_train_samples=max_train_samples,
            eval_every=10,
            eval_per_tenant_every=10,
            device=device,
        )
        for s in seeds
    ]
    sweep_summary["scaler_ablation"]["insdn_binary_natural_n3_local_per_tenant"] = (
        summarize_multi_seed_fl(local_scaler_runs)
    )

    sweep_summary["total_sweep_wall_time_sec"] = float(time.perf_counter() - t_sweep_start)

    # Save CSVs & JSON
    pd.DataFrame(all_per_round_rows).to_csv(target_dir / "per_round_metrics.csv", index=False)
    pd.DataFrame(all_per_seed_rows).to_csv(target_dir / "per_seed_summary.csv", index=False)
    (target_dir / "summary.json").write_text(
        json.dumps(sweep_summary, indent=2),
        encoding="utf-8",
    )

    # Generate all 3 Step 03 plots
    plot_convergence_curves(
        sweep_summary,
        baselines_02,
        target_dir / "convergence_curves.png",
    )
    plot_centralized_vs_fedavg_vs_local_only(
        sweep_summary,
        baselines_02,
        target_dir / "centralized_vs_fedavg_vs_local_only.png",
    )
    plot_non_iid_and_scalability(
        sweep_summary,
        target_dir / "non_iid_and_scalability_comparison.png",
    )
    return sweep_summary


STEP04_AGG_ORDER: list[tuple[str, str, str, str, str]] = [
    ("fedavg", "FedAvg (Unprotected Baseline)", "#d62728", "o", "-"),
    ("static_weight", "StaticWeight (Ali et al., 2023)", "#ff7f0e", "s", "--"),
    ("trimmed_mean_b0.1", "TrimmedMean (β = 0.1)", "#9467bd", "^", "-."),
    ("trimmed_mean_b0.2", "TrimmedMean (β = 0.2)", "#2ca02c", "D", "--"),
    ("median", "Coordinate Median", "#17becf", "v", ":"),
    ("clustering", "CosineClustering (Norm + Clique)", "#1f77b4", "P", "-"),
]


def _get_step04_entry(
    grid: dict[str, Any],
    attack_key: str,
    pf_val: float,
    agg_key: str,
) -> dict[str, Any] | None:
    """Lookup `(attack_key, pf_val, agg_key)` in a Step 04 regime dictionary, falling back to `none` at `pf=0.0`."""
    pf_str = f"{pf_val:.2f}"
    key = f"{attack_key}__pf{pf_str}__{agg_key}"
    if key in grid:
        return grid[key]
    if abs(pf_val) < 1e-9:
        clean_key = f"none__pf0.00__{agg_key}"
        if clean_key in grid:
            return grid[clean_key]
    return None


def plot_accuracy_vs_poison_fraction(
    sweep_04: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot Global Test Macro-F1 (%) vs. Poisoned Tenant Fraction (0%..30%) across all 3 attacks & 6 aggregators."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=160)

    attacks = [
        ("label_flip", "Label Flipping Attack (100% Bidirectional + 8.5× Scale)"),
        ("backdoor", "Feature-Trigger Backdoor + Model Replacement (6.0× Scale)"),
        ("scaling", "Update Scaling & Sign-Flipping Attack (-10.0× Scale)"),
    ]
    pfs_insdn = [0.00, 0.10, 0.20, 0.30]
    pfs_aseados = [0.00, 0.20, 0.30]

    insdn_dir_grid = sweep_04["grids"].get("insdn_dirichlet_a0.5_n10", {})
    aseados_dir_grid = sweep_04["grids"].get("aseados_dirichlet_a0.5_n10", {})

    # Top row: InSDN (Dirichlet alpha=0.5, N=10)
    for col_idx, (atk_key, atk_title) in enumerate(attacks):
        ax = axes[0, col_idx]
        for agg_key, agg_label, color, marker, lstyle in STEP04_AGG_ORDER:
            xs, means, stds = [], [], []
            for pf in pfs_insdn:
                entry = _get_step04_entry(insdn_dir_grid, atk_key, pf, agg_key)
                if entry is None:
                    continue
                xs.append(pf * 100.0)
                means.append(float(entry["metrics"]["f1_macro"]["mean"]) * 100.0)
                stds.append(float(entry["metrics"]["f1_macro"]["std"]) * 100.0)
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    label=agg_label,
                    color=color,
                    marker=marker,
                    linestyle=lstyle,
                    linewidth=2.1,
                    markersize=6.0,
                    capsize=3.5,
                )
        ax.set_title(f"InSDN (Dirichlet α=0.5, N=10)\n{atk_title}", fontsize=9.8, fontweight="bold")
        ax.set_xlabel("Poisoned Tenant Fraction (%)")
        ax.set_ylabel("Global Test Macro-F1 (%)")
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylim(0.0, 103.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 0:
            ax.legend(loc="lower left", fontsize=7.8, framealpha=0.92)

    # Bottom row: ASEADOS-SDN-IoT (2026 Modern SDN-IoT, Dirichlet alpha=0.5, N=10)
    for col_idx, (atk_key, atk_title) in enumerate(attacks):
        ax = axes[1, col_idx]
        for agg_key, agg_label, color, marker, lstyle in STEP04_AGG_ORDER:
            xs, means, stds = [], [], []
            for pf in pfs_aseados:
                entry = _get_step04_entry(aseados_dir_grid, atk_key, pf, agg_key)
                if entry is None:
                    continue
                xs.append(pf * 100.0)
                means.append(float(entry["metrics"]["f1_macro"]["mean"]) * 100.0)
                stds.append(float(entry["metrics"]["f1_macro"]["std"]) * 100.0)
            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    label=agg_label,
                    color=color,
                    marker=marker,
                    linestyle=lstyle,
                    linewidth=2.1,
                    markersize=6.0,
                    capsize=3.5,
                )
        ax.set_title(
            f"ASEADOS-SDN-IoT (2026, Dirichlet α=0.5, N=10)\n{atk_title}",
            fontsize=9.8,
            fontweight="bold",
        )
        ax.set_xlabel("Poisoned Tenant Fraction (%)")
        ax.set_ylabel("Global Test Macro-F1 (%)")
        ax.set_xticks([0, 20, 30])
        ax.set_ylim(0.0, 103.5)
        ax.grid(True, linestyle="--", alpha=0.35)
        if col_idx == 0:
            ax.legend(loc="lower left", fontsize=7.8, framealpha=0.92)

    fig.suptitle(
        "Step 04 — Detection Macro-F1 (%) vs. Poisoned Tenant Fraction Across 3 Attacks & 6 Aggregators (5 Seeds)",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_backdoor_asr_comparison(
    sweep_04: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot Backdoor Attack Success Rate (ASR %, lower is better) across aggregators & stealth regime."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=160)

    insdn_dir_grid = sweep_04["grids"].get("insdn_dirichlet_a0.5_n10", {})
    abl_grid = sweep_04["grids"].get("failure_mode_ablations", {})

    # Panel A: Backdoor ASR (%) vs Poisoned Fraction (0%, 10%, 20%, 30%) on InSDN
    ax0 = axes[0]
    pfs = [0.00, 0.10, 0.20, 0.30]
    for agg_key, agg_label, color, marker, lstyle in STEP04_AGG_ORDER:
        xs, means, stds = [], [], []
        for pf in pfs:
            entry = _get_step04_entry(insdn_dir_grid, "backdoor", pf, agg_key)
            if entry is None:
                continue
            xs.append(pf * 100.0)
            means.append(float(entry["metrics"]["backdoor_asr"]["mean"]) * 100.0)
            stds.append(float(entry["metrics"]["backdoor_asr"]["std"]) * 100.0)
        if xs:
            ax0.errorbar(
                xs,
                means,
                yerr=stds,
                label=agg_label,
                color=color,
                marker=marker,
                linestyle=lstyle,
                linewidth=2.2,
                markersize=6.5,
                capsize=4.0,
            )
    ax0.set_title(
        "InSDN (N=10): Backdoor Attack Success Rate (ASR %)\nvs. Poisoned Tenant Fraction (Aggressive 6.0× Replacement)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax0.set_xlabel("Poisoned Tenant Fraction (%)")
    ax0.set_ylabel("Backdoor ASR (%) — Lower is More Robust")
    ax0.set_xticks([0, 10, 20, 30])
    ax0.set_ylim(-3.0, 105.0)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="center right", fontsize=8.2, framealpha=0.92)

    # Panel B: Standard Backdoor (6.0x) vs Norm-Bounded Stealth Backdoor (1.4x) at 30% Poisoning
    ax1 = axes[1]
    compare_aggs = [
        ("fedavg", "FedAvg"),
        ("static_weight", "StaticWeight\n(Ali et al.)"),
        ("trimmed_mean_b0.2", "TrimmedMean\n(β=0.2)"),
        ("clustering", "CosineClustering\n(Norm+Clique)"),
    ]
    x = np.arange(len(compare_aggs))
    width = 0.34

    std_asr_means, std_asr_stds = [], []
    stl_asr_means, stl_asr_stds = [], []

    for agg_key, _ in compare_aggs:
        e_std = _get_step04_entry(insdn_dir_grid, "backdoor", 0.30, agg_key)
        e_stl = abl_grid.get(f"stealth_backdoor_pf0.30__{agg_key}")
        std_asr_means.append(
            float(e_std["metrics"]["backdoor_asr"]["mean"]) * 100.0 if e_std else 0.0
        )
        std_asr_stds.append(
            float(e_std["metrics"]["backdoor_asr"]["std"]) * 100.0 if e_std else 0.0
        )
        stl_asr_means.append(
            float(e_stl["metrics"]["backdoor_asr"]["mean"]) * 100.0 if e_stl else 0.0
        )
        stl_asr_stds.append(
            float(e_stl["metrics"]["backdoor_asr"]["std"]) * 100.0 if e_stl else 0.0
        )

    rects1 = ax1.bar(
        x - width / 2,
        std_asr_means,
        width,
        yerr=std_asr_stds,
        capsize=4,
        label="Aggressive Replacement Backdoor (scale = 6.0×)",
        color="#2980b9",
        edgecolor="white",
    )
    rects2 = ax1.bar(
        x + width / 2,
        stl_asr_means,
        width,
        yerr=stl_asr_stds,
        capsize=4,
        label="Norm-Bounded Stealth Backdoor (scale = 1.4×, Adaptive)",
        color="#c0392b",
        edgecolor="white",
    )
    for rects in (rects1, rects2):
        for r in rects:
            h = r.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in compare_aggs], fontsize=9.2)
    ax1.set_ylabel("Backdoor Attack Success Rate (ASR %)")
    ax1.set_ylim(0.0, 114.0)
    ax1.set_title(
        "Why Geometric Defenses Fail Under Adaptive Stealth:\nStandard (6.0×) vs. Norm-Bounded (1.4×) Backdoor at 30% Poisoning",
        fontsize=10.2,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    fig.suptitle(
        "Step 04 — Backdoor Evasion Analysis: Coordinate-Wise Failure & Norm-Bounded Stealth Vulnerability (5 Seeds)",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def plot_honest_false_rejection_rate(
    sweep_04: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot Honest-Tenant False-Rejection Rate across Non-IID regimes and Few-Tenant (N=3) breakdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=160)

    insdn_dir_grid = sweep_04["grids"].get("insdn_dirichlet_a0.5_n10", {})
    insdn_nat_grid = sweep_04["grids"].get("insdn_natural_n10", {})
    abl_grid = sweep_04["grids"].get("failure_mode_ablations", {})

    # Panel A: Honest False-Rejection Rate (%) of CosineClustering across Non-IID spectrum
    ax0 = axes[0]
    regimes = [
        (
            "IID Control\n(N=10)",
            abl_grid.get("iid_n10_clean__clustering"),
            abl_grid.get("iid_n10_scaling_pf0.20__clustering"),
        ),
        (
            "Natural Non-IID\n(Domain, N=10)",
            _get_step04_entry(insdn_nat_grid, "none", 0.0, "clustering"),
            _get_step04_entry(insdn_nat_grid, "scaling", 0.20, "clustering"),
        ),
        (
            "Moderate Non-IID\n(Dirichlet α=0.5)",
            _get_step04_entry(insdn_dir_grid, "none", 0.0, "clustering"),
            _get_step04_entry(insdn_dir_grid, "scaling", 0.20, "clustering"),
        ),
        (
            "Extreme Non-IID\n(Dirichlet α=0.1)",
            abl_grid.get("dir01_n10_clean__clustering"),
            abl_grid.get("dir01_n10_scaling_pf0.20__clustering"),
        ),
    ]

    x0 = np.arange(len(regimes))
    w0 = 0.34
    clean_frr_m, clean_frr_s = [], []
    pois_frr_m, pois_frr_s = [], []

    for _, e_clean, e_pois in regimes:
        clean_frr_m.append(
            float(e_clean["metrics"]["mean_honest_false_rejection_rate"]["mean"]) * 100.0
            if e_clean
            else 0.0
        )
        clean_frr_s.append(
            float(e_clean["metrics"]["mean_honest_false_rejection_rate"]["std"]) * 100.0
            if e_clean
            else 0.0
        )
        pois_frr_m.append(
            float(e_pois["metrics"]["mean_honest_false_rejection_rate"]["mean"]) * 100.0
            if e_pois
            else 0.0
        )
        pois_frr_s.append(
            float(e_pois["metrics"]["mean_honest_false_rejection_rate"]["std"]) * 100.0
            if e_pois
            else 0.0
        )

    b1 = ax0.bar(
        x0 - w0 / 2,
        clean_frr_m,
        w0,
        yerr=clean_frr_s,
        capsize=4,
        label="Clean Regime (0% Poisoned — Pure Non-IID Drift)",
        color="#e67e22",
        edgecolor="white",
    )
    b2 = ax0.bar(
        x0 + w0 / 2,
        pois_frr_m,
        w0,
        yerr=pois_frr_s,
        capsize=4,
        label="Under 20% Update-Scaling Attack (2 of 10 Malicious)",
        color="#8e44ad",
        edgecolor="white",
    )
    for rects in (b1, b2):
        for r in rects:
            h = r.get_height()
            ax0.annotate(
                f"{h:.1f}%",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
            )

    ax0.set_xticks(x0)
    ax0.set_xticklabels([lbl for lbl, _, _ in regimes], fontsize=9.0)
    ax0.set_ylabel("Honest-Tenant False-Rejection Rate (%)")
    ax0.set_ylim(0.0, max(38.0, max(clean_frr_m + pois_frr_m) * 1.25))
    ax0.set_title(
        "CosineClustering Honest-Tenant False-Rejection Rate (%)\nAcross Data Heterogeneity Regimes (InSDN, N=10)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="upper left", fontsize=8.2, framealpha=0.92)

    # Panel B: Few-Tenant Regime (N=3, Natural Non-IID, 1 of 3 = 33.3% Poisoned)
    ax1 = axes[1]
    n3_aggs = [
        ("fedavg", "FedAvg"),
        ("trimmed_mean_b0.2", "TrimmedMean\n(β=0.2, ⌊0.2×3⌋=0)"),
        ("median", "Coordinate\nMedian"),
        ("clustering", "CosineClustering\n(Norm+Clique)"),
    ]
    x1 = np.arange(len(n3_aggs))
    w1 = 0.26

    c_m, c_s = [], []
    lf_m, lf_s = [], []
    sc_m, sc_s = [], []

    for agg_key, _ in n3_aggs:
        ec = abl_grid.get(f"natural_n3_clean__{agg_key}")
        elf = abl_grid.get(f"natural_n3_label_flip_pf0.33__{agg_key}")
        esc = abl_grid.get(f"natural_n3_scaling_pf0.33__{agg_key}")
        c_m.append(float(ec["metrics"]["f1_macro"]["mean"]) * 100.0 if ec else 0.0)
        c_s.append(float(ec["metrics"]["f1_macro"]["std"]) * 100.0 if ec else 0.0)
        lf_m.append(float(elf["metrics"]["f1_macro"]["mean"]) * 100.0 if elf else 0.0)
        lf_s.append(float(elf["metrics"]["f1_macro"]["std"]) * 100.0 if elf else 0.0)
        sc_m.append(float(esc["metrics"]["f1_macro"]["mean"]) * 100.0 if esc else 0.0)
        sc_s.append(float(esc["metrics"]["f1_macro"]["std"]) * 100.0 if esc else 0.0)

    r_c = ax1.bar(
        x1 - w1,
        c_m,
        w1,
        yerr=c_s,
        capsize=3,
        label="Clean (0% Poisoned)",
        color="#27ae60",
        edgecolor="white",
    )
    r_lf = ax1.bar(
        x1,
        lf_m,
        w1,
        yerr=lf_s,
        capsize=3,
        label="Label Flip (1 of 3 = 33% Malicious)",
        color="#e74c3c",
        edgecolor="white",
    )
    r_sc = ax1.bar(
        x1 + w1,
        sc_m,
        w1,
        yerr=sc_s,
        capsize=3,
        label="Update Scaling (1 of 3 = 33% Malicious)",
        color="#2980b9",
        edgecolor="white",
    )

    for rects in (r_c, r_lf, r_sc):
        for r in rects:
            h = r.get_height()
            ax1.annotate(
                f"{h:.1f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.3,
            )

    ax1.set_xticks(x1)
    ax1.set_xticklabels([lbl for _, lbl in n3_aggs], fontsize=9.0)
    ax1.set_ylabel("Global Test Macro-F1 (%)")
    ax1.set_ylim(0.0, 110.0)
    ax1.set_title(
        "Few-Tenant Breakdown (N=3 Domain-Specialized Tenants):\nWhy TrimmedMean Collapses When ⌊β·N⌋ = 0",
        fontsize=10.2,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Step 04 — Non-IID Honest False-Rejection Trade-Off & Few-Tenant (N=3) Breakdown (5 Seeds)",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def run_step04_full_sweep(
    *,
    out_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS_DEFAULT,
    rounds: int = 16,
    local_epochs: int = 2,
    batch_size: int = 256,
    lr: float = 0.0015,
    max_train_samples: int = 6000,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute the complete Step 04 5-seed Poisoning & Robust Aggregation benchmark with checkpointing."""
    if device is None:
        device = get_default_device(prefer_gpu=True)
    target_dir = out_dir or (REPO_ROOT / "results" / "04_robust")
    target_dir.mkdir(parents=True, exist_ok=True)

    attack_catalog: dict[str, tuple[str, dict[str, Any] | None]] = {
        "none": ("none", None),
        "label_flip": (
            "label_flip",
            {"flip_fraction": 1.0, "bidirectional": True, "update_scale": 8.5},
        ),
        "backdoor": (
            "backdoor",
            {
                "trigger_features": (3, 6, 8),
                "trigger_value": 4.5,
                "target_class": 0,
                "poison_ratio": 0.5,
                "replacement_scale": 6.0,
                "collateral_flip_ratio": 0.5,
            },
        ),
        "scaling": (
            "scaling",
            {"scale_factor": 10.0, "sign_flip": True},
        ),
        "backdoor_stealth": (
            "backdoor",
            {
                "trigger_features": (3, 6, 8),
                "trigger_value": 4.5,
                "target_class": 0,
                "poison_ratio": 0.5,
                "replacement_scale": 1.4,
                "collateral_flip_ratio": 0.0,
            },
        ),
    }

    all_6_aggs = [
        "fedavg",
        "static_weight",
        "trimmed_mean_b0.1",
        "trimmed_mean_b0.2",
        "median",
        "clustering",
    ]
    core_4_aggs = ["fedavg", "static_weight", "trimmed_mean_b0.2", "clustering"]

    # Build the list of (grid_name, config_key, ds_name, ptype, n_ten, alpha, atk_label, pf, agg_name, ev_every)
    jobs: list[tuple[str, str, str, str, int, float, str, float, str, int]] = []

    # 1. Primary Grid: InSDN Dirichlet alpha=0.5, N=10 (60 configs)
    for agg in all_6_aggs:
        jobs.append(
            (
                "insdn_dirichlet_a0.5_n10",
                f"none__pf0.00__{agg}",
                "insdn",
                "dirichlet",
                10,
                0.5,
                "none",
                0.00,
                agg,
                8,
            )
        )
    for atk_lbl in ("label_flip", "backdoor", "scaling"):
        for pf in (0.10, 0.20, 0.30):
            ev = 8 if abs(pf - 0.20) < 1e-9 else 0
            for agg in all_6_aggs:
                jobs.append(
                    (
                        "insdn_dirichlet_a0.5_n10",
                        f"{atk_lbl}__pf{pf:.2f}__{agg}",
                        "insdn",
                        "dirichlet",
                        10,
                        0.5,
                        atk_lbl,
                        pf,
                        agg,
                        ev,
                    )
                )

    # 2. Natural Domain-Specialty Non-IID Grid: InSDN Natural, N=10 (16 configs)
    for agg in core_4_aggs:
        jobs.append(
            (
                "insdn_natural_n10",
                f"none__pf0.00__{agg}",
                "insdn",
                "natural",
                10,
                0.0,
                "none",
                0.00,
                agg,
                0,
            )
        )
    for atk_lbl in ("label_flip", "backdoor", "scaling"):
        for agg in core_4_aggs:
            jobs.append(
                (
                    "insdn_natural_n10",
                    f"{atk_lbl}__pf0.20__{agg}",
                    "insdn",
                    "natural",
                    10,
                    0.0,
                    atk_lbl,
                    0.20,
                    agg,
                    0,
                )
            )

    # 3. Modern 2026 SDN-IoT Dataset Grid: ASEADOS-SDN-IoT Dirichlet alpha=0.5, N=10 (28 configs)
    for agg in core_4_aggs:
        jobs.append(
            (
                "aseados_dirichlet_a0.5_n10",
                f"none__pf0.00__{agg}",
                "aseados_sdn_iot",
                "dirichlet",
                10,
                0.5,
                "none",
                0.00,
                agg,
                0,
            )
        )
    for atk_lbl in ("label_flip", "backdoor", "scaling"):
        for pf in (0.20, 0.30):
            for agg in core_4_aggs:
                jobs.append(
                    (
                        "aseados_dirichlet_a0.5_n10",
                        f"{atk_lbl}__pf{pf:.2f}__{agg}",
                        "aseados_sdn_iot",
                        "dirichlet",
                        10,
                        0.5,
                        atk_lbl,
                        pf,
                        agg,
                        0,
                    )
                )

    # 4. Failure-Mode & Ablation Diagnostics on InSDN (28 configs)
    # 4a. IID Control (N=10)
    for agg in ("fedavg", "clustering"):
        jobs.append(
            (
                "failure_mode_ablations",
                f"iid_n10_clean__{agg}",
                "insdn",
                "iid",
                10,
                0.0,
                "none",
                0.00,
                agg,
                0,
            )
        )
        jobs.append(
            (
                "failure_mode_ablations",
                f"iid_n10_scaling_pf0.20__{agg}",
                "insdn",
                "iid",
                10,
                0.0,
                "scaling",
                0.20,
                agg,
                0,
            )
        )
    # 4b. Extreme Non-IID (Dirichlet alpha=0.1, N=10)
    for agg in ("fedavg", "trimmed_mean_b0.2", "median", "clustering"):
        jobs.append(
            (
                "failure_mode_ablations",
                f"dir01_n10_clean__{agg}",
                "insdn",
                "dirichlet",
                10,
                0.1,
                "none",
                0.00,
                agg,
                0,
            )
        )
        jobs.append(
            (
                "failure_mode_ablations",
                f"dir01_n10_scaling_pf0.20__{agg}",
                "insdn",
                "dirichlet",
                10,
                0.1,
                "scaling",
                0.20,
                agg,
                0,
            )
        )
    # 4c. Few-Tenant Breakdown (Natural Non-IID, N=3, 1 of 3 = 33.3% poisoned)
    for agg in ("fedavg", "trimmed_mean_b0.2", "median", "clustering"):
        jobs.append(
            (
                "failure_mode_ablations",
                f"natural_n3_clean__{agg}",
                "insdn",
                "natural",
                3,
                0.0,
                "none",
                0.00,
                agg,
                0,
            )
        )
        jobs.append(
            (
                "failure_mode_ablations",
                f"natural_n3_label_flip_pf0.33__{agg}",
                "insdn",
                "natural",
                3,
                0.0,
                "label_flip",
                0.33,
                agg,
                0,
            )
        )
        jobs.append(
            (
                "failure_mode_ablations",
                f"natural_n3_scaling_pf0.33__{agg}",
                "insdn",
                "natural",
                3,
                0.0,
                "scaling",
                0.33,
                agg,
                0,
            )
        )
    # 4d. Norm-Bounded Stealth Backdoor (scale = 1.4x at 30% poisoning)
    for agg in core_4_aggs:
        jobs.append(
            (
                "failure_mode_ablations",
                f"stealth_backdoor_pf0.30__{agg}",
                "insdn",
                "dirichlet",
                10,
                0.5,
                "backdoor_stealth",
                0.30,
                agg,
                0,
            )
        )

    sweep_summary: dict[str, Any] = {
        "step": "04_robust",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "fl_defaults": {
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "optimizer": "Adam",
            "lr": float(lr),
            "weight_decay": 1e-4,
            "seeds": list(seeds),
            "max_train_samples_stratified": int(max_train_samples),
            "attack_catalog": {
                k: {"attack_name": v[0], "kwargs": v[1]} for k, v in attack_catalog.items()
            },
        },
        "grids": {
            "insdn_dirichlet_a0.5_n10": {},
            "insdn_natural_n10": {},
            "aseados_dirichlet_a0.5_n10": {},
            "failure_mode_ablations": {},
        },
        "statistical_tests": {},
    }

    ckpt_path = target_dir / "summary_checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            for gname in sweep_summary["grids"]:
                sweep_summary["grids"][gname] = ckpt.get("grids", {}).get(gname, {})
        except Exception:
            pass

    bundles_cache: dict[str, dict[str, Any]] = {}
    model_cfg = {
        "type": "gru",
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "seq_mode": "single_step",
        "num_chunks": 1,
    }

    t_sweep_start = time.perf_counter()
    total_jobs = len(jobs)

    for job_idx, (
        grid_name,
        cfg_key,
        ds_name,
        ptype,
        n_ten,
        alpha,
        atk_label,
        pf,
        agg_name,
        ev_every,
    ) in enumerate(jobs, start=1):
        if cfg_key in sweep_summary["grids"][grid_name]:
            print(
                f"[Step 04] ({job_idx:3d}/{total_jobs}) Resuming from checkpoint: "
                f"{grid_name} | {cfg_key} already done.",
                flush=True,
            )
            continue

        if ds_name not in bundles_cache:
            bundles_cache[ds_name] = load_dataset_artifacts(ds_name)
        bundle = bundles_cache[ds_name]
        atk_real_name, atk_kwargs = attack_catalog[atk_label]

        t_job = time.perf_counter()
        seed_runs: list[dict[str, Any]] = []
        for s in seeds:
            sr = run_single_seed_fl(
                bundle,
                task="binary",
                partition_type=ptype,
                num_tenants=n_ten,
                dirichlet_alpha=alpha,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=s,
                model_cfg=model_cfg,
                aggregator_name=agg_name,
                attack_name=atk_real_name,
                poison_fraction=pf,
                attack_kwargs=atk_kwargs,
                scaler_mode="global",
                max_train_samples=max_train_samples,
                eval_every=ev_every,
                eval_per_tenant_every=0,
                device=device,
            )
            seed_runs.append(sr)

        reg_summary = summarize_multi_seed_fl(seed_runs)
        reg_summary["attack_label"] = atk_label
        reg_summary["seed_f1_macros"] = [float(sr["final_metrics"]["f1_macro"]) for sr in seed_runs]
        reg_summary["seed_accuracies"] = [
            float(sr["final_metrics"]["accuracy"]) for sr in seed_runs
        ]
        reg_summary["seed_backdoor_asrs"] = [
            float(sr["final_metrics"]["backdoor_asr"]) for sr in seed_runs
        ]
        reg_summary["seed_honest_frrs"] = [
            float(sr["final_metrics"]["mean_honest_false_rejection_rate"]) for sr in seed_runs
        ]
        reg_summary["seed_malicious_trrs"] = [
            float(sr["final_metrics"]["mean_malicious_rejection_rate"]) for sr in seed_runs
        ]

        sweep_summary["grids"][grid_name][cfg_key] = reg_summary
        ckpt_path.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

        f1_m = reg_summary["metrics"]["f1_macro"]["mean"] * 100.0
        f1_s = reg_summary["metrics"]["f1_macro"]["std"] * 100.0
        asr_m = reg_summary["metrics"]["backdoor_asr"]["mean"] * 100.0
        frr_m = reg_summary["metrics"]["mean_honest_false_rejection_rate"]["mean"] * 100.0
        trr_m = reg_summary["metrics"]["mean_malicious_rejection_rate"]["mean"] * 100.0
        dt = time.perf_counter() - t_job
        print(
            f"[Step 04] ({job_idx:3d}/{total_jobs}) {grid_name:24s} | {cfg_key:34s} "
            f"-> F1={f1_m:6.2f}±{f1_s:4.2f}% | ASR={asr_m:5.1f}% | H-FRR={frr_m:4.1f}% | M-TRR={trr_m:5.1f}% ({dt:.1f}s)",
            flush=True,
        )

    # Reconstruct flat per_seed_summary.csv and per_round_metrics.csv from completed sweep_summary
    all_per_seed_rows: list[dict[str, Any]] = []
    all_per_round_rows: list[dict[str, Any]] = []

    for grid_name, grid_dict in sweep_summary["grids"].items():
        for cfg_key, reg_sum in grid_dict.items():
            m = reg_sum["metrics"]
            seeds_list = reg_sum.get("seeds", list(seeds))
            f1_list = reg_sum.get("seed_f1_macros", [])
            acc_list = reg_sum.get("seed_accuracies", [])
            asr_list = reg_sum.get("seed_backdoor_asrs", [])
            frr_list = reg_sum.get("seed_honest_frrs", [])
            trr_list = reg_sum.get("seed_malicious_trrs", [])
            for idx_s, s_val in enumerate(seeds_list):
                all_per_seed_rows.append(
                    {
                        "grid": grid_name,
                        "config_key": cfg_key,
                        "dataset": reg_sum["dataset"],
                        "partition_type": reg_sum["partition_type"],
                        "num_tenants": reg_sum["num_tenants"],
                        "dirichlet_alpha": reg_sum["dirichlet_alpha"],
                        "aggregator": reg_sum["aggregator"],
                        "attack_label": reg_sum.get("attack_label", reg_sum.get("attack_name")),
                        "poison_fraction": reg_sum["poison_fraction"],
                        "num_malicious_clients": reg_sum["num_malicious_clients"],
                        "seed": s_val,
                        "f1_macro": (
                            f1_list[idx_s] if idx_s < len(f1_list) else m["f1_macro"]["mean"]
                        ),
                        "accuracy": (
                            acc_list[idx_s] if idx_s < len(acc_list) else m["accuracy"]["mean"]
                        ),
                        "backdoor_asr": (
                            asr_list[idx_s] if idx_s < len(asr_list) else m["backdoor_asr"]["mean"]
                        ),
                        "honest_false_rejection_rate": (
                            frr_list[idx_s]
                            if idx_s < len(frr_list)
                            else m["mean_honest_false_rejection_rate"]["mean"]
                        ),
                        "malicious_rejection_rate": (
                            trr_list[idx_s]
                            if idx_s < len(trr_list)
                            else m["mean_malicious_rejection_rate"]["mean"]
                        ),
                        "mean_round_aggregation_time_sec": m["mean_round_aggregation_time_sec"][
                            "mean"
                        ],
                        "bytes_per_round": int(m["bytes_per_round"]["mean"]),
                    }
                )
            for c in reg_sum.get("round_curves", []):
                all_per_round_rows.append(
                    {
                        "grid": grid_name,
                        "config_key": cfg_key,
                        "dataset": reg_sum["dataset"],
                        "aggregator": reg_sum["aggregator"],
                        "attack_label": reg_sum.get("attack_label", reg_sum.get("attack_name")),
                        "poison_fraction": reg_sum["poison_fraction"],
                        "round": c["round"],
                        "train_loss_mean": c.get("train_loss_mean"),
                        "f1_macro_mean": c.get("f1_macro_mean"),
                        "accuracy_mean": c.get("accuracy_mean"),
                        "aggregation_time_sec_mean": c.get("aggregation_time_sec_mean"),
                    }
                )

    # Paired Wilcoxon tests comparing robust aggregators vs FedAvg at pf=0.20 & 0.30
    insdn_g = sweep_summary["grids"]["insdn_dirichlet_a0.5_n10"]
    stat_tests: dict[str, Any] = {}
    for atk_lbl in ("label_flip", "backdoor", "scaling"):
        for pf in (0.20, 0.30):
            fed_entry = _get_step04_entry(insdn_g, atk_lbl, pf, "fedavg")
            if not fed_entry:
                continue
            for agg in (
                "static_weight",
                "trimmed_mean_b0.1",
                "trimmed_mean_b0.2",
                "median",
                "clustering",
            ):
                rob_entry = _get_step04_entry(insdn_g, atk_lbl, pf, agg)
                if not rob_entry:
                    continue
                p_f1 = compute_paired_wilcoxon(
                    fed_entry["seed_f1_macros"],
                    rob_entry["seed_f1_macros"],
                )
                p_asr = compute_paired_wilcoxon(
                    fed_entry["seed_backdoor_asrs"],
                    rob_entry["seed_backdoor_asrs"],
                )
                stat_tests[f"{atk_lbl}_pf{pf:.2f}_{agg}_vs_fedavg"] = {
                    "fedavg_f1_mean": fed_entry["metrics"]["f1_macro"]["mean"],
                    "robust_f1_mean": rob_entry["metrics"]["f1_macro"]["mean"],
                    "f1_gain_pts": (
                        rob_entry["metrics"]["f1_macro"]["mean"]
                        - fed_entry["metrics"]["f1_macro"]["mean"]
                    )
                    * 100.0,
                    "wilcoxon_p_f1": p_f1,
                    "fedavg_asr_mean": fed_entry["metrics"]["backdoor_asr"]["mean"],
                    "robust_asr_mean": rob_entry["metrics"]["backdoor_asr"]["mean"],
                    "wilcoxon_p_asr": p_asr,
                }
    sweep_summary["statistical_tests"] = stat_tests
    sweep_summary["total_sweep_wall_time_sec"] = float(time.perf_counter() - t_sweep_start)

    pd.DataFrame(all_per_seed_rows).to_csv(target_dir / "per_seed_summary.csv", index=False)
    pd.DataFrame(all_per_round_rows).to_csv(target_dir / "per_round_metrics.csv", index=False)
    (target_dir / "summary.json").write_text(
        json.dumps(sweep_summary, indent=2),
        encoding="utf-8",
    )

    # Generate all 3 Step 04 plots
    plot_accuracy_vs_poison_fraction(
        sweep_summary,
        target_dir / "accuracy_vs_poison_fraction.png",
    )
    plot_backdoor_asr_comparison(
        sweep_summary,
        target_dir / "backdoor_asr_comparison.png",
    )
    plot_honest_false_rejection_rate(
        sweep_summary,
        target_dir / "honest_false_rejection_rate.png",
    )
    return sweep_summary


def run_single_seed_dp_fl(
    bundle: dict[str, Any],
    *,
    dp_policy: str = "adaptive",
    mean_epsilon: float | None = 2.0,
    delta: float = 1e-5,
    max_grad_norm: float = 0.5,
    expected_batch_size: int = 512,
    rounds: int = 12,
    local_epochs: int = 1,
    lr: float = 0.0025,
    weight_decay: float = 1e-4,
    max_train_samples: int = 12000,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run one 3-tenant Natural Non-IID FL experiment with Uniform/Adaptive DP-SGD or No-DP."""
    if device is None:
        device = get_default_device(prefer_gpu=True)

    sp = bundle["splits"]
    y_train_all = sp["y_bin_train"]
    y_val_all = sp["y_bin_val"]
    y_test_all = sp["y_bin_test"]
    X_train_global = sp["X_train_global_scaled"]
    X_val_global = sp["X_val_global_scaled"]
    X_test_global = sp["X_test_global_scaled"]
    input_dim = int(X_train_global.shape[1])

    train_map_raw, test_map = resolve_tenant_partitions(
        bundle,
        partition_type="natural",
        num_tenants=3,
        seed=42,
    )
    train_map = subsample_tenant_indices_stratified(
        train_map_raw,
        sp["y_mul_train"],
        max_total_samples=max_train_samples,
        seed=42 + seed,
    )

    tenant_names = sorted(train_map.keys())
    num_tenants = len(tenant_names)
    tenant_labels_raw = {t: sp["y_mul_train"][train_map_raw[t]] for t in tenant_names}
    sensitivities = compute_sensitivity_scores(
        tenant_labels_raw,
        num_classes=len(bundle["class_order"]),
    )
    scores_map = {t: sensitivities[t].score for t in tenant_names}

    pol_clean = dp_policy.strip().lower()
    is_dp = pol_clean != "none" and mean_epsilon is not None and math.isfinite(float(mean_epsilon))
    if is_dp:
        eps_total = float(num_tenants) * float(mean_epsilon)
        allocated_eps = allocate_budgets(scores_map, eps_total, policy=pol_clean)
    else:
        eps_total = float("inf")
        allocated_eps = {t: float("inf") for t in tenant_names}

    set_all_seeds(seed)
    global_model = build_model(
        "gru",
        input_dim=input_dim,
        num_classes=2,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        seq_mode="single_step",
        num_chunks=1,
    ).to(device)
    init_flat = global_model.get_flat_params()

    clients: list[FLClient] = []
    for idx, t_name in enumerate(tenant_names):
        tr_idx = train_map[t_name]
        te_idx = test_map[t_name]
        c_model = build_model(
            "gru",
            input_dim=input_dim,
            num_classes=2,
            hidden_dim=64,
            num_layers=2,
            dropout=0.0,
            seq_mode="single_step",
            num_chunks=1,
        ).to(device)
        c_model.set_flat_params(init_flat)

        if is_dp:
            dpcfg = DPConfig(
                target_epsilon=float(allocated_eps[t_name]),
                delta=float(delta),
                max_grad_norm=float(max_grad_norm),
                expected_batch_size=int(expected_batch_size),
            )
            clients.append(
                DPFLClient(
                    client_id=t_name,
                    client_index=idx,
                    model=c_model,
                    X_train=X_train_global[tr_idx],
                    y_train=y_train_all[tr_idx],
                    dp_config=dpcfg,
                    rounds=int(rounds),
                    local_epochs=int(local_epochs),
                    X_test=X_test_global[te_idx],
                    y_test=y_test_all[te_idx],
                    device=device,
                )
            )
        else:
            clients.append(
                FLClient(
                    client_id=t_name,
                    client_index=idx,
                    model=c_model,
                    X_train=X_train_global[tr_idx],
                    y_train=y_train_all[tr_idx],
                    X_test=X_test_global[te_idx],
                    y_test=y_test_all[te_idx],
                    device=device,
                )
            )

    server = FLServer(
        global_model=global_model,
        clients=clients,
        aggregator=build_aggregator("fedavg"),
        X_test=X_test_global,
        y_test=y_test_all,
        X_val=X_val_global,
        y_val=y_val_all,
        class_names=["Benign", "Attack"],
        device=device,
    )
    fit_res = server.fit(
        rounds=int(rounds),
        local_epochs=int(local_epochs),
        batch_size=int(expected_batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        seed=int(seed),
        eval_every=max(1, int(rounds) // 2),
        eval_per_tenant_every=int(rounds),
    )

    # Evaluate per-tenant local & global MIA and collect per-tenant DP metadata
    per_tenant_details: dict[str, dict[str, Any]] = {}
    local_mia_aucs: list[float] = []
    global_mia_aucs: list[float] = []
    local_mia_accs: list[float] = []
    global_mia_accs: list[float] = []

    for c in clients:
        t_name = c.client_id
        tr_idx = train_map[t_name]
        te_idx = test_map[t_name]
        X_t_tr = X_train_global[tr_idx]
        y_t_tr = y_train_all[tr_idx]
        X_t_te = X_test_global[te_idx]
        y_t_te = y_test_all[te_idx]

        mia_local = evaluate_membership_inference(
            c.model,
            X_t_tr,
            y_t_tr,
            X_t_te,
            y_t_te,
            max_eval_samples=1500,
            seed=42 + seed,
            device=device,
        )
        mia_global = evaluate_membership_inference(
            server.global_model,
            X_t_tr,
            y_t_tr,
            X_t_te,
            y_t_te,
            max_eval_samples=1500,
            seed=42 + seed,
            device=device,
        )
        t_eval = c.evaluate_local_test(server.global_model)
        ts = sensitivities[t_name]

        if isinstance(c, DPFLClient):
            eps_spent = float(c.epsilon_spent)
            sigma_val = float(c.noise_multiplier)
            q_val = float(c.sample_rate)
            dp_steps_val = int(c.accountant.steps)
        else:
            eps_spent = float("inf")
            sigma_val = 0.0
            q_val = float(min(1.0, expected_batch_size / max(1, c.num_samples)))
            dp_steps_val = 0

        local_mia_aucs.append(float(mia_local["mia_auc"]))
        global_mia_aucs.append(float(mia_global["mia_auc"]))
        local_mia_accs.append(float(mia_local["mia_accuracy"]))
        global_mia_accs.append(float(mia_global["mia_accuracy"]))

        per_tenant_details[t_name] = {
            "tenant": t_name,
            "num_samples": int(c.num_samples),
            "domain_profile": float(ts.domain_profile),
            "rare_share": float(ts.rare_share),
            "imbalance": float(ts.imbalance),
            "data_score": float(ts.data_score),
            "sensitivity_score": float(ts.score),
            "allocated_epsilon": float(allocated_eps[t_name]),
            "epsilon_spent": eps_spent,
            "delta": float(delta) if is_dp else 0.0,
            "noise_multiplier": sigma_val,
            "sample_rate": q_val,
            "dp_steps_total": dp_steps_val,
            "f1_macro": float(t_eval.get("f1_macro", 0.0)),
            "accuracy": float(t_eval.get("accuracy", 0.0)),
            "mia_local": mia_local,
            "mia_global": mia_global,
        }

    fit_res["dataset"] = bundle["dataset"]
    fit_res["task"] = "binary"
    fit_res["partition_type"] = "natural"
    fit_res["num_tenants"] = int(num_tenants)
    fit_res["dirichlet_alpha"] = None
    fit_res["scaler_mode"] = "global"
    fit_res["attack_name"] = "none"
    fit_res["poison_fraction"] = 0.0
    fit_res["num_malicious_clients"] = 0
    fit_res["final_metrics"].setdefault("backdoor_asr", 0.0)
    fit_res["final_metrics"].setdefault("clean_evasion_rate", 0.0)
    fit_res["final_metrics"].setdefault("asr_gain", 0.0)
    fit_res["final_metrics"].setdefault("mean_honest_false_rejection_rate", 0.0)
    fit_res["final_metrics"].setdefault("mean_malicious_rejection_rate", 0.0)
    fit_res["dp_policy"] = pol_clean
    fit_res["mean_epsilon"] = float(mean_epsilon) if is_dp and mean_epsilon is not None else None
    fit_res["epsilon_total"] = float(eps_total) if is_dp else None
    fit_res["delta"] = float(delta) if is_dp else 0.0
    fit_res["max_grad_norm"] = float(max_grad_norm) if is_dp else None
    fit_res["per_tenant_dp"] = per_tenant_details
    fit_res["final_metrics"]["mia_local_auc_mean"] = float(np.mean(local_mia_aucs))
    fit_res["final_metrics"]["mia_global_auc_mean"] = float(np.mean(global_mia_aucs))
    fit_res["final_metrics"]["mia_local_acc_mean"] = float(np.mean(local_mia_accs))
    fit_res["final_metrics"]["mia_global_acc_mean"] = float(np.mean(global_mia_accs))
    fit_res["final_metrics"]["most_sensitive_tenant_f1"] = float(
        per_tenant_details["tenant_1_finance"]["f1_macro"]
    )
    fit_res["final_metrics"]["most_sensitive_tenant_acc"] = float(
        per_tenant_details["tenant_1_finance"]["accuracy"]
    )
    return fit_res


def run_rare_cohort_mia_audit(
    bundle: dict[str, Any],
    *,
    dp_policy: str,
    mean_epsilon: float | None,
    scores_map: dict[str, float],
    delta: float = 1e-5,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run high-risk rare-attack cohort MIA audit on `tenant_1_finance` for a single seed."""
    if device is None:
        device = get_default_device(prefer_gpu=True)
    sp = bundle["splits"]
    train_map_raw, test_map = resolve_tenant_partitions(
        bundle,
        partition_type="natural",
        num_tenants=3,
        seed=42,
    )
    rng = np.random.default_rng(42 + int(seed) * 97)
    fin_tr = train_map_raw["tenant_1_finance"]
    fin_te = test_map["tenant_1_finance"]
    y_tr_m = sp["y_mul_train"][fin_tr]
    y_te_m = sp["y_mul_test"][fin_te]

    m_list, nm_list = [], []
    for c_id in np.unique(y_tr_m):
        tr_c = fin_tr[y_tr_m == c_id]
        te_c = fin_te[y_te_m == c_id]
        k = min(12, len(tr_c), len(te_c))
        if k >= 2:
            m_list.append(rng.choice(tr_c, size=k, replace=False))
            nm_list.append(rng.choice(te_c, size=k, replace=False))
    m_idx = np.concatenate(m_list)
    nm_idx = np.concatenate(nm_list)
    X_mem = sp["X_train_global_scaled"][m_idx]
    y_mem = sp["y_mul_train"][m_idx]
    X_non = sp["X_test_global_scaled"][nm_idx]
    y_non = sp["y_mul_test"][nm_idx]
    num_cls = len(bundle["class_order"])

    set_all_seeds(seed)
    m = build_model(
        "gru",
        input_dim=int(X_mem.shape[1]),
        num_classes=num_cls,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
        seq_mode="single_step",
        num_chunks=1,
    ).to(device)
    init_flat = m.get_flat_params()

    pol_clean = dp_policy.strip().lower()
    is_dp = pol_clean != "none" and mean_epsilon is not None and math.isfinite(float(mean_epsilon))
    if not is_dp:
        c = FLClient("tenant_1_finance", 0, m, X_mem, y_mem, device=device)
        c.local_train(
            init_flat,
            round_idx=1,
            local_epochs=45,
            batch_size=32,
            lr=0.008,
            weight_decay=0.0,
            seed=seed,
        )
        eps_alloc = float("inf")
        sigma_val = 0.0
    else:
        alloc = allocate_budgets(scores_map, 3.0 * float(mean_epsilon), policy=pol_clean)
        eps_alloc = float(alloc["tenant_1_finance"])
        dpcfg = DPConfig(
            target_epsilon=eps_alloc,
            delta=float(delta),
            max_grad_norm=0.5,
            expected_batch_size=32,
        )
        c = DPFLClient(
            "tenant_1_finance",
            0,
            m,
            X_mem,
            y_mem,
            dp_config=dpcfg,
            rounds=1,
            local_epochs=25,
            device=device,
        )
        c.local_train(
            init_flat,
            round_idx=1,
            local_epochs=25,
            batch_size=32,
            lr=0.008,
            weight_decay=0.0,
            seed=seed,
        )
        sigma_val = float(c.noise_multiplier)

    mia = evaluate_membership_inference(
        c.model,
        X_mem,
        y_mem,
        X_non,
        y_non,
        seed=42 + seed,
        device=device,
    )
    mia["allocated_epsilon_finance"] = eps_alloc
    mia["noise_multiplier_finance"] = sigma_val
    return mia


def plot_privacy_utility_tradeoff(
    sweep_05: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot Global & Per-Tenant Macro-F1 (%) vs. Mean Per-Tenant Budget eps in {0.5, 1, 2, 4, 8, inf}."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=160)

    eps_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    datasets = [
        ("insdn", "InSDN (SDN Native, Natural Non-IID, N=3)"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT (2026 Modern SDN-IoT, Natural Non-IID, N=3)"),
    ]

    for col_idx, (ds_key, ds_title) in enumerate(datasets):
        ds_grid = sweep_05["datasets"].get(ds_key, {})
        no_dp = ds_grid.get("none__eps_inf", {})
        no_dp_f1 = float(no_dp.get("metrics", {}).get("f1_macro", {}).get("mean", 0.0)) * 100.0

        # Top row: Global Test Macro-F1 (%) — Uniform vs. Adaptive vs. No-DP
        ax_top = axes[0, col_idx]
        ax_top.axhline(
            no_dp_f1,
            color="#111111",
            linestyle="--",
            linewidth=1.8,
            label=f"No-DP Baseline (ε = ∞): {no_dp_f1:.2f}%",
        )
        for pol_key, pol_label, color, marker, lstyle in [
            ("uniform", "Uniform Budget (ε_i = ε_total / N)", "#d62728", "s", "--"),
            ("adaptive", "Adaptive Budget (ε_i ∝ 1 / s_i)", "#1f77b4", "o", "-"),
        ]:
            xs, means, stds = [], [], []
            for eps_val in eps_grid:
                entry = ds_grid.get(f"{pol_key}__eps_{eps_val:.1f}")
                if not entry:
                    continue
                xs.append(eps_val)
                means.append(float(entry["metrics"]["f1_macro"]["mean"]) * 100.0)
                stds.append(float(entry["metrics"]["f1_macro"]["std"]) * 100.0)
            if xs:
                ax_top.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    label=pol_label,
                    color=color,
                    marker=marker,
                    linestyle=lstyle,
                    linewidth=2.2,
                    markersize=6.5,
                    capsize=4.0,
                )
        ax_top.set_xscale("log", base=2)
        ax_top.set_xticks(eps_grid)
        ax_top.set_xticklabels([str(e) for e in eps_grid])
        ax_top.set_xlabel("Mean Per-Tenant Privacy Budget ε̄ (δ = 10⁻⁵, log₂ scale)")
        ax_top.set_ylabel("Global Test Macro-F1 (%)")
        ax_top.set_title(
            f"{ds_title}\nGlobal Detection Macro-F1 vs. Privacy Budget ε̄",
            fontsize=10.2,
            fontweight="bold",
        )
        ax_top.grid(True, linestyle="--", alpha=0.35)
        ax_top.legend(loc="lower right", fontsize=8.4, framealpha=0.92)

        # Bottom row: Per-Tenant Test Macro-F1 (%) under Adaptive vs Uniform DP
        ax_bot = axes[1, col_idx]
        tenant_specs = [
            ("tenant_1_finance", "Finance (s₁=1.00, Most Sensitive)", "#c0392b"),
            ("tenant_2_healthcare", "Healthcare (High Sensitivity)", "#8e44ad"),
            ("tenant_3_education", "Education (Lower Sensitivity)", "#27ae60"),
        ]
        for t_key, t_lbl, color in tenant_specs:
            for pol_key, lstyle, marker, pol_short in [
                ("adaptive", "-", "o", "Adaptive"),
                ("uniform", ":", "s", "Uniform"),
            ]:
                xs, means = [], []
                for eps_val in eps_grid:
                    entry = ds_grid.get(f"{pol_key}__eps_{eps_val:.1f}")
                    if not entry:
                        continue
                    t_info = entry.get("per_tenant_summary", {}).get(t_key, {})
                    xs.append(eps_val)
                    means.append(float(t_info.get("f1_macro_mean", 0.0)) * 100.0)
                if xs:
                    ax_bot.plot(
                        xs,
                        means,
                        label=f"{t_lbl} — {pol_short}",
                        color=color,
                        linestyle=lstyle,
                        marker=marker,
                        linewidth=2.0 if pol_key == "adaptive" else 1.5,
                        markersize=5.5,
                        alpha=0.95 if pol_key == "adaptive" else 0.72,
                    )
        ax_bot.set_xscale("log", base=2)
        ax_bot.set_xticks(eps_grid)
        ax_bot.set_xticklabels([str(e) for e in eps_grid])
        ax_bot.set_xlabel("Mean Per-Tenant Privacy Budget ε̄ (δ = 10⁻⁵, log₂ scale)")
        ax_bot.set_ylabel("Per-Tenant Local Test Macro-F1 (%)")
        ax_bot.set_title(
            f"{ds_title}\nPer-Tenant Macro-F1 Trajectory (Adaptive Solid vs. Uniform Dotted)",
            fontsize=10.2,
            fontweight="bold",
        )
        ax_bot.grid(True, linestyle="--", alpha=0.35)
        ax_bot.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.92)

    fig.suptitle(
        "Step 05 — Privacy-Utility Trade-Off Across ε̄ ∈ {0.5, 1, 2, 4, 8, ∞}: Adaptive vs. Uniform Per-Tenant DP-SGD (5 Seeds)",
        fontsize=11.8,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_tenant_dp_comparison(
    sweep_05: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot per-tenant sensitivity score breakdown, allocated eps_i, and calibrated sigma_i at eps_bar=2.0."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), dpi=160)

    insdn_grid = sweep_05["datasets"].get("insdn", {})
    adap_2 = insdn_grid.get("adaptive__eps_2.0", {}).get("per_tenant_summary", {})
    unif_2 = insdn_grid.get("uniform__eps_2.0", {}).get("per_tenant_summary", {})

    tenants = [
        ("tenant_1_finance", "Tenant 1:\nFinance"),
        ("tenant_2_healthcare", "Tenant 2:\nHealthcare"),
        ("tenant_3_education", "Tenant 3:\nEducation"),
    ]
    x = np.arange(len(tenants))
    w = 0.25

    # Panel A: Sensitivity Score Breakdown (Domain Profile, Data Score, Hybrid Score s_i)
    ax0 = axes[0]
    dom_vals = [float(adap_2.get(t, {}).get("domain_profile", 0.0)) for t, _ in tenants]
    dat_vals = [float(adap_2.get(t, {}).get("data_score", 0.0)) for t, _ in tenants]
    hyb_vals = [float(adap_2.get(t, {}).get("sensitivity_score", 0.0)) for t, _ in tenants]

    b_dom = ax0.bar(
        x - w, dom_vals, w, label="Domain Profile (0.5 weight)", color="#34495e", edgecolor="white"
    )
    b_dat = ax0.bar(
        x,
        dat_vals,
        w,
        label="Data Vulnerability Score (0.5 weight)",
        color="#e67e22",
        edgecolor="white",
    )
    b_hyb = ax0.bar(
        x + w,
        hyb_vals,
        w,
        label="Hybrid Sensitivity Score s_i",
        color="#c0392b",
        edgecolor="white",
    )
    for rects in (b_dom, b_dat, b_hyb):
        for r in rects:
            h = r.get_height()
            ax0.annotate(
                f"{h:.2f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )
    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for _, lbl in tenants], fontsize=9.2)
    ax0.set_ylim(0.0, 1.18)
    ax0.set_ylabel("Normalized Score in [0.5, 1.0]")
    ax0.set_title(
        "Hybrid Per-Tenant Sensitivity Score (s_i)\nBreakdown on InSDN (Natural Non-IID)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    # Panel B: Allocated Privacy Budget eps_i at eps_bar = 2.0 (eps_total = 6.0)
    ax1 = axes[1]
    w2 = 0.34
    u_eps = [float(unif_2.get(t, {}).get("allocated_epsilon", 2.0)) for t, _ in tenants]
    a_eps = [float(adap_2.get(t, {}).get("allocated_epsilon", 2.0)) for t, _ in tenants]

    r_ue = ax1.bar(
        x - w2 / 2,
        u_eps,
        w2,
        label="Uniform Policy (ε_i = 2.000)",
        color="#95a5a6",
        edgecolor="white",
    )
    r_ae = ax1.bar(
        x + w2 / 2,
        a_eps,
        w2,
        label="Adaptive Policy (ε_i ∝ 1/s_i, Σε_i = 6.0)",
        color="#2980b9",
        edgecolor="white",
    )
    for rects in (r_ue, r_ae):
        for r in rects:
            h = r.get_height()
            ax1.annotate(
                f"ε={h:.2f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in tenants], fontsize=9.2)
    ax1.set_ylim(0.0, max(u_eps + a_eps) * 1.22)
    ax1.set_ylabel("Allocated Privacy Budget ε_i (Lower = Stronger Privacy)")
    ax1.set_title(
        "Per-Tenant Privacy Budget Allocation at ε̄ = 2.0\n(Finance Gets 28.8% Tighter ε₁ = 1.42 vs. 2.00)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    # Panel C: Calibrated Noise Multiplier sigma_i across Mean Budgets eps_bar in {0.5, 1, 2, 4, 8}
    ax2 = axes[2]
    eps_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    for t_key, t_lbl, color, marker in [
        ("tenant_1_finance", "Finance (Adaptive σ₁)", "#c0392b", "o"),
        ("tenant_2_healthcare", "Healthcare (Adaptive σ₂)", "#8e44ad", "^"),
        ("tenant_3_education", "Education (Adaptive σ₃)", "#27ae60", "D"),
    ]:
        sigs = []
        for eps_val in eps_grid:
            t_sum = (
                insdn_grid.get(f"adaptive__eps_{eps_val:.1f}", {})
                .get("per_tenant_summary", {})
                .get(t_key, {})
            )
            sigs.append(float(t_sum.get("noise_multiplier", 0.0)))
        ax2.plot(
            eps_grid, sigs, label=t_lbl, color=color, marker=marker, linewidth=2.1, markersize=6.0
        )

    unif_fin_sigs = [
        float(
            insdn_grid.get(f"uniform__eps_{e:.1f}", {})
            .get("per_tenant_summary", {})
            .get("tenant_1_finance", {})
            .get("noise_multiplier", 0.0)
        )
        for e in eps_grid
    ]
    ax2.plot(
        eps_grid,
        unif_fin_sigs,
        label="Finance (Uniform σ₁)",
        color="#c0392b",
        linestyle="--",
        marker="s",
        linewidth=1.7,
        alpha=0.75,
    )
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(eps_grid)
    ax2.set_xticklabels([str(e) for e in eps_grid])
    ax2.set_xlabel("Mean Per-Tenant Budget ε̄ (δ = 10⁻⁵)")
    ax2.set_ylabel("Calibrated Gaussian Noise Multiplier σ_i")
    ax2.set_title(
        "RDP-Calibrated Noise Multiplier σ_i vs. Budget ε̄\n(Higher σ_i Enforces Stronger Record-Level Protection)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Step 05 — Per-Tenant Sensitivity Scoring (s_i), Budget Allocation (ε_i), and RDP Noise Calibration (σ_i)",
        fontsize=11.8,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def plot_mia_attack_evaluation(
    sweep_05: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot Membership Inference Attack (MIA) AUC & Accuracy across standard FL and rare-cohort audit."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.5), dpi=160)

    eps_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
    insdn_grid = sweep_05["datasets"].get("insdn", {})
    cohort_insdn = sweep_05["rare_cohort_mia"].get("insdn", {})

    # Panel A: High-Risk Rare-Attack Cohort MIA Audit on Finance (InSDN, 5 Seeds)
    ax0 = axes[0]
    no_dp_c = cohort_insdn.get("none__eps_inf", {})
    no_dp_auc = float(no_dp_c.get("mia_auc_mean", 0.5)) * 100.0
    no_dp_acc = float(no_dp_c.get("mia_accuracy_mean", 0.5)) * 100.0

    ax0.axhline(
        50.0,
        color="#7f8c8d",
        linestyle=":",
        linewidth=1.6,
        label="Ideal Random-Guessing Baseline (50.0%)",
    )
    ax0.axhline(
        no_dp_auc,
        color="#111111",
        linestyle="--",
        linewidth=1.8,
        label=f"No-DP Baseline MIA AUC ({no_dp_auc:.2f}%)",
    )
    ax0.axhline(
        no_dp_acc,
        color="#555555",
        linestyle="-.",
        linewidth=1.4,
        label=f"No-DP Baseline MIA Accuracy ({no_dp_acc:.2f}%)",
    )

    for pol_key, pol_lbl, color, marker in [
        ("uniform", "Uniform DP — Finance MIA AUC", "#e74c3c", "s"),
        ("adaptive", "Adaptive DP — Finance MIA AUC (Tighter ε₁)", "#1f77b4", "o"),
    ]:
        xs, aucs, stds = [], [], []
        for e_val in eps_grid:
            c_entry = cohort_insdn.get(f"{pol_key}__eps_{e_val:.1f}")
            if not c_entry:
                continue
            xs.append(e_val)
            aucs.append(float(c_entry["mia_auc_mean"]) * 100.0)
            stds.append(float(c_entry["mia_auc_std"]) * 100.0)
        if xs:
            ax0.errorbar(
                xs,
                aucs,
                yerr=stds,
                label=pol_lbl,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=6.5,
                capsize=3.5,
            )

    ax0.set_xscale("log", base=2)
    ax0.set_xticks(eps_grid)
    ax0.set_xticklabels([str(e) for e in eps_grid])
    ax0.set_ylim(48.5, max(64.0, no_dp_acc + 4.0))
    ax0.set_xlabel("Mean Per-Tenant Privacy Budget ε̄ (δ = 10⁻⁵, log₂ scale)")
    ax0.set_ylabel("Membership Inference Attack Score (%) — Lower is More Private")
    ax0.set_title(
        "High-Risk Rare-Attack Cohort MIA Audit (Tenant 1: Finance)\nAdaptive DP Suppresses Leakage Faster Than Uniform DP",
        fontsize=10.0,
        fontweight="bold",
    )
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    # Panel B: Standard Federated Regime Local vs. Global Model MIA AUC & Accuracy
    ax1 = axes[1]
    regimes_bar = [
        ("none__eps_inf", "No-DP\n(ε = ∞)"),
        ("uniform__eps_8.0", "Uniform\n(ε̄ = 8.0)"),
        ("adaptive__eps_8.0", "Adaptive\n(ε̄ = 8.0)"),
        ("uniform__eps_2.0", "Uniform\n(ε̄ = 2.0)"),
        ("adaptive__eps_2.0", "Adaptive\n(ε̄ = 2.0)"),
        ("adaptive__eps_0.5", "Adaptive\n(ε̄ = 0.5)"),
    ]
    xb = np.arange(len(regimes_bar))
    wb = 0.26

    loc_aucs, glob_aucs, coh_aucs = [], [], []
    for r_key, _ in regimes_bar:
        e_fl = insdn_grid.get(r_key, {}).get("metrics", {})
        e_ch = cohort_insdn.get(r_key, {})
        loc_aucs.append(float(e_fl.get("mia_local_auc_mean", {}).get("mean", 0.5)) * 100.0)
        glob_aucs.append(float(e_fl.get("mia_global_auc_mean", {}).get("mean", 0.5)) * 100.0)
        coh_aucs.append(float(e_ch.get("mia_auc_mean", 0.5)) * 100.0)

    ax1.axhline(50.0, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Random Guess (50.0%)")
    b1 = ax1.bar(
        xb - wb,
        loc_aucs,
        wb,
        label="Full-Partition Local Model MIA AUC",
        color="#2980b9",
        edgecolor="white",
    )
    b2 = ax1.bar(
        xb,
        glob_aucs,
        wb,
        label="Full-Partition Global Model MIA AUC",
        color="#27ae60",
        edgecolor="white",
    )
    b3 = ax1.bar(
        xb + wb,
        coh_aucs,
        wb,
        label="Rare-Attack Cohort Local MIA AUC (Finance)",
        color="#c0392b",
        edgecolor="white",
    )

    for rects in (b1, b2, b3):
        for r in rects:
            h = r.get_height()
            ax1.annotate(
                f"{h:.1f}",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    ax1.set_xticks(xb)
    ax1.set_xticklabels([lbl for _, lbl in regimes_bar], fontsize=8.8)
    ax1.set_ylim(45.0, 64.0)
    ax1.set_ylabel("Membership Inference Attack AUC (%)")
    ax1.set_title(
        "Empirical MIA Audit: Full-Partition (Local & Global Models)\nvs. High-Risk Rare-Attack Cohort (InSDN, 5 Seeds)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.0, framealpha=0.92)

    fig.suptitle(
        "Step 05 — Empirical Privacy Evaluation via White-Box & Black-Box Membership Inference Attacks (5 Seeds)",
        fontsize=11.6,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def run_step05_full_sweep(
    *,
    out_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS_DEFAULT,
    rounds: int = 12,
    local_epochs: int = 1,
    expected_batch_size: int = 512,
    delta: float = 1e-5,
    max_train_samples: int = 12000,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute the complete Step 05 5-seed Adaptive & Uniform Per-Tenant DP-SGD & MIA benchmark."""
    if device is None:
        device = get_default_device(prefer_gpu=True)
    target_dir = out_dir or (REPO_ROOT / "results" / "05_dp")
    target_dir.mkdir(parents=True, exist_ok=True)

    datasets_cfg = {
        "insdn": {"lr": 0.0025, "max_grad_norm": 0.5},
        "aseados_sdn_iot": {"lr": 0.006, "max_grad_norm": 0.8},
    }
    eps_values = [0.5, 1.0, 2.0, 4.0, 8.0]

    # Build jobs: (ds_name, cfg_key, dp_policy, mean_eps)
    jobs: list[tuple[str, str, str, float | None]] = []
    for ds_name in ("insdn", "aseados_sdn_iot"):
        jobs.append((ds_name, "none__eps_inf", "none", None))
        for eps_val in eps_values:
            for pol in ("uniform", "adaptive"):
                jobs.append((ds_name, f"{pol}__eps_{eps_val:.1f}", pol, eps_val))

    sweep_summary: dict[str, Any] = {
        "step": "05_dp",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "dp_defaults": {
            "granularity": "record_level_dp_sgd",
            "accountant": "rdp_sampled_gaussian_mironov2019_balle2020",
            "delta": float(delta),
            "num_tenants": 3,
            "partition_type": "natural",
            "aggregator": "fedavg",
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "expected_batch_size": int(expected_batch_size),
            "max_train_samples_stratified": int(max_train_samples),
            "seeds": list(seeds),
            "mean_epsilon_sweep": eps_values,
            "sensitivity_formula": "s_i = 0.5 * domain_profile_i + 0.5 * data_score_i",
            "budget_policy_formula": "eps_i = eps_total * (1 / s_i) / sum_j(1 / s_j)",
        },
        "datasets": {"insdn": {}, "aseados_sdn_iot": {}},
        "rare_cohort_mia": {"insdn": {}, "aseados_sdn_iot": {}},
        "statistical_tests": {},
    }

    ckpt_path = target_dir / "summary_checkpoint.json"
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            for ds_k in ("insdn", "aseados_sdn_iot"):
                sweep_summary["datasets"][ds_k] = ckpt.get("datasets", {}).get(ds_k, {})
                sweep_summary["rare_cohort_mia"][ds_k] = ckpt.get("rare_cohort_mia", {}).get(
                    ds_k, {}
                )
        except Exception:
            pass

    bundles_cache: dict[str, dict[str, Any]] = {}
    t_sweep_start = time.perf_counter()
    total_jobs = len(jobs)

    for job_idx, (ds_name, cfg_key, pol, mean_eps) in enumerate(jobs, start=1):
        if (
            cfg_key in sweep_summary["datasets"][ds_name]
            and cfg_key in sweep_summary["rare_cohort_mia"][ds_name]
        ):
            print(
                f"[Step 05] ({job_idx:2d}/{total_jobs}) Resuming from checkpoint: "
                f"{ds_name} | {cfg_key} already done.",
                flush=True,
            )
            continue

        if ds_name not in bundles_cache:
            bundles_cache[ds_name] = load_dataset_artifacts(ds_name)
        bundle = bundles_cache[ds_name]
        dcfg = datasets_cfg[ds_name]

        t_job = time.perf_counter()
        seed_runs: list[dict[str, Any]] = []
        cohort_runs: list[dict[str, Any]] = []

        for s in seeds:
            sr = run_single_seed_dp_fl(
                bundle,
                dp_policy=pol,
                mean_epsilon=mean_eps,
                delta=delta,
                max_grad_norm=float(dcfg["max_grad_norm"]),
                expected_batch_size=expected_batch_size,
                rounds=rounds,
                local_epochs=local_epochs,
                lr=float(dcfg["lr"]),
                max_train_samples=max_train_samples,
                seed=s,
                device=device,
            )
            seed_runs.append(sr)
            scores_map = {t: d["sensitivity_score"] for t, d in sr["per_tenant_dp"].items()}
            c_mia = run_rare_cohort_mia_audit(
                bundle,
                dp_policy=pol,
                mean_epsilon=mean_eps,
                scores_map=scores_map,
                delta=delta,
                seed=s,
                device=device,
            )
            cohort_runs.append(c_mia)

        reg_summary = summarize_multi_seed_fl(seed_runs)
        reg_summary["dp_policy"] = pol
        reg_summary["mean_epsilon"] = float(mean_eps) if mean_eps is not None else None
        reg_summary["epsilon_total"] = 3.0 * float(mean_eps) if mean_eps is not None else None
        reg_summary["delta"] = float(delta) if mean_eps is not None else 0.0
        reg_summary["seed_f1_macros"] = [float(sr["final_metrics"]["f1_macro"]) for sr in seed_runs]
        reg_summary["seed_accuracies"] = [
            float(sr["final_metrics"]["accuracy"]) for sr in seed_runs
        ]
        reg_summary["seed_mia_local_aucs"] = [
            float(sr["final_metrics"]["mia_local_auc_mean"]) for sr in seed_runs
        ]
        reg_summary["seed_mia_global_aucs"] = [
            float(sr["final_metrics"]["mia_global_auc_mean"]) for sr in seed_runs
        ]

        # Summarize per-tenant metrics across the 5 seeds
        tenant_names = sorted(seed_runs[0]["per_tenant_dp"].keys())
        per_tenant_summary: dict[str, Any] = {}
        for t_name in tenant_names:
            t_items = [sr["per_tenant_dp"][t_name] for sr in seed_runs]
            f1s = [float(x["f1_macro"]) for x in t_items]
            accs = [float(x["accuracy"]) for x in t_items]
            l_aucs = [float(x["mia_local"]["mia_auc"]) for x in t_items]
            l_accs = [float(x["mia_local"]["mia_accuracy"]) for x in t_items]
            g_aucs = [float(x["mia_global"]["mia_auc"]) for x in t_items]
            g_accs = [float(x["mia_global"]["mia_accuracy"]) for x in t_items]
            first_t = t_items[0]
            per_tenant_summary[t_name] = {
                "tenant": t_name,
                "num_samples": int(first_t["num_samples"]),
                "domain_profile": float(first_t["domain_profile"]),
                "rare_share": float(first_t["rare_share"]),
                "imbalance": float(first_t["imbalance"]),
                "data_score": float(first_t["data_score"]),
                "sensitivity_score": float(first_t["sensitivity_score"]),
                "allocated_epsilon": (
                    float(first_t["allocated_epsilon"])
                    if math.isfinite(float(first_t["allocated_epsilon"]))
                    else None
                ),
                "epsilon_spent": (
                    float(first_t["epsilon_spent"])
                    if math.isfinite(float(first_t["epsilon_spent"]))
                    else None
                ),
                "delta": float(first_t["delta"]),
                "noise_multiplier": float(first_t["noise_multiplier"]),
                "sample_rate": float(first_t["sample_rate"]),
                "dp_steps_total": int(first_t["dp_steps_total"]),
                "f1_macro_mean": float(np.mean(f1s)),
                "f1_macro_std": float(np.std(f1s, ddof=0)),
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs, ddof=0)),
                "mia_local_auc_mean": float(np.mean(l_aucs)),
                "mia_local_auc_std": float(np.std(l_aucs, ddof=0)),
                "mia_local_acc_mean": float(np.mean(l_accs)),
                "mia_global_auc_mean": float(np.mean(g_aucs)),
                "mia_global_auc_std": float(np.std(g_aucs, ddof=0)),
                "mia_global_acc_mean": float(np.mean(g_accs)),
                "seed_f1_macros": f1s,
            }
        reg_summary["per_tenant_summary"] = per_tenant_summary
        reg_summary["metrics"]["mia_local_auc_mean"] = {
            "mean": float(np.mean(reg_summary["seed_mia_local_aucs"])),
            "std": float(np.std(reg_summary["seed_mia_local_aucs"], ddof=0)),
        }
        reg_summary["metrics"]["mia_global_auc_mean"] = {
            "mean": float(np.mean(reg_summary["seed_mia_global_aucs"])),
            "std": float(np.std(reg_summary["seed_mia_global_aucs"], ddof=0)),
        }

        c_aucs = [float(x["mia_auc"]) for x in cohort_runs]
        c_accs = [float(x["mia_accuracy"]) for x in cohort_runs]
        c_advs = [float(x["mia_advantage"]) for x in cohort_runs]
        c_tprs = [float(x["mia_tpr_at_5pct_fpr"]) for x in cohort_runs]
        c_loss_aucs = [float(x["loss_threshold"]["auc"]) for x in cohort_runs]
        c_marg_aucs = [float(x["logit_margin"]["auc"]) for x in cohort_runs]
        c_grad_aucs = [float(x["grad_norm"]["auc"]) for x in cohort_runs]

        cohort_summary = {
            "dp_policy": pol,
            "mean_epsilon": float(mean_eps) if mean_eps is not None else None,
            "allocated_epsilon_finance": (
                float(cohort_runs[0]["allocated_epsilon_finance"])
                if math.isfinite(float(cohort_runs[0]["allocated_epsilon_finance"]))
                else None
            ),
            "noise_multiplier_finance": float(cohort_runs[0]["noise_multiplier_finance"]),
            "mia_auc_mean": float(np.mean(c_aucs)),
            "mia_auc_std": float(np.std(c_aucs, ddof=0)),
            "mia_accuracy_mean": float(np.mean(c_accs)),
            "mia_accuracy_std": float(np.std(c_accs, ddof=0)),
            "mia_advantage_mean": float(np.mean(c_advs)),
            "mia_tpr_at_5pct_fpr_mean": float(np.mean(c_tprs)),
            "loss_threshold_auc_mean": float(np.mean(c_loss_aucs)),
            "logit_margin_auc_mean": float(np.mean(c_marg_aucs)),
            "grad_norm_auc_mean": float(np.mean(c_grad_aucs)),
            "seed_mia_aucs": c_aucs,
        }

        sweep_summary["datasets"][ds_name][cfg_key] = reg_summary
        sweep_summary["rare_cohort_mia"][ds_name][cfg_key] = cohort_summary
        ckpt_path.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

        f1_m = reg_summary["metrics"]["f1_macro"]["mean"] * 100.0
        f1_s = reg_summary["metrics"]["f1_macro"]["std"] * 100.0
        fin_f1 = per_tenant_summary["tenant_1_finance"]["f1_macro_mean"] * 100.0
        fin_eps = per_tenant_summary["tenant_1_finance"]["allocated_epsilon"]
        c_auc_m = cohort_summary["mia_auc_mean"] * 100.0
        dt = time.perf_counter() - t_job
        print(
            f"[Step 05] ({job_idx:2d}/{total_jobs}) {ds_name:16s} | {cfg_key:18s} "
            f"-> Global F1={f1_m:6.2f}±{f1_s:4.2f}% | Fin F1={fin_f1:6.2f}% (eps₁={str(round(fin_eps,3)) if fin_eps else 'inf'}) "
            f"| Cohort MIA AUC={c_auc_m:5.2f}% ({dt:.1f}s)",
            flush=True,
        )

    # Reconstruct flat tables: per_seed_summary.csv, per_tenant_dp_table.csv, mia_evaluation.csv
    per_seed_rows: list[dict[str, Any]] = []
    per_tenant_rows: list[dict[str, Any]] = []
    mia_rows: list[dict[str, Any]] = []

    for ds_name in ("insdn", "aseados_sdn_iot"):
        ds_grid = sweep_summary["datasets"][ds_name]
        ch_grid = sweep_summary["rare_cohort_mia"][ds_name]
        for cfg_key, reg_sum in ds_grid.items():
            pol = reg_sum["dp_policy"]
            m_eps = reg_sum["mean_epsilon"]
            e_tot = reg_sum["epsilon_total"]
            f1_list = reg_sum.get("seed_f1_macros", [])
            acc_list = reg_sum.get("seed_accuracies", [])
            l_mia_list = reg_sum.get("seed_mia_local_aucs", [])
            g_mia_list = reg_sum.get("seed_mia_global_aucs", [])
            ch_sum = ch_grid.get(cfg_key, {})
            ch_mia_list = ch_sum.get("seed_mia_aucs", [])

            for idx_s, s_val in enumerate(reg_sum.get("seeds", list(seeds))):
                per_seed_rows.append(
                    {
                        "dataset": ds_name,
                        "config_key": cfg_key,
                        "dp_policy": pol,
                        "mean_epsilon": m_eps if m_eps is not None else np.inf,
                        "epsilon_total": e_tot if e_tot is not None else np.inf,
                        "delta": reg_sum["delta"],
                        "seed": s_val,
                        "global_f1_macro": f1_list[idx_s] if idx_s < len(f1_list) else np.nan,
                        "global_accuracy": acc_list[idx_s] if idx_s < len(acc_list) else np.nan,
                        "mia_local_auc_mean": (
                            l_mia_list[idx_s] if idx_s < len(l_mia_list) else np.nan
                        ),
                        "mia_global_auc_mean": (
                            g_mia_list[idx_s] if idx_s < len(g_mia_list) else np.nan
                        ),
                        "rare_cohort_finance_mia_auc": (
                            ch_mia_list[idx_s] if idx_s < len(ch_mia_list) else np.nan
                        ),
                    }
                )

            for t_name, t_sum in reg_sum.get("per_tenant_summary", {}).items():
                per_tenant_rows.append(
                    {
                        "dataset": ds_name,
                        "config_key": cfg_key,
                        "dp_policy": pol,
                        "mean_epsilon": m_eps if m_eps is not None else np.inf,
                        "epsilon_total": e_tot if e_tot is not None else np.inf,
                        "tenant": t_name,
                        "num_samples": t_sum["num_samples"],
                        "domain_profile": t_sum["domain_profile"],
                        "rare_share": t_sum["rare_share"],
                        "imbalance": t_sum["imbalance"],
                        "data_score": t_sum["data_score"],
                        "sensitivity_score": t_sum["sensitivity_score"],
                        "allocated_epsilon": (
                            t_sum["allocated_epsilon"]
                            if t_sum["allocated_epsilon"] is not None
                            else np.inf
                        ),
                        "epsilon_spent": (
                            t_sum["epsilon_spent"] if t_sum["epsilon_spent"] is not None else np.inf
                        ),
                        "delta": t_sum["delta"],
                        "noise_multiplier": t_sum["noise_multiplier"],
                        "sample_rate": t_sum["sample_rate"],
                        "dp_steps_total": t_sum["dp_steps_total"],
                        "tenant_f1_macro_mean": t_sum["f1_macro_mean"],
                        "tenant_f1_macro_std": t_sum["f1_macro_std"],
                        "tenant_accuracy_mean": t_sum["accuracy_mean"],
                        "tenant_accuracy_std": t_sum["accuracy_std"],
                        "mia_local_auc_mean": t_sum["mia_local_auc_mean"],
                        "mia_local_acc_mean": t_sum["mia_local_acc_mean"],
                        "mia_global_auc_mean": t_sum["mia_global_auc_mean"],
                        "mia_global_acc_mean": t_sum["mia_global_acc_mean"],
                    }
                )

            mia_rows.append(
                {
                    "dataset": ds_name,
                    "config_key": cfg_key,
                    "dp_policy": pol,
                    "mean_epsilon": m_eps if m_eps is not None else np.inf,
                    "allocated_epsilon_finance": (
                        ch_sum.get("allocated_epsilon_finance")
                        if ch_sum.get("allocated_epsilon_finance") is not None
                        else np.inf
                    ),
                    "noise_multiplier_finance": ch_sum.get("noise_multiplier_finance", 0.0),
                    "full_partition_local_mia_auc_mean": reg_sum["metrics"]["mia_local_auc_mean"][
                        "mean"
                    ],
                    "full_partition_local_mia_auc_std": reg_sum["metrics"]["mia_local_auc_mean"][
                        "std"
                    ],
                    "full_partition_global_mia_auc_mean": reg_sum["metrics"]["mia_global_auc_mean"][
                        "mean"
                    ],
                    "full_partition_global_mia_auc_std": reg_sum["metrics"]["mia_global_auc_mean"][
                        "std"
                    ],
                    "rare_cohort_mia_auc_mean": ch_sum.get("mia_auc_mean"),
                    "rare_cohort_mia_auc_std": ch_sum.get("mia_auc_std"),
                    "rare_cohort_mia_accuracy_mean": ch_sum.get("mia_accuracy_mean"),
                    "rare_cohort_mia_advantage_mean": ch_sum.get("mia_advantage_mean"),
                    "rare_cohort_tpr_at_5pct_fpr_mean": ch_sum.get("mia_tpr_at_5pct_fpr_mean"),
                    "rare_cohort_loss_threshold_auc_mean": ch_sum.get("loss_threshold_auc_mean"),
                    "rare_cohort_logit_margin_auc_mean": ch_sum.get("logit_margin_auc_mean"),
                    "rare_cohort_grad_norm_auc_mean": ch_sum.get("grad_norm_auc_mean"),
                }
            )

    # Paired Wilcoxon tests comparing Adaptive vs Uniform at equal total budget
    stat_tests: dict[str, Any] = {}
    for ds_name in ("insdn", "aseados_sdn_iot"):
        ds_g = sweep_summary["datasets"][ds_name]
        ch_g = sweep_summary["rare_cohort_mia"][ds_name]
        for eps_val in eps_values:
            u_entry = ds_g.get(f"uniform__eps_{eps_val:.1f}")
            a_entry = ds_g.get(f"adaptive__eps_{eps_val:.1f}")
            u_ch = ch_g.get(f"uniform__eps_{eps_val:.1f}")
            a_ch = ch_g.get(f"adaptive__eps_{eps_val:.1f}")
            if not u_entry or not a_entry:
                continue
            p_f1 = compute_paired_wilcoxon(u_entry["seed_f1_macros"], a_entry["seed_f1_macros"])
            p_mia = (
                compute_paired_wilcoxon(u_ch["seed_mia_aucs"], a_ch["seed_mia_aucs"])
                if u_ch and a_ch
                else 1.0
            )
            u_fin = u_entry["per_tenant_summary"]["tenant_1_finance"]
            a_fin = a_entry["per_tenant_summary"]["tenant_1_finance"]
            stat_tests[f"{ds_name}_eps{eps_val:.1f}_adaptive_vs_uniform"] = {
                "mean_epsilon": eps_val,
                "epsilon_total": 3.0 * eps_val,
                "uniform_global_f1_mean": u_entry["metrics"]["f1_macro"]["mean"],
                "adaptive_global_f1_mean": a_entry["metrics"]["f1_macro"]["mean"],
                "global_f1_diff_pts": (
                    a_entry["metrics"]["f1_macro"]["mean"] - u_entry["metrics"]["f1_macro"]["mean"]
                )
                * 100.0,
                "wilcoxon_p_global_f1": p_f1,
                "finance_epsilon_uniform": u_fin["allocated_epsilon"],
                "finance_epsilon_adaptive": a_fin["allocated_epsilon"],
                "finance_epsilon_reduction_pct": (
                    (u_fin["allocated_epsilon"] - a_fin["allocated_epsilon"])
                    / u_fin["allocated_epsilon"]
                )
                * 100.0,
                "finance_sigma_uniform": u_fin["noise_multiplier"],
                "finance_sigma_adaptive": a_fin["noise_multiplier"],
                "finance_cohort_mia_auc_uniform": u_ch["mia_auc_mean"] if u_ch else None,
                "finance_cohort_mia_auc_adaptive": a_ch["mia_auc_mean"] if a_ch else None,
                "wilcoxon_p_cohort_mia_auc": p_mia,
            }

    sweep_summary["statistical_tests"] = stat_tests
    sweep_summary["total_sweep_wall_time_sec"] = float(time.perf_counter() - t_sweep_start)

    pd.DataFrame(per_seed_rows).to_csv(target_dir / "per_seed_summary.csv", index=False)
    pd.DataFrame(per_tenant_rows).to_csv(target_dir / "per_tenant_dp_table.csv", index=False)
    pd.DataFrame(mia_rows).to_csv(target_dir / "mia_evaluation.csv", index=False)
    (target_dir / "summary.json").write_text(
        json.dumps(sweep_summary, indent=2),
        encoding="utf-8",
    )

    # Generate all 3 Step 05 plots
    plot_privacy_utility_tradeoff(
        sweep_summary,
        target_dir / "privacy_utility_tradeoff.png",
    )
    plot_per_tenant_dp_comparison(
        sweep_summary,
        target_dir / "per_tenant_dp_comparison.png",
    )
    plot_mia_attack_evaluation(
        sweep_summary,
        target_dir / "mia_attack_evaluation.png",
    )
    return sweep_summary


# ==============================================================================
# STEP 06: Verifiable Secure Aggregation Sweep & Tamper/Overhead Benchmarks
# ==============================================================================


def benchmark_step06_tamper_and_honest_runs(
    *,
    param_dim: int,
    tenant_counts: tuple[int, ...] = (3, 5, 10),
    honest_rounds_per_n: int = 100,
    tamper_trials_per_attack: int = 100,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run honest false-alarm audit and 4-type tamper detection benchmark across N."""
    rows: list[dict[str, Any]] = []
    summary_by_n: dict[str, Any] = {}

    attack_types = [
        ("honest_rounds", "Honest Rounds (No Tamper — Target 0% Alarm)"),
        ("modified_aggregate_float", "Tamper 1a: Modified Aggregate (Float +0.05)"),
        ("modified_aggregate_lsb", "Tamper 1b: Modified Aggregate (+1 Quantized LSB)"),
        ("dropped_tenant", "Tamper 2: Dropped Tenant Submission"),
        ("replayed_update", "Tamper 3: Replayed Old-Round Update"),
        ("substituted_update", "Tamper 4: Substituted / Forged Tenant Update"),
    ]

    for num_tenants in tenant_counts:
        client_ids = [f"tenant_{i}" for i in range(num_tenants)]
        materials, registry, group_key = generate_tenant_keys(client_ids, seed=100 + num_tenants)
        gen = torch.Generator(device="cpu").manual_seed(2026 + num_tenants)
        server_nonces: set[tuple[str, bytes]] = set()
        client_nonces: set[tuple[str, bytes]] = set()

        honest_false_alarms = 0
        verify_ms_list: list[float] = []
        round_submissions_cache: list[list[Any]] = []
        round_proofs_cache: list[Any] = []

        for round_idx in range(1, honest_rounds_per_n + 1):
            subs = [
                create_signed_submission(
                    cid,
                    torch.randn(param_dim, generator=gen) * 0.25,
                    num_samples=500 + 200 * idx,
                    round_idx=round_idx,
                    key_material=materials[cid],
                    registry=registry,
                    group_key=group_key,
                    scale_bits=scale_bits,
                    seed=100 + num_tenants,
                )
                for idx, cid in enumerate(client_ids)
            ]
            proof = server_secure_aggregate(
                subs,
                registry,
                round_idx=round_idx,
                seen_nonces=server_nonces,
                scale_bits=scale_bits,
            )
            verdict = verify_aggregate(
                proof,
                registry,
                group_key,
                expected_round_idx=round_idx,
                seen_nonces=client_nonces,
                scale_bits=scale_bits,
            )
            verify_ms_list.append(float(verdict.verify_time_ms))
            if not verdict.ok:
                honest_false_alarms += 1
            round_submissions_cache.append(subs)
            round_proofs_cache.append(proof)

        # Evaluate tamper attacks across rounds 2..honest_rounds_per_n (using fresh nonces view
        # matching round t before round t's acceptance, plus cached previous-round submissions)
        det_counts: dict[str, int] = {k: 0 for k, _ in attack_types if k != "honest_rounds"}
        det_reasons: dict[str, str] = {}

        for trial_idx in range(tamper_trials_per_attack):
            src_idx = (trial_idx % (honest_rounds_per_n - 1)) + 1  # index >= 1 (round >= 2)
            round_idx = src_idx + 1
            subs = round_submissions_cache[src_idx]
            prev_sub = round_submissions_cache[src_idx - 1][trial_idx % num_tenants]
            proof = round_proofs_cache[src_idx]
            # Snapshot nonces seen BEFORE round_idx
            seen_before_round = {
                (s.header.client_id, s.header.nonce)
                for r in range(src_idx)
                for s in round_submissions_cache[r]
            }

            bad_mod_f = tamper_modified_aggregate(
                proof, lsb_only=False, delta_float=0.05, scale_bits=scale_bits
            )
            v1a = verify_aggregate(
                bad_mod_f,
                registry,
                group_key,
                expected_round_idx=round_idx,
                seen_nonces=set(seen_before_round),
                scale_bits=scale_bits,
            )
            if not v1a.ok:
                det_counts["modified_aggregate_float"] += 1
                det_reasons.setdefault("modified_aggregate_float", v1a.reason)

            bad_mod_lsb = tamper_modified_aggregate(proof, lsb_only=True, scale_bits=scale_bits)
            v1b = verify_aggregate(
                bad_mod_lsb,
                registry,
                group_key,
                expected_round_idx=round_idx,
                seen_nonces=set(seen_before_round),
                scale_bits=scale_bits,
            )
            if not v1b.ok:
                det_counts["modified_aggregate_lsb"] += 1
                det_reasons.setdefault("modified_aggregate_lsb", v1b.reason)

            bad_drop = tamper_dropped_tenant(
                subs, round_idx=round_idx, drop_index=trial_idx % num_tenants, scale_bits=scale_bits
            )
            v2 = verify_aggregate(
                bad_drop,
                registry,
                group_key,
                expected_round_idx=round_idx,
                seen_nonces=set(seen_before_round),
                scale_bits=scale_bits,
            )
            if not v2.ok:
                det_counts["dropped_tenant"] += 1
                det_reasons.setdefault("dropped_tenant", v2.reason)

            bad_replay = tamper_replayed_update(
                subs, prev_sub, round_idx=round_idx, scale_bits=scale_bits
            )
            v3 = verify_aggregate(
                bad_replay,
                registry,
                group_key,
                expected_round_idx=round_idx,
                seen_nonces=set(seen_before_round),
                scale_bits=scale_bits,
            )
            if not v3.ok:
                det_counts["replayed_update"] += 1
                det_reasons.setdefault("replayed_update", v3.reason)

            bad_sub = tamper_substituted_update(
                subs,
                round_idx=round_idx,
                replace_index=trial_idx % num_tenants,
                seed=5000 + trial_idx,
                scale_bits=scale_bits,
            )
            v4 = verify_aggregate(
                bad_sub,
                registry,
                group_key,
                expected_round_idx=round_idx,
                seen_nonces=set(seen_before_round),
                scale_bits=scale_bits,
            )
            if not v4.ok:
                det_counts["substituted_update"] += 1
                det_reasons.setdefault("substituted_update", v4.reason)

        n_key = f"n{num_tenants}"
        summary_by_n[n_key] = {
            "num_tenants": int(num_tenants),
            "honest_trials": int(honest_rounds_per_n),
            "honest_false_alarms": int(honest_false_alarms),
            "honest_false_alarm_rate_pct": (
                100.0 * float(honest_false_alarms) / float(honest_rounds_per_n)
            ),
            "mean_verify_ms": float(np.mean(verify_ms_list)),
            "attacks": {},
        }
        rows.append(
            {
                "num_tenants": int(num_tenants),
                "scenario_key": "honest_rounds",
                "scenario_label": attack_types[0][1],
                "trials": int(honest_rounds_per_n),
                "detected_or_flagged": int(honest_false_alarms),
                "rate_pct": 100.0 * float(honest_false_alarms) / float(honest_rounds_per_n),
                "sample_reason": "verified",
            }
        )
        for atk_key, atk_label in attack_types[1:]:
            cnt = int(det_counts[atk_key])
            rate = 100.0 * float(cnt) / float(tamper_trials_per_attack)
            summary_by_n[n_key]["attacks"][atk_key] = {
                "trials": int(tamper_trials_per_attack),
                "detected": cnt,
                "detection_rate_pct": rate,
                "sample_reason": det_reasons.get(atk_key, ""),
            }
            rows.append(
                {
                    "num_tenants": int(num_tenants),
                    "scenario_key": atk_key,
                    "scenario_label": atk_label,
                    "trials": int(tamper_trials_per_attack),
                    "detected_or_flagged": cnt,
                    "rate_pct": rate,
                    "sample_reason": det_reasons.get(atk_key, ""),
                }
            )

    return rows, summary_by_n


def benchmark_step06_overhead(
    *,
    param_dim: int,
    tenant_counts: tuple[int, ...] = (3, 5, 10),
    num_timed_rounds: int = 20,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure exact Step 06 latency and byte overhead breakdown vs Plain FedAvg."""
    rows: list[dict[str, Any]] = []
    summary_by_n: dict[str, Any] = {}

    for num_tenants in tenant_counts:
        client_ids = [f"tenant_{i}" for i in range(num_tenants)]
        sec_agg = build_aggregator("secure_fedavg", scale_bits=scale_bits, seed=77 + num_tenants)
        plain_agg = FedAvgAggregator()
        gen = torch.Generator(device="cpu").manual_seed(333 + num_tenants)
        init_global = torch.zeros(param_dim, dtype=torch.float32)

        q_times: list[float] = []
        m_times: list[float] = []
        c_times: list[float] = []
        s_times: list[float] = []
        srv_times: list[float] = []
        v_times: list[float] = []
        plain_agg_times: list[float] = []
        sec_upload_bytes_list: list[int] = []
        proof_wire_bytes_list: list[int] = []
        plain_upload_bytes_list: list[int] = []

        # Warmup 2 rounds to avoid first-call allocation jitter
        for r_idx in range(1, num_timed_rounds + 3):
            updates = [
                ClientUpdate(
                    client_id=cid,
                    flat_params=torch.randn(param_dim, generator=gen) * 0.25,
                    flat_delta=torch.zeros(param_dim, dtype=torch.float32),
                    num_samples=500 + 250 * i,
                    train_loss=0.2,
                    local_train_time_sec=0.05,
                    upload_bytes=24 + 4 * param_dim,
                    round_idx=r_idx,
                )
                for i, cid in enumerate(client_ids)
            ]
            res_plain = plain_agg.aggregate(init_global, updates)
            res_sec = sec_agg.aggregate(init_global, updates)
            if r_idx <= 2:
                continue

            md = res_sec.metadata
            q_times.append(float(md["quantize_ms_per_tenant"]))
            m_times.append(float(md["mask_ms_per_tenant"]))
            c_times.append(float(md["checksum_ms_per_tenant"]))
            s_times.append(float(md["sign_ms_per_tenant"]))
            srv_times.append(float(md["server_agg_ms"]))
            v_times.append(float(md["verify_ms_per_tenant"]))
            plain_agg_times.append(float(res_plain.aggregation_time_sec) * 1000.0)
            sec_upload_bytes_list.append(int(md["secure_upload_bytes_per_tenant"]))
            proof_wire_bytes_list.append(int(md["proof_wire_bytes"]))
            plain_upload_bytes_list.append(int(md["plain_upload_bytes_per_tenant"]))

        plain_up = int(round(np.mean(plain_upload_bytes_list)))
        plain_down = plain_up
        plain_rt = plain_up + plain_down

        sec_up_48 = int(round(np.mean(sec_upload_bytes_list)))
        proof_bytes = int(round(np.mean(proof_wire_bytes_list)))
        # Tenant downloads global float32 model + verification proof bundle
        sec_down = plain_down + proof_bytes
        sec_rt_48 = sec_up_48 + sec_down

        # Contrast against naive uncompressed 64-bit masked upload
        sec_up_64 = sec_up_48 + (2 * param_dim)
        sec_rt_64 = sec_up_64 + sec_down

        q_ms = float(np.mean(q_times))
        m_ms = float(np.mean(m_times))
        c_ms = float(np.mean(c_times))
        s_ms = float(np.mean(s_times))
        srv_ms = float(np.mean(srv_times))
        v_ms = float(np.mean(v_times))
        p_agg_ms = float(np.mean(plain_agg_times))

        row = {
            "num_tenants": int(num_tenants),
            "param_dim": int(param_dim),
            "scale_bits": int(scale_bits),
            "quantize_ms_per_tenant": q_ms,
            "mask_ms_per_tenant": m_ms,
            "checksum_commit_ms_per_tenant": c_ms,
            "sign_ms_per_tenant": s_ms,
            "client_total_crypto_ms": q_ms + m_ms + c_ms + s_ms,
            "server_agg_merkle_ms": srv_ms,
            "verify_ms_per_tenant": v_ms,
            "total_secure_protocol_ms": q_ms + m_ms + c_ms + s_ms + srv_ms + v_ms,
            "plain_fedavg_agg_ms": p_agg_ms,
            "plain_upload_bytes_per_tenant": plain_up,
            "plain_roundtrip_bytes_per_tenant": plain_rt,
            "secure_48bit_upload_bytes_per_tenant": sec_up_48,
            "secure_proof_bytes_per_tenant": proof_bytes,
            "secure_48bit_roundtrip_bytes_per_tenant": sec_rt_48,
            "secure_64bit_roundtrip_bytes_per_tenant": sec_rt_64,
            "upload_ratio_48bit": float(sec_up_48) / float(plain_up),
            "roundtrip_ratio_48bit": float(sec_rt_48) / float(plain_rt),
            "roundtrip_ratio_64bit": float(sec_rt_64) / float(plain_rt),
        }
        rows.append(row)
        summary_by_n[f"n{num_tenants}"] = row

    return rows, summary_by_n


def plot_step06_tamper_matrix(
    tamper_summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot 100% tamper detection across all 4 attack classes + 0% false alarms on 100 honest rounds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.4, 5.4), dpi=160)

    n_keys = ["n3", "n5", "n10"]
    n_labels = ["N = 3 Tenants", "N = 5 Tenants", "N = 10 Tenants"]
    scenarios = [
        ("modified_aggregate_float", "Tamper 1a:\nModified Agg\n(Float +0.05)"),
        ("modified_aggregate_lsb", "Tamper 1b:\nModified Agg\n(+1 LSB Int)"),
        ("dropped_tenant", "Tamper 2:\nDropped\nTenant"),
        ("replayed_update", "Tamper 3:\nReplayed Old\nUpdate"),
        ("substituted_update", "Tamper 4:\nSubstituted\nUpdate"),
        ("honest_rounds", "Honest Baseline:\nFalse Alarm Rate\n(100 Rounds)"),
    ]

    matrix = np.zeros((len(n_keys), len(scenarios)), dtype=np.float64)
    for r_i, n_k in enumerate(n_keys):
        ns = tamper_summary[n_k]
        for c_i, (s_k, _) in enumerate(scenarios):
            if s_k == "honest_rounds":
                matrix[r_i, c_i] = float(ns["honest_false_alarm_rate_pct"])
            else:
                matrix[r_i, c_i] = float(ns["attacks"][s_k]["detection_rate_pct"])

    # Left panel: detection rate vs false alarm rate grouped bars for N=3, 5, 10
    x = np.arange(len(scenarios))
    width = 0.25
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for idx_n, (n_lbl, col) in enumerate(zip(n_labels, colors, strict=True)):
        offset = (idx_n - 1) * width
        vals = matrix[idx_n, :]
        bars = ax0.bar(x + offset, vals, width=width, label=n_lbl, color=col, alpha=0.88)
        for b_i, bar in enumerate(bars):
            h = bar.get_height()
            txt = f"{h:.0f}%" if b_i < 5 else f"{h:.1f}%"
            y_offset = 1.6 + (idx_n % 2) * 3.8
            ax0.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + y_offset,
                txt,
                ha="center",
                va="bottom",
                fontsize=6.4,
                fontweight="bold",
            )

    ax0.axvline(4.5, color="#333333", linestyle="--", linewidth=1.2)
    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for _, lbl in scenarios], fontsize=8.0)
    ax0.set_ylim(0.0, 120.0)
    ax0.set_ylabel("Detection Rate (%)  |  False-Alarm Rate (%)")
    ax0.set_title(
        "Tamper Detection Rate (Target 100%) vs. Honest False-Alarm Rate (Target 0%)\n"
        "Evaluated over 100 Honest Rounds + 100 Tamper Trials per Attack",
        fontsize=10.0,
        fontweight="bold",
    )
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="center right", fontsize=8.3, framealpha=0.92)

    # Right panel: Tenant verify() latency scaling across N in {3, 5, 10}
    ver_ms = [float(tamper_summary[k]["mean_verify_ms"]) for k in n_keys]
    bars2 = ax1.bar(
        n_labels,
        ver_ms,
        color=["#2c7fb8", "#41b6c4", "#253494"],
        width=0.45,
        edgecolor="#222222",
    )
    for bar, v in zip(bars2, ver_ms, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.35,
            f"{v:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
        )
    ax1.set_ylim(0.0, max(ver_ms) * 1.28)
    ax1.set_ylabel("Per-Round Tenant verify() Time (ms)")
    ax1.set_title(
        "Tenant Verification Latency vs. Federation Scale (GRU, d = 43,554 Params)\n"
        "Constant O(d + K log K) Verification Cost per Round",
        fontsize=10.0,
        fontweight="bold",
    )
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Step 06 — Cryptographic Integrity Verification: 100% Tamper Detection & 0% False Alarms",
        fontsize=11.6,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def plot_step06_overhead_breakdown(
    overhead_summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot compute latency breakdown and wire communication ratio vs Plain FedAvg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.4, 5.4), dpi=160)

    n_keys = ["n3", "n5", "n10"]
    n_labels = ["N = 3 Tenants", "N = 5 Tenants", "N = 10 Tenants"]
    x = np.arange(len(n_keys))

    stages = [
        ("quantize_ms_per_tenant", "1. Fixed-Point Quantize", "#4e79a7"),
        ("mask_ms_per_tenant", "2. X25519 + SHAKE-256 Mask", "#f28e2b"),
        ("checksum_commit_ms_per_tenant", "3. Dual-Prime Linear Checksum", "#59a14f"),
        ("sign_ms_per_tenant", "4. Ed25519 Signature + Hash", "#b07aa1"),
        ("server_agg_merkle_ms", "5. Server Unmask + Merkle Tree", "#76b7b2"),
        ("verify_ms_per_tenant", "6. Tenant verify() Proof Check", "#e15759"),
    ]

    bottom = np.zeros(len(n_keys), dtype=np.float64)
    for key, lbl, col in stages:
        vals = np.array([float(overhead_summary[nk][key]) for nk in n_keys], dtype=np.float64)
        ax0.bar(x, vals, bottom=bottom, width=0.48, label=lbl, color=col, edgecolor="white")
        bottom += vals

    for idx_n, total_ms in enumerate(bottom):
        ax0.text(
            x[idx_n],
            total_ms + max(bottom) * 0.025,
            f"Total:\n{total_ms:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax0.set_xticks(x)
    ax0.set_xticklabels(n_labels, fontsize=9.2)
    ax0.set_ylim(0.0, max(bottom) * 1.24)
    ax0.set_ylabel("Wall-Clock Crypto & Aggregation Time (ms / round)")
    ax0.set_title(
        "Per-Round Cryptographic Compute Breakdown\n(GRU d = 43,554 Params, Measured Across 20 Rounds)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax0.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="upper left", fontsize=7.6, framealpha=0.92)

    # Right panel: Wire communication ratio vs Plain FedAvg (48-bit packed vs 64-bit naive)
    rt_48 = [float(overhead_summary[nk]["roundtrip_ratio_48bit"]) for nk in n_keys]
    up_48 = [float(overhead_summary[nk]["upload_ratio_48bit"]) for nk in n_keys]
    rt_64 = [float(overhead_summary[nk]["roundtrip_ratio_64bit"]) for nk in n_keys]

    w = 0.24
    ax1.axhspan(1.2, 1.5, color="#59a14f", alpha=0.14, label="Poster Target Zone (1.20×–1.50×)")
    ax1.axhline(
        1.0, color="#333333", linestyle="--", linewidth=1.2, label="Plain FedAvg Baseline (1.00×)"
    )

    b1 = ax1.bar(x - w, rt_48, width=w, label="48-Bit Packed Round-Trip Ratio", color="#1f77b4")
    b2 = ax1.bar(x, up_48, width=w, label="48-Bit Packed Upload Ratio", color="#2ca02c")
    b3 = ax1.bar(x + w, rt_64, width=w, label="Unpacked 64-Bit Round-Trip Ratio", color="#ff7f0e")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.025,
                f"{h:.2f}×",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(n_labels, fontsize=9.2)
    ax1.set_ylim(0.0, 2.08)
    ax1.set_ylabel("Wire Communication Overhead Ratio vs. Plain FedAvg")
    ax1.set_title(
        "Per-Tenant Wire Byte Ratio (Download + Upload + Merkle Proof)\n"
        "48-Bit Packing Meets the 1.2×–1.5× Poster Target",
        fontsize=10.0,
        fontweight="bold",
    )
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", ncol=2, fontsize=7.3, framealpha=0.92)

    fig.suptitle(
        "Step 06 — Computational & Communication Overhead vs. Plain FedAvg (GRU, 43,554 Params)",
        fontsize=11.6,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def plot_step06_accuracy_equivalence(
    eq_summary: dict[str, Any],
    quant_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Plot detection Macro-F1 convergence equivalence & quantization bit-width ablation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), dpi=160)

    for ax_i, (ds_key, ds_title) in enumerate(
        [
            ("insdn", "InSDN (Natural Non-IID, N=3)"),
            ("aseados_sdn_iot", "ASEADOS-SDN-IoT (Natural Non-IID, N=3)"),
        ]
    ):
        ax = axes[ax_i]
        ds_data = eq_summary["datasets"][ds_key]
        styles = [
            ("plain_fedavg", "Plain FedAvg (Float32 Baseline)", "#111111", "-", 2.3),
            ("secure_q20", "SecureAgg (20-bit Default, S=2^20)", "#1f77b4", "--", 2.1),
            ("secure_q16", "SecureAgg (16-bit, S=2^16)", "#ff7f0e", "-.", 1.8),
            ("secure_q24", "SecureAgg (24-bit, S=2^24)", "#2ca02c", ":", 1.9),
        ]
        for cfg_key, lbl, col, ls, lw in styles:
            if cfg_key not in ds_data:
                continue
            curves = ds_data[cfg_key]["round_curves"]
            rds = [int(c["round"]) for c in curves if "f1_macro_mean" in c]
            vals = [float(c["f1_macro_mean"]) * 100.0 for c in curves if "f1_macro_mean" in c]
            final_f1 = float(ds_data[cfg_key]["metrics"]["f1_macro"]["mean"]) * 100.0
            ax.plot(
                rds,
                vals,
                color=col,
                linestyle=ls,
                linewidth=lw,
                label=f"{lbl} ({final_f1:.2f}%)",
            )
        ax.set_xlabel("Federated Round (t = 1..30)")
        ax.set_ylabel("Global Test Macro-F1 (%)")
        ax.set_title(ds_title, fontsize=10.0, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", fontsize=7.6, framealpha=0.92)

    # Panel 3: Quantization Precision vs Round-Trip Error & SNR
    ax2 = axes[2]
    bits = [int(r["scale_bits"]) for r in quant_rows]
    errs = [float(r["max_abs_error"]) for r in quant_rows]
    snrs = [float(r["snr_db"]) for r in quant_rows]
    x = np.arange(len(bits))
    b = ax2.bar(
        x,
        snrs,
        width=0.45,
        color=["#ff7f0e", "#1f77b4", "#2ca02c"],
        edgecolor="#222222",
    )
    for bar, snr, err in zip(b, snrs, errs, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            snr + 2.5,
            f"{snr:.1f} dB\n(max err {err:.1e})",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b_val}-bit (2^{b_val})" for b_val in bits], fontsize=8.8)
    ax2.set_ylim(0.0, max(snrs) * 1.24)
    ax2.set_ylabel("Quantization Signal-to-Noise Ratio (dB)")
    ax2.set_title(
        "Fixed-Point Precision Trade-Off\n(20-bit Chosen Default: < 5e-7 Error, > 107 dB SNR)",
        fontsize=10.0,
        fontweight="bold",
    )
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Step 06 — Accuracy Equivalence: Verifiable Secure Aggregation Matches Plain FedAvg (5 Seeds)",
        fontsize=11.6,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def run_step06_full_sweep(
    *,
    out_dir: Path | None = None,
    seeds: tuple[int, ...] = SEEDS_DEFAULT,
    rounds: int = 30,
    local_epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_train_samples: int = 24000,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Execute the complete Step 06 Verifiable Secure Aggregation evaluation suite."""
    t_sweep_start = time.perf_counter()
    target_dir = out_dir or (REPO_ROOT / "results" / "06_verify")
    target_dir.mkdir(parents=True, exist_ok=True)
    eff_device = device if device is not None else get_default_device(prefer_gpu=True)

    # Probe exact GRU parameter dimension on InSDN binary
    probe_bundle = load_dataset_artifacts("insdn")
    probe_model = build_model(
        "gru",
        input_dim=int(probe_bundle["splits"]["X_train_global_scaled"].shape[1]),
        num_classes=2,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        seq_mode="single_step",
        num_chunks=1,
    )
    param_dim = probe_model.count_parameters()
    flat_probe = probe_model.get_flat_params()

    # 1. Fixed-point quantization error across {16, 20, 24} bits
    quant_rows: list[dict[str, Any]] = []
    for sb in (16, 20, 24):
        qe = measure_quantization_error(flat_probe, scale_bits=sb)
        quant_rows.append(qe)
    pd.DataFrame(quant_rows).to_csv(target_dir / "quantization_error.csv", index=False)

    # 2. Tamper detection + 100 honest rounds across N in {3, 5, 10}
    tamper_rows, tamper_summary = benchmark_step06_tamper_and_honest_runs(
        param_dim=param_dim,
        tenant_counts=(3, 5, 10),
        honest_rounds_per_n=100,
        tamper_trials_per_attack=100,
        scale_bits=DEFAULT_SCALE_BITS,
    )
    pd.DataFrame(tamper_rows).to_csv(target_dir / "tamper_test_table.csv", index=False)

    # 3. Latency and wire byte overhead breakdown across N in {3, 5, 10}
    overhead_rows, overhead_summary = benchmark_step06_overhead(
        param_dim=param_dim,
        tenant_counts=(3, 5, 10),
        num_timed_rounds=20,
        scale_bits=DEFAULT_SCALE_BITS,
    )
    pd.DataFrame(overhead_rows).to_csv(target_dir / "overhead_breakdown.csv", index=False)

    # 4. 5-seed accuracy equivalence: Plain FedAvg vs SecureAgg (16, 20, 24 bits)
    eq_configs = [
        ("plain_fedavg", "fedavg", None, None),
        ("secure_q16", "secure_fedavg", {"scale_bits": 16}, 16),
        ("secure_q20", "secure_fedavg", {"scale_bits": 20}, 20),
        ("secure_q24", "secure_fedavg", {"scale_bits": 24}, 24),
    ]
    model_cfg = {
        "type": "gru",
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "seq_mode": "single_step",
        "num_chunks": 1,
    }

    sweep_summary: dict[str, Any] = {
        "step": "06",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": str(eff_device),
        "param_dim": int(param_dim),
        "seeds": list(seeds),
        "quantization_error": quant_rows,
        "tamper_tests": tamper_summary,
        "overhead": overhead_summary,
        "datasets": {},
        "statistical_tests": {},
    }

    per_seed_rows: list[dict[str, Any]] = []
    eq_table_rows: list[dict[str, Any]] = []
    total_jobs = 2 * len(eq_configs)
    job_idx = 0

    for ds_name in ("insdn", "aseados_sdn_iot"):
        bundle = load_dataset_artifacts(ds_name)
        sweep_summary["datasets"][ds_name] = {}

        for cfg_key, agg_name, agg_kw, scale_bits in eq_configs:
            job_idx += 1
            t_job = time.perf_counter()
            seed_runs: list[dict[str, Any]] = []
            for seed in seeds:
                kw = dict(agg_kw, seed=seed) if agg_kw is not None else None
                res = run_single_seed_fl(
                    bundle,
                    task="binary",
                    partition_type="natural",
                    num_tenants=3,
                    dirichlet_alpha=0.5,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    seed=seed,
                    model_cfg=model_cfg,
                    aggregator_name=agg_name,
                    aggregator_kwargs=kw,
                    scaler_mode="global",
                    max_train_samples=max_train_samples,
                    eval_every=1,
                    eval_per_tenant_every=5,
                    device=eff_device,
                )
                seed_runs.append(res)
                fm = res["final_metrics"]
                per_seed_rows.append(
                    {
                        "dataset": ds_name,
                        "config_key": cfg_key,
                        "aggregator": agg_name,
                        "scale_bits": scale_bits if scale_bits is not None else 32,
                        "seed": int(seed),
                        "accuracy": float(fm["accuracy"]),
                        "f1_macro": float(fm["f1_macro"]),
                        "attack_detection_rate": float(fm["attack_detection_rate"]),
                        "false_positive_rate": float(fm["false_positive_rate"]),
                        "roc_auc": float(fm["roc_auc"]),
                    }
                )

            cfg_sum = summarize_multi_seed_fl(seed_runs)
            cfg_sum["scale_bits"] = scale_bits
            cfg_sum["seed_f1_macros"] = [float(sr["final_metrics"]["f1_macro"]) for sr in seed_runs]
            cfg_sum["seed_accuracies"] = [
                float(sr["final_metrics"]["accuracy"]) for sr in seed_runs
            ]
            sweep_summary["datasets"][ds_name][cfg_key] = cfg_sum

            m = cfg_sum["metrics"]
            eq_table_rows.append(
                {
                    "dataset": ds_name,
                    "config_key": cfg_key,
                    "aggregator": agg_name,
                    "scale_bits": scale_bits if scale_bits is not None else 32,
                    "f1_macro_mean": float(m["f1_macro"]["mean"]),
                    "f1_macro_std": float(m["f1_macro"]["std"]),
                    "accuracy_mean": float(m["accuracy"]["mean"]),
                    "accuracy_std": float(m["accuracy"]["std"]),
                    "adr_mean": float(m["attack_detection_rate"]["mean"]),
                    "fpr_mean": float(m["false_positive_rate"]["mean"]),
                    "roc_auc_mean": float(m["roc_auc"]["mean"]),
                }
            )
            dt = time.perf_counter() - t_job
            print(
                f"[Step 06] ({job_idx:2d}/{total_jobs}) {ds_name:16s} | {cfg_key:14s} "
                f"-> F1={m['f1_macro']['mean']*100.0:6.2f}±{m['f1_macro']['std']*100.0:4.2f}% "
                f"| Acc={m['accuracy']['mean']*100.0:6.2f}% ({dt:.1f}s)",
                flush=True,
            )

        plain_s = sweep_summary["datasets"][ds_name]["plain_fedavg"]
        for q_key in ("secure_q16", "secure_q20", "secure_q24"):
            sec_s = sweep_summary["datasets"][ds_name][q_key]
            p_f1 = compute_paired_wilcoxon(plain_s["seed_f1_macros"], sec_s["seed_f1_macros"])
            diff_pts = (
                sec_s["metrics"]["f1_macro"]["mean"] - plain_s["metrics"]["f1_macro"]["mean"]
            ) * 100.0
            sweep_summary["statistical_tests"][f"{ds_name}_{q_key}_vs_plain"] = {
                "plain_f1_mean": plain_s["metrics"]["f1_macro"]["mean"],
                "secure_f1_mean": sec_s["metrics"]["f1_macro"]["mean"],
                "f1_diff_pts": diff_pts,
                "wilcoxon_p_f1": p_f1,
            }

    sweep_summary["total_sweep_wall_time_sec"] = float(time.perf_counter() - t_sweep_start)

    pd.DataFrame(per_seed_rows).to_csv(target_dir / "per_seed_summary.csv", index=False)
    pd.DataFrame(eq_table_rows).to_csv(target_dir / "accuracy_equivalence.csv", index=False)
    (target_dir / "summary.json").write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")

    # Generate all 3 Step 06 plots
    plot_step06_tamper_matrix(
        tamper_summary,
        target_dir / "tamper_detection_matrix.png",
    )
    plot_step06_overhead_breakdown(
        overhead_summary,
        target_dir / "overhead_breakdown.png",
    )
    plot_step06_accuracy_equivalence(
        sweep_summary,
        quant_rows,
        target_dir / "accuracy_equivalence.png",
    )
    return sweep_summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `sim/run_experiment.py`."""
    parser = argparse.ArgumentParser(
        description="TrustFed-SDN Federated Learning Simulator (Steps 03, 04, and 05)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--sweep-step03",
        action="store_true",
        help="Run the complete Step 03 5-seed FedAvg benchmark sweep and generate plots.",
    )
    parser.add_argument(
        "--sweep-step04",
        action="store_true",
        help="Run the complete Step 04 5-seed Poisoning & Robust Aggregation sweep and generate plots.",
    )
    parser.add_argument(
        "--sweep-step05",
        action="store_true",
        help="Run the complete Step 05 5-seed Adaptive Per-Tenant DP & MIA sweep and generate plots.",
    )
    parser.add_argument(
        "--sweep-step06",
        action="store_true",
        help="Run the complete Step 06 Verifiable Secure Aggregation benchmark sweep and plots.",
    )
    args = parser.parse_args(argv)

    if args.sweep_step03:
        run_step03_full_sweep()
        return 0

    if args.sweep_step04:
        run_step04_full_sweep()
        return 0

    if args.sweep_step05:
        run_step05_full_sweep()
        return 0

    if args.sweep_step06:
        run_step06_full_sweep()
        return 0

    if not args.config:
        parser.error(
            "Provide `--config <path.yaml>`, `--sweep-step03`, `--sweep-step04`, "
            "`--sweep-step05`, or `--sweep-step06`."
        )

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    run_experiment_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

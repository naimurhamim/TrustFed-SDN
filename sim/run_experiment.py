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

from trustfed.data.clean import load_label_map  # noqa: E402
from trustfed.data.partition import (  # noqa: E402
    partition_dirichlet,
    partition_iid,
    partition_natural_non_iid,
)
from trustfed.fl import (  # noqa: E402
    Aggregator,
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

SEEDS_DEFAULT: tuple[int, ...] = (0, 1, 2, 3, 4)


def build_aggregator(name: str) -> Aggregator:
    """Instantiate an `Aggregator` by name (`fedavg` in Step 03)."""
    key = name.strip().lower()
    if key == "fedavg":
        return FedAvgAggregator()
    raise ValueError(f"Unsupported aggregator '{name}' in Step 03. Expected 'fedavg'.")


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
        if num_tenants != 3:
            raise ValueError(
                f"Natural domain-specialty partition requires num_tenants=3, got {num_tenants}."
            )
        train_map: dict[str, np.ndarray] = {}
        prefix = "natural_n3__"
        for k, v in part_npz.items():
            if k.startswith(prefix):
                train_map[k[len(prefix) :]] = v.astype(np.int64)
        if not train_map:
            train_map = partition_natural_non_iid(
                y_mul_train,
                class_order=bundle["class_order"],
                specialty_map=bundle["specialty_map"],
                primary_share=0.70,
                seed=seed,
            )
        test_map = partition_natural_non_iid(
            y_mul_test,
            class_order=bundle["class_order"],
            specialty_map=bundle["specialty_map"],
            primary_share=0.70,
            seed=seed + 99,
        )
        return train_map, test_map

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
            )
        )

    aggregator = build_aggregator(aggregator_name)
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
    fit_res["dataset"] = bundle["dataset"]
    fit_res["task"] = task
    fit_res["partition_type"] = partition_type
    fit_res["num_tenants"] = int(num_tenants)
    fit_res["dirichlet_alpha"] = float(dirichlet_alpha) if partition_type == "dirichlet" else None
    fit_res["scaler_mode"] = scaler_mode
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `sim/run_experiment.py`."""
    parser = argparse.ArgumentParser(
        description="TrustFed-SDN Step 03 Federated Learning Simulator & FedAvg Runner"
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
    args = parser.parse_args(argv)

    if args.sweep_step03:
        run_step03_full_sweep()
        return 0

    if not args.config:
        parser.error("Provide either `--config <path.yaml>` or `--sweep-step03`.")

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

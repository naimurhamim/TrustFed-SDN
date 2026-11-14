"""Step 07 Sim-to-Live Gap Evaluation, Manifest Generation, and Publication Figures.

Evaluates Step 02/03 offline-trained GRU models directly against live multi-tenant
SDN testbed flows captured in ``data/live/`` (zero-shot transfer + local feature
calibration) and generates ``results/07_testbed/`` plots and ``data/live/MANIFEST.md``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.run_experiment import load_dataset_artifacts, resolve_tenant_partitions  # noqa: E402
from testbed.capture.flow_extractor import (  # noqa: E402
    extract_features_for_dataset,
    verify_feature_alignment,
)
from trustfed.fl import FLClient, FLServer  # noqa: E402
from trustfed.metrics import summarize_seed_metrics  # noqa: E402
from trustfed.models import (  # noqa: E402
    build_model,
    evaluate_pytorch_model,
    get_default_device,
    set_all_seeds,
)
from trustfed.models.trainer import train_pytorch_model  # noqa: E402

EVAL_SEEDS: tuple[int, ...] = (0, 1, 2)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def generate_live_manifest(live_dir: Path) -> Path:
    """Write ``data/live/MANIFEST.md`` with exact sizes and SHA-256 digests."""
    files = sorted(
        [
            p
            for p in live_dir.rglob("*")
            if p.is_file() and p.name not in ("MANIFEST.md", ".gitkeep")
        ]
    )
    lines = [
        "# Live SDN Testbed Dataset Manifest (`data/live/`)",
        "",
        f"- **Generated (UTC):** `{datetime.now(UTC).isoformat()}`",
        "- **Testbed Stack:** `Mininet 2.3.0 + Open vSwitch 2.17.12 + OS-Ken 3.0.1 (OpenFlow 1.3)`",
        "- **Isolation Level:** `L1` (3 isolated `/24` subnets under `10.10.0.0/16`)",
        "",
        "| Relative Path | Size (Bytes) | SHA-256 |",
        "|---|---|---|",
    ]
    for fp in files:
        rel = fp.relative_to(live_dir).as_posix()
        lines.append(f"| `{rel}` | `{fp.stat().st_size}` | `{_sha256_file(fp)}` |")

    manifest_path = live_dir / "MANIFEST.md"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _train_fedavg_offline_gru(
    bundle: dict[str, Any],
    seed: int,
    device: torch.device,
    rounds: int = 20,
) -> tuple[torch.nn.Module, dict[str, float]]:
    """Train Step 03 FedAvg GRU on offline training data (same global scaler space)."""
    set_all_seeds(seed)
    sp = bundle["splits"]
    X_train = sp["X_train_global_scaled"]
    y_train = sp["y_bin_train"]
    X_val = sp["X_val_global_scaled"]
    y_val = sp["y_bin_val"]
    X_test = sp["X_test_global_scaled"]
    y_test = sp["y_bin_test"]

    train_parts, _ = resolve_tenant_partitions(bundle, "natural", num_tenants=3, seed=seed)

    def factory() -> torch.nn.Module:
        return build_model(
            model_type="gru",
            input_dim=int(X_train.shape[1]),
            num_classes=2,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            seq_mode="feature_chunks",
        )

    clients = []
    for c_idx, (t_id, idx) in enumerate(sorted(train_parts.items())):
        clients.append(
            FLClient(
                client_id=t_id,
                client_index=c_idx,
                model=factory(),
                X_train=X_train[idx],
                y_train=y_train[idx],
                X_val=X_val,
                y_val=y_val,
                device=device,
            )
        )

    server = FLServer(
        global_model=factory(),
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        device=device,
    )
    server.fit(rounds=rounds, local_epochs=3, batch_size=128, lr=1e-3, seed=seed)
    offline_metrics = evaluate_pytorch_model(server.global_model, X_test, y_test, device=device)
    return server.global_model, offline_metrics


def _evaluate_sim_to_live_for_dataset(
    dataset_name: str,
    feature_yaml: Path,
    live_full_df: pd.DataFrame,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Measure offline sim test, zero-shot live, and domain-calibrated live scores."""
    bundle = load_dataset_artifacts(dataset_name)
    sp = bundle["splits"]
    mean = sp["global_scaler_mean"].astype(np.float64)
    scale = sp["global_scaler_scale"].astype(np.float64)
    scale_safe = np.where(scale == 0.0, 1.0, scale)

    X_live_raw = extract_features_for_dataset(live_full_df, feature_yaml).to_numpy(dtype=np.float64)
    y_live = live_full_df["y_binary"].to_numpy(dtype=np.int64)
    tenants_live = live_full_df["tenant_id"].to_numpy()

    # Zero-shot transformation using offline training scaler (exact sim->live boundary)
    X_live_offline_scaled = ((X_live_raw - mean) / scale_safe).astype(np.float32)

    # Z-score calibrated view (models live link distribution calibration)
    live_scaler = StandardScaler()
    X_live_zcal = live_scaler.fit_transform(X_live_raw).astype(np.float32)

    rows: list[dict[str, Any]] = []
    for seed in EVAL_SEEDS:
        model, off_m = _train_fedavg_offline_gru(bundle, seed=seed, device=device)
        zs_m = evaluate_pytorch_model(model, X_live_offline_scaled, y_live, device=device)
        cal_m = evaluate_pytorch_model(model, X_live_zcal, y_live, device=device)

        # Stratified 50/50 live fine-tuned evaluation (simulates FL adaptation on testbed)
        tr_idx, te_idx = train_test_split(
            np.arange(len(y_live)),
            test_size=0.5,
            stratify=y_live,
            random_state=seed + 100,
        )
        ft_scaler = StandardScaler()
        X_ft_tr = ft_scaler.fit_transform(X_live_raw[tr_idx]).astype(np.float32)
        X_ft_te = ft_scaler.transform(X_live_raw[te_idx]).astype(np.float32)

        ft_model = build_model(
            model_type="gru",
            input_dim=int(X_live_raw.shape[1]),
            num_classes=2,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            seq_mode="feature_chunks",
        )
        ft_model.load_state_dict(model.state_dict())
        train_pytorch_model(
            ft_model,
            X_ft_tr,
            y_live[tr_idx],
            X_ft_te,
            y_live[te_idx],
            epochs=8,
            batch_size=64,
            lr=1e-3,
            device=device,
            seed=seed,
        )
        ft_m = evaluate_pytorch_model(ft_model, X_ft_te, y_live[te_idx], device=device)

        # Per-tenant zero-shot breakdown
        for t_id in sorted(np.unique(tenants_live)):
            mask = tenants_live == t_id
            tm = evaluate_pytorch_model(
                model,
                X_live_offline_scaled[mask],
                y_live[mask],
                device=device,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "regime": f"zero_shot_{t_id}",
                    "accuracy": tm["accuracy"],
                    "f1_macro": tm["f1_macro"],
                    "precision_macro": tm["precision_macro"],
                    "recall_macro": tm["recall_macro"],
                    "attack_detection_rate": tm["attack_detection_rate"],
                    "false_positive_rate": tm["false_positive_rate"],
                    "roc_auc": tm.get("roc_auc", float("nan")),
                }
            )

        for reg_name, metrics in (
            ("offline_sim_fedavg", off_m),
            ("live_zero_shot", zs_m),
            ("live_feature_zcal", cal_m),
            ("live_finetuned_adapted", ft_m),
        ):
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "regime": reg_name,
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "attack_detection_rate": metrics["attack_detection_rate"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "roc_auc": metrics.get("roc_auc", float("nan")),
                }
            )
    return rows


def _plot_tenant_traffic_distribution(
    live_full_df: pd.DataFrame,
    capture_summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot per-tenant attack/benign class counts + L1 isolation verification bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=150)

    tenants = ["tenant_1_finance", "tenant_2_healthcare", "tenant_3_education"]
    t_labels = ["T1: Finance", "T2: Healthcare", "T3: Education"]
    classes = sorted(live_full_df["canonical_label"].unique())
    cmap = plt.get_cmap("tab10")
    bottom = np.zeros(len(tenants), dtype=np.float64)

    for idx, cls_name in enumerate(classes):
        counts = [
            int(
                (
                    (live_full_df["tenant_id"] == t) & (live_full_df["canonical_label"] == cls_name)
                ).sum()
            )
            for t in tenants
        ]
        vals = np.asarray(counts, dtype=np.float64)
        ax1.bar(
            t_labels, vals, bottom=bottom, label=cls_name, color=cmap(idx % 10), edgecolor="white"
        )
        bottom += vals

    for i, total in enumerate(bottom):
        ax1.text(
            i, total + 12, f"{int(total)}", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax1.set_ylim(0, max(bottom) * 1.16)
    ax1.set_ylabel("Captured Bidirectional Live Flows")
    ax1.set_title("Live SDN Flow Distribution by Tenant & Class", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    iso = capture_summary["isolation_stats"]
    cats = ["Intra-Tenant\nReachability (%)", "Cross-Tenant\nIsolation Block (%)"]
    vals_iso = [100.0 - float(iso["intra_loss_pct"]), float(iso["cross_block_pct"])]
    colors = ["#2ca02c", "#1f77b4"]
    bars = ax2.bar(cats, vals_iso, color=colors, width=0.5)
    for bar, val in zip(bars, vals_iso, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Measured Verification Rate (%)")
    ax2.set_title("L1 Mininet + OVS + OS-Ken Policy Isolation", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_sim_to_live_gap(df_eval: pd.DataFrame, out_path: Path) -> None:
    """Plot Offline Sim vs. Zero-Shot Live vs. Calibrated vs. Adapted Macro-F1."""
    regimes = [
        ("offline_sim_fedavg", "Offline Sim\n(FedAvg GRU)"),
        ("live_zero_shot", "Live Testbed\n(Zero-Shot)"),
        ("live_feature_zcal", "Live Testbed\n(Z-Calibrated)"),
        ("live_finetuned_adapted", "Live Testbed\n(Adapted GRU)"),
    ]
    datasets = [
        ("insdn", "InSDN (19 features)"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT (12 features)"),
    ]

    x = np.arange(len(regimes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=150)

    for d_idx, (ds_key, ds_label) in enumerate(datasets):
        means = []
        stds = []
        for reg_key, _ in regimes:
            sub = df_eval[(df_eval["dataset"] == ds_key) & (df_eval["regime"] == reg_key)]
            means.append(100.0 * float(sub["f1_macro"].mean()))
            stds.append(100.0 * float(sub["f1_macro"].std(ddof=1)) if len(sub) > 1 else 0.0)
        offset = (d_idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=ds_label,
        )
        for bar, mean_v, std_v in zip(bars, means, stds, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mean_v + std_v + 2.5,
                f"{mean_v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in regimes], fontsize=9.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Binary Macro-F1 (%)")
    ax.set_title(
        "Sim-to-Live Gap: Offline Datasets vs. Live Multi-Tenant SDN Testbed", fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    live_dir = REPO_ROOT / "data" / "live"
    res_dir = REPO_ROOT / "results" / "07_testbed"
    live_full_path = live_dir / "live_flows_full.csv"
    cap_sum_path = res_dir / "capture_summary.json"
    if not live_full_path.exists() or not cap_sum_path.exists():
        raise FileNotFoundError(
            f"Missing live capture artifacts in {live_dir} or {res_dir}. "
            "Run `sudo bash scripts/run_step07_testbed.sh` inside Ubuntu 22.04 VM first."
        )

    live_full_df = pd.read_csv(live_full_path)
    capture_summary = json.loads(cap_sum_path.read_text(encoding="utf-8"))

    # Verify strict Step 01 feature schema match
    insdn_yaml = REPO_ROOT / "configs" / "data" / "features_insdn.yaml"
    ase_yaml = REPO_ROOT / "configs" / "data" / "features_aseados_sdn_iot.yaml"
    align_insdn = verify_feature_alignment(live_full_df, insdn_yaml)
    align_ase = verify_feature_alignment(live_full_df, ase_yaml)
    assert align_insdn["exact_names_and_order"] and align_ase["exact_names_and_order"]

    device = get_default_device()
    rows: list[dict[str, Any]] = []
    rows.extend(_evaluate_sim_to_live_for_dataset("insdn", insdn_yaml, live_full_df, device))
    rows.extend(
        _evaluate_sim_to_live_for_dataset("aseados_sdn_iot", ase_yaml, live_full_df, device)
    )

    df_eval = pd.DataFrame(rows)
    df_eval.to_csv(res_dir / "sim_to_live_per_seed.csv", index=False)

    summary_rows = []
    for (ds_name, regime), grp in df_eval.groupby(["dataset", "regime"], sort=False):
        stats = summarize_seed_metrics(grp.to_dict("records"))
        entry: dict[str, Any] = {"dataset": ds_name, "regime": regime}
        for k, v in stats.items():
            if isinstance(v, dict):
                entry[f"{k}_mean"] = round(float(v["mean"]), 4)
                entry[f"{k}_std"] = round(float(v["std"]), 4)
        summary_rows.append(entry)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(res_dir / "sim_to_live_gap.csv", index=False)

    manifest_path = generate_live_manifest(live_dir)
    _plot_tenant_traffic_distribution(
        live_full_df, capture_summary, res_dir / "tenant_traffic_distribution.png"
    )
    _plot_sim_to_live_gap(df_eval, res_dir / "sim_to_live_gap.png")

    master_summary = {
        "analyzed_at_utc": datetime.now(UTC).isoformat(),
        "capture_summary": capture_summary,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT).as_posix()),
        "sim_to_live_summary": summary_rows,
    }
    (res_dir / "summary.json").write_text(json.dumps(master_summary, indent=2), encoding="utf-8")
    print(f"Step 07 analysis complete -> {res_dir}", flush=True)


if __name__ == "__main__":
    main()

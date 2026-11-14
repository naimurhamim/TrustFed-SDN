"""Step 08 Live Integration Runner & Demo Orchestrator (`scripts/run_step08_live.py`).

Executes the full TrustFed-SDN pipeline over live Mininet testbed flows
(`data/live/live_flows_insdn.csv`) and localhost HTTP RPC (`LiveAggregationHTTPServer`):
1. Honest Full Pipeline (`12` rounds x `5` seeds; or `10` rounds x `1` seed with `--demo`)
   with Adaptive Per-Tenant DP-SGD (`Step 05`) + Verifiable Secure Aggregation (`Step 06`)
2. Live Malicious Tenant Poisoning Filtering Demo (Unguarded FedAvg vs. TrustFed-SDN)
3. Live Server Tampered Aggregate Rejection Demo (all 4 tamper modes from `Step 06`)
4. Real-Time Off-Event-Loop Detection & OVS Mitigation Timeline (`AsyncFlowInferenceWorker`)
5. Publication PNG/PDF figures & JSON/CSV artifacts under `results/08_live/`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testbed.fl import (  # noqa: E402
    LiveAggregationHTTPServer,
    LiveAggregationServer,
    LiveBundle,
    LiveTenantFLClient,
    LiveTenantHTTPClient,
    build_live_gru,
    load_live_tenant_splits,
)
from testbed.ryu_apps.model_deploy import (  # noqa: E402
    AsyncFlowInferenceWorker,
    FlowClassificationJob,
    MitigationPolicyManager,
    calibrate_attack_threshold,
)
from testbed.topology.multi_tenant_topo import TOPOLOGY_SPEC  # noqa: E402
from trustfed.metrics import summarize_seed_metrics  # noqa: E402
from trustfed.models import set_all_seeds  # noqa: E402
from trustfed.verify import generate_tenant_keys  # noqa: E402

FULL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEMO_SEEDS: tuple[int, ...] = (0,)
TAMPER_MODES: tuple[str, ...] = (
    "modified_aggregate",
    "dropped_tenant",
    "replayed_update",
    "substituted_update",
)
SUBNET_HOSTS = {
    t.tenant_id: {
        "switch": t.switch_name,
        "benign_ip": next(h.ip for h in t.hosts if h.role == "benign"),
        "victim_ip": next(h.ip for h in t.hosts if h.role == "victim"),
        "attacker_ip": next(h.ip for h in t.hosts if h.role == "attacker"),
    }
    for t in TOPOLOGY_SPEC
}


def _create_clients_and_server(
    bundle: LiveBundle,
    *,
    rounds: int,
    local_epochs: int,
    use_dp: bool,
    robust_defense: bool,
    seed: int,
) -> tuple[dict[str, LiveTenantFLClient], LiveAggregationServer]:
    set_all_seeds(seed)
    tenant_ids = sorted(bundle.tenants.keys())
    materials, registry, group_key = generate_tenant_keys(tenant_ids, seed=seed)
    device = torch.device("cpu")

    clients: dict[str, LiveTenantFLClient] = {}
    for idx, tid in enumerate(tenant_ids):
        tds = bundle.tenants[tid]
        clients[tid] = LiveTenantFLClient(
            tenant_id=tid,
            client_index=idx,
            X_train=tds.X_train,
            y_train=tds.y_train,
            X_val=tds.X_val,
            y_val=tds.y_val,
            X_test=tds.X_test,
            y_test=tds.y_test,
            key_material=materials[tid],
            registry=registry,
            group_key=group_key,
            rounds=rounds,
            local_epochs=local_epochs,
            dp_epsilon=bundle.adaptive_epsilons[tid] if use_dp else None,
            dp_delta=1e-5,
            dp_max_grad_norm=0.5,
            dp_batch_size=64,
            device=device,
            use_class_weights=True,
        )

    server = LiveAggregationServer(
        build_live_gru(len(bundle.feature_names)),
        registry=registry,
        robust_defense=robust_defense,
        device=device,
    )
    return clients, server


def run_live_fl_scenario(
    bundle: LiveBundle,
    *,
    scenario_name: str,
    rounds: int,
    local_epochs: int,
    use_dp: bool,
    robust_defense: bool,
    use_verifiable_secagg: bool,
    malicious_from_round: int | None,
    seed: int,
    worker: AsyncFlowInferenceWorker | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one complete multi-round live FL scenario over real localhost HTTP RPC."""
    clients, server = _create_clients_and_server(
        bundle,
        rounds=rounds,
        local_epochs=local_epochs,
        use_dp=use_dp,
        robust_defense=robust_defense,
        seed=seed,
    )
    tenant_ids = sorted(clients.keys())
    malicious_tid = "tenant_3_education"
    round_rows: list[dict[str, Any]] = []
    malicious_rounds_total = 0
    malicious_rounds_rejected = 0
    honest_false_rejections = 0
    honest_checks = 0

    with LiveAggregationHTTPServer(server, host="127.0.0.1", port=0) as http_srv:
        rpc = LiveTenantHTTPClient(http_srv.base_url)

        for r_idx in range(1, rounds + 1):
            if malicious_from_round is not None and r_idx == malicious_from_round:
                clients[malicious_tid].enable_attack("scaling", scale_factor=10.0, sign_flip=True)

            g_flat, _, dl_model_ms, dl_model_bytes = rpc.fetch_global_model()
            subs_map: dict[str, Any] = {}
            up_times: list[float] = []
            up_bytes_total = 0

            for tid in tenant_ids:
                prep = clients[tid].train_and_prepare_submission(
                    g_flat,
                    round_idx=r_idx,
                    participants=tenant_ids,
                    lr=3.0e-3,
                    batch_size=64,
                    weight_decay=1.0e-3,
                    seed=seed,
                )
                subs_map[tid] = prep
                up_ms, up_b = rpc.submit_update(prep.submission, prep.update)
                up_times.append(up_ms)
                up_bytes_total += up_b

            cohort = list(tenant_ids)
            if robust_defense:
                screen = rpc.screen_round(r_idx)
                cohort = list(screen["accepted_clients"])
                rejected = list(screen["rejected_clients"])
                if rejected and use_verifiable_secagg and len(cohort) >= 2:
                    for tid in cohort:
                        resigned = clients[tid].resign_for_cohort(
                            subs_map[tid].update.flat_params,
                            num_samples=subs_map[tid].update.num_samples,
                            round_idx=r_idx,
                            participants=cohort,
                            seed=seed,
                        )
                        rpc.submit_resigned_cohort(resigned)
            else:
                rejected = []

            if malicious_from_round is not None and r_idx >= malicious_from_round:
                malicious_rounds_total += 1
                if malicious_tid in rejected:
                    malicious_rounds_rejected += 1
            else:
                honest_checks += len(tenant_ids)
                honest_false_rejections += len(rejected)

            agg = rpc.trigger_aggregate(r_idx, use_verifiable_secagg=use_verifiable_secagg)
            proof = agg.get("proof_obj")

            verify_times: list[float] = []
            all_verified = True
            if proof is not None:
                for tid in cohort:
                    verdict = clients[tid].verify_and_deploy(
                        proof, expected_round_idx=r_idx, expected_clients=cohort
                    )
                    verify_times.append(float(verdict.verify_time_ms))
                    all_verified = all_verified and verdict.ok
            else:
                for tid in cohort:
                    clients[tid].deployed_flat_params = agg["global_tensor"].clone()
                    clients[tid].model.set_flat_params(agg["global_tensor"])

            deploy_ms = 0.0
            if worker is not None:
                deploy_ms = worker.update_model_weights(
                    server.get_global_flat_params(), round_idx=r_idx
                )

            g_eval = server.evaluate_global(bundle.X_global_test, bundle.y_global_test)
            mean_train_ms = float(np.mean([subs_map[t].local_train_ms for t in tenant_ids]))
            mean_dp_ms = float(np.mean([subs_map[t].dp_ms for t in tenant_ids]))
            train_stage_ms = mean_dp_ms if use_dp else mean_train_ms
            mean_crypto_ms = float(np.mean([subs_map[t].mask_commit_sign_ms for t in tenant_ids]))
            mean_up_ms = float(np.mean(up_times))
            dl_proof_ms = float(agg["download_ms"])
            verify_ms = float(np.mean(verify_times)) if verify_times else 0.0
            agg_ms = float(agg["robust_screen_ms"]) + float(agg["server_agg_proof_ms"])
            total_round_ms = (
                train_stage_ms
                + mean_crypto_ms
                + dl_model_ms
                + mean_up_ms
                + agg_ms
                + dl_proof_ms
                + verify_ms
                + deploy_ms
            )

            row = {
                "scenario": scenario_name,
                "seed": int(seed),
                "round": int(r_idx),
                "accuracy": float(g_eval["accuracy"]),
                "precision_macro": float(g_eval["precision_macro"]),
                "recall_macro": float(g_eval["recall_macro"]),
                "f1_macro": float(g_eval["f1_macro"]),
                "f1_attack": float(g_eval["f1_attack"]),
                "attack_detection_rate": float(g_eval["attack_detection_rate"]),
                "false_positive_rate": float(g_eval["false_positive_rate"]),
                "roc_auc": float(g_eval["roc_auc"]),
                "accepted_count": len(cohort),
                "rejected_clients": ";".join(rejected),
                "verified_ok": bool(all_verified),
                "eps_t1": float(clients["tenant_1_finance"].epsilon_spent),
                "eps_t2": float(clients["tenant_2_healthcare"].epsilon_spent),
                "eps_t3": float(clients["tenant_3_education"].epsilon_spent),
                "upload_bytes_total": int(up_bytes_total),
                "download_bytes_total": int(agg["download_bytes"]) * len(cohort)
                + int(dl_model_bytes) * len(tenant_ids),
                "proof_wire_bytes": int(proof.proof_wire_bytes) if proof is not None else 0,
                "local_train_ms": float(mean_train_ms),
                "dp_ms": float(mean_dp_ms),
                "mask_commit_sign_ms": float(mean_crypto_ms),
                "upload_ms": float(mean_up_ms),
                "server_agg_proof_ms": float(agg_ms),
                "download_ms": float(dl_model_ms + dl_proof_ms),
                "verify_ms": float(verify_ms),
                "controller_deploy_ms": float(deploy_ms),
                "total_round_ms": float(total_round_ms),
            }
            round_rows.append(row)

    final_row = round_rows[-1]
    summary = {
        "scenario": scenario_name,
        "seed": int(seed),
        "rounds": int(rounds),
        "final_metrics": {
            k: final_row[k]
            for k in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_attack",
                "attack_detection_rate",
                "false_positive_rate",
                "roc_auc",
            )
        },
        "final_epsilons": {
            "tenant_1_finance": round(final_row["eps_t1"], 4),
            "tenant_2_healthcare": round(final_row["eps_t2"], 4),
            "tenant_3_education": round(final_row["eps_t3"], 4),
        },
        "malicious_rounds_total": int(malicious_rounds_total),
        "malicious_rounds_rejected": int(malicious_rounds_rejected),
        "malicious_rejection_rate": (
            float(malicious_rounds_rejected / malicious_rounds_total)
            if malicious_rounds_total
            else 0.0
        ),
        "honest_false_rejection_rate": float(honest_false_rejections / max(1, honest_checks)),
        "final_global_weights": server.get_global_flat_params().cpu(),
    }
    return round_rows, summary


def run_tamper_rejection_demo(bundle: LiveBundle, *, seed: int = 0) -> dict[str, Any]:
    """Demonstrate live cryptographic rejection of all 4 server tamper modes over HTTP."""
    clients, server = _create_clients_and_server(
        bundle,
        rounds=6,
        local_epochs=1,
        use_dp=False,
        robust_defense=True,
        seed=seed,
    )
    tenant_ids = sorted(clients.keys())
    trials: list[dict[str, Any]] = []

    with LiveAggregationHTTPServer(server, host="127.0.0.1", port=0) as http_srv:
        rpc = LiveTenantHTTPClient(http_srv.base_url)

        # Round 1 honest baseline (initializes previous_submissions for replay attack)
        g1, _, _, _ = rpc.fetch_global_model()
        for tid in tenant_ids:
            prep = clients[tid].train_and_prepare_submission(
                g1, round_idx=1, participants=tenant_ids, seed=seed
            )
            rpc.submit_update(prep.submission, prep.update)
        r1 = rpc.trigger_aggregate(1, use_verifiable_secagg=True)
        honest_ok = all(
            clients[tid]
            .verify_and_deploy(r1["proof_obj"], expected_round_idx=1, expected_clients=tenant_ids)
            .ok
            for tid in tenant_ids
        )
        trials.append(
            {
                "round": 1,
                "tamper_mode": "none_honest",
                "all_tenants_accepted": bool(honest_ok),
                "false_alarm": not honest_ok,
                "reason": "verified",
            }
        )

        # Rounds 2..5 test each of the 4 server tamper attacks over live HTTP RPC
        for offset, mode in enumerate(TAMPER_MODES, start=2):
            ref_weights = clients["tenant_1_finance"].deployed_flat_params.clone()
            g_cur, _, _, _ = rpc.fetch_global_model()
            for tid in tenant_ids:
                prep = clients[tid].train_and_prepare_submission(
                    g_cur, round_idx=offset, participants=tenant_ids, seed=seed
                )
                rpc.submit_update(prep.submission, prep.update)
            r_tampered = rpc.trigger_aggregate(offset, use_verifiable_secagg=True, tamper_mode=mode)
            verdicts = [
                clients[tid].verify_and_deploy(
                    r_tampered["proof_obj"],
                    expected_round_idx=offset,
                    expected_clients=tenant_ids,
                )
                for tid in tenant_ids
            ]
            all_rejected = all(not v.ok for v in verdicts)
            weights_preserved = all(
                torch.allclose(clients[tid].deployed_flat_params, ref_weights) for tid in tenant_ids
            )
            trials.append(
                {
                    "round": offset,
                    "tamper_mode": mode,
                    "all_tenants_rejected": bool(all_rejected),
                    "weights_preserved_on_rollback": bool(weights_preserved),
                    "verify_ms_mean": round(
                        float(np.mean([v.verify_time_ms for v in verdicts])), 3
                    ),
                    "reasons": [v.reason for v in verdicts],
                }
            )

    tamper_trials = [t for t in trials if t["tamper_mode"] != "none_honest"]
    detected_count = sum(1 for t in tamper_trials if t["all_tenants_rejected"])
    return {
        "seed": int(seed),
        "honest_false_alarms": 0 if honest_ok else 1,
        "tamper_modes_tested": len(tamper_trials),
        "tamper_modes_rejected": detected_count,
        "detection_rate_pct": round(100.0 * detected_count / max(1, len(tamper_trials)), 2),
        "trials": trials,
    }


def run_live_mitigation_evaluation(
    bundle: LiveBundle,
    deployed_weights: torch.Tensor,
    *,
    blocklist_dir: Path,
) -> dict[str, Any]:
    """Stream held-out live tenant flows through `AsyncFlowInferenceWorker` & OVS policy."""
    blocklist_file = blocklist_dir / "demo_blocklist.json"
    policy = MitigationPolicyManager(
        blocklist_path=blocklist_file,
        apply_ovs_cli=False,
        priority=300,
        idle_timeout=60,
        hard_timeout=120,
    )
    model = build_live_gru(len(bundle.feature_names))
    model.set_flat_params(deployed_weights.detach().cpu().clone())

    X_val = np.concatenate([t.X_val for t in bundle.tenants.values()], axis=0)
    y_val = np.concatenate([t.y_val for t in bundle.tenants.values()], axis=0)
    cal_threshold, val_cal_stats = calibrate_attack_threshold(model, X_val, y_val, max_fpr=0.12)

    worker = AsyncFlowInferenceWorker(model, policy, attack_threshold=cal_threshold)
    deploy_ms = worker.update_model_weights(deployed_weights, round_idx=1)

    worker.start()
    t0 = time.perf_counter()
    try:
        for tid, tds in sorted(bundle.tenants.items()):
            topo = SUBNET_HOSTS[tid]
            for i in range(len(tds.X_test)):
                label = int(tds.y_test[i])
                src_ip = topo["attacker_ip"] if label == 1 else topo["benign_ip"]
                worker.submit_flow(
                    FlowClassificationJob(
                        flow_id=f"{tid}_flow_{i}",
                        tenant_id=tid,
                        src_ip=src_ip,
                        dst_ip=topo["victim_ip"],
                        feature_vector=tds.X_test[i],
                        bridge=topo["switch"],
                        ground_truth_label=label,
                    )
                )
        worker.wait_until_empty(timeout=15.0)
    finally:
        total_stream_ms = (time.perf_counter() - t0) * 1000.0
        worker.stop()

    stats = worker.summarize_mitigation()
    y_true = np.array([e.ground_truth_label for e in worker.events], dtype=np.int64)
    y_pred = np.array([e.predicted_label for e in worker.events], dtype=np.int64)
    from sklearn.metrics import accuracy_score, f1_score

    stats["calibrated_attack_threshold"] = round(float(cal_threshold), 4)
    stats["val_calibration"] = {k: round(float(v), 4) for k, v in val_cal_stats.items()}
    stats["calibrated_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    stats["calibrated_f1_macro"] = round(
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
    )
    stats["controller_deploy_ms"] = round(float(deploy_ms), 3)
    stats["total_stream_elapsed_ms"] = round(float(total_stream_ms), 2)
    stats["sample_events"] = [
        {
            "flow_id": e.flow_id,
            "tenant_id": e.tenant_id,
            "ipv4_src": e.ipv4_src,
            "ground_truth": e.ground_truth_label,
            "predicted": e.predicted_label,
            "prob_attack": round(e.prob_attack, 4),
            "inference_ms": round(e.inference_latency_ms, 3),
            "mitigation_ms": round(e.mitigation_latency_ms, 3),
            "action": e.action,
        }
        for e in worker.events[:15]
    ]
    return stats


def _plot_round_latency(df_honest: pd.DataFrame, out_stem: Path) -> None:
    stages = [
        ("dp_ms", "Local Training + DP-SGD"),
        ("mask_commit_sign_ms", "Quantize + Mask + Commit + Sign"),
        ("upload_ms", "HTTP Upload"),
        ("server_agg_proof_ms", "Server Screen + Unmask + Merkle"),
        ("download_ms", "HTTP Download (Model + Proof)"),
        ("verify_ms", "Tenant Crypto Verify"),
        ("controller_deploy_ms", "Controller Weight Deploy"),
    ]
    means = [float(df_honest[col].mean()) for col, _ in stages]
    stds = [float(df_honest[col].std(ddof=1)) if len(df_honest) > 1 else 0.0 for col, _ in stages]
    labels = [lbl for _, lbl in stages]

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=150)
    y_pos = np.arange(len(stages))
    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color="#1f77b4", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Per-Round Wall-Clock Latency (ms)")
    ax.set_title("Step 08 Live TrustFed-SDN Per-Round Stage Latency Breakdown", fontweight="bold")
    max_extent = max(m + s for m, s in zip(means, stds, strict=True))
    ax.set_xlim(0, max_extent * 1.20 + 2.0)
    for bar, m_val, s_val in zip(bars, means, stds, strict=True):
        ax.text(
            m_val + s_val + max_extent * 0.015 + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{m_val:.2f} ms",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def _plot_live_scenarios(df_all: pd.DataFrame, out_stem: Path) -> None:
    scenarios = [
        ("honest_full", "TrustFed-SDN (Honest)", "#2ca02c", "-"),
        ("malicious_trustfed", "TrustFed-SDN (33% Poisoning)", "#1f77b4", "--"),
        ("malicious_unguarded_fedavg", "Unguarded FedAvg (33% Poisoning)", "#d62728", ":"),
    ]
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=150)
    for key, label, color, ls in scenarios:
        sub = df_all[df_all["scenario"] == key]
        if sub.empty:
            continue
        grp = sub.groupby("round")["f1_macro"]
        rounds = grp.mean().index.to_numpy()
        mean_v = 100.0 * grp.mean().to_numpy()
        std_v = 100.0 * grp.std(ddof=1).fillna(0.0).to_numpy()
        ax.plot(rounds, mean_v, label=label, color=color, linestyle=ls, linewidth=2.0, marker="o")
        ax.fill_between(rounds, mean_v - std_v, mean_v + std_v, color=color, alpha=0.14)

    ax.axvline(x=4, color="#555555", linestyle="-.", linewidth=1.2, label="Attack Start (Round 4)")
    ax.set_xlabel("Live Federated Round")
    ax.set_ylabel("Global Live Macro-F1 (%)")
    ax.set_title("Live SDN Testbed Convergence: Honest vs. Poisoning Defense", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"))
    plt.close(fig)


def _plot_mitigation_overview(
    mit_stats: dict[str, Any],
    vm_stats: dict[str, Any] | None,
    out_stem: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=150)

    labels_left = ["Worker Attack\nDetection Rate", "False Block Rate\n(Benign Flows)"]
    vals_left = [
        100.0 * float(mit_stats.get("attack_block_rate", 0.0)),
        100.0 * float(mit_stats.get("false_block_rate", 0.0)),
    ]
    if vm_stats and "live_mitigation" in vm_stats:
        labels_left = [
            "OVS Attacker\nBlock Rate (VM)",
            "OVS Benign False\nBlock Rate (VM)",
        ]
        vals_left = [
            float(vm_stats["live_mitigation"]["attack_block_rate_pct"]),
            float(vm_stats["live_mitigation"]["false_block_rate_pct"]),
        ]

    bars1 = ax1.bar(labels_left, vals_left, color=["#2ca02c", "#d62728"], width=0.48)
    for b, v in zip(bars1, vals_left, strict=True):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.0,
            f"{v:.1f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Live OVS Mitigation Accuracy vs. Collateral Impact", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    if vm_stats and "control_plane" in vm_stats:
        cp = vm_stats["control_plane"]
        labels_right = [
            "Flow Infer\n(Worker)",
            "OVS Rule\nInstall (VM)",
            "Control Ping\nIdle (VM)",
            "Control Ping\nUnder FL (VM)",
        ]
        vals_right = [
            float(mit_stats.get("mean_inference_ms", 0.0)),
            float(vm_stats["live_mitigation"].get("mean_ovs_drop_install_ms", 0.0)),
            float(cp["idle_ping"].get("mean_rtt_ms", 0.0)),
            float(cp["under_fl_load_ping"].get("mean_rtt_ms", 0.0)),
        ]
    else:
        labels_right = ["Mean Flow Infer", "P95 Flow Infer", "Mean Mitigation"]
        vals_right = [
            float(mit_stats.get("mean_inference_ms", 0.0)),
            float(mit_stats.get("p95_inference_ms", 0.0)),
            float(mit_stats.get("mean_mitigation_ms", 0.0)),
        ]
    bars2 = ax2.bar(labels_right, vals_right, color="#1f77b4", width=0.52)
    for b, v in zip(bars2, vals_right, strict=True):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + max(vals_right) * 0.03 + 0.05,
            f"{v:.2f} ms",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_ylim(0, max(vals_right) * 1.25 + 0.5)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Off-Loop Detection & Control-Plane Latency", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustFed-SDN Step 08 Live Integration Runner")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick 1-seed x 10-round clean end-to-end demo (used by `make demo`)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of live FL rounds (default: 10 for --demo, 12 for full run)",
    )
    args = parser.parse_args()

    out_dir = REPO_ROOT / "results" / "08_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = DEMO_SEEDS if args.demo else FULL_SEEDS
    rounds = int(args.rounds) if args.rounds is not None else (10 if args.demo else 12)
    local_epochs = 1

    print(
        f"[Step 08] Mode={'DEMO' if args.demo else 'FULL'} | seeds={seeds} | rounds={rounds}",
        flush=True,
    )

    all_round_rows: list[dict[str, Any]] = []
    honest_summaries: list[dict[str, Any]] = []
    mal_def_summaries: list[dict[str, Any]] = []
    mal_raw_summaries: list[dict[str, Any]] = []
    last_deployed_weights: torch.Tensor | None = None
    last_bundle: LiveBundle | None = None

    for seed in seeds:
        bundle = load_live_tenant_splits(seed=seed, epsilon_total=12.0)
        last_bundle = bundle

        # Track controller deployment timing inside honest run using a live worker handle
        dummy_policy = MitigationPolicyManager(blocklist_path=None, apply_ovs_cli=False)
        deploy_worker = AsyncFlowInferenceWorker(
            build_live_gru(len(bundle.feature_names)), dummy_policy
        )

        # Scenario 1: Honest Full TrustFed-SDN (Adaptive DP + Verifiable SecAgg + Robust Screen)
        h_rows, h_sum = run_live_fl_scenario(
            bundle,
            scenario_name="honest_full",
            rounds=rounds,
            local_epochs=local_epochs,
            use_dp=True,
            robust_defense=True,
            use_verifiable_secagg=True,
            malicious_from_round=None,
            seed=seed,
            worker=deploy_worker,
        )
        all_round_rows.extend(h_rows)
        honest_summaries.append(h_sum)
        last_deployed_weights = h_sum["final_global_weights"]

        # Scenario 2a: Poisoned Tenant (Round 4+) WITH TrustFed-SDN Defense
        md_rows, md_sum = run_live_fl_scenario(
            bundle,
            scenario_name="malicious_trustfed",
            rounds=rounds,
            local_epochs=local_epochs,
            use_dp=True,
            robust_defense=True,
            use_verifiable_secagg=True,
            malicious_from_round=4,
            seed=seed,
        )
        all_round_rows.extend(md_rows)
        mal_def_summaries.append(md_sum)

        # Scenario 2b: Poisoned Tenant (Round 4+) WITH Unguarded FedAvg Baseline
        mr_rows, mr_sum = run_live_fl_scenario(
            bundle,
            scenario_name="malicious_unguarded_fedavg",
            rounds=rounds,
            local_epochs=local_epochs,
            use_dp=False,
            robust_defense=False,
            use_verifiable_secagg=False,
            malicious_from_round=4,
            seed=seed,
        )
        all_round_rows.extend(mr_rows)
        mal_raw_summaries.append(mr_sum)

        rej_cnt = md_sum["malicious_rounds_rejected"]
        rej_tot = md_sum["malicious_rounds_total"]
        print(
            f"  [Seed {seed}] Honest F1={h_sum['final_metrics']['f1_macro']:.4f} | "
            f"Defended F1={md_sum['final_metrics']['f1_macro']:.4f} "
            f"(Rejected={rej_cnt}/{rej_tot}) | "
            f"Unguarded F1={mr_sum['final_metrics']['f1_macro']:.4f}",
            flush=True,
        )

    assert last_bundle is not None and last_deployed_weights is not None
    tamper_demo = run_tamper_rejection_demo(last_bundle, seed=seeds[0])
    mit_stats = run_live_mitigation_evaluation(
        last_bundle, last_deployed_weights, blocklist_dir=out_dir
    )

    vm_path = out_dir / "vm_live_mitigation.json"
    vm_stats = json.loads(vm_path.read_text(encoding="utf-8")) if vm_path.exists() else None

    df_all = pd.DataFrame(all_round_rows)
    mal_demo_payload = {
        "seeds": list(seeds),
        "rounds": rounds,
        "attack": (
            "UpdateScalingAttack(scale_factor=10.0, sign_flip=True) "
            "from Round 4 on tenant_3_education"
        ),
        "trustfed_defended": summarize_seed_metrics(
            [s["final_metrics"] for s in mal_def_summaries]
        ),
        "unguarded_fedavg": summarize_seed_metrics([s["final_metrics"] for s in mal_raw_summaries]),
        "malicious_rejection_rate_mean": float(
            np.mean([s["malicious_rejection_rate"] for s in mal_def_summaries])
        ),
    }
    if not args.demo:
        df_all.to_csv(out_dir / "per_round_metrics.csv", index=False)
        (out_dir / "malicious_tenant_demo.json").write_text(
            json.dumps(mal_demo_payload, indent=2), encoding="utf-8"
        )
        (out_dir / "tamper_rejection_demo.json").write_text(
            json.dumps(tamper_demo, indent=2), encoding="utf-8"
        )
        (out_dir / "mitigation_timeline.json").write_text(
            json.dumps(mit_stats, indent=2), encoding="utf-8"
        )

    df_honest = df_all[df_all["scenario"] == "honest_full"]
    stage_latency_summary = {
        col: {
            "mean_ms": round(float(df_honest[col].mean()), 3),
            "std_ms": round(float(df_honest[col].std(ddof=1)) if len(df_honest) > 1 else 0.0, 3),
        }
        for col in (
            "dp_ms",
            "mask_commit_sign_ms",
            "upload_ms",
            "server_agg_proof_ms",
            "download_ms",
            "verify_ms",
            "controller_deploy_ms",
            "total_round_ms",
        )
    }

    summary_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "mode": "demo" if args.demo else "full",
        "seeds": list(seeds),
        "rounds_per_seed": rounds,
        "adaptive_dp_budgets": {
            tid: round(eps, 4) for tid, eps in last_bundle.adaptive_epsilons.items()
        },
        "honest_full_metrics": summarize_seed_metrics(
            [s["final_metrics"] for s in honest_summaries]
        ),
        "stage_latency_ms": stage_latency_summary,
        "wire_bytes_mean_per_round": {
            "upload_bytes_total": int(df_honest["upload_bytes_total"].mean()),
            "download_bytes_total": int(df_honest["download_bytes_total"].mean()),
            "proof_wire_bytes": int(df_honest["proof_wire_bytes"].mean()),
        },
        "malicious_tenant_filtering": mal_demo_payload,
        "tamper_rejection": tamper_demo,
        "live_worker_mitigation": {k: v for k, v in mit_stats.items() if k != "sample_events"},
        "vm_testbed_mitigation": vm_stats,
    }
    summary_name = "demo_summary.json" if args.demo else "summary.json"
    (out_dir / summary_name).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if not args.demo:
        _plot_round_latency(df_honest, out_dir / "round_latency_breakdown")
        _plot_live_scenarios(df_all, out_dir / "live_scenarios_f1")
        _plot_mitigation_overview(mit_stats, vm_stats, out_dir / "mitigation_timeline")

    print(
        f"[Step 08] Complete! Tamper Rejection={tamper_demo['detection_rate_pct']:.1f}% | "
        f"Malicious Rejection={100.0 * mal_demo_payload['malicious_rejection_rate_mean']:.1f}% | "
        f"Summary saved to results/08_live/{summary_name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

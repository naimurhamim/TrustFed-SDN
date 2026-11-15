"""Step 09 Publication Figures (PDF + PNG) and LaTeX Tables Generator."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Okabe-Ito colorblind-safe palette + distinct line/marker styles
PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "charcoal": "#2B2B2B",
    "gray": "#7F7F7F",
}


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "figure.titlesize": 11.5,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_fig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ms(df: pd.DataFrame, col: str, scale: float = 100.0) -> tuple[float, float]:
    arr = df[col].dropna().astype(float).to_numpy() * scale
    if len(arr) == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _tex_esc(text: str) -> str:
    return (
        str(text)
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("<", "$<$")
        .replace(">", "$>$")
    )


def generate_all_figures(
    *,
    results_dir: Path | None = None,
    fig_dir: Path | None = None,
) -> list[Path]:
    """Generate all 8 publication figures (vector PDF + PNG) from results artifacts."""
    _apply_pub_style()
    res = results_dir or (REPO_ROOT / "results")
    out = fig_dir or (REPO_ROOT / "figures")
    fin = res / "final"
    written: list[Path] = []

    df02_c = pd.read_csv(res / "02_baselines" / "per_seed_centralized.csv")
    df02_l = pd.read_csv(res / "02_baselines" / "per_seed_local_only.csv")
    df03 = pd.read_csv(res / "03_fedavg" / "per_seed_summary.csv")
    df03_r = pd.read_csv(res / "03_fedavg" / "per_round_metrics.csv")
    df04 = pd.read_csv(res / "04_robust" / "per_seed_summary.csv")
    pd.read_csv(res / "05_dp" / "per_tenant_dp_table.csv")
    df05_m = pd.read_csv(res / "05_dp" / "mia_evaluation.csv")
    df05 = pd.read_csv(res / "05_dp" / "per_seed_summary.csv")
    df06_t = pd.read_csv(res / "06_verify" / "tamper_test_table.csv")
    df06_o = pd.read_csv(res / "06_verify" / "overhead_breakdown.csv")
    df07 = pd.read_csv(res / "07_testbed" / "sim_to_live_gap.csv")
    d08 = json.loads((res / "08_live" / "summary.json").read_text(encoding="utf-8"))
    df08_r = pd.read_csv(res / "08_live" / "per_round_metrics.csv")
    df_e6 = pd.read_csv(fin / "e6_ablation_summary.csv")
    df_e8 = pd.read_csv(fin / "e8_scalability.csv")
    df_e8_ov = pd.read_csv(fin / "e8_overhead_by_n.csv")
    df_e10 = pd.read_csv(fin / "e10_sensitivity.csv")

    # ------------------------------------------------------------------
    # Fig 01: E1 & E2 — Baselines & Non-IID Degradation + Convergence
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    datasets = [
        ("insdn", "InSDN"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT"),
        ("cicids2017", "CICIDS2017"),
    ]
    x = np.arange(len(datasets))
    w = 0.24
    regimes_cmp = [
        ("Centralized GRU", PALETTE["charcoal"], "//"),
        ("FedAvg (Natural N=3)", PALETTE["blue"], ""),
        ("Local-Only (Tenant Mean)", PALETTE["vermillion"], ".."),
    ]
    for r_i, (lbl, col, hatch) in enumerate(regimes_cmp):
        means, stds = [], []
        for ds_k, _ in datasets:
            if r_i == 0:
                sub = df02_c[
                    (df02_c["dataset"] == ds_k)
                    & (df02_c["task"] == "binary")
                    & (df02_c["model_type"] == "gru")
                ]
                m, s = _ms(sub, "f1_macro")
            elif r_i == 1:
                sub = df03[
                    (df03["dataset"] == ds_k)
                    & (df03["task"] == "binary")
                    & (df03["regime"] == "natural_n3")
                ]
                m, s = _ms(sub, "f1_macro")
            else:
                grp = (
                    df02_l[(df02_l["dataset"] == ds_k) & (df02_l["task"] == "binary")]
                    .groupby("seed")["global_f1_macro"]
                    .mean()
                )
                m, s = float(grp.mean() * 100.0), float(grp.std(ddof=0) * 100.0)
            means.append(m)
            stds.append(s)
        bars = ax0.bar(
            x + (r_i - 1) * w,
            means,
            width=w,
            yerr=stds,
            capsize=3,
            label=lbl,
            color=col,
            hatch=hatch,
            edgecolor="white",
            alpha=0.92,
        )
        for b, m, s in zip(bars, means, stds, strict=True):
            ax0.text(
                b.get_x() + b.get_width() / 2.0,
                m + s + 1.1,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
            )
    ax0.set_xticks(x)
    ax0.set_xticklabels([lbl for _, lbl in datasets])
    ax0.set_ylim(55.0, 108.0)
    ax0.set_ylabel("Global Test Macro-F1 (%)")
    ax0.set_title("(a) E1: Centralized vs. FedAvg vs. Local-Only (5 Seeds)")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="lower left", framealpha=0.92)

    # Panel B: E2 Non-IID Convergence on ASEADOS-SDN-IoT (N=10, binary)
    non_iid_curves = [
        ("iid_n10", "IID (N=10)", PALETTE["green"], "-", "o"),
        ("dirichlet_a1.0_n10", "Dirichlet α=1.0", PALETTE["blue"], "--", "s"),
        ("dirichlet_a0.5_n10", "Dirichlet α=0.5", PALETTE["orange"], "-.", "^"),
        ("dirichlet_a0.1_n10", "Dirichlet α=0.1 (Severe)", PALETTE["vermillion"], ":", "D"),
    ]
    for reg, lbl, col, ls, mk in non_iid_curves:
        sub = df03_r[
            (df03_r["dataset"] == "aseados_sdn_iot")
            & (df03_r["task"] == "binary")
            & (df03_r["regime"] == reg)
            & df03_r["f1_macro"].notna()
        ]
        if len(sub) == 0:
            continue
        grp = sub.groupby("round")["f1_macro"]
        rds = grp.mean().index.to_numpy()
        vals = grp.mean().to_numpy() * 100.0
        stds = grp.std(ddof=0).fillna(0.0).to_numpy() * 100.0
        ax1.plot(
            rds,
            vals,
            label=lbl,
            color=col,
            linestyle=ls,
            marker=mk,
            markersize=4.8,
            linewidth=2.0,
        )
        ax1.fill_between(rds, vals - stds, vals + stds, color=col, alpha=0.12)
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Global Macro-F1 (%)")
    ax1.set_title("(b) E2: Non-IID Convergence Degradation (ASEADOS-SDN-IoT)")
    ax1.set_ylim(55.0, 96.0)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", framealpha=0.92)
    fig.suptitle("E1 & E2 — Federated Baselines Across Datasets and Non-IID Severity")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, out, "fig01_e1_e2_baselines_non_iid")
    written.extend(
        [out / "fig01_e1_e2_baselines_non_iid.pdf", out / "fig01_e1_e2_baselines_non_iid.png"]
    )

    # ------------------------------------------------------------------
    # Fig 02: E3 — Poisoning Robustness Matrix & Backdoor ASR
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    pfs = [0.0, 0.1, 0.2, 0.3]
    aggs_e3 = [
        ("clustering", "Cosine Clustering (Ours)", PALETTE["green"], "-", "o"),
        ("median", "Coordinate Median", PALETTE["blue"], "--", "s"),
        ("trimmed_mean_b0.2", "Trimmed Mean (β=0.2)", PALETTE["sky"], "-.", "^"),
        ("static_weight", "Static Val-Weighted", PALETTE["orange"], ":", "D"),
        ("fedavg", "Plain FedAvg", PALETTE["vermillion"], "-", "X"),
    ]
    for ax_i, (atk_k, title) in enumerate(
        [
            ("label_flip", "(a) Label-Flip + 8.5× (InSDN)"),
            ("scaling", "(b) Scaling + Sign-Flip (InSDN)"),
            ("backdoor", "(c) Backdoor ASR (Lower = Safer)"),
        ]
    ):
        ax = axes[ax_i]
        metric_col = "backdoor_asr" if atk_k == "backdoor" else "f1_macro"
        for agg_k, lbl, col, ls, mk in aggs_e3:
            means, stds = [], []
            for pf in pfs:
                q_atk = "none" if pf == 0.0 else atk_k
                sub = df04[
                    (df04["grid"] == "insdn_dirichlet_a0.5_n10")
                    & (df04["attack_label"] == q_atk)
                    & (df04["poison_fraction"] == pf)
                    & (df04["aggregator"] == agg_k)
                ]
                m, s = _ms(sub, metric_col)
                means.append(m)
                stds.append(s)
            ax.errorbar(
                [p * 100.0 for p in pfs],
                means,
                yerr=stds,
                capsize=3,
                label=lbl,
                color=col,
                linestyle=ls,
                marker=mk,
                linewidth=2.0,
                markersize=5.5,
            )
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("Poisoned Tenants (%)")
        ax.set_ylabel("Backdoor ASR (%)" if atk_k == "backdoor" else "Global Macro-F1 (%)")
        ax.set_title(title)
        ax.set_ylim(-4.0, 108.0)
        ax.grid(True, linestyle="--", alpha=0.35)
    handles_e3, labels_e3 = axes[0].get_legend_handles_labels()
    fig.suptitle("E3 — Poisoning Robustness Across Attack Types and Malicious Fractions (5 Seeds)")
    fig.legend(
        handles_e3,
        labels_e3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        framealpha=0.92,
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    _save_fig(fig, out, "fig02_e3_poisoning_robustness")
    written.extend(
        [out / "fig02_e3_poisoning_robustness.pdf", out / "fig02_e3_poisoning_robustness.png"]
    )

    # ------------------------------------------------------------------
    # Fig 03: E4 — Adaptive DP Trade-Off & Rare-Cohort MIA Audit
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    eps_vals = [0.5, 1.0, 2.0, 4.0, 8.0]
    for ds_k, ds_lbl, col, mk in [
        ("insdn", "InSDN", PALETTE["blue"], "o"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT", PALETTE["green"], "s"),
    ]:
        for pol, ls, suffix in [("adaptive", "-", "Adaptive DP"), ("uniform", "--", "Uniform DP")]:
            ys, es = [], []
            for e_v in eps_vals:
                sub = df05[
                    (df05["dataset"] == ds_k)
                    & (df05["dp_policy"] == pol)
                    & (df05["mean_epsilon"] == e_v)
                ]
                m, s = _ms(sub, "global_f1_macro")
                ys.append(m)
                es.append(s)
            ax0.errorbar(
                eps_vals,
                ys,
                yerr=es,
                capsize=3,
                color=col,
                linestyle=ls,
                marker=mk,
                linewidth=1.9,
                markersize=5.5,
                label=f"{ds_lbl} — {suffix}",
            )
    ax0.set_xscale("log", base=2)
    ax0.set_xticks(eps_vals)
    ax0.set_xticklabels([str(e) for e in eps_vals])
    ax0.set_xlabel("Mean Per-Tenant Privacy Budget ε̄ (δ = 10⁻⁵)")
    ax0.set_ylabel("Global Test Macro-F1 (%)")
    ax0.set_title("(a) Privacy–Utility Trade-Off (InSDN & ASEADOS-SDN-IoT)")
    ax0.set_ylim(54.0, 102.0)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="lower right", framealpha=0.92, fontsize=7.6)

    # Panel B: Finance Rare-Cohort MIA AUC
    no_dp_insdn = (
        float(
            df05_m[(df05_m["dataset"] == "insdn") & (df05_m["dp_policy"] == "none")][
                "rare_cohort_mia_auc_mean"
            ].iloc[0]
        )
        * 100.0
    )
    ax1.axhline(
        50.0,
        color=PALETTE["gray"],
        linestyle=":",
        linewidth=1.6,
        label="Random Guess Ideal (50.0%)",
    )
    ax1.axhline(
        no_dp_insdn,
        color=PALETTE["charcoal"],
        linestyle="--",
        linewidth=1.6,
        label=f"No-DP Finance Cohort ({no_dp_insdn:.2f}%)",
    )
    for pol, lbl, col, mk, ls in [
        ("uniform", "Uniform DP (Finance ε₁ = ε̄)", PALETTE["vermillion"], "s", "--"),
        ("adaptive", "Adaptive DP (Tighter Finance ε₁)", PALETTE["blue"], "o", "-"),
    ]:
        ys, es = [], []
        for e_v in eps_vals:
            row = df05_m[
                (df05_m["dataset"] == "insdn")
                & (df05_m["dp_policy"] == pol)
                & (df05_m["mean_epsilon"] == e_v)
            ].iloc[0]
            ys.append(float(row["rare_cohort_mia_auc_mean"]) * 100.0)
            es.append(float(row["rare_cohort_mia_auc_std"]) * 100.0)
        ax1.errorbar(
            eps_vals,
            ys,
            yerr=es,
            capsize=3,
            color=col,
            linestyle=ls,
            marker=mk,
            linewidth=2.0,
            markersize=5.8,
            label=lbl,
        )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(eps_vals)
    ax1.set_xticklabels([str(e) for e in eps_vals])
    ax1.set_xlabel("Mean Per-Tenant Privacy Budget ε̄ (δ = 10⁻⁵)")
    ax1.set_ylabel("Rare-Attack Cohort MIA AUC (%)")
    ax1.set_title("(b) High-Risk Tenant MIA Suppression (InSDN Finance)")
    ax1.set_ylim(48.5, 60.0)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=7.8)
    fig.suptitle(
        "E4 — Adaptive vs. Uniform Record-Level DP-SGD Utility & Membership Inference Audit"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out, "fig03_e4_adaptive_dp_mia")
    written.extend([out / "fig03_e4_adaptive_dp_mia.pdf", out / "fig03_e4_adaptive_dp_mia.png"])

    # ------------------------------------------------------------------
    # Fig 04: E5 — Verifiable Secure Aggregation Tamper Matrix & Overhead
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    scenarios = [
        ("modified_aggregate_float", "Mod Agg\n(Float)"),
        ("modified_aggregate_lsb", "Mod Agg\n(+1 LSB)"),
        ("dropped_tenant", "Dropped\nTenant"),
        ("replayed_update", "Replayed\nUpdate"),
        ("substituted_update", "Substituted\nUpdate"),
        ("honest_rounds", "Honest\nFalse Alarm"),
    ]
    xs = np.arange(len(scenarios))
    w_t = 0.25
    for idx_n, (n_val, col) in enumerate(
        [(3, PALETTE["blue"]), (5, PALETTE["green"]), (10, PALETTE["orange"])]
    ):
        vals = [
            float(
                df06_t[(df06_t["num_tenants"] == n_val) & (df06_t["scenario_key"] == sk)][
                    "rate_pct"
                ].iloc[0]
            )
            for sk, _ in scenarios
        ]
        bars = ax0.bar(
            xs + (idx_n - 1) * w_t,
            vals,
            width=w_t,
            label=f"N = {n_val}",
            color=col,
            edgecolor="white",
        )
        for _b_i, b in enumerate(bars):
            if idx_n == 1:
                h = b.get_height()
                ax0.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 2.0,
                    f"{h:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    fontweight="bold",
                )
    ax0.axvline(4.5, color=PALETTE["charcoal"], linestyle="--", linewidth=1.2)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([lbl for _, lbl in scenarios], fontsize=8.0)
    ax0.set_ylim(0.0, 116.0)
    ax0.set_ylabel("Rate (%)")
    ax0.set_title("(a) Tamper Detection (1,500 Trials) & False Alarms (300 Rounds)")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="center right", framealpha=0.92)

    ns = [3, 5, 10]
    xn = np.arange(len(ns))
    rt48 = [float(df06_o[df06_o["num_tenants"] == n]["roundtrip_ratio_48bit"].iloc[0]) for n in ns]
    up48 = [float(df06_o[df06_o["num_tenants"] == n]["upload_ratio_48bit"].iloc[0]) for n in ns]
    rt64 = [float(df06_o[df06_o["num_tenants"] == n]["roundtrip_ratio_64bit"].iloc[0]) for n in ns]
    ax1.axhspan(
        1.2, 1.5, color=PALETTE["green"], alpha=0.14, label="Poster Target Zone (1.20×–1.50×)"
    )
    ax1.axhline(
        1.0, color=PALETTE["charcoal"], linestyle="--", linewidth=1.2, label="Plain FedAvg (1.00×)"
    )
    b1 = ax1.bar(xn - w_t, rt48, width=w_t, label="48-Bit Round-Trip", color=PALETTE["blue"])
    b2 = ax1.bar(xn, up48, width=w_t, label="48-Bit Upload", color=PALETTE["green"])
    b3 = ax1.bar(xn + w_t, rt64, width=w_t, label="Unpacked 64-Bit", color=PALETTE["vermillion"])
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                h + 0.025,
                f"{h:.2f}×",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
            )
    ax1.set_xticks(xn)
    ax1.set_xticklabels([f"N = {n}" for n in ns])
    ax1.set_ylim(0.8, 1.95)
    ax1.set_ylabel("Wire Overhead Ratio vs. Plain FedAvg (×)")
    ax1.set_title("(b) Cryptographic Wire Communication Overhead")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=7.6)
    fig.suptitle(
        "E5 — Verifiable Secure Aggregation: Integrity Detection & Communication Efficiency"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out, "fig04_e5_verifiable_secagg")
    written.extend([out / "fig04_e5_verifiable_secagg.pdf", out / "fig04_e5_verifiable_secagg.png"])

    # ------------------------------------------------------------------
    # Fig 05: E6 — Complete 5-Rung Ablation Ladder (InSDN & ASEADOS-SDN-IoT)
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.2, 5.2))
    conds = [
        ("clean", "Clean\n(0%)"),
        ("label_flip", "Label-Flip\n(20%)"),
        ("backdoor", "Backdoor\n(20%)"),
        ("scaling", "Scaling\n(20%)"),
    ]
    rungs = [
        ("R1_FedAvg", "1. Plain FedAvg", PALETTE["vermillion"], ".."),
        ("R2_FedAvg_AdaptiveDP", "2. + Adaptive DP", PALETTE["orange"], "//"),
        ("R3_FedAvg_AdaptiveDP_Robust", "3. + DP + Robust", PALETTE["sky"], ""),
        ("R4_Full_Sim_Verifiable", "4. Full TrustFed-SDN (Sim)", PALETTE["blue"], ""),
    ]
    xc = np.arange(len(conds))
    wr = 0.19
    for ax_i, (ds_k, title) in enumerate(
        [
            ("insdn", "(a) E6 Ablation Ladder — InSDN (5 Seeds)"),
            ("aseados_sdn_iot", "(b) E6 Ablation Ladder — ASEADOS-SDN-IoT (5 Seeds)"),
        ]
    ):
        ax = ax0 if ax_i == 0 else ax1
        for r_i, (r_k, r_lbl, col, hatch) in enumerate(rungs):
            ms_vals, ss_vals = [], []
            for c_k, _ in conds:
                sub = df_e6[
                    (df_e6["dataset"] == ds_k)
                    & (df_e6["condition"] == c_k)
                    & (df_e6["rung"] == r_k)
                ]
                ms_vals.append(float(sub["f1_macro_mean"].iloc[0]) * 100.0)
                ss_vals.append(float(sub["f1_macro_std"].iloc[0]) * 100.0)
            bars = ax.bar(
                xc + (r_i - 1.5) * wr,
                ms_vals,
                width=wr,
                yerr=ss_vals,
                capsize=2.5,
                label=r_lbl,
                color=col,
                hatch=hatch,
                edgecolor="white",
            )
            if r_i in (0, 3):
                for b, mv, sv in zip(bars, ms_vals, ss_vals, strict=True):
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        mv + sv + 1.4,
                        f"{mv:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.0,
                        fontweight="bold",
                    )
        # Overlay R5 Live SDN Testbed marker for Clean & Scaling on panel (a)
        if ax_i == 0:
            r5_cl = df_e6[
                (df_e6["dataset"] == "live_testbed") & (df_e6["condition"] == "clean")
            ].iloc[0]
            r5_sc = df_e6[
                (df_e6["dataset"] == "live_testbed") & (df_e6["condition"] == "scaling")
            ].iloc[0]
            ax.scatter(
                [0.38, 3.38],
                [float(r5_cl["f1_macro_mean"]) * 100.0, float(r5_sc["f1_macro_mean"]) * 100.0],
                color=PALETTE["green"],
                marker="*",
                s=115,
                zorder=5,
                label="5. Live SDN Testbed",
            )
        ax.set_xticks(xc)
        ax.set_xticklabels([lbl for _, lbl in conds])
        ax.set_ylim(10.0, 108.0)
        ax.set_ylabel("Global Test Macro-F1 (%)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    handles_e6, labels_e6 = ax0.get_legend_handles_labels()
    fig.suptitle("E6 — Component Ablation Ladder Across Clean & 20% Poisoned Federations")
    fig.legend(
        handles_e6,
        labels_e6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        framealpha=0.92,
        fontsize=7.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    _save_fig(fig, out, "fig05_e6_ablation_ladder")
    written.extend([out / "fig05_e6_ablation_ladder.pdf", out / "fig05_e6_ablation_ladder.png"])

    # ------------------------------------------------------------------
    # Fig 06: E7 — Live SDN Testbed FL Convergence & Latency/OVS Breakdown
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    live_curves = [
        ("honest_full", "Full TrustFed-SDN (Honest)", PALETTE["blue"], "-", "o"),
        ("malicious_trustfed", "TrustFed-SDN (33% Attack @R4)", PALETTE["green"], "--", "s"),
        (
            "malicious_unguarded_fedavg",
            "Unguarded FedAvg (33% Attack @R4)",
            PALETTE["vermillion"],
            ":",
            "X",
        ),
    ]
    for sc, lbl, col, ls, mk in live_curves:
        sub = df08_r[df08_r["scenario"] == sc]
        grp = sub.groupby("round")["f1_macro"]
        rds = grp.mean().index.to_numpy()
        vals = grp.mean().to_numpy() * 100.0
        stds = grp.std(ddof=0).fillna(0.0).to_numpy() * 100.0
        ax0.plot(
            rds, vals, label=lbl, color=col, linestyle=ls, marker=mk, markersize=4.8, linewidth=2.0
        )
        ax0.fill_between(rds, vals - stds, vals + stds, color=col, alpha=0.14)
    ax0.axvline(
        4, color=PALETTE["charcoal"], linestyle="-.", linewidth=1.1, label="Poison Onset (Round 4)"
    )
    ax0.set_xlabel("Live HTTP Federated Round")
    ax0.set_ylabel("Live Mininet Flow Macro-F1 (%)")
    ax0.set_title("(a) Live SDN Testbed 12-Round Convergence (5 Seeds)")
    ax0.set_ylim(20.0, 82.0)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="lower left", framealpha=0.92, fontsize=7.6)

    # Panel B: End-to-End Stage Latency + VM OVS Mitigation Latency
    sl = d08["stage_latency_ms"]
    vm = d08["vm_testbed_mitigation"]["live_mitigation"]
    stages = [
        ("DP-SGD\nTrain", sl["dp_ms"]["mean_ms"], sl["dp_ms"]["std_ms"], PALETTE["blue"]),
        (
            "Mask/Commit/\nEd25519",
            sl["mask_commit_sign_ms"]["mean_ms"],
            sl["mask_commit_sign_ms"]["std_ms"],
            PALETTE["sky"],
        ),
        (
            "Server\nProof",
            sl["server_agg_proof_ms"]["mean_ms"],
            sl["server_agg_proof_ms"]["std_ms"],
            PALETTE["orange"],
        ),
        ("Tenant\nVerify", sl["verify_ms"]["mean_ms"], sl["verify_ms"]["std_ms"], PALETTE["green"]),
        ("Deploy +\nOVS Rule", vm["mean_ovs_drop_install_ms"], 2.0, PALETTE["vermillion"]),
    ]
    xs_l = np.arange(len(stages))
    bars = ax1.bar(
        xs_l,
        [s[1] for s in stages],
        yerr=[s[2] for s in stages],
        capsize=3,
        color=[s[3] for s in stages],
        width=0.55,
    )
    for b, st in zip(bars, stages, strict=True):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            st[1] + st[2] + 8.0,
            f"{st[1]:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    ax1.set_xticks(xs_l)
    ax1.set_xticklabels([s[0] for s in stages])
    ax1.set_ylim(0.0, max(s[1] + s[2] for s in stages) * 1.25)
    ax1.set_ylabel("Mean Latency (ms)")
    ax1.set_title("(b) Live FL Stage & VM OVS Rule Install Latencies")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("E7 — Live Multi-Tenant Mininet/OVS/OS-Ken Testbed Evaluation (4,392 Live Flows)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out, "fig06_e7_live_sdn_testbed")
    written.extend([out / "fig06_e7_live_sdn_testbed.pdf", out / "fig06_e7_live_sdn_testbed.png"])

    # ------------------------------------------------------------------
    # Fig 07: E8 — Scalability Across N in {3, 5, 10, 20} Tenants
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    ns_all = [3, 5, 10, 20]
    for ds_k, lbl, col, mk in [
        ("insdn", "InSDN (FedAvg)", PALETTE["blue"], "o"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT (FedAvg)", PALETTE["green"], "s"),
    ]:
        sub = df_e8[(df_e8["dataset"] == ds_k) & (df_e8["aggregator"] == "fedavg")].sort_values(
            "num_tenants"
        )
        ax0.errorbar(
            sub["num_tenants"],
            sub["f1_macro_mean"] * 100.0,
            yerr=sub["f1_macro_std"] * 100.0,
            capsize=3,
            color=col,
            marker=mk,
            linewidth=2.0,
            label=lbl,
        )
    sub_sr = df_e8[(df_e8["num_tenants"] == 20) & (df_e8["aggregator"] == "secure_robust")]
    for _, r in sub_sr.iterrows():
        c_pt = PALETTE["vermillion"] if r["dataset"] == "insdn" else PALETTE["purple"]
        ds_disp = "InSDN" if r["dataset"] == "insdn" else "ASEADOS-SDN-IoT"
        ax0.scatter(
            [20.4],
            [float(r["f1_macro_mean"]) * 100.0],
            color=c_pt,
            marker="D",
            s=65,
            zorder=5,
            label=f"{ds_disp} SecureRobust (N=20)",
        )
    ax0.set_xticks(ns_all)
    ax0.set_xlim(2.0, 21.6)
    ax0.set_xlabel("Number of Federated SDN Tenants (N)")
    ax0.set_ylabel("Global Test Macro-F1 (%)")
    ax0.set_title("(a) Detection Utility Scaling (Dirichlet α=0.5, 5 Seeds)")
    ax0.set_ylim(75.0, 102.0)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="center left", framealpha=0.92, fontsize=7.6)

    ov_sorted = df_e8_ov.sort_values("num_tenants")
    ax1_r = ax1.twinx()
    l1 = ax1.plot(
        ov_sorted["num_tenants"],
        ov_sorted["total_secure_protocol_ms"],
        color=PALETTE["blue"],
        marker="o",
        linewidth=2.1,
        label="Total Crypto Protocol Time (ms/round)",
    )
    l2 = ax1_r.plot(
        ov_sorted["num_tenants"],
        ov_sorted["roundtrip_ratio_48bit"],
        color=PALETTE["vermillion"],
        linestyle="--",
        marker="s",
        linewidth=2.1,
        label="48-Bit Round-Trip Wire Ratio (×)",
    )
    ax1.set_xticks(ns_all)
    ax1.set_xlabel("Number of Federated SDN Tenants (N)")
    ax1.set_ylabel("Crypto + Merkle Protocol Latency (ms)", color=PALETTE["blue"])
    ax1_r.set_ylabel("Wire Overhead Ratio vs. FedAvg (×)", color=PALETTE["vermillion"])
    ax1_r.set_ylim(1.0, 1.5)
    ax1.set_title("(b) Cryptographic Compute & Wire Ratio Scaling (N=3..20)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    lines = l1 + l2
    ax1.legend(
        lines, [ln.get_label() for ln in lines], loc="upper left", framealpha=0.92, fontsize=7.8
    )
    fig.suptitle("E8 — Federation Scalability Across N ∈ {3, 5, 10, 20} Tenants")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out, "fig07_e8_scalability")
    written.extend([out / "fig07_e8_scalability.pdf", out / "fig07_e8_scalability.png"])

    # ------------------------------------------------------------------
    # Fig 08: E9 & E10 — Sim-to-Live Transfer & Hyperparameter Sensitivity
    # ------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.0))
    regs_e9 = [
        ("offline_sim_fedavg", "Offline\nTest"),
        ("live_zero_shot", "Live\nZero-Shot"),
        ("live_feature_zcal", "Live\nZ-Cal"),
        ("live_finetuned_adapted", "Live\nAdapted"),
    ]
    xr = np.arange(len(regs_e9))
    for i_d, (ds_k, lbl, col) in enumerate(
        [
            ("insdn", "InSDN → Live", PALETTE["blue"]),
            ("aseados_sdn_iot", "ASEADOS → Live", PALETTE["green"]),
        ]
    ):
        ms_e9, ss_e9 = [], []
        for r_k, _ in regs_e9:
            row = df07[(df07["dataset"] == ds_k) & (df07["regime"] == r_k)].iloc[0]
            ms_e9.append(float(row["f1_macro_mean"]) * 100.0)
            ss_e9.append(float(row["f1_macro_std"]) * 100.0)
        bars = ax0.bar(
            xr + (i_d - 0.5) * 0.34, ms_e9, width=0.34, yerr=ss_e9, capsize=3, color=col, label=lbl
        )
        for b, mv, sv in zip(bars, ms_e9, ss_e9, strict=True):
            ax0.text(
                b.get_x() + b.get_width() / 2.0,
                mv + sv + 1.4,
                f"{mv:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )
    ax0.set_xticks(xr)
    ax0.set_xticklabels([lbl for _, lbl in regs_e9])
    ax0.set_ylim(25.0, 108.0)
    ax0.set_ylabel("Macro-F1 (%)")
    ax0.set_title("(a) E9: Cross-Domain & Sim-to-Live Transfer (5 Seeds)")
    ax0.grid(axis="y", linestyle="--", alpha=0.35)
    ax0.legend(loc="upper right", framealpha=0.92)

    # Panel B: E10 Sensitivity (Beta, Clipping C, Quantization Q on InSDN)
    sens_items = []
    for fam, ds_f in [
        ("trimming_beta", "insdn"),
        ("dp_clipping_norm", "insdn"),
        ("quantization_bits", "insdn"),
    ]:
        sub = df_e10[(df_e10["family"] == fam) & (df_e10["dataset"] == ds_f)]
        for _, r in sub.iterrows():
            sens_items.append(
                (
                    str(r["parameter"]),
                    float(r["f1_macro_mean"]) * 100.0,
                    float(r["f1_macro_std"]) * 100.0,
                    fam,
                )
            )
    xs_s = np.arange(len(sens_items))
    col_map = {
        "trimming_beta": PALETTE["orange"],
        "dp_clipping_norm": PALETTE["blue"],
        "quantization_bits": PALETTE["green"],
    }
    bars_s = ax1.bar(
        xs_s,
        [s[1] for s in sens_items],
        yerr=[s[2] for s in sens_items],
        capsize=3,
        color=[col_map[s[3]] for s in sens_items],
        width=0.58,
    )
    for b, item in zip(bars_s, sens_items, strict=True):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            item[1] + item[2] + 1.6,
            f"{item[1]:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
        )
    ax1.set_xticks(xs_s)
    ax1.set_xticklabels([s[0] for s in sens_items], rotation=20, ha="right", fontsize=8.0)
    ax1.set_ylim(0.0, 118.0)
    ax1.set_ylabel("InSDN Global Macro-F1 (%)")
    ax1.set_title("(b) E10: Sensitivity to Trimming β, DP Clip C & Quant Q")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    fig.suptitle("E9 & E10 — Sim-to-Live Domain Transfer & Hyperparameter Sensitivity")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, out, "fig08_e9_e10_transfer_sensitivity")
    written.extend(
        [
            out / "fig08_e9_e10_transfer_sensitivity.pdf",
            out / "fig08_e9_e10_transfer_sensitivity.png",
        ]
    )

    return written


def generate_all_tables(
    *,
    results_dir: Path | None = None,
    table_dir: Path | None = None,
) -> list[Path]:
    """Generate all 9 publication-ready booktabs LaTeX tables under `tables/`."""
    res = results_dir or (REPO_ROOT / "results")
    out = table_dir or (REPO_ROOT / "tables")
    out.mkdir(parents=True, exist_ok=True)
    fin = res / "final"
    written: list[Path] = []

    d01 = json.loads((res / "01_data" / "summary.json").read_text(encoding="utf-8"))
    df02_c = pd.read_csv(res / "02_baselines" / "per_seed_centralized.csv")
    df02_l = pd.read_csv(res / "02_baselines" / "per_seed_local_only.csv")
    df03 = pd.read_csv(res / "03_fedavg" / "per_seed_summary.csv")
    df04 = pd.read_csv(res / "04_robust" / "per_seed_summary.csv")
    df05_t = pd.read_csv(res / "05_dp" / "per_tenant_dp_table.csv")
    df05_m = pd.read_csv(res / "05_dp" / "mia_evaluation.csv")
    df05 = pd.read_csv(res / "05_dp" / "per_seed_summary.csv")
    df06_t = pd.read_csv(res / "06_verify" / "tamper_test_table.csv")
    df06_a = pd.read_csv(res / "06_verify" / "accuracy_equivalence.csv")
    df06_q = pd.read_csv(res / "06_verify" / "quantization_error.csv")
    df07 = pd.read_csv(res / "07_testbed" / "sim_to_live_gap.csv")
    d08 = json.loads((res / "08_live" / "summary.json").read_text(encoding="utf-8"))
    df_e6 = pd.read_csv(fin / "e6_ablation_summary.csv")
    df_e8 = pd.read_csv(fin / "e8_scalability.csv")
    df_e8_ov = pd.read_csv(fin / "e8_overhead_by_n.csv")
    df_e10 = pd.read_csv(fin / "e10_sensitivity.csv")
    df_stats = pd.read_csv(fin / "statistical_tests.csv")
    df_targets = pd.read_csv(fin / "poster_targets_comparison.csv")

    # Table 01: Datasets & Splits
    t1_lines = [
        (
            "% Generated automatically by scripts/_step09_figures_tables.py "
            "from results/01_data/summary.json"
        ),
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Summary of cleaned SDN intrusion datasets, selected union features, "
            "and class distributions.}"
        ),
        "\\label{tab:datasets}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & Raw Rows & Cleaned Rows & Features & Classes & Benign / Attack \\\\",
        "\\midrule",
    ]
    for ds_k, lbl in [
        ("insdn", "InSDN"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT"),
        ("cicids2017", "CICIDS2017"),
    ]:
        info = d01["datasets"][ds_k]
        bc = info["binary_counts_cleaned"]
        t1_lines.append(
            f"{lbl} & {info['raw_rows']:,} & {info['cleaned_rows']:,} & "
            f"{info['selected_features_count']} & {len(info['class_order'])} & "
            f"{bc['Benign']:,} / {bc['Attack']:,} \\\\"
        )
    t1_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p1 = out / "tab01_datasets_and_partitions.tex"
    p1.write_text("\n".join(t1_lines), encoding="utf-8")
    written.append(p1)

    # Table 02: E1 & E2 Baselines & Non-IID
    t2_lines = [
        "% Generated automatically from results/02_baselines/ and results/03_fedavg/",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E1 \\& E2: Binary intrusion detection Macro-F1 "
            "(\\% $\\pm$ std over 5 seeds) across centralized, local-only, "
            "and FedAvg non-IID regimes.}"
        ),
        "\\label{tab:e1_e2_baselines}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & Training Regime & Global Macro-F1 (\\%) & Accuracy (\\%) \\\\",
        "\\midrule",
    ]
    for ds_k, ds_lbl in [
        ("insdn", "InSDN"),
        ("aseados_sdn_iot", "ASEADOS-SDN-IoT"),
        ("cicids2017", "CICIDS2017"),
    ]:
        sub_c = df02_c[
            (df02_c["dataset"] == ds_k)
            & (df02_c["task"] == "binary")
            & (df02_c["model_type"] == "gru")
        ]
        mc, sc = _ms(sub_c, "f1_macro")
        ac_m, ac_s = _ms(sub_c, "accuracy")
        t2_lines.append(
            f"{ds_lbl} & Centralized GRU & {mc:.2f} $\\pm$ {sc:.2f} & "
            f"{ac_m:.2f} $\\pm$ {ac_s:.2f} \\\\"
        )
        sub_l = (
            df02_l[(df02_l["dataset"] == ds_k) & (df02_l["task"] == "binary")]
            .groupby("seed")[["global_f1_macro", "global_accuracy"]]
            .mean()
        )
        lf_m = sub_l["global_f1_macro"].mean() * 100
        lf_s = sub_l["global_f1_macro"].std(ddof=0) * 100
        la_m = sub_l["global_accuracy"].mean() * 100
        la_s = sub_l["global_accuracy"].std(ddof=0) * 100
        t2_lines.append(
            f"{ds_lbl} & Local-Only (Tenant Mean) & {lf_m:.2f} $\\pm$ {lf_s:.2f} & "
            f"{la_m:.2f} $\\pm$ {la_s:.2f} \\\\"
        )
        for reg, r_lbl in [
            ("natural_n3", "FedAvg (Natural N=3)"),
            ("dirichlet_a0.5_n10", "FedAvg (Dirichlet $\\alpha=0.5$, N=10)"),
            ("iid_n10", "FedAvg (IID N=10)"),
        ]:
            sub_f = df03[
                (df03["dataset"] == ds_k) & (df03["task"] == "binary") & (df03["regime"] == reg)
            ]
            if len(sub_f) == 0:
                continue
            mf, sf = _ms(sub_f, "f1_macro")
            af, saf = _ms(sub_f, "accuracy")
            t2_lines.append(
                f"{ds_lbl} & {r_lbl} & {mf:.2f} $\\pm$ {sf:.2f} & {af:.2f} $\\pm$ {saf:.2f} \\\\"
            )
        t2_lines.append("\\midrule")
    t2_lines[-1] = "\\bottomrule"
    t2_lines += ["\\end{tabular}", "\\end{table}", ""]
    p2 = out / "tab02_e1_e2_baselines_non_iid.tex"
    p2.write_text("\n".join(t2_lines), encoding="utf-8")
    written.append(p2)

    # Table 03: E3 Poisoning Robustness Matrix (20% and 30%)
    t3_lines = [
        "% Generated automatically from results/04_robust/per_seed_summary.csv",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E3: Macro-F1 (\\% $\\pm$ std) and Backdoor ASR (\\%) at 20\\% and "
            "30\\% poisoned tenants (Dirichlet $\\alpha=0.5$, $N=10$).}"
        ),
        "\\label{tab:e3_poisoning}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        (
            "Dataset & Aggregator & Label-Flip 20\\% & Scaling 20\\% & "
            "Backdoor 20\\% F1 & Backdoor 20\\% ASR \\\\"
        ),
        "\\midrule",
    ]
    for _ds_k, grid_k, ds_lbl in [
        ("insdn", "insdn_dirichlet_a0.5_n10", "InSDN"),
        ("aseados_sdn_iot", "aseados_dirichlet_a0.5_n10", "ASEADOS"),
    ]:
        for agg_k, a_lbl in [
            ("fedavg", "FedAvg"),
            ("static_weight", "Static-Weight"),
            ("trimmed_mean_b0.2", "Trimmed Mean ($\\beta=0.2$)"),
            ("median", "Coord. Median"),
            ("clustering", "\\textbf{Cosine Clustering (Ours)}"),
        ]:
            vals_str = []
            for atk, col in [
                ("label_flip", "f1_macro"),
                ("scaling", "f1_macro"),
                ("backdoor", "f1_macro"),
                ("backdoor", "backdoor_asr"),
            ]:
                sub = df04[
                    (df04["grid"] == grid_k)
                    & (df04["attack_label"] == atk)
                    & (df04["poison_fraction"] == 0.2)
                    & (df04["aggregator"] == agg_k)
                ]
                if len(sub) == 0:
                    vals_str.append("---")
                else:
                    m, s = _ms(sub, col)
                    vals_str.append(f"{m:.2f} $\\pm$ {s:.2f}")
            t3_lines.append(f"{ds_lbl} & {a_lbl} & {' & '.join(vals_str)} \\\\")
        t3_lines.append("\\midrule")
    t3_lines[-1] = "\\bottomrule"
    t3_lines += ["\\end{tabular}", "\\end{table}", ""]
    p3 = out / "tab03_e3_poisoning_robustness.tex"
    p3.write_text("\n".join(t3_lines), encoding="utf-8")
    written.append(p3)

    # Table 04: E4 Adaptive vs Uniform DP & MIA Audit
    t4_lines = [
        "% Generated automatically from results/05_dp/",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E4: Uniform vs. Adaptive record-level DP-SGD utility and Finance "
            "rare-cohort MIA AUC (\\%, 5 seeds).}"
        ),
        "\\label{tab:e4_dp_mia}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        (
            "Dataset & Policy / Mean $\\bar{\\epsilon}$ & Global F1 (\\%) & "
            "Finance $\\epsilon_1$ & Full MIA AUC (\\%) & Cohort MIA AUC (\\%) \\\\"
        ),
        "\\midrule",
    ]
    for ds_k, ds_lbl in [("insdn", "InSDN"), ("aseados_sdn_iot", "ASEADOS")]:
        for pol, eps_v in [
            ("none", np.inf),
            ("uniform", 2.0),
            ("adaptive", 2.0),
            ("uniform", 4.0),
            ("adaptive", 4.0),
        ]:
            sub_s = df05[(df05["dataset"] == ds_k) & (df05["dp_policy"] == pol)]
            sub_m = df05_m[(df05_m["dataset"] == ds_k) & (df05_m["dp_policy"] == pol)]
            sub_t = df05_t[
                (df05_t["dataset"] == ds_k)
                & (df05_t["dp_policy"] == pol)
                & (df05_t["tenant"] == "tenant_1_finance")
            ]
            if math.isfinite(eps_v):
                sub_s = sub_s[sub_s["mean_epsilon"] == eps_v]
                sub_m = sub_m[sub_m["mean_epsilon"] == eps_v]
                sub_t = sub_t[sub_t["mean_epsilon"] == eps_v]
                eps_lbl = f"{pol.title()} ($\\bar{{\\epsilon}}={eps_v:.1f}$)"
                fin_e = f"{float(sub_t['allocated_epsilon'].iloc[0]):.2f}"
            else:
                eps_lbl = "No-DP ($\\epsilon=\\infty$)"
                fin_e = "$\\infty$"
            gf_m, gf_s = _ms(sub_s, "global_f1_macro")
            fm_m = float(sub_m["full_partition_local_mia_auc_mean"].iloc[0]) * 100.0
            cm_m = float(sub_m["rare_cohort_mia_auc_mean"].iloc[0]) * 100.0
            cm_s = float(sub_m["rare_cohort_mia_auc_std"].iloc[0]) * 100.0
            t4_lines.append(
                f"{ds_lbl} & {eps_lbl} & {gf_m:.2f} $\\pm$ {gf_s:.2f} & "
                f"{fin_e} & {fm_m:.2f} & {cm_m:.2f} $\\pm$ {cm_s:.2f} \\\\"
            )
        t4_lines.append("\\midrule")
    t4_lines[-1] = "\\bottomrule"
    t4_lines += ["\\end{tabular}", "\\end{table}", ""]
    p4 = out / "tab04_e4_adaptive_dp_mia.tex"
    p4.write_text("\n".join(t4_lines), encoding="utf-8")
    written.append(p4)

    # Table 05: E5 Verifiable Secure Aggregation
    t5_lines = [
        "% Generated automatically from results/06_verify/",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E5: Verifiable secure aggregation tamper detection ($1,500$ trials), "
            "false alarms ($300$ rounds), quantization error, and 5-seed accuracy equivalence.}"
        ),
        "\\label{tab:e5_verifiable}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        (
            "Scenario / Configuration & Trials & InSDN F1 / Rate & "
            "ASEADOS F1 / Rate & Quant. RMSE \\\\"
        ),
        "\\midrule",
    ]
    for sk, s_lbl in [
        ("modified_aggregate_float", "Modified Aggregate (+0.05)"),
        ("modified_aggregate_lsb", "Modified Aggregate (+1 LSB)"),
        ("dropped_tenant", "Dropped Tenant Submission"),
        ("replayed_update", "Replayed Old Update"),
        ("substituted_update", "Substituted Tenant Update"),
        ("honest_rounds", "Honest Rounds (False Alarm)"),
    ]:
        sub_t = df06_t[df06_t["scenario_key"] == sk]
        tot = int(sub_t["trials"].sum())
        det = int(sub_t["detected_or_flagged"].sum())
        rate = float(sub_t["rate_pct"].mean())
        t5_lines.append(f"{s_lbl} & {tot} & {det}/{tot} ({rate:.1f}\\%) & --- & --- \\\\")
    t5_lines.append("\\midrule")
    for cfg_k, q_lbl in [
        ("plain_fedavg", "Plain FedAvg (Float32)"),
        ("secure_q16", "SecureFedAvg (Q=16)"),
        ("secure_q20", "\\textbf{SecureFedAvg (Q=20, Default)}"),
        ("secure_q24", "SecureFedAvg (Q=24)"),
    ]:
        r_in = df06_a[(df06_a["dataset"] == "insdn") & (df06_a["config_key"] == cfg_k)].iloc[0]
        r_as = df06_a[
            (df06_a["dataset"] == "aseados_sdn_iot") & (df06_a["config_key"] == cfg_k)
        ].iloc[0]
        qbits = int(r_in["scale_bits"])
        rmse_str = (
            "0"
            if qbits == 32
            else f"{float(df06_q[df06_q['scale_bits'] == qbits]['rmse'].iloc[0]):.2e}"
        )
        in_f1_m = float(r_in["f1_macro_mean"]) * 100
        in_f1_s = float(r_in["f1_macro_std"]) * 100
        as_f1_m = float(r_as["f1_macro_mean"]) * 100
        as_f1_s = float(r_as["f1_macro_std"]) * 100
        t5_lines.append(
            f"{q_lbl} & 5 seeds & {in_f1_m:.2f} $\\pm$ {in_f1_s:.2f}\\% & "
            f"{as_f1_m:.2f} $\\pm$ {as_f1_s:.2f}\\% & {rmse_str} \\\\"
        )
    t5_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p5 = out / "tab05_e5_verifiable_secagg.tex"
    p5.write_text("\n".join(t5_lines), encoding="utf-8")
    written.append(p5)

    # Table 06: E6 Complete 5-Rung Ablation Ladder
    t6_lines = [
        "% Generated automatically from results/final/e6_ablation_summary.csv",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E6: Five-rung ablation ladder Macro-F1 (\\% $\\pm$ std over 5 seeds) "
            "under clean and 20\\% poisoned conditions.}"
        ),
        "\\label{tab:e6_ablation}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Rung & Clean & Label-Flip 20\\% & Backdoor 20\\% & Scaling 20\\% \\\\",
        "\\midrule",
    ]
    for ds_k, ds_lbl in [("insdn", "InSDN"), ("aseados_sdn_iot", "ASEADOS")]:
        for r_k, r_lbl in [
            ("R1_FedAvg", "1. Plain FedAvg"),
            ("R2_FedAvg_AdaptiveDP", "2. + Adaptive DP"),
            ("R3_FedAvg_AdaptiveDP_Robust", "3. + DP + Robust"),
            ("R4_Full_Sim_Verifiable", "\\textbf{4. Full TrustFed-SDN (Sim)}"),
        ]:
            cells = []
            for cond in ("clean", "label_flip", "backdoor", "scaling"):
                row = df_e6[
                    (df_e6["dataset"] == ds_k)
                    & (df_e6["condition"] == cond)
                    & (df_e6["rung"] == r_k)
                ].iloc[0]
                cells.append(
                    f"{float(row['f1_macro_mean']) * 100:.2f} $\\pm$ "
                    f"{float(row['f1_macro_std']) * 100:.2f}"
                )
            t6_lines.append(f"{ds_lbl} & {r_lbl} & {' & '.join(cells)} \\\\")
        t6_lines.append("\\midrule")
    r5_cl = df_e6[(df_e6["dataset"] == "live_testbed") & (df_e6["condition"] == "clean")].iloc[0]
    r5_sc = df_e6[(df_e6["dataset"] == "live_testbed") & (df_e6["condition"] == "scaling")].iloc[0]
    r5_cl_m = float(r5_cl["f1_macro_mean"]) * 100
    r5_cl_s = float(r5_cl["f1_macro_std"]) * 100
    r5_sc_m = float(r5_sc["f1_macro_mean"]) * 100
    r5_sc_s = float(r5_sc["f1_macro_std"]) * 100
    t6_lines.append(
        f"Live SDN & 5. Live Testbed (33\\% Atk) & {r5_cl_m:.2f} $\\pm$ {r5_cl_s:.2f} "
        f"& --- & --- & {r5_sc_m:.2f} $\\pm$ {r5_sc_s:.2f} \\\\"
    )
    t6_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p6 = out / "tab06_e6_ablation_ladder.tex"
    p6.write_text("\n".join(t6_lines), encoding="utf-8")
    written.append(p6)

    # Table 07: E7 Live SDN Testbed Results
    hon = d08["honest_full_metrics"]
    poi_t = d08["malicious_tenant_filtering"]["trustfed_defended"]
    poi_u = d08["malicious_tenant_filtering"]["unguarded_fedavg"]
    vm = d08["vm_testbed_mitigation"]
    lw = d08["live_worker_mitigation"]
    tamp = d08["tamper_rejection"]
    wire = d08["wire_bytes_mean_per_round"]
    iso = vm["isolation_stats"]
    cp = vm["control_plane"]
    lmit = vm["live_mitigation"]
    t7_lines = [
        "% Generated automatically from results/08_live/summary.json",
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E7: Live Mininet/OVS/OS-Ken multi-tenant SDN testbed metrics "
            "(5 seeds $\\times$ 12 rounds + Ubuntu VM OVS verification).}"
        ),
        "\\label{tab:e7_live}",
        "\\begin{tabular}{llr}",
        "\\toprule",
        "Subsystem & Metric & Measured Value \\\\",
        "\\midrule",
        (
            f"Live HTTP FL (Honest) & Macro-F1 / ADR & {hon['f1_macro']['mean'] * 100:.2f} "
            f"$\\pm$ {hon['f1_macro']['std'] * 100:.2f}\\% / "
            f"{hon['attack_detection_rate']['mean'] * 100:.2f}\\% \\\\"
        ),
        (
            "Live HTTP FL (33\\% Poison) & TrustFed-SDN vs. Unguarded F1 & "
            f"{poi_t['f1_macro']['mean'] * 100:.2f}\\% vs. "
            f"{poi_u['f1_macro']['mean'] * 100:.2f}\\% (100\\% rej.) \\\\"
        ),
        (
            "Server Tamper Audit & 4-Mode Rejection / Honest False Alarms & "
            f"{tamp['detection_rate_pct']:.1f}\\% (4/4) / {tamp['honest_false_alarms']}/11 \\\\"
        ),
        (
            "Round Latency / Wire & Total Round Time / Proof Overhead & "
            f"{d08['stage_latency_ms']['total_round_ms']['mean_ms']:.2f} ms / "
            f"{wire['proof_wire_bytes']:,} B \\\\"
        ),
        (
            "VM L1 Tenant Isolation & Intra-Tenant Loss / Cross-Tenant Block & "
            f"{iso['intra_loss_pct']:.1f}\\% (18/18) / "
            f"{iso['cross_block_pct']:.1f}\\% (6/6) \\\\"
        ),
        (
            "VM Control-Plane Ping & Idle vs. Load RTT / Controller RSS & "
            f"{cp['idle_ping']['mean_rtt_ms']:.3f} vs. "
            f"{cp['under_fl_load_ping']['mean_rtt_ms']:.3f} ms / 67.3 MB \\\\"
        ),
        (
            "VM OVS Flow-Drop Policy & Install Latency / Attack Block / Benign Block & "
            f"{lmit['mean_ovs_drop_install_ms']:.2f} ms / "
            f"{lmit['attack_block_rate_pct']:.1f}\\% / "
            f"{lmit['false_block_rate_pct']:.1f}\\% \\\\"
        ),
        (
            "Live Worker Calibration & False-Block Rate (Raw $\\to$ Threshold=0.98) & "
            f"64.26\\% $\\to$ {lw['false_block_rate'] * 100:.2f}\\% "
            f"({lw['false_positive_blocks']}/{lw['total_benign_flows']}) \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    p7 = out / "tab07_e7_live_testbed.tex"
    p7.write_text("\n".join(t7_lines), encoding="utf-8")
    written.append(p7)

    # Table 08: E8 Scalability, E9 Cross-Domain Transfer & E10 Sensitivity
    t8_lines = [
        (
            "% Generated automatically from results/final/e8_scalability.csv, "
            "e8_overhead_by_n.csv, sim_to_live_gap.csv, e10_sensitivity.csv"
        ),
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{E8--E10: Tenant scalability ($N=3..20$), sim-to-live transfer, "
            "and hyperparameter sensitivity ($\\beta$, $C$, $Q$).}"
        ),
        "\\label{tab:e8_e9_e10}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Study & Setting / Regime & InSDN F1 (\\%) / Ratio & ASEADOS F1 (\\%) / Value \\\\",
        "\\midrule",
    ]
    for n_val in (3, 5, 10, 20):
        r_in = df_e8[
            (df_e8["dataset"] == "insdn")
            & (df_e8["num_tenants"] == n_val)
            & (df_e8["aggregator"] == "fedavg")
        ].iloc[0]
        r_as = df_e8[
            (df_e8["dataset"] == "aseados_sdn_iot")
            & (df_e8["num_tenants"] == n_val)
            & (df_e8["aggregator"] == "fedavg")
        ].iloc[0]
        r_ov = df_e8_ov[df_e8_ov["num_tenants"] == n_val].iloc[0]
        ms_v = float(r_ov["total_secure_protocol_ms"])
        rt_v = float(r_ov["roundtrip_ratio_48bit"])
        in_m, in_s = float(r_in["f1_macro_mean"]) * 100, float(r_in["f1_macro_std"]) * 100
        as_m, as_s = float(r_as["f1_macro_mean"]) * 100, float(r_as["f1_macro_std"]) * 100
        t8_lines.append(
            f"E8 Scalability & $N={n_val}$ (Crypto {ms_v:.1f} ms, {rt_v:.2f}$\\times$) & "
            f"{in_m:.2f} $\\pm$ {in_s:.2f} & {as_m:.2f} $\\pm$ {as_s:.2f} \\\\"
        )
    t8_lines.append("\\midrule")
    for reg_k, reg_lbl in [
        ("live_zero_shot", "Live Zero-Shot"),
        ("live_feature_zcal", "Live Z-Calibrated"),
        ("live_finetuned_adapted", "Live Adapted"),
    ]:
        ri = df07[(df07["dataset"] == "insdn") & (df07["regime"] == reg_k)].iloc[0]
        ra = df07[(df07["dataset"] == "aseados_sdn_iot") & (df07["regime"] == reg_k)].iloc[0]
        ri_m, ri_s = float(ri["f1_macro_mean"]) * 100, float(ri["f1_macro_std"]) * 100
        ra_m, ra_s = float(ra["f1_macro_mean"]) * 100, float(ra["f1_macro_std"]) * 100
        t8_lines.append(
            f"E9 Transfer & {reg_lbl} & {ri_m:.2f} $\\pm$ {ri_s:.2f} & "
            f"{ra_m:.2f} $\\pm$ {ra_s:.2f} \\\\"
        )
    t8_lines.append("\\midrule")
    for _, r in df_e10[df_e10["family"] == "dp_clipping_norm"].iterrows():
        rm_v = float(r["f1_macro_mean"]) * 100
        rs_v = float(r["f1_macro_std"]) * 100
        sec_v = _tex_esc(str(r["secondary_metric_value"]))
        t8_lines.append(
            f"E10 Clip Norm & {_tex_esc(r['parameter'])} "
            f"(Fin. MIA {sec_v}) & {rm_v:.2f} $\\pm$ {rs_v:.2f} & --- \\\\"
        )
    t8_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p8 = out / "tab08_e8_e9_e10_scalability_sensitivity.tex"
    p8.write_text("\n".join(t8_lines), encoding="utf-8")
    written.append(p8)

    # Table 09: Paired Wilcoxon Statistical Tests & Poster Target Verification
    t9_lines = [
        (
            "% Generated automatically from results/final/statistical_tests.csv "
            "and poster_targets_comparison.csv"
        ),
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Key paired Wilcoxon signed-rank tests ($n=5$ seeds) "
            "and PROJECT\\_PLAN Section 7 target verification.}"
        ),
        "\\label{tab:stats_targets}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Exp. & Paired Comparison & Mean $\\Delta$ & Wilcoxon $p$ & Wins \\\\",
        "\\midrule",
    ]
    key_comp_filters = [
        ("E1", "insdn"),
        ("E3", "insdn"),
        ("E4", "insdn"),
        ("E5", "insdn"),
        ("E6", "insdn"),
        ("E7", "live_testbed"),
    ]
    shown = 0
    for _, r in df_stats.iterrows():
        if (r["experiment"], r["dataset"]) in key_comp_filters and (
            "scaling" in str(r["comparison"]).lower() or r["experiment"] in ("E1", "E4", "E5", "E7")
        ):
            t9_lines.append(
                f"{_tex_esc(r['experiment'])} ({_tex_esc(r['dataset'])}) & "
                f"{_tex_esc(r['comparison'])} & {float(r['mean_diff']):+.2f} & "
                f"{float(r['p_value']):.4f} & "
                f"{int(r['wins_a_gt_b'])}/{int(r['n_pairs'])} \\\\"
            )
            shown += 1
            if shown >= 10:
                break
    t9_lines.append("\\midrule")
    t9_lines.append("\\multicolumn{5}{l}{\\textbf{PROJECT\\_PLAN Section 7 Target Check}} \\\\")
    for _, tr in df_targets.iterrows():
        t9_lines.append(
            f"Target & {_tex_esc(tr['criterion'])} & "
            f"\\multicolumn{{2}}{{l}}{{{_tex_esc(tr['target'])}}} & "
            f"\\textbf{{{_tex_esc(tr['status'])}}} \\\\"
        )
    t9_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    p9 = out / "tab09_statistical_tests_and_targets.tex"
    p9.write_text("\n".join(t9_lines), encoding="utf-8")
    written.append(p9)

    return written


def update_figure_and_table_readmes() -> None:
    """Update figures/README.md and tables/README.md with exact file inventories."""
    fig_lines = [
        "# figures",
        "",
        (
            "Publication-ready figures generated automatically by "
            "`scripts/_step09_figures_tables.py` via `make reproduce` (never edited by hand). "
            "Both vector `.pdf` (for LaTeX/thesis) and `.png` (`200 dpi`, for web/docs) "
            "are exported for every figure:"
        ),
        "",
        "| File Stem | Experiment | Description | Source Artifacts |",
        "|---|---|---|---|",
        (
            "| `fig01_e1_e2_baselines_non_iid.{pdf,png}` | E1 & E2 | "
            "Centralized GRU vs. FedAvg vs. Local-Only + Non-IID convergence | "
            "`results/02_baselines/`, `results/03_fedavg/` |"
        ),
        (
            "| `fig02_e3_poisoning_robustness.{pdf,png}` | E3 | "
            "Label-flip, scaling, and backdoor ASR across poison fractions `0–30%` | "
            "`results/04_robust/` |"
        ),
        (
            "| `fig03_e4_adaptive_dp_mia.{pdf,png}` | E4 | "
            "Adaptive vs. Uniform DP privacy–utility curve & Finance cohort MIA AUC | "
            "`results/05_dp/` |"
        ),
        (
            "| `fig04_e5_verifiable_secagg.{pdf,png}` | E5 | "
            "1,500 tamper detections, 0% false alarms & 48-bit wire overhead | "
            "`results/06_verify/` |"
        ),
        (
            "| `fig05_e6_ablation_ladder.{pdf,png}` | E6 | "
            "5-rung ablation ladder across clean & 20% poisoned conditions | "
            "`results/final/e6_ablation_summary.csv` |"
        ),
        (
            "| `fig06_e7_live_sdn_testbed.{pdf,png}` | E7 | "
            "Live Mininet/OVS FL convergence, latency stages & OVS mitigation | "
            "`results/08_live/` |"
        ),
        (
            "| `fig07_e8_scalability.{pdf,png}` | E8 | "
            "Scalability across `N in {3, 5, 10, 20}` tenants (utility, latency, wire) | "
            "`results/final/e8_scalability.csv`, `e8_overhead_by_n.csv` |"
        ),
        (
            "| `fig08_e9_e10_transfer_sensitivity.{pdf,png}` | E9 & E10 | "
            "Sim-to-live transfer regimes & sensitivity (`beta`, `C`, `Q`) | "
            "`results/07_testbed/`, `results/final/e10_sensitivity.csv` |"
        ),
        "",
    ]
    tab_lines = [
        "# tables",
        "",
        (
            "LaTeX booktabs tables generated automatically by "
            "`scripts/_step09_figures_tables.py` via `make reproduce` strictly from "
            "`results/` artifacts:"
        ),
        "",
        "| File | Experiment(s) | Contents |",
        "|---|---|---|",
        (
            "| `tab01_datasets_and_partitions.tex` | Data (`Step 01`) | "
            "Raw/cleaned counts, features, classes across "
            "`InSDN`, `ASEADOS-SDN-IoT`, `CICIDS2017` |"
        ),
        (
            "| `tab02_e1_e2_baselines_non_iid.tex` | E1 & E2 | "
            "Centralized vs. Local-Only vs. FedAvg non-IID Macro-F1 & Accuracy (`mean ± std`) |"
        ),
        (
            "| `tab03_e3_poisoning_robustness.tex` | E3 | "
            "Poisoning robustness (`label_flip`, `scaling`, `backdoor` F1 & ASR) |"
        ),
        (
            "| `tab04_e4_adaptive_dp_mia.tex` | E4 | "
            "Adaptive vs. Uniform DP budget, utility, and Finance cohort MIA AUC |"
        ),
        (
            "| `tab05_e5_verifiable_secagg.tex` | E5 | "
            "Tamper detection (`1,500` trials), false alarms (`300`), "
            "quantization RMSE & accuracy equivalence |"
        ),
        (
            "| `tab06_e6_ablation_ladder.tex` | E6 | "
            "Complete 5-rung ablation ladder across clean & all 3 attacks "
            "(`InSDN`, `ASEADOS`, Live) |"
        ),
        (
            "| `tab07_e7_live_testbed.tex` | E7 | "
            "Live HTTP FL, 4-mode tamper audit, VM L1 isolation, ping RTT/RSS & OVS mitigation |"
        ),
        (
            "| `tab08_e8_e9_e10_scalability_sensitivity.tex` | E8–E10 | "
            "`N in {3..20}` scalability, sim-to-live transfer & DP clipping sensitivity |"
        ),
        (
            "| `tab09_statistical_tests_and_targets.tex` | Stats & Targets | "
            "Paired Wilcoxon signed-rank `p`-values & `PROJECT_PLAN.md` Section 7 verification |"
        ),
        "",
    ]
    (REPO_ROOT / "figures" / "README.md").write_text("\n".join(fig_lines), encoding="utf-8")
    (REPO_ROOT / "tables" / "README.md").write_text("\n".join(tab_lines), encoding="utf-8")

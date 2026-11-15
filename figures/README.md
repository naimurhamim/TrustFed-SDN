# figures

Publication-ready figures generated automatically by `scripts/_step09_figures_tables.py` via `make reproduce` (never edited by hand). Both vector `.pdf` (for LaTeX/thesis) and `.png` (`200 dpi`, for web/docs) are exported for every figure:

| File Stem | Experiment | Description | Source Artifacts |
|---|---|---|---|
| `fig01_e1_e2_baselines_non_iid.{pdf,png}` | E1 & E2 | Centralized GRU vs. FedAvg vs. Local-Only + Non-IID convergence | `results/02_baselines/`, `results/03_fedavg/` |
| `fig02_e3_poisoning_robustness.{pdf,png}` | E3 | Label-flip, scaling, and backdoor ASR across poison fractions `0–30%` | `results/04_robust/` |
| `fig03_e4_adaptive_dp_mia.{pdf,png}` | E4 | Adaptive vs. Uniform DP privacy–utility curve & Finance cohort MIA AUC | `results/05_dp/` |
| `fig04_e5_verifiable_secagg.{pdf,png}` | E5 | 1,500 tamper detections, 0% false alarms & 48-bit wire overhead | `results/06_verify/` |
| `fig05_e6_ablation_ladder.{pdf,png}` | E6 | 5-rung ablation ladder across clean & 20% poisoned conditions | `results/final/e6_ablation_summary.csv` |
| `fig06_e7_live_sdn_testbed.{pdf,png}` | E7 | Live Mininet/OVS FL convergence, latency stages & OVS mitigation | `results/08_live/` |
| `fig07_e8_scalability.{pdf,png}` | E8 | Scalability across `N in {3, 5, 10, 20}` tenants (utility, latency, wire) | `results/final/e8_scalability.csv`, `e8_overhead_by_n.csv` |
| `fig08_e9_e10_transfer_sensitivity.{pdf,png}` | E9 & E10 | Sim-to-live transfer regimes & sensitivity (`beta`, `C`, `Q`) | `results/07_testbed/`, `results/final/e10_sensitivity.csv` |

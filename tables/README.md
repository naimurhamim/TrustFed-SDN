# tables

LaTeX booktabs tables generated automatically by `scripts/_step09_figures_tables.py` via `make reproduce` strictly from `results/` artifacts:

| File | Experiment(s) | Contents |
|---|---|---|
| `tab01_datasets_and_partitions.tex` | Data (`Step 01`) | Raw/cleaned counts, features, classes across `InSDN`, `ASEADOS-SDN-IoT`, `CICIDS2017` |
| `tab02_e1_e2_baselines_non_iid.tex` | E1 & E2 | Centralized vs. Local-Only vs. FedAvg non-IID Macro-F1 & Accuracy (`mean ± std`) |
| `tab03_e3_poisoning_robustness.tex` | E3 | Poisoning robustness (`label_flip`, `scaling`, `backdoor` F1 & ASR) |
| `tab04_e4_adaptive_dp_mia.tex` | E4 | Adaptive vs. Uniform DP budget, utility, and Finance cohort MIA AUC |
| `tab05_e5_verifiable_secagg.tex` | E5 | Tamper detection (`1,500` trials), false alarms (`300`), quantization RMSE & accuracy equivalence |
| `tab06_e6_ablation_ladder.tex` | E6 | Complete 5-rung ablation ladder across clean & all 3 attacks (`InSDN`, `ASEADOS`, Live) |
| `tab07_e7_live_testbed.tex` | E7 | Live HTTP FL, 4-mode tamper audit, VM L1 isolation, ping RTT/RSS & OVS mitigation |
| `tab08_e8_e9_e10_scalability_sensitivity.tex` | E8–E10 | `N in {3..20}` scalability, sim-to-live transfer & DP clipping sensitivity |
| `tab09_statistical_tests_and_targets.tex` | Stats & Targets | Paired Wilcoxon signed-rank `p`-values & `PROJECT_PLAN.md` Section 7 verification |

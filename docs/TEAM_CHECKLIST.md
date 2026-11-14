# TrustFed-SDN — Team Checklist & Handoff Board (`@Naimur` & `@Pritom`)

> **How this file works:**
> 1. Before starting work, run `git pull` and `.\make.cmd status` (Windows) or `make status` (Linux), or tell the AI agent: *"Check team status and continue."*
> 2. Whenever you finish one or more sub-tasks, mark `[x]`, append `(@YourName, YYYY-MM-DD)`, update the **Latest Handoff Note** below, commit, and `git push`.
> 3. Do not start Step N+1 until the team explicitly approves Step N (`approved step N`).

---

## Latest Handoff Note

- **Last Updated By:** `@Naimur`
- **Date:** `2026-11-14`
- **Active Step:** `Step 07 - Multi-Tenant SDN Testbed` (`waiting for approval`)
- **What Was Just Completed:** Completed Step 07 live Ubuntu 22.04 VMware testbed (`L1`: Mininet `2.3.0` + OVS `2.17.12` + 3 OS-Ken `3.0.1` OpenFlow 1.3 controllers + core isolation switch `s0`) with **`0.0%` intra-tenant loss (`18/18`)** and **`100.0%` cross-tenant block (`6/6`)**, captured **`4,392` labeled bidirectional flows** into `data/live/` + `data/live/MANIFEST.md`, verified `13` new unit tests (`63/63` pytest passing via `make check`), ran 3-seed sim-to-live evaluation (`results/07_testbed/`), and logged Section 8 in `docs/EXPERIMENT_RESULTS_LOG.md`.
- **Next Task for Incoming Teammate:** Review `docs/TESTBED.md`, `results/07_testbed/`, and `docs/EXPERIMENT_RESULTS_LOG.md` Section 8; after explicit `approved step 07`, tag `step-07-done` and start `Step 08 - Live Integration`.

---

## Step-by-Step Sub-Task Checklist

### Step 00 — Environment and User Preferences
- [x] `00.1` Answer all 3 preference batches and record in `docs/USER_PREFERENCES.md` (@Naimur, 2026-10-28)
- [x] `00.2` Create repository skeleton, `pyproject.toml`, `Makefile`, `make.cmd`, `.gitignore`, and `docs/DEVIATIONS.md` (@Naimur, 2026-10-28)
- [x] `00.3` Create project-local Python `3.12.7` environment (`.venv`) and install ML + dev dependencies (@Naimur, 2026-10-28)
- [x] `00.4` Implement `scripts/probe_hardware.py` and record host specs in `results/00_environment/summary.json` & `docs/ENVIRONMENT.md` (@Naimur, 2026-10-28)
- [x] `00.5` Run live Linux SDN testbed smoke test (`Mininet 2.3.0 + OVS 2.17.12 + OS-Ken 3.0.1` OpenFlow 1.3 `pingAll`, `0%` loss) on Ubuntu 22.04 VM and save logs in `docs/smoke_test/` (@Naimur, 2026-10-28)
- [x] `00.6` Document isolated testbed IP ranges (`10.10.0.0/16`) and propose Step 07 isolation level (`L1`) in `docs/ENVIRONMENT.md` (@Naimur, 2026-10-28)
- [x] `00.7` Pass `make check` (`ruff`, `black --check`, `pytest`, `import trustfed`) and set up GitHub team collaboration tools (@Naimur, 2026-10-29)
- [x] `00.8` Explicit user approval (`approved step 00` + `L1` isolation confirmation) and git tag `step-00-done` (@Naimur, 2026-11-05)

### Step 01 — Data (`docs/steps/STEP_01_data.md`)
- [x] `01.1` Answer Step 01 questions (dataset availability, disk budget, download mode, tenant profiles) in `docs/USER_PREFERENCES.md` (@Naimur, 2026-11-05)
- [x] `01.2` Acquire raw datasets (`InSDN` primary, `ASEADOS-SDN-IoT` 2026 SDN, `CICIDS2017` supplementary) into `data/raw/` and generate `data/MANIFEST.md` (size + SHA-256) (@Naimur, 2026-11-05)
- [x] `01.3` Implement dataset loaders (`trustfed/data/load.py`) and profiling script (`scripts/profile_dataset.py`) -> `results/01_data/` (@Naimur, 2026-11-05)
- [x] `01.4` Implement ordered cleaning pipeline (`trustfed/data/clean.py`) and log row counts per rule to `results/01_data/cleaning_log.csv` (@Naimur, 2026-11-05)
- [x] `01.5` Create binary & multi-class label mapping (`configs/data/label_map.yaml`) and stratified train/val/test split before any fitting (@Naimur, 2026-11-05)
- [x] `01.6` Run training-only top-k per-class feature selection and save `configs/data/features_*.yaml` + ranking plot (@Naimur, 2026-11-05)
- [x] `01.7` Implement natural non-IID, Dirichlet (`alpha` in `{0.1, 0.5, 1.0}`), and IID partitions (`trustfed/data/partition.py`) + per-tenant class plots (@Naimur, 2026-11-05)
- [x] `01.8` Implement privacy-preserving per-tenant local scaling (+ global-scaler ablation), save `data/processed/`, pass `tests/test_data.py` & leakage hash checks (@Naimur, 2026-11-05)
- [x] `01.9` Log Step 01 dataset stats to `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 01`, and tag `step-01-done` (@Naimur, 2026-11-05)

### Step 02 — Centralized Baselines & Local Model Decision (`docs/steps/STEP_02_centralized_baselines.md`)
- [x] `02.1` Confirm Step 02 evaluation targets (binary/multi-class) and tuning budget (@Naimur, 2026-11-05)
- [x] `02.2` Implement `DNN`, `GRU`, and `LSTM` in `trustfed/models/` with unified input interface + `trustfed/metrics/` (@Naimur, 2026-11-05)
- [x] `02.3` Train centralized pooled baselines across 5 seeds with equal tuning budget + classical context baselines (LogReg, RandomForest) (@Naimur, 2026-11-05)
- [x] `02.4` Train local-only single-tenant baselines for every tenant across 5 seeds (@Naimur, 2026-11-05)
- [x] `02.5` Record parameter counts, training time, inference latency per flow, and write evidence-based choice to `docs/MODEL_DECISION.md` (@Naimur, 2026-11-05)
- [x] `02.6` Pass `tests/test_models.py`, log tables to `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 02`, and tag `step-02-done` (@Naimur, 2026-11-05)

### Step 03 — FL Simulation Core & FedAvg Baseline (`docs/steps/STEP_03_fl_core_and_fedavg.md`)
- [x] `03.1` Confirm FL hyperparameter defaults (`30` rounds, `3` local epochs, batch `128`, Adam `1e-3`) (@Naimur, 2026-11-05)
- [x] `03.2` Implement `Client`, `Server`, pluggable flat-tensor `Aggregator` interface, and weighted `FedAvg` in `trustfed/fl/` (@Naimur, 2026-11-10)
- [x] `03.3` Build config-driven runner `sim/run_experiment.py` with per-round metrics, serialized byte counts, and wall-clock timing (@Naimur, 2026-11-10)
- [x] `03.4` Run FedAvg across natural non-IID, Dirichlet `{0.1, 0.5, 1.0}`, and IID for `N in {3, 5, 10}` (5 seeds) and plot convergence vs centralized/local-only (@Naimur, 2026-11-10)
- [x] `03.5` Pass `tests/test_fl.py` (determinism + weighted-mean tests), update `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 03`, and tag `step-03-done` (@Naimur, 2026-11-10)

### Step 04 — Poisoning Attacks & Robust Aggregation (`docs/steps/STEP_04_attacks_and_robust_aggregation.md`)
- [x] `04.1` Confirm attack matrix and aggregator baselines (@Naimur, 2026-11-12)
- [x] `04.2` Implement label flipping, feature-trigger backdoor, and update scaling/sign-flip attacks in `trustfed/attacks/` + backdoor ASR metric (@Naimur, 2026-11-12)
- [x] `04.3` Implement coordinate-wise trimmed mean (`beta` `0.1`, `0.2`), cosine-similarity clustering filter, and static-weight baseline in `trustfed/aggregation/` (@Naimur, 2026-11-12)
- [x] `04.4` Run full poisoning sweep (`N=10`, `{0, 10, 20, 30}%` poisoned, 3 attacks, 6 aggregators, 5 seeds) and measure honest-tenant false-rejection rates (@Naimur, 2026-11-12)
- [x] `04.5` Pass `tests/test_attacks.py` & `tests/test_aggregation.py` (`38/38` pytest), log tables to `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 04`, and tag `step-04-done` (@Naimur, 2026-11-12)

### Step 05 — Adaptive Per-Tenant Differential Privacy (`docs/steps/STEP_05_adaptive_dp.md`)
- [x] `05.1` Approve DP design checkpoint (record-level vs update-level granularity, sensitivity score `s_i`, budget policy, `delta`) (@Naimur, 2026-11-12)
- [x] `05.2` Implement L2 clipping, Gaussian mechanism, RDP/moments privacy accountant, uniform DP, and adaptive per-tenant DP in `trustfed/privacy/` (@Naimur, 2026-11-12)
- [x] `05.3` Run epsilon sweep `{0.5, 1, 2, 4, 8, inf}` comparing uniform vs adaptive DP at equal total budget (5 seeds) (@Naimur, 2026-11-12)
- [x] `05.4` Implement and evaluate Membership Inference Attack (MIA accuracy & AUC with and without DP) and write `docs/DP_DESIGN.md` (@Naimur, 2026-11-12)
- [x] `05.5` Pass `tests/test_privacy.py` (`43/43` pytest), log results to `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 05`, and tag `step-05-done` (@Naimur, 2026-11-12)

### Step 06 — Verifiable Secure Aggregation (`docs/steps/STEP_06_verifiable_secure_aggregation.md`)
- [x] `06.1` Approve verification checkpoint (threat model, homomorphic commitment vs keyed linear checksum, dropout policy, fixed-point quantization precision) (@Naimur, 2026-11-13)
- [x] `06.2` Implement fixed-point quantization (`quantize.py`) and exact-canceling pairwise masking (`masking.py`) in `trustfed/verify/` (@Naimur, 2026-11-13)
- [x] `06.3` Implement commitments/checksums, digital signatures, replay nonces, secure aggregator wrapper, and tenant `verify()` (@Naimur, 2026-11-13)
- [x] `06.4` Execute all 4 tamper tests (modified aggregate, dropped tenant, replay, substitution) + 100-round honest false-alarm test + byte/time overhead breakdown (@Naimur, 2026-11-13)
- [x] `06.5` Write `docs/VERIFY_DESIGN.md`, pass `tests/test_verify.py`, update `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 06`, and tag `step-06-done` (@Naimur, 2026-11-13)

### Step 07 — Multi-Tenant SDN Testbed (`docs/steps/STEP_07_sdn_testbed.md`)
- [x] `07.1` Confirm final testbed checkpoint (approved isolation level, 3 tenant subnets, allowed attack tools, live flow feature extractor) (@Naimur, 2026-11-14)
- [x] `07.2` Build multi-tenant Mininet + OVS topology (`testbed/topology/`) and per-tenant OS-Ken apps (`testbed/ryu_apps/`: switching, isolation, flow stats, block API) (@Naimur, 2026-11-14)
- [x] `07.3` Build SD-Traffic-Regulator & flow feature extractor (`testbed/capture/`) verified against Step 01 training feature order/units (@Naimur, 2026-11-14)
- [x] `07.4` Build benign & guarded attack traffic generators (`testbed/traffic/`) enforcing `10.10.0.0/16` target-range guard (@Naimur, 2026-11-14)
- [x] `07.5` Collect labeled live testbed dataset (`data/live/` + `data/live/MANIFEST.md`), measure sim-to-live gap, write `docs/TESTBED.md` (`63/63` pytest; awaiting `approved step 07` to tag `step-07-done`) (@Naimur, 2026-11-14)

### Step 08 — Live Integration on the SDN Testbed (`docs/steps/STEP_08_live_integration.md`)
- [ ] `08.1` Confirm live communication stack (Flower / gRPC), aggregator placement, mitigation expiry policy, and demo scenarios
- [ ] `08.2` Implement live tenant FL clients & secure/robust/verifiable aggregation server (`testbed/fl/`) reusing `trustfed/`
- [ ] `08.3` Implement real-time controller model deployment & automated drop-rule mitigation (`testbed/ryu_apps/model_deploy.py`) outside the OpenFlow event loop
- [ ] `08.4` Run >=10 live rounds + live malicious-tenant filtering demo + live tampered-aggregate rejection demo + latency/CPU/control-plane measurements
- [ ] `08.5` Package `make demo` & `docs/LIVE_PIPELINE.md`, log metrics to `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 08`, and tag `step-08-done`

### Step 09 — Full Evaluation, Ablation, Figures, and Tables (`docs/steps/STEP_09_full_evaluation.md`)
- [ ] `09.1` Confirm final experiment matrix (`E1`–`E8` primary, `E9`–`E10` optional), seed count, and figure styling
- [ ] `09.2` Finalize `configs/experiments/final/` and implement `make reproduce` & `make reproduce-quick`
- [ ] `09.3` Run complete ablation ladder (`FedAvg -> +DP -> +Robust -> +Verifiable -> Full TrustFed-SDN`) and paired Wilcoxon statistical tests
- [ ] `09.4` Generate all publication PDF/PNG figures (`figures/`) and LaTeX tables (`tables/`) strictly from scripts
- [ ] `09.5` Write `docs/RESULTS_SUMMARY.md` & `results/final/key_numbers.json`, update `docs/EXPERIMENT_RESULTS_LOG.md`, get `approved step 09`, and tag `step-09-done`

### Step 10 — Website, QR Code, Documentation, and Release (`docs/steps/STEP_10_website_docs_release.md`)
- [ ] `10.1` Confirm public visibility scope, dataset-license compliance, and open-source code license
- [ ] `10.2` Build responsive project website in `site/` loading all numbers dynamically from `results/final/key_numbers.json`
- [ ] `10.3` Deploy to `https://trustfed.naimurrashid.dev/` and verify high-res print QR code via `scripts/make_qr.py`
- [ ] `10.4` Complete `docs/ARCHITECTURE.md`, `docs/REPRODUCE.md`, `CITATION.cff`, `LICENSE`, and `docs/FINAL_AUDIT.md`
- [ ] `10.5` Verify fresh-clone `make reproduce-quick`, tag `v1.0` & `step-10-done`, and get final `approved step 10`

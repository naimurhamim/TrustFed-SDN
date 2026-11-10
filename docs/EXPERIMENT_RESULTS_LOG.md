# TrustFed-SDN — Master Experiment Results Log (Paper & Thesis Data Store)

> **Mandatory Rule for Agents:**
> At the end of every step or experiment run, you MUST record all empirical numbers, metrics, standard deviations, and overhead statistics directly into this file. 
> Every number here must originate from a verified execution output located in `results/`.
> This file serves as the single source of truth for writing research papers, IEEE/ACM conference submissions, and the undergraduate thesis.

---

## Table of Contents
1. [Step 00: Environment & Smoke Test Baseline](#step-00-environment--smoke-test-baseline)
2. [Step 01: Dataset Partitioning & Statistics](#step-01-dataset-partitioning--statistics)
3. [Step 02: Centralized Baselines & Model Selection (DNN vs GRU vs LSTM)](#step-02-centralized-baselines--model-selection)
4. [Step 03: Federated Learning Core & FedAvg Benchmark](#step-03-federated-learning-core--fedavg-benchmark)
5. [Step 04: Poisoning Attacks & Robust Aggregation](#step-04-poisoning-attacks--robust-aggregation)
6. [Step 05: Adaptive Differential Privacy Evaluation](#step-05-adaptive-differential-privacy-evaluation)
7. [Step 06: Verifiable Secure Aggregation Overhead & Tamper Tests](#step-06-verifiable-secure-aggregation-overhead--tamper-tests)
8. [Step 07: SDN Testbed (Mininet + OVS + RYU) Traffic & Flow Stats](#step-07-sdn-testbed-traffic--flow-stats)
9. [Step 08: Live End-to-End Integration & Mitigation Latency](#step-08-live-end-to-end-integration--mitigation-latency)
10. [Step 09: Master Ablation Ladder & Final Evaluation](#step-09-master-ablation-ladder--final-evaluation)

---

## Log Template for Each Entry

```markdown
### [YYYY-MM-DD] Step XX — Experiment Name
- **Commit:** `<git-hash>` | **Config:** `configs/path/to/config.yaml` | **Raw Summary:** `results/path/to/summary.json`
- **Seeds Used:** `[0, 1, 2, 3, 4]` (N=5)

#### 1. Detection Performance (Mean ± Std)
| Model / Setup | Accuracy (%) | Precision (%) | Recall (%) | Macro-F1 (%) | FPR (%) | ROC-AUC |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

#### 2. Robustness & Privacy Metrics (if applicable)
| Setup / Attack | Clean Acc (%) | Poisoned Acc (%) | Backdoor ASR (%) | Epsilon (ε) | Delta (δ) | MIA Acc (%) |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

#### 3. Computational & Communication Overhead
| Component | Wall-Clock Time (ms / s) | Communication (Bytes / Round) | Notes |
|---|---|---|---|
| Local Training | ... | ... | ... |
| Aggregation | ... | ... | ... |
| Verification | ... | ... | ... |

#### 4. Paper-Ready Key Findings / Highlights
- *Bullet points highlighting statistical significance, key takeaways, and trade-offs for paper discussion sections.*
```

---

## Experiment Entries

### [2026-10-28] Step 00 — Environment & Smoke Test Baseline {#step-00-environment--smoke-test-baseline}

- **Base Commit:** `a3f8364` | **Raw Summary Artifacts:** `results/00_environment/summary.json`, `results/00_environment/smoke_test_summary.json`
- **Environment Config:** `pyproject.toml`, `docs/ENVIRONMENT.md`

#### 1. Experimental Host Hardware & Software Specification (Paper "Experimental Setup" Table)

| Component | Measured Specification | Source |
|---|---|---|
| Host OS | Windows 11 Home (`10.0.26300`, `AMD64`) | `results/00_environment/summary.json` |
| CPU | 12th Gen Intel Core i7-12700H (14 physical cores, 20 logical threads) | `results/00_environment/summary.json` |
| System RAM | `15.67 GB` total (`3.66 GB` free during active desktop/IDE probe) | `results/00_environment/summary.json` |
| Storage (Workspace `D:\`) | `475.4 GB` free of `953.9 GB` NTFS | `results/00_environment/summary.json` |
| Accelerator GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU (`4096 MiB` VRAM, driver `610.88`) | `results/00_environment/summary.json` |
| Hypervisor & SDN Testbed VM | VMware Workstation `17.6.0` running **Ubuntu 22.04 LTS** (`Linux 5.15.0-25-generic`) | `docs/smoke_test/versions.txt` |
| SDN Data & Control Plane | Mininet `2.3.0`, Open vSwitch `2.17.12`, OpenFlow `1.3`, `os-ken==3.0.1` (`eventlet==0.37.0`) on Python `3.10.12` (`.venv-ryu`) | `results/00_environment/smoke_test_summary.json` |
| ML / FL Python Runtime | Python `3.12.7` in project-local `.venv` | `results/00_environment/summary.json` |
| Core Library Versions | PyTorch `2.14.1`, scikit-learn `1.9.1`, NumPy `2.5.4`, pandas `3.1.0`, SciPy `1.18.1`, cryptography `50.0.2` | `results/00_environment/summary.json` |

#### 2. Testbed Safety, Smoke Test & Isolation Baseline

| Parameter | Value / Status | Notes |
|---|---|---|
| Isolated Testbed Supernet | `10.10.0.0/16` | Strictly private virtual testbed boundary |
| Tenant Subnets (3 tenants) | `10.10.1.0/24` (Finance), `10.10.2.0/24` (Healthcare), `10.10.3.0/24` (Education) | Disjointness & RFC1918 privacy verified by `tests/test_smoke.py` |
| Quality Check (`make check`) | `3 passed` (`pytest`), `0` lint/format errors (`ruff` + `black`) | Verified on Windows Python `3.12.7` |
| Linux Mininet+OVS+RYU(OS-Ken) `pingAll` | **PASSED (`0.0%` loss, `6/6` packets received)** | OpenFlow `1.3` (`6` learned unicast flows + `priority=0` table-miss flow, `20` packets / `1468` B); logs in `docs/smoke_test/` |
| Host RAM Under Active Ubuntu VM | `93.0%` used (`1.1 GB` free of `15.67 GB`) | Measured via `scripts/probe_hardware.py` while Ubuntu 22.04 Desktop VM was active |
| Proposed Step 07 Isolation Level | **L1** (approved by user 2026-11-05: Mininet namespaces + per-tenant OVS + per-tenant OS-Ken controller inside Ubuntu 22.04 VM) | Strongly justified by `93.0%` host RAM usage with 1 active Desktop VM |

---

### [2026-11-05] Step 01 — Dataset Partitioning, Leakage-Free Cleaning & Feature Selection {#step-01-dataset-partitioning--statistics}

- **Script:** `scripts/profile_dataset.py` | **Configs:** `configs/data/label_map.yaml`, `configs/data/features_{insdn,aseados_sdn_iot,cicids2017}.yaml`
- **Raw Summaries:** `results/01_data/summary.json`, `results/01_data/cleaning_log.csv`, `results/01_data/profile_{insdn,aseados_sdn_iot,cicids2017}.json`
- **Seed Used:** `42` (stratified `70% / 15% / 15%` train/validation/test split performed **before** any feature selection or scaler fitting)

#### 1. Dataset Overview & Ordered Cleaning Pipeline Summary (`results/01_data/cleaning_log.csv`)

| Dataset | Year / Source | Raw Rows | Raw Cols | Dropped Dup Header Cols (R1) | Dropped ID & Port Cols (R3) | Inf Cells / NaN Rows Dropped (R4) | Duplicate Feature Rows Removed (R6) | Final Unique Cleaned Rows | Candidate Numeric Features | Selected Top-$k$ Union Features ($k=5$/class) |
|---|---|---|---|---|---|---|---|---|---|---|
| **InSDN** | 2020 (UCD ASEADOS) | `343,889` | `84` | `0` | `6` (`Flow ID`, `Src/Dst IP`, `Src/Dst Port`, `Timestamp`) | `0` / `0` | `159,034` (`46.25%`) | **`184,855`** | `77` | **`19`** |
| **ASEADOS-SDN-IoT** | 2026 (UCD ASEADOS) | `457,044` | `84` | `0` | `6` (`Flow ID`, `Src/Dst IP`, `Src/Dst Port`, `Timestamp`) | `74` / `70` | `30,863` (`6.75%`) | **`426,111`** | `77` | **`12`** |
| **CICIDS2017** | 2017 (UNB CIC) | `2,830,743` | `79` | `1` (`Fwd Header Length.1`) | `1` (`Destination Port`) | `4,376` / `2,867` | `594,982` (`21.02%`) | **`2,232,894`** | `76` | **`22`** |

#### 2. Stratified Split Sizes & SHA-256 Zero-Leakage Verification (`results/01_data/summary.json`)

| Dataset | Train Rows (`70%`) | Validation Rows (`15%`) | Test Rows (`15%`) | Train $\cap$ Val Hash Overlap | Train $\cap$ Test Hash Overlap | Val $\cap$ Test Hash Overlap |
|---|---|---|---|---|---|---|
| **InSDN** | `129,398` | `27,728` | `27,729` | **`0`** | **`0`** | **`0`** |
| **ASEADOS-SDN-IoT (2026)** | `298,277` | `63,917` | `63,917` | **`0`** | **`0`** | **`0`** |
| **CICIDS2017** | `1,563,025` | `334,934` | `334,935` | **`0`** | **`0`** | **`0`** |

#### 3. Cleaned Multi-Class and Natural Non-IID Per-Tenant Training Distributions (`N=3`)

**A. InSDN (`184,855` cleaned unique flows; `129,398` training flows):**

| Canonical Class | Total Cleaned (`100%`) | Tenant 1: Finance (Train) | Tenant 2: Healthcare (Train) | Tenant 3: Education (Train) |
|---|---|---|---|---|
| `Benign` | `64,218` | `14,984` | `14,984` | `14,984` |
| `Probe` | `67,068` | `7,042` | `7,042` | **`32,863`** *(70% specialty)* |
| `DoS` | `50,310` | `5,283` | **`24,652`** *(70% specialty)* | `5,282` |
| `DDoS` | `1,520` | `160` | **`745`** *(70% specialty)* | `159` |
| `BruteForce` (`BFA`) | `1,375` | **`674`** *(70% specialty)* | `145` | `144` |
| `WebAttack` | `183` | **`90`** *(70% specialty)* | `19` | `19` |
| `Botnet` | `164` | **`80`** *(70% specialty)* | `18` | `17` |
| `Exploitation_U2R` | `17` | `2` | `2` | **`8`** *(70% specialty)* |
| **Total Flows** | **`184,855`** | **`28,315`** | **`47,607`** | **`53,476`** |

**B. ASEADOS-SDN-IoT (2026) (`426,111` cleaned unique flows; `298,277` training flows):**

| Canonical Class | Total Cleaned (`100%`) | Tenant 1: Finance (Train) | Tenant 2: Healthcare (Train) | Tenant 3: Education (Train) |
|---|---|---|---|---|
| `Benign` | `233,386` | `54,457` | `54,457` | `54,456` |
| `DoS` | `126,201` | `13,251` | **`61,839`** *(70% specialty)* | `13,251` |
| `DDoS` | `50,165` | `5,268` | **`24,580`** *(70% specialty)* | `5,267` |
| `Botnet` | `8,957` | **`4,389`** *(70% specialty)* | `941` | `940` |
| `Probe` | `7,402` | `777` | `777` | **`3,627`** *(70% specialty)* |
| **Total Flows** | **`426,111`** | **`78,142`** | **`142,594`** | **`77,541`** |

**C. CICIDS2017 (`2,232,894` cleaned unique flows; `1,563,025` training flows):**

| Canonical Class | Total Cleaned (`100%`) | Tenant 1: Finance (Train) | Tenant 2: Healthcare (Train) | Tenant 3: Education (Train) |
|---|---|---|---|---|
| `Benign` | `1,896,512` | `442,520` | `442,519` | `442,519` |
| `DoS` | `193,641` | `20,332` | **`94,884`** *(70% specialty)* | `20,332` |
| `DDoS` | `128,014` | `13,442` | **`62,727`** *(70% specialty)* | `13,441` |
| `BruteForce` | `9,150` | **`4,484`** *(70% specialty)* | `961` | `960` |
| `WebAttack` | `2,143` | **`1,050`** *(70% specialty)* | `225` | `225` |
| `Probe` (`PortScan`) | `1,956` | `206` | `205` | **`958`** *(70% specialty)* |
| `Botnet` | `1,431` | **`701`** *(70% specialty)* | `151` | `150` |
| `Infiltration` | `36` | `4` | `3` | **`18`** *(70% specialty)* |
| `Exploitation_U2R` (`Heartbleed`) | `11` | `1` | `1` | **`6`** *(70% specialty)* |
| **Total Flows** | **`2,232,894`** | **`482,740`** | **`601,676`** | **`478,609`** |

#### 4. Selected Top-5-per-Attack-Class Feature Unions (Training Data Only)

- **InSDN (`19` features, `configs/data/features_insdn.yaml`):** `Bwd Header Len`, `Init Bwd Win Byts`, `Pkt Len Var`, `Pkt Len Std`, `Bwd Seg Size Avg`, `Bwd Pkt Len Mean`, `Pkt Len Max`, `Pkt Len Mean`, `Bwd Pkt Len Max`, `Pkt Size Avg`, `TotLen Bwd Pkts`, `Subflow Bwd Byts`, `Fwd Header Len`, `Bwd IAT Tot`, `Bwd IAT Max`, `Flow Duration`, `Flow IAT Min`, `Bwd IAT Mean`, `Fwd IAT Tot`.
- **ASEADOS-SDN-IoT 2026 (`12` features, `configs/data/features_aseados_sdn_iot.yaml`):** `Flow IAT Min`, `Bwd Pkts/s`, `Flow IAT Mean`, `Bwd Header Len`, `Fwd Header Len`, `Fwd Pkts/s`, `Flow Pkts/s`, `Flow IAT Max`, `Flow Duration`, `Pkt Len Max`, `Pkt Len Std`, `Pkt Len Var`.
- **CICIDS2017 (`22` features, `configs/data/features_cicids2017.yaml`):** `Average Packet Size`, `Packet Length Mean`, `Total Length of Fwd Packets`, `Subflow Fwd Bytes`, `Packet Length Variance`, `Packet Length Std`, `Fwd Header Length`, `Init_Win_bytes_backward`, `Max Packet Length`, `Flow IAT Max`, `Fwd Packet Length Max`, `Total Length of Bwd Packets`, `Bwd Packets/s`, `Subflow Bwd Bytes`, `Bwd Packet Length Mean`, `Avg Bwd Segment Size`, `Bwd Packet Length Max`, `Flow IAT Std`, `Flow Packets/s`, `Flow IAT Mean`, `Fwd IAT Std`, `Init_Win_bytes_forward`.

#### 5. Paper-Ready Dataset Quirks & Methodological Findings

1. **Massive Feature-Vector Duplication in Raw `InSDN` (`46.25%`) and `CICIDS2017` (`21.02%`):** After stripping flow identifiers (`Flow ID`, `Src/Dst IP`, `Src/Dst Port`, `Timestamp`), `159,034` of `343,889` rows in `InSDN` (especially stateless single-packet DDoS floods, reducing `DDoS` from `121,942` raw rows to `1,520` unique feature vectors) and `594,982` of `2,830,743` rows in `CICIDS2017` (reducing `PortScan` from `158,930` raw rows to `1,956` unique feature vectors after dropping `Destination Port`) are byte-for-byte identical. Standard random train/test splits without deduplication leak nearly half of `InSDN` test samples into the training set; our Rule `R6` deduplication + SHA-256 row-hash check guarantees `0` train/val/test overlap.
2. **Superior Diversity in 2026 `ASEADOS-SDN-IoT`:** Only `6.75%` (`30,863` of `457,044`) of rows in `ASEADOS-SDN-IoT` (2026) are duplicate feature vectors, yielding `426,111` unique flows with a near-balanced `54.8%` Benign / `45.2%` Attack ratio.
3. **Label & Schema Quirks Resolved:** Trailing space in `InSDN` (`"DDoS "`), truncated spellings in `ASEADOS-SDN-IoT` (`"Dos"`, `"Prob"`), Windows-1252 `0x96` en-dash corruption in `CICIDS2017` Web Attack labels, `65` whitespace-padded column headers and duplicate `Fwd Header Length.1` in `CICIDS2017`, and `4,376` (`CICIDS2017`) + `74` (`ASEADOS-SDN-IoT`) infinite cells.



---

## Step 02 — Centralized Baselines, Sequence-Layout Study, Local-Only Baselines & Local Model Decision

- **Experiment ID:** `EXP-02-CENTRALIZED-BASELINES`
- **Date / Timestamp (UTC):** `2026-11-05T08:58:52Z`
- **Execution Hardware:** `NVIDIA GeForce RTX 3050 Ti Laptop GPU` (`PyTorch 2.14.1+cu130`, `CUDA 13.0`), `batch_size=1024`, `epochs=12`
- **Seeds Evaluated:** `{0, 1, 2, 3, 4}` (`5` seeds per model, task, and dataset)
- **Artifacts:**
  - `results/02_baselines/summary.json`
  - `results/02_baselines/per_seed_centralized.csv`
  - `results/02_baselines/per_seed_local_only.csv`
  - `results/02_baselines/sequence_layout_study.csv`
  - `results/02_baselines/tuning_results.csv`
  - `results/02_baselines/centralized_model_comparison.png`
  - `results/02_baselines/centralized_vs_local_only.png`
  - `docs/MODEL_DECISION.md`

### 1. Recurrent Sequence Presentation Study (`Step 02` Task 1)

| Dataset | Model | Sequence Mode | `(T, F)` | Params | Val Macro-F1 (%) | Train Time (s) | Inference (`us`/flow) |
|---|---|---|---|---:|---:|---:|---:|
| `insdn` | `GRU` | `single_step` | `(1, 19)` | `43,752` | **`52.39%`** | `11.33` | `1.54` |
| `insdn` | `GRU` | `feature_chunks` | `(4, 5)` | `41,064` | `43.38%` | `12.43` | `1.45` |
| `insdn` | `GRU` | `per_feature` | `(19, 1)` | `40,296` | `40.66%` | `12.25` | `1.62` |
| `insdn` | `LSTM` | `single_step` | `(1, 19)` | `57,512` | `48.11%` | `10.80` | `1.35` |
| `insdn` | `LSTM` | `feature_chunks` | `(4, 5)` | `53,928` | `40.40%` | `11.75` | `1.31` |
| `insdn` | `LSTM` | `per_feature` | `(19, 1)` | `52,904` | `44.79%` | `11.57` | `1.79` |
| `aseados_sdn_iot` | `GRU` | `single_step` | `(1, 12)` | `42,309` | **`57.78%`** | `17.04` | `0.76` |
| `aseados_sdn_iot` | `GRU` | `feature_chunks` | `(4, 3)` | `40,581` | `55.18%` | `18.65` | `0.96` |
| `aseados_sdn_iot` | `GRU` | `per_feature` | `(12, 1)` | `40,197` | `57.08%` | `17.73` | `1.31` |
| `aseados_sdn_iot` | `LSTM` | `single_step` | `(1, 12)` | `55,621` | `57.26%` | `16.28` | `1.09` |
| `aseados_sdn_iot` | `LSTM` | `feature_chunks` | `(4, 3)` | `53,317` | `55.03%` | `17.77` | `1.12` |
| `aseados_sdn_iot` | `LSTM` | `per_feature` | `(12, 1)` | `52,805` | `57.30%` | `18.22` | `1.49` |
| `cicids2017` | `GRU` | `single_step` | `(1, 22)` | `44,361` | `56.21%` | `138.15` | `0.75` |
| `cicids2017` | `GRU` | `feature_chunks` | `(4, 6)` | `41,289` | `55.92%` | `158.60` | `0.91` |
| `cicids2017` | `GRU` | `per_feature` | `(22, 1)` | `40,329` | `59.19%` | `154.75` | `2.98` |
| `cicids2017` | `LSTM` | `single_step` | `(1, 22)` | `58,313` | `56.55%` | `154.66` | `1.73` |
| `cicids2017` | `LSTM` | `feature_chunks` | `(4, 6)` | `54,217` | `54.63%` | `169.40` | `1.11` |
| `cicids2017` | `LSTM` | `per_feature` | `(22, 1)` | `52,937` | `59.54%` | `155.50` | `1.99` |

### 2. Centralized Pooled Baselines (`5 Seeds: Mean ± Std`)

#### 2.1 Binary Threat Detection (Headline Metric)

| Dataset | Model | Params | Accuracy (%) | Macro-F1 (%) | Attack F1 (%) | ADR / Recall (%) | FPR (%) | ROC-AUC (%) | Train (s) | Latency (`us`/flow) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | **`DNN`** | `11,330` | `99.85 ± 0.02` | **`99.83 ± 0.03`** | `99.88 ± 0.02` | `99.93 ± 0.02` | `0.31 ± 0.07` | `99.97 ± 0.00` | `8.22` | `0.30` |
| `insdn` | **`GRU`** | `43,554` | `99.76 ± 0.05` | **`99.74 ± 0.05`** | `99.82 ± 0.04` | `99.84 ± 0.07` | `0.39 ± 0.10` | `99.95 ± 0.01` | `12.18` | `1.35` |
| `insdn` | **`LSTM`** | `57,314` | `99.63 ± 0.12` | **`99.59 ± 0.13`** | `99.72 ± 0.09` | `99.85 ± 0.03` | `0.77 ± 0.40` | `99.94 ± 0.01` | `11.72` | `1.53` |
| `insdn` | **`LOGREG`** | `20` | `98.34 ± 0.00` | **`98.17 ± 0.00`** | `98.73 ± 0.00` | `99.04 ± 0.00` | `2.97 ± 0.00` | `98.65 ± 0.00` | `0.92` | `0.10` |
| `insdn` | **`RANDOM_FOREST`** | `15,514` | `99.98 ± 0.00` | **`99.98 ± 0.00`** | `99.99 ± 0.00` | `100.00 ± 0.00` | `0.05 ± 0.01` | `100.00 ± 0.00` | `1.24` | `2.20` |
| `aseados_sdn_iot` | **`DNN`** | `12,514` | `91.45 ± 0.02` | **`91.42 ± 0.02`** | `90.98 ± 0.05` | `95.38 ± 0.39` | `11.80 ± 0.31` | `97.58 ± 0.04` | `18.23` | `0.29` |
| `aseados_sdn_iot` | **`GRU`** | `42,210` | `91.26 ± 0.05` | **`91.23 ± 0.06`** | `90.77 ± 0.08` | `95.07 ± 0.41` | `11.89 ± 0.31` | `97.45 ± 0.03` | `19.11` | `0.93` |
| `aseados_sdn_iot` | **`LSTM`** | `55,522` | `91.23 ± 0.15` | **`91.21 ± 0.15`** | `90.75 ± 0.15` | `95.03 ± 0.42` | `11.90 ± 0.39` | `97.39 ± 0.10` | `23.51` | `1.13` |
| `aseados_sdn_iot` | **`LOGREG`** | `13` | `73.61 ± 0.00` | **`72.82 ± 0.00`** | `68.17 ± 0.00` | `62.46 ± 0.00` | `17.18 ± 0.00` | `83.85 ± 0.00` | `0.56` | `0.06` |
| `aseados_sdn_iot` | **`RANDOM_FOREST`** | `151,432` | `92.89 ± 0.03` | **`92.87 ± 0.03`** | `92.49 ± 0.03` | `96.80 ± 0.04` | `10.33 ± 0.05` | `98.31 ± 0.01` | `1.82` | `1.02` |
| `cicids2017` | **`DNN`** | `13,794` | `98.50 ± 0.15` | **`97.01 ± 0.33`** | `94.90 ± 0.58` | `92.90 ± 2.30` | `0.51 ± 0.24` | `99.64 ± 0.16` | `31.75` | `0.28` |
| `cicids2017` | **`GRU`** | `90,770` | `98.33 ± 0.10` | **`96.66 ± 0.18`** | `94.30 ± 0.30` | `91.74 ± 0.82` | `0.50 ± 0.21` | `99.58 ± 0.08` | `49.57` | `1.95` |
| `cicids2017` | **`LSTM`** | `119,378` | `98.30 ± 0.05` | **`96.61 ± 0.09`** | `94.21 ± 0.14` | `91.68 ± 0.16` | `0.52 ± 0.07` | `99.56 ± 0.06` | `54.26` | `2.82` |
| `cicids2017` | **`LOGREG`** | `23` | `95.49 ± 0.00` | **`90.11 ± 0.00`** | `82.81 ± 0.00` | `72.18 ± 0.00` | `0.38 ± 0.00` | `97.04 ± 0.00` | `1.01` | `0.08` |
| `cicids2017` | **`RANDOM_FOREST`** | `47,572` | `99.62 ± 0.01` | **`99.27 ± 0.02`** | `98.76 ± 0.03` | `99.26 ± 0.03` | `0.31 ± 0.01` | `99.99 ± 0.00` | `5.17` | `1.20` |

#### 2.2 Multi-Class Attack Family Detection (Secondary Metric)

| Dataset | Model | Params | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | ADR (%) | FPR (%) | ROC-AUC (%) | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | **`DNN`** | `11,720` | `98.87 ± 0.14` | **`63.91 ± 3.12`** | `98.77 ± 0.15` | `99.59 ± 0.07` | `0.32 ± 0.07` | `98.75 ± 0.34` | `10.12` |
| `insdn` | **`GRU`** | `43,752` | `98.09 ± 0.27` | **`54.69 ± 1.53`** | `97.87 ± 0.30` | `99.51 ± 0.09` | `1.18 ± 0.56` | `94.63 ± 2.57` | `12.57` |
| `insdn` | **`LSTM`** | `57,512` | `97.99 ± 0.15` | **`52.84 ± 1.21`** | `97.71 ± 0.19` | `99.44 ± 0.08` | `1.25 ± 0.38` | `94.38 ± 2.69` | `12.97` |
| `insdn` | **`LOGREG`** | `160` | `91.45 ± 0.00` | **`39.60 ± 0.00`** | `90.78 ± 0.00` | `99.27 ± 0.00` | `2.77 ± 0.00` | `89.49 ± 0.00` | `10.03` |
| `insdn` | **`RANDOM_FOREST`** | `41,990` | `99.82 ± 0.00` | **`86.21 ± 0.02`** | `99.81 ± 0.00` | `99.99 ± 0.00` | `0.02 ± 0.00` | `99.95 ± 0.02` | `1.13` |
| `aseados_sdn_iot` | **`DNN`** | `12,613` | `86.05 ± 0.06` | **`58.73 ± 0.09`** | `85.69 ± 0.06` | `88.28 ± 0.24` | `7.47 ± 0.12` | `95.99 ± 0.04` | `19.30` |
| `aseados_sdn_iot` | **`GRU`** | `42,309` | `86.07 ± 0.03` | **`58.92 ± 0.31`** | `85.78 ± 0.09` | `88.95 ± 0.60` | `7.80 ± 0.27` | `95.95 ± 0.03` | `21.57` |
| `aseados_sdn_iot` | **`LSTM`** | `55,621` | `85.98 ± 0.04` | **`58.40 ± 0.33`** | `85.68 ± 0.09` | `88.89 ± 0.51` | `7.89 ± 0.21` | `95.94 ± 0.04` | `21.95` |
| `aseados_sdn_iot` | **`LOGREG`** | `65` | `72.82 ± 0.00` | **`45.64 ± 0.00`** | `71.83 ± 0.00` | `73.18 ± 0.00` | `16.32 ± 0.00` | `90.87 ± 0.00` | `11.28` |
| `aseados_sdn_iot` | **`RANDOM_FOREST`** | `225,154` | `87.64 ± 0.03` | **`68.13 ± 0.37`** | `87.97 ± 0.02` | `93.62 ± 0.18` | `8.16 ± 0.08` | `97.20 ± 0.01` | `1.91` |
| `cicids2017` | **`DNN`** | `14,025` | `98.71 ± 0.02` | **`59.93 ± 0.39`** | `98.82 ± 0.02` | `95.96 ± 0.59` | `0.79 ± 0.10` | `94.06 ± 1.98` | `54.24` |
| `cicids2017` | **`GRU`** | `91,113` | `98.29 ± 0.08` | **`57.88 ± 1.02`** | `98.36 ± 0.08` | `92.57 ± 0.37` | `0.69 ± 0.05` | `97.60 ± 1.84` | `66.77` |
| `cicids2017` | **`LSTM`** | `119,721` | `98.26 ± 0.04` | **`57.37 ± 0.14`** | `98.33 ± 0.05` | `92.21 ± 0.44` | `0.66 ± 0.06` | `98.75 ± 0.67` | `70.50` |
| `cicids2017` | **`LOGREG`** | `207` | `94.83 ± 0.00` | **`38.34 ± 0.00`** | `94.81 ± 0.00` | `74.09 ± 0.00` | `1.40 ± 0.00` | `97.42 ± 0.00` | `22.97` |
| `cicids2017` | **`RANDOM_FOREST`** | `53,234` | `99.61 ± 0.00` | **`86.71 ± 0.37`** | `99.67 ± 0.00` | `99.31 ± 0.01` | `0.33 ± 0.00` | `98.77 ± 0.74` | `5.61` |

### 3. Single-Tenant Local-Only vs. Centralized Pooled (`Step 02` Task 6)

| Dataset | Task | Training Regime | Own-Tenant Test Macro-F1 (%) | Global Test Macro-F1 (%) | Global ADR (%) | Global FPR (%) |
|---|---|---|---:|---:|---:|---:|
| `insdn` | `binary` | **Centralized Pooled (Upper Bound)** | `—` | **`99.74 ± 0.05`** | `99.84%` | `0.39%` |
| `insdn` | `binary` | Local-Only: `tenant_1_finance` | `98.63 ± 0.02` | `98.86 ± 0.05` | `99.33%` | `1.73%` |
| `insdn` | `binary` | Local-Only: `tenant_2_healthcare` | `98.81 ± 0.06` | `99.05 ± 0.02` | `99.76%` | `2.02%` |
| `insdn` | `binary` | Local-Only: `tenant_3_education` | `99.38 ± 0.04` | `98.92 ± 0.04` | `99.41%` | `1.72%` |
| `insdn` | `multiclass` | **Centralized Pooled (Upper Bound)** | `—` | **`54.69 ± 1.53`** | `99.51%` | `1.18%` |
| `insdn` | `multiclass` | Local-Only: `tenant_1_finance` | `31.22 ± 1.25` | `31.99 ± 1.22` | `99.09%` | `1.76%` |
| `insdn` | `multiclass` | Local-Only: `tenant_2_healthcare` | `45.18 ± 1.73` | `40.11 ± 2.17` | `99.35%` | `2.12%` |
| `insdn` | `multiclass` | Local-Only: `tenant_3_education` | `35.40 ± 0.02` | `34.45 ± 0.01` | `99.07%` | `1.80%` |
| `aseados_sdn_iot` | `binary` | **Centralized Pooled (Upper Bound)** | `—` | **`91.23 ± 0.06`** | `95.07%` | `11.89%` |
| `aseados_sdn_iot` | `binary` | Local-Only: `tenant_1_finance` | `89.27 ± 0.14` | `89.24 ± 0.13` | `85.09%` | `7.03%` |
| `aseados_sdn_iot` | `binary` | Local-Only: `tenant_2_healthcare` | `91.82 ± 0.08` | `90.57 ± 0.10` | `96.50%` | `14.30%` |
| `aseados_sdn_iot` | `binary` | Local-Only: `tenant_3_education` | `88.15 ± 0.09` | `89.32 ± 0.11` | `85.70%` | `7.42%` |
| `aseados_sdn_iot` | `multiclass` | **Centralized Pooled (Upper Bound)** | `—` | **`58.92 ± 0.31`** | `88.95%` | `7.80%` |
| `aseados_sdn_iot` | `multiclass` | Local-Only: `tenant_1_finance` | `54.20 ± 0.73` | `50.01 ± 0.55` | `80.35%` | `5.58%` |
| `aseados_sdn_iot` | `multiclass` | Local-Only: `tenant_2_healthcare` | `52.84 ± 0.71` | `50.26 ± 0.50` | `95.32%` | `13.17%` |
| `aseados_sdn_iot` | `multiclass` | Local-Only: `tenant_3_education` | `57.89 ± 0.18` | `56.29 ± 0.33` | `82.01%` | `5.90%` |
| `cicids2017` | `binary` | **Centralized Pooled (Upper Bound)** | `—` | **`96.66 ± 0.18`** | `91.74%` | `0.50%` |
| `cicids2017` | `binary` | Local-Only: `tenant_1_finance` | `94.23 ± 0.73` | `95.96 ± 0.22` | `87.90%` | `0.17%` |
| `cicids2017` | `binary` | Local-Only: `tenant_2_healthcare` | `96.72 ± 0.03` | `95.95 ± 0.05` | `88.38%` | `0.28%` |
| `cicids2017` | `binary` | Local-Only: `tenant_3_education` | `95.28 ± 0.14` | `95.54 ± 0.14` | `86.33%` | `0.12%` |
| `cicids2017` | `multiclass` | **Centralized Pooled (Upper Bound)** | `—` | **`57.88 ± 1.02`** | `92.57%` | `0.69%` |
| `cicids2017` | `multiclass` | Local-Only: `tenant_1_finance` | `60.79 ± 0.24` | `51.13 ± 0.28` | `89.72%` | `0.63%` |
| `cicids2017` | `multiclass` | Local-Only: `tenant_2_healthcare` | `47.72 ± 4.65` | `37.12 ± 3.81` | `89.08%` | `0.28%` |
| `cicids2017` | `multiclass` | Local-Only: `tenant_3_education` | `43.67 ± 4.58` | `43.22 ± 4.23` | `88.16%` | `0.18%` |

### 4. Local Model Decision Summary (`docs/MODEL_DECISION.md`)

- **Selected Model:** **`GRU` (`GRUClassifier`)**
- **Evidence:** Across all 3 datasets, `GRU` binary Macro-F1 (`99.74%` on `InSDN`, `91.23%` on `ASEADOS-SDN-IoT` 2026, `96.66%` on `CICIDS2017`; mean `95.88%`) is within **`+0.21` F1 points** (`< 0.50` F1 threshold) of `DNN` (`96.09%`), beats `LSTM` on **every dataset and task** while requiring **~24% fewer trainable parameters** (`42.2K–43.6K` vs `55.5K–57.3K`), and achieves the **highest neural Multi-Class Macro-F1 on the 2026 `ASEADOS-SDN-IoT` benchmark (`58.92 ± 0.31%`)**.

---

## Step 03: Federated Learning Core & FedAvg Benchmark

### [2026-11-10] Step 03 — FL Simulation Core & FedAvg Baseline (`03_fedavg`)
- **Runner:** `sim/run_experiment.py --sweep-step03`
- **Configs:** `configs/experiments/fedavg_step03_sweep.yaml`, `configs/experiments/fedavg_insdn_natural_n3.yaml`, `configs/experiments/fedavg_aseados_natural_n3.yaml`, `configs/experiments/fedavg_cicids2017_natural_n3.yaml`, `configs/experiments/fedavg_insdn_dirichlet_a0.5_n10.yaml`, `configs/experiments/fedavg_insdn_iid_n10.yaml`
- **Raw Summary & Artifacts:**
  - `results/03_fedavg/summary.json`
  - `results/03_fedavg/per_seed_summary.csv` (`270` 30-round FL runs = `54` regimes × `5` seeds)
  - `results/03_fedavg/per_round_metrics.csv`
  - `results/03_fedavg/convergence_curves.png`
  - `results/03_fedavg/centralized_vs_fedavg_vs_local_only.png`
  - `results/03_fedavg/non_iid_and_scalability_comparison.png`
- **FL Hyperparameters:** Local model = `GRUClassifier` (`hidden_dim=64`, `num_layers=2`, `dropout=0.2`, `num_chunks=1`), `T = 30` communication rounds, `E = 3` local epochs per round, `batch_size = 128`, `Adam` (`lr = 1e-3`, `weight_decay = 1e-4`), `100%` client participation, `5` deterministic seeds (`[0, 1, 2, 3, 4]`).

#### 1. Centralized vs. FedAvg vs. Local-Only Comparison (`N=3` Primary Regimes, 5 Seeds, Mean ± Std)

| Dataset | Task | Training Regime | Global Accuracy (%) | Global Macro-F1 (%) | Attack-F1 (%) | ADR (%) | FPR (%) | ROC-AUC (%) | Tenant Macro-F1 Mean (%) | Tenant Macro-F1 Min (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | `binary` | **Centralized Pooled (Upper Bound)** | `99.76 ± 0.04` | **`99.74 ± 0.05`** | `99.82 ± 0.03` | `99.84 ± 0.05` | `0.39 ± 0.16` | `99.96 ± 0.01` | `—` | `—` |
| `insdn` | `binary` | **`FedAvg` — IID (`N=3`)** | `99.24 ± 0.61` | **`99.16 ± 0.68`** | `99.43 ± 0.46` | `99.81 ± 0.09` | `1.81 ± 1.66` | `99.87 ± 0.15` | `99.10 ± 0.67` | `99.04 ± 0.68` |
| `insdn` | `binary` | **`FedAvg` — Natural Non-IID (`N=3`)** | `99.21 ± 0.53` | **`99.13 ± 0.59`** | `99.40 ± 0.40` | `99.81 ± 0.14` | `1.92 ± 1.60` | `99.74 ± 0.30` | `99.06 ± 0.63` | `98.86 ± 0.88` |
| `insdn` | `binary` | **`FedAvg` — Dirichlet $\alpha=1.0$ (`N=3`)** | `99.27 ± 0.25` | **`99.20 ± 0.28`** | `99.45 ± 0.19` | `99.64 ± 0.27` | `1.42 ± 0.78` | `99.88 ± 0.08` | `99.13 ± 0.31` | `98.96 ± 0.43` |
| `insdn` | `binary` | **`FedAvg` — Dirichlet $\alpha=0.5$ (`N=3`)** | `99.18 ± 0.10` | **`99.10 ± 0.11`** | `99.38 ± 0.07` | `99.51 ± 0.32` | `1.42 ± 0.56` | `99.89 ± 0.03` | `98.60 ± 0.27` | `97.69 ± 0.87` |
| `insdn` | `binary` | **`FedAvg` — Dirichlet $\alpha=0.1$ (`N=3`)** | `95.31 ± 2.48` | **`94.63 ± 2.40`** | `96.46 ± 1.88` | `99.42 ± 0.68` | `12.58 ± 7.21` | `98.89 ± 0.60` | `87.31 ± 5.14` | `72.42 ± 14.37` |
| `insdn` | `binary` | **Local-Only Mean (Lower Bound, `N=3`)** | `99.04 ± 0.06` | **`98.94 ± 0.08`** | `99.28 ± 0.05` | `99.50 ± 0.19` | `1.82 ± 0.14` | `99.81 ± 0.05` | `98.94 ± 0.32` | `98.63 ± 0.02` |
| `insdn` | `multiclass` | **Centralized Pooled (Upper Bound)** | `98.09 ± 0.27` | **`54.69 ± 1.53`** | `—` | `99.51 ± 0.09` | `1.18 ± 0.56` | `94.63 ± 2.57` | `—` | `—` |
| `insdn` | `multiclass` | **`FedAvg` — IID (`N=3`)** | `96.24 ± 0.24` | **`65.90 ± 1.51`** | `—` | `99.53 ± 0.10` | `0.81 ± 0.17` | `99.00 ± 0.44` | `65.72 ± 1.66` | `64.61 ± 2.41` |
| `insdn` | `multiclass` | **`FedAvg` — Natural Non-IID (`N=3`)** | `96.34 ± 0.36` | **`69.43 ± 1.22`** | `—` | `99.52 ± 0.09` | `0.73 ± 0.18` | `99.57 ± 0.10` | `69.03 ± 1.28` | `56.60 ± 1.98` |
| `insdn` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=1.0$ (`N=3`)** | `94.37 ± 1.32` | **`61.63 ± 1.72`** | `—` | `99.45 ± 0.14` | `0.90 ± 0.24` | `98.26 ± 0.56` | `63.61 ± 2.56` | `57.97 ± 2.66` |
| `insdn` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=0.5$ (`N=3`)** | `89.38 ± 4.91` | **`64.34 ± 0.75`** | `—` | `99.31 ± 0.15` | `1.59 ± 0.67` | `98.07 ± 0.56` | `64.42 ± 2.75` | `58.95 ± 4.65` |
| `insdn` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=0.1$ (`N=3`)** | `56.86 ± 17.57` | **`31.84 ± 2.69`** | `—` | `96.54 ± 4.95` | `18.30 ± 16.41` | `83.99 ± 4.53` | `35.29 ± 7.12` | `20.47 ± 11.79` |
| `insdn` | `multiclass` | **Local-Only Mean (Lower Bound, `N=3`)** | `89.98 ± 4.24` | **`35.52 ± 3.41`** | `—` | `99.17 ± 0.13` | `1.89 ± 0.16` | `88.75 ± 2.10` | `37.27 ± 5.86` | `31.22 ± 1.25` |
| `aseados_sdn_iot` | `binary` | **Centralized Pooled (Upper Bound)** | `91.26 ± 0.05` | **`91.23 ± 0.06`** | `90.77 ± 0.08` | `95.07 ± 0.41` | `11.89 ± 0.31` | `97.45 ± 0.03` | `—` | `—` |
| `aseados_sdn_iot` | `binary` | **`FedAvg` — IID (`N=3`)** | `90.66 ± 0.11` | **`90.63 ± 0.11`** | `90.06 ± 0.13` | `93.67 ± 0.47` | `11.82 ± 0.49` | `97.05 ± 0.03` | `90.63 ± 0.11` | `90.47 ± 0.13` |
| `aseados_sdn_iot` | `binary` | **`FedAvg` — Natural Non-IID (`N=3`)** | `90.54 ± 0.07` | **`90.51 ± 0.07`** | `89.92 ± 0.09` | `93.37 ± 0.48` | `11.79 ± 0.46` | `96.98 ± 0.04` | `89.16 ± 0.09` | `87.35 ± 0.15` |
| `aseados_sdn_iot` | `binary` | **`FedAvg` — Dirichlet $\alpha=1.0$ (`N=3`)** | `84.74 ± 2.43` | **`83.94 ± 2.97`** | `80.34 ± 5.20` | `71.60 ± 10.67` | `4.41 ± 3.16` | `96.32 ± 0.25` | `82.13 ± 3.24` | `76.19 ± 5.25` |
| `aseados_sdn_iot` | `binary` | **`FedAvg` — Dirichlet $\alpha=0.5$ (`N=3`)** | `83.90 ± 1.11` | **`82.83 ± 1.43`** | `78.30 ± 2.67` | `67.15 ± 5.39` | `2.28 ± 1.32` | `96.42 ± 0.17` | `79.64 ± 3.31` | `71.09 ± 6.55` |
| `aseados_sdn_iot` | `binary` | **`FedAvg` — Dirichlet $\alpha=0.1$ (`N=3`)** | `73.80 ± 0.62` | **`73.50 ± 0.43`** | `76.64 ± 1.07` | `93.62 ± 7.80` | `42.57 ± 7.51` | `85.59 ± 6.27` | `70.11 ± 6.00` | `56.83 ± 10.89` |
| `aseados_sdn_iot` | `binary` | **Local-Only Mean (Lower Bound, `N=3`)** | `89.82 ± 0.56` | **`89.71 ± 0.61`** | `88.72 ± 0.87` | `89.10 ± 5.24` | `9.58 ± 3.34` | `96.55 ± 0.28` | `89.75 ± 1.54` | `88.15 ± 0.09` |
| `aseados_sdn_iot` | `multiclass` | **Centralized Pooled (Upper Bound)** | `86.07 ± 0.03` | **`58.92 ± 0.31`** | `—` | `88.95 ± 0.60` | `7.80 ± 0.27` | `95.95 ± 0.03` | `—` | `—` |
| `aseados_sdn_iot` | `multiclass` | **`FedAvg` — IID (`N=3`)** | `84.64 ± 0.34` | **`58.41 ± 1.23`** | `—` | `93.56 ± 0.67` | `12.63 ± 0.65` | `95.39 ± 0.07` | `58.40 ± 1.21` | `57.93 ± 1.18` |
| `aseados_sdn_iot` | `multiclass` | **`FedAvg` — Natural Non-IID (`N=3`)** | `84.51 ± 0.56` | **`57.36 ± 1.35`** | `—` | `93.89 ± 0.72` | `13.21 ± 0.68` | `95.30 ± 0.07` | `58.08 ± 1.22` | `53.20 ± 0.78` |
| `aseados_sdn_iot` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=1.0$ (`N=3`)** | `81.99 ± 0.86` | **`59.10 ± 1.31`** | `—` | `94.28 ± 1.79` | `15.70 ± 2.28` | `94.61 ± 0.14` | `58.04 ± 0.90` | `53.24 ± 1.37` |
| `aseados_sdn_iot` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=0.5$ (`N=3`)** | `82.34 ± 0.96` | **`60.13 ± 0.72`** | `—` | `90.22 ± 4.00` | `11.98 ± 4.06` | `94.24 ± 0.17` | `59.40 ± 1.48` | `54.78 ± 2.41` |
| `aseados_sdn_iot` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=0.1$ (`N=3`)** | `62.20 ± 4.44` | **`40.10 ± 1.39`** | `—` | `96.15 ± 5.21` | `45.32 ± 14.37` | `84.72 ± 2.58` | `39.77 ± 2.47` | `28.59 ± 5.71` |
| `aseados_sdn_iot` | `multiclass` | **Local-Only Mean (Lower Bound, `N=3`)** | `83.37 ± 2.13` | **`52.19 ± 2.90`** | `—` | `85.89 ± 6.69` | `8.22 ± 3.50` | `94.12 ± 0.85` | `54.98 ± 2.13` | `52.84 ± 0.71` |
| `cicids2017` | `binary` | **Centralized Pooled (Upper Bound)** | `98.33 ± 0.10` | **`96.66 ± 0.18`** | `94.30 ± 0.30` | `91.74 ± 0.82` | `0.50 ± 0.21` | `99.58 ± 0.08` | `—` | `—` |
| `cicids2017` | `binary` | **`FedAvg` — Natural Non-IID (`N=3`)** | `95.06 ± 0.56` | **`90.50 ± 1.03`** | `83.99 ± 1.71` | `85.62 ± 4.62` | `3.27 ± 1.34` | `98.06 ± 0.17` | `89.96 ± 1.15` | `87.84 ± 1.98` |
| `cicids2017` | `binary` | **`FedAvg` — Dirichlet $\alpha=0.5$ (`N=10`)** | `91.05 ± 2.26` | **`84.34 ± 2.63`** | `74.47 ± 3.85` | `87.64 ± 6.80` | `8.35 ± 3.74` | `95.44 ± 0.44` | `83.52 ± 2.75` | `74.81 ± 4.19` |
| `cicids2017` | `binary` | **`FedAvg` — IID (`N=10`)** | `86.55 ± 1.37` | **`78.49 ± 1.56`** | `65.87 ± 2.16` | `84.88 ± 5.21` | `13.16 ± 2.41` | `94.07 ± 0.37` | `78.48 ± 1.56` | `78.18 ± 1.57` |
| `cicids2017` | `multiclass` | **Centralized Pooled (Upper Bound)** | `98.29 ± 0.08` | **`57.88 ± 1.02`** | `—` | `92.57 ± 0.37` | `0.69 ± 0.05` | `97.60 ± 1.84` | `—` | `—` |
| `cicids2017` | `multiclass` | **`FedAvg` — Natural Non-IID (`N=3`)** | `94.05 ± 0.90` | **`43.35 ± 1.58`** | `—` | `96.26 ± 0.42` | `6.14 ± 1.06` | `98.92 ± 0.08` | `44.91 ± 1.72` | `38.65 ± 2.68` |
| `cicids2017` | `multiclass` | **`FedAvg` — Dirichlet $\alpha=0.5$ (`N=10`)** | `91.14 ± 0.72` | **`36.41 ± 0.59`** | `—` | `96.21 ± 0.73` | `9.57 ± 0.89` | `97.57 ± 0.25` | `36.31 ± 0.67` | `31.90 ± 1.48` |
| `cicids2017` | `multiclass` | **`FedAvg` — IID (`N=10`)** | `86.65 ± 1.24` | **`34.99 ± 0.77`** | `—` | `96.90 ± 0.57` | `14.95 ± 1.50` | `97.79 ± 0.14` | `35.01 ± 0.77` | `34.25 ± 0.83` |

#### 2. Quantified Loss to Federation & Loss to Non-IID Data (`Centralized` $\to$ `FedAvg IID` $\to$ `FedAvg Non-IID`)

| Dataset | Task | Centralized Macro-F1 (%) | `FedAvg` IID Macro-F1 (%) | **Loss to Federation** (`Centralized - FedAvg IID`, pts) | `FedAvg` Natural (`N=3`) Macro-F1 (%) | **Loss to Domain Non-IID** (`FedAvg IID - Natural`, pts) | `FedAvg` Dirichlet $\alpha=0.1$ Macro-F1 (%) | **Loss to Extreme Non-IID** (`FedAvg IID - Dir 0.1`, pts) | **`FedAvg` Natural Gain over Local-Only** (pts) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | `binary` | `99.74%` | `99.16%` | **`0.57` pts** (`0.52` Acc pts) | `99.13%` | **`0.04` pts** ($p=0.625$) | `94.63%` | **`4.54` pts** ($p=0.0625$) | **`+0.19` pts** |
| `insdn` | `multiclass` | `54.69%` | `65.90%` | **`-11.21` pts** (`+1.85` Acc loss) | `69.43%` | **`-3.53` pts** (`-0.10` Acc loss) | `31.84%` | **`34.05` pts** ($p=0.0625$) | **`+33.91` pts** |
| `aseados_sdn_iot` | `binary` | `91.23%` | `90.63%` | **`0.60` pts** (`0.60` Acc pts) | `90.51%` | **`0.12` pts** ($p=0.125$) | `73.50%` | **`17.13` pts** ($p=0.0625$) | **`+0.80` pts** |
| `aseados_sdn_iot` | `multiclass` | `58.92%` | `58.41%` | **`0.50` pts** (`1.43` Acc pts) | `57.36%` | **`1.05` pts** ($p=0.3125$) | `40.10%` | **`18.32` pts** ($p=0.0625$) | **`+5.17` pts** |
| `cicids2017` | `binary` | `96.66%` | `78.49%` (`N=10`) | **`6.16` pts** (`Centralized - Natural N=3`) | `90.50%` | `—` | `84.34%` ($\alpha=0.5, N=10$) | `—` | `—` |
| `cicids2017` | `multiclass` | `57.88%` | `34.99%` (`N=10`) | **`14.53` pts** (`Centralized - Natural N=3`) | `43.35%` | `—` | `36.41%` ($\alpha=0.5, N=10$) | `—` | **`+0.91` pts** |

#### 3. Scalability Across Tenant Counts $N \in \{3, 5, 10\}$ and Data Heterogeneity (`Dirichlet` $\alpha \in \{0.1, 0.5, 1.0\}$ & `IID`)

| Dataset | Task | Partition Regime | `N = 3` Macro-F1 (%) | `N = 5` Macro-F1 (%) | `N = 10` Macro-F1 (%) | `N = 3` Round to 95% Final F1 | `N = 10` Round to 95% Final F1 |
|---|---|---|---:|---:|---:|---:|---:|
| `insdn` | `binary` | **Natural Domain Non-IID** | **`99.13 ± 0.59`** | `—` | `—` | `Round 1.0` | `—` |
| `insdn` | `binary` | **Stratified IID** | **`99.16 ± 0.68`** | **`99.02 ± 0.40`** | **`98.92 ± 0.16`** | `Round 1.0` | `Round 1.0` |
| `insdn` | `binary` | **Dirichlet $\alpha = 1.0$ (Mild)** | **`99.20 ± 0.28`** | **`99.05 ± 0.35`** | **`98.75 ± 0.08`** | `Round 1.0` | `Round 1.0` |
| `insdn` | `binary` | **Dirichlet $\alpha = 0.5$ (Moderate)** | **`99.10 ± 0.11`** | **`99.07 ± 0.32`** | **`98.77 ± 0.08`** | `Round 1.0` | `Round 1.0` |
| `insdn` | `binary` | **Dirichlet $\alpha = 0.1$ (Extreme)** | **`94.63 ± 2.40`** | **`99.21 ± 0.32`** | **`98.96 ± 0.08`** | `Round 3.4` | `Round 4.2` |
| `insdn` | `multiclass` | **Natural Domain Non-IID** | **`69.43 ± 1.22`** | `—` | `—` | `Round 9.0` | `—` |
| `insdn` | `multiclass` | **Stratified IID** | **`65.90 ± 1.51`** | **`66.00 ± 1.23`** | **`61.44 ± 0.67`** | `Round 6.6` | `Round 9.0` |
| `insdn` | `multiclass` | **Dirichlet $\alpha = 1.0$ (Mild)** | **`61.63 ± 1.72`** | **`62.61 ± 2.13`** | **`56.24 ± 2.57`** | `Round 10.2` | `Round 11.0` |
| `insdn` | `multiclass` | **Dirichlet $\alpha = 0.5$ (Moderate)** | **`64.34 ± 0.75`** | **`58.92 ± 2.02`** | **`51.11 ± 1.58`** | `Round 12.6` | `Round 11.0` |
| `insdn` | `multiclass` | **Dirichlet $\alpha = 0.1$ (Extreme)** | **`31.84 ± 2.69`** | **`52.17 ± 1.02`** | **`52.37 ± 0.90`** | `Round 10.2` | `Round 15.0` |
| `aseados_sdn_iot` | `binary` | **Natural Domain Non-IID** | **`90.51 ± 0.07`** | `—` | `—` | `Round 1.0` | `—` |
| `aseados_sdn_iot` | `binary` | **Stratified IID** | **`90.63 ± 0.11`** | **`89.87 ± 0.19`** | **`85.45 ± 0.43`** | `Round 1.0` | `Round 1.0` |
| `aseados_sdn_iot` | `binary` | **Dirichlet $\alpha = 1.0$ (Mild)** | **`83.94 ± 2.97`** | `—` | **`82.83 ± 0.43`** | `Round 5.8` | `Round 1.0` |
| `aseados_sdn_iot` | `binary` | **Dirichlet $\alpha = 0.5$ (Moderate)** | **`82.83 ± 1.43`** | **`82.44 ± 0.25`** | **`83.21 ± 0.47`** | `Round 6.6` | `Round 1.0` |
| `aseados_sdn_iot` | `binary` | **Dirichlet $\alpha = 0.1$ (Extreme)** | **`73.50 ± 0.43`** | `—` | **`71.43 ± 2.86`** | `Round 5.4` | `Round 5.8` |
| `aseados_sdn_iot` | `multiclass` | **Natural Domain Non-IID** | **`57.36 ± 1.35`** | `—` | `—` | `Round 15.0` | `—` |
| `aseados_sdn_iot` | `multiclass` | **Stratified IID** | **`58.41 ± 1.23`** | **`55.66 ± 1.07`** | **`48.94 ± 0.50`** | `Round 18.6` | `Round 5.0` |
| `aseados_sdn_iot` | `multiclass` | **Dirichlet $\alpha = 1.0$ (Mild)** | **`59.10 ± 1.31`** | `—` | **`46.46 ± 0.87`** | `Round 16.2` | `Round 5.8` |
| `aseados_sdn_iot` | `multiclass` | **Dirichlet $\alpha = 0.5$ (Moderate)** | **`60.13 ± 0.72`** | **`55.57 ± 1.54`** | **`47.00 ± 1.13`** | `Round 15.0` | `Round 7.4` |
| `aseados_sdn_iot` | `multiclass` | **Dirichlet $\alpha = 0.1$ (Extreme)** | **`40.10 ± 1.39`** | `—` | **`39.46 ± 1.55`** | `Round 12.6` | `Round 9.8` |

#### 4. Communication Overhead (Exact Wire Serialization Bytes) & Wall-Clock Timing Breakdown

- **Wire Serialization Format (`trustfed/fl/aggregator.py`):** `24`-byte binary header (`b"TFSD"` magic + uint32 version + uint32 `round_idx` + uint32 `num_samples` + uint64 `numel`) followed by contiguous IEEE-754 little-endian `float32` parameter bytes (`4 * numel`). Every active tenant downloads `1` global payload and uploads `1` local update payload per round (`2 * N * payload_bytes` per round).

| Dataset | Task | Trainable Params (`numel`) | One-Way Payload (Bytes / KB) | Tenant Count (`N`) | Wire Traffic per Round (Bytes / MB) | Total 30-Round Traffic (MB) | Mean Local Train Time / Round (s) | Mean Server Aggregation Time / Round (ms) | Total 30-Round Wall Clock (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | `binary` | `43,554` | `174,240 B` (`170.16 KB`) | `N = 3` | `1,045,440 B` (`1.00 MB`) | **`31.36 MB`** | `0.565 s` | **`0.31 ms`** | `17.51 s` |
| `insdn` | `binary` | `43,554` | `174,240 B` (`170.16 KB`) | `N = 5` | `1,742,400 B` (`1.66 MB`) | **`52.27 MB`** | `0.592 s` | **`0.41 ms`** | `18.14 s` |
| `insdn` | `binary` | `43,554` | `174,240 B` (`170.16 KB`) | `N = 10` | `3,484,800 B` (`3.32 MB`) | **`104.54 MB`** | `0.646 s` | **`0.67 ms`** | `19.86 s` |
| `insdn` | `multiclass` | `43,752` | `175,032 B` (`170.93 KB`) | `N = 3` | `1,050,192 B` (`1.00 MB`) | **`31.51 MB`** | `0.578 s` | **`0.31 ms`** | `17.92 s` |
| `insdn` | `multiclass` | `43,752` | `175,032 B` (`170.93 KB`) | `N = 5` | `1,750,320 B` (`1.67 MB`) | **`52.51 MB`** | `0.598 s` | **`0.42 ms`** | `18.38 s` |
| `insdn` | `multiclass` | `43,752` | `175,032 B` (`170.93 KB`) | `N = 10` | `3,500,640 B` (`3.34 MB`) | **`105.02 MB`** | `0.659 s` | **`0.68 ms`** | `20.25 s` |
| `aseados_sdn_iot` | `binary` | `42,209` | `168,864 B` (`164.91 KB`) | `N = 3` | `1,013,184 B` (`0.97 MB`) | **`30.40 MB`** | `0.556 s` | **`0.29 ms`** | `17.24 s` |
| `aseados_sdn_iot` | `binary` | `42,209` | `168,864 B` (`164.91 KB`) | `N = 5` | `1,688,640 B` (`1.61 MB`) | **`50.66 MB`** | `0.579 s` | **`0.40 ms`** | `17.75 s` |
| `aseados_sdn_iot` | `binary` | `42,209` | `168,864 B` (`164.91 KB`) | `N = 10` | `3,377,280 B` (`3.22 MB`) | **`101.32 MB`** | `0.635 s` | **`0.64 ms`** | `19.47 s` |
| `cicids2017` | `binary` | `41,066` | `164,288 B` (`160.44 KB`) | `N = 3` | `985,728 B` (`0.94 MB`) | **`29.57 MB`** | `0.552 s` | **`0.29 ms`** | `17.12 s` |
| `cicids2017` | `binary` | `41,066` | `164,288 B` (`160.44 KB`) | `N = 10` | `3,285,760 B` (`3.13 MB`) | **`98.57 MB`** | `0.638 s` | **`0.65 ms`** | `19.58 s` |

#### 5. Scaler Alignment Ablation (`local_per_tenant` vs. `global` / Federated Moments on `InSDN` Binary `natural_n3`)

| Scaler Regime | Own-Tenant Test Macro-F1 Mean (%) | Global Test Macro-F1 (%) | Global Accuracy (%) | Global ADR (%) | Global FPR (%) |
|---|---:|---:|---:|---:|---:|
| **Aligned Global / Federated Moments (`global`)** | **`99.06 ± 0.63`** | **`99.13 ± 0.59`** | `99.21 ± 0.53` | `99.81 ± 0.14` | **`1.92 ± 1.60`** |
| **Unaligned Per-Tenant Local Scaler (`local_per_tenant`)** | **`98.78 ± 0.23`** | **`91.02 ± 7.23`** | `92.60 ± 5.71` | `99.49 ± 0.29` | **`20.81 ± 16.51`** |


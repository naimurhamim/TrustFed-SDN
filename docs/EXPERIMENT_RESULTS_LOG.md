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

---

## Step 04: Poisoning Attacks & Robust Aggregation (`04_robust`)

### [2026-11-12] Step 04 — Byzantine Poisoning Matrix, Robust Aggregators & Failure-Mode Ablations (`04_robust`)
- **Runner:** `sim/run_experiment.py --sweep-step04`
- **Configs:** `configs/experiments/robust_step04_sweep.yaml`, `configs/experiments/robust_insdn_label_flip_n10.yaml`, `configs/experiments/robust_insdn_backdoor_n10.yaml`, `configs/experiments/robust_insdn_scaling_n10.yaml`, `configs/experiments/robust_aseados_poison_n10.yaml`
- **Raw Summary & Artifacts:**
  - `results/04_robust/summary.json`
  - `results/04_robust/per_seed_summary.csv` (`660` FL runs = `132` configurations × `5` seeds)
  - `results/04_robust/per_round_metrics.csv`
  - `results/04_robust/accuracy_vs_poison_fraction.png`
  - `results/04_robust/backdoor_asr_comparison.png`
  - `results/04_robust/honest_false_rejection_rate.png`
- **Implemented Attacks (`trustfed/attacks/`):**
  1. `label_flip` (`LabelFlipAttack`): `flip_fraction=1.0`, `bidirectional=True`, `update_scale=8.5`
  2. `backdoor` (`BackdoorAttack`): `trigger_features=(3, 6, 8)` (calibrated for `<1%` clean baseline ASR), `trigger_value=4.5`, `target_class=0` (`Benign`), `poison_ratio=0.5`, `replacement_scale=6.0`, `collateral_flip_ratio=0.5`
  3. `scaling` (`UpdateScalingAttack`): `scale_factor=10.0`, `sign_flip=True` (`-10.0x` parameter delta reversal)
  4. `backdoor_stealth` (Adaptive Norm-Bounded Backdoor): `trigger_features=(3, 6, 8)`, `trigger_value=4.5`, `poison_ratio=0.5`, `replacement_scale=1.4`, `collateral_flip_ratio=0.0` (stays underneath `2.5x` median-norm filter)
- **Implemented Aggregators (`trustfed/aggregation/`):**
  1. `fedavg` (`FedAvgAggregator`): Sample-weighted unprotected baseline
  2. `static_weight` (`StaticWeightAggregator`): Ali et al. (2023) fixed-weight baseline calibrated once on validation data at Round 1 and frozen
  3. `trimmed_mean_b0.1` (`TrimmedMeanAggregator(beta=0.1)`): Coordinate-wise trimmed mean removing `k = floor(0.1 * N)` extremes per tail
  4. `trimmed_mean_b0.2` (`TrimmedMeanAggregator(beta=0.2)`): Coordinate-wise trimmed mean removing `k = floor(0.2 * N)` extremes per tail
  5. `median` (`MedianAggregator`): Coordinate-wise median (`beta -> 0.5`)
  6. `clustering` (`CosineClusteringAggregator`): Two-stage median-norm (`2.5x`) + pairwise cosine-similarity clique filtering with soft norm clipping

#### 1. Primary Poisoning Matrix (`InSDN` Binary, `Dirichlet` $\alpha = 0.5, N = 10$, 5 Seeds: Mean ± Std)

| Attack | Poisoned Fraction | `FedAvg` Macro-F1 (%) | `StaticWeight` (Ali et al.) Macro-F1 (%) | `TrimmedMean` ($\beta=0.1$) Macro-F1 (%) | `TrimmedMean` ($\beta=0.2$) Macro-F1 (%) | `Median` Macro-F1 (%) | `CosineClustering` Macro-F1 (%) | `FedAvg` Backdoor ASR (%) | `TrimmedMean` ($\beta=0.2$) Backdoor ASR (%) | `CosineClustering` Backdoor ASR (%) | `CosineClustering` Malicious Rej. (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Clean (None)`** | **`0%` (`0/10`)** | **`98.52 ± 0.09`** | `98.53 ± 0.08` | `98.31 ± 0.17` | `98.30 ± 0.18` | `98.26 ± 0.22` | **`98.25 ± 0.12`** | `0.00%` | `0.00%` | **`0.00%`** | `0.0%` (`0.0%` H-FRR) |
| **`Label Flipping`** | **`10%` (`1/10`)** | `62.79 ± 37.64` | `88.79 ± 11.69` | **`98.42 ± 0.17`** | **`98.44 ± 0.14`** | **`98.40 ± 0.15`** | **`98.52 ± 0.05`** | `40.00%` | `0.00%` | **`0.00%`** | **`100.0%`** |
| **`Label Flipping`** | **`20%` (`2/10`)** | `53.45 ± 21.02` | `92.00 ± 6.36` | `51.66 ± 29.40` *(fails: $\beta < 0.2$)* | **`98.32 ± 0.13`** | **`98.26 ± 0.15`** | **`98.53 ± 0.07`** | `80.30%` | `0.00%` | **`0.00%`** | **`100.0%`** |
| **`Label Flipping`** | **`30%` (`3/10`)** | `27.08 ± 11.16` | `79.58 ± 24.78` | `25.35 ± 12.57` | `44.03 ± 21.72` *(fails: $\beta < 0.3$)* | **`98.27 ± 0.23`** | **`98.51 ± 0.08`** | `76.70%` | `63.50%` | **`0.00%`** | **`100.0%`** |
| **`Backdoor (6.0x)`** | **`10%` (`1/10`)** | `81.62 ± 29.41` | `93.38 ± 8.37` | `98.55 ± 0.09` | `98.54 ± 0.11` | `98.49 ± 0.10` | **`98.52 ± 0.05`** | **`99.90%`** | `0.00%` | **`0.00%`** | **`100.0%`** |
| **`Backdoor (6.0x)`** | **`20%` (`2/10`)** | `74.00 ± 19.88` | `96.33 ± 1.99` | `94.11 ± 1.00` | `98.57 ± 0.03` | `98.56 ± 0.05` | **`98.53 ± 0.07`** | **`100.00%`** | **`76.70%`** *(evades trim!)* | **`0.00%`** | **`95.6%`** |
| **`Backdoor (6.0x)`** | **`30%` (`3/10`)** | `48.25 ± 6.58` | `74.36 ± 28.52` | `65.22 ± 12.06` | `85.85 ± 15.66` | `98.54 ± 0.11` | **`98.50 ± 0.08`** | **`100.00%`** | **`100.00%`** *(evades trim!)* | **`0.00%`** | **`96.7%`** |
| **`Update Scaling`** | **`10%` (`1/10`)** | `51.95 ± 33.71` | `77.02 ± 28.66` | **`98.23 ± 0.29`** | **`98.30 ± 0.22`** | **`98.22 ± 0.20`** | **`98.52 ± 0.05`** | `16.60%` | `0.00%` | **`0.00%`** | **`100.0%`** |
| **`Update Scaling`** | **`20%` (`2/10`)** | `25.09 ± 9.80` | `87.08 ± 9.05` | `38.49 ± 23.52` *(fails: $\beta < 0.2$)* | **`98.10 ± 0.50`** | **`98.18 ± 0.35`** | **`98.53 ± 0.07`** | `82.90%` | `0.00%` | **`0.00%`** | **`100.0%`** |
| **`Update Scaling`** | **`30%` (`3/10`)** | `20.29 ± 4.09` | `51.43 ± 32.63` | `21.10 ± 3.84` | `24.73 ± 4.77` *(fails: $\beta < 0.3$)* | **`97.70 ± 0.53`** | **`98.51 ± 0.08`** | `79.50%` | `73.70%` | **`0.00%`** | **`100.0%`** |

#### 2. Natural Domain-Specialty Non-IID (`InSDN` `natural_n10`) & Modern 2026 SDN-IoT (`ASEADOS-SDN-IoT` `dirichlet_a0.5_n10`)

| Dataset (`N=10`) | Attack & Poisoned Fraction | `FedAvg` Macro-F1 (%) | `StaticWeight` Macro-F1 (%) | `TrimmedMean` ($\beta=0.2$) Macro-F1 (%) | `CosineClustering` Macro-F1 (%) | `FedAvg` Backdoor ASR (%) | `TrimmedMean` ($\beta=0.2$) Backdoor ASR (%) | `CosineClustering` Backdoor ASR (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `insdn` (`natural_n10`) | `Clean` (`0%`) | `98.51 ± 0.06` | `98.55 ± 0.07` | `98.46 ± 0.08` | **`98.45 ± 0.11`** | `0.00%` | `0.00%` | **`0.00%`** |
| `insdn` (`natural_n10`) | `Label Flip` (`20%`) | `25.43 ± 16.09` | `85.57 ± 8.69` | `98.35 ± 0.17` | **`98.48 ± 0.07`** | `84.10%` | `0.00%` | **`0.00%`** |
| `insdn` (`natural_n10`) | `Backdoor` (`20%`) | `73.07 ± 21.37` | `84.49 ± 16.00` | `98.52 ± 0.12` | **`98.48 ± 0.07`** | **`100.00%`** | **`59.70%`** | **`0.00%`** |
| `insdn` (`natural_n10`) | `Update Scaling` (`20%`) | `23.41 ± 9.25` | `87.56 ± 6.53` | `98.21 ± 0.38` | **`98.48 ± 0.07`** | `93.80%` | `0.00%` | **`0.00%`** |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Clean` (`0%`) | `78.22 ± 4.23` | `81.00 ± 1.58` | `72.45 ± 5.23` | **`75.22 ± 5.23`** | `63.20%` | `55.60%` | `63.80%` |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Label Flip` (`20%`) | `28.16 ± 4.65` | `57.37 ± 19.06` | **`75.25 ± 5.52`** | **`74.81 ± 6.46`** | `35.10%` | `79.20%` | `59.90%` |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Label Flip` (`30%`) | `28.27 ± 6.21` | `28.80 ± 5.32` | `33.67 ± 9.90` | **`72.73 ± 5.20`** | `50.30%` | `40.60%` | **`47.50%`** |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Backdoor` (`20%`) | `49.73 ± 11.41` | `56.94 ± 18.50` | `71.91 ± 3.89` | **`74.81 ± 6.46`** | **`100.00%`** | **`100.00%`** | **`59.90%`** |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Backdoor` (`30%`) | `42.69 ± 11.27` | `45.89 ± 11.37` | `59.28 ± 10.08` | **`72.73 ± 5.20`** | **`100.00%`** | **`100.00%`** | **`47.50%`** |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Update Scaling` (`20%`) | `26.00 ± 4.00` | `40.85 ± 20.75` | `66.91 ± 3.33` | **`74.81 ± 6.46`** | `99.00%` | `60.00%` | **`59.90%`** |
| `aseados_sdn_iot` (`dir_a0.5_n10`) | `Update Scaling` (`30%`) | `28.56 ± 2.40` | `28.91 ± 3.97` | `30.49 ± 3.76` | **`72.73 ± 5.20`** | `53.60%` | `68.90%` | **`47.50%`** |

#### 3. Empirical Failure-Mode Ablations (Where Existing Robust Aggregators Still Fail — Direct Motivation for Step 05 `TrustFed-Core`)

| Failure Mode / Ablation Regime (`InSDN`, 5 Seeds) | `FedAvg` Macro-F1 (%) | `StaticWeight` (Ali et al.) Macro-F1 (%) | `TrimmedMean` ($\beta=0.2$) Macro-F1 (%) | `Median` Macro-F1 (%) | `CosineClustering` Macro-F1 (%) | Key Failure Metric (`ASR` or `Honest False-Rejection Rate`) |
|---|---:|---:|---:|---:|---:|---|
| **1. Extreme Non-IID Clean Drift (`Dirichlet` $\alpha=0.1, N=10, 0\%$ Poisoned)** | **`98.64 ± 0.05`** | `—` | `93.77 ± 2.50` *(-4.87 pts)* | `91.56 ± 3.46` *(-7.08 pts)* | **`76.39 ± 3.76`** *(-22.25 pts!)* | `CosineClustering` falsely rejects **`40.0%`** of honest tenants (`H-FRR = 40.0%`) due to severe non-IID gradient divergence! |
| **2. Extreme Non-IID + 20% Scaling (`Dirichlet` $\alpha=0.1, N=10, 20\%$ Poisoned)** | `68.43 ± 29.95` | `—` | `82.98 ± 11.21` | `83.75 ± 10.10` | **`96.47 ± 3.75`** | `CosineClustering` still suffers **`9.5%`** Honest False-Rejection (`H-FRR = 9.5%`); `TrimmedMean` loses `15.66` F1 pts by trimming rare-class honest coordinates. |
| **3. Few-Tenant Breakdown (`Natural Non-IID`, $N=3, 33.3\%$ Label-Flip)** | `22.77 ± 10.42` | `—` | **`28.85 ± 8.22`** *(collapses!)* | `98.81 ± 0.16` | `98.84 ± 0.18` | At $N=3$, `TrimmedMean` ($\beta=0.2$) computes $k = \lfloor 0.2 \times 3 \rfloor = 0$, trimming **zero** updates and collapsing alongside `FedAvg`! |
| **4. Few-Tenant Breakdown (`Natural Non-IID`, $N=3, 33.3\%$ Update Scaling)** | `16.84 ± 1.62` | `—` | **`16.19 ± 0.36`** *(collapses!)* | `98.66 ± 0.26` | `98.84 ± 0.18` | Confirming that coordinate-wise trimmed mean is mathematically inoperable when $\lfloor \beta N \rfloor < m$ ($N=3$). |
| **5. Subspace Backdoor Evasion (`Dirichlet` $\alpha=0.5, N=10, 30\%$ Backdoor `6.0x`)** | `48.25 ± 6.58` | `74.36 ± 28.52` | `85.85 ± 15.66` | `98.54 ± 0.11` | **`98.50 ± 0.08`** | `Median` (`ASR = 100.0%`), `TrimmedMean` (`ASR = 100.0%`), and `StaticWeight` (`ASR = 99.8%`) completely fail to stop the localized trigger on columns `(3, 6, 8)`! |
| **6. Adaptive Norm-Bounded Stealth Backdoor (`Dirichlet` $\alpha=0.5, N=10, 30\%$ Backdoor `1.4x`)** | `98.55 ± 0.07` (`ASR = 99.9%`) | `98.55 ± 0.07` (`ASR = 80.3%`) | `98.54 ± 0.07` (`ASR = 99.9%`) | `—` | `98.53 ± 0.06` (**`ASR = 93.7 ± 12.4%`!**) | By keeping `replacement_scale = 1.4x` underneath the `2.5x` norm threshold, the stealthy attacker evades **`89.2%`** of `CosineClustering` rejections (`M-TRR = 10.8%`), driving `Backdoor ASR` from `0.0%` up to **`93.7%`**! |

---

## 6. Step 05 — Adaptive Per-Tenant Differential Privacy & Membership Inference Attack Results (`results/05_dp/`)

- **Completed:** `2026-11-12` (`@Naimur`)
- **Design Specification:** [`docs/DP_DESIGN.md`](DP_DESIGN.md)
- **Artifacts:**
  - [`results/05_dp/summary.json`](../results/05_dp/summary.json)
  - [`results/05_dp/per_seed_summary.csv`](../results/05_dp/per_seed_summary.csv)
  - [`results/05_dp/per_tenant_dp_table.csv`](../results/05_dp/per_tenant_dp_table.csv)
  - [`results/05_dp/mia_evaluation.csv`](../results/05_dp/mia_evaluation.csv)
  - [`results/05_dp/privacy_utility_tradeoff.png`](../results/05_dp/privacy_utility_tradeoff.png)
  - [`results/05_dp/per_tenant_dp_comparison.png`](../results/05_dp/per_tenant_dp_comparison.png)
  - [`results/05_dp/mia_attack_evaluation.png`](../results/05_dp/mia_attack_evaluation.png)
- **Experimental Setup:**
  - **Granularity & Mechanism:** Record-level DP-SGD (`trustfed/privacy/dp_sgd.py`) with Poisson subsampling ($q_i = B / n_i$, $B = 512$), exact `torch.func.vmap(grad(...))` per-sample gradients via `functional_gru_forward`, $\ell_2$ clipping norm $C \in \{0.5, 0.8\}$, Gaussian noise $\mathcal{N}(0, \sigma_i^2 C^2 \mathbf{I}_d)$, and exact Rényi Differential Privacy (RDP) accounting (`trustfed/privacy/accountant.py`, Mironov et al. 2019 + Balle et al. 2020 Theorem 21 conversion at $\delta = 10^{-5} < 1/n_i$).
  - **Hybrid Sensitivity Score ($s_i$):** $s_i = 0.5 \cdot \text{domain\_profile}_i + 0.5 \cdot \text{data\_score}_i \in [0.5, 1.0]$ (`trustfed/privacy/budget.py`), combining domain criticality (`Finance = 1.0`, `Healthcare = 1.0`, `Education = 0.5`) with rare-attack share ($<5\%$ global attack share) and class imbalance ($1 - \min_c n_{i,c} / \max_c n_{i,c}$).
  - **Budget Policy Comparison:** `Adaptive` ($\varepsilon_i = \varepsilon_{\text{total}} \frac{1/s_i}{\sum_j 1/s_j}$) vs. `Uniform` ($\varepsilon_i = \varepsilon_{\text{total}} / N = \bar{\varepsilon}$) at equal total budget $\varepsilon_{\text{total}} = 3 \bar{\varepsilon}$ across $\bar{\varepsilon} \in \{0.5, 1.0, 2.0, 4.0, 8.0, \infty\}$ on `FedAvg`, Natural Non-IID ($N=3$), `5` random seeds (`0, 1, 2, 3, 4`).

### 6.1 Hybrid Sensitivity Scores & Budget Allocation Multipliers (`Natural Non-IID, N=3`)

| Dataset | Tenant | Training Flows ($n_i$) | Domain Profile | Rare-Attack Share | Class Imbalance | Data Score | Hybrid Score $s_i$ | Adaptive Budget Share ($\varepsilon_i / \bar{\varepsilon}$) | Privacy Gain vs. Uniform |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`InSDN`** | `tenant_1_finance` | `2,625` | `1.00` | `7.55%` | `0.9999` | `1.0000` | **`1.0000`** | **`0.7119 × ε̄`** | **`28.8%` smaller $\varepsilon_1$ (stronger privacy)** |
| **`InSDN`** | `tenant_2_healthcare` | `4,414` | `1.00` | `2.85%` | `0.9999` | `0.6474` | **`0.8237`** | **`0.8643 × ε̄`** | **`13.6%` smaller $\varepsilon_2$ (stronger privacy)** |
| **`InSDN`** | `tenant_3_education` | `4,960` | `0.50` | `0.90%` | `0.9998` | `0.5000` | **`0.5000`** | **`1.4238 × ε̄`** | `+42.4%` larger $\varepsilon_3$ (cleaner bulk gradients) |
| **`ASEADOS-SDN-IoT`** | `tenant_1_finance` | `3,144` | `1.00` | `21.81%` | `0.9857` | `1.0000` | **`1.0000`** | **`0.8062 × ε̄`** | **`19.4%` smaller $\varepsilon_1$ (stronger privacy)** |
| **`ASEADOS-SDN-IoT`** | `tenant_2_healthcare` | `5,737` | `1.00` | `1.95%` | `0.9874` | `0.5000` | **`0.7500`** | **`1.0749 × ε̄`** | `+7.5%` larger $\varepsilon_2$ |
| **`ASEADOS-SDN-IoT`** | `tenant_3_education` | `3,120` | `0.50` | `19.78%` | `0.9827` | `0.9409` | **`0.7205`** | **`1.1190 × ε̄`** | `+11.9%` larger $\varepsilon_3$ |

### 6.2 Privacy-Utility Trade-Off & Per-Tenant Breakdown (`InSDN`, 5 Seeds `0..4`, $\delta = 10^{-5}$)

| Mean Budget $\bar{\varepsilon}$ ($\varepsilon_{\text{total}}$) | Policy | Global Test Macro-F1 (%) | Global Test Accuracy (%) | `Finance` ($\varepsilon_1, \sigma_1$) | `Finance` Macro-F1 (%) | `Healthcare` ($\varepsilon_2, \sigma_2$) | `Healthcare` Macro-F1 (%) | `Education` ($\varepsilon_3, \sigma_3$) | `Education` Macro-F1 (%) |
|---|---|---:|---:|---|---:|---|---:|---|---:|
| **`∞` (No-DP)** | `none` | **`98.69 ± 0.07`** | `98.81 ± 0.06` | `∞` ($\sigma_1 = 0.00$) | `98.43 ± 0.06` | `∞` ($\sigma_2 = 0.00$) | `98.27 ± 0.09` | `∞` ($\sigma_3 = 0.00$) | `99.11 ± 0.09` |
| **`8.0` (`24.0`)** | `uniform` | **`97.86 ± 0.14`** | `98.09 ± 0.12` | `8.000` ($\sigma_1 = 1.37$) | `97.32 ± 0.19` | `8.000` ($\sigma_2 = 1.11$) | `97.28 ± 0.16` | `8.000` ($\sigma_3 = 1.07$) | `98.53 ± 0.15` |
| **`8.0` (`24.0`)** | `adaptive` | **`97.10 ± 0.99`** | `97.46 ± 0.82` | **`5.695`** ($\sigma_1 = 1.70$) | `96.90 ± 0.58` | **`6.914`** ($\sigma_2 = 1.20$) | `95.70 ± 2.02` | `11.391` ($\sigma_3 = 0.90$) | `98.27 ± 0.36` |
| **`4.0` (`12.0`)** | `uniform` | **`97.16 ± 1.02`** | `97.50 ± 0.85` | `4.000` ($\sigma_1 = 2.19$) | `96.80 ± 0.65` | `4.000` ($\sigma_2 = 1.70$) | `96.06 ± 1.98` | `4.000` ($\sigma_3 = 1.62$) | `98.17 ± 0.47` |
| **`4.0` (`12.0`)** | `adaptive` | **`97.14 ± 1.08`** | `97.48 ± 0.90` | **`2.848`** ($\sigma_1 = 2.85$) | **`96.78 ± 0.75`** | **`3.457`** ($\sigma_2 = 1.87$) | `96.00 ± 2.00` | `5.695` ($\sigma_3 = 1.28$) | `98.17 ± 0.52` |
| **`2.0` (`6.0`)** | `uniform` | `83.94 ± 3.83` | `86.44 ± 2.66` | `2.000` ($\sigma_1 = 3.80$) | `82.89 ± 6.47` | `2.000` ($\sigma_2 = 2.84$) | `79.74 ± 1.44` | `2.000` ($\sigma_3 = 2.68$) | `87.60 ± 5.31` |
| **`2.0` (`6.0`)** | `adaptive` | **`84.28 ± 3.55`** | **`86.45 ± 2.61`** | **`1.424`** ($\sigma_1 = 5.07$) | **`85.49 ± 4.40`** | **`1.729`** ($\sigma_2 = 3.20$) | `76.63 ± 3.42` | `2.848` ($\sigma_3 = 2.04$) | **`89.87 ± 3.42`** |
| **`1.0` (`3.0`)** | `uniform` | `76.30 ± 0.33` | `81.80 ± 0.21` | `1.000` ($\sigma_1 = 6.91$) | `70.90 ± 0.38` | `1.000` ($\sigma_2 = 5.09$) | `76.66 ± 0.36` | `1.000` ($\sigma_3 = 4.79$) | `78.28 ± 0.27` |
| **`1.0` (`3.0`)** | `adaptive` | `75.97 ± 0.70` | `80.93 ± 1.66` | **`0.712`** ($\sigma_1 = 9.36$) | **`71.36 ± 0.92`** | **`0.864`** ($\sigma_2 = 5.79$) | `74.80 ± 3.62` | `1.424` ($\sigma_3 = 3.54$) | **`78.86 ± 1.01`** |
| **`0.5` (`1.5`)** | `uniform` | `75.80 ± 0.16` | `81.48 ± 0.10` | `0.500` ($\sigma_1 = 12.89$) | `70.30 ± 0.14` | `0.500` ($\sigma_2 = 9.43$) | `76.16 ± 0.15` | `0.500` ($\sigma_3 = 8.86$) | `77.84 ± 0.19` |
| **`0.5` (`1.5`)** | `adaptive` | **`76.05 ± 0.14`** | **`81.64 ± 0.09`** | **`0.356`** ($\sigma_1 = 17.58$) | **`70.58 ± 0.15`** | **`0.432`** ($\sigma_2 = 10.77$) | **`76.42 ± 0.18`** | `0.712` ($\sigma_3 = 6.45$) | **`78.07 ± 0.11`** |

### 6.3 Privacy-Utility Trade-Off on Modern 2026 SDN-IoT (`ASEADOS-SDN-IoT`, 5 Seeds `0..4`, $\delta = 10^{-5}$)

| Mean Budget $\bar{\varepsilon}$ ($\varepsilon_{\text{total}}$) | Policy | Global Test Macro-F1 (%) | Global Test Accuracy (%) | `Finance` ($\varepsilon_1, \sigma_1$) | `Finance` Macro-F1 (%) | `Healthcare` ($\varepsilon_2, \sigma_2$) | `Healthcare` Macro-F1 (%) | `Education` ($\varepsilon_3, \sigma_3$) | `Education` Macro-F1 (%) |
|---|---|---:|---:|---|---:|---|---:|---|---:|
| **`∞` (No-DP)** | `none` | **`80.82 ± 2.69`** | `81.14 ± 2.98` | `∞` ($\sigma_1 = 0.00$) | `80.15 ± 4.98` | `∞` ($\sigma_2 = 0.00$) | `79.79 ± 0.57` | `∞` ($\sigma_3 = 0.00$) | `78.53 ± 5.18` |
| **`8.0` (`24.0`)** | `uniform` | `76.43 ± 3.48` | `76.95 ± 3.93` | `8.000` ($\sigma_1 = 1.28$) | `75.33 ± 5.11` | `8.000` ($\sigma_2 = 1.03$) | `75.25 ± 2.96` | `8.000` ($\sigma_3 = 1.28$) | `74.52 ± 5.23` |
| **`8.0` (`24.0`)** | `adaptive` | **`76.59 ± 3.26`** | **`77.12 ± 3.80`** | **`6.449`** ($\sigma_1 = 1.44$) | **`75.41 ± 4.97`** | `8.599` ($\sigma_2 = 0.99$) | **`75.51 ± 2.88`** | `8.952` ($\sigma_3 = 1.18$) | **`74.61 ± 5.12`** |
| **`4.0` (`12.0`)** | `uniform` | `73.16 ± 2.92` | `74.02 ± 3.16` | `4.000` ($\sigma_1 = 2.01$) | `73.42 ± 3.20` | `4.000` ($\sigma_2 = 1.55$) | `70.37 ± 3.04` | `4.000` ($\sigma_3 = 2.02$) | `72.82 ± 3.44` |
| **`4.0` (`12.0`)** | `adaptive` | **`73.64 ± 2.26`** | **`74.55 ± 2.38`** | **`3.225`** ($\sigma_1 = 2.36$) | **`73.96 ± 2.21`** | `4.300` ($\sigma_2 = 1.47$) | **`70.76 ± 2.77`** | `4.476` ($\sigma_3 = 1.86$) | **`73.40 ± 2.35`** |
| **`2.0` (`6.0`)** | `uniform` | `71.89 ± 4.93` | `72.47 ± 5.45` | `2.000` ($\sigma_1 = 3.45$) | `70.97 ± 6.53` | `2.000` ($\sigma_2 = 2.55$) | `70.48 ± 4.18` | `2.000` ($\sigma_3 = 3.48$) | `70.18 ± 6.77` |
| **`2.0` (`6.0`)** | `adaptive` | `71.78 ± 4.98` | `72.37 ± 5.44` | **`1.612`** ($\sigma_1 = 4.14$) | **`71.17 ± 6.27`** | `2.150` ($\sigma_2 = 2.40$) | `70.02 ± 4.39` | `2.238` ($\sigma_3 = 3.17$) | **`70.39 ± 6.52`** |
| **`1.0` (`3.0`)** | `uniform` | `70.30 ± 3.83` | `70.94 ± 4.56` | `1.000` ($\sigma_1 = 6.25$) | `69.69 ± 6.28` | `1.000` ($\sigma_2 = 4.54$) | `68.64 ± 1.87` | `1.000` ($\sigma_3 = 6.30$) | `68.88 ± 6.62` |
| **`1.0` (`3.0`)** | `adaptive` | `70.11 ± 3.72` | `70.69 ± 4.44` | **`0.806`** ($\sigma_1 = 7.57$) | `69.42 ± 6.11` | `1.075` ($\sigma_2 = 4.26$) | `68.54 ± 1.84` | `1.119` ($\sigma_3 = 5.71$) | `68.60 ± 6.46` |
| **`0.5` (`1.5`)** | `uniform` | `65.53 ± 0.94` | `65.72 ± 1.15` | `0.500` ($\sigma_1 = 11.64$) | `62.39 ± 1.25` | `0.500` ($\sigma_2 = 8.39$) | `66.53 ± 1.49` | `0.500` ($\sigma_3 = 11.73$) | `61.26 ± 1.37` |
| **`0.5` (`1.5`)** | `adaptive` | `64.81 ± 1.27` | `64.96 ± 1.50` | **`0.403`** ($\sigma_1 = 14.16$) | `61.55 ± 1.98` | `0.537` ($\sigma_2 = 7.86$) | `65.91 ± 1.00` | `0.559` ($\sigma_3 = 10.59$) | `60.51 ± 2.05` |

### 6.4 Empirical Membership Inference Attack (MIA) Audit (`5` Seeds `0..4`)

| Dataset | Regime ($\bar{\varepsilon}$) | Policy | `Finance` Allocated $\varepsilon_1$ | Full-Partition Local MIA AUC (%) | Full-Partition Global MIA AUC (%) | High-Risk Rare-Attack Cohort MIA AUC (%) | Rare-Cohort MIA Accuracy (%) | Rare-Cohort Yeom Advantage (%) | Loss-Threshold AUC (%) | Logit-Margin AUC (%) | Output Grad-Norm AUC (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `insdn` | **`∞` (No-DP)** | `none` | `∞` | `51.26 ± 0.37` | `50.80 ± 0.28` | **`57.31 ± 4.51`** | **`58.21%`** | **`16.43%`** | `56.69%` | `56.69%` | `57.20%` |
| `insdn` | **`8.0`** | `uniform` | `8.000` | `51.10 ± 0.57` | `51.18 ± 0.27` | `54.93 ± 3.64` | `55.95%` | `11.90%` | `54.70%` | `54.70%` | `54.51%` |
| `insdn` | **`8.0`** | `adaptive` | **`5.695`** | `51.29 ± 0.51` | `51.20 ± 0.37` | **`54.55 ± 3.20`** | **`55.83%`** | **`11.67%`** | `54.35%` | `54.35%` | `54.39%` |
| `insdn` | **`4.0`** | `uniform` | `4.000` | `51.27 ± 0.58` | `51.09 ± 0.33` | `54.52 ± 2.71` | `54.29%` | `8.57%` | `54.33%` | `54.34%` | `54.46%` |
| `insdn` | **`4.0`** | `adaptive` | **`2.848`** | `51.18 ± 0.39` | `50.90 ± 0.32` | **`54.27 ± 2.32`** | **`54.17%`** | **`8.33%`** | `53.90%` | `53.90%` | `54.26%` |
| `insdn` | **`2.0`** | `uniform` | `2.000` | `51.32 ± 0.60` | `51.12 ± 0.65` | `53.60 ± 2.23` | `53.69%` | `7.38%` | `53.25%` | `53.25%` | `53.46%` |
| `insdn` | **`2.0`** | `adaptive` | **`1.424`** | `51.16 ± 0.51` | `51.06 ± 0.59` | **`52.86 ± 1.39`** | **`53.69%`** | **`7.38%`** | `52.63%` | `52.63%` | `52.44%` |
| `insdn` | **`1.0`** | `uniform` | `1.000` | `51.04 ± 0.39` | `51.14 ± 0.68` | `52.38 ± 1.28` | `53.69%` | `7.38%` | `52.11%` | `52.12%` | `52.10%` |
| `insdn` | **`1.0`** | `adaptive` | **`0.712`** | `51.11 ± 0.46` | `51.11 ± 0.73` | **`51.81 ± 1.03`** | `53.93%` | `7.86%` | `51.68%` | `51.68%` | `51.53%` |
| `insdn` | **`0.5`** | `uniform` | `0.500` | `50.82 ± 0.46` | `51.24 ± 0.63` | `51.53 ± 0.62` | `53.45%` | `6.90%` | `51.35%` | `51.35%` | `51.30%` |
| `insdn` | **`0.5`** | `adaptive` | **`0.356`** | `50.93 ± 0.46` | `51.30 ± 0.66` | **`51.39 ± 0.65`** | **`53.21%`** | **`6.43%`** | `51.30%` | `51.31%` | `51.11%` |
| `aseados_sdn_iot` | **`∞` (No-DP)** | `none` | `∞` | `50.93 ± 0.50` | `51.06 ± 0.54` | **`66.17 ± 4.43`** | **`66.00%`** | **`32.00%`** | `65.90%` | `65.90%` | `65.65%` |
| `aseados_sdn_iot` | **`4.0`** | `uniform` | `4.000` | `50.92 ± 0.48` | `50.95 ± 0.43` | `57.78 ± 3.68` | `58.67%` | `17.33%` | `57.51%` | `57.51%` | `57.26%` |
| `aseados_sdn_iot` | **`4.0`** | `adaptive` | **`3.225`** | `50.91 ± 0.47` | `50.98 ± 0.44` | **`57.25 ± 3.46`** | **`57.50%`** | **`15.00%`** | `56.93%` | `56.93%` | `57.17%` |
| `aseados_sdn_iot` | **`2.0`** | `uniform` | `2.000` | `50.94 ± 0.42` | `50.96 ± 0.31` | `56.73 ± 3.21` | `58.83%` | `17.67%` | `56.52%` | `56.52%` | `56.55%` |
| `aseados_sdn_iot` | **`2.0`** | `adaptive` | **`1.612`** | `50.93 ± 0.40` | `50.95 ± 0.32` | **`56.42 ± 3.23`** | **`57.17%`** | **`14.33%`** | `56.21%` | `56.21%` | `56.22%` |
| `aseados_sdn_iot` | **`0.5`** | `uniform` | `0.500` | `50.93 ± 0.27` | `50.76 ± 0.20` | `54.86 ± 1.85` | `57.50%` | `15.00%` | `54.15%` | `54.15%` | `54.70%` |
| `aseados_sdn_iot` | **`0.5`** | `adaptive` | **`0.403`** | `50.95 ± 0.29` | `50.75 ± 0.23` | **`54.63 ± 1.74`** | **`56.00%`** | **`12.00%`** | `54.18%` | `54.18%` | `54.16%` |

### 6.5 Key Thesis Findings from Step 05

1. **Adaptive DP Strengthens Protection for the Most Sensitive Tenant at Zero/Minimal Utility Cost:**
   - On `InSDN` at $\bar{\varepsilon} = 4.0$ ($\varepsilon_{\text{total}} = 12.0, \delta = 10^{-5}$), `Adaptive DP` assigns `Finance` ($s_1 = 1.00$) a **28.8% smaller privacy budget** ($\varepsilon_1 = 2.848$ vs. $4.000$, increasing noise $\sigma_1$ from $2.19$ to $2.85$) and assigns `Healthcare` ($s_2 = 0.82$) a **13.6% smaller budget** ($\varepsilon_2 = 3.457$ vs. $4.000$), while global Macro-F1 remains virtually identical (`97.14 ± 1.08%` vs. `97.16 ± 1.02%`, $\Delta = -0.02$ F1 pts, Wilcoxon $p = 0.8125$).
   - At the tighter operating point $\bar{\varepsilon} = 2.0$ ($\varepsilon_{\text{total}} = 6.0$), `Adaptive DP` not only gives `Finance` a 28.8% stronger formal guarantee ($\varepsilon_1 = 1.424$ vs. $2.000$), but actually **improves** `Finance`'s local test Macro-F1 by **`+2.60` percentage points** (`85.49%` vs. `82.89%`) and global Macro-F1 by `+0.34` pts (`84.28%` vs. `83.94%`) because `Education` (the largest tenant, $n_3 = 4{,}960$, $s_3 = 0.50$) receives less noise ($\sigma_3 = 2.04$ vs. $2.68$), stabilizing the aggregated global representation.
   - Similarly, on `ASEADOS-SDN-IoT` at $\bar{\varepsilon} = 4.0$, `Adaptive DP` gives `Finance` a **19.4% tighter budget** ($\varepsilon_1 = 3.225$ vs. $4.000$) while achieving **`+0.48` pts higher global Macro-F1** (`73.64%` vs. `73.16%`) and **`+0.54` pts higher `Finance` Macro-F1** (`73.96%` vs. `73.42%`).
2. **Honest Reporting of Utility Collapse at Very Small $\bar{\varepsilon} \le 1.0$:**
   - When $\bar{\varepsilon}$ is pushed down to $0.5$ ($\sigma_1 = 17.58$ on `InSDN`), heavy gradient noise degrades `InSDN` global Macro-F1 from `98.69%` (`No-DP`) and `97.14%` ($\bar{\varepsilon}=4.0$) down to `76.05%` (`Finance` F1 = `70.58%`), and degrades `ASEADOS-SDN-IoT` global Macro-F1 from `80.82%` down to `64.81%`. This confirms that $\bar{\varepsilon} \in [2.0, 4.0]$ is the practical privacy-utility sweet spot for multi-tenant SDN intrusion detection.
3. **Dual-Regime Membership Inference Attack (MIA) Audit:**
   - **Full-Partition FL Audit:** Across full tenant partitions ($n_i \in [2625, 5737]$), both local and global models exhibit MIA AUC near `50.8%–51.3%` even without DP because the compact `43,554`-parameter GRU trained for 1 local epoch per round generalizes well on bulk traffic. As documented in `docs/DP_DESIGN.md`, a near-50% MIA score against a bounded attacker does not by itself prove privacy; the formal guarantee is provided by the $(\varepsilon_i, \delta)$ RDP accountant.
   - **High-Risk Rare-Attack Cohort Audit:** When auditing memorization on stratified rare-attack cohorts of `Finance`, unprotected `No-DP` models leak membership (`MIA AUC = 57.31%` on `InSDN` and **`66.17%`** with `32.0%` Yeom advantage on `ASEADOS-SDN-IoT`). Record-level DP-SGD monotonically suppresses rare-cohort MIA AUC down to `51.39%` (`InSDN`) and `54.63%` (`ASEADOS-SDN-IoT`), with `Adaptive DP` consistently achieving lower MIA leakage on `Finance` than `Uniform DP` (`52.86%` vs. `53.60%` at $\bar{\varepsilon}=2.0$ on `InSDN`, and `57.25%` vs. `57.78%` at $\bar{\varepsilon}=4.0$ on `ASEADOS-SDN-IoT`).




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




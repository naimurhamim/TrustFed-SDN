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

- **Base Commit:** `067df86` | **Raw Summary Artifact:** `results/00_environment/summary.json` (probed `2026-10-28T11:20:18Z`)
- **Environment Config:** `pyproject.toml`, `docs/ENVIRONMENT.md`

#### 1. Experimental Host Hardware & Software Specification (Paper "Experimental Setup" Table)

| Component | Measured Specification | Source |
|---|---|---|
| Host OS | Windows 11 Home (`10.0.26300`, `AMD64`) | `results/00_environment/summary.json` |
| CPU | 12th Gen Intel Core i7-12700H (14 physical cores, 20 logical threads) | `results/00_environment/summary.json` |
| System RAM | `15.67 GB` total (`3.66 GB` free during active desktop/IDE probe) | `results/00_environment/summary.json` |
| Storage (Workspace `D:\`) | `475.4 GB` free of `953.9 GB` NTFS | `results/00_environment/summary.json` |
| Accelerator GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU (`4096 MiB` VRAM, driver `610.88`) | `results/00_environment/summary.json` |
| Hypervisor | VMware Workstation `17.6.0` (existing guest: Kali Linux 2025.2, `4 GB` RAM) | `docs/ENVIRONMENT.md` |
| ML / FL Python Runtime | Python `3.12.7` in project-local `.venv` | `results/00_environment/summary.json` |
| Core Library Versions | PyTorch `2.14.1`, scikit-learn `1.9.1`, NumPy `2.5.4`, pandas `3.1.0`, SciPy `1.18.1`, cryptography `50.0.2` | `results/00_environment/summary.json` |

#### 2. Testbed Safety & Isolation Baseline

| Parameter | Value / Status | Notes |
|---|---|---|
| Isolated Testbed Supernet | `10.10.0.0/16` | Strictly private virtual testbed boundary |
| Tenant Subnets (3 tenants) | `10.10.1.0/24` (Finance), `10.10.2.0/24` (Healthcare), `10.10.3.0/24` (Education) | Disjointness & RFC1918 privacy verified by `tests/test_smoke.py` |
| Quality Check (`make check`) | `3 passed` (`pytest`), `0` lint/format errors (`ruff` + `black`) | Verified on Windows Python `3.12.7` |
| Linux Mininet+OVS+RYU `pingall` | `unverified` on Windows host (`DEV-001`) | Automated script `scripts/smoke_test_linux.sh` + Ubuntu 22.04 VM fallback documented in `docs/smoke_test/README.md` |
| Proposed Step 07 Isolation Level | **L1** (Ubuntu 22.04 Server VM, 4 GB RAM) + optional **Mixed L3** (Kali VM for Tenant 1) | Proposed based on `15.67 GB` host RAM constraint; awaiting explicit user approval |



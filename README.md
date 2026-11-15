# TrustFed-SDN

**Privacy-Preserving, Byzantine-Robust, and Cryptographically Verifiable Federated Intrusion Detection for Multi-Tenant Software-Defined Networks**

[![Release](https://img.shields.io/badge/release-v1.0.0-E8123C.svg)](https://github.com/naimurhamim/TrustFed-SDN/releases/tag/v1.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12-38BDF8.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981.svg)](LICENSE)
[![Companion Monograph](https://img.shields.io/badge/Monograph-trustfed.naimurrashid.dev-FF7A1A.svg)](https://trustfed.naimurrashid.dev/)

---

## Executive Summary & Headline Empirical Results

Multi-tenant SDN cloud fabrics expose critical east-west and north-south telemetry. **TrustFed-SDN** enables independent tenant networks to collaboratively train a compact sequential GRU intrusion detector without sharing raw flows, leaking minority-tenant membership, falling victim to poisoned tenant updates, or trusting an unverified aggregation server.

All headline numbers below come directly from verified 5-seed runs in [`results/final/key_numbers.json`](results/final/key_numbers.json):

| Capability | Empirical Metric | Measured Result (`results/final/key_numbers.json`) |
| :--- | :--- | :--- |
| **Centralized & Clean FL Utility** | Centralized GRU vs. FedAvg (InSDN, Natural `N=3`) | `99.74 ± 0.05%` vs. `99.13 ± 0.06%` Macro-F1 |
| **Byzantine Poisoning Robustness** | 20% Scaling Attack in Full Stack (`R4` vs. Plain `R1`) | `75.97 ± 0.29%` vs. `21.73 ± 8.70%` Macro-F1 (`100%` mal. rejected) |
| **Adaptive R&eacute;nyi DP Privacy** | Membership Inference AUC (`eps_bar = 4.0`, InSDN) | `51.18%` full partition · `54.27%` rare Finance cohort |
| **Cryptographic SecAgg Integrity** | Tampered Proof Detection / Honest False Alarms | `100.0%` (`1,500/1,500`) detected · `0` (`0/300`) false alarms |
| **48-Bit Packed SecAgg Wire Cost** | Round-Trip Wire Overhead vs. Plain FedAvg | `1.256x` (`N=3`) · `1.270x` (`N=10`) · `1.290x` (`N=20`) |
| **Live OVS + OS-Ken Enforcement** | OpenFlow `priority=300` Rule Install &amp; Live FBR | `27.38 ms` install · `100%` attack block · `0%` benign block · `2.70%` calibrated FBR |

---

## Four Core Architectural Pillars

1. **Sample-Aware Adaptive Differential Privacy (`trustfed/privacy/`):** Allocates per-tenant DP-SGD noise calibrated to sample volume so rare tenant cohorts (e.g., InSDN `Finance`) resist Membership Inference Attacks near the 50% random-guess floor.
2. **Two-Stage Byzantine Defense (`trustfed/aggregation/`):** Filters poisoned label-flip, backdoor-trigger, and scaling updates via pairwise cosine clustering (`cutoff=0.5`) and coordinate-wise trimmed-mean / median aggregation (`beta=0.2`).
3. **48-Bit Packed Verifiable SecAgg (`trustfed/verify/`):** Fixed-point `Q=20` quantization packed into 48-bit words, blinded by X25519 + SHAKE-256 pairwise masks modulo `2^61 - 1`, and authenticated with canonical SHA-256 Merkle commitments + Ed25519 server/client transcript signatures (`1,781 B` proof overhead).
4. **Closed-Loop Live SDN Mitigation (`testbed/`):** Live Ubuntu 22.04 L1 Linux namespaces (`t1..t3`) and per-tenant Open vSwitch switches (`br_t1..br_t3`) driven by `os-ken==3.0.1` OpenFlow 1.3 controllers with automatic TTL flow-rule expiry (`486.714 ms` end-to-end 7-stage round latency).

---

## Quick Start & Verification Commands

### Windows / Cross-Platform Verification (`.\make.cmd`)
```powershell
git clone https://github.com/naimurhamim/TrustFed-SDN.git
cd TrustFed-SDN
py -3.12 -m venv .venv
.\make.cmd install          # Editable install + pytest, ruff, black
.\make.cmd check            # Lint + full pytest suite + import check
.\make.cmd reproduce-quick  # Fast Step 09 regeneration + verification
.\make.cmd qr               # Generate & decode-verify SVG/PNG/PDF QR assets
.\make.cmd site-preview     # Browse the 10-chapter Research Monograph at :8080
```

### Ubuntu 22.04 Live Testbed (`make`)
```bash
source .venv-ryu/bin/activate
make demo                   # Boots OVS br_t1..br_t3, t1..t3 namespaces & OS-Ken 1.3 loop
make reproduce              # Full 5-seed E1-E10 evaluation + 8 PNG/PDF figures + LaTeX tables
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for raw dataset placement (`InSDN`, `ASEADOS-SDN-IoT`, `CICIDS2017`) and full reproduction instructions.

---

## Repository Structure

```text
TrustFed-SDN/
├── trustfed/          # Core ML, FL, Adaptive DP, Robust Agg & 48-bit Verifiable SecAgg
├── testbed/           # Ubuntu L1 namespaces, OVS mitigator, OS-Ken 3.0.1 apps & HTTP RPC
├── sim/               # Multi-seed simulation experiment runners
├── scripts/           # Step 01-09 CLI pipelines + ISO/IEC 18004 QR generator (make_qr.py)
├── tests/             # Unit, integration, crypto, live-testbed & release pytest suites
├── configs/           # Reproducible YAML experiment configurations
├── docs/              # Architecture, reproduction guide, final audit, deviations & checklists
├── results/           # Summarized JSON/CSV outputs (incl. results/final/key_numbers.json)
├── figures/ & tables/ # 8 publication-grade PNG/PDF plots & LaTeX tables
├── site/              # Self-contained 10-chapter Interactive Research Monograph + QR assets
├── LICENSE            # MIT License
└── CITATION.cff       # Machine-readable academic citation metadata (v1.0.0)
```

---

## Documentation Index

- **System Design & Threat Model:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Step-by-Step Reproducibility Guide:** [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
- **Final Release & Consistency Audit:** [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md)
- **Documented Deviations Log (`DEV-001`–`DEV-004`):** [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)
- **Live Interactive Monograph:** [`https://trustfed.naimurrashid.dev/`](https://trustfed.naimurrashid.dev/) (`site/README.md`)

---

## Citation & License

Released under the [MIT License](LICENSE). If you build upon this codebase, evaluation suite, or live SDN controller integration, please cite [`CITATION.cff`](CITATION.cff):

```bibtex
@misc{bhowmik2026trustfedsdn,
  title        = {TrustFed-SDN: Privacy-Preserving and Verifiable Federated Learning for Multi-Tenant SDN Security},
  author       = {Bhowmik, Pritom Kumar and Rashid, MD Naimur and Das, Saurav Chandra},
  year         = {2026},
  month        = nov,
  institution  = {Department of IoT and Robotics Engineering, University of Frontier Technology, Bangladesh},
  version      = {v1.0.0},
  url          = {https://github.com/naimurhamim/TrustFed-SDN},
  note         = {Companion Monograph: https://trustfed.naimurrashid.dev/}
}
```

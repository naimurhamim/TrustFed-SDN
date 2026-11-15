# Reproducibility & Verification Guide (`TrustFed-SDN` v1.0)

This guide gives step-by-step instructions to recreate our Python environments, acquire datasets under upstream licenses, execute unit tests, regenerate all Step 09 empirical figures/tables, generate & decode QR assets, and boot the Ubuntu live SDN testbed.

---

## 1. Environments

### A. Windows 10/11 (CUDA ML Training, Crypto Verification & Quick Reproduction)
Tested on Python `3.12` (`.venv`):

```powershell
git clone https://github.com/naimurhamim/TrustFed-SDN.git
cd TrustFed-SDN
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\make.cmd install          # installs trustfed_sdn in editable mode + dev deps (pytest, ruff, black)
.\make.cmd check            # ruff lint + black check + pytest suite + import check
```

> **Note:** On Windows without GNU `make`, `.\make.cmd <target>` provides native PowerShell/CMD fallback to `.venv\Scripts\python.exe`.

### B. Ubuntu 22.04 VMware VM (Live OVS + OS-Ken SDN Testbed)
Requires Open vSwitch (`openvswitch-switch`), `iproute2`, `tcpdump`, and `os-ken==3.0.1` inside `.venv-ryu`:

```bash
sudo apt update && sudo apt install -y openvswitch-switch iproute2 tcpdump python3-venv
python3 -m venv .venv-ryu
source .venv-ryu/bin/activate
pip install -e ".[dev]" os-ken==3.0.1
```

---

## 2. Dataset Acquisition & Placement

Raw benchmark CSVs are **excluded from git** (`data/` is gitignored) to respect upstream redistribution terms. Download directly from official academic sources and place into `TrustFed-SDN/data/raw/`:

| Dataset | Upstream Source | Cleaned Size (Step 01) | Role |
| :--- | :--- | :--- | :--- |
| **InSDN** | Elsayed et al., University College Dublin SDN IDS Dataset | `343,889 -> 184,855` rows (`19` features) | Primary native SDN benchmark |
| **ASEADOS-SDN-IoT (2026)** | Modern SDN-IoT multi-attack dataset | `457,044 -> 426,111` rows (`12` features) | High-volume Non-IID SDN-IoT benchmark |
| **CICIDS2017** | Canadian Institute for Cybersecurity | `2,830,743 -> 2,232,894` rows (`22` features) | Cross-domain baseline verification |

Run preprocessing & leakage checks (Step 01) once raw CSVs are placed:
```powershell
.\.venv\Scripts\python.exe scripts\run_step01_data.py
```

---

## 3. Standard Commands & Expected Runtimes

| Command (`Windows` / `Linux`) | Action | Expected Runtime | Output Location |
| :--- | :--- | :--- | :--- |
| `.\make.cmd check` / `make check` | Runs `ruff`, `black --check`, `pytest` suite & import check | ~5–15 sec | Terminal stdout (`exit 0`) |
| `.\make.cmd reproduce-quick` | Runs fast end-to-end E1–E10 verification & rebuilds plots/tables from cached/smoke outputs | ~5–15 sec | `results/final/`, `figures/`, `tables/` |
| `.\make.cmd reproduce` | Full 5-seed E1–E10 evaluation sweep, paired Wilcoxon tests, 8 PNG/PDF plots & LaTeX tables | ~25–90 min (GPU dependent) | `results/final/key_numbers.json`, `figures/fig01..08.*`, `tables/` |
| `.\make.cmd qr` | Generates SVG, 1080px PNG, PDF QR for `https://trustfed.naimurrashid.dev/` & runs independent RS decoder check | ~1–2 sec | `figures/qr/`, `../07_Website/trustfed-sdn-site/images/qr/` |
| `.\make.cmd site-preview` | Serves the 10-chapter Research Monograph locally at `http://127.0.0.1:8080/` | Interactive | Local HTTP server (`../07_Website/trustfed-sdn-site/`) |
| `make demo` *(Ubuntu root)* | Boots OVS bridges `br_t1..br_t3`, namespaces `t1..t3`, OS-Ken 1.3 controller & closed-loop FL mitigation demo | ~30–60 sec | `results/08_live/summary.json` |

---

## 4. Verifying Clean-Tree Reproducibility

Every committed thesis result in `results/final/`, `figures/`, and `tables/` can be cross-checked deterministically without touching raw datasets:

```powershell
.\make.cmd format
.\make.cmd lint
.\make.cmd reproduce-quick
.\make.cmd check
```

All assertions in `tests/test_final_evaluation.py` validate `results/final/key_numbers.json`, the 8 PNG/PDF figures, LaTeX `tables/`, and the Step 10 QR + documentation + website sync checks.

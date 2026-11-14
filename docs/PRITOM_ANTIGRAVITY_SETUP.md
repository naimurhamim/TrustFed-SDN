# TrustFed-SDN — One-Command Antigravity Onboarding & Setup Guide (`@Pritom`)

> **Project:** **TrustFed-SDN** — Privacy-Preserving, Verifiable, Poisoning-Robust Federated Learning for Cross-Tenant Threat Detection in SDN-Enabled Cloud Networks  
> **GitHub Repository (Public):** `https://github.com/naimurhamim/TrustFed-SDN.git`  
> **Team:** `@Naimur` (MD Naimur Rashid) & `@Pritom` (Pritom Kumar Bhowmik)

---

## Quick Start for `@Pritom` (Only 3 Steps)

1. **Accept GitHub Collaborator Invite:** Open `https://github.com/naimurhamim/TrustFed-SDN/invitations` in your browser and accept `@Naimur`'s invitation.
2. **Open Workspace in Antigravity:** Put this file (`PRITOM_ANTIGRAVITY_SETUP.md`) inside an empty folder where you want the thesis code stored (for example, `D:\Thesis\` or `C:\Projects\Thesis\`), and open that folder in **Google Antigravity**.
3. **Paste This Single Command in Antigravity Chat:**
   ```text
   Read PRITOM_ANTIGRAVITY_SETUP.md and autonomously set up the TrustFed-SDN project on my computer step by step, connect to the project checklist, and tell me what our next task is.
   ```

---

## Autonomous Execution Protocol for Antigravity Agent (on `@Pritom`'s Machine)

> **AGENT SYSTEM DIRECTIVE (SETUP-MATCH GATE):**
> Whenever `@Pritom` asks you to read this file or continue thesis work:
> 1. **Verify Setup First:** Check whether the local machine already matches **Phases 1 through 5** below (`TrustFed-SDN/` cloned and synced with `git pull --tags`, Git identity configured, `.venv` active with all dependencies, `.\make.cmd check` passing, and `.\make.cmd status` clean).
> 2. **If Setup Matches:** Do **not** redo completed setup steps—proceed directly to the next active task in `PROGRESS.md` and `docs/TEAM_CHECKLIST.md`.
> 3. **If Setup Does NOT Match:** Stop any new step work and **complete/fix the setup first** following Phases 1 through 5 in strict order before doing anything else.
> 4. Always obey `TrustFed-SDN/AGENTS.md`: never skip verification checks, never invent metrics, and never advance past an unapproved step.

### Agent Identity & Language Rules
- You are assisting **`@Pritom` (Pritom Kumar Bhowmik)**, collaborating with **`@Naimur` (MD Naimur Rashid)** via GitHub (`main` branch).
- **Chat language:** Communicate with `@Pritom` in **Bengali (Bengali script)** with technical terms in English (keep responses concise and clear), unless `@Pritom` explicitly asks to chat in English.
- **Repository file language:** Keep **100% of code, comments, docstrings, commit messages, logs, and Markdown files in English only** (zero Bangla or Banglish inside any file).

---

### Phase 1 — Pre-Flight System & Prerequisite Check

1. **Inspect OS & Free Disk Space:**
   - Detect OS (Windows PowerShell vs Linux/macOS) and check free disk space on the workspace drive (`>= 15 GB` free recommended for virtual environments, raw datasets, and processed splits).
2. **Verify `git` Installation:**
   - Run `git --version`. If Git is missing on Windows, ask `@Pritom` for permission to run `winget install --id Git.Git -e --source winget`.
3. **Verify `python` (`3.10`–`3.12`, with `3.12` Preferred):**
   - On Windows, check `py -0` and `py -3.12 --version` (or `python --version`).
   - `@Naimur`'s reference environment uses **Python `3.12.7`**. Any Python `3.11.x` or `3.12.x` works cleanly (`requires-python = ">=3.10"` in `pyproject.toml`; prefer `3.12` over `3.13+` for full PyTorch wheel compatibility).
   - If no compatible Python (`3.10`–`3.12`) is installed, ask `@Pritom` before installing Python 3.12 (`winget install --id Python.Python.3.12 -e --source winget`).
4. **Confirm `@Pritom`'s GitHub Identity Before Cloning:**
   - Check `git config --global user.name` and `git config --global user.email`.
   - Ask `@Pritom` for his **exact GitHub username** (and Git email if unset) so you can bind local credentials cleanly and prevent multi-account popup hangs on Windows.

---

### Phase 2 — Clone `TrustFed-SDN` & Bind GitHub Credentials (Auto-Pull / Auto-Push)

1. **Clone the Repository (or Pull if Already Cloned):**
   - Target directory: `./TrustFed-SDN` inside the current workspace folder.
   - If `./TrustFed-SDN/.git` does **not** exist yet, run:
     ```powershell
     git clone https://github.com/naimurhamim/TrustFed-SDN.git TrustFed-SDN
     ```
     *(Note: If Windows Git Credential Manager opens a one-time browser login window for `@Pritom`, let `@Pritom` complete that single login.)*
   - If `./TrustFed-SDN/.git` already exists, `cd TrustFed-SDN` and run `git pull --ff-only`.
2. **Configure Local Multi-Account Credential Binding (Prevents Popup Hangs):**
   - Inside `TrustFed-SDN/`, bind `@Pritom`'s GitHub username locally (writes only to `.git/config`, never committed, zero plain-text tokens stored):
     ```powershell
     cd TrustFed-SDN
     git config --local credential.https://github.com.username <PRITOM_GITHUB_USERNAME>
     ```
   - Set local `user.name` (`Pritom Kumar Bhowmik`) and `user.email` if global Git identity was empty.
   - Test non-interactive remote access (`$env:GCM_INTERACTIVE="never"; $env:GIT_TERMINAL_PROMPT="0"; git fetch origin`) to verify automated agent pulls and pushes work without any GUI account-picker popup.

---

### Phase 3 — Create Isolated `.venv` & Install Project Dependencies

All subsequent commands MUST run inside the `TrustFed-SDN/` directory.

1. **Create `.venv`:**
   - **Windows (PowerShell):**
     ```powershell
     py -3.12 -m venv .venv
     ```
     *(Fallback to `py -3.11 -m venv .venv` or `python -m venv .venv` if `3.12` is unavailable.)*
   - **Linux / WSL / macOS:**
     ```bash
     python3 -m venv .venv
     ```
2. **Upgrade `pip` & Install Editable Package + Dev Suite (`pyproject.toml`):**
   - **Windows (PowerShell):** *(Always invoke `python.exe -m pip` rather than `pip.exe` when upgrading pip on Windows)*
     ```powershell
     .\.venv\Scripts\python.exe -m pip install --upgrade pip
     .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
     ```
   - **Linux / WSL / macOS:**
     ```bash
     .venv/bin/python -m pip install --upgrade pip
     .venv/bin/python -m pip install -e ".[dev]"
     ```
   - Installed core + dev stack includes: `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `torch`, `PyYAML`, `matplotlib`, `psutil`, `cryptography`, `pytest`, `ruff`, `black`.

---

### Phase 4 — Run Code Quality, Smoke Tests & Local Hardware Probe

1. **Run the Full Verification Suite (`make check`):**
   - **Windows:** `.\make.cmd check` *(works out-of-the-box with or without GNU Make / MSYS2 installed!)*
   - **Linux / macOS:** `make check`
   - Confirm all 4 gates pass:
     - `ruff check trustfed scripts tests` -> `All checks passed!`
     - `black --check trustfed scripts tests` -> clean
     - `pytest` -> `100% passed`
     - `import trustfed` -> `0.1.0`
2. **Inspect `@Pritom`'s Hardware Without Overwriting `@Naimur`'s Primary Summary:**
   - Run `.\make.cmd probe` (or `.venv\Scripts\python.exe scripts/probe_hardware.py`) to inspect `@Pritom`'s CPU, RAM, disk free space, GPU/CUDA availability, and VMware/WSL2 status.
   - **Important:** Do **not** overwrite `results/00_environment/summary.json` (which records `@Naimur`'s primary host specs). If you want to record `@Pritom`'s hardware specs in the repo, write to `results/00_environment/summary_pritom.json` via `--json --out results/00_environment/summary_pritom.json`.

---

### Phase 5 — Load the Project Brain & Run Team Sync Checker (`make status`)

Read these repository documents in **strict order** (mandatory per `TrustFed-SDN/AGENTS.md`):

1. `TrustFed-SDN/AGENTS.md` — 12 non-negotiable engineering, scientific honesty, network safety (`10.10.0.0/16` only), paper-ready logging, and 2-person collaboration rules.
2. `TrustFed-SDN/PROJECT_PLAN.md` — Master research architecture, threat model, and 11-step execution plan (`Step 00` – `Step 10`).
3. `TrustFed-SDN/PROGRESS.md` — High-level milestone tracker, commit history, and blockers.
4. `TrustFed-SDN/docs/TEAM_CHECKLIST.md` — **Latest Handoff Note** banner at top + **63 granular `[ ]` / `[x]` sub-tasks** (`00.1` through `10.5`) showing exactly who completed what and what comes next.
5. `TrustFed-SDN/docs/TEAM_WORKFLOW.md` — Turn-by-turn collaboration workflow and dataset sharing rules.
6. `TrustFed-SDN/docs/USER_PREFERENCES.md` & `TrustFed-SDN/docs/ENVIRONMENT.md` — Agreed project decisions, hardware constraints, and isolated testbed CIDR ranges.
7. `TrustFed-SDN/docs/EXPERIMENT_SPEC.md`, `TrustFed-SDN/docs/DATASETS.md`, `TrustFed-SDN/docs/DEVIATIONS.md`, and `TrustFed-SDN/docs/EXPERIMENT_RESULTS_LOG.md`.
8. The **currently active step specification** inside `TrustFed-SDN/docs/steps/STEP_XX_*.md`.

Then immediately run the automated team sync checker:
```powershell
.\make.cmd status
```
*(or `.venv\Scripts\python.exe scripts/team_status.py --verify-hashes` once `data/MANIFEST.md` exists).*

Check all 4 sections of the `status` output:
- **`[GIT]`**: Must report `Sync state: UP TO DATE with GitHub`. If `BEHIND`, run `git pull --ff-only`.
- **`[ENV]`**: Must report all required packages installed (`OK`). If any new package was added by `@Naimur`, run `.\make.cmd install`.
- **`[DATA]`**: Raw datasets (`data/raw/insdn/`, `data/raw/aseados_sdn_iot/`, `data/raw/cicids2017/`) and processed arrays (`data/processed/`) are git-ignored because they exceed GitHub size limits. `data/MANIFEST.md` tracks all 14 raw files with byte sizes and SHA-256 digests:
  1. If `data/raw/aseados_sdn_iot/ASEADOS_SDN_IoT.csv` (`230.93 MB`) is missing, download it automatically via:
     `curl.exe -L --fail --retry 3 -o "data\raw\aseados_sdn_iot\ASEADOS_SDN_IoT.csv" "https://aseados.ucd.ie/datasets/SDN-IoT/ASEADOS_SDN_IoT.csv"`
  2. If `@Pritom` placed `InSDN_DatasetCSV.zip` in `data/raw/insdn/` or `MachineLearningCSV.zip` in `data/raw/cicids2017/` (from Google Drive) without unzipping, extract them into those folders automatically.
  3. If `data/processed/` arrays are not present on disk after `data/raw/` is verified (`14/14` in `.\make.cmd status`), run `.\.venv\Scripts\python.exe scripts/profile_dataset.py` once to generate `data/processed/`.
- **`[PLAN]`**: Read the **Latest Handoff Note** and the **Next Unchecked Sub-Tasks**.

---

### Phase 6 — Connect to the 11 Project Steps & Report Ready State to `@Pritom`

Use this step map throughout the project so every session connects cleanly to the exact specification file, checklist items, and runtime environment:

| Step | Specification File (`TrustFed-SDN/`) | Checklist IDs (`docs/TEAM_CHECKLIST.md`) | Execution Environment | Core Deliverables & Gate |
|---|---|---|---|---|
| **Step 00** | `docs/steps/STEP_00_environment_and_preferences.md` | `00.1` – `00.8` | Windows `.venv` + Ubuntu 22.04 VM | Repo skeleton, `.venv`, hardware probe, OpenFlow 1.3 `pingAll` (`0%` loss), team sync system -> tag `step-00-done` |
| **Step 01** | `docs/steps/STEP_01_data.md` | `01.1` – `01.9` | Windows / Linux `.venv` | `InSDN` + `CICIDS2017` `data/MANIFEST.md`, leakage-free cleaning, top-k features, non-IID/Dirichlet/IID tenant splits, per-tenant scaling -> tag `step-01-done` |
| **Step 02** | `docs/steps/STEP_02_centralized_baselines.md` | `02.1` – `02.6` | Windows / Linux `.venv` (CPU/GPU) | Unified `DNN`, `GRU`, `LSTM` + classical baselines, centralized vs local-only across 5 seeds, `docs/MODEL_DECISION.md` -> tag `step-02-done` |
| **Step 03** | `docs/steps/STEP_03_fl_core_and_fedavg.md` | `03.1` – `03.5` | Windows / Linux `.venv` (CPU/GPU) | `Client`, `Server`, pluggable `Aggregator`, weighted `FedAvg` across `N in {3,5,10}` & heterogeneity regimes (5 seeds) -> tag `step-03-done` |
| **Step 04** | `docs/steps/STEP_04_attacks_and_robust_aggregation.md` | `04.1` – `04.5` | Windows / Linux `.venv` (CPU/GPU) | Label-flip, backdoor, scaling/sign-flip attacks vs Trimmed Mean, Cosine Clustering, Static Weight (`0–30%` poisoned, 5 seeds) -> tag `step-04-done` |
| **Step 05** | `docs/steps/STEP_05_adaptive_dp.md` | `05.1` – `05.5` | Windows / Linux `.venv` (CPU/GPU) | L2 clipping, Gaussian mechanism, RDP accountant, uniform vs adaptive per-tenant DP (`epsilon` sweep) + MIA evaluation -> tag `step-05-done` |
| **Step 06** | `docs/steps/STEP_06_verifiable_secure_aggregation.md` | `06.1` – `06.5` | Windows / Linux `.venv` | Fixed-point quantization, pairwise masking, commitments/checksums, signatures, nonces, 4 tamper tests (`100%` detection, `0%` false alarm) -> tag `step-06-done` |
| **Step 07** | `docs/steps/STEP_07_sdn_testbed.md` | `07.1` – `07.5` | **Ubuntu 22.04 VM** (`Mininet + OVS + OS-Ken`) | Multi-tenant SDN topology (`10.10.0.0/16` ONLY), per-tenant isolation, flow extractor, guarded traffic gen, `data/live/` capture -> tag `step-07-done` |
| **Step 08** | `docs/steps/STEP_08_live_integration.md` | `08.1` – `08.5` | **Ubuntu 22.04 VM** (`Mininet + OVS + OS-Ken`) | Live FL clients + secure/robust/verifiable server, controller model deploy & automated OpenFlow drop-rule mitigation (`>= 10` live rounds) -> tag `step-08-done` |
| **Step 09** | `docs/steps/STEP_09_full_evaluation.md` | `09.1` – `09.5` | Windows / Linux `.venv` (+ VM replay) | Full ablation matrix (Rows `A`–`F`, 5 seeds), paired Wilcoxon tests, publication PDF/PNG figures, LaTeX tables, `make reproduce` -> tag `step-09-done` |
| **Step 10** | `docs/steps/STEP_10_website_and_packaging.md` | `10.1` – `10.5` | Windows / Linux `.venv` | Populate `site/` strictly from `results/` & `figures/` (`scripts/verify_site_numbers.py`), final `README.md`, `CITATION.cff`, `v1.0.0` release -> tag `step-10-done` |

> **Note on Ubuntu 22.04 SDN VM (`Steps 07–08`):**  
> Steps `01` through `06`, `09`, and `10` run natively on Windows (or Linux) inside `.venv` without needing a VM. Only **Steps `07` and `08`** require an Ubuntu 22.04 LTS environment (VMware/VirtualBox/native Linux) with `mininet`, `openvswitch-switch`, and `.venv-ryu` (`os-ken==3.0.1` — verified via `sudo bash scripts/smoke_test_linux.sh`).

#### Final Onboarding Action (Report to `@Pritom`)
Once Phases 1–5 succeed, send `@Pritom` a concise Bengali summary confirming:
1. Repository cloned and bound to `<PRITOM_GITHUB_USERNAME>` (auto-pull/push verified),
2. Python `.venv` created and `.\make.cmd check` passed (`ruff`, `black`, `pytest`, `import trustfed`),
3. Current project progress from `docs/TEAM_CHECKLIST.md` (**Latest Handoff Note** from `@Naimur`),
4. Any missing git-ignored datasets (if Step 01 has begun), and
5. **STOP and ask `@Pritom`** whether to start working on the next unchecked sub-task (following the `AGENTS.md` rule: ask any step questions -> post a max-10-line plan -> wait for `"go"`).

---

## Ongoing Daily Commands for `@Pritom` (Every Future Session)

- **When starting a work session**, tell Antigravity:
  ```text
  cd into TrustFed-SDN, run .\make.cmd status, pull any new GitHub changes from @Naimur, and show me the next unchecked sub-tasks in docs/TEAM_CHECKLIST.md.
  ```
- **When finishing a work session**, Antigravity MUST automatically:
  1. Run `.\make.cmd format` and `.\make.cmd check` (must pass with `0` errors),
  2. Mark completed sub-tasks `[x]` with `(@Pritom, YYYY-MM-DD)` and update **Latest Handoff Note** in `TrustFed-SDN/docs/TEAM_CHECKLIST.md`,
  3. Update `TrustFed-SDN/PROGRESS.md` (and `TrustFed-SDN/docs/EXPERIMENT_RESULTS_LOG.md` whenever experiments produced numbers),
  4. Commit in clean English and `git push` to `origin main` so `@Naimur` immediately sees the update on his next `.\make.cmd status`.

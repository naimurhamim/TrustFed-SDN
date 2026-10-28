# Plan Deviations Log

Every difference between `PROJECT_PLAN.md` / `docs/EXPERIMENT_SPEC.md` / `docs/steps/` and what was actually executed is recorded here with **what**, **why**, and **impact**.

---

## DEV-001 — Step 00 Linux Testbed Smoke Test Deferred to Linux VM Boot (Before Step 07)

- **Date:** 2026-10-28
- **Step:** Step 00 (`docs/steps/STEP_00_environment_and_preferences.md`, Task 4)
- **What:** The live `pingall` execution of Mininet + Open vSwitch + RYU was not executed during Step 00 on the Windows 11 host (`unverified` live run). Instead, the automated verification script `scripts/smoke_test_linux.sh` and full Ubuntu 22.04 Server VM setup & fallback documentation (`docs/smoke_test/README.md`) were created.
- **Why:** Development runs on Windows 11 Home, where Mininet and Open vSwitch kernel datapaths cannot run natively (and `STEP_00` forbids faking Mininet on Windows). On VMware Workstation 17.6.0, only a powered-off Kali Linux 2025.2 VM currently exists, and the user chose the recommended path: proceed immediately with dataset and FL algorithm steps (Steps 01–06) on Windows Python 3.12 while provisioning a clean Ubuntu 22.04 LTS Server VM before Step 07.
- **Impact:** Zero impact on Steps 01–06 (pure Python/PyTorch dataset and federated learning experiments). Before Step 07 begins, `sudo bash scripts/smoke_test_linux.sh` MUST be executed inside the Ubuntu 22.04 VM and its logs committed to `docs/smoke_test/` as an entry gate to Step 07.


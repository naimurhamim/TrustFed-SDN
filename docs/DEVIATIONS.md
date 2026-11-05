# Plan Deviations Log

Every difference between `PROJECT_PLAN.md` / `docs/EXPERIMENT_SPEC.md` / `docs/steps/` and what was actually executed is recorded here with **what**, **why**, and **impact**.

---

## DEV-001 — Use Maintained RYU Fork (`os-ken==3.0.1`) on Python 3.10+ in `.venv-ryu`

- **Date:** 2026-10-28
- **Step:** Step 00 (`docs/steps/STEP_00_environment_and_preferences.md`, Task 4)
- **What:** Inside the Ubuntu 22.04 LTS VM's dedicated SDN controller virtualenv (`.venv-ryu`), we use **`os-ken==3.0.1`** (`osken-manager`, `os_ken.base.app_manager.OSKenApp`) rather than legacy unmaintained `ryu==4.34`.
- **Why:** Ubuntu 22.04 LTS ships with `Python 3.10.12`. During Step 00 smoke testing on the Ubuntu 22.04 VM, legacy `ryu==4.34` + `eventlet==0.30.2` failed at startup because `collections.MutableMapping` and legacy `distutils`/`wsgi` hooks were removed in Python 3.10+. `PROJECT_PLAN.md` Section 8 explicitly specifies *"use a maintained fork"* as the approved fallback for RYU Python incompatibility. `os-ken` is OpenStack Neutron's officially maintained fork of RYU with the exact same OpenFlow 1.3 event/flow API (`EventOFPSwitchFeatures`, `EventOFPPacketIn`, `OFPFlowMod`, `OFPPacketOut`). Note that `os-ken` is pinned to `3.0.1` (`<4.0`) because `os-ken 4.x` removed the `osken-manager` CLI entry point.
- **Impact:** OpenFlow 1.3 switch control, flow-stats polling, and mitigation rule installation in Steps 07–08 work natively on Ubuntu 22.04 (`Python 3.10.12`) with `0%` packet loss verified in `docs/smoke_test/mininet_pingall.log`. RYU apps under `testbed/ryu_apps/` import from `os_ken.*` instead of `ryu.*`.

---

## DEV-002 — Include Official 2026 `ASEADOS-SDN-IoT` Dataset Alongside `InSDN` (2020) and `CICIDS2017`

- **Date:** 2026-11-05
- **Step:** Step 01 (`docs/steps/STEP_01_data.md`, Task 1)
- **What:** In addition to `InSDN` (2020) and `CICIDS2017` (2017), we include **`ASEADOS-SDN-IoT` (2026)** (`data/raw/aseados_sdn_iot/ASEADOS_SDN_IoT.csv`, `242,144,496` bytes, `457,044` flow records, `83` CICFlowMeter-compatible features) published by the same University College Dublin (UCD) ASEADOS Lab (`https://aseados.ucd.ie/datasets/SDN-IoT/`, DOI: `10.1016/j.iot.2026.101891`, *Elsevier Internet of Things*, 2026).
- **Why:** The user requested including a current 2026 SDN intrusion dataset to strengthen the 2026 thesis defense while keeping classic `InSDN` (for direct literature comparison) and `CICIDS2017` (supplementary benchmark).
- **Impact:** Strictly additive — all original `InSDN` and `CICIDS2017` requirements remain intact, and the data pipeline (`trustfed/data/`) additionally supports `aseados_sdn_iot` with full profiling, cleaning, feature selection, partitioning, and scaling.

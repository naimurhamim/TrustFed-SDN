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

---

## DEV-003 — HTTP/JSON + Binary Wire RPC Transport in Step 08 (Instead of Flower Wrapper)

- **Date:** 2026-11-14
- **Step:** Step 08 (`docs/steps/STEP_08_live_integration.md`, Ask the User Q1)
- **What:** For live multi-tenant FL orchestration in `testbed/fl/protocol_http.py`, we implemented a direct `ThreadingHTTPServer` + `LiveTenantHTTPClient` HTTP/JSON RPC transport carrying base64-encoded 48-bit packed masked tensors, SHA-256 Merkle proofs, and Ed25519 headers rather than wrapping `trustfed/` inside `flwr` (Flower).
- **Why:** `docs/steps/STEP_08_live_integration.md` explicitly allows *"Flower with custom strategies reusing the trustfed/ aggregation code, or a simple gRPC/HTTP service"*. Direct HTTP RPC avoids introducing heavy external Flower/gRPC runtime dependencies across both the Windows `.venv` (Python `3.12.7`) and Ubuntu VM `.venv-ryu` (Python `3.10.12`), preserves exact byte-for-byte wire accounting of `VerificationProof` (`proof_wire_bytes`) and `48-bit` packed payloads, and supports clean mid-round abort/retry (`/abort` and HTTP `409` on incomplete cohorts) as required by Step 08.
- **Impact:** All core FL, Adaptive DP-SGD, CosineClustering poisoning filtering, X25519/SHAKE-256 masking, Merkle commitments, and Ed25519 verification algorithms are reused verbatim from `trustfed/` over real localhost socket RPC with exact per-stage latency and byte telemetry.

---

## DEV-004 — L1 Namespace + Per-Tenant OVS Testbed Scope, Calibrated Live Threshold & Interactive Monograph Website

- **Date:** 2026-11-15
- **Step:** Step 08 / Step 10 (`docs/steps/STEP_08_live_integration.md`, `docs/steps/STEP_10_website_docs_release.md`)
- **What:** (1) The Ubuntu 22.04 live SDN testbed validates **L1 Linux network namespace (`t1..t3`) + per-tenant Open vSwitch (`br_t1..br_t3`) isolation** inside a VMware guest rather than multi-VM L3 hypervisor isolation, and applies a validation-calibrated classification threshold (`theta = 0.98`) in live OVS mitigation (reducing False Block Rate from `64.26%` raw zero-shot to `2.70%`); (2) the companion web portal (`../07_Website/trustfed-sdn-site/`, `naimurhamim/trust-fed-website`) is delivered as a zero-build static **10-Chapter Interactive Research Monograph** HTML/CSS/JS application hydrating live KPIs from `data/key_numbers.json`.
- **Why:** Earlier proposal notes mentioned multi-VM L3 isolation and default `0.5` thresholding; `L1` namespace + per-tenant OVS switches provide full OpenFlow 1.3 data-plane packet telemetry and deterministic reproducibility inside a single Ubuntu VM, calibration resolves sim-to-live domain shift transparently, and static HTML/CSS/JS ensures effortless Cloudflare Worker / GitHub Pages deployment with zero build toolchain drift.
- **Impact:** Every published figure, live testbed metric (`results/08_live/summary.json`), website KPI card, and audit note honestly reports L1 namespace isolation and domain-shift calibration without overclaiming hypervisor-level L3 isolation.


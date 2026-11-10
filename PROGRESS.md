# PROGRESS

The agent updates this file at the end of every step. Status values: `not started`, `in progress`, `waiting for approval`, `approved`.

| Step | Name | Status | Date | Commit | Notes |
|---|---|---|---|---|---|
| 00 | Environment and preferences | approved | 2026-11-05 | `567b2f9` (tag `step-00-done`) | `.venv` (Py 3.12.7), `make check` passing (4/4), Ubuntu 22.04 Mininet 2.3.0 + OVS 2.17.12 + OS-Ken 3.0.1 OpenFlow 1.3 `pingAll` verified (`0%` loss); L1 isolation approved; team sync system on GitHub |
| 01 | Data | approved | 2026-11-05 | `ff6931d` (tag `step-01-done`) | Processed `InSDN` (`184,855` cleaned), `ASEADOS-SDN-IoT` 2026 (`426,111` cleaned), and `CICIDS2017` (`2,232,894` cleaned); `0` split hash overlap; `12/12` pytest passing |
| 02 | Centralized baselines and model choice | approved | 2026-11-05 | `3b7adbd` (tag `step-02-done`) | Completed 5-seed centralized (`DNN`, `GRU`, `LSTM`, `LogReg`, `RandomForest`) & single-tenant local-only baselines across `InSDN`, `ASEADOS-SDN-IoT` 2026, and `CICIDS2017` on RTX 3050 Ti GPU; `GRU` selected per spec (`docs/MODEL_DECISION.md`); `27/27` pytest passing |
| 03 | FL core and FedAvg | approved | 2026-11-10 | (tag `step-03-done`) | Completed `trustfed/fl/*`, `sim/run_experiment.py`, `tests/test_fl.py` (`31/31` pytest passing), and full 5-seed `270`-run FedAvg sweep (`results/03_fedavg/`) across `InSDN`, `ASEADOS-SDN-IoT` 2026, and `CICIDS2017` |
| 04 | Attacks and robust aggregation | not started | | | |
| 05 | Adaptive DP | not started | | | |
| 06 | Verifiable secure aggregation | not started | | | |
| 07 | SDN testbed | not started | | | |
| 08 | Live integration | not started | | | |
| 09 | Full evaluation | not started | | | |
| 10 | Website, docs, release | not started | | | |

## Open questions for the user

(none yet)

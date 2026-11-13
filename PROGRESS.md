# PROGRESS

The agent updates this file at the end of every step. Status values: `not started`, `in progress`, `waiting for approval`, `approved`.

| Step | Name | Status | Date | Commit | Notes |
|---|---|---|---|---|---|
| 00 | Environment and preferences | approved | 2026-11-05 | `567b2f9` (tag `step-00-done`) | `.venv` (Py 3.12.7), `make check` passing (4/4), Ubuntu 22.04 Mininet 2.3.0 + OVS 2.17.12 + OS-Ken 3.0.1 OpenFlow 1.3 `pingAll` verified (`0%` loss); L1 isolation approved; team sync system on GitHub |
| 01 | Data | approved | 2026-11-05 | `ff6931d` (tag `step-01-done`) | Processed `InSDN` (`184,855` cleaned), `ASEADOS-SDN-IoT` 2026 (`426,111` cleaned), and `CICIDS2017` (`2,232,894` cleaned); `0` split hash overlap; `12/12` pytest passing |
| 02 | Centralized baselines and model choice | approved | 2026-11-05 | `3b7adbd` (tag `step-02-done`) | Completed 5-seed centralized (`DNN`, `GRU`, `LSTM`, `LogReg`, `RandomForest`) & single-tenant local-only baselines across `InSDN`, `ASEADOS-SDN-IoT` 2026, and `CICIDS2017` on RTX 3050 Ti GPU; `GRU` selected per spec (`docs/MODEL_DECISION.md`); `27/27` pytest passing |
| 03 | FL core and FedAvg | approved | 2026-11-10 | `cd1b8b7` (tag `step-03-done`) | Completed `trustfed/fl/*`, `sim/run_experiment.py`, `tests/test_fl.py` (`31/31` pytest passing), and full 5-seed `270`-run FedAvg sweep (`results/03_fedavg/`) across `InSDN`, `ASEADOS-SDN-IoT` 2026, and `CICIDS2017` |
| 04 | Attacks and robust aggregation | approved | 2026-11-12 | `06647a5` (tag `step-04-done`) | Completed `trustfed/attacks/*` (`label_flip`, `backdoor`, `scaling`, `compute_backdoor_asr`), `trustfed/aggregation/*` (`static_weight`, `trimmed_mean_b0.1`, `trimmed_mean_b0.2`, `median`, `clustering`), `tests/test_attacks.py` & `tests/test_aggregation.py` (`38/38` pytest passing), and full 5-seed `660`-run Step 04 poisoning & failure-mode sweep (`results/04_robust/`) |
| 05 | Adaptive DP | approved | 2026-11-12 | `step-05-done` | Completed `trustfed/privacy/*` (`accountant.py`, `budget.py`, `dp_sgd.py`, `mia.py`), `docs/DP_DESIGN.md`, `tests/test_privacy.py` (`43/43` pytest passing), and full 5-seed `220`-run Adaptive vs. Uniform DP-SGD & MIA sweep (`results/05_dp/`) across `InSDN` & `ASEADOS-SDN-IoT` |
| 06 | Verifiable secure aggregation | approved | 2026-11-13 | `step-06-done` | Completed `trustfed/verify/*` (`quantize.py`, `masking.py`, `merkle.py`, `commitment.py`, `protocol.py`), `trustfed/aggregation/secure.py` (`SecureAggregator`), `docs/VERIFY_DESIGN.md`, `tests/test_verify.py` (`50/50` pytest passing), `100%` tamper detection (`1,500` trials) / `0%` false alarms (`300` honest rounds), `1.26×` wire byte overhead, and 5-seed accuracy equivalence (`results/06_verify/`) |
| 07 | SDN testbed | not started | | | |
| 08 | Live integration | not started | | | |
| 09 | Full evaluation | not started | | | |
| 10 | Website, docs, release | not started | | | |

## Open questions for the user

(none yet)

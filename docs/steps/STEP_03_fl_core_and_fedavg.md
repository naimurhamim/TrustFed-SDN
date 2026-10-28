# STEP 03 — FL Simulation Core and FedAvg Baseline

## Goal
Build a transparent federated learning simulator and the undefended FedAvg baseline that every later component is compared against.

## Prerequisites
Step 02 approved (model decided).

## Ask the user first
1. Confirm the default FL settings from `docs/EXPERIMENT_SPEC.md` (30 rounds, 3 local epochs, batch 128, Adam 1e-3), or change them.
2. Should partial client participation be studied now or skipped? (Default: skipped.)

## Tasks
1. Implement the FL core in `trustfed/fl/`: `Client` (local training on a tenant partition), `Server` (aggregation loop), and a pluggable `Aggregator` interface. All aggregators operate on flat parameter tensors so they can be reused later by the live testbed.
2. Implement FedAvg with sample-count weighting.
3. Implement per-round logging: global and per-tenant metrics, bytes sent, wall-clock time split into local training and aggregation. Save to `results/03_fedavg/`.
4. Write a runner `sim/run_experiment.py --config configs/experiments/<name>.yaml` that supports N tenants, partition type, seeds, and rounds.
5. Run: FedAvg on natural non-IID partitions, Dirichlet alpha in {0.1, 0.5, 1.0}, and IID, with N in {3, 5, 10}, 5 seeds.
6. Compare FedAvg with the centralized upper bound and the local-only lower bound from Step 02. Plot convergence curves.
7. Add unit tests: aggregation equals a manual weighted mean on a toy example; determinism with the same seed; the runner reproduces a saved result.

## Deliverables
- `trustfed/fl/*`, `sim/run_experiment.py`, `configs/experiments/fedavg_*.yaml`
- `results/03_fedavg/` and convergence figures
- `tests/test_fl.py`

## Acceptance criteria
- [ ] Same seed gives the same result (determinism test passes)
- [ ] FedAvg, centralized, and local-only are compared in one table and one figure
- [ ] Per-round time and bytes are logged
- [ ] `make check` passes

## Report back, then stop
State clearly how much accuracy is lost to federation and to non-IID data. Wait for "approved step 03".

## Pitfalls
- Keep the aggregator interface strict: later steps (robust, DP, verifiable) plug into it.
- Count communication bytes from actual serialized update sizes, not guesses.

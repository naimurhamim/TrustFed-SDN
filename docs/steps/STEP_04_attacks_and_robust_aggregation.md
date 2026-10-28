# STEP 04 — Poisoning Attacks and Robust Aggregation

## Goal
Show that poisoning hurts FedAvg, then show how much the robust aggregators recover.

## Prerequisites
Step 03 approved.

## Ask the user first
1. Confirm the three attacks and the poisoned fractions in `docs/EXPERIMENT_SPEC.md`, or add or remove items.
2. Should Krum and median be included as extra baselines? (Default: median only if time allows.)

## Tasks
1. Implement the attacks in `trustfed/attacks/`: label flipping, backdoor / model replacement (trigger on a fixed feature subset, target class benign), and update scaling or sign flipping. Make strength parameters configurable.
2. Implement aggregators in `trustfed/aggregation/`: coordinate-wise trimmed mean (beta 0.1 and 0.2), cosine-similarity clustering filter, and the static-weight baseline (grid-search the fixed weights on validation data, then freeze them).
3. Implement the backdoor attack success rate metric.
4. Run the grid: N = 10, poisoned fraction in {0, 10, 20, 30}%, every attack, every aggregator (FedAvg, static-weight, trimmed mean, clustering), natural non-IID and Dirichlet 0.5, 5 seeds.
5. Specifically test the risk that honest but heterogeneous tenants look "different" and are wrongly filtered. Report false-rejection rates of honest tenants, especially under non-IID.
6. Report the accuracy drop at 20% poisoning for each method and compare against the targets in `PROJECT_PLAN.md` section 7.
7. Add tests: each aggregator on toy vectors with known outliers; a tenant that sends a huge update is removed by clustering; trimmed mean with beta = 0 equals the plain mean.

## Deliverables
- `trustfed/attacks/*`, `trustfed/aggregation/*`
- `configs/experiments/robust_*.yaml`
- `results/04_robust/` (tables, accuracy-versus-poison-fraction plots, backdoor success plots, honest false-rejection table)
- `tests/test_attacks.py`, `tests/test_aggregation.py`

## Acceptance criteria
- [ ] Every attack measurably hurts FedAvg (otherwise the attack is too weak; report and fix)
- [ ] All aggregators evaluated on identical seeds and partitions
- [ ] Honest-tenant false-rejection rates reported
- [ ] Real numbers compared against the poster targets, with no tuning to hit them

## Report back, then stop
Give a one-table summary and an honest statement of where robustness fails. Wait for "approved step 04".

## Pitfalls
- With few tenants, trimmed mean with large beta removes honest information. Show the trade-off.
- Do not cherry-pick an attack strength that favors our method; use a documented range.

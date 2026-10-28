# STEP 02 — Centralized Baselines and Local Model Decision

## Goal
Establish the best-case (centralized) performance and choose the local model with evidence.

## Prerequisites
Step 01 approved.

## Ask the user first
1. Confirm the evaluation targets: binary detection, multi-class detection, or both? (Default: both, with binary as the headline.)
2. Time budget for hyperparameter search: small (a few configurations) or larger?

## Tasks
1. Implement DNN, GRU, and LSTM in `trustfed/models/` with the same input interface. Decide and document how the tabular feature vector is presented to recurrent layers (for example a sequence of length 1 or chunks of features) and test the alternatives briefly.
2. Train each model **centrally** on the pooled training data from all tenants, with a small, fixed hyperparameter search (same budget for every model). Use 5 seeds.
3. Evaluate with all detection metrics from `docs/EXPERIMENT_SPEC.md` on the global held-out test set and on each tenant's test set.
4. Record parameter counts, training time, and inference time per flow for each model.
5. Apply the selection rule from the spec and write the decision to `docs/MODEL_DECISION.md` with the evidence table.
6. Also train the models on one tenant only (local-only baseline) to show the value of federation later.
7. Add a simple baseline set (for example logistic regression and a random forest) for context.

## Deliverables
- `trustfed/models/*`, `trustfed/metrics/*`
- `results/02_baselines/summary.json`, per-seed CSV, plots
- `docs/MODEL_DECISION.md`
- `tests/test_models.py`

## Acceptance criteria
- [ ] All three models trained with equal tuning budgets and 5 seeds
- [ ] Mean and standard deviation reported for every metric
- [ ] Decision written down using the rule in the spec
- [ ] Local-only baselines exist for every tenant

## Report back, then stop
Show the comparison table and ask the user to approve the model choice explicitly. Wait for "approved step 02".

## Pitfalls
- Very high accuracy (above 99.9%) on CIC-style data often hides duplicates or leakage. Re-check the split before celebrating.
- Do not tune on the test set.

# STEP 05 — Adaptive Per-Tenant Differential Privacy

## Goal
Implement differential privacy with a per-tenant budget, measure the privacy-utility trade-off, and test the privacy empirically with a membership-inference attack.

## Prerequisites
Step 03 approved. (Step 04 is not required, so this step may be run before it if the user prefers.)

## Ask the user first (design checkpoint, needs approval)
1. **Granularity.** Which guarantee do you want? (a) Record-level DP using DP-SGD during local training (clip per-sample gradients, add noise), or (b) update-level DP where each tenant clips its whole update and adds noise to it before sending. Explain the difference in two sentences and recommend one with reasons. (Update-level is simpler; record-level gives a standard guarantee for individual flow records.)
2. **Sensitivity score s_i.** Propose a definition and ask for approval. Options: a fixed value from the tenant profile (for example Finance and Healthcare high, Education low), a data-derived score (for example class imbalance, share of rare attack types, feature variance), or a weighted combination of both. The score must be documented and reproducible.
3. **Budget policy.** The poster formula is epsilon_i = epsilon_total * (1 / s_i) / (sum_j 1 / s_j). Explain what this means (the budgets sum to epsilon_total, higher sensitivity gives a smaller epsilon and therefore stronger privacy) and ask whether the user and supervisor accept it as a policy. Note clearly that the sum constraint is a design policy for allocating protection, not a composition theorem across tenants.
4. Confirm delta (default: smaller than 1 / number of local training samples) and the epsilon sweep.

## Tasks
1. Implement the chosen DP mechanism in `trustfed/privacy/` (clipping, Gaussian noise, and a privacy accountant using a standard method such as RDP). Prefer a well-tested library for accounting or per-sample gradients when the license and compatibility allow; otherwise implement carefully and test.
2. Implement the sensitivity score and the adaptive budget allocation exactly as approved. Implement the uniform-budget variant for comparison.
3. Run the epsilon sweep {0.5, 1, 2, 4, 8, no DP} for uniform and adaptive allocation at equal total budget, on FedAvg, natural non-IID, 5 seeds. Report accuracy, per-tenant accuracy, and the per-tenant epsilon values.
4. Implement the membership-inference attack (loss-threshold attack minimum). Evaluate on the global model and on local models, with and without DP. Report attack accuracy and AUC.
5. Report whether adaptive allocation helps the most sensitive tenant's privacy at a small cost in utility, using real numbers.
6. Tests: noise scale matches the formula; clipping bounds the norm; accountant output is monotonic in the number of rounds; budgets sum to epsilon_total.

## Deliverables
- `trustfed/privacy/*`, `configs/experiments/dp_*.yaml`
- `results/05_dp/` (privacy-utility curves, per-tenant table, membership-inference results)
- `docs/DP_DESIGN.md` (granularity, score definition, budget policy, exact guarantee statement, limitations)
- `tests/test_privacy.py`

## Acceptance criteria
- [ ] `docs/DP_DESIGN.md` states precisely what is protected and what is not
- [ ] Uniform versus adaptive comparison at equal total budget exists
- [ ] Membership-inference results exist with and without DP
- [ ] No privacy claim in any report goes beyond what the accountant and the attack results support

## Report back, then stop
Wait for "approved step 05".

## Pitfalls
- Report epsilon and delta together; epsilon alone is incomplete.
- A very small epsilon can destroy accuracy on small tenants; show it, do not hide it.
- Membership inference near 50% on a weak attack does not prove privacy; say that in the limitations.

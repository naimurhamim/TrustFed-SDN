# STEP 09 — Full Evaluation, Ablation, Figures, and Tables

## Goal
Run the complete, final experiment set, produce every figure and table from code, and write an honest results summary that the thesis, poster, and website can copy from.

## Prerequisites
Step 08 approved.

## Ask the user first
1. Confirm the final experiment list below, or remove items if time is short (primary items are marked P).
2. Seeds: 5 is the minimum. Is more compute time available for 10?
3. Preferred figure style and format (PDF vector for the thesis, PNG for the website) and color scheme (accessible, readable when printed in grayscale).

## Experiment list

| ID | Experiment | Priority |
|---|---|---|
| E1 | Centralized, local-only, and FedAvg comparison on all datasets | P |
| E2 | Non-IID study (natural, Dirichlet 0.1 / 0.5 / 1.0, IID) | P |
| E3 | Poisoning matrix: 3 attacks × 4 poison fractions × 4 aggregators | P |
| E4 | DP sweep: uniform versus adaptive at equal total budget, plus membership inference | P |
| E5 | Verification: tamper detection and overhead breakdown | P |
| E6 | **Ablation ladder** (FedAvg → +DP → +Robust → +Verifiable → Full), clean and under each attack | P |
| E7 | Live testbed: detection, mitigation, latency, control-plane impact | P |
| E8 | Scalability in the number of tenants (3, 5, 10, and 20 if feasible) | P |
| E9 | Cross-dataset generalization (train on InSDN, test on CICIDS2017 and live data, and the reverse) | Optional |
| E10 | Sensitivity of key hyperparameters (beta, clipping norm, quantization precision) | Optional |

## Tasks
1. Create or finalize configs for every experiment under `configs/experiments/final/`.
2. Implement `make reproduce` to run all experiments (with a `--quick` mode for a smoke run and a full mode) and write results to `results/final/`.
3. Generate all figures (`figures/`) and LaTeX tables (`tables/`) by scripts only. Include confidence intervals or standard deviations.
4. Run statistical tests for key claims (paired Wilcoxon with seeds as pairs), and state the number of seeds and p-values. Do not claim significance without a test.
5. Compare each result with the targets in `PROJECT_PLAN.md` section 7 in a table with three columns: target, measured, met or not met.
6. Write `docs/RESULTS_SUMMARY.md`: plain-language findings, what worked, what did not, limitations, threats to validity (dataset, emulation versus real deployment, small number of tenants, simulated attacks), and a list of claims that are safe to make versus claims that are not supported.
7. Prepare a "poster and thesis numbers" file `results/final/key_numbers.json` listing every number intended for public use with the file it came from.

## Deliverables
- `results/final/*`, `figures/*`, `tables/*`
- `docs/RESULTS_SUMMARY.md`, `results/final/key_numbers.json`
- `make reproduce` and `make reproduce-quick`

## Acceptance criteria
- [ ] All primary experiments (P) complete with at least 5 seeds
- [ ] Every figure and table regenerates from scripts
- [ ] Target-versus-measured table is complete and honest
- [ ] `key_numbers.json` exists and every entry points to a source file
- [ ] Limitations section is specific, not generic

## Report back, then stop
Wait for "approved step 09".

## Pitfalls
- Do not drop failed runs. Report them and explain.
- Keep the same data splits and seeds across all compared methods.
- Do not compare directly with accuracy numbers from other papers unless the dataset and protocol match.

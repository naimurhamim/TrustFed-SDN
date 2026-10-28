# STEP 01 — Data

## Goal
Obtain, clean, feature-select, and partition the datasets into reproducible per-tenant train/validation/test sets.

## Prerequisites
Step 00 approved. Read `docs/DATASETS.md` and `docs/EXPERIMENT_SPEC.md` first.

## Ask the user first
1. Do you already have InSDN, CICIDS2017, or CICDDoS2019 on this computer? Where?
2. How much free disk space do you have? Do you want CICDDoS2019 at all, or only InSDN and CICIDS2017?
3. Automatic download or manual download? (Follow `docs/DATASETS.md` section 3. If anything fails, print the manual instructions and wait for the user.)
4. Confirm the tenant profiles for the natural partition (default: Finance, Healthcare, Education) and which attack families each tenant should be heavy in. Propose a default and ask for approval.

## Tasks
1. Acquire the data into `data/raw/` and write `data/MANIFEST.md` (file, size, SHA-256, source URL, date).
2. Write `trustfed/data/load.py` loaders for each dataset and a profile report script (`scripts/profile_dataset.py`) that saves schema, class counts, missing values, and duplicate counts to `results/01_data/`.
3. Implement the cleaning rules from `docs/DATASETS.md` section 5 in `trustfed/data/clean.py`. Log counts per rule to `results/01_data/cleaning_log.csv`.
4. Implement label mapping (binary and multi-class) in `configs/data/label_map.yaml`.
5. Implement the split (train/validation/test, stratified, seeded) **before** any fitting.
6. Implement top-k per-class feature selection on training data only. Save the feature lists to `configs/data/features_*.yaml` and a ranking plot.
7. Implement the natural non-IID partition and the Dirichlet partition (alpha 0.1, 0.5, 1.0) in `trustfed/data/partition.py`. Save per-tenant class distribution plots.
8. Implement per-tenant scaling (fit on local training data only) plus the global-scaler ablation option.
9. Save processed arrays under `data/processed/` and write unit tests (shape checks, leakage hash check, partition sizes, label mapping).

## Deliverables
- `data/MANIFEST.md`, `trustfed/data/*`, `configs/data/*`
- `results/01_data/` (profiles, cleaning log, class distribution plots, feature rankings)
- `tests/test_data.py`

## Acceptance criteria
- [ ] Row counts before and after every cleaning rule are logged
- [ ] Zero overlap between train, validation, and test sets (hash check passes)
- [ ] Per-tenant class distributions are plotted for all partition types
- [ ] Scalers and selectors were fitted on training data only (test proves it)
- [ ] `make check` passes

## Report back, then stop
Include the final feature list, class counts per tenant, and any dataset quirks found. Wait for "approved step 01".

## Pitfalls
- Duplicate header rows inside CSV files, infinite values, and trailing spaces in column names are common in CIC datasets.
- Do not use identifier columns (IP addresses, flow ID, timestamp) as features.
- Be careful that attack families that exist only in one dataset are not silently dropped.

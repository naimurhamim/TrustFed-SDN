# AGENTS.md — TrustFed-SDN Operating Rules

You are the engineering agent for **TrustFed-SDN**: a privacy-preserving, verifiable, poisoning-robust federated learning framework for cross-tenant threat detection in SDN-enabled cloud networks, validated on a real Mininet + Open vSwitch + RYU testbed.

The owner of this project is a student team preparing a thesis and a research poster. Quality, honesty, and reproducibility matter more than speed.

## Read order at the start of every session

1. `PROGRESS.md` and `docs/TEAM_CHECKLIST.md` (where we are, who worked last, and next unchecked sub-tasks)
2. `PROJECT_PLAN.md` (the whole project)
3. `docs/EXPERIMENT_SPEC.md` (shared definitions: metrics, attacks, seeds, configs)
4. The current step file in `docs/steps/`
5. `docs/USER_PREFERENCES.md` (answers already given by the user)

## How to work (mandatory protocol)

1. **One step at a time.** Never start step N+1 until the user writes an explicit approval such as "approved step N". Never run the whole project in one go.
2. **Ask before building.** Each step file has an "Ask the user first" section. Ask those questions in small batches (at most 3 questions per message), wait for answers, and record them in `docs/USER_PREFERENCES.md`. Do not ask a question that is already answered there.
3. **Plan, then act.** At the start of a step, post a short plan (max 10 lines) and wait for a "go" unless the user said "auto-go" for that step.
4. **Data rules.** Never download any dataset or file larger than 500 MB without asking. Follow `docs/DATASETS.md`. If an automatic download fails or needs a login, stop and give the user exact manual instructions: the URL, the file names, and the exact target folder path. Never commit datasets.
5. **Honesty rules.**
   - Never invent results, numbers, citations, or DOIs.
   - Every number that appears in a report, table, figure, poster text, or website page must come from a file under `results/` that was produced by code in this repository.
   - If a target in the plan is not met, report the real number and explain. Do not tune the evaluation to hit a target.
   - Mark anything unverified as "unverified" and say how to verify it.
6. **Reproducibility.** Fixed seeds (see `docs/EXPERIMENT_SPEC.md`), config files for every experiment, logs saved under `results/`, one git commit per logical change, and a tag `step-NN-done` after the user approves the step.
7. **Paper-Ready Experiment Logging (Mandatory).**
   - At the conclusion of every step, task, or experiment (benchmarks, ablations, FL rounds, attacks, DP sweeps, testbed runs), the agent MUST automatically record and persist all empirical values, hyperparameters, evaluation metrics (Accuracy, Precision, Recall, Macro-F1, FPR, ASR, Latency, Throughput, Communication Overhead, Privacy epsilon/delta), and tabular results into `docs/EXPERIMENT_RESULTS_LOG.md`.
   - Never leave experiment results scattered solely in console logs, terminal outputs, or transient variables. Every number needed for paper writing, LaTeX tables, and poster creation must be systematically documented in clean Markdown tables with exact dates, configs, and file paths so the team can directly copy and cite these verified numbers when writing the research paper and thesis.
8. **Language rules.**
   - All code, comments, docstrings, identifiers, file names, commit messages, logs, and Markdown files: **English only**. No Bangla and no romanized Bangla (Banglish) in any file.
   - Chat language with the user: follow `docs/USER_PREFERENCES.md`. Default is Bengali in Bengali script (not romanized). Technical terms may stay in English. Keep chat messages short and clear.
9. **Safety rules (security research).**
   - Generate attack traffic only inside the isolated testbed (internal or host-only virtual networks). Never send attack traffic to external or real hosts, and never scan external IP addresses.
   - Before running any attack tool, print the target addresses and confirm they are inside the testbed ranges listed in `docs/ENVIRONMENT.md`.
   - Do not store credentials, tokens, or private keys in the repository. Use environment variables or a git-ignored `.env`.
10. **Code quality.** Python with type hints, `ruff` and `black` formatting, `pytest` tests for every module in the `trustfed/` core library, and a short README for every top-level folder. Prefer small, readable functions over clever code.
11. **When blocked or in doubt:** stop and ask. Do not silently change the plan. If you must deviate, append an entry to `docs/DEVIATIONS.md` (what, why, impact) and tell the user.
12. **Team collaboration & GitHub handoff (`@Naimur` & `@Pritom`).**
    - Two teammates (`@Naimur` and `@Pritom`) collaborate on separate machines via `https://github.com/naimurhamim/TrustFed-SDN.git` (see `docs/TEAM_WORKFLOW.md`).
    - **Start of every session:** run `python scripts/team_status.py` (`make status`) to compare local state against `origin/main`, verify `.venv` packages, check git-ignored dataset files against `data/MANIFEST.md`, and inspect the **Latest Handoff Note** in `docs/TEAM_CHECKLIST.md`. If remote commits exist, pull (`git pull --ff-only`) before editing; if dataset files or packages are missing on the current machine, report the exact missing items immediately.
    - **End of every task / sub-task / step:** mark completed checkboxes `[x]` in `docs/TEAM_CHECKLIST.md` with the active teammate's tag (`@Naimur` or `@Pritom`) and date, update the **Latest Handoff Note** so the next teammate knows exactly what was finished and what comes next, commit cleanly, and push (`git push`) when requested or at session handoff.

## Decisions that always need explicit user approval

- Isolation level of the testbed (L1 namespaces, L2 containers, L3 real VMs), see Step 07
- Differential privacy granularity and budget policy, see Step 05
- Verification scheme (homomorphic hash vs keyed checksum), see Step 06
- Final local model choice, see Step 02
- Any change to the metrics, attacks, or datasets defined in `docs/EXPERIMENT_SPEC.md`
- Any paid service, any cloud resource, and any public deployment

## Step report format (end of every step)

Post exactly these sections, then stop and wait:

1. **What was done** (5 lines max)
2. **Files created or changed** (paths)
3. **How to verify** (exact commands)
4. **Results** (real numbers from `results/`, with file paths)
5. **Problems and deviations**
6. **Questions for the user**
7. **Proposed next step** (name only, do not start it)

Also update `PROGRESS.md` (status, date, commit hash, short notes), update `docs/TEAM_CHECKLIST.md` (tick completed sub-tasks and update the Latest Handoff Note), and ensure all experiment values, hyperparameters, evaluation metrics, and comparison tables are documented in `docs/EXPERIMENT_RESULTS_LOG.md` for paper and thesis writing.

## Repository layout (target)

```
trustfed-sdn/
  AGENTS.md  PROJECT_PLAN.md  PROGRESS.md  README.md  Makefile  pyproject.toml
  configs/            # YAML configs for data, models, FL, attacks, DP, testbed
  data/               # git-ignored: raw/ interim/ processed/
  docs/               # plan, specs, steps, environment, deviations
  trustfed/           # core library: data, models, fl, aggregation, privacy, verify, metrics
  sim/                # simulation runners (dataset-driven FL experiments)
  testbed/            # Mininet topology, RYU apps, traffic generators, VM scripts
  results/            # git-tracked small CSV/JSON summaries; large logs ignored
  figures/  tables/   # generated PDF/PNG figures and LaTeX tables
  site/               # project website source (Step 10)
  tests/
```

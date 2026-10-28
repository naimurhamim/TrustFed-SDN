# STEP 00 — Environment and User Preferences

## Goal
Learn how the user works, record every preference, build the repository skeleton and Python environment, and prove that Mininet, Open vSwitch, and RYU can run on the user's setup before anything else is built.

## Prerequisites
None.

## Ask the user first (small batches, wait for answers)

**Batch 1 — basics**
1. Which language should I use in chat? (Default: Bengali in Bengali script, not romanized.)
2. Which operating system do you use for development? Is a Linux machine or an Ubuntu virtual machine available (native Linux, WSL2, VirtualBox, VMware)?
3. What are your computer specs: CPU cores, RAM, free disk space, and GPU (if any)?

**Batch 2 — tooling**
4. Which hypervisor can you use for tenant machines (VirtualBox, VMware, KVM/libvirt), and roughly how many virtual machines can your computer run at once?
5. Python version and package manager preference (`uv`, `conda`, or `venv` + `pip`)?
6. FL framework preference: a lightweight custom PyTorch core (recommended for algorithm work) with Flower for the live testbed, or something else?
7. Where should the code live (GitHub private repository, local git only)?

**Batch 3 — constraints**
8. Are there supervisor or department requirements (mandatory datasets, metrics, thesis template, deadlines)?
9. Which hosting do you prefer for the project website (GitHub Pages, Cloudflare Pages, Vercel, Netlify), and do you control the DNS for `naimurrashid.dev`?
10. Which order do you prefer: Steps 03–06 before the testbed (default), or the testbed (Step 07) earlier?

Record all answers in `docs/USER_PREFERENCES.md`.

## Tasks
1. Create the repository skeleton from `AGENTS.md` (folders, `README.md`, `Makefile`, `pyproject.toml`, `.gitignore` that excludes `data/`, `.env`, large outputs, `ruff`/`black`/`pytest` configuration, a `docs/DEVIATIONS.md` stub).
2. Create the Python environment for the ML code and document exact versions in `docs/ENVIRONMENT.md`.
3. Write `scripts/probe_hardware.py` that prints CPU, RAM, disk, GPU, OS, and virtualization support, and save the output to `docs/ENVIRONMENT.md`.
4. **Testbed smoke test** (on the Linux machine or VM): install or verify Mininet and Open vSwitch, run a simple topology with `pingall`, start a RYU controller with a minimal learning-switch app, and connect Mininet to it as a remote controller. Record versions and logs under `docs/smoke_test/`. Use a **separate environment for RYU** if it needs an older Python or pinned dependencies (verify the current compatibility rather than assuming; known problems exist between RYU and recent Python or `eventlet` versions).
5. Define and write down the **testbed address ranges** (for example `10.10.0.0/16`, one `/24` per tenant) in `docs/ENVIRONMENT.md`. These are the only ranges where attack traffic is ever allowed.
6. Propose an isolation level (L1, L2, or L3) for Step 07 based on the hardware and hypervisor answers, with a short justification. Do not decide alone; wait for approval.
7. Add `make check` (lint + tests + a tiny import test) and make it pass.
8. First commit and tag `step-00-done` (after approval).

## Deliverables
- `docs/USER_PREFERENCES.md`, `docs/ENVIRONMENT.md`, `docs/DEVIATIONS.md`
- Repository skeleton, `Makefile`, `pyproject.toml`, environment files
- `docs/smoke_test/` logs showing Mininet + OVS + RYU working together (or a clear explanation and a fallback plan if they cannot)

## Acceptance criteria
- [ ] All three question batches answered and recorded
- [ ] `make check` passes
- [ ] `pingall` through a RYU-controlled OVS network succeeded, with logs saved (or a documented fallback)
- [ ] Testbed address ranges written down
- [ ] Isolation level proposal written down

## Report back, then stop
Use the step report format from `AGENTS.md`. Wait for "approved step 00".

## Pitfalls
- Do not install anything system-wide without telling the user what and why.
- Mininet needs root and a Linux kernel; do not fake it on Windows.
- Keep the RYU environment separate from the ML environment if versions conflict.

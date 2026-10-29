# Team Collaboration & GitHub Sync Guide (`@Naimur` & `@Pritom`)

Repository URL (Private): `https://github.com/naimurhamim/TrustFed-SDN.git`

This guide ensures two teammates working on separate computers (each with their own AI coding agent) stay 100% synchronized without merge conflicts, missing packages, or mismatched datasets.

---

## 1. One-Time Setup on the Second Computer (`@Pritom`)

1. Accept the GitHub collaborator invitation for `naimurhamim/TrustFed-SDN`.
2. Ensure **Python 3.12** and **Git** are installed.
3. Clone and initialize the local environment:
   ```powershell
   git clone https://github.com/naimurhamim/TrustFed-SDN.git
   cd TrustFed-SDN
   py -3.12 -m venv .venv
   .\.venv\Scripts\pip.exe install -e ".[dev]"
   .\make.cmd status
   .\make.cmd check
   ```
   *(If `mingw32-make` is not installed on Windows, you can run `.venv\Scripts\python.exe scripts/team_status.py` and `.venv\Scripts\python.exe -m pytest` directly, or install `make` via `winget install ezwinports.make`.)*

---

## 2. Fixing the Multi-Account GitHub Popup on Windows (Auto-Push for Agents)

When Windows Credential Manager stores **two GitHub accounts**, Git Credential Manager (`credential.helper=manager`) opens an interactive account-picker window during `git push` or `git fetch`, which blocks automated AI agent commands.

Run this **once** inside `TrustFed-SDN/` to bind this repository to your exact GitHub username (stored only in local `.git/config`, never pushed, zero tokens stored in plain text):

```powershell
# On Naimur's computer:
git config --local credential.https://github.com.username naimurhamim

# On Pritom's computer (if Pritom also has multiple GitHub accounts):
git config --local credential.https://github.com.username <pritom-github-username>
```

The very first `git push` may ask you to sign in once for that username in your browser; after Windows Credential Manager caches that token under that username, every future `git fetch`, `git pull`, and `git push` by you or the AI agent runs **100% non-interactively with zero account-chooser popups**.

---

## 3. Daily Turn-by-Turn Workflow (Every Work Session)

### A. Before Starting Work (Compare & Pull)

Run:
```powershell
.\make.cmd status
```
or tell the AI agent:
> **"Run `scripts/team_status.py`, pull any remote changes, and show where we are in `docs/TEAM_CHECKLIST.md`."**

What `scripts/team_status.py` checks automatically:
1. **GitHub Remote Comparison:** Fetches `origin/main` and lists any new commits and changed files pushed by your teammate. If behind, run `git pull --ff-only`.
2. **Python Package Sync:** Checks that every dependency in `pyproject.toml` is installed in your local `.venv`. If anything was added, run `.\make.cmd install`.
3. **Dataset Sync (`data/MANIFEST.md`):** Because raw datasets in `data/raw/` exceed GitHub size limits and are git-ignored, `team_status.py` verifies every file listed in `data/MANIFEST.md` exists on your disk with the exact expected byte size (and optional `--verify-hashes` SHA-256 check).
4. **Checklist & Handoff:** Displays the **Latest Handoff Note** from `docs/TEAM_CHECKLIST.md` and the exact next unchecked `[ ]` sub-tasks.

### B. Sharing Datasets Between Both Computers

- Raw datasets (`data/raw/insdn/`, `data/raw/cicids2017/`) are **never** pushed to GitHub.
- Share the raw archive once (via Google Drive or USB) and extract into the exact same relative folder (`TrustFed-SDN/data/raw/...`).
- `data/MANIFEST.md` **is** tracked in GitHub, so `.\make.cmd status` immediately confirms whether both computers have identical dataset files.

### C. Finishing a Work Session (Checklist, Test, Push)

1. Run `.\make.cmd check` (`ruff` + `black --check` + `pytest`) and ensure it passes.
2. Tick `[x]` on completed sub-tasks in `docs/TEAM_CHECKLIST.md` with `(@Naimur, YYYY-MM-DD)` or `(@Pritom, YYYY-MM-DD)` and update the **Latest Handoff Note** at the top.
3. Update `PROGRESS.md` (and `docs/EXPERIMENT_RESULTS_LOG.md` if any experiment ran).
4. Commit and push:
   ```powershell
   git add -A
   git commit -m "step XX: concise English summary of completed sub-tasks"
   git push
   ```
5. Message your teammate that you pushed so they know `main` is ready for their turn.

"""Zero-dependency Telegram notification sender for TrustFed-SDN milestones.

Supports both:
- Option A: GitHub Actions (reads TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID from env/secrets).
- Option B: Local CLI (reads from ../08_Telegram_Bot/.env or --env-file).

Produces structured, zero-emoji HTML messages summarizing:
1. Current step accomplishments
2. Overall progress across all thesis steps (Step 00..10)
3. Next actions, teammate instructions (Pritom), links, and optional file attachments
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_ROOT = REPO_ROOT.parent
DEFAULT_ENV_FILE = THESIS_ROOT / "08_Telegram_Bot" / ".env"

GITHUB_REPO_URL = "https://github.com/naimurhamim/TrustFed-SDN"
WEBSITE_URL = "https://trustfed.naimurrashid.dev/"
MAX_TELEGRAM_TEXT_LEN = 3900


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs into os.environ without overwriting existing vars."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ and val:
            os.environ[key] = val


def parse_progress_md(progress_path: Path) -> list[dict[str, str]]:
    """Parse the step table from PROGRESS.md."""
    rows: list[dict[str, str]] = []
    if not progress_path.exists():
        return rows
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line_s = line.strip()
        if not line_s.startswith("|"):
            continue
        parts = [p.strip() for p in line_s.strip("|").split("|")]
        if len(parts) < 6 or not re.fullmatch(r"\d{2}", parts[0]):
            continue
        rows.append(
            {
                "step": parts[0],
                "name": parts[1],
                "status": parts[2].lower(),
                "date": parts[3],
                "commit": parts[4].strip("`"),
                "notes": parts[5],
            }
        )
    return rows


def parse_team_checklist(checklist_path: Path) -> dict[str, Any]:
    """Extract Latest Handoff Note and per-step checklist items from TEAM_CHECKLIST.md."""
    info: dict[str, Any] = {
        "updated_by": "@Naimur",
        "date": "",
        "active_step": "",
        "what_completed": "",
        "next_task": "",
        "step_items": {},
    }
    if not checklist_path.exists():
        return info

    text = checklist_path.read_text(encoding="utf-8")
    current_step: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- **Last Updated By:**"):
            info["updated_by"] = line.split(":", 1)[1].strip().strip("`")
        elif line.startswith("- **Date:**"):
            info["date"] = line.split(":", 1)[1].strip().strip("`")
        elif line.startswith("- **Active Step:**"):
            info["active_step"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **What Was Just Completed:**"):
            info["what_completed"] = line.split(":", 1)[1].strip()
        elif line.startswith("- **Next Task for"):
            info["next_task"] = line.split(":", 1)[1].strip()
        else:
            m_hdr = re.match(r"^###\s+Step\s+(\d{2})\b", line)
            if m_hdr:
                current_step = m_hdr.group(1)
                info["step_items"].setdefault(current_step, [])
                continue
            if current_step is not None:
                m_item = re.match(r"^- \[(x| )\]\s+`(\d{2}\.\d+)`\s+(.*)$", line)
                if m_item:
                    info["step_items"][current_step].append(
                        {
                            "done": m_item.group(1) == "x",
                            "id": m_item.group(2),
                            "text": m_item.group(3),
                        }
                    )
    return info


def _clean_md_inline(text: str) -> str:
    """Strip markdown bold/code wrappers for clean plain text before HTML escaping."""
    cleaned = text.replace("**", "").replace("`", "")
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _select_target_step(
    rows: list[dict[str, str]], explicit_step: str | None
) -> dict[str, str] | None:
    if not rows:
        return None
    if explicit_step:
        norm = explicit_step.zfill(2)
        for row in rows:
            if row["step"] == norm:
                return row
    for status_target in ("waiting for approval", "in progress", "approved"):
        matches = [r for r in rows if r["status"] == status_target]
        if matches:
            return matches[-1]
    return rows[0]


def _status_tag(status: str) -> str:
    if status == "approved":
        return "DONE"
    if status == "waiting for approval":
        return "WAITING APPROVAL"
    if status == "in progress":
        return "IN PROGRESS"
    return "PENDING"


def build_message(
    rows: list[dict[str, str]],
    checklist: dict[str, Any],
    *,
    explicit_step: str | None = None,
    custom_note: str | None = None,
) -> str:
    """Build structured zero-emoji Telegram HTML message."""
    target = _select_target_step(rows, explicit_step)
    if target is None:
        return "<b>TrustFed-SDN Update</b>\nNo steps found in PROGRESS.md."

    step_num = target["step"]
    step_name = html.escape(_clean_md_inline(target["name"]))
    status_label = _status_tag(target["status"])
    updated_by = html.escape(_clean_md_inline(checklist.get("updated_by") or "@Naimur"))
    date_str = html.escape(target["date"] or checklist.get("date") or "N/A")
    commit_str = html.escape(_clean_md_inline(target["commit"]) or "local / pending")

    lines: list[str] = [
        f"<b>TrustFed-SDN | Step {step_num}: {step_name} [{status_label}]</b>",
        f"<b>Updated By:</b> {updated_by} | <b>Date:</b> {date_str}",
        f"<b>Commit / Tag:</b> <code>{commit_str}</code>",
        "",
        f"<b>1. Current Step Summary (Step {step_num})</b>",
    ]

    if custom_note:
        lines.append(f"- {html.escape(_clean_md_inline(custom_note))}")

    notes = _clean_md_inline(target["notes"])
    if notes:
        for chunk in [c.strip() for c in notes.split(";") if c.strip()]:
            lines.append(f"- {html.escape(chunk)}")

    step_items = checklist.get("step_items", {}).get(step_num, [])
    done_items = [it for it in step_items if it["done"]]
    if done_items:
        lines.append(f"- Checklist Verified: {len(done_items)}/{len(step_items)} sub-tasks done")

    done_count = sum(1 for r in rows if r["status"] == "approved")
    lines.extend(
        [
            "",
            f"<b>2. Overall Progress So Far ({done_count}/{len(rows)} Steps Approved)</b>",
        ]
    )
    for r in rows:
        tag = _status_tag(r["status"])
        r_name = html.escape(_clean_md_inline(r["name"]))
        if r["status"] == "approved":
            lines.append(f"- Step {r['step']} [{tag}] {r_name} ({html.escape(r['date'])})")
        else:
            lines.append(f"- Step {r['step']} [{tag}] {r_name}")

    next_task = _clean_md_inline(
        checklist.get("next_task") or "Run git pull --tags and review TEAM_CHECKLIST.md."
    )
    lines.extend(
        [
            "",
            "<b>3. Team &amp; Pritom Sync Instructions</b>",
            "- Run: <code>git pull --tags</code>",
            f"- Next Action: {html.escape(next_task)}",
            "- Key Docs: <code>PRITOM_ANTIGRAVITY_SETUP.md</code>, "
            "<code>docs/TEAM_CHECKLIST.md</code>, <code>docs/EXPERIMENT_RESULTS_LOG.md</code>",
            "",
            "<b>4. Project Links</b>",
            f'- GitHub Repo: <a href="{GITHUB_REPO_URL}">naimurhamim/TrustFed-SDN</a>',
            f'- Team Checklist: <a href="{GITHUB_REPO_URL}/blob/main/docs/TEAM_CHECKLIST.md">'
            "docs/TEAM_CHECKLIST.md</a>",
            f'- Results Log: <a href="{GITHUB_REPO_URL}/blob/main/docs/EXPERIMENT_RESULTS_LOG.md">'
            "docs/EXPERIMENT_RESULTS_LOG.md</a>",
            f'- Live Website: <a href="{WEBSITE_URL}">trustfed.naimurrashid.dev</a>',
        ]
    )

    msg = "\n".join(lines)
    if len(msg) > MAX_TELEGRAM_TEXT_LEN:
        msg = msg[: MAX_TELEGRAM_TEXT_LEN - 20] + "\n...(truncated)"
    return msg


def _run_git(repo_root: Path, args: list[str]) -> str:
    """Execute a git subcommand cleanly and return stripped stdout or empty string."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_git_push_info(
    repo_root: Path | None = None,
    *,
    before_sha: str | None = None,
    after_sha: str | None = None,
) -> dict[str, Any]:
    """Collect commit subject, description, author, and changed files for a git push."""
    root = repo_root or REPO_ROOT
    target_ref = (after_sha or "HEAD").strip() or "HEAD"
    head_sha = _run_git(root, ["rev-parse", target_ref])
    short_sha = _run_git(root, ["rev-parse", "--short", target_ref]) or target_ref[:7]
    author = _run_git(root, ["log", "-1", "--pretty=%an", target_ref]) or "Team Member"
    commit_date = _run_git(root, ["log", "-1", "--pretty=%ad", "--date=short", target_ref])
    subject = _run_git(root, ["log", "-1", "--pretty=%s", target_ref]) or "Repository update"
    body = _run_git(root, ["log", "-1", "--pretty=%b", target_ref])

    has_before = (
        bool(before_sha) and set(before_sha.strip()) != {"0"} and before_sha.strip() != head_sha
    )
    commit_list: list[str] = []
    if has_before and before_sha:
        raw_log = _run_git(
            root,
            ["log", "--pretty=format:%h %s (%an)", f"{before_sha.strip()}..{target_ref}"],
        )
        if raw_log:
            commit_list = [ln.strip() for ln in raw_log.splitlines() if ln.strip()]

    diff_args = (
        ["diff", "--name-status", f"{before_sha.strip()}..{target_ref}"]
        if has_before and before_sha and commit_list
        else ["diff-tree", "--no-commit-id", "--name-status", "-r", target_ref]
    )
    raw_files = _run_git(root, diff_args)
    changed_files: list[tuple[str, str]] = []
    for line in raw_files.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            changed_files.append((parts[0][:1], parts[-1].strip()))

    milestone_paths = {"PROGRESS.md", "docs/TEAM_CHECKLIST.md"}
    is_milestone = any(path in milestone_paths for _, path in changed_files)
    commit_url = f"{GITHUB_REPO_URL}/commit/{head_sha or short_sha}"
    if has_before and before_sha and len(commit_list) > 1:
        commit_url = f"{GITHUB_REPO_URL}/compare/{before_sha[:7]}...{short_sha}"

    return {
        "head_sha": head_sha or short_sha,
        "short_sha": short_sha,
        "author": author,
        "date": commit_date,
        "subject": subject,
        "body": body,
        "commits": commit_list,
        "changed_files": changed_files,
        "is_milestone": is_milestone,
        "commit_url": commit_url,
    }


def build_push_message(
    push_info: dict[str, Any],
    *,
    actor: str = "",
    branch: str = "main",
    custom_note: str | None = None,
) -> str:
    """Build concise, zero-emoji Telegram HTML summary describing what was pushed."""
    short_sha = html.escape(push_info.get("short_sha", "HEAD"))
    author = html.escape(push_info.get("author", "Unknown"))
    actor_clean = actor.strip()
    by_line = f"{author} (@{html.escape(actor_clean)})" if actor_clean else author
    date_str = html.escape(push_info.get("date") or "today")
    subject = html.escape(push_info.get("subject") or "Repository update")
    body = (push_info.get("body") or "").strip()
    commits = push_info.get("commits") or []
    changed_files = push_info.get("changed_files") or []
    commit_url = push_info.get("commit_url") or GITHUB_REPO_URL

    lines: list[str] = [
        f"<b>TrustFed-SDN | Git Push ({html.escape(branch)})</b>",
        f"<b>Pushed By:</b> {by_line} | <b>Date:</b> {date_str}",
        f"<b>Commit:</b> <code>{short_sha}</code>",
        "",
        "<b>What Was Pushed &amp; Description:</b>",
        f"- <b>Title:</b> {subject}",
    ]
    if body:
        for para_line in body.splitlines():
            clean_p = para_line.strip()
            if clean_p:
                lines.append(f"- {html.escape(clean_p)}")
    if len(commits) > 1:
        lines.append("")
        lines.append(f"<b>Commits Included ({len(commits)}):</b>")
        for c_line in commits[:8]:
            lines.append(f"- <code>{html.escape(c_line)}</code>")
        if len(commits) > 8:
            lines.append(f"- ... (+{len(commits) - 8} more commits)")
    if custom_note:
        lines.append(f"- <b>Note:</b> {html.escape(_clean_md_inline(custom_note))}")

    if changed_files:
        lines.append("")
        lines.append(f"<b>Changed Files ({len(changed_files)}):</b>")
        for st, fpath in changed_files[:14]:
            lines.append(f"- <code>[{html.escape(st)}] {html.escape(fpath)}</code>")
        if len(changed_files) > 14:
            lines.append(f"- ... (+{len(changed_files) - 14} more files)")

    lines.extend(
        [
            "",
            "<b>Sync &amp; Links:</b>",
            "- Team Sync: <code>git pull</code>",
            f'- View Changes: <a href="{commit_url}">GitHub Commit / Diff</a>',
            f'- Live Website: <a href="{WEBSITE_URL}">trustfed.naimurrashid.dev</a>',
        ]
    )
    msg = "\n".join(lines)
    if len(msg) > MAX_TELEGRAM_TEXT_LEN:
        msg = msg[: MAX_TELEGRAM_TEXT_LEN - 20] + "\n...(truncated)"
    return msg


def build_inline_keyboard(commit_url: str | None = None) -> dict[str, Any]:
    """Build inline URL buttons attached below the Telegram message."""
    first_btn = (
        {"text": "View Commit Diff", "url": commit_url}
        if commit_url
        else {"text": "GitHub Repo", "url": GITHUB_REPO_URL}
    )
    return {
        "inline_keyboard": [
            [
                first_btn,
                {
                    "text": "Team Checklist",
                    "url": f"{GITHUB_REPO_URL}/blob/main/docs/TEAM_CHECKLIST.md",
                },
            ],
            [
                {
                    "text": "Results Log",
                    "url": f"{GITHUB_REPO_URL}/blob/main/docs/EXPERIMENT_RESULTS_LOG.md",
                },
                {"text": "Live Website", "url": WEBSITE_URL},
            ],
        ]
    }


def telegram_send_message(
    bot_token: str,
    chat_id: str,
    text: str,
    *,
    reply_markup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send an HTML message via Telegram Bot API sendMessage."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return dict(json.loads(resp.read().decode("utf-8")))


def telegram_send_file(
    bot_token: str,
    chat_id: str,
    file_path: Path,
    caption: str = "",
) -> dict[str, Any]:
    """Upload a document or PNG photo via Telegram Bot API multipart/form-data."""
    is_photo = file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    endpoint = "sendPhoto" if is_photo else "sendDocument"
    field_name = "photo" if is_photo else "document"
    url = f"https://api.telegram.org/bot{bot_token}/{endpoint}"

    boundary = f"----TrustFedBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts: list[bytes] = []
    for k, v in (("chat_id", chat_id), ("caption", caption)):
        if not v:
            continue
        parts.append(
            (
                f"--{boundary}\r\n" f'Content-Disposition: form-data; name="{k}"\r\n\r\n' f"{v}\r\n"
            ).encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{file_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return dict(json.loads(resp.read().decode("utf-8")))


def load_chat_ids(subscribers_file: Path | None = None) -> list[str]:
    """Collect unique Telegram chat IDs from env vars and optional subscribers.json."""
    ids: list[str] = []
    for var_name in ("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_IDS"):
        raw = os.environ.get(var_name, "").strip()
        if raw:
            for item in raw.split(","):
                cid = item.strip()
                if cid and cid not in ids:
                    ids.append(cid)
    sub_path = subscribers_file or (THESIS_ROOT / "08_Telegram_Bot" / "subscribers.json")
    if sub_path.exists():
        try:
            data = json.loads(sub_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    cid = str(item.get("chat_id", "") if isinstance(item, dict) else item).strip()
                    if cid and cid not in ids:
                        ids.append(cid)
        except Exception:
            pass
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Send TrustFed-SDN update to Telegram subscribers")
    parser.add_argument("--step", default=None, help="Specific step number (e.g. 07)")
    parser.add_argument("--note", default=None, help="Optional extra summary line")
    parser.add_argument(
        "--attach",
        action="append",
        default=[],
        help="File path to send after the message (repeatable)",
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE), help="Path to .env file")
    parser.add_argument(
        "--push-event",
        action="store_true",
        help="Format commit diff/description message for git push events",
    )
    parser.add_argument("--before", default="", help="Before commit SHA (from GitHub push event)")
    parser.add_argument("--after", default="", help="After commit SHA (from GitHub push event)")
    parser.add_argument("--actor", default="", help="GitHub pusher username")
    parser.add_argument("--branch", default="main", help="Git branch name")
    parser.add_argument("--dry-run", action="store_true", help="Print message without sending")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))

    push_info: dict[str, Any] | None = None
    if args.push_event:
        push_info = collect_git_push_info(
            REPO_ROOT, before_sha=args.before or None, after_sha=args.after or None
        )

    if push_info and not args.step and not push_info["is_milestone"]:
        msg = build_push_message(
            push_info,
            actor=args.actor,
            branch=args.branch,
            custom_note=args.note,
        )
        keyboard = build_inline_keyboard(commit_url=push_info["commit_url"])
    else:
        rows = parse_progress_md(REPO_ROOT / "PROGRESS.md")
        checklist = parse_team_checklist(REPO_ROOT / "docs" / "TEAM_CHECKLIST.md")
        note_parts: list[str] = []
        if push_info:
            note_parts.append(f"Push {push_info['short_sha']}: {push_info['subject']}")
            if push_info["body"]:
                note_parts.append(push_info["body"])
        if args.note:
            note_parts.append(args.note)
        combined_note = " | ".join(note_parts) if note_parts else None
        msg = build_message(rows, checklist, explicit_step=args.step, custom_note=combined_note)
        keyboard = build_inline_keyboard(commit_url=push_info["commit_url"] if push_info else None)

    if args.dry_run:
        print("=== TELEGRAM DRY RUN ===")
        print(msg)
        print("=== ATTACHMENTS ===")
        print(args.attach or "(none)")
        return 0

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_ids = load_chat_ids()
    if not bot_token or not chat_ids:
        print(
            "ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set. "
            "Send /start to @trustfed_bot and run get_chat_id.py --save (or bot_server.py).",
            file=sys.stderr,
        )
        return 1

    try:
        for chat_id in chat_ids:
            res = telegram_send_message(bot_token, chat_id, msg, reply_markup=keyboard)
            print(
                f"Telegram message sent to {chat_id}: ok={res.get('ok')} "
                f"message_id={res['result']['message_id']}"
            )
            for raw_attach in args.attach:
                fp = Path(raw_attach)
                if not fp.is_absolute():
                    fp = (Path.cwd() / fp).resolve()
                if not fp.exists():
                    print(f"WARNING: Attachment not found, skipping: {fp}", file=sys.stderr)
                    continue
                fres = telegram_send_file(bot_token, chat_id, fp, caption=f"Attachment: {fp.name}")
                print(f"Sent attachment {fp.name} to {chat_id}: ok={fres.get('ok')}")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        print(f"Telegram API HTTP {exc.code}: {err_body}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

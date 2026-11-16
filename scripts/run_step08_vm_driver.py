"""Linux root driver verifying live OVS drop-rule mitigation & control-plane responsiveness.

Intended to run inside the Ubuntu 22.04 LTS VM via `scripts/run_step08_vm_testbed.sh`
while 3 OS-Ken controllers (ports 6653, 6654, 6655) serve switches s1, s2, s3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Append system dist-packages at the end of sys.path to access system Mininet
# without shadowing any virtualenv packages (e.g. numpy, pandas).
for p in ["/usr/lib/python3/dist-packages", "/usr/local/lib/python3/dist-packages"]:
    if p not in sys.path and Path(p).exists():
        sys.path.append(p)

from testbed.ryu_apps.model_deploy import MitigationPolicyManager  # noqa: E402
from testbed.topology.multi_tenant_topo import (  # noqa: E402
    TOPOLOGY_SPEC,
    run_isolation_verification,
    start_mininet_topology,
)

PING_RTT_RE = re.compile(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/([\d.]+)/")


def _parse_ping_stats(stdout: str) -> tuple[bool, float]:
    """Return `(all_received, avg_rtt_ms)` from ping output."""
    ok = " 0% packet loss" in stdout
    match = PING_RTT_RE.search(stdout)
    avg_ms = float(match.group(1)) if match else 0.0
    return ok, avg_ms


def _measure_ctrl_rss_mb(pid_file: Path) -> list[dict[str, Any]]:
    """Read controller PID RSS memory (MB) and CPU ticks from `/proc`."""
    if not pid_file.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in pid_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, pid_str = line.split(":", 1)
        pid = int(pid_str.strip())
        status_path = Path(f"/proc/{pid}/status")
        rss_kb = 0
        if status_path.exists():
            for sline in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if sline.startswith("VmRSS:"):
                    parts = sline.split()
                    if len(parts) >= 2:
                        rss_kb = int(parts[1])
                    break
        records.append({"controller": name, "pid": pid, "rss_mb": round(rss_kb / 1024.0, 2)})
    return records


def _cpu_agg_load_worker(stop_evt: threading.Event) -> None:
    """Simulate concurrent FL round crypto hashing & verification load on host CPU."""
    payload = os.urandom(4096)
    while not stop_evt.is_set():
        h = hashlib.sha256(payload).digest()
        for _ in range(150):
            h = hashlib.sha256(h + b"trustfed").digest()


def _measure_tenant_pings(net: Any, count: int = 5) -> dict[str, Any]:
    rtts: list[float] = []
    passed = 0
    total = 0
    for t_spec in TOPOLOGY_SPEC:
        ben = net.get(next(h.name for h in t_spec.hosts if h.role == "benign"))
        vic = net.get(next(h.name for h in t_spec.hosts if h.role == "victim"))
        total += 1
        out = ben.cmd(f"ping -c {count} -i 0.2 -W 1 {vic.IP()}")
        ok, avg_ms = _parse_ping_stats(out)
        if ok:
            passed += 1
        if avg_ms > 0.0:
            rtts.append(avg_ms)
    return {
        "probes_passed": passed,
        "probes_total": total,
        "mean_rtt_ms": round(sum(rtts) / max(1, len(rtts)), 3),
        "max_rtt_ms": round(max(rtts), 3) if rtts else 0.0,
    }


def _dump_ovs_flows(out_path: Path) -> str:
    blocks: list[str] = []
    for sw in ["s0", "s1", "s2", "s3"]:
        res = subprocess.run(
            ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", sw],
            capture_output=True,
            text=True,
            check=False,
        )
        blocks.append(f"=== OpenFlow 1.3 Flow Table: {sw} ===\n{res.stdout}{res.stderr}\n")
    text = "\n".join(blocks)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    res_dir = REPO_ROOT / "results" / "08_live"
    res_dir.mkdir(parents=True, exist_ok=True)
    pid_file = res_dir / "controller_pids.txt"

    print("[1/5] Starting multi-tenant Mininet + OVS topology for Step 08...", flush=True)
    net = start_mininet_topology(controller_ip="127.0.0.1")

    try:
        iso_stats = run_isolation_verification(net)
        print(f"      Isolation baseline: {iso_stats}", flush=True)

        print("[2/5] Measuring idle control-plane RTT and controller footprint...", flush=True)
        idle_pings = _measure_tenant_pings(net, count=5)
        ctrl_mem_idle = _measure_ctrl_rss_mb(pid_file)

        print(
            "[3/5] Measuring control-plane responsiveness during concurrent FL load...", flush=True
        )
        stop_evt = threading.Event()
        load_thread = threading.Thread(target=_cpu_agg_load_worker, args=(stop_evt,), daemon=True)
        load_thread.start()
        time.sleep(0.3)
        busy_pings = _measure_tenant_pings(net, count=5)
        ctrl_mem_busy = _measure_ctrl_rss_mb(pid_file)
        stop_evt.set()
        load_thread.join(timeout=2.0)

        print(
            "[4/5] Verifying pre-mitigation reachability + live OVS drop-rule enforcement...",
            flush=True,
        )
        pre_attack_ok = 0
        pre_benign_ok = 0
        for t_spec in TOPOLOGY_SPEC:
            vic = net.get(next(h.name for h in t_spec.hosts if h.role == "victim"))
            atk = net.get(next(h.name for h in t_spec.hosts if h.role == "attacker"))
            ben = net.get(next(h.name for h in t_spec.hosts if h.role == "benign"))
            if _parse_ping_stats(atk.cmd(f"ping -c 2 -W 1 {vic.IP()}"))[0]:
                pre_attack_ok += 1
            if _parse_ping_stats(ben.cmd(f"ping -c 2 -W 1 {vic.IP()}"))[0]:
                pre_benign_ok += 1

        policy = MitigationPolicyManager(
            apply_ovs_cli=True,
            priority=300,
            idle_timeout=60,
            hard_timeout=120,
        )
        install_latencies_ms: dict[str, float] = {}
        for t_spec in TOPOLOGY_SPEC:
            atk_spec = next(h for h in t_spec.hosts if h.role == "attacker")
            _, inst_ms, _ = policy.enforce_drop_rule(
                tenant_id=t_spec.tenant_id,
                ipv4_src=atk_spec.ip,
                prob_attack=0.99,
                bridge=t_spec.switch_name,
            )
            install_latencies_ms[t_spec.tenant_id] = round(inst_ms, 3)

        time.sleep(0.3)
        _dump_ovs_flows(res_dir / "ovs_drop_flows.txt")

        post_attack_blocked = 0
        post_benign_allowed = 0
        for t_spec in TOPOLOGY_SPEC:
            vic = net.get(next(h.name for h in t_spec.hosts if h.role == "victim"))
            atk = net.get(next(h.name for h in t_spec.hosts if h.role == "attacker"))
            ben = net.get(next(h.name for h in t_spec.hosts if h.role == "benign"))
            atk_out = atk.cmd(f"ping -c 3 -i 0.2 -W 1 {vic.IP()}")
            if "100% packet loss" in atk_out or "0 received" in atk_out:
                post_attack_blocked += 1
            ben_out = ben.cmd(f"ping -c 3 -i 0.2 -W 1 {vic.IP()}")
            if _parse_ping_stats(ben_out)[0]:
                post_benign_allowed += 1

        print("[5/5] Verifying automatic temporary drop-rule TTL expiry on s1...", flush=True)
        t1_spec = TOPOLOGY_SPEC[0]
        t1_atk = next(h for h in t1_spec.hosts if h.role == "attacker")
        t1_vic = net.get(next(h.name for h in t1_spec.hosts if h.role == "victim"))
        t1_atk_host = net.get(t1_atk.name)
        subprocess.run(
            [
                "ovs-ofctl",
                "-O",
                "OpenFlow13",
                "del-flows",
                t1_spec.switch_name,
                f"ip,nw_src={t1_atk.ip}",
            ],
            check=True,
        )
        subprocess.run(
            [
                "ovs-ofctl",
                "-O",
                "OpenFlow13",
                "add-flow",
                t1_spec.switch_name,
                f"priority=300,hard_timeout=4,ip,nw_src={t1_atk.ip},actions=drop",
            ],
            check=True,
        )
        during_expiry_blocked = "0 received" in t1_atk_host.cmd(
            f"ping -c 2 -i 0.2 -W 1 {t1_vic.IP()}"
        )
        time.sleep(5.0)
        after_expiry_restored = _parse_ping_stats(
            t1_atk_host.cmd(f"ping -c 2 -i 0.2 -W 1 {t1_vic.IP()}")
        )[0]

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "vm_environment": "Ubuntu 22.04 LTS + Mininet 2.3.0 + OVS 2.17.12 + OS-Ken 3.0.1",
            "isolation_stats": iso_stats,
            "control_plane": {
                "idle_ping": idle_pings,
                "under_fl_load_ping": busy_pings,
                "rtt_delta_ms": round(busy_pings["mean_rtt_ms"] - idle_pings["mean_rtt_ms"], 3),
                "controller_rss_idle_mb": ctrl_mem_idle,
                "controller_rss_busy_mb": ctrl_mem_busy,
            },
            "live_mitigation": {
                "tenants_tested": len(TOPOLOGY_SPEC),
                "pre_mitigation_attacker_reachable": pre_attack_ok,
                "pre_mitigation_benign_reachable": pre_benign_ok,
                "ovs_drop_install_latency_ms": install_latencies_ms,
                "mean_ovs_drop_install_ms": round(
                    sum(install_latencies_ms.values()) / max(1, len(install_latencies_ms)), 3
                ),
                "post_mitigation_attackers_blocked": post_attack_blocked,
                "attack_block_rate_pct": round(100.0 * post_attack_blocked / len(TOPOLOGY_SPEC), 2),
                "post_mitigation_benign_allowed": post_benign_allowed,
                "false_block_rate_pct": round(
                    100.0 * (len(TOPOLOGY_SPEC) - post_benign_allowed) / len(TOPOLOGY_SPEC), 2
                ),
                "ttl_expiry_blocked_before": bool(during_expiry_blocked),
                "ttl_expiry_restored_after": bool(after_expiry_restored),
            },
        }
        (res_dir / "vm_live_mitigation.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print("VM Step 08 summary:", json.dumps(summary, indent=2), flush=True)
    finally:
        net.stop()


if __name__ == "__main__":
    main()

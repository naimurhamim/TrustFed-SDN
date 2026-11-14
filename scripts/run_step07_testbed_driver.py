"""Linux root driver for Step 07 Multi-Tenant SDN Testbed capture and isolation verification.

Intended to be invoked by ``scripts/run_step07_testbed.sh`` inside the Ubuntu 22.04 VM
while the 3 OS-Ken controllers (ports 6653, 6654, 6655) are listening.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testbed.capture.flow_extractor import (  # noqa: E402
    extract_features_for_dataset,
    extract_flows_from_pcap,
    parse_openflow_stats_jsonl,
    verify_feature_alignment,
)
from testbed.topology.multi_tenant_topo import (  # noqa: E402
    CORE_SWITCH_SPEC,
    TOPOLOGY_SPEC,
    run_isolation_verification,
    start_mininet_topology,
)

SCENARIO_PLAN: dict[str, list[dict[str, Any]]] = {
    "tenant_1_finance": [
        {"scenario": "benign", "role": "benign", "label": "Benign", "y_bin": 0, "count": 220},
        {
            "scenario": "brute_force",
            "role": "attacker",
            "attack_type": "brute_force",
            "label": "BruteForce",
            "y_bin": 1,
            "count": 150,
        },
        {
            "scenario": "web_attack",
            "role": "attacker",
            "attack_type": "web_attack",
            "label": "WebAttack",
            "y_bin": 1,
            "count": 150,
        },
        {
            "scenario": "dos_http",
            "role": "attacker",
            "attack_type": "dos_http",
            "label": "DoS",
            "y_bin": 1,
            "count": 140,
        },
    ],
    "tenant_2_healthcare": [
        {"scenario": "benign", "role": "benign", "label": "Benign", "y_bin": 0, "count": 220},
        {
            "scenario": "syn_flood",
            "role": "attacker",
            "attack_type": "syn_flood",
            "label": "DoS",
            "y_bin": 1,
            "count": 200,
        },
        {
            "scenario": "udp_flood",
            "role": "attacker",
            "attack_type": "udp_flood",
            "label": "DDoS",
            "y_bin": 1,
            "count": 200,
        },
        {
            "scenario": "icmp_flood",
            "role": "attacker",
            "attack_type": "icmp_flood",
            "label": "DDoS",
            "y_bin": 1,
            "count": 40,
        },
    ],
    "tenant_3_education": [
        {"scenario": "benign", "role": "benign", "label": "Benign", "y_bin": 0, "count": 220},
        {
            "scenario": "port_scan",
            "role": "attacker",
            "attack_type": "port_scan",
            "label": "Probe",
            "y_bin": 1,
            "count": 220,
        },
        {
            "scenario": "dos_http",
            "role": "attacker",
            "attack_type": "dos_http",
            "label": "DoS",
            "y_bin": 1,
            "count": 130,
        },
        {
            "scenario": "udp_flood",
            "role": "attacker",
            "attack_type": "udp_flood",
            "label": "DDoS",
            "y_bin": 1,
            "count": 110,
        },
    ],
}


def _dump_ovs_flows(out_path: Path) -> None:
    """Capture OpenFlow 1.3 flow tables from s0, s1, s2, s3."""
    switches = [CORE_SWITCH_SPEC["switch_name"], *[t.switch_name for t in TOPOLOGY_SPEC]]
    blocks = []
    for sw in switches:
        res = subprocess.run(
            ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", sw],
            capture_output=True,
            text=True,
            check=False,
        )
        blocks.append(f"=== OpenFlow 1.3 Flow Table: {sw} ===\n{res.stdout}{res.stderr}\n")
    out_path.write_text("\n".join(blocks), encoding="utf-8")


def _run_host_scenario(
    net: Any,
    tenant_spec: Any,
    item: dict[str, Any],
    pcap_path: Path,
    python_bin: str,
) -> dict[str, Any]:
    """Capture one labeled scenario PCAP on the tenant's victim interface."""
    victim_host = net.get(next(h.name for h in tenant_spec.hosts if h.role == "victim"))
    victim_ip = next(h.ip for h in tenant_spec.hosts if h.role == "victim")
    src_role = item["role"]
    src_host = net.get(next(h.name for h in tenant_spec.hosts if h.role == src_role))
    iface = f"{victim_host.name}-eth0"

    import shutil

    cap_tmp_dir = Path("/tmp/trustfed_step07_pcaps")
    cap_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pcap = cap_tmp_dir / pcap_path.name
    if tmp_pcap.exists():
        tmp_pcap.unlink()

    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    cap_proc = victim_host.popen(
        ["tcpdump", "-U", "-n", "-s", "0", "-i", iface, "-w", str(tmp_pcap)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.6)

    start_utc = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    t0 = time.monotonic()
    stderr_text = ""
    try:
        if src_role == "benign":
            cmd = (
                f"PYTHONPATH='{REPO_ROOT}' '{python_bin}' -m testbed.traffic.benign_traffic "
                f"--mode client --target {victim_ip} --tenant {tenant_spec.tenant_id} "
                f"--flows {int(item['count'])} --seed 42"
            )
        else:
            cmd = (
                f"PYTHONPATH='{REPO_ROOT}' '{python_bin}' -m testbed.traffic.attack_traffic "
                f"--target {victim_ip} --attack-type {item['attack_type']} "
                f"--tenant {tenant_spec.tenant_id} --count {int(item['count'])} --seed 42"
            )
        stdout = src_host.cmd(cmd)
    finally:
        time.sleep(0.8)
        cap_proc.send_signal(signal.SIGINT)
        try:
            _, err_bytes = cap_proc.communicate(timeout=5)
            stderr_text = err_bytes.decode("utf-8", errors="ignore")
        except subprocess.TimeoutExpired:
            cap_proc.kill()
            cap_proc.wait()
        if tmp_pcap.exists():
            shutil.copy2(tmp_pcap, pcap_path)
        time.sleep(0.2)

    elapsed = round(time.monotonic() - t0, 3)
    if not pcap_path.exists() or pcap_path.stat().st_size < 25:
        raise RuntimeError(f"tcpdump failed for {pcap_path.name}: {stderr_text}")
    return {
        "tenant_id": tenant_spec.tenant_id,
        "scenario": item["scenario"],
        "label": item["label"],
        "y_binary": int(item["y_bin"]),
        "source_host": src_host.name,
        "target_ip": victim_ip,
        "pcap_path": str(pcap_path.relative_to(REPO_ROOT)),
        "started_utc": start_utc,
        "elapsed_sec": elapsed,
        "generator_tail": stdout.strip().splitlines()[-1] if stdout.strip() else "",
    }


def main() -> None:
    live_root = REPO_ROOT / "data" / "live"
    raw_pcap_dir = live_root / "raw_pcaps"
    res_dir = REPO_ROOT / "results" / "07_testbed"
    raw_pcap_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    print("[1/5] Building multi-tenant Mininet + OVS topology...", flush=True)
    net = start_mininet_topology(controller_ip="127.0.0.1")

    schedule_records: list[dict[str, Any]] = []
    server_procs: list[Any] = []
    try:
        print("[2/5] Running intra-tenant & cross-tenant isolation verification...", flush=True)
        iso_stats = run_isolation_verification(net)
        (res_dir / "isolation_test.json").write_text(
            json.dumps(iso_stats, indent=2), encoding="utf-8"
        )
        print(f"      Isolation stats: {iso_stats}", flush=True)

        print("[3/5] Starting victim multi-service responders inside h1..h3_victim...", flush=True)
        for tenant in TOPOLOGY_SPEC:
            victim_spec = next(h for h in tenant.hosts if h.role == "victim")
            victim_host = net.get(victim_spec.name)
            env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
            srv = victim_host.popen(
                [
                    python_bin,
                    "-m",
                    "testbed.traffic.benign_traffic",
                    "--mode",
                    "server",
                    "--target",
                    victim_spec.ip,
                    "--tenant",
                    tenant.tenant_id,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            server_procs.append(srv)
        time.sleep(1.0)

        print(
            "[4/5] Running labeled benign & attack traffic schedules + PCAP capture...", flush=True
        )
        for tenant in TOPOLOGY_SPEC:
            for item in SCENARIO_PLAN[tenant.tenant_id]:
                pcap_path = raw_pcap_dir / f"{tenant.tenant_id}__{item['scenario']}.pcap"
                print(
                    f"      -> {tenant.tenant_id} | scenario={item['scenario']} "
                    f"| label={item['label']} | count={item['count']}",
                    flush=True,
                )
                rec = _run_host_scenario(net, tenant, item, pcap_path, python_bin)
                schedule_records.append(rec)

        _dump_ovs_flows(res_dir / "ovs_flows_snapshot.txt")
    finally:
        for srv in server_procs:
            srv.terminate()
        net.stop()

    (res_dir / "schedule_log.json").write_text(
        json.dumps(schedule_records, indent=2), encoding="utf-8"
    )

    print("[5/5] Extracting CICFlowMeter-compatible features + OpenFlow stats...", flush=True)
    import pandas as pd

    all_frames: list[pd.DataFrame] = []
    for rec in schedule_records:
        pcap_full = REPO_ROOT / rec["pcap_path"]
        df = extract_flows_from_pcap(pcap_full)
        if len(df) == 0:
            continue
        df["tenant_id"] = rec["tenant_id"]
        df["scenario"] = rec["scenario"]
        df["canonical_label"] = rec["label"]
        df["y_binary"] = rec["y_binary"]
        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No flows extracted from testbed PCAP captures.")

    full_df = pd.concat(all_frames, ignore_index=True)
    full_csv = live_root / "live_flows_full.csv"
    full_df.to_csv(full_csv, index=False)

    insdn_align = verify_feature_alignment(full_df, REPO_ROOT / "configs/data/features_insdn.yaml")
    aseados_align = verify_feature_alignment(
        full_df, REPO_ROOT / "configs/data/features_aseados_sdn_iot.yaml"
    )

    insdn_df = extract_features_for_dataset(full_df, REPO_ROOT / "configs/data/features_insdn.yaml")
    insdn_df["tenant_id"] = full_df["tenant_id"]
    insdn_df["scenario"] = full_df["scenario"]
    insdn_df["canonical_label"] = full_df["canonical_label"]
    insdn_df["y_binary"] = full_df["y_binary"]
    insdn_df.to_csv(live_root / "live_flows_insdn.csv", index=False)

    ase_df = extract_features_for_dataset(
        full_df, REPO_ROOT / "configs/data/features_aseados_sdn_iot.yaml"
    )
    ase_df["tenant_id"] = full_df["tenant_id"]
    ase_df["scenario"] = full_df["scenario"]
    ase_df["canonical_label"] = full_df["canonical_label"]
    ase_df["y_binary"] = full_df["y_binary"]
    ase_df.to_csv(live_root / "live_flows_aseados.csv", index=False)

    of_stats_summary = {}
    for idx, tenant in enumerate(TOPOLOGY_SPEC, start=1):
        stats_file = res_dir / f"of_stats_t{idx}.jsonl"
        of_stats_summary[tenant.tenant_id] = parse_openflow_stats_jsonl(stats_file)

    capture_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "isolation_level": "L1",
        "isolation_stats": iso_stats,
        "total_live_flows": int(len(full_df)),
        "per_tenant_counts": {k: int(v) for k, v in full_df["tenant_id"].value_counts().items()},
        "per_class_counts": {
            k: int(v) for k, v in full_df["canonical_label"].value_counts().items()
        },
        "binary_counts": {str(k): int(v) for k, v in full_df["y_binary"].value_counts().items()},
        "feature_alignment_insdn": insdn_align,
        "feature_alignment_aseados": aseados_align,
        "openflow_controller_stats": of_stats_summary,
    }
    (res_dir / "capture_summary.json").write_text(
        json.dumps(capture_summary, indent=2), encoding="utf-8"
    )
    print(f"DONE: Extracted {len(full_df)} labeled live SDN flows -> {live_root}", flush=True)


if __name__ == "__main__":
    main()

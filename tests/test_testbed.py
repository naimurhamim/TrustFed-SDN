"""Step 07 unit tests for safety target guard, multi-tenant topology, and flow extractor."""

from __future__ import annotations

import socket
import struct
from pathlib import Path

import pytest

from testbed.capture.flow_extractor import (
    extract_features_for_dataset,
    extract_flows_from_pcap,
    parse_openflow_stats_jsonl,
    verify_feature_alignment,
)
from testbed.topology.multi_tenant_topo import (
    CORE_SWITCH_SPEC,
    TOPOLOGY_SPEC,
    get_all_host_ips,
    render_ovs_core_isolation_commands,
)
from testbed.traffic.attack_traffic import run_attack_traffic
from testbed.traffic.safety_guard import SafetyGuardError, validate_testbed_target


def _build_eth_ipv4_tcp_frame(
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    payload: bytes,
    *,
    win: int = 64240,
    flags: int = 0x18,
) -> bytes:
    """Construct a valid Ethernet + IPv4 + TCP frame for unit testing."""
    eth = b"\x00\x00\x00\x01\x00\x50\x00\x00\x00\x01\x00\x10\x08\x00"
    tcp_hdr_len = 20
    ip_total_len = 20 + tcp_hdr_len + len(payload)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        ip_total_len,
        1,
        0,
        64,
        6,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    offset_flags = (5 << 12) | flags
    tcp_hdr = struct.pack(
        "!HHIIHHHH",
        sport,
        dport,
        1000,
        2000,
        offset_flags,
        win,
        0,
        0,
    )
    return eth + ip_hdr + tcp_hdr + payload


def _build_eth_ipv4_udp_frame(
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    payload: bytes,
) -> bytes:
    """Construct a valid Ethernet + IPv4 + UDP frame for unit testing."""
    eth = b"\x00\x00\x00\x02\x00\x50\x00\x00\x00\x02\x00\x10\x08\x00"
    udp_len = 8 + len(payload)
    ip_total_len = 20 + udp_len
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        ip_total_len,
        2,
        0,
        64,
        17,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    udp_hdr = struct.pack("!HHHH", sport, dport, udp_len, 0)
    return eth + ip_hdr + udp_hdr + payload


def _write_test_pcap(path: Path, packets: list[tuple[int, int, bytes]]) -> None:
    """Write classic little-endian microsecond libpcap file."""
    header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    chunks = [header]
    for ts_sec, ts_usec, frame in packets:
        chunks.append(struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame)))
        chunks.append(frame)
    path.write_bytes(b"".join(chunks))


def test_safety_guard_accepts_valid_tenant_targets() -> None:
    """Every benign/victim/attacker IP inside 10.10.1-3.0/24 passes validation."""
    assert (
        str(validate_testbed_target("10.10.1.50", tenant_id="tenant_1_finance", verbose=False))
        == "10.10.1.50"
    )
    assert (
        str(validate_testbed_target("10.10.2.100", tenant_id="tenant_2_healthcare", verbose=False))
        == "10.10.2.100"
    )
    assert (
        str(validate_testbed_target("10.10.3.10", tenant_id="tenant_3_education", verbose=False))
        == "10.10.3.10"
    )


@pytest.mark.parametrize(
    "bad_ip",
    [
        "8.8.8.8",
        "1.1.1.1",
        "192.168.1.1",
        "127.0.0.1",
        "10.10.4.50",
        "10.10.1.0",
        "10.10.1.255",
        "not-an-ip",
        "::1",
    ],
)
def test_safety_guard_rejects_outside_and_invalid_targets(bad_ip: str) -> None:
    """Any IP outside 10.10.1-3.0/24 or malformed must fail immediately."""
    with pytest.raises(SafetyGuardError):
        validate_testbed_target(bad_ip, verbose=False)


def test_safety_guard_rejects_tenant_mismatch_and_blocks_attack() -> None:
    """Attack generator refuses wrong-tenant or external IP before emitting traffic."""
    with pytest.raises(SafetyGuardError):
        validate_testbed_target("10.10.2.50", tenant_id="tenant_1_finance", verbose=False)
    with pytest.raises(SafetyGuardError):
        run_attack_traffic("8.8.8.8", attack_type="syn_flood", count=5)


def test_topology_spec_and_isolation_commands() -> None:
    """Multi-tenant topology defines 3 isolated tenants, 9 hosts, and drop rules."""
    assert CORE_SWITCH_SPEC["switch_name"] == "s0"
    assert len(TOPOLOGY_SPEC) == 3
    all_ips = get_all_host_ips()
    assert len(all_ips) == 9
    assert len(set(all_ips.values())) == 9
    for host_name, ip_str in all_ips.items():
        # Linux kernel IFNAMSIZ-1 interface name length limit (15 chars) for <host>-eth0
        assert len(f"{host_name}-eth0") <= 15
        validate_testbed_target(ip_str, verbose=False)

    cmds = render_ovs_core_isolation_commands()
    # 1 default drop + 6 directed pairs * 2 (IP + ARP) = 13 OVS flow commands
    assert len(cmds) == 13


def test_hybrid_flow_extractor_matches_step01_features(tmp_path: Path) -> None:
    """Extracted PCAP flows match exact names, order, and units of Step 01 configs."""
    pcap_file = tmp_path / "sample.pcap"
    frames = [
        (
            1700000000,
            0,
            _build_eth_ipv4_tcp_frame(
                "10.10.1.10",
                "10.10.1.50",
                41000,
                8080,
                b"GET / HTTP/1.1\r\n",
                win=64240,
                flags=0x02,
            ),
        ),
        (
            1700000000,
            1000,
            _build_eth_ipv4_tcp_frame(
                "10.10.1.50",
                "10.10.1.10",
                8080,
                41000,
                b"HTTP/1.1 200 OK\r\n" * 4,
                win=65535,
                flags=0x12,
            ),
        ),
        (
            1700000000,
            4000,
            _build_eth_ipv4_tcp_frame(
                "10.10.1.10", "10.10.1.50", 41000, 8080, b"", win=64240, flags=0x11
            ),
        ),
        (
            1700000001,
            0,
            _build_eth_ipv4_udp_frame("10.10.2.100", "10.10.2.50", 51000, 5353, b"X" * 128),
        ),
        (
            1700000001,
            2500,
            _build_eth_ipv4_udp_frame("10.10.2.100", "10.10.2.50", 51000, 5353, b"X" * 128),
        ),
    ]
    _write_test_pcap(pcap_file, frames)

    df = extract_flows_from_pcap(pcap_file)
    assert len(df) == 2

    root = Path(__file__).resolve().parents[1]
    for yaml_rel in (
        "configs/data/features_insdn.yaml",
        "configs/data/features_aseados_sdn_iot.yaml",
    ):
        report = verify_feature_alignment(df, root / yaml_rel)
        assert report["exact_names_and_order"] is True
        assert report["nan_count"] == 0
        assert report["inf_count"] == 0
        sub = extract_features_for_dataset(df, root / yaml_rel)
        assert len(sub) == 2

    tcp_row = df[df["Protocol"] == 6].iloc[0]
    assert tcp_row["Flow Duration"] == pytest.approx(4000.0)
    assert tcp_row["Init Bwd Win Byts"] == pytest.approx(65535.0)
    assert tcp_row["Fwd Header Len"] == pytest.approx(40.0)
    assert tcp_row["Bwd Header Len"] == pytest.approx(20.0)

    udp_row = df[df["Protocol"] == 17].iloc[0]
    assert udp_row["Init Bwd Win Byts"] == pytest.approx(-1.0)
    assert udp_row["Flow Pkts/s"] == pytest.approx(2.0 / 0.0025)

    # Also test OpenFlow stats JSONL summary helper
    jsonl_file = tmp_path / "stats.jsonl"
    jsonl_file.write_text(
        '{"type":"flow_stats","dpid":1,"flows":[{"priority":10},{"priority":10}]}\n'
        '{"type":"port_stats","dpid":1,"ports":[{"port_no":1,"rx_bytes":2048}]}\n',
        encoding="utf-8",
    )
    of_summary = parse_openflow_stats_jsonl(jsonl_file)
    assert of_summary["max_active_flows"] == 2
    assert of_summary["max_port_rx_bytes"] == 2048

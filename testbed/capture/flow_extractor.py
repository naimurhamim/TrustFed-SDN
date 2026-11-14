"""CICFlowMeter-compatible PCAP + OpenFlow 1.3 hybrid flow feature extractor.

Computes exact training-aligned flow features (``configs/data/features_insdn.yaml``
19 features and ``configs/data/features_aseados_sdn_iot.yaml`` 12 features) directly
from raw libpcap ``.pcap`` captures in pure Python, alongside SDN-native OpenFlow
1.3 flow/port telemetry parsed from ``TenantController13`` JSONL logs.

Verified units against ``InSDN`` and ``ASEADOS-SDN-IoT`` raw CSVs:
- ``Flow Duration``, ``*IAT*``: microseconds (us).
- ``Fwd Header Len``, ``Bwd Header Len``: L4 transport header bytes (TCP/UDP).
- ``*Pkt Len*``, ``TotLen*``, ``Subflow Bwd Byts``: L4 payload bytes.
- ``Init Bwd Win Byts``: initial backward TCP window size (``-1`` when absent).
- ``*Pkts/s``: packets per second (``count / (Flow Duration / 1e6)``).
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LINKTYPE_ETHERNET = 1
LINKTYPE_LINUX_SLL = 113

# Full superset of CICFlowMeter columns required by InSDN (19) and ASEADOS-SDN-IoT (12)
ALL_EXTRACTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Mean",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Max",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "Pkt Size Avg",
    "Bwd Seg Size Avg",
    "Subflow Bwd Byts",
    "Init Bwd Win Byts",
)


@dataclass(frozen=True)
class ParsedPacket:
    """Normalized IPv4 L3/L4 packet metadata parsed from a PCAP record."""

    ts_us: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int
    l4_header_len: int
    payload_len: int
    tcp_window: int
    tcp_flags: int


def parse_pcap_packets(pcap_path: str | Path) -> list[ParsedPacket]:
    """Parse IPv4 packets from a standard ``libpcap`` capture file.

    Supports both little-endian and big-endian magic headers (microsecond and
    nanosecond timestamps) on Ethernet (``LINKTYPE_ETHERNET``) and Linux cooked
    (``LINKTYPE_LINUX_SLL``) interfaces.
    """
    raw = Path(pcap_path).read_bytes()
    if len(raw) < 24:
        return []

    magic_le = raw[:4]
    if magic_le == b"\xd4\xc3\xb2\xa1":
        endian, ts_scale_div = "<", 1
    elif magic_le == b"\xa1\xb2\xc3\xd4":
        endian, ts_scale_div = ">", 1
    elif magic_le == b"\x4d\x3c\xb2\xa1":
        endian, ts_scale_div = "<", 1000
    elif magic_le == b"\xa1\xb2\x3c\x4d":
        endian, ts_scale_div = ">", 1000
    else:
        raise ValueError(f"Unsupported capture format (expected classic libpcap) in {pcap_path}")

    _, _, _, _, _, _, network = struct.unpack(f"{endian}IHHIIII", raw[:24])
    if network not in (LINKTYPE_ETHERNET, LINKTYPE_LINUX_SLL):
        raise ValueError(f"Unsupported PCAP linktype {network} in {pcap_path}")

    l2_offset = 14 if network == LINKTYPE_ETHERNET else 16
    packets: list[ParsedPacket] = []
    pos = 24
    total_bytes = len(raw)

    while pos + 16 <= total_bytes:
        ts_sec, ts_sub, incl_len, _ = struct.unpack(f"{endian}IIII", raw[pos : pos + 16])
        pos += 16
        frame = raw[pos : pos + incl_len]
        pos += incl_len
        if len(frame) < l2_offset:
            continue

        eth_type = (
            struct.unpack("!H", frame[12:14])[0]
            if network == LINKTYPE_ETHERNET
            else struct.unpack("!H", frame[14:16])[0]
        )
        ip_offset = l2_offset
        if eth_type == 0x8100 and len(frame) >= l2_offset + 4:
            eth_type = struct.unpack("!H", frame[l2_offset + 2 : l2_offset + 4])[0]
            ip_offset += 4
        if eth_type != 0x0800 or len(frame) < ip_offset + 20:
            continue

        ip_hdr = frame[ip_offset:]
        ihl = (ip_hdr[0] & 0x0F) * 4
        if ihl < 20 or len(ip_hdr) < ihl:
            continue
        ip_total_len = struct.unpack("!H", ip_hdr[2:4])[0]
        proto = ip_hdr[9]
        src_ip = socket.inet_ntoa(ip_hdr[12:16])
        dst_ip = socket.inet_ntoa(ip_hdr[16:20])

        l4 = ip_hdr[ihl:]
        src_port, dst_port = 0, 0
        l4_hdr_len = 0
        tcp_win = -1
        tcp_flags = 0

        if proto == 6 and len(l4) >= 20:
            src_port, dst_port, _, _, offset_flags, tcp_win = struct.unpack("!HHIIHH", l4[:16])
            l4_hdr_len = ((offset_flags >> 12) & 0x0F) * 4
            if l4_hdr_len < 20:
                l4_hdr_len = 20
            tcp_flags = offset_flags & 0x3F
        elif proto == 17 and len(l4) >= 8:
            src_port, dst_port = struct.unpack("!HH", l4[:4])
            l4_hdr_len = 8

        payload_len = max(0, ip_total_len - ihl - l4_hdr_len)
        ts_us = ts_sec * 1_000_000 + (ts_sub // ts_scale_div)
        packets.append(
            ParsedPacket(
                ts_us=int(ts_us),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=int(src_port),
                dst_port=int(dst_port),
                proto=int(proto),
                l4_header_len=int(l4_hdr_len),
                payload_len=int(payload_len),
                tcp_window=int(tcp_win),
                tcp_flags=int(tcp_flags),
            )
        )

    packets.sort(key=lambda p: p.ts_us)
    return packets


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std_var(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    var = float(np.var(arr, ddof=1))
    return float(np.sqrt(max(0.0, var))), var


def _summarize_flow(
    pkts: list[ParsedPacket],
    fwd_src_ip: str,
    fwd_src_port: int,
    fwd_dst_ip: str,
    fwd_dst_port: int,
    proto: int,
) -> dict[str, Any]:
    """Compute bidirectional CICFlowMeter features for one completed flow."""
    pkts_sorted = sorted(pkts, key=lambda p: p.ts_us)
    first_ts = pkts_sorted[0].ts_us
    last_ts = pkts_sorted[-1].ts_us
    duration_us = float(max(0, last_ts - first_ts))
    dur_sec = duration_us / 1_000_000.0

    fwd_pkts = [p for p in pkts_sorted if p.src_ip == fwd_src_ip and p.src_port == fwd_src_port]
    bwd_pkts = [p for p in pkts_sorted if p not in fwd_pkts]

    all_ts = [p.ts_us for p in pkts_sorted]
    flow_iats = [float(all_ts[i] - all_ts[i - 1]) for i in range(1, len(all_ts))]

    fwd_ts = [p.ts_us for p in fwd_pkts]
    fwd_iats = [float(fwd_ts[i] - fwd_ts[i - 1]) for i in range(1, len(fwd_ts))]

    bwd_ts = [p.ts_us for p in bwd_pkts]
    bwd_iats = [float(bwd_ts[i] - bwd_ts[i - 1]) for i in range(1, len(bwd_ts))]

    all_lens = [float(p.payload_len) for p in pkts_sorted]
    # Matches CICFlowMeter FlowLengthStats (initial 0 anchor included in Pkt Len Mean/Std/Var)
    cic_lens = [0.0, *all_lens]
    pkt_std, pkt_var = _safe_std_var(cic_lens)

    bwd_lens = [float(p.payload_len) for p in bwd_pkts]
    init_bwd_win = -1
    if proto == 6:
        for bp in bwd_pkts:
            if bp.tcp_window >= 0:
                init_bwd_win = bp.tcp_window
                break

    tot_fwd = len(fwd_pkts)
    tot_bwd = len(bwd_pkts)
    tot_all = tot_fwd + tot_bwd

    record: dict[str, Any] = {
        "Src IP": fwd_src_ip,
        "Src Port": fwd_src_port,
        "Dst IP": fwd_dst_ip,
        "Dst Port": fwd_dst_port,
        "Protocol": proto,
        "Timestamp_us": first_ts,
        "Flow Duration": duration_us,
        "Tot Fwd Pkts": tot_fwd,
        "Tot Bwd Pkts": tot_bwd,
        "TotLen Fwd Pkts": float(sum(p.payload_len for p in fwd_pkts)),
        "TotLen Bwd Pkts": float(sum(p.payload_len for p in bwd_pkts)),
        "Bwd Pkt Len Max": max(bwd_lens) if bwd_lens else 0.0,
        "Bwd Pkt Len Mean": _safe_mean(bwd_lens),
        "Flow Pkts/s": float(tot_all / dur_sec) if dur_sec > 0.0 else 0.0,
        "Flow IAT Mean": _safe_mean(flow_iats),
        "Flow IAT Std": _safe_std_var(flow_iats)[0],
        "Flow IAT Max": max(flow_iats) if flow_iats else 0.0,
        "Flow IAT Min": min(flow_iats) if flow_iats else 0.0,
        "Fwd IAT Tot": float(sum(fwd_iats)),
        "Bwd IAT Tot": float(sum(bwd_iats)),
        "Bwd IAT Mean": _safe_mean(bwd_iats),
        "Bwd IAT Max": max(bwd_iats) if bwd_iats else 0.0,
        "Fwd Header Len": float(sum(p.l4_header_len for p in fwd_pkts)),
        "Bwd Header Len": float(sum(p.l4_header_len for p in bwd_pkts)),
        "Fwd Pkts/s": float(tot_fwd / dur_sec) if dur_sec > 0.0 else 0.0,
        "Bwd Pkts/s": float(tot_bwd / dur_sec) if dur_sec > 0.0 else 0.0,
        "Pkt Len Min": min(all_lens) if all_lens else 0.0,
        "Pkt Len Max": max(all_lens) if all_lens else 0.0,
        "Pkt Len Mean": float(sum(all_lens) / (tot_all + 1)),
        "Pkt Len Std": pkt_std,
        "Pkt Len Var": pkt_var,
        "Pkt Size Avg": float(sum(all_lens) / max(1, tot_all)),
        "Bwd Seg Size Avg": _safe_mean(bwd_lens),
        "Subflow Bwd Byts": float(sum(p.payload_len for p in bwd_pkts)),
        "Init Bwd Win Byts": float(init_bwd_win),
    }
    return record


def extract_flows_from_pcap(
    pcap_path: str | Path,
    flow_timeout_us: int = 120_000_000,
) -> pd.DataFrame:
    """Aggregate packets from ``pcap_path`` into bidirectional flow feature rows."""
    packets = parse_pcap_packets(pcap_path)
    if not packets:
        return pd.DataFrame(
            columns=["Src IP", "Dst IP", "Protocol", *ALL_EXTRACTED_FEATURE_COLUMNS]
        )

    active_flows: dict[tuple[tuple[str, int], tuple[str, int], int], dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []

    for pkt in packets:
        ep_a = (pkt.src_ip, pkt.src_port)
        ep_b = (pkt.dst_ip, pkt.dst_port)
        key = (ep_a, ep_b, pkt.proto) if ep_a <= ep_b else (ep_b, ep_a, pkt.proto)

        state = active_flows.get(key)
        if state is not None and (pkt.ts_us - state["start_ts"]) > flow_timeout_us:
            completed.append(
                _summarize_flow(
                    state["pkts"],
                    state["fwd_src_ip"],
                    state["fwd_src_port"],
                    state["fwd_dst_ip"],
                    state["fwd_dst_port"],
                    pkt.proto,
                )
            )
            state = None

        if state is None:
            state = {
                "start_ts": pkt.ts_us,
                "fwd_src_ip": pkt.src_ip,
                "fwd_src_port": pkt.src_port,
                "fwd_dst_ip": pkt.dst_ip,
                "fwd_dst_port": pkt.dst_port,
                "pkts": [pkt],
            }
            active_flows[key] = state
        else:
            state["pkts"].append(pkt)

        # Close TCP flow on FIN (0x01) or RST (0x04) like CICFlowMeter
        if pkt.proto == 6 and (pkt.tcp_flags & 0x05):
            completed.append(
                _summarize_flow(
                    state["pkts"],
                    state["fwd_src_ip"],
                    state["fwd_src_port"],
                    state["fwd_dst_ip"],
                    state["fwd_dst_port"],
                    pkt.proto,
                )
            )
            del active_flows[key]

    for (_, _, proto), state in active_flows.items():
        completed.append(
            _summarize_flow(
                state["pkts"],
                state["fwd_src_ip"],
                state["fwd_src_port"],
                state["fwd_dst_ip"],
                state["fwd_dst_port"],
                proto,
            )
        )

    return pd.DataFrame(completed)


def load_selected_feature_names(feature_yaml_path: str | Path) -> list[str]:
    """Load ordered selected feature list saved in Step 01 YAML artifacts."""
    payload = yaml.safe_load(Path(feature_yaml_path).read_text(encoding="utf-8"))
    return list(payload["selected_features"])


def extract_features_for_dataset(
    flows_df: pd.DataFrame,
    feature_yaml_path: str | Path,
) -> pd.DataFrame:
    """Project extracted flows onto exact Step 01 training feature names and order."""
    cols = load_selected_feature_names(feature_yaml_path)
    missing = [c for c in cols if c not in flows_df.columns]
    if missing:
        raise KeyError(f"Live extracted flows missing required features: {missing}")
    return flows_df[cols].astype(np.float64).copy()


def verify_feature_alignment(
    flows_df: pd.DataFrame,
    feature_yaml_path: str | Path,
) -> dict[str, Any]:
    """Verify live extracted feature names, order, and finiteness against Step 01 spec."""
    expected = load_selected_feature_names(feature_yaml_path)
    aligned = extract_features_for_dataset(flows_df, feature_yaml_path)
    actual = list(aligned.columns)
    values = aligned.to_numpy(dtype=np.float64)
    return {
        "feature_yaml": str(feature_yaml_path),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "exact_names_and_order": actual == expected,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "num_rows": int(len(aligned)),
    }


def parse_openflow_stats_jsonl(jsonl_path: str | Path) -> dict[str, Any]:
    """Summarize SDN-native OpenFlow 1.3 FlowStats and PortStats JSONL telemetry."""
    path = Path(jsonl_path)
    if not path.exists():
        return {
            "flow_stat_polls": 0,
            "port_stat_polls": 0,
            "max_active_flows": 0,
            "max_port_rx_bytes": 0,
        }

    flow_polls = 0
    port_polls = 0
    max_flows = 0
    max_rx_bytes = 0
    dpids: set[int] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        dpids.add(int(rec.get("dpid", 0)))
        if rec.get("type") == "flow_stats":
            flow_polls += 1
            max_flows = max(max_flows, len(rec.get("flows", [])))
        elif rec.get("type") == "port_stats":
            port_polls += 1
            total_rx = sum(int(p.get("rx_bytes", 0)) for p in rec.get("ports", []))
            max_rx_bytes = max(max_rx_bytes, total_rx)

    return {
        "dpids": sorted(dpids),
        "flow_stat_polls": flow_polls,
        "port_stat_polls": port_polls,
        "max_active_flows": max_flows,
        "max_port_rx_bytes": max_rx_bytes,
    }

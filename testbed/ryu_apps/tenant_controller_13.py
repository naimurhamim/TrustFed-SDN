"""Per-tenant OpenFlow 1.3 SDN controller app for TrustFed-SDN (OS-Ken / RYU API).

Provides:
1. L2/L3 OpenFlow 1.3 switching within the owning tenant subnet.
2. Default cross-tenant IPv4 & ARP isolation drop rules (priority=200).
3. Per-5-tuple IPv4 flow rule installation so OVS maintains granular packet/byte stats.
4. Periodic OpenFlow 1.3 FlowStats + PortStats collection written to JSONL.
5. Dynamic mitigation block rule helper + policy file watcher for Step 08.
"""

from __future__ import annotations

import ipaddress
import json
import os
import time
from pathlib import Path
from typing import Any

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.lib import hub
from os_ken.lib.packet import arp, ether_types, ethernet, ipv4, packet, tcp, udp
from os_ken.ofproto import ofproto_v1_3

DEFAULT_TENANT_SUBNETS = (
    "10.10.1.0/24",
    "10.10.2.0/24",
    "10.10.3.0/24",
)


def _same_tenant_subnet(src_ip: str, dst_ip: str, subnets: tuple[str, ...]) -> bool:
    """Return True if ``src_ip`` and ``dst_ip`` belong to the same tenant /24 subnet."""
    try:
        src = ipaddress.ip_address(src_ip)
        dst = ipaddress.ip_address(dst_ip)
    except ValueError:
        return False
    for cidr in subnets:
        net = ipaddress.ip_network(cidr)
        if src in net:
            return dst in net
    return False


class TenantController13(app_manager.OSKenApp):  # type: ignore[misc]
    """OpenFlow 1.3 tenant switch controller with isolation, flow telemetry, and blocking."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mac_to_port: dict[int, dict[str, int]] = {}
        self.datapaths: dict[int, Any] = {}
        self.stats_log_path = Path(
            os.environ.get("TENANT_FLOW_STATS_LOG", "tenant_flow_stats.jsonl")
        )
        self.blocklist_path = os.environ.get("TENANT_BLOCKLIST_JSON", "")
        self.poll_interval_sec = float(os.environ.get("TENANT_STATS_POLL_SEC", "2.0"))
        self.installed_blocks: set[str] = set()
        self.monitor_thread = hub.spawn(self._monitor_loop)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)  # type: ignore[misc]
    def switch_features_handler(self, ev: Any) -> None:
        """Install table-miss controller punt and default cross-tenant drop rules."""
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Priority 0: table-miss -> send unmatched packet to controller
        match_all = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        datapath.send_msg(
            parser.OFPFlowMod(datapath=datapath, priority=0, match=match_all, instructions=inst)
        )

        # Priority 200: cross-tenant IPv4 isolation drop rules
        for src_cidr in DEFAULT_TENANT_SUBNETS:
            for dst_cidr in DEFAULT_TENANT_SUBNETS:
                if src_cidr == dst_cidr:
                    continue
                src_net = ipaddress.ip_network(src_cidr)
                dst_net = ipaddress.ip_network(dst_cidr)
                drop_match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=(str(src_net.network_address), str(src_net.netmask)),
                    ipv4_dst=(str(dst_net.network_address), str(dst_net.netmask)),
                )
                datapath.send_msg(
                    parser.OFPFlowMod(
                        datapath=datapath,
                        priority=200,
                        match=drop_match,
                        instructions=[],
                    )
                )

    def install_mitigation_drop_rule(
        self,
        datapath: Any,
        ipv4_src: str,
        priority: int = 300,
        idle_timeout: int = 120,
    ) -> None:
        """Install an OpenFlow 1.3 drop rule for a flagged malicious source IP."""
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=ipv4_src)
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            idle_timeout=idle_timeout,
            match=match,
            instructions=[],
        )
        datapath.send_msg(mod)
        self.installed_blocks.add(f"{datapath.id}:{ipv4_src}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)  # type: ignore[misc]
    def packet_in_handler(self, ev: Any) -> None:
        """Enforce tenant subnet isolation, learn MACs, and install granular OpenFlow rules."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        arp_pkt = pkt.get_protocol(arp.arp)
        if ip_pkt is not None and not _same_tenant_subnet(
            ip_pkt.src, ip_pkt.dst, DEFAULT_TENANT_SUBNETS
        ):
            return
        if arp_pkt is not None and not _same_tenant_subnet(
            arp_pkt.src_ip, arp_pkt.dst_ip, DEFAULT_TENANT_SUBNETS
        ):
            return

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][eth.src] = in_port
        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
            if ip_pkt is not None:
                tcp_pkt = pkt.get_protocol(tcp.tcp)
                udp_pkt = pkt.get_protocol(udp.udp)
                if tcp_pkt is not None:
                    match = parser.OFPMatch(
                        in_port=in_port,
                        eth_type=ether_types.ETH_TYPE_IP,
                        ip_proto=ip_pkt.proto,
                        ipv4_src=ip_pkt.src,
                        ipv4_dst=ip_pkt.dst,
                        tcp_src=tcp_pkt.src_port,
                        tcp_dst=tcp_pkt.dst_port,
                    )
                elif udp_pkt is not None:
                    match = parser.OFPMatch(
                        in_port=in_port,
                        eth_type=ether_types.ETH_TYPE_IP,
                        ip_proto=ip_pkt.proto,
                        ipv4_src=ip_pkt.src,
                        ipv4_dst=ip_pkt.dst,
                        udp_src=udp_pkt.src_port,
                        udp_dst=udp_pkt.dst_port,
                    )
                else:
                    match = parser.OFPMatch(
                        in_port=in_port,
                        eth_type=ether_types.ETH_TYPE_IP,
                        ip_proto=ip_pkt.proto,
                        ipv4_src=ip_pkt.src,
                        ipv4_dst=ip_pkt.dst,
                    )
                datapath.send_msg(
                    parser.OFPFlowMod(
                        datapath=datapath,
                        priority=10,
                        idle_timeout=45,
                        hard_timeout=180,
                        match=match,
                        instructions=inst,
                    )
                )
            elif arp_pkt is not None:
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=ether_types.ETH_TYPE_ARP,
                    eth_src=eth.src,
                    eth_dst=eth.dst,
                )
                datapath.send_msg(
                    parser.OFPFlowMod(
                        datapath=datapath,
                        priority=5,
                        idle_timeout=60,
                        match=match,
                        instructions=inst,
                    )
                )

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        datapath.send_msg(
            parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=data,
            )
        )

    def _monitor_loop(self) -> None:
        """Periodically request OpenFlow 1.3 flow/port stats and sync policy blocklists."""
        while True:
            for dpid, dp in list(self.datapaths.items()):
                parser = dp.ofproto_parser
                ofp = dp.ofproto
                dp.send_msg(parser.OFPFlowStatsRequest(dp))
                dp.send_msg(parser.OFPPortStatsRequest(dp, 0, ofp.OFPP_ANY))
                self._sync_blocklist(dpid, dp)
            hub.sleep(self.poll_interval_sec)

    def _sync_blocklist(self, dpid: int, datapath: Any) -> None:
        """Read external JSON blocklist commands for automated mitigation if configured."""
        if not self.blocklist_path:
            return
        path = Path(self.blocklist_path)
        if not path.exists():
            return
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        for entry in items:
            src_ip = str(entry.get("ipv4_src", ""))
            key = f"{dpid}:{src_ip}"
            if src_ip and key not in self.installed_blocks:
                self.install_mitigation_drop_rule(
                    datapath,
                    ipv4_src=src_ip,
                    priority=int(entry.get("priority", 300)),
                    idle_timeout=int(entry.get("idle_timeout", 120)),
                )

    def _append_stat_record(self, record: dict[str, Any]) -> None:
        """Write one JSONL telemetry line safely."""
        self.stats_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stats_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)  # type: ignore[misc]
    def flow_stats_reply_handler(self, ev: Any) -> None:
        """Record per-flow packet/byte counts and durations reported by OVS."""
        dpid = ev.msg.datapath.id
        ts = time.time()
        flows = []
        for stat in ev.msg.body:
            if stat.priority < 5:
                continue
            flows.append(
                {
                    "priority": stat.priority,
                    "packet_count": stat.packet_count,
                    "byte_count": stat.byte_count,
                    "duration_sec": stat.duration_sec,
                    "duration_nsec": stat.duration_nsec,
                    "match": {k: str(v) for k, v in stat.match.items()},
                }
            )
        self._append_stat_record(
            {"type": "flow_stats", "timestamp": ts, "dpid": dpid, "flows": flows}
        )

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)  # type: ignore[misc]
    def port_stats_reply_handler(self, ev: Any) -> None:
        """Record per-port TX/RX counters reported by OVS."""
        dpid = ev.msg.datapath.id
        ts = time.time()
        ports = [
            {
                "port_no": stat.port_no,
                "rx_packets": stat.rx_packets,
                "tx_packets": stat.tx_packets,
                "rx_bytes": stat.rx_bytes,
                "tx_bytes": stat.tx_bytes,
                "rx_dropped": stat.rx_dropped,
                "tx_dropped": stat.tx_dropped,
            }
            for stat in ev.msg.body
        ]
        self._append_stat_record(
            {"type": "port_stats", "timestamp": ts, "dpid": dpid, "ports": ports}
        )

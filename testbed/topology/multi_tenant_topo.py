"""Multi-tenant Mininet + Open vSwitch topology for TrustFed-SDN Step 07.

Isolation Level L1 (approved 2026-11-14):
- One Ubuntu 22.04 LTS VMware guest VM.
- Three isolated tenant sub-networks (`10.10.1.0/24`, `10.10.2.0/24`, `10.10.3.0/24`)
  each with a dedicated OVS switch (`s1`, `s2`, `s3`), a dedicated OS-Ken OpenFlow 1.3
  controller instance (`6653`, `6654`, `6655`), and 3 namespace hosts (`benign`,
  `victim`, `attacker`).
- One core policy switch (`s0`, port `6656` or standalone OVS policy rules)
  interconnecting `s1..s3` while dropping 100% of cross-tenant IP/ARP traffic by default.
"""

from __future__ import annotations

import argparse
import ipaddress
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TenantHostSpec:
    """Specification for one Mininet namespace host inside a tenant subnet."""

    name: str
    role: str  # "benign" | "victim" | "attacker"
    ip: str
    prefix_len: int
    mac: str

    @property
    def cidr_ip(self) -> str:
        return f"{self.ip}/{self.prefix_len}"


@dataclass(frozen=True)
class TenantTopoSpec:
    """Specification for one isolated tenant OVS switch + controller + 3 hosts."""

    tenant_id: str
    display_name: str
    subnet: str
    switch_name: str
    switch_dpid: str
    controller_name: str
    controller_port: int
    specialty_attacks: tuple[str, ...]
    hosts: tuple[TenantHostSpec, ...]


CORE_SWITCH_SPEC: dict[str, Any] = {
    "switch_name": "s0",
    "dpid": "0000000000000010",
    "controller_port": 6656,
}

TOPOLOGY_SPEC: tuple[TenantTopoSpec, ...] = (
    TenantTopoSpec(
        tenant_id="tenant_1_finance",
        display_name="Tenant 1 — Finance",
        subnet="10.10.1.0/24",
        switch_name="s1",
        switch_dpid="0000000000000001",
        controller_name="c1",
        controller_port=6653,
        specialty_attacks=("brute_force", "web_attack", "dos_http"),
        hosts=(
            TenantHostSpec("h1_ben", "benign", "10.10.1.10", 24, "00:00:00:01:00:10"),
            TenantHostSpec("h1_vic", "victim", "10.10.1.50", 24, "00:00:00:01:00:50"),
            TenantHostSpec("h1_atk", "attacker", "10.10.1.100", 24, "00:00:00:01:01:00"),
        ),
    ),
    TenantTopoSpec(
        tenant_id="tenant_2_healthcare",
        display_name="Tenant 2 — Healthcare",
        subnet="10.10.2.0/24",
        switch_name="s2",
        switch_dpid="0000000000000002",
        controller_name="c2",
        controller_port=6654,
        specialty_attacks=("syn_flood", "udp_flood", "icmp_flood"),
        hosts=(
            TenantHostSpec("h2_ben", "benign", "10.10.2.10", 24, "00:00:00:02:00:10"),
            TenantHostSpec("h2_vic", "victim", "10.10.2.50", 24, "00:00:00:02:00:50"),
            TenantHostSpec("h2_atk", "attacker", "10.10.2.100", 24, "00:00:00:02:01:00"),
        ),
    ),
    TenantTopoSpec(
        tenant_id="tenant_3_education",
        display_name="Tenant 3 — Education",
        subnet="10.10.3.0/24",
        switch_name="s3",
        switch_dpid="0000000000000003",
        controller_name="c3",
        controller_port=6655,
        specialty_attacks=("port_scan", "dos_http", "udp_flood"),
        hosts=(
            TenantHostSpec("h3_ben", "benign", "10.10.3.10", 24, "00:00:00:03:00:10"),
            TenantHostSpec("h3_vic", "victim", "10.10.3.50", 24, "00:00:00:03:00:50"),
            TenantHostSpec("h3_atk", "attacker", "10.10.3.100", 24, "00:00:00:03:01:00"),
        ),
    ),
)


def get_tenant_spec(tenant_id: str) -> TenantTopoSpec:
    """Return the ``TenantTopoSpec`` matching ``tenant_id``."""
    for spec in TOPOLOGY_SPEC:
        if spec.tenant_id == tenant_id:
            return spec
    raise KeyError(f"Unknown tenant_id: {tenant_id!r}")


def get_all_host_ips() -> dict[str, str]:
    """Return mapping of host name -> IPv4 string across all tenants."""
    mapping: dict[str, str] = {}
    for tenant in TOPOLOGY_SPEC:
        for host in tenant.hosts:
            mapping[host.name] = host.ip
    return mapping


def render_ovs_core_isolation_commands() -> list[list[str]]:
    """Build deterministic ``ovs-ofctl`` OpenFlow 1.3 drop rules on core switch ``s0``.

    Any IPv4 or ARP packet traversing ``s0`` between different tenant ``/24``
    subnets is dropped unconditionally at priority 200.
    """
    sw = CORE_SWITCH_SPEC["switch_name"]
    cmds: list[list[str]] = [
        ["ovs-ofctl", "-O", "OpenFlow13", "add-flow", sw, "priority=0,actions=drop"]
    ]
    for src_t in TOPOLOGY_SPEC:
        for dst_t in TOPOLOGY_SPEC:
            if src_t.tenant_id == dst_t.tenant_id:
                continue
            cmds.append(
                [
                    "ovs-ofctl",
                    "-O",
                    "OpenFlow13",
                    "add-flow",
                    sw,
                    f"priority=200,ip,nw_src={src_t.subnet},nw_dst={dst_t.subnet},actions=drop",
                ]
            )
            cmds.append(
                [
                    "ovs-ofctl",
                    "-O",
                    "OpenFlow13",
                    "add-flow",
                    sw,
                    f"priority=200,arp,arp_spa={src_t.subnet},arp_tpa={dst_t.subnet},actions=drop",
                ]
            )
    return cmds


def _validate_topology_invariants() -> None:
    """Assert disjoint /24 subnets and distinct controller ports before launch."""
    nets = [ipaddress.ip_network(t.subnet) for t in TOPOLOGY_SPEC]
    ports = [t.controller_port for t in TOPOLOGY_SPEC]
    if len(set(ports)) != len(ports):
        raise ValueError("Tenant controller ports must be distinct.")
    for i, net_a in enumerate(nets):
        for net_b in nets[i + 1 :]:
            if net_a.overlaps(net_b):
                raise ValueError(f"Tenant subnets overlap: {net_a} vs {net_b}")


def start_mininet_topology(controller_ip: str = "127.0.0.1") -> Any:
    """Construct and start the multi-tenant Mininet + OVS network (Linux root only)."""
    _validate_topology_invariants()
    from mininet.net import Mininet
    from mininet.node import OVSSwitch, RemoteController
    from mininet.topo import Topo

    class MultiTenantSDNTopo(Topo):
        def build(self) -> None:
            core_sw = self.addSwitch(
                CORE_SWITCH_SPEC["switch_name"],
                dpid=CORE_SWITCH_SPEC["dpid"],
                protocols="OpenFlow13",
            )
            for tenant in TOPOLOGY_SPEC:
                sw = self.addSwitch(
                    tenant.switch_name,
                    dpid=tenant.switch_dpid,
                    protocols="OpenFlow13",
                )
                self.addLink(sw, core_sw)
                for h_spec in tenant.hosts:
                    host = self.addHost(
                        h_spec.name,
                        ip=h_spec.cidr_ip,
                        mac=h_spec.mac,
                    )
                    self.addLink(host, sw)

    net = Mininet(
        topo=MultiTenantSDNTopo(),
        switch=OVSSwitch,
        controller=None,
        build=False,
        autoSetMacs=False,
        autoStaticArp=False,
    )

    controllers: dict[str, Any] = {}
    for tenant in TOPOLOGY_SPEC:
        ctrl = net.addController(
            tenant.controller_name,
            controller=RemoteController,
            ip=controller_ip,
            port=tenant.controller_port,
        )
        controllers[tenant.switch_name] = ctrl

    net.build()
    for ctrl in controllers.values():
        ctrl.start()

    core_sw = net.get(CORE_SWITCH_SPEC["switch_name"])
    core_sw.start([])
    for tenant in TOPOLOGY_SPEC:
        net.get(tenant.switch_name).start([controllers[tenant.switch_name]])

    # Commit batched OVS bridge/port/controller configuration created by switch.start().
    if hasattr(OVSSwitch, "batchStartup"):
        OVSSwitch.batchStartup(net.switches)

    for cmd in render_ovs_core_isolation_commands():
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Wait up to 8s for all tenant switches to connect and receive table-miss OpenFlow 1.3 rules.
    tenant_switches = [tenant.switch_name for tenant in TOPOLOGY_SPEC]
    for _ in range(16):
        ready = True
        for sw_name in tenant_switches:
            res = subprocess.run(
                ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", sw_name],
                check=False,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0 or "CONTROLLER" not in res.stdout:
                ready = False
                break
        if ready:
            break
        time.sleep(0.5)
    return net


def run_isolation_verification(net: Any) -> dict[str, Any]:
    """Measure intra-tenant reachability and cross-tenant isolation via live ICMP probes."""
    intra_sent = 0
    intra_recv = 0
    for tenant in TOPOLOGY_SPEC:
        hosts = [net.get(h.name) for h in tenant.hosts]
        for i, src in enumerate(hosts):
            for j, dst in enumerate(hosts):
                if i == j:
                    continue
                intra_sent += 1
                out = src.cmd(f"ping -c 1 -W 1 {dst.IP()}")
                if " 0% packet loss" in out or "1 received" in out:
                    intra_recv += 1

    cross_sent = 0
    cross_blocked = 0
    cross_leaked = 0
    for src_t in TOPOLOGY_SPEC:
        src_host = net.get(src_t.hosts[0].name)
        for dst_t in TOPOLOGY_SPEC:
            if src_t.tenant_id == dst_t.tenant_id:
                continue
            dst_host = net.get(dst_t.hosts[1].name)
            # Also add explicit on-link route inside namespace so ARP/IP goes out to s1/s0
            src_host.cmd(f"ip route replace {dst_t.subnet} dev {src_t.hosts[0].name}-eth0")
            cross_sent += 1
            out = src_host.cmd(f"ping -c 1 -W 1 {dst_host.IP()}")
            src_host.cmd(f"ip route del {dst_t.subnet} dev {src_t.hosts[0].name}-eth0 2>/dev/null")
            if " 0% packet loss" in out or "1 received" in out:
                cross_leaked += 1
            else:
                cross_blocked += 1

    return {
        "intra_probes_sent": intra_sent,
        "intra_probes_received": intra_recv,
        "intra_loss_pct": round(100.0 * (intra_sent - intra_recv) / max(1, intra_sent), 2),
        "cross_probes_sent": cross_sent,
        "cross_probes_blocked": cross_blocked,
        "cross_probes_leaked": cross_leaked,
        "cross_block_pct": round(100.0 * cross_blocked / max(1, cross_sent), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustFed-SDN multi-tenant topology CLI")
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--verify", action="store_true", help="Verify intra/cross isolation & exit")
    parser.add_argument("--cli", action="store_true", help="Drop into interactive Mininet CLI")
    args = parser.parse_args()

    net = start_mininet_topology(controller_ip=args.controller_ip)
    try:
        if args.verify:
            stats = run_isolation_verification(net)
            print(stats, flush=True)
        elif args.cli:
            from mininet.cli import CLI

            CLI(net)
    finally:
        net.stop()


if __name__ == "__main__":
    main()

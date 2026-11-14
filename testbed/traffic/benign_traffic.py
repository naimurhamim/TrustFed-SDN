"""Benign multi-service traffic generator for TrustFed-SDN hosts.

Simulates realistic HTTP GET/POST, SSH-banner TCP exchanges, DNS-like UDP queries,
and short bulk TCP transfers strictly inside ``10.10.1-3.0/24``.
"""

from __future__ import annotations

import argparse
import http.server
import random
import socket
import socketserver
import threading
import time

from testbed.traffic.safety_guard import validate_testbed_target

DEFAULT_HTTP_PORT = 8080
DEFAULT_ECHO_PORT = 9000
DEFAULT_UDP_PORT = 5353
DEFAULT_SSH_PORT = 2222


class _QuietHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Lightweight HTTP responder returning synthetic telemetry payloads."""

    def do_GET(self) -> None:
        payload = (
            b'{"service":"trustfed-tenant-app","status":"ok","path":"'
            + self.path.encode("ascii", errors="ignore")
            + b'"}'
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 0:
            _ = self.rfile.read(min(length, 65536))
        reply = b'{"accepted":true}'
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, format: str, *args: object) -> None:
        return


class _ReusableTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _serve_tcp_echo(bind_ip: str, port: int, stop_event: threading.Event, banner: bytes) -> None:
    """Serve simple TCP banner/echo exchanges until ``stop_event`` is set."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_ip, port))
    srv.listen(128)
    srv.settimeout(0.5)

    def handle_conn(conn: socket.socket) -> None:
        with conn:
            conn.settimeout(1.0)
            try:
                if banner:
                    conn.sendall(banner)
                data = conn.recv(4096)
                if data:
                    conn.sendall(data[:1024])
            except OSError:
                pass

    try:
        while not stop_event.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=handle_conn, args=(conn,), daemon=True).start()
    finally:
        srv.close()


def _serve_udp_dns(bind_ip: str, port: int, stop_event: threading.Event) -> None:
    """Serve small UDP DNS-like query/response datagrams until ``stop_event`` is set."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_ip, port))
    sock.settimeout(0.5)
    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError:
                break
            resp = b"\x00\x01\x81\x80\x00\x01\x00\x01" + data[:64]
            try:
                sock.sendto(resp, addr)
            except OSError:
                pass
    finally:
        sock.close()


def run_benign_server(
    bind_ip: str,
    tenant_id: str | None = None,
    duration_sec: float = 0.0,
    http_port: int = DEFAULT_HTTP_PORT,
    echo_port: int = DEFAULT_ECHO_PORT,
    udp_port: int = DEFAULT_UDP_PORT,
    ssh_port: int = DEFAULT_SSH_PORT,
) -> None:
    """Start multi-service benign victim listeners on validated ``bind_ip``."""
    target = str(validate_testbed_target(bind_ip, tenant_id=tenant_id))
    stop_event = threading.Event()
    http_srv = _ReusableTCPServer((target, http_port), _QuietHTTPHandler)

    threads = [
        threading.Thread(target=http_srv.serve_forever, daemon=True),
        threading.Thread(
            target=_serve_tcp_echo, args=(target, echo_port, stop_event, b""), daemon=True
        ),
        threading.Thread(
            target=_serve_tcp_echo,
            args=(target, ssh_port, stop_event, b"SSH-2.0-OpenSSH_8.9p1 Ubuntu\r\n"),
            daemon=True,
        ),
        threading.Thread(target=_serve_udp_dns, args=(target, udp_port, stop_event), daemon=True),
    ]
    for th in threads:
        th.start()

    print(
        f"[BENIGN SERVER] listening on {target} ports http={http_port}, "
        f"echo={echo_port}, ssh={ssh_port}, udp={udp_port}",
        flush=True,
    )
    start = time.monotonic()
    try:
        while True:
            if duration_sec > 0.0 and (time.monotonic() - start) >= duration_sec:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        http_srv.shutdown()
        http_srv.server_close()


def _send_http_session(target_ip: str, port: int, rng: random.Random) -> bool:
    """Perform one short HTTP GET or POST request/response session."""
    try:
        with socket.create_connection((target_ip, port), timeout=1.5) as sock:
            if rng.random() < 0.7:
                req = (
                    f"GET /api/v1/status?q={rng.randint(1, 500)} HTTP/1.1\r\n"
                    f"Host: {target_ip}\r\nConnection: close\r\n\r\n"
                ).encode("ascii")
            else:
                body = ("sample=" + ("A" * rng.randint(64, 512))).encode("ascii")
                req = (
                    f"POST /api/v1/ingest HTTP/1.1\r\nHost: {target_ip}\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
                ).encode("ascii") + body
            sock.sendall(req)
            while sock.recv(2048):
                pass
        return True
    except OSError:
        return False


def _send_ssh_like_session(target_ip: str, port: int, rng: random.Random) -> bool:
    """Exchange realistic SSH banner/KEX-sized payloads over TCP."""
    try:
        with socket.create_connection((target_ip, port), timeout=1.5) as sock:
            _ = sock.recv(512)
            sock.sendall(b"SSH-2.0-OpenSSH_8.9p1\r\n" + bytes(rng.randint(32, 160)))
            _ = sock.recv(512)
        return True
    except OSError:
        return False


def _send_bulk_transfer(target_ip: str, port: int, rng: random.Random) -> bool:
    """Transfer a moderate TCP payload (1-6 KB) resembling benign app traffic."""
    try:
        with socket.create_connection((target_ip, port), timeout=1.5) as sock:
            chunk = bytes([rng.randint(65, 90)]) * rng.randint(512, 3072)
            sock.sendall(chunk)
            _ = sock.recv(2048)
        return True
    except OSError:
        return False


def _send_udp_query(target_ip: str, port: int, rng: random.Random) -> bool:
    """Send a UDP DNS-like query and read the reply."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            query = b"\xab\xcd\x01\x00\x00\x01\x00\x00" + bytes(rng.randint(16, 48))
            sock.sendto(query, (target_ip, port))
            _ = sock.recvfrom(1024)
        return True
    except OSError:
        return False


def run_benign_client(
    target_ip: str,
    tenant_id: str | None = None,
    flows: int = 80,
    seed: int = 42,
    http_port: int = DEFAULT_HTTP_PORT,
    echo_port: int = DEFAULT_ECHO_PORT,
    udp_port: int = DEFAULT_UDP_PORT,
    ssh_port: int = DEFAULT_SSH_PORT,
    min_gap_sec: float = 0.01,
    max_gap_sec: float = 0.06,
) -> int:
    """Generate ``flows`` randomized benign sessions toward validated ``target_ip``."""
    target = str(validate_testbed_target(target_ip, tenant_id=tenant_id))
    rng = random.Random(seed)
    succeeded = 0
    for idx in range(flows):
        roll = rng.random()
        if roll < 0.45:
            ok = _send_http_session(target, http_port, rng)
        elif roll < 0.70:
            ok = _send_udp_query(target, udp_port, rng)
        elif roll < 0.85:
            ok = _send_ssh_like_session(target, ssh_port, rng)
        else:
            ok = _send_bulk_transfer(target, echo_port, rng)
        if ok:
            succeeded += 1
        time.sleep(rng.uniform(min_gap_sec, max_gap_sec))
        if (idx + 1) % 50 == 0:
            print(f"[BENIGN CLIENT] target={target} completed={idx + 1}/{flows}", flush=True)
    print(f"[BENIGN CLIENT DONE] target={target} succeeded={succeeded}/{flows}", flush=True)
    return succeeded


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrustFed-SDN benign traffic generator")
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--target", required=True, help="Target/bind IP inside 10.10.1-3.0/24")
    parser.add_argument("--tenant", default=None, help="Optional tenant id check")
    parser.add_argument("--flows", type=int, default=80)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "server":
        run_benign_server(args.target, tenant_id=args.tenant, duration_sec=args.duration)
    else:
        run_benign_client(args.target, tenant_id=args.tenant, flows=args.flows, seed=args.seed)


if __name__ == "__main__":
    main()

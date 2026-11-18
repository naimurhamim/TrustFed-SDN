"""Local web server for the TrustFed-SDN Interactive Visual Studio (Option A & B)."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.engine import TrustFedStudioEngine  # noqa: E402

STATIC_DIR = REPO_ROOT / "demo" / "static"
_MAX_BODY_BYTES = 25 * 1024 * 1024
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)


class StudioRequestHandler(BaseHTTPRequestHandler):
    """Serve the interactive frontend and real Python TrustFed-SDN API endpoints."""

    engine: TrustFedStudioEngine

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # Keep terminal output clean during live demos

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except _CLIENT_DISCONNECT_ERRORS:
            pass

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise ValueError("Payload exceeds 25 MB safety limit.")
        raw = self.rfile.read(length)
        return dict(json.loads(raw.decode("utf-8")))

    def _serve_static(self, rel_path: str) -> None:
        clean = "index.html" if rel_path in ("", "/") else rel_path.lstrip("/")
        target = (STATIC_DIR / clean).resolve()
        if not target.is_file() or STATIC_DIR.resolve() not in target.parents:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        ctype, _ = mimetypes.guess_type(str(target))
        data = target.read_bytes()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except _CLIENT_DISCONNECT_ERRORS:
            pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/init":
            self._send_json(HTTPStatus.OK, self.engine.get_initial_state())
            return
        if parsed.path == "/api/vm/status":
            self._send_json(HTTPStatus.OK, self.engine.get_vm_status())
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._read_json_body()
            if parsed.path == "/api/step":
                step = int(body.get("step", 0))
                self._send_json(HTTPStatus.OK, self.engine.run_pipeline_step(step))
                return
            if parsed.path == "/api/reset":
                self._send_json(HTTPStatus.OK, self.engine.reset_pipeline())
                return
            if parsed.path == "/api/inspect-sample":
                name = str(body.get("pcap_name", "")).strip()
                threshold = body.get("threshold")
                raw_dir = (REPO_ROOT / "data" / "live" / "raw_pcaps").resolve()
                pcap_path = (raw_dir / name).resolve()
                if not pcap_path.is_file() or raw_dir not in pcap_path.parents:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"ok": False, "error": f"Invalid PCAP sample: {name}"},
                    )
                    return
                res = self.engine.inspect_pcap_path(
                    pcap_path, threshold=float(threshold) if threshold is not None else None
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "result": res})
                return
            if parsed.path == "/api/inspect-upload":
                fname = str(body.get("filename", "upload.pcap"))
                b64 = str(body.get("content_b64", ""))
                threshold = body.get("threshold")
                raw = base64.b64decode(b64)
                res = self.engine.inspect_uploaded_bytes(
                    fname, raw, threshold=float(threshold) if threshold is not None else None
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "result": res})
                return
            if parsed.path == "/api/lab/poison":
                res = self.engine.simulate_poisoning(
                    malicious_pct=int(body.get("malicious_pct", 33)),
                    attack_type=str(body.get("attack_type", "scaling")),
                )
                self._send_json(HTTPStatus.OK, {"ok": True, "result": res})
                return
            if parsed.path == "/api/lab/tamper":
                res = self.engine.simulate_tamper(tamper_mode=str(body.get("tamper_mode", "none")))
                self._send_json(HTTPStatus.OK, {"ok": True, "result": res})
                return
            if parsed.path == "/api/vm/step":
                step = int(body.get("step", 0))
                safe_mode = bool(body.get("safe_mode", False))
                auto_fallback = bool(body.get("auto_fallback", True))
                res = self.engine.run_vm_step(
                    step,
                    safe_mode=safe_mode,
                    auto_fallback=auto_fallback,
                    vm_user=body.get("vm_user"),
                    vm_pass=body.get("vm_pass"),
                    vmx_path=body.get("vmx_path"),
                )
                self._send_json(HTTPStatus.OK, res)
                return
            if parsed.path == "/api/vm/stop":
                self._send_json(
                    HTTPStatus.OK,
                    self.engine.stop_vm(
                        vm_user=body.get("vm_user"),
                        vm_pass=body.get("vm_pass"),
                        vmx_path=body.get("vmx_path"),
                    ),
                )
                return
            if parsed.path == "/api/vm/reset":
                self._send_json(HTTPStatus.OK, self.engine.reset_vm())
                return
            if parsed.path == "/api/vm/setup":
                res = self.engine.auto_setup_vm(
                    vm_user=body.get("vm_user"),
                    vm_pass=body.get("vm_pass"),
                    vmx_path=body.get("vmx_path"),
                    safe_mode=bool(body.get("safe_mode", False)),
                )
                self._send_json(HTTPStatus.OK, res)
                return
            if parsed.path == "/api/vm/inspect-file":
                rel_path = str(body.get("path", "")).strip()
                res = self.engine.inspect_vm_file(rel_path)
                self._send_json(HTTPStatus.OK, {"ok": True, "result": res})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown API route"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})


def make_studio_server(
    engine: TrustFedStudioEngine | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> ThreadingHTTPServer:
    """Create a configured ``ThreadingHTTPServer`` serving the TrustFed Studio UI and API."""
    eng = engine or TrustFedStudioEngine()
    eng.ensure_model_bundle()

    class _Handler(StudioRequestHandler):
        engine = eng

    return ThreadingHTTPServer((host, int(port)), _Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description="TrustFed-SDN Interactive Studio Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("[TrustFed Studio] Loading live GRU model & captured SDN flows...", flush=True)
    server: ThreadingHTTPServer | None = None
    for offset in range(11):
        try_port = args.port + offset
        try:
            server = make_studio_server(host=args.host, port=try_port)
            if offset > 0:
                print(
                    f"[TrustFed Studio] Port {args.port} busy; bound safely to {try_port}.",
                    flush=True,
                )
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError(f"Ports {args.port}..{args.port + 10} are all busy.")
    actual_port = int(server.server_address[1])
    url = f"http://{args.host}:{actual_port}/"
    print(f"[TrustFed Studio] Ready at {url} (Ctrl+C to stop)", flush=True)

    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[TrustFed Studio] Stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Lightweight HTTP/JSON + binary wire RPC transport for live SDN FL rounds (`Step 08`).

Provides real localhost socket communication (`http.server.ThreadingHTTPServer` +
`urllib.request`) between live tenant FL clients and `LiveAggregationServer` while
measuring true wire bytes and network transfer latency.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch

from testbed.fl.aggregation_server import LiveAggregationServer
from trustfed.fl.aggregator import (
    ClientUpdate,
    deserialize_tensor_payload,
    serialize_tensor_payload,
)
from trustfed.verify import (
    MerkleInclusionProof,
    SignedTenantSubmission,
    TenantCommitmentHeader,
    VerificationProof,
    pack_masked_48bit,
    unpack_masked_48bit,
)


def _b64_encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64_decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def submission_to_wire_dict(
    sub: SignedTenantSubmission,
    update: ClientUpdate,
) -> dict[str, Any]:
    """Serialize a `SignedTenantSubmission` + `ClientUpdate` to a JSON-safe wire dict."""
    model_wire = serialize_tensor_payload(
        update.flat_params,
        num_samples=update.num_samples,
        round_idx=update.round_idx,
    )
    return {
        "client_id": sub.header.client_id,
        "round_idx": int(sub.header.round_idx),
        "numel": int(sub.masked_update.numel()),
        "header_b64": _b64_encode(sub.header_wire_bytes),
        "masked_b64": _b64_encode(sub.masked_wire_bytes),
        "model_b64": _b64_encode(model_wire),
        "num_samples": int(update.num_samples),
        "train_loss": float(update.train_loss),
        "local_train_time_sec": float(update.local_train_time_sec),
        "timings_ms": {
            "quantize_ms": float(sub.quantize_ms),
            "mask_ms": float(sub.mask_ms),
            "checksum_ms": float(sub.checksum_ms),
            "sign_ms": float(sub.sign_ms),
        },
        "metadata": dict(update.metadata),
    }


def wire_dict_to_submission(
    data: dict[str, Any],
    global_flat_ref: torch.Tensor,
) -> tuple[SignedTenantSubmission, ClientUpdate]:
    """Reconstruct `(SignedTenantSubmission, ClientUpdate)` from wire JSON."""
    header_wire = _b64_decode(data["header_b64"])
    masked_wire = _b64_decode(data["masked_b64"])
    model_wire = _b64_decode(data["model_b64"])

    header = TenantCommitmentHeader.from_bytes(header_wire)
    masked_tensor = unpack_masked_48bit(masked_wire, int(data["numel"]))
    flat_params, num_samples, round_idx = deserialize_tensor_payload(model_wire)
    flat_delta = flat_params - global_flat_ref.detach().to(device="cpu", dtype=torch.float32)

    t = data.get("timings_ms", {})
    sub = SignedTenantSubmission(
        header=header,
        masked_update=masked_tensor,
        masked_wire_bytes=masked_wire,
        header_wire_bytes=header_wire,
        quantize_ms=float(t.get("quantize_ms", 0.0)),
        mask_ms=float(t.get("mask_ms", 0.0)),
        checksum_ms=float(t.get("checksum_ms", 0.0)),
        sign_ms=float(t.get("sign_ms", 0.0)),
    )
    upd = ClientUpdate(
        client_id=str(data["client_id"]),
        flat_params=flat_params,
        flat_delta=flat_delta,
        num_samples=int(num_samples),
        train_loss=float(data.get("train_loss", 0.0)),
        local_train_time_sec=float(data.get("local_train_time_sec", 0.0)),
        upload_bytes=len(masked_wire) + len(header_wire),
        round_idx=int(round_idx),
        metadata=dict(data.get("metadata", {})),
    )
    return sub, upd


def proof_to_wire_dict(proof: VerificationProof) -> dict[str, Any]:
    """Serialize `VerificationProof` cleanly for HTTP download."""
    sum_bytes = proof.unmasked_sum_int.detach().cpu().to(dtype=torch.int64).numpy().tobytes()
    agg_wire = serialize_tensor_payload(
        proof.aggregate_float,
        num_samples=proof.total_samples,
        round_idx=proof.round_idx,
    )
    merkle_dict = {
        cid: {
            "leaf_index": int(mp.leaf_index),
            "leaf_hash_b64": _b64_encode(mp.leaf_hash),
            "siblings": [[_b64_encode(sib), side] for sib, side in mp.siblings],
            "root_hash_b64": _b64_encode(mp.root_hash),
        }
        for cid, mp in proof.merkle_proofs.items()
    }
    headers_dict = {cid: _b64_encode(hdr.to_bytes()) for cid, hdr in proof.signed_headers.items()}
    return {
        "round_idx": int(proof.round_idx),
        "merkle_root_b64": _b64_encode(proof.merkle_root),
        "included_client_ids": list(proof.included_client_ids),
        "total_samples": int(proof.total_samples),
        "unmasked_sum_int_b64": _b64_encode(sum_bytes),
        "aggregate_b64": _b64_encode(agg_wire),
        "signed_headers": headers_dict,
        "merkle_proofs": merkle_dict,
        "server_time_ms": float(proof.server_time_ms),
        "proof_wire_bytes": int(proof.proof_wire_bytes),
    }


def wire_dict_to_proof(data: dict[str, Any]) -> VerificationProof:
    """Reconstruct `VerificationProof` from wire JSON."""
    sum_raw = _b64_decode(data["unmasked_sum_int_b64"])
    sum_arr = np.frombuffer(sum_raw, dtype=np.int64).copy()
    unmasked_sum = torch.from_numpy(sum_arr)

    agg_float, _, _ = deserialize_tensor_payload(_b64_decode(data["aggregate_b64"]))
    signed_headers = {
        cid: TenantCommitmentHeader.from_bytes(_b64_decode(b64_str))
        for cid, b64_str in data["signed_headers"].items()
    }
    merkle_proofs = {
        cid: MerkleInclusionProof(
            leaf_index=int(mp["leaf_index"]),
            leaf_hash=_b64_decode(mp["leaf_hash_b64"]),
            siblings=tuple((_b64_decode(s_b64), str(side)) for s_b64, side in mp["siblings"]),
            root_hash=_b64_decode(mp["root_hash_b64"]),
        )
        for cid, mp in data["merkle_proofs"].items()
    }
    return VerificationProof(
        round_idx=int(data["round_idx"]),
        merkle_root=_b64_decode(data["merkle_root_b64"]),
        included_client_ids=tuple(data["included_client_ids"]),
        total_samples=int(data["total_samples"]),
        unmasked_sum_int=unmasked_sum,
        aggregate_float=agg_float,
        signed_headers=signed_headers,
        merkle_proofs=merkle_proofs,
        server_time_ms=float(data["server_time_ms"]),
        proof_wire_bytes=int(data["proof_wire_bytes"]),
    )


def repack_submission_header_b64(sub: SignedTenantSubmission) -> dict[str, str]:
    """Serialize only the updated header and masked vector after cohort re-masking."""
    packed = pack_masked_48bit(sub.masked_update)
    return {
        "client_id": sub.header.client_id,
        "round_idx": str(sub.header.round_idx),
        "numel": str(sub.masked_update.numel()),
        "header_b64": _b64_encode(sub.header_wire_bytes),
        "masked_b64": _b64_encode(packed),
    }


class LiveAggregationHTTPServer:
    """Multi-threaded HTTP server wrapping `LiveAggregationServer` on localhost."""

    def __init__(
        self,
        agg_server: LiveAggregationServer,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.agg_server = agg_server
        self._lock = threading.Lock()
        self._round_submissions: dict[
            int, dict[str, tuple[SignedTenantSubmission, ClientUpdate]]
        ] = {}
        self._last_resign_cohort: dict[int, dict[str, SignedTenantSubmission]] = {}

        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                _ = (format, args)

            def _send_json(self, status: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                return dict(json.loads(raw.decode("utf-8")))

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send_json(200, {"ok": True, "round_idx": parent.agg_server.current_round})
                    return
                if self.path == "/model":
                    with parent._lock:
                        g_flat = parent.agg_server.get_global_flat_params()
                        wire = serialize_tensor_payload(
                            g_flat, num_samples=0, round_idx=parent.agg_server.current_round
                        )
                        self._send_json(
                            200,
                            {
                                "round_idx": int(parent.agg_server.current_round),
                                "model_b64": _b64_encode(wire),
                                "model_bytes": len(wire),
                            },
                        )
                    return
                self._send_json(404, {"error": f"Unknown GET path: {self.path}"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    data = self._read_json()
                except Exception as exc:
                    self._send_json(400, {"error": f"Invalid JSON payload: {exc}"})
                    return

                if self.path == "/submit":
                    with parent._lock:
                        g_flat = parent.agg_server.get_global_flat_params()
                        sub, upd = wire_dict_to_submission(data, g_flat)
                        r_idx = int(upd.round_idx)
                        parent._round_submissions.setdefault(r_idx, {})[upd.client_id] = (sub, upd)
                        self._send_json(
                            200,
                            {
                                "ok": True,
                                "client_id": upd.client_id,
                                "round_idx": r_idx,
                                "upload_bytes": int(sub.total_upload_bytes),
                            },
                        )
                    return

                if self.path == "/resign":
                    with parent._lock:
                        r_idx = int(data["round_idx"])
                        header = TenantCommitmentHeader.from_bytes(_b64_decode(data["header_b64"]))
                        masked_wire = _b64_decode(data["masked_b64"])
                        masked_tensor = unpack_masked_48bit(masked_wire, int(data["numel"]))
                        sub = SignedTenantSubmission(
                            header=header,
                            masked_update=masked_tensor,
                            masked_wire_bytes=masked_wire,
                            header_wire_bytes=_b64_decode(data["header_b64"]),
                            quantize_ms=0.0,
                            mask_ms=0.0,
                            checksum_ms=0.0,
                            sign_ms=0.0,
                        )
                        parent._last_resign_cohort.setdefault(r_idx, {})[header.client_id] = sub
                        self._send_json(200, {"ok": True, "client_id": header.client_id})
                    return

                if self.path == "/abort":
                    with parent._lock:
                        r_idx = int(data["round_idx"])
                        parent._round_submissions.pop(r_idx, None)
                        parent._last_resign_cohort.pop(r_idx, None)
                        self._send_json(
                            200,
                            {
                                "ok": True,
                                "aborted": True,
                                "round_idx": r_idx,
                                "preserved_round": int(parent.agg_server.current_round),
                            },
                        )
                    return

                if self.path == "/aggregate":
                    with parent._lock:
                        r_idx = int(data["round_idx"])
                        use_secagg = bool(data.get("use_verifiable_secagg", True))
                        tamper_mode = data.get("tamper_mode")
                        expected_clients = data.get("expected_clients")
                        bucket = parent._round_submissions.get(r_idx, {})
                        if not bucket:
                            self._send_json(400, {"error": f"No submissions for round {r_idx}."})
                            return

                        ordered_ids = sorted(bucket.keys())
                        if expected_clients is not None:
                            missing = sorted(set(expected_clients) - set(ordered_ids))
                            if missing:
                                parent._round_submissions.pop(r_idx, None)
                                parent._last_resign_cohort.pop(r_idx, None)
                                self._send_json(
                                    409,
                                    {
                                        "ok": False,
                                        "aborted": True,
                                        "error": "incomplete_round_missing_clients",
                                        "round_idx": r_idx,
                                        "missing_clients": missing,
                                    },
                                )
                                return
                        subs = [bucket[cid][0] for cid in ordered_ids]
                        updates = [bucket[cid][1] for cid in ordered_ids]
                        g_flat = parent.agg_server.get_global_flat_params()

                        resigned_map = parent._last_resign_cohort.get(r_idx, {})

                        def _cohort_cb(
                            accepted: Any, round_no: int
                        ) -> list[SignedTenantSubmission]:
                            _ = round_no
                            return [
                                resigned_map[cid] for cid in sorted(accepted) if cid in resigned_map
                            ]

                        cb = _cohort_cb if resigned_map else None
                        outcome = parent.agg_server.aggregate_live_round(
                            round_idx=r_idx,
                            global_flat_params=g_flat,
                            client_updates=updates,
                            submissions=subs,
                            resign_cohort_fn=cb,
                            use_verifiable_secagg=use_secagg,
                            tamper_mode=tamper_mode,
                        )
                        resp: dict[str, Any] = {
                            "ok": True,
                            "round_idx": int(r_idx),
                            "accepted_clients": outcome.accepted_clients,
                            "rejected_clients": outcome.rejected_clients,
                            "robust_screen_ms": outcome.robust_screen_ms,
                            "server_agg_proof_ms": outcome.server_agg_proof_ms,
                            "tamper_injected": outcome.tamper_injected,
                            "model_b64": _b64_encode(
                                serialize_tensor_payload(
                                    outcome.new_global_flat_params, num_samples=0, round_idx=r_idx
                                )
                            ),
                        }
                        if outcome.proof is not None:
                            resp["proof"] = proof_to_wire_dict(outcome.proof)
                        self._send_json(200, resp)
                    return

                if self.path == "/screen":
                    with parent._lock:
                        r_idx = int(data["round_idx"])
                        bucket = parent._round_submissions.get(r_idx, {})
                        ordered_ids = sorted(bucket.keys())
                        updates = [bucket[cid][1] for cid in ordered_ids]
                        g_flat = parent.agg_server.get_global_flat_params()
                        t0 = time.perf_counter()
                        res = (
                            parent.agg_server._robust.aggregate(g_flat, updates)
                            if parent.agg_server.robust_defense and len(updates) >= 3
                            else parent.agg_server._fedavg.aggregate(g_flat, updates)
                        )
                        screen_ms = (time.perf_counter() - t0) * 1000.0
                        self._send_json(
                            200,
                            {
                                "accepted_clients": res.accepted_clients,
                                "rejected_clients": res.rejected_clients,
                                "robust_screen_ms": float(screen_ms),
                            },
                        )
                    return

                self._send_json(404, {"error": f"Unknown POST path: {self.path}"})

        self._httpd = ThreadingHTTPServer((host, int(port)), _Handler)
        self.host, self.port = self._httpd.server_address
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start serving requests in a background daemon thread."""
        self._thread.start()

    def stop(self) -> None:
        """Cleanly shut down the HTTP server socket."""
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> LiveAggregationHTTPServer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class LiveTenantHTTPClient:
    """HTTP RPC client used by tenant orchestrators to talk to `LiveAggregationHTTPServer`."""

    def __init__(self, base_url: str, *, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def _get_json(self, path: str) -> tuple[dict[str, Any], float, int]:
        t0 = time.perf_counter()
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self.timeout) as resp:
            raw = resp.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return dict(json.loads(raw.decode("utf-8"))), float(elapsed_ms), len(raw)

    def _post_json(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float, int, int]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return dict(json.loads(raw.decode("utf-8"))), float(elapsed_ms), len(body), len(raw)

    def fetch_global_model(self) -> tuple[torch.Tensor, int, float, int]:
        """Download global flat model parameters from server."""
        data, dl_ms, raw_bytes = self._get_json("/model")
        tensor, _, _ = deserialize_tensor_payload(_b64_decode(data["model_b64"]))
        return tensor, int(data["round_idx"]), dl_ms, raw_bytes

    def submit_update(
        self,
        sub: SignedTenantSubmission,
        update: ClientUpdate,
    ) -> tuple[float, int]:
        """Upload one tenant's signed & masked update to server."""
        wire_payload = submission_to_wire_dict(sub, update)
        _, up_ms, up_bytes, _ = self._post_json("/submit", wire_payload)
        return up_ms, up_bytes

    def screen_round(self, round_idx: int) -> dict[str, Any]:
        """Run server-side robust screening to identify accepted/rejected cohort."""
        resp, _, _, _ = self._post_json("/screen", {"round_idx": int(round_idx)})
        return resp

    def submit_resigned_cohort(self, sub: SignedTenantSubmission) -> None:
        """Upload re-masked cohort submission after excluding malicious tenants."""
        self._post_json("/resign", repack_submission_header_b64(sub))

    def abort_round(self, round_idx: int) -> dict[str, Any]:
        """Abort and discard an incomplete/failed round without changing global weights."""
        resp, _, _, _ = self._post_json("/abort", {"round_idx": int(round_idx)})
        return resp

    def trigger_aggregate(
        self,
        round_idx: int,
        *,
        use_verifiable_secagg: bool = True,
        tamper_mode: str | None = None,
        expected_clients: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Trigger server aggregation and return proof, metrics, and download size."""
        payload: dict[str, Any] = {
            "round_idx": int(round_idx),
            "use_verifiable_secagg": bool(use_verifiable_secagg),
            "tamper_mode": tamper_mode,
        }
        if expected_clients is not None:
            payload["expected_clients"] = list(expected_clients)
        resp, rpc_ms, _, dl_bytes = self._post_json("/aggregate", payload)
        resp["download_ms"] = float(rpc_ms)
        resp["download_bytes"] = int(dl_bytes)
        if "proof" in resp:
            resp["proof_obj"] = wire_dict_to_proof(resp["proof"])
        resp["global_tensor"], _, _ = deserialize_tensor_payload(_b64_decode(resp["model_b64"]))
        return resp

"""Verifiable Secure Aggregation wrapper implementing `Aggregator` (`Step 06`)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from trustfed.aggregation.clustering import CosineClusteringAggregator
from trustfed.fl.aggregator import AggregationResult, Aggregator, ClientUpdate
from trustfed.verify.masking import (
    MODULUS_M,
    TenantKeyMaterial,
    TenantPublicRegistry,
    generate_tenant_keys,
)
from trustfed.verify.protocol import (
    SignedTenantSubmission,
    VerificationProof,
    VerificationVerdict,
    create_signed_submission,
    server_secure_aggregate,
    verify_aggregate,
)
from trustfed.verify.quantize import DEFAULT_SCALE_BITS


class SecureAggregator(Aggregator):
    """Privacy-preserving & value-verifiable `FedAvg` aggregator (`Step 06`).

    Wraps fixed-point quantization, X25519+SHAKE256 pairwise additive masking,
    dual-prime keyed linear homomorphic checksums, SHA-256 Merkle tree commitments,
    and Ed25519 digital signatures into the unified `Aggregator` interface so it
    plugs directly into `FLServer` (`sim/`) and the live testbed (`testbed/`).
    When `robust_screening=True`, updates are first screened with
    `CosineClusteringAggregator` and cryptographic secure aggregation runs over
    the accepted honest cohort.
    """

    name: str = "secure_fedavg"

    def __init__(
        self,
        *,
        scale_bits: int = DEFAULT_SCALE_BITS,
        seed: int = 0,
        modulus: int = MODULUS_M,
        expected_clients: Sequence[str] | None = None,
        verify_on_aggregate: bool = True,
        raise_on_tamper: bool = True,
        robust_screening: bool = False,
        max_norm_factor: float = 2.5,
        min_cosine_to_median: float = 0.0,
    ) -> None:
        self.scale_bits = int(scale_bits)
        self.seed = int(seed)
        self.modulus = int(modulus)
        self.expected_clients = (
            tuple(sorted(str(c) for c in expected_clients))
            if expected_clients is not None
            else None
        )
        self.verify_on_aggregate = bool(verify_on_aggregate)
        self.raise_on_tamper = bool(raise_on_tamper)
        self.robust_screening = bool(robust_screening)
        self.max_norm_factor = float(max_norm_factor)
        self.min_cosine_to_median = float(min_cosine_to_median)
        if self.robust_screening:
            self.name = "secure_robust"

        self._robust_screener = (
            CosineClusteringAggregator(
                max_norm_factor=self.max_norm_factor,
                min_cosine_to_median=self.min_cosine_to_median,
            )
            if self.robust_screening
            else None
        )
        self._materials: dict[str, TenantKeyMaterial] | None = None
        self._registry: TenantPublicRegistry | None = None
        self._group_key: bytes | None = None
        self._server_seen_nonces: set[tuple[str, bytes]] = set()
        self._client_seen_nonces: set[tuple[str, bytes]] = set()
        self.last_proof: VerificationProof | None = None
        self.last_verdict: VerificationVerdict | None = None

    def _ensure_keys(self, client_ids: Sequence[str]) -> None:
        ordered = tuple(sorted(str(c) for c in client_ids))
        if self._registry is not None:
            registered = set(self._registry.client_ids)
            if set(ordered) == registered:
                return
            if self.robust_screening and set(ordered).issubset(registered):
                return
            missing = sorted(registered - set(ordered))
            raise RuntimeError(
                f"Round aborted: tenant set changed or dropout occurred (missing={missing})."
            )
        target_ids = self.expected_clients if self.expected_clients is not None else ordered
        materials, registry, group_key = generate_tenant_keys(target_ids, seed=self.seed)
        self._materials = materials
        self._registry = registry
        self._group_key = group_key

    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("SecureAggregator received an empty updates sequence.")

        round_idx = int(updates[0].round_idx)
        for upd in updates[1:]:
            if int(upd.round_idx) != round_idx:
                raise RuntimeError(
                    f"Inconsistent round_idx in updates for SecureAggregator: "
                    f"{upd.round_idx} != {round_idx}."
                )

        self._ensure_keys([u.client_id for u in updates])
        assert self._materials is not None
        assert self._registry is not None
        assert self._group_key is not None

        device = global_flat_params.device
        dtype = global_flat_params.dtype
        numel = global_flat_params.numel()

        t_total_start = time.perf_counter()

        # 0. Optional Byzantine update screening before cryptographic secure aggregation
        robust_screen_ms = 0.0
        rejected_clients: list[str] = []
        active_updates: list[ClientUpdate] = list(updates)
        if self._robust_screener is not None and len(updates) >= 3:
            t_screen = time.perf_counter()
            screen_res = self._robust_screener.aggregate(global_flat_params, updates)
            robust_screen_ms = (time.perf_counter() - t_screen) * 1000.0
            accepted_set = set(screen_res.accepted_clients)
            rejected_clients = list(screen_res.rejected_clients)
            active_updates = [u for u in updates if u.client_id in accepted_set]

        active_participants = tuple(sorted(u.client_id for u in active_updates))

        # 1. Tenant-side: quantize -> pairwise mask -> linear checksum -> Ed25519 sign
        submissions: list[SignedTenantSubmission] = []
        weights_map: dict[str, float] = {u.client_id: 0.0 for u in updates}
        total_samples = sum(max(1, int(u.num_samples)) for u in active_updates)
        for upd in active_updates:
            if upd.flat_params.numel() != numel:
                raise ValueError(
                    f"Client '{upd.client_id}' flat_params length ({upd.flat_params.numel()}) "
                    f"!= global_flat_params length ({numel})."
                )
            sub = create_signed_submission(
                client_id=upd.client_id,
                flat_params=upd.flat_params,
                num_samples=upd.num_samples,
                round_idx=round_idx,
                key_material=self._materials[upd.client_id],
                registry=self._registry,
                group_key=self._group_key,
                participants=active_participants,
                scale_bits=self.scale_bits,
                modulus=self.modulus,
                seed=self.seed,
            )
            submissions.append(sub)
            weights_map[upd.client_id] = float(max(1, int(upd.num_samples))) / float(total_samples)

        # 2. Server-side: verify signatures/nonces, build Merkle tree, unmask & dequantize
        proof = server_secure_aggregate(
            submissions,
            self._registry,
            round_idx=round_idx,
            expected_clients=active_participants,
            seen_nonces=self._server_seen_nonces,
            scale_bits=self.scale_bits,
            modulus=self.modulus,
        )
        self.last_proof = proof

        # 3. Tenant-side: verify proof before accepting global model update
        verdict: VerificationVerdict | None = None
        if self.verify_on_aggregate:
            verdict = verify_aggregate(
                proof,
                self._registry,
                self._group_key,
                expected_round_idx=round_idx,
                expected_clients=active_participants,
                seen_nonces=self._client_seen_nonces,
                scale_bits=self.scale_bits,
            )
            self.last_verdict = verdict
            if self.raise_on_tamper and not verdict.ok:
                raise RuntimeError(f"SecureAggregator verification failed: {verdict.reason}")

        total_elapsed = time.perf_counter() - t_total_start
        new_global = proof.aggregate_float.to(device=device, dtype=dtype)

        n_clients = len(submissions)
        mean_q_ms = sum(s.quantize_ms for s in submissions) / n_clients
        mean_m_ms = sum(s.mask_ms for s in submissions) / n_clients
        mean_c_ms = sum(s.checksum_ms for s in submissions) / n_clients
        mean_s_ms = sum(s.sign_ms for s in submissions) / n_clients
        upload_bytes_per_tenant = int(
            round(sum(s.total_upload_bytes for s in submissions) / n_clients)
        )
        plain_bytes_per_tenant = 24 + 4 * int(numel)

        meta: dict[str, Any] = {
            "total_samples": int(total_samples),
            "num_clients": n_clients,
            "num_submitted": len(updates),
            "robust_screen_ms": float(robust_screen_ms),
            "scale_bits": self.scale_bits,
            "merkle_root_hex": proof.merkle_root.hex(),
            "proof_wire_bytes": int(proof.proof_wire_bytes),
            "secure_upload_bytes_per_tenant": upload_bytes_per_tenant,
            "plain_upload_bytes_per_tenant": plain_bytes_per_tenant,
            "upload_overhead_ratio": float(upload_bytes_per_tenant) / float(plain_bytes_per_tenant),
            "quantize_ms_per_tenant": float(mean_q_ms),
            "mask_ms_per_tenant": float(mean_m_ms),
            "checksum_ms_per_tenant": float(mean_c_ms),
            "sign_ms_per_tenant": float(mean_s_ms),
            "server_agg_ms": float(proof.server_time_ms),
            "verify_ms_per_tenant": float(verdict.verify_time_ms) if verdict is not None else 0.0,
            "verified": bool(verdict.ok) if verdict is not None else None,
        }

        return AggregationResult(
            new_global_flat_params=new_global,
            aggregation_time_sec=float(total_elapsed),
            accepted_clients=list(proof.included_client_ids),
            rejected_clients=rejected_clients,
            weights=weights_map,
            metadata=meta,
        )

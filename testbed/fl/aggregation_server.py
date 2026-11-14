"""Live SDN Federated Aggregation Server (`Step 08`).

Combines:
- Robust Byzantine filtering (`trustfed.aggregation.clustering.CosineClusteringAggregator`)
- Verifiable Secure Aggregation (`trustfed.verify.protocol.server_secure_aggregate`)
- Optional server tamper injection (`tamper_*`) to verify tenant rejection
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from trustfed.aggregation import CosineClusteringAggregator
from trustfed.fl.aggregator import ClientUpdate, FedAvgAggregator
from trustfed.models import BaseFlowClassifier, evaluate_pytorch_model, get_default_device
from trustfed.verify import (
    DEFAULT_SCALE_BITS,
    MODULUS_M,
    SignedTenantSubmission,
    TenantPublicRegistry,
    VerificationProof,
    server_secure_aggregate,
    tamper_dropped_tenant,
    tamper_modified_aggregate,
    tamper_replayed_update,
    tamper_substituted_update,
)

ResignCallback = Callable[[Sequence[str], int], list[SignedTenantSubmission]]


@dataclass
class LiveAggregationOutcome:
    """Complete result returned by `LiveAggregationServer.aggregate_live_round`."""

    round_idx: int
    proof: VerificationProof | None
    new_global_flat_params: torch.Tensor
    accepted_clients: list[str]
    rejected_clients: list[str]
    robust_screen_ms: float
    server_agg_proof_ms: float
    tamper_injected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LiveAggregationServer:
    """Central aggregation server coordinating robust + cryptographically verifiable rounds."""

    def __init__(
        self,
        global_model: BaseFlowClassifier,
        registry: TenantPublicRegistry,
        *,
        robust_defense: bool = True,
        max_norm_factor: float = 2.5,
        min_cosine_to_median: float = 0.0,
        scale_bits: int = DEFAULT_SCALE_BITS,
        modulus: int = MODULUS_M,
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else get_default_device(prefer_gpu=False)
        self.global_model = global_model.to(self.device)
        self.registry = registry
        self.robust_defense = bool(robust_defense)
        self.scale_bits = int(scale_bits)
        self.modulus = int(modulus)
        self.seen_nonces: set[tuple[str, bytes]] = set()
        self.previous_submissions: list[SignedTenantSubmission] = []
        self.current_round: int = 0
        self.last_proof: VerificationProof | None = None

        self._robust = CosineClusteringAggregator(
            max_norm_factor=float(max_norm_factor),
            min_cosine_to_median=float(min_cosine_to_median),
        )
        self._fedavg = FedAvgAggregator()

    def get_global_flat_params(self) -> torch.Tensor:
        """Return detached copy of current global flat parameter tensor."""
        return self.global_model.get_flat_params().clone()

    def set_global_flat_params(self, flat_params: torch.Tensor) -> None:
        """Update server global model parameters."""
        self.global_model.set_flat_params(flat_params.detach().to(device=self.device))

    def _apply_tamper(
        self,
        proof: VerificationProof,
        submissions: Sequence[SignedTenantSubmission],
        round_idx: int,
        tamper_mode: str,
    ) -> VerificationProof:
        mode = tamper_mode.strip().lower()
        if mode == "modified_aggregate":
            return tamper_modified_aggregate(proof, delta_float=0.25, scale_bits=self.scale_bits)
        if mode == "dropped_tenant":
            return tamper_dropped_tenant(
                submissions,
                round_idx=int(round_idx),
                drop_index=-1,
                scale_bits=self.scale_bits,
                modulus=self.modulus,
            )
        if mode == "replayed_update":
            if not self.previous_submissions:
                raise RuntimeError("No previous round submissions available for replayed_update.")
            return tamper_replayed_update(
                submissions,
                self.previous_submissions[0],
                round_idx=int(round_idx),
                scale_bits=self.scale_bits,
                modulus=self.modulus,
            )
        if mode == "substituted_update":
            return tamper_substituted_update(
                submissions,
                round_idx=int(round_idx),
                replace_index=0,
                scale_bits=self.scale_bits,
                modulus=self.modulus,
            )
        raise ValueError(f"Unknown tamper_mode '{tamper_mode}'.")

    def aggregate_live_round(
        self,
        round_idx: int,
        global_flat_params: torch.Tensor,
        client_updates: Sequence[ClientUpdate],
        submissions: Sequence[SignedTenantSubmission],
        *,
        resign_cohort_fn: ResignCallback | None = None,
        use_verifiable_secagg: bool = True,
        tamper_mode: str | None = None,
    ) -> LiveAggregationOutcome:
        """Screen updates robustly, then run verifiable secure aggregation over honest cohort."""
        t0 = time.perf_counter()
        if self.robust_defense and len(client_updates) >= 3:
            screen_res = self._robust.aggregate(global_flat_params, client_updates)
        else:
            screen_res = self._fedavg.aggregate(global_flat_params, client_updates)
        robust_ms = (time.perf_counter() - t0) * 1000.0

        accepted = list(screen_res.accepted_clients)
        rejected = list(screen_res.rejected_clients)

        if not use_verifiable_secagg:
            self.set_global_flat_params(screen_res.new_global_flat_params)
            self.current_round = int(round_idx)
            return LiveAggregationOutcome(
                round_idx=int(round_idx),
                proof=None,
                new_global_flat_params=screen_res.new_global_flat_params,
                accepted_clients=accepted,
                rejected_clients=rejected,
                robust_screen_ms=float(robust_ms),
                server_agg_proof_ms=float(screen_res.aggregation_time_sec * 1000.0),
            )

        active_subs = [s for s in submissions if s.header.client_id in accepted]
        if rejected and len(accepted) >= 2 and resign_cohort_fn is not None:
            active_subs = list(resign_cohort_fn(accepted, int(round_idx)))

        if len(active_subs) < 2:
            # Degenerate single-client fallback (pairwise masking needs >=2 participants)
            self.set_global_flat_params(screen_res.new_global_flat_params)
            self.current_round = int(round_idx)
            return LiveAggregationOutcome(
                round_idx=int(round_idx),
                proof=None,
                new_global_flat_params=screen_res.new_global_flat_params,
                accepted_clients=accepted,
                rejected_clients=rejected,
                robust_screen_ms=float(robust_ms),
                server_agg_proof_ms=float(screen_res.aggregation_time_sec * 1000.0),
                metadata={"fallback": "single_survivor_no_secagg"},
            )

        proof = server_secure_aggregate(
            active_subs,
            self.registry,
            round_idx=int(round_idx),
            expected_clients=accepted,
            seen_nonces=self.seen_nonces,
            scale_bits=self.scale_bits,
            modulus=self.modulus,
        )
        # Apply soft norm-clipped / secure aggregated parameters cleanly
        new_flat = proof.aggregate_float.detach().to(device=self.device).clone()

        if tamper_mode is not None:
            tampered_proof = self._apply_tamper(proof, active_subs, int(round_idx), tamper_mode)
            return LiveAggregationOutcome(
                round_idx=int(round_idx),
                proof=tampered_proof,
                new_global_flat_params=new_flat,
                accepted_clients=accepted,
                rejected_clients=rejected,
                robust_screen_ms=float(robust_ms),
                server_agg_proof_ms=float(proof.server_time_ms),
                tamper_injected=tamper_mode,
            )

        self.previous_submissions = list(submissions)
        self.last_proof = proof
        self.set_global_flat_params(new_flat)
        self.current_round = int(round_idx)

        return LiveAggregationOutcome(
            round_idx=int(round_idx),
            proof=proof,
            new_global_flat_params=new_flat,
            accepted_clients=accepted,
            rejected_clients=rejected,
            robust_screen_ms=float(robust_ms),
            server_agg_proof_ms=float(proof.server_time_ms),
        )

    def evaluate_global(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        batch_size: int = 2048,
    ) -> dict[str, Any]:
        """Evaluate the server's current global model against held-out live test flows."""
        return evaluate_pytorch_model(
            self.global_model,
            X_test,
            y_test,
            batch_size=batch_size,
            device=self.device,
        )

"""Base interface for federated poisoning attacks (`Step 04`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from trustfed.fl.aggregator import ClientUpdate, serialize_tensor_payload


class PoisoningAttack:
    """Base class for data-level and model-update poisoning attacks."""

    name: str = "none"

    def poison_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        num_classes: int = 2,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optionally poison a malicious tenant's local training dataset `(X, y)`.

        Default implementation returns `(X, y)` unchanged (for pure model-update attacks).
        """
        _ = (num_classes, seed)
        return X, y

    def poison_update(
        self,
        global_flat_params: torch.Tensor,
        honest_update: ClientUpdate,
        *,
        round_idx: int = 0,
        seed: int = 0,
    ) -> ClientUpdate:
        """Optionally transform a malicious tenant's `ClientUpdate` before upload."""
        _ = (global_flat_params, round_idx, seed)
        return honest_update

    @staticmethod
    def rebuild_client_update(
        global_flat_params: torch.Tensor,
        base_update: ClientUpdate,
        new_flat_delta: torch.Tensor,
        *,
        attack_name: str,
        extra_meta: dict[str, Any] | None = None,
    ) -> ClientUpdate:
        """Construct a valid `ClientUpdate` from a modified parameter delta `new_flat_delta`."""
        ref_global = global_flat_params.detach().to(
            device=new_flat_delta.device, dtype=new_flat_delta.dtype
        )
        new_flat_params = ref_global + new_flat_delta
        wire_payload = serialize_tensor_payload(
            new_flat_params,
            num_samples=base_update.num_samples,
            round_idx=base_update.round_idx,
        )
        meta = dict(base_update.metadata)
        meta["is_malicious"] = True
        meta["attack"] = attack_name
        if extra_meta:
            meta.update(extra_meta)

        return ClientUpdate(
            client_id=base_update.client_id,
            flat_params=new_flat_params,
            flat_delta=new_flat_delta,
            num_samples=base_update.num_samples,
            train_loss=base_update.train_loss,
            local_train_time_sec=base_update.local_train_time_sec,
            upload_bytes=len(wire_payload),
            round_idx=base_update.round_idx,
            metadata=meta,
        )

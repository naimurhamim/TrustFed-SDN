"""Pluggable flat-tensor Aggregator interface and FedAvg implementation (`Step 03`).

All aggregators in TrustFed-SDN (FedAvg in Step 03, Trimmed Mean / Cosine Clustering /
Static-Weight in Step 04, DP-aware aggregation in Step 05, and Verifiable Secure
Aggregation in Step 06) operate on 1-D flattened parameter tensors (`torch.Tensor`)
so they can be composed cleanly in both simulation (`sim/`) and the live SDN testbed (`testbed/`).
"""

from __future__ import annotations

import abc
import struct
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


def serialize_tensor_payload(
    flat_tensor: torch.Tensor,
    *,
    num_samples: int = 0,
    round_idx: int = 0,
) -> bytes:
    """Serialize a 1-D float32 parameter tensor + header metadata into actual wire bytes.

    Wire format:
      - 4-byte magic `b"TFSD"`
      - 4-byte uint32 `round_idx`
      - 8-byte uint64 `num_samples`
      - 8-byte uint64 `numel`
      - `4 * numel` raw IEEE-754 float32 bytes
    Total header = 24 bytes + `4 * numel` payload bytes.
    """
    arr_cpu = flat_tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw_bytes = arr_cpu.numpy().tobytes()
    header = struct.pack(
        "<4sIQQ",
        b"TFSD",
        int(max(0, round_idx)),
        int(max(0, num_samples)),
        int(arr_cpu.numel()),
    )
    return header + raw_bytes


def deserialize_tensor_payload(payload: bytes) -> tuple[torch.Tensor, int, int]:
    """Deserialize wire bytes produced by `serialize_tensor_payload`."""
    if len(payload) < 24:
        raise ValueError(f"Payload too short ({len(payload)} bytes < 24-byte header).")
    magic, round_idx, num_samples, numel = struct.unpack("<4sIQQ", payload[:24])
    if magic != b"TFSD":
        raise ValueError(f"Invalid payload magic: {magic!r}")
    expected_len = 24 + 4 * int(numel)
    if len(payload) != expected_len:
        raise ValueError(
            f"Payload byte length mismatch: got {len(payload)}, expected {expected_len}."
        )
    tensor = torch.frombuffer(bytearray(payload[24:]), dtype=torch.float32).clone()
    return tensor, int(num_samples), int(round_idx)


@dataclass
class ClientUpdate:
    """Single tenant's update submitted to the FL server at the end of a round."""

    client_id: str
    flat_params: torch.Tensor
    flat_delta: torch.Tensor
    num_samples: int
    train_loss: float
    local_train_time_sec: float
    upload_bytes: int
    round_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result returned by an `Aggregator.aggregate(...)` call."""

    new_global_flat_params: torch.Tensor
    aggregation_time_sec: float
    accepted_clients: list[str]
    rejected_clients: list[str]
    weights: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class Aggregator(abc.ABC):
    """Abstract base class for all flat-tensor FL aggregators."""

    name: str = "base"

    @abc.abstractmethod
    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """Combine client updates into the next global flat parameter tensor."""


class FedAvgAggregator(Aggregator):
    """Standard McMahan et al. (2017) Federated Averaging weighted by local sample count `n_i`.

    Computes:
        `w^{(t+1)} = sum_{i=1}^K (n_i / sum_j n_j) * w_i^{(t)}`
    If all `n_i == 0` (degenerate edge case), falls back to uniform `1 / K` weights.
    """

    name: str = "fedavg"

    def aggregate(
        self,
        global_flat_params: torch.Tensor,
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        if not updates:
            raise ValueError("FedAvgAggregator received an empty updates sequence.")

        t0 = time.perf_counter()
        device = global_flat_params.device
        dtype = global_flat_params.dtype

        total_samples = sum(max(0, int(u.num_samples)) for u in updates)
        n_clients = len(updates)

        weights_map: dict[str, float] = {}
        accum = torch.zeros_like(global_flat_params, device=device, dtype=torch.float64)

        for u in updates:
            if u.flat_params.numel() != global_flat_params.numel():
                raise ValueError(
                    f"Client '{u.client_id}' flat_params length ({u.flat_params.numel()}) "
                    f"!= global_flat_params length ({global_flat_params.numel()})."
                )
            w_i = (
                float(max(0, int(u.num_samples))) / float(total_samples)
                if total_samples > 0
                else 1.0 / float(n_clients)
            )
            weights_map[u.client_id] = w_i
            accum.add_(u.flat_params.to(device=device, dtype=torch.float64), alpha=w_i)

        new_global = accum.to(dtype=dtype)
        elapsed = time.perf_counter() - t0

        return AggregationResult(
            new_global_flat_params=new_global,
            aggregation_time_sec=float(elapsed),
            accepted_clients=[u.client_id for u in updates],
            rejected_clients=[],
            weights=weights_map,
            metadata={"total_samples": int(total_samples), "num_clients": n_clients},
        )

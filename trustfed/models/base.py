"""Unified base model interface and sequence-layout helpers for TrustFed-SDN."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

SequenceMode = Literal["single_step", "feature_chunks", "per_feature"]
VALID_SEQ_MODES: tuple[str, ...] = ("single_step", "feature_chunks", "per_feature")


def compute_sequence_Dimensions(
    input_dim: int,
    seq_mode: SequenceMode = "feature_chunks",
    num_chunks: int = 4,
) -> tuple[int, int, int]:
    """Compute `(seq_len, step_input_size, pad_len)` for presenting a 2-D tabular
    feature vector `(B, input_dim)` to a recurrent layer (`GRU` or `LSTM`).

    Args:
        input_dim: Number of selected tabular flow features `D`.
        seq_mode: One of:
            - `"single_step"`: Sequence of length `T=1`, feature width `F=D`.
            - `"feature_chunks"`: Splits features into `T=num_chunks` steps of width
              `F=ceil(D / num_chunks)` (zero-padded if `D` is not divisible by `num_chunks`).
            - `"per_feature"`: Sequence of length `T=D`, scalar width `F=1` per step.
        num_chunks: Number of chunks when `seq_mode == "feature_chunks"`.

    Returns:
        Tuple `(seq_len, step_input_size, pad_len)`.
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if seq_mode not in VALID_SEQ_MODES:
        raise ValueError(f"Unsupported seq_mode '{seq_mode}'. Expected one of {VALID_SEQ_MODES}.")

    if seq_mode == "single_step":
        return 1, input_dim, 0
    if seq_mode == "per_feature":
        return input_dim, 1, 0

    # feature_chunks
    eff_chunks = max(1, min(num_chunks, input_dim))
    step_width = math.ceil(input_dim / eff_chunks)
    total_len = eff_chunks * step_width
    pad_len = total_len - input_dim
    return eff_chunks, step_width, pad_len


def reshape_tabular_to_sequence(
    x: torch.Tensor,
    seq_mode: SequenceMode = "feature_chunks",
    num_chunks: int = 4,
) -> torch.Tensor:
    """Reshape a 2-D tabular tensor `(B, D)` (or already 3-D `(B, T, F)`) into `(B, T, F)`."""
    if x.ndim == 3:
        return x
    if x.ndim != 2:
        raise ValueError(
            f"Expected 2-D tensor (B, D) or 3-D tensor (B, T, F), got shape {tuple(x.shape)}"
        )

    batch_size, input_dim = x.shape
    seq_len, step_width, pad_len = compute_sequence_Dimensions(
        input_dim, seq_mode=seq_mode, num_chunks=num_chunks
    )
    if pad_len > 0:
        pad = torch.zeros((batch_size, pad_len), dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad], dim=1)
    return x.view(batch_size, seq_len, step_width)


class BaseFlowClassifier(nn.Module):
    """Abstract base class for all TrustFed-SDN flow classifiers (`DNN`, `GRU`, `LSTM`).

    Guarantees an identical input/output signature:
        `logits = model(x)` where `x` has shape `(batch_size, input_dim)`
        and `logits` has shape `(batch_size, num_classes)`.
    Also provides flat parameter vector serialization for Federated Learning (Steps 03-08).
    """

    model_type: str = "base"

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flat_params(self) -> torch.Tensor:
        """Flatten all trainable parameters into a single 1-D float32 CPU/device tensor."""
        params = [p.detach().reshape(-1) for p in self.parameters() if p.requires_grad]
        if not params:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(params, dim=0).clone()

    @torch.no_grad()
    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Load a 1-D flattened parameter tensor back into trainable parameters."""
        expected = self.count_parameters()
        if flat_params.numel() != expected:
            raise ValueError(
                f"Flat parameter length mismatch: got {flat_params.numel()}, expected {expected}"
            )
        offset = 0
        for p in self.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            chunk = flat_params[offset : offset + numel].view_as(p)
            p.copy_(chunk.to(device=p.device, dtype=p.dtype))
            offset += numel

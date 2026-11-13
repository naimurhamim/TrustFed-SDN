"""Fixed-point integer quantization utilities for Verifiable Secure Aggregation (`Step 06`)."""

from __future__ import annotations

import math

import torch

DEFAULT_SCALE_BITS: int = 20


def quantize_tensor(
    tensor: torch.Tensor,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> torch.Tensor:
    """Quantize a floating-point tensor into signed `int64` fixed-point integers.

    Computes `round(x * 2^scale_bits)` in float64 to avoid float32 rounding artifacts.
    """
    if scale_bits < 1 or scale_bits > 30:
        raise ValueError(f"scale_bits must be in [1, 30], got {scale_bits}.")
    scale = float(1 << int(scale_bits))
    arr64 = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not torch.isfinite(arr64).all():
        raise ValueError("Cannot quantize tensor containing NaN or Inf values.")
    return torch.round(arr64 * scale).to(dtype=torch.int64)


def dequantize_tensor(
    q_tensor: torch.Tensor,
    scale_bits: int = DEFAULT_SCALE_BITS,
    *,
    denominator: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Recover floating-point values from fixed-point `int64` representation."""
    if scale_bits < 1 or scale_bits > 30:
        raise ValueError(f"scale_bits must be in [1, 30], got {scale_bits}.")
    if denominator <= 0.0:
        raise ValueError(f"denominator must be > 0, got {denominator}.")
    denom = float(1 << int(scale_bits)) * float(denominator)
    out64 = q_tensor.detach().to(device="cpu", dtype=torch.float64) / denom
    out = out64.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


def quantize_weighted_update(
    flat_params: torch.Tensor,
    num_samples: int,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> torch.Tensor:
    """Quantize a tenant parameter vector and pre-multiply by integer weight `n_i`.

    Exact algebraic form:
        `x_tilde_i = max(1, n_i) * round(w_i * 2^scale_bits)`
    """
    weight_int = max(1, int(num_samples))
    q_w = quantize_tensor(flat_params, scale_bits=scale_bits)
    return q_w * weight_int


def dequantize_weighted_sum(
    sum_int: torch.Tensor,
    total_samples: int,
    scale_bits: int = DEFAULT_SCALE_BITS,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Divide the recovered exact integer weighted sum by `(sum_i n_i) * 2^scale_bits`."""
    denom_samples = max(1, int(total_samples))
    return dequantize_tensor(
        sum_int,
        scale_bits=scale_bits,
        denominator=float(denom_samples),
        dtype=dtype,
        device=device,
    )


def measure_quantization_error(
    tensor: torch.Tensor,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> dict[str, float]:
    """Measure round-trip quantization error (`max_abs_error`, `mean_abs_error`, `rmse`, `snr_db`).

    Returns float summary metrics for the given `scale_bits`.
    """
    orig = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    q = quantize_tensor(orig, scale_bits=scale_bits)
    rec = dequantize_tensor(q, scale_bits=scale_bits, dtype=torch.float64)
    diff = orig - rec
    abs_diff = diff.abs()

    max_abs = float(abs_diff.max().item()) if diff.numel() > 0 else 0.0
    mean_abs = float(abs_diff.mean().item()) if diff.numel() > 0 else 0.0
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() > 0 else 0.0
    signal_power = float(torch.mean(orig * orig).item()) if diff.numel() > 0 else 0.0
    noise_power = float(torch.mean(diff * diff).item()) if diff.numel() > 0 else 0.0
    if noise_power == 0.0:
        snr_db = float("inf")
    elif signal_power == 0.0:
        snr_db = 0.0
    else:
        snr_db = 10.0 * math.log10(signal_power / noise_power)

    return {
        "scale_bits": float(scale_bits),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "snr_db": snr_db,
    }

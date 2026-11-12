"""Record-level DP-SGD with per-sample gradient clipping and Poisson sampling (`Step 05`).

Guarantee implemented (approved design, `docs/DP_DESIGN.md`):
    Each tenant runs DP-SGD on its own flow records. In every local step, each record is
    included independently with probability `q` (Poisson sampling), the gradient of every
    included record is clipped to L2 norm `C`, the clipped gradients are summed, Gaussian
    noise `N(0, sigma^2 C^2 I)` is added, and the result is divided by the expected batch size
    `q * n`. The RDP accountant in `trustfed.privacy.accountant` tracks the cumulative
    `(epsilon, delta)` of the tenant over all rounds and local steps.

    Adjacent datasets differ in one flow record of that tenant (add/remove). The guarantee
    covers individual flow records; it does not hide aggregate tenant-level properties
    (for example the tenant's overall attack mix).

Per-sample gradients use `torch.func.vmap(grad(...))`. Because `nn.GRU` has no vmap batching
rule, `GRUClassifier` is evaluated with an exact functional re-implementation of the PyTorch
GRU equations (`functional_gru_forward`) that reads the same parameter tensors, so the flat
parameter layout and the FL wire format are unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from trustfed.fl.aggregator import ClientUpdate, serialize_tensor_payload
from trustfed.fl.client import FLClient
from trustfed.models.base import BaseFlowClassifier, reshape_tabular_to_sequence
from trustfed.models.gru import GRUClassifier
from trustfed.models.trainer import set_all_seeds
from trustfed.privacy.accountant import RDPAccountant, calibrate_noise_multiplier

if TYPE_CHECKING:
    from trustfed.attacks.base import PoisoningAttack


@dataclass
class DPConfig:
    """Per-tenant DP-SGD configuration."""

    target_epsilon: float
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    expected_batch_size: int = 128
    noise_multiplier: float | None = None  # calibrated if None

    def sample_rate(self, num_samples: int) -> float:
        """Poisson sampling rate `q = expected_batch_size / n` (capped at 1)."""
        return float(min(1.0, self.expected_batch_size / max(1, int(num_samples))))


def steps_per_epoch(sample_rate: float) -> int:
    """Number of Poisson-sampled steps that make up one expected pass over the data."""
    return max(1, int(math.ceil(1.0 / max(sample_rate, 1e-12))))


def total_dp_steps(sample_rate: float, rounds: int, local_epochs: int) -> int:
    """Total DP-SGD steps a tenant performs across all FL rounds."""
    return int(rounds) * int(local_epochs) * steps_per_epoch(sample_rate)


def calibrate_tenant_noise(
    cfg: DPConfig,
    num_samples: int,
    rounds: int,
    local_epochs: int,
) -> float:
    """Noise multiplier that spends exactly (at most) `cfg.target_epsilon` over training."""
    q = cfg.sample_rate(num_samples)
    steps = total_dp_steps(q, rounds, local_epochs)
    return calibrate_noise_multiplier(cfg.target_epsilon, q, steps, cfg.delta)


def _gru_cell(
    x_t: torch.Tensor,
    h: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: torch.Tensor | None,
    b_hh: torch.Tensor | None,
) -> torch.Tensor:
    gi = F.linear(x_t, w_ih, b_ih)
    gh = F.linear(h, w_hh, b_hh)
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1.0 - z) * n + z * h


def functional_gru_forward(
    model: GRUClassifier,
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """Exact functional GRUClassifier forward (no inter-layer dropout) using `params`.

    Accepts `x` of shape `(B, D)` and returns logits `(B, num_classes)`.
    """
    if model.bidirectional:
        raise NotImplementedError("functional_gru_forward does not support bidirectional GRU.")
    seq = reshape_tabular_to_sequence(x, seq_mode=model.seq_mode, num_chunks=model.num_chunks)
    batch = seq.shape[0]
    layer_in = seq
    for layer in range(model.num_layers):
        w_ih = params[f"gru.weight_ih_l{layer}"]
        w_hh = params[f"gru.weight_hh_l{layer}"]
        b_ih = params.get(f"gru.bias_ih_l{layer}")
        b_hh = params.get(f"gru.bias_hh_l{layer}")
        h = torch.zeros(batch, model.hidden_dim, dtype=seq.dtype, device=seq.device)
        outs = []
        for t in range(layer_in.shape[1]):
            h = _gru_cell(layer_in[:, t, :], h, w_ih, w_hh, b_ih, b_hh)
            outs.append(h)
        layer_in = torch.stack(outs, dim=1)
    if layer_in.shape[1] > 1:
        feat = 0.5 * (layer_in[:, -1, :] + layer_in.mean(dim=1))
    else:
        feat = layer_in[:, -1, :]
    head_params = {k[len("head.") :]: v for k, v in params.items() if k.startswith("head.")}
    return functional_call(model.head, head_params, (feat,))


def _forward_with_params(
    model: BaseFlowClassifier,
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    if isinstance(model, GRUClassifier):
        return functional_gru_forward(model, params, x)
    return functional_call(model, params, (x,))


def compute_per_sample_grads(
    model: BaseFlowClassifier,
    params: dict[str, torch.Tensor],
    xb: torch.Tensor,
    yb: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return per-sample gradients `{name: (B, *param_shape)}` of the cross-entropy loss."""

    def _loss(p: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = _forward_with_params(model, p, x.unsqueeze(0))
        return F.cross_entropy(logits, y.unsqueeze(0), weight=class_weights)

    return vmap(grad(_loss), in_dims=(None, 0, 0), randomness="different")(params, xb, yb)


def clip_and_sum_per_sample_grads(
    per_sample: dict[str, torch.Tensor],
    max_grad_norm: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Clip every sample's full gradient to L2 norm `max_grad_norm` and sum over the batch.

    Returns `(summed_clipped_grads, per_sample_norms_before_clipping)`.
    """
    names = list(per_sample.keys())
    batch = per_sample[names[0]].shape[0]
    sq = torch.zeros(batch, dtype=per_sample[names[0]].dtype, device=per_sample[names[0]].device)
    for n in names:
        sq = sq + per_sample[n].reshape(batch, -1).pow(2).sum(dim=1)
    norms = sq.sqrt()
    factors = torch.clamp(float(max_grad_norm) / (norms + 1e-6), max=1.0)
    summed = {n: torch.einsum("b,b...->...", factors, per_sample[n]) for n in names}
    return summed, norms


def add_gaussian_noise(
    summed: dict[str, torch.Tensor],
    noise_multiplier: float,
    max_grad_norm: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Add `N(0, (noise_multiplier * max_grad_norm)^2)` noise to every coordinate."""
    std = float(noise_multiplier) * float(max_grad_norm)
    if std <= 0.0:
        return summed
    noisy: dict[str, torch.Tensor] = {}
    for n, g in summed.items():
        noise = torch.randn(g.shape, generator=generator, dtype=g.dtype, device="cpu").to(g.device)
        noisy[n] = g + std * noise
    return noisy


class DPFLClient(FLClient):
    """FL tenant client that trains locally with record-level DP-SGD."""

    def __init__(
        self,
        client_id: str,
        client_index: int,
        model: BaseFlowClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        dp_config: DPConfig,
        rounds: int,
        local_epochs: int,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        device: torch.device | None = None,
        use_class_weights: bool = False,
        attack: PoisoningAttack | None = None,
    ) -> None:
        super().__init__(
            client_id,
            client_index,
            model,
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            device=device,
            use_class_weights=use_class_weights,
            attack=attack,
        )
        self.dp_config = dp_config
        self.sample_rate = dp_config.sample_rate(self.num_samples)
        self.steps_per_epoch = steps_per_epoch(self.sample_rate)
        if dp_config.noise_multiplier is None:
            self.noise_multiplier = calibrate_tenant_noise(
                dp_config, self.num_samples, rounds, local_epochs
            )
        else:
            self.noise_multiplier = float(dp_config.noise_multiplier)
        self.accountant = RDPAccountant()
        self.planned_steps = total_dp_steps(self.sample_rate, rounds, local_epochs)

    @property
    def epsilon_spent(self) -> float:
        """Cumulative epsilon spent so far (at `dp_config.delta`)."""
        return self.accountant.get_epsilon(self.dp_config.delta)

    def local_train(
        self,
        global_flat_params: torch.Tensor,
        *,
        round_idx: int,
        local_epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> ClientUpdate:
        """Run `local_epochs` of DP-SGD (Adam on privatized gradients) from the global model."""
        _ = batch_size  # expected batch size comes from dp_config (Poisson sampling)
        round_seed = int(seed) * 100003 + int(round_idx) * 1009 + int(self.client_index)
        set_all_seeds(round_seed)

        self.model.set_flat_params(global_flat_params)
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        gen_sample = torch.Generator(device="cpu")
        gen_sample.manual_seed(round_seed)
        gen_noise = torch.Generator(device="cpu")
        gen_noise.manual_seed(round_seed + 7919)

        cfg = self.dp_config
        q = self.sample_rate
        expected_b = q * self.num_samples
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        total_loss = 0.0
        loss_batches = 0
        clipped_frac_sum = 0.0
        n_steps = max(1, int(local_epochs)) * self.steps_per_epoch
        for _step in range(n_steps):
            mask = torch.rand(self.num_samples, generator=gen_sample) < q
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1).to(self.device)
            params = {n: p.detach() for n, p in named}
            if idx.numel() > 0:
                xb = self.X_train_tensor.index_select(0, idx)
                yb = self.y_train_tensor.index_select(0, idx)
                per_sample = compute_per_sample_grads(
                    self.model, params, xb, yb, self._class_weights
                )
                summed, norms = clip_and_sum_per_sample_grads(per_sample, cfg.max_grad_norm)
                clipped_frac_sum += float((norms > cfg.max_grad_norm).float().mean().item())
                with torch.no_grad():
                    logits = _forward_with_params(self.model, params, xb)
                    total_loss += float(
                        F.cross_entropy(logits, yb, weight=self._class_weights).item()
                    )
                    loss_batches += 1
            else:
                summed = {n: torch.zeros_like(p) for n, p in named}
            noisy = add_gaussian_noise(
                summed, self.noise_multiplier, cfg.max_grad_norm, generator=gen_noise
            )
            optimizer.zero_grad(set_to_none=True)
            for n, p in named:
                p.grad = noisy[n] / float(expected_b)
            optimizer.step()
            self.accountant.step(q, self.noise_multiplier)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0

        new_flat = self.model.get_flat_params()
        ref_flat = global_flat_params.detach().to(device=new_flat.device, dtype=new_flat.dtype)
        flat_delta = new_flat - ref_flat
        wire = serialize_tensor_payload(
            new_flat, num_samples=self.num_samples, round_idx=int(round_idx)
        )
        meta: dict[str, Any] = {
            "local_epochs": int(local_epochs),
            "dp": True,
            "noise_multiplier": float(self.noise_multiplier),
            "sample_rate": float(q),
            "max_grad_norm": float(cfg.max_grad_norm),
            "epsilon_spent": float(self.epsilon_spent),
            "delta": float(cfg.delta),
            "dp_steps_total": int(self.accountant.steps),
            "clipped_fraction": float(clipped_frac_sum / max(1, n_steps)),
            "is_malicious": self.is_malicious,
            "attack": self.attack.name if self.attack is not None else "none",
        }
        update = ClientUpdate(
            client_id=self.client_id,
            flat_params=new_flat,
            flat_delta=flat_delta,
            num_samples=self.num_samples,
            train_loss=float(total_loss / max(1, loss_batches)),
            local_train_time_sec=float(elapsed),
            upload_bytes=len(wire),
            round_idx=int(round_idx),
            metadata=meta,
        )
        if self.attack is not None:
            return self.attack.poison_update(
                global_flat_params, update, round_idx=int(round_idx), seed=round_seed
            )
        return update

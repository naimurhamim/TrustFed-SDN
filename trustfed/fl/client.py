"""Federated Learning Client (`FLClient`) for local tenant training (`Step 03`)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from trustfed.fl.aggregator import ClientUpdate, serialize_tensor_payload
from trustfed.models.base import BaseFlowClassifier
from trustfed.models.trainer import evaluate_pytorch_model, get_default_device, set_all_seeds


class FLClient:
    """Represents a single SDN tenant in federated training.

    Pre-stages local training tensors `(X_train, y_train)` on `device` (`cuda` or `cpu`)
    to avoid repeated host-to-device copies across FL rounds while keeping every
    round's mini-batch order strictly deterministic with respect to
    `(seed, round_idx, client_index)`.
    """

    def __init__(
        self,
        client_id: str,
        client_index: int,
        model: BaseFlowClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        device: torch.device | None = None,
        use_class_weights: bool = False,
    ) -> None:
        if len(X_train) == 0:
            raise ValueError(f"FLClient '{client_id}' received empty X_train.")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"FLClient '{client_id}' X_train ({len(X_train)}) and y_train ({len(y_train)}) "
                "length mismatch."
            )

        self.client_id = str(client_id)
        self.client_index = int(client_index)
        self.device = device if device is not None else get_default_device(prefer_gpu=True)
        self.model = model.to(self.device)

        self.X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train_tensor = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self._num_samples = int(self.X_train_tensor.shape[0])

        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self._class_weights: torch.Tensor | None = None
        if use_class_weights:
            counts = np.bincount(
                np.asarray(y_train, dtype=np.int64), minlength=self.model.num_classes
            )
            counts_safe = np.maximum(counts, 1).astype(np.float64)
            inv = np.sqrt(float(self._num_samples) / (self.model.num_classes * counts_safe))
            inv = np.clip(inv, 0.5, 5.0)
            self._class_weights = torch.as_tensor(inv, dtype=torch.float32, device=self.device)

    @property
    def num_samples(self) -> int:
        """Number of local training flows held by this tenant."""
        return self._num_samples

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
        """Run `local_epochs` of Adam local training starting from `global_flat_params`."""
        round_seed = int(seed) * 100003 + int(round_idx) * 1009 + int(self.client_index)
        set_all_seeds(round_seed)

        self.model.set_flat_params(global_flat_params)
        self.model.train()

        criterion = nn.CrossEntropyLoss(weight=self._class_weights)
        use_fused = self.device.type == "cuda"
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
            fused=use_fused,
        )

        gen = torch.Generator(device="cpu")
        gen.manual_seed(round_seed)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        total_loss_tensor = torch.zeros((), dtype=torch.float64, device=self.device)
        total_batches = 0
        eff_batch_size = max(1, int(batch_size))

        for _epoch in range(max(1, int(local_epochs))):
            perm = torch.randperm(self._num_samples, generator=gen).to(self.device)
            X_shuf = self.X_train_tensor.index_select(0, perm)
            y_shuf = self.y_train_tensor.index_select(0, perm)
            for start in range(0, self._num_samples, eff_batch_size):
                xb = X_shuf[start : start + eff_batch_size]
                yb = y_shuf[start : start + eff_batch_size]

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss_tensor = total_loss_tensor + loss.detach().to(dtype=torch.float64)
                total_batches += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_sec = time.perf_counter() - t0
        total_loss = float(total_loss_tensor.item())

        new_flat = self.model.get_flat_params()
        ref_flat = global_flat_params.detach().to(device=new_flat.device, dtype=new_flat.dtype)
        flat_delta = new_flat - ref_flat

        wire_payload = serialize_tensor_payload(
            new_flat,
            num_samples=self._num_samples,
            round_idx=int(round_idx),
        )
        upload_bytes = len(wire_payload)
        avg_loss = total_loss / max(1, total_batches)

        return ClientUpdate(
            client_id=self.client_id,
            flat_params=new_flat,
            flat_delta=flat_delta,
            num_samples=self._num_samples,
            train_loss=float(avg_loss),
            local_train_time_sec=float(elapsed_sec),
            upload_bytes=int(upload_bytes),
            round_idx=int(round_idx),
            metadata={"local_epochs": int(local_epochs), "batch_size": eff_batch_size},
        )

    def evaluate_local_test(
        self,
        model: BaseFlowClassifier | None = None,
        *,
        batch_size: int = 4096,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate `model` (or `self.model`) on this tenant's local test partition."""
        if self.X_test is None or self.y_test is None or len(self.X_test) == 0:
            raise ValueError(f"FLClient '{self.client_id}' has no local test set configured.")
        target_model = model if model is not None else self.model
        return evaluate_pytorch_model(
            target_model,
            self.X_test,
            self.y_test,
            batch_size=batch_size,
            device=self.device,
            class_names=class_names,
        )

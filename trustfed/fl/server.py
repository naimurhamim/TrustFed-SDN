"""Federated Learning Server (`FLServer`) and round coordination loop (`Step 03`)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from trustfed.fl.aggregator import (
    AggregationResult,
    Aggregator,
    ClientUpdate,
    FedAvgAggregator,
    serialize_tensor_payload,
)
from trustfed.fl.client import FLClient
from trustfed.models.base import BaseFlowClassifier
from trustfed.models.trainer import evaluate_pytorch_model, get_default_device


class FLServer:
    """Coordinates synchronous federated learning rounds across `FLClient` instances.

    Logs per-round:
      - Global test metrics (`accuracy`, `precision_macro`, `recall_macro`, `f1_macro`,
        `f1_attack`, `attack_detection_rate`, `false_positive_rate`, `roc_auc`)
      - Per-tenant test metrics (`tenant_f1_macro_mean`, `tenant_f1_macro_std`,
        `tenant_f1_macro_min`, and individual tenant Macro-F1 / ADR / FPR)
      - Communication overhead in actual serialized wire bytes (`download_bytes`,
        `upload_bytes`, `total_comm_bytes`, and cumulative bytes)
      - Wall-clock time split into `local_train_time_sec` (sum & max across tenants)
        and `aggregation_time_sec`.
    """

    def __init__(
        self,
        global_model: BaseFlowClassifier,
        clients: Sequence[FLClient],
        *,
        aggregator: Aggregator | None = None,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_names: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        if not clients:
            raise ValueError("FLServer requires at least one FLClient.")
        self.device = device if device is not None else get_default_device(prefer_gpu=True)
        self.global_model = global_model.to(self.device)
        self.clients = list(clients)
        self.aggregator: Aggregator = aggregator if aggregator is not None else FedAvgAggregator()
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.class_names = class_names
        self._cumulative_comm_bytes = 0

    def get_global_flat_params(self) -> torch.Tensor:
        """Return a detached 1-D copy of the current global model parameters."""
        return self.global_model.get_flat_params()

    def evaluate_global(
        self,
        *,
        batch_size: int = 8192,
    ) -> dict[str, Any]:
        """Evaluate `self.global_model` on the held-out global test set and per-tenant test sets."""
        t0 = time.perf_counter()
        result: dict[str, Any] = {}

        if self.X_test is not None and self.y_test is not None and len(self.X_test) > 0:
            global_metrics = evaluate_pytorch_model(
                self.global_model,
                self.X_test,
                self.y_test,
                batch_size=batch_size,
                device=self.device,
                class_names=self.class_names,
            )
            result["global_test"] = global_metrics

        if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
            val_metrics = evaluate_pytorch_model(
                self.global_model,
                self.X_val,
                self.y_val,
                batch_size=batch_size,
                device=self.device,
                class_names=self.class_names,
            )
            result["global_val"] = val_metrics

        per_tenant_metrics: dict[str, dict[str, Any]] = {}
        tenant_f1s: list[float] = []
        tenant_adrs: list[float] = []
        for client in self.clients:
            if client.X_test is not None and client.y_test is not None and len(client.X_test) > 0:
                tm = client.evaluate_local_test(
                    self.global_model,
                    batch_size=batch_size,
                    class_names=self.class_names,
                )
                per_tenant_metrics[client.client_id] = tm
                tenant_f1s.append(float(tm["f1_macro"]))
                tenant_adrs.append(float(tm["attack_detection_rate"]))

        if tenant_f1s:
            arr_f1 = np.asarray(tenant_f1s, dtype=np.float64)
            arr_adr = np.asarray(tenant_adrs, dtype=np.float64)
            result["per_tenant_test"] = per_tenant_metrics
            result["tenant_f1_macro_mean"] = float(np.mean(arr_f1))
            result["tenant_f1_macro_std"] = float(np.std(arr_f1, ddof=0))
            result["tenant_f1_macro_min"] = float(np.min(arr_f1))
            result["tenant_f1_macro_max"] = float(np.max(arr_f1))
            result["tenant_adr_mean"] = float(np.mean(arr_adr))

        result["eval_time_sec"] = float(time.perf_counter() - t0)
        return result

    def run_round(
        self,
        round_idx: int,
        *,
        local_epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        evaluate: bool = True,
        eval_per_tenant: bool = True,
    ) -> dict[str, Any]:
        """Execute a single synchronous FL round across all participating clients."""
        t_round_start = time.perf_counter()
        global_flat = self.global_model.get_flat_params()

        # Measure actual broadcast wire payload bytes sent from server to K clients
        broadcast_wire = serialize_tensor_payload(
            global_flat,
            num_samples=0,
            round_idx=int(round_idx),
        )
        download_bytes = len(broadcast_wire) * len(self.clients)

        updates: list[ClientUpdate] = []
        for client in self.clients:
            upd = client.local_train(
                global_flat,
                round_idx=int(round_idx),
                local_epochs=int(local_epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                weight_decay=float(weight_decay),
                seed=int(seed),
            )
            updates.append(upd)

        agg_result: AggregationResult = self.aggregator.aggregate(global_flat, updates)
        self.global_model.set_flat_params(agg_result.new_global_flat_params)

        upload_bytes = sum(int(u.upload_bytes) for u in updates)
        round_comm_bytes = int(download_bytes + upload_bytes)
        self._cumulative_comm_bytes += round_comm_bytes

        local_train_times = [float(u.local_train_time_sec) for u in updates]
        local_train_time_sum_sec = float(sum(local_train_times))
        local_train_time_max_sec = float(max(local_train_times))
        local_train_time_mean_sec = float(np.mean(local_train_times))

        total_samples = sum(max(1, int(u.num_samples)) for u in updates)
        weighted_train_loss = sum(
            float(u.train_loss) * (max(1, int(u.num_samples)) / float(total_samples))
            for u in updates
        )

        round_log: dict[str, Any] = {
            "round": int(round_idx),
            "aggregator": self.aggregator.name,
            "num_clients": len(self.clients),
            "accepted_clients": len(agg_result.accepted_clients),
            "rejected_clients": len(agg_result.rejected_clients),
            "train_loss_weighted": float(weighted_train_loss),
            "local_train_time_sec": local_train_time_sum_sec,
            "local_train_time_max_sec": local_train_time_max_sec,
            "local_train_time_mean_sec": local_train_time_mean_sec,
            "aggregation_time_sec": float(agg_result.aggregation_time_sec),
            "download_bytes": int(download_bytes),
            "upload_bytes": int(upload_bytes),
            "total_comm_bytes": int(round_comm_bytes),
            "cumulative_comm_bytes": int(self._cumulative_comm_bytes),
        }

        if evaluate:
            eval_start = time.perf_counter()
            if self.X_test is not None and self.y_test is not None and len(self.X_test) > 0:
                gm = evaluate_pytorch_model(
                    self.global_model,
                    self.X_test,
                    self.y_test,
                    batch_size=8192,
                    device=self.device,
                    class_names=self.class_names,
                )
                round_log["accuracy"] = float(gm["accuracy"])
                round_log["precision_macro"] = float(gm["precision_macro"])
                round_log["recall_macro"] = float(gm["recall_macro"])
                round_log["f1_macro"] = float(gm["f1_macro"])
                round_log["f1_attack"] = float(gm["f1_attack"])
                round_log["attack_detection_rate"] = float(gm["attack_detection_rate"])
                round_log["false_positive_rate"] = float(gm["false_positive_rate"])
                round_log["roc_auc"] = float(gm["roc_auc"])
                round_log["inference_us_per_flow"] = float(gm["inference_us_per_flow"])

            if eval_per_tenant:
                tenant_f1s: list[float] = []
                tenant_adrs: list[float] = []
                tenant_fprs: list[float] = []
                per_tenant_f1_map: dict[str, float] = {}
                for client in self.clients:
                    if (
                        client.X_test is not None
                        and client.y_test is not None
                        and len(client.X_test) > 0
                    ):
                        tm = client.evaluate_local_test(
                            self.global_model,
                            batch_size=8192,
                            class_names=self.class_names,
                        )
                        f1_val = float(tm["f1_macro"])
                        tenant_f1s.append(f1_val)
                        tenant_adrs.append(float(tm["attack_detection_rate"]))
                        tenant_fprs.append(float(tm["false_positive_rate"]))
                        per_tenant_f1_map[client.client_id] = f1_val

                if tenant_f1s:
                    arr_f1 = np.asarray(tenant_f1s, dtype=np.float64)
                    round_log["tenant_f1_macro_mean"] = float(np.mean(arr_f1))
                    round_log["tenant_f1_macro_std"] = float(np.std(arr_f1, ddof=0))
                    round_log["tenant_f1_macro_min"] = float(np.min(arr_f1))
                    round_log["tenant_f1_macro_max"] = float(np.max(arr_f1))
                    round_log["tenant_adr_mean"] = float(np.mean(tenant_adrs))
                    round_log["tenant_fpr_mean"] = float(np.mean(tenant_fprs))
                    round_log["per_tenant_f1_macro"] = per_tenant_f1_map

            round_log["eval_time_sec"] = float(time.perf_counter() - eval_start)

        round_log["round_wall_time_sec"] = float(time.perf_counter() - t_round_start)
        return round_log

    def fit(
        self,
        *,
        rounds: int = 30,
        local_epochs: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        eval_every: int = 1,
        eval_per_tenant_every: int = 1,
    ) -> dict[str, Any]:
        """Run `rounds` synchronous FL rounds and return full round history + final summary."""
        self._cumulative_comm_bytes = 0
        history: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for r in range(1, int(rounds) + 1):
            is_last = r == int(rounds)
            do_eval = is_last or (r == 1) or (eval_every > 0 and r % eval_every == 0)
            do_tenant_eval = is_last or (
                eval_per_tenant_every > 0 and r % eval_per_tenant_every == 0
            )
            r_log = self.run_round(
                r,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                evaluate=do_eval,
                eval_per_tenant=do_tenant_eval,
            )
            history.append(r_log)

        total_wall_sec = time.perf_counter() - t0
        final_round = history[-1] if history else {}

        # Compute convergence round (first round reaching >= 95% of final Macro-F1)
        final_f1 = float(final_round.get("f1_macro", 0.0))
        target_95 = 0.95 * final_f1
        round_to_95pct_f1 = int(rounds)
        for h in history:
            if "f1_macro" in h and float(h["f1_macro"]) >= target_95:
                round_to_95pct_f1 = int(h["round"])
                break

        return {
            "rounds": int(rounds),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "seed": int(seed),
            "aggregator": self.aggregator.name,
            "num_clients": len(self.clients),
            "param_count": int(self.global_model.count_parameters()),
            "total_wall_time_sec": float(total_wall_sec),
            "total_local_train_time_sec": float(
                sum(float(h["local_train_time_sec"]) for h in history)
            ),
            "total_aggregation_time_sec": float(
                sum(float(h["aggregation_time_sec"]) for h in history)
            ),
            "mean_round_local_train_time_sec": float(
                np.mean([float(h["local_train_time_sec"]) for h in history])
            ),
            "mean_round_aggregation_time_sec": float(
                np.mean([float(h["aggregation_time_sec"]) for h in history])
            ),
            "total_comm_bytes": int(self._cumulative_comm_bytes),
            "bytes_per_round": int(final_round.get("total_comm_bytes", 0)),
            "round_to_95pct_f1": int(round_to_95pct_f1),
            "final_metrics": final_round,
            "history": history,
        }

"""Robust aggregation defenses and baselines (`Step 04`)."""

from __future__ import annotations

from typing import Any

from trustfed.aggregation.clustering import CosineClusteringAggregator
from trustfed.aggregation.secure import SecureAggregator
from trustfed.aggregation.static_weight import StaticWeightAggregator
from trustfed.aggregation.trimmed_mean import MedianAggregator, TrimmedMeanAggregator
from trustfed.fl.aggregator import AggregationResult, Aggregator, ClientUpdate, FedAvgAggregator


def build_aggregator(name: str, **kwargs: Any) -> Aggregator:
    """Instantiate an `Aggregator` by name."""
    norm = str(name).strip().lower()
    if norm in ("fedavg", "fed_avg", "mean"):
        return FedAvgAggregator()
    if norm in ("static_weight", "static", "static_weights"):
        return StaticWeightAggregator(**kwargs)
    if norm in ("trimmed_mean", "trimmed_mean_b0.1", "trimmed_b0.1"):
        beta = float(kwargs.pop("beta", 0.1))
        return TrimmedMeanAggregator(beta=beta)
    if norm in ("trimmed_mean_b0.2", "trimmed_b0.2"):
        beta = float(kwargs.pop("beta", 0.2))
        return TrimmedMeanAggregator(beta=beta)
    if norm in ("trimmed_mean_b0", "trimmed_mean_b0.0"):
        return TrimmedMeanAggregator(beta=0.0)
    if norm in ("median", "coord_median", "coordinate_median"):
        return MedianAggregator()
    if norm in ("clustering", "cosine_clustering", "trustfed_robust"):
        return CosineClusteringAggregator(**kwargs)
    if norm in ("secure", "secure_fedavg", "verifiable"):
        return SecureAggregator(**kwargs)
    raise ValueError(
        f"Unknown aggregator '{name}'. Choose from: 'fedavg', 'static_weight', "
        "'trimmed_mean_b0.1', 'trimmed_mean_b0.2', 'median', 'clustering', 'secure_fedavg'."
    )


__all__ = [
    "AggregationResult",
    "Aggregator",
    "ClientUpdate",
    "CosineClusteringAggregator",
    "FedAvgAggregator",
    "MedianAggregator",
    "SecureAggregator",
    "StaticWeightAggregator",
    "TrimmedMeanAggregator",
    "build_aggregator",
]

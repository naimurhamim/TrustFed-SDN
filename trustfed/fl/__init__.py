"""Federated Learning simulation core and pluggable aggregators (`Step 03`)."""

from trustfed.fl.aggregator import (
    AggregationResult,
    Aggregator,
    ClientUpdate,
    FedAvgAggregator,
    deserialize_tensor_payload,
    serialize_tensor_payload,
)
from trustfed.fl.client import FLClient
from trustfed.fl.server import FLServer

__all__ = [
    "AggregationResult",
    "Aggregator",
    "ClientUpdate",
    "FLClient",
    "FLServer",
    "FedAvgAggregator",
    "deserialize_tensor_payload",
    "serialize_tensor_payload",
]

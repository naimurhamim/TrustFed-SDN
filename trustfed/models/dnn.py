"""Multi-Layer Perceptron (DNN) classifier for tabular SDN flow features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from trustfed.models.base import BaseFlowClassifier


class DNNClassifier(BaseFlowClassifier):
    """Feed-forward Deep Neural Network (MLP) classifier with LayerNorm and Dropout."""

    model_type = "dnn"

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if hidden_dims is not None:
            dims = [int(d) for d in hidden_dims]
        else:
            dims = [max(32, int(hidden_dim // (2**i))) for i in range(max(1, num_layers))]
        self.hidden_dims = dims

        layers: list[nn.Module] = []
        in_features = self.input_dim
        for h_dim in dims:
            layers.append(nn.Linear(in_features, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = h_dim
        layers.append(nn.Linear(in_features, self.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim != 2:
            raise ValueError(f"DNNClassifier expects 2-D input (B, D), got {tuple(x.shape)}")
        return self.net(x)

"""Gated Recurrent Unit (GRU) classifier for SDN flow feature vectors."""

from __future__ import annotations

import torch
import torch.nn as nn

from trustfed.models.base import (
    BaseFlowClassifier,
    SequenceMode,
    compute_sequence_Dimensions,
    reshape_tabular_to_sequence,
)


class GRUClassifier(BaseFlowClassifier):
    """GRU classifier accepting `(batch_size, input_dim)` tabular SDN flow vectors.

    Supports three sequence representations (`seq_mode`):
    - `"feature_chunks"` (default): Reshapes `D` features into `num_chunks` steps
      of `ceil(D / num_chunks)` features so recurrent gates capture cross-feature
      interactions across steps while keeping sequence length compact (`T=4`).
    - `"single_step"`: Sequence of length `T=1`, width `D`.
    - `"per_feature"`: Sequence of length `T=D`, width `1`.
    """

    model_type = "gru"

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_mode: SequenceMode = "feature_chunks",
        num_chunks: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super().__init__(input_dim=input_dim, num_classes=num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.seq_mode: SequenceMode = seq_mode
        self.num_chunks = int(num_chunks)
        self.bidirectional = bool(bidirectional)

        seq_len, step_width, pad_len = compute_sequence_Dimensions(
            input_dim, seq_mode=seq_mode, num_chunks=num_chunks
        )
        self.seq_len = seq_len
        self.step_width = step_width
        self.pad_len = pad_len

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=step_width,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        head_mid = max(32, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(out_dim, head_mid),
            nn.GELU(),
            nn.Linear(head_mid, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = reshape_tabular_to_sequence(x, seq_mode=self.seq_mode, num_chunks=self.num_chunks)
        out, _ = self.gru(seq)
        # Pool mean + final step representation for richer sequence summary when T > 1
        if out.shape[1] > 1:
            feat = 0.5 * (out[:, -1, :] + out.mean(dim=1))
        else:
            feat = out[:, -1, :]
        return self.head(feat)

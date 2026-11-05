"""Long Short-Term Memory (LSTM) classifier for SDN flow feature vectors."""

from __future__ import annotations

import torch
import torch.nn as nn

from trustfed.models.base import (
    BaseFlowClassifier,
    SequenceMode,
    compute_sequence_Dimensions,
    reshape_tabular_to_sequence,
)


class LSTMClassifier(BaseFlowClassifier):
    """LSTM classifier with identical input/output and head architecture to `GRUClassifier`."""

    model_type = "lstm"

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

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=step_width,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
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
        out, _ = self.lstm(seq)
        if out.shape[1] > 1:
            feat = 0.5 * (out[:, -1, :] + out.mean(dim=1))
        else:
            feat = out[:, -1, :]
        return self.head(feat)

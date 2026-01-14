from typing import Tuple

import torch
from torch import nn


class LSTMNet(nn.Module):
    """Simple LSTM encoder over time series.

    Input:  x of shape [B, T, C]
    Output: z of shape [B, D]
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True, bidirectional=False)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        h = h_n[-1]  # [B, H]
        z = self.proj(h)
        return z


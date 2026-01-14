from typing import List

import torch
from torch import nn


class SPP1D(nn.Module):
    """Spatial Pyramid Pooling over temporal axis for 1D sequences.
    Applies adaptive max pooling with different bin sizes and concatenates the outputs.
    """

    def __init__(self, bins: List[int]):
        super().__init__()
        self.bins = bins
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool1d(b) for b in bins])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        outs = [p(x) for p in self.pools]  # list of [B, C, b]
        outs = [o.flatten(start_dim=1) for o in outs]  # [B, C*b]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


class SPPNet(nn.Module):
    """1D CNN + SPP encoder for time series.

    Input: x [B, T, C]
    Output: z [B, D]
    """

    def __init__(self, in_dim: int = 2, channels: int = 64, hidden_dim: int = 64, bins: List[int] = [1, 2, 4, 8], dropout: float = 0.1):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.spp = SPP1D(bins=bins)
        spp_out_dim = channels * sum(bins)
        self.proj = nn.Sequential(
            nn.Linear(spp_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        feat = self.feature(x)
        pooled = self.spp(feat)
        z = self.proj(pooled)
        return z


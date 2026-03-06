"""
Ablation study for temporal branch.

This module provides different variants to isolate the effect of SPP:
1. ParallelTemporalBranch: LSTM + SPP (baseline)
2. LSTMOnlyBranch: LSTM only (removes SPP)
3. MultiScaleTemporal: Multi-scale LSTM (replaces SPP with multi-scale processing)
"""

import torch
import torch.nn as nn
from typing import List
import config


class LSTMOnlyBranch(nn.Module):
    """
    LSTM-only temporal branch (removes SPP from baseline).

    This is identical to ParallelTemporalBranch but without the SPP branch,
    allowing us to isolate SPP's contribution.
    """

    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = config.LSTM_HIDDEN_SIZE,
                 num_layers: int = config.LSTM_LAYERS,
                 dropout: float = config.LSTM_DROPOUT,
                 output_size: int = 256):
        """
        Initialize LSTM-only branch

        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Output feature size
        """
        super(LSTMOnlyBranch, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Shared LSTM for both years
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Projection layers for each year
        self.proj_2021 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.proj_2024 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_2021, x_2024):
        """
        Forward pass for both years

        Args:
            x_2021: Input tensor for 2021 (batch_size, 168, 1)
            x_2024: Input tensor for 2024 (batch_size, 168, 1)

        Returns:
            features: Stacked features (batch_size, 3, output_size)
                     [lstm_2021, lstm_2024, diff_lstm]
        """
        # Process 2021
        lstm_out_2021, (h_n_2021, _) = self.lstm(x_2021)
        h_2021 = h_n_2021[-1]  # (batch_size, hidden_size)
        lstm_2021 = self.proj_2021(h_2021)

        # Process 2024
        lstm_out_2024, (h_n_2024, _) = self.lstm(x_2024)
        h_2024 = h_n_2024[-1]
        lstm_2024 = self.proj_2024(h_2024)

        # Compute difference
        diff_lstm = lstm_2024 - lstm_2021

        # Stack features: (batch, 3, output_size)
        features = torch.stack([lstm_2021, lstm_2024, diff_lstm], dim=1)

        return features


class TemporalBranchWithMultiScale(nn.Module):
    """
    Multi-scale temporal branch (replaces SPP with multi-scale LSTM).

    This keeps the baseline structure but replaces SPP with multi-scale processing.
    """

    def __init__(self,
                 input_size: int = 1,
                 hidden_size: int = config.LSTM_HIDDEN_SIZE,
                 num_layers: int = config.LSTM_LAYERS,
                 dropout: float = config.LSTM_DROPOUT,
                 output_size: int = 256):
        super(TemporalBranchWithMultiScale, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Standard LSTM for baseline comparison
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-scale LSTM: hourly (full) + daily (aggregated)
        self.lstm_daily = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Weekly statistics network
        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 3, 64),  # mean, std, trend
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

        # Projection layers
        self.proj_lstm = nn.Linear(hidden_size, output_size)
        self.proj_daily = nn.Linear(hidden_size, output_size)

    def forward(self, x_2021, x_2024):
        """
        Forward pass with multi-scale processing

        Args:
            x_2021: Input tensor for 2021 (batch_size, 168, 1)
            x_2024: Input tensor for 2024 (batch_size, 168, 1)

        Returns:
            features: Stacked features (batch_size, 6, output_size)
        """
        all_features = []

        for x in [x_2021, x_2024]:
            # Scale 1: Hourly (standard LSTM)
            _, (h_n, _) = self.lstm(x)
            h_hourly = self.proj_lstm(h_n[-1])

            # Scale 2: Daily (aggregate 168h → 7d)
            batch_size = x.size(0)
            x_daily = x.view(batch_size, 7, 24, self.input_size).mean(dim=2)
            _, (h_daily_n, _) = self.lstm_daily(x_daily)
            h_daily = self.proj_daily(h_daily_n[-1])

            # Scale 3: Weekly statistics
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            trend = (x[:, -1, :] - x[:, 0, :]) / 168
            stats = torch.cat([mean, std, trend], dim=1)
            h_weekly = self.weekly_net(stats)

            # Stack multi-scale features for this year
            multi_scale = torch.stack([h_hourly, h_daily, h_weekly], dim=1)
            all_features.append(multi_scale)

        # Compute difference
        diff = all_features[1] - all_features[0]

        # Concatenate: 2 years × 3 scales + 1 difference = 7 features
        combined = torch.cat([all_features[0], all_features[1], diff], dim=1)

        return combined


# Test the modules
if __name__ == "__main__":
    print("Testing Temporal Branch Ablation Variants:")

    batch_size = 8
    timesteps = 168
    input_size = 1

    x_2021 = torch.randn(batch_size, timesteps, input_size)
    x_2024 = torch.randn(batch_size, timesteps, input_size)

    # Test LSTM-only
    print("\n1. LSTMOnlyBranch:")
    model1 = LSTMOnlyBranch(input_size=input_size)
    output1 = model1(x_2021, x_2024)
    print(f"   Output shape: {output1.shape}")
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"   Parameters: {params1:,}")

    # Test Multi-scale
    print("\n2. TemporalBranchWithMultiScale:")
    model2 = TemporalBranchWithMultiScale(input_size=input_size)
    output2 = model2(x_2021, x_2024)
    print(f"   Output shape: {output2.shape}")
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"   Parameters: {params2:,}")

    print(f"\nParameter difference: {params2 - params1:,} ({(params2/params1 - 1)*100:.1f}%)")

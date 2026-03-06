"""
Multi-scale temporal branch for processing time series at multiple granularities.

Captures patterns at three scales:
1. Hourly (168 timesteps): Fine-grained patterns
2. Daily (7 timesteps): Day-level trends
3. Weekly (statistical): Overall statistics (mean, std, trend)

This allows the model to learn both short-term fluctuations and long-term trends.
"""

import torch
import torch.nn as nn


class MultiScaleTemporalBranch(nn.Module):
    """
    Multi-scale temporal processing branch.

    Processes time series at hourly, daily, and weekly scales simultaneously,
    then fuses the multi-scale representations.

    Args:
        input_size: Number of features per timestep (default: 1)
        hidden_size: Hidden dimension for projections (default: 256)
        lstm_hidden: Hidden dimension for LSTM layers (default: 128)
        lstm_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.4)
    """

    def __init__(self, input_size=1, hidden_size=256, lstm_hidden=128, lstm_layers=2, dropout=0.4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden

        # Scale 1: Hourly LSTM (processes all 168 timesteps)
        self.lstm_hourly = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.proj_hourly = nn.Linear(lstm_hidden, hidden_size)

        # Scale 2: Daily LSTM (processes 7 daily aggregations)
        self.lstm_daily = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.proj_daily = nn.Linear(lstm_hidden, hidden_size)

        # Scale 3: Weekly statistical features
        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 3, 64),  # 3 stats per feature
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        # Fusion layer for multi-scale features
        self.scale_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.dropout = nn.Dropout(dropout)

    def extract_hourly_features(self, x):
        """
        Extract hourly-scale features using LSTM.

        Args:
            x: (batch, 168, input_size)

        Returns:
            h: (batch, hidden_size)
        """
        # LSTM processing
        lstm_out, (h_n, _) = self.lstm_hourly(x)  # h_n: (layers, batch, hidden)

        # Use final hidden state
        h = h_n[-1]  # (batch, lstm_hidden)

        # Project to hidden_size
        h = self.proj_hourly(h)
        h = self.dropout(h)

        return h  # (batch, hidden_size)

    def extract_daily_features(self, x):
        """
        Extract daily-scale features using LSTM.

        Args:
            x: (batch, 168, input_size)

        Returns:
            h: (batch, hidden_size)
        """
        batch_size = x.size(0)

        # Reshape to daily: (batch, 168, input_size) -> (batch, 7, 24, input_size)
        # Then aggregate by averaging over hours
        x_daily = x.view(batch_size, 7, 24, self.input_size).mean(dim=2)
        # x_daily: (batch, 7, input_size)

        # LSTM processing
        lstm_out, (h_n, _) = self.lstm_daily(x_daily)

        # Use final hidden state
        h = h_n[-1]  # (batch, lstm_hidden)

        # Project to hidden_size
        h = self.proj_daily(h)
        h = self.dropout(h)

        return h  # (batch, hidden_size)

    def extract_weekly_features(self, x):
        """
        Extract weekly-scale statistical features.

        Args:
            x: (batch, 168, input_size)

        Returns:
            h: (batch, hidden_size)
        """
        # Compute statistics over time dimension
        mean = x.mean(dim=1)  # (batch, input_size)
        std = x.std(dim=1)    # (batch, input_size)

        # Compute trend (linear slope over time)
        batch_size = x.size(0)
        time_points = torch.arange(168, dtype=x.dtype, device=x.device).view(1, 168, 1)
        time_points = time_points.expand(batch_size, 168, self.input_size)

        # Simple trend: last value - first value
        trend = (x[:, -1, :] - x[:, 0, :]) / 168  # (batch, input_size)

        # Concatenate statistics
        stats = torch.cat([mean, std, trend], dim=1)  # (batch, input_size * 3)

        # Process through MLP
        h = self.weekly_net(stats)
        h = self.dropout(h)

        return h  # (batch, hidden_size)

    def forward(self, x_2021, x_2024):
        """
        Forward pass with multi-scale processing.

        Args:
            x_2021: (batch, 168, input_size)
            x_2024: (batch, 168, input_size)

        Returns:
            fused: (batch, hidden_size)
        """
        batch_size = x_2021.size(0)

        # Process each year at multiple scales
        all_features = []

        for x in [x_2021, x_2024]:
            # Extract multi-scale features
            h_hourly = self.extract_hourly_features(x)    # (batch, hidden_size)
            h_daily = self.extract_daily_features(x)      # (batch, hidden_size)
            h_weekly = self.extract_weekly_features(x)    # (batch, hidden_size)

            # Stack multi-scale features
            multi_scale = torch.stack([h_hourly, h_daily, h_weekly], dim=1)
            # multi_scale: (batch, 3, hidden_size)

            all_features.append(multi_scale)

        # Compute difference between years
        diff = all_features[1] - all_features[0]  # (batch, 3, hidden_size)

        # Concatenate all features: 2 years × 3 scales + 1 difference = 7 features
        combined = torch.cat([all_features[0], all_features[1], diff], dim=1)
        # combined: (batch, 7, hidden_size)

        # Fuse multi-scale features using attention
        fused, _ = self.scale_fusion(combined, combined, combined)
        # fused: (batch, 7, hidden_size)

        # Aggregate across scales (mean pooling)
        fused = fused.mean(dim=1)  # (batch, hidden_size)

        # Final projection
        output = self.output_proj(fused)  # (batch, hidden_size)

        return output


class SimplifiedMultiScaleTemporal(nn.Module):
    """
    Simplified version with reduced parameters for faster training.

    Uses:
    - Separate LSTMs for hourly and daily scales (FIXED: no longer sharing LSTM)
    - Simpler weekly aggregation
    - Fewer parameters
    """

    def __init__(self, input_size=1, hidden_size=256, lstm_hidden=128, dropout=0.4):
        super().__init__()

        # FIX: Separate LSTMs for different scales to avoid sequence length mismatch
        self.lstm_hourly = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.lstm_daily = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Projections
        self.proj_hourly = nn.Linear(lstm_hidden, hidden_size)
        self.proj_daily = nn.Linear(lstm_hidden, hidden_size)

        # Weekly statistics
        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x_2021, x_2024):
        """
        Args:
            x_2021, x_2024: (batch, 168, input_size)

        Returns:
            output: (batch, 3, hidden_size) - [2021_features, 2024_features, diff]
        """
        # Process each year separately and fuse multi-scale features
        yearly_features = []

        for x in [x_2021, x_2024]:
            # Process hourly with dedicated LSTM
            _, (h_hourly_n, _) = self.lstm_hourly(x)
            h_hourly = self.proj_hourly(h_hourly_n[-1])

            # Process daily (average over 24 hours) with dedicated LSTM
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).mean(dim=2)
            _, (h_daily_n, _) = self.lstm_daily(x_daily)
            h_daily = self.proj_daily(h_daily_n[-1])

            # Weekly stats
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            trend = (x[:, -1, :] - x[:, 0, :]) / 168
            stats = torch.cat([mean, std, trend], dim=1)
            h_weekly = self.weekly_net(stats)

            # Fuse multi-scale features for this year
            multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
            fused_year = self.fusion(multi_scale)
            yearly_features.append(fused_year)

        # Compute difference between years
        diff = yearly_features[1] - yearly_features[0]

        # Stack: [2021_fused, 2024_fused, diff] -> (batch, 3, hidden_size)
        output = torch.stack([yearly_features[0], yearly_features[1], diff], dim=1)

        return output


# Test the modules
if __name__ == "__main__":
    # Test with dummy data
    batch_size = 8
    timesteps = 168
    input_size = 1

    x_2021 = torch.randn(batch_size, timesteps, input_size)
    x_2024 = torch.randn(batch_size, timesteps, input_size)

    # Test full multi-scale model
    model = MultiScaleTemporalBranch(input_size=input_size, hidden_size=256)
    output = model(x_2021, x_2024)

    print(f"Multi-Scale Temporal Branch:")
    print(f"  Input shape: {x_2021.shape}")
    print(f"  Output shape: {output.shape}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Test simplified model
    model_simple = SimplifiedMultiScaleTemporal(input_size=input_size, hidden_size=256)
    output_simple = model_simple(x_2021, x_2024)

    print(f"\nSimplified Multi-Scale Temporal:")
    print(f"  Output shape: {output_simple.shape}")

    params_simple = sum(p.numel() for p in model_simple.parameters())
    print(f"  Parameters: {params_simple:,}")
    print(f"  Reduction: {(1 - params_simple/params) * 100:.1f}%")

"""
Multi-scale temporal branch for processing time series at multiple granularities.

Captures patterns at three scales:
1. Hourly (168 timesteps): Fine-grained patterns
2. Daily (7 timesteps): Day-level trends
3. Weekly (statistical): Overall statistics (mean, std, trend)

This allows the model to learn both short-term fluctuations and long-term trends.
"""

import math
import torch
import torch.nn as nn


def _aggregate_daily_wamd(x: torch.Tensor) -> torch.Tensor:
    """
    Aggregate hourly [total_h, wamd_h] to daily [total_d, wamd_d].

    total_d = sum of hourly totals over 24h
    wamd_d  = flow-weighted average of hourly wamd values
              = Σ(total_h * wamd_h) / Σ(total_h)

    Args:
        x: (batch, 168, 2) where x[:,:,0]=total_h, x[:,:,1]=wamd_h
    Returns:
        (batch, 7, 2)
    """
    b = x.size(0)
    x_r = x.view(b, 7, 24, 2)
    total_h = x_r[:, :, :, 0]                                          # (b, 7, 24)
    wamd_h  = x_r[:, :, :, 1]                                          # (b, 7, 24)
    total_d = total_h.sum(dim=2)                                        # (b, 7)
    wamd_d  = (total_h * wamd_h).sum(dim=2) / total_d.clamp_min(1e-6)  # (b, 7)
    return torch.stack([total_d, wamd_d], dim=2)                        # (b, 7, 2)


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

    def __init__(self, input_size=1, hidden_size=256, lstm_hidden=128, lstm_layers=2, dropout=0.4, daily_agg_mode='sum'):
        super().__init__()
        self.daily_agg_mode = daily_agg_mode

        # FIX: Separate LSTMs for different scales to avoid sequence length mismatch
        self.lstm_hourly = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.lstm_daily = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Projections
        self.proj_hourly = nn.Linear(lstm_hidden, hidden_size)
        self.proj_daily = nn.Linear(lstm_hidden, hidden_size)

        # Weekly statistics
        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 4, 64),  # Changed to support 2-feature input: 2 * 4 = 8
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

            # Process daily (sum over 24 hours for true daily flow) with dedicated LSTM
            if self.daily_agg_mode == 'wamd':
                x_daily = _aggregate_daily_wamd(x)
            else:
                x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2)
            _, (h_daily_n, _) = self.lstm_daily(x_daily)
            h_daily = self.proj_daily(h_daily_n[-1])

            # Weekly stats (sum, max, mean for traffic characteristics)
            weekly_sum = x.sum(dim=1)      # Total weekly flow
            weekly_max = x.max(dim=1)[0]   # Peak flow
            weekly_mean = x.mean(dim=1)    # Average flow
            trend = (x[:, -1, :] - x[:, 0, :]) / 168
            stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
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

    def extract_features_single(self, x):
        """Extract fused multi-scale features from a single year's sequence."""
        _, (h_hourly_n, _) = self.lstm_hourly(x)
        h_hourly = self.proj_hourly(h_hourly_n[-1])

        if self.daily_agg_mode == 'wamd':
            x_daily = _aggregate_daily_wamd(x)
        else:
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2)
        _, (h_daily_n, _) = self.lstm_daily(x_daily)
        h_daily = self.proj_daily(h_daily_n[-1])

        weekly_sum = x.sum(dim=1)
        weekly_max = x.max(dim=1)[0]
        weekly_mean = x.mean(dim=1)
        trend = (x[:, -1, :] - x[:, 0, :]) / 168
        stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
        h_weekly = self.weekly_net(stats)

        multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
        return self.fusion(multi_scale)


class SimplifiedMultiScaleTemporalGRU(nn.Module):
    """GRU variant of the simplified multi-scale temporal branch."""

    def __init__(self, input_size=1, hidden_size=256, gru_hidden=128, gru_layers=2, dropout=0.4, daily_agg_mode='sum'):
        super().__init__()
        self.daily_agg_mode = daily_agg_mode

        self.gru_hourly = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.gru_daily = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        self.proj_hourly = nn.Linear(gru_hidden, hidden_size)
        self.proj_daily = nn.Linear(gru_hidden, hidden_size)

        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def extract_features_single(self, x):
        """Extract fused multi-scale features from a single year's sequence."""
        _, h_hourly_n = self.gru_hourly(x)
        h_hourly = self.proj_hourly(h_hourly_n[-1])

        if self.daily_agg_mode == 'wamd':
            x_daily = _aggregate_daily_wamd(x)
        else:
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2)
        _, h_daily_n = self.gru_daily(x_daily)
        h_daily = self.proj_daily(h_daily_n[-1])

        weekly_sum = x.sum(dim=1)
        weekly_max = x.max(dim=1)[0]
        weekly_mean = x.mean(dim=1)
        trend = (x[:, -1, :] - x[:, 0, :]) / 168
        stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
        h_weekly = self.weekly_net(stats)

        multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
        return self.fusion(multi_scale)

    def forward(self, x_2021, x_2024):
        yearly_features = [
            self.extract_features_single(x_2021),
            self.extract_features_single(x_2024)
        ]
        diff = yearly_features[1] - yearly_features[0]
        return torch.stack([yearly_features[0], yearly_features[1], diff], dim=1)


class _TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for fixed-length temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 168):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _TemporalConvBlock(nn.Module):
    """A lightweight residual TCN block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SimplifiedMultiScaleTemporalTCN(nn.Module):
    """TCN variant of the simplified multi-scale temporal branch."""

    def __init__(self, input_size=1, hidden_size=256, tcn_hidden=128, tcn_layers=3, dropout=0.4, daily_agg_mode='sum'):
        super().__init__()
        self.daily_agg_mode = daily_agg_mode

        self.hourly_in = nn.Sequential(
            nn.Conv1d(input_size, tcn_hidden, kernel_size=1),
            nn.ReLU(),
        )
        self.hourly_tcn = nn.Sequential(*[
            _TemporalConvBlock(tcn_hidden, kernel_size=3, dilation=2 ** i, dropout=dropout)
            for i in range(tcn_layers)
        ])

        self.daily_in = nn.Sequential(
            nn.Conv1d(input_size, tcn_hidden, kernel_size=1),
            nn.ReLU(),
        )
        self.daily_tcn = nn.Sequential(*[
            _TemporalConvBlock(tcn_hidden, kernel_size=3, dilation=1, dropout=dropout)
            for _ in range(max(1, tcn_layers - 1))
        ])

        self.proj_hourly = nn.Linear(tcn_hidden, hidden_size)
        self.proj_daily = nn.Linear(tcn_hidden, hidden_size)

        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def extract_features_single(self, x):
        x_h = x.transpose(1, 2)
        h_hourly = self.hourly_tcn(self.hourly_in(x_h)).mean(dim=2)
        h_hourly = self.proj_hourly(h_hourly)

        if self.daily_agg_mode == 'wamd':
            x_daily = _aggregate_daily_wamd(x)
        else:
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2).transpose(1, 2)
        if self.daily_agg_mode != 'wamd':
            pass  # already transposed above
        else:
            x_daily = x_daily.transpose(1, 2)
        h_daily = self.daily_tcn(self.daily_in(x_daily)).mean(dim=2)
        h_daily = self.proj_daily(h_daily)

        weekly_sum = x.sum(dim=1)
        weekly_max = x.max(dim=1)[0]
        weekly_mean = x.mean(dim=1)
        trend = (x[:, -1, :] - x[:, 0, :]) / 168
        stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
        h_weekly = self.weekly_net(stats)

        multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
        return self.fusion(multi_scale)

    def forward(self, x_2021, x_2024):
        yearly_features = [
            self.extract_features_single(x_2021),
            self.extract_features_single(x_2024)
        ]
        diff = yearly_features[1] - yearly_features[0]
        return torch.stack([yearly_features[0], yearly_features[1], diff], dim=1)


class SimplifiedMultiScaleTemporalTransformer(nn.Module):
    """Lightweight Transformer variant of the simplified multi-scale temporal branch."""

    def __init__(self, input_size=1, hidden_size=256, model_dim=128, num_layers=2, dropout=0.4, daily_agg_mode='sum'):
        super().__init__()
        self.daily_agg_mode = daily_agg_mode

        self.in_proj_hourly = nn.Linear(input_size, model_dim)
        self.in_proj_daily = nn.Linear(input_size, model_dim)
        self.pos_hourly = _TemporalPositionalEncoding(model_dim, max_len=168)
        self.pos_daily = _TemporalPositionalEncoding(model_dim, max_len=7)

        nhead = 8 if model_dim % 8 == 0 else 4
        enc_layer_hourly = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        enc_layer_daily = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.hourly_encoder = nn.TransformerEncoder(enc_layer_hourly, num_layers=num_layers)
        self.daily_encoder = nn.TransformerEncoder(enc_layer_daily, num_layers=max(1, num_layers - 1))

        self.proj_hourly = nn.Linear(model_dim, hidden_size)
        self.proj_daily = nn.Linear(model_dim, hidden_size)

        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def extract_features_single(self, x):
        x_hourly = self.pos_hourly(self.in_proj_hourly(x))
        h_hourly = self.hourly_encoder(x_hourly).mean(dim=1)
        h_hourly = self.proj_hourly(h_hourly)

        if self.daily_agg_mode == 'wamd':
            x_daily = _aggregate_daily_wamd(x)
        else:
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2)
        x_daily = self.pos_daily(self.in_proj_daily(x_daily))
        h_daily = self.daily_encoder(x_daily).mean(dim=1)
        h_daily = self.proj_daily(h_daily)

        weekly_sum = x.sum(dim=1)
        weekly_max = x.max(dim=1)[0]
        weekly_mean = x.mean(dim=1)
        trend = (x[:, -1, :] - x[:, 0, :]) / 168
        stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
        h_weekly = self.weekly_net(stats)

        multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
        return self.fusion(multi_scale)

    def forward(self, x_2021, x_2024):
        yearly_features = [
            self.extract_features_single(x_2021),
            self.extract_features_single(x_2024)
        ]
        diff = yearly_features[1] - yearly_features[0]
        return torch.stack([yearly_features[0], yearly_features[1], diff], dim=1)


class SimplifiedMultiScaleTemporalBiGRU(nn.Module):
    """Bidirectional GRU variant of the simplified multi-scale temporal branch."""

    def __init__(self, input_size=1, hidden_size=256, gru_hidden=128, gru_layers=2, dropout=0.4, daily_agg_mode='sum'):
        super().__init__()
        self.daily_agg_mode = daily_agg_mode

        self.gru_hourly = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )

        self.gru_daily = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=True
        )

        bi_hidden = gru_hidden * 2
        self.proj_hourly = nn.Linear(bi_hidden, hidden_size)
        self.proj_daily = nn.Linear(bi_hidden, hidden_size)

        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def _last_bidirectional(self, h_n):
        # h_n shape: (num_layers * 2, batch, hidden)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        return torch.cat([h_forward, h_backward], dim=1)

    def extract_features_single(self, x):
        _, h_hourly_n = self.gru_hourly(x)
        h_hourly = self.proj_hourly(self._last_bidirectional(h_hourly_n))

        if self.daily_agg_mode == 'wamd':
            x_daily = _aggregate_daily_wamd(x)
        else:
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2)
        _, h_daily_n = self.gru_daily(x_daily)
        h_daily = self.proj_daily(self._last_bidirectional(h_daily_n))

        weekly_sum = x.sum(dim=1)
        weekly_max = x.max(dim=1)[0]
        weekly_mean = x.mean(dim=1)
        trend = (x[:, -1, :] - x[:, 0, :]) / 168
        stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
        h_weekly = self.weekly_net(stats)

        multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
        return self.fusion(multi_scale)

    def forward(self, x_2021, x_2024):
        yearly_features = [
            self.extract_features_single(x_2021),
            self.extract_features_single(x_2024)
        ]
        diff = yearly_features[1] - yearly_features[0]
        return torch.stack([yearly_features[0], yearly_features[1], diff], dim=1)


class FullMultiScaleTemporalTransformer(nn.Module):
    """Full-capacity Transformer variant for temporal branch ablation."""

    def __init__(
        self,
        input_size=1,
        hidden_size=256,
        model_dim=256,
        hourly_layers=4,
        daily_layers=3,
        nhead=8,
        ff_multiplier=4,
        dropout=0.3,
        daily_agg_mode='sum',
    ):
        super().__init__()
        self.daily_agg_mode = daily_agg_mode

        if model_dim % nhead != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by nhead={nhead}")

        self.in_proj_hourly = nn.Linear(input_size, model_dim)
        self.in_proj_daily = nn.Linear(input_size, model_dim)

        self.cls_token_hourly = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.cls_token_daily = nn.Parameter(torch.zeros(1, 1, model_dim))

        self.pos_hourly = _TemporalPositionalEncoding(model_dim, max_len=169)  # 168 + CLS
        self.pos_daily = _TemporalPositionalEncoding(model_dim, max_len=8)      # 7 + CLS

        ff_dim = model_dim * ff_multiplier
        hourly_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        daily_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.hourly_encoder = nn.TransformerEncoder(hourly_layer, num_layers=hourly_layers)
        self.daily_encoder = nn.TransformerEncoder(daily_layer, num_layers=daily_layers)

        self.hourly_readout = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.daily_readout = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def _encode_with_cls(self, x, in_proj, pos_enc, cls_token, encoder):
        x = in_proj(x)
        batch_size = x.size(0)
        cls = cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = pos_enc(x)
        h = encoder(x)
        return h[:, 0, :]  # CLS token

    def extract_features_single(self, x):
        h_hourly = self._encode_with_cls(
            x,
            self.in_proj_hourly,
            self.pos_hourly,
            self.cls_token_hourly,
            self.hourly_encoder,
        )
        h_hourly = self.hourly_readout(h_hourly)

        if self.daily_agg_mode == 'wamd':
            x_daily = _aggregate_daily_wamd(x)
        else:
            x_daily = x.view(x.size(0), 7, 24, x.size(2)).sum(dim=2)
        h_daily = self._encode_with_cls(
            x_daily,
            self.in_proj_daily,
            self.pos_daily,
            self.cls_token_daily,
            self.daily_encoder,
        )
        h_daily = self.daily_readout(h_daily)

        weekly_sum = x.sum(dim=1)
        weekly_max = x.max(dim=1)[0]
        weekly_mean = x.mean(dim=1)
        trend = (x[:, -1, :] - x[:, 0, :]) / 168
        stats = torch.cat([weekly_sum, weekly_max, weekly_mean, trend], dim=1)
        h_weekly = self.weekly_net(stats)

        multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
        return self.fusion(multi_scale)

    def forward(self, x_2021, x_2024):
        yearly_features = [
            self.extract_features_single(x_2021),
            self.extract_features_single(x_2024)
        ]
        diff = yearly_features[1] - yearly_features[0]
        return torch.stack([yearly_features[0], yearly_features[1], diff], dim=1)


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

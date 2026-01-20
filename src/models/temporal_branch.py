"""
Temporal branch: LSTM-Net + SPP-Net for time series processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import config


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer"""

    def __init__(self, levels: List[int] = [1, 2, 4]):
        """
        Initialize SPP layer

        Args:
            levels: Pyramid levels (e.g., [1, 2, 4] for 1x1, 2x2, 4x4 pooling)
        """
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, channels, length)

        Returns:
            Pooled features (batch_size, channels * sum(levels))
        """
        batch_size, channels, length = x.size()
        pooled_features = []

        for level in self.levels:
            # CRITICAL FIX: Use adaptive pooling to avoid dimension mismatch issues
            # This ensures output size is exactly 'level' regardless of input length
            pooled = F.adaptive_max_pool1d(x, output_size=level)

            # Flatten and append - use reshape instead of view for non-contiguous tensors
            pooled = pooled.reshape(batch_size, -1)
            pooled_features.append(pooled)

        # Concatenate all levels
        output = torch.cat(pooled_features, dim=1)

        return output


class LSTMSPPNet(nn.Module):
    """LSTM network with Spatial Pyramid Pooling"""

    def __init__(self,
                 input_size: int = 2,  # inflow + outflow
                 hidden_size: int = config.LSTM_HIDDEN_SIZE,
                 num_layers: int = config.LSTM_LAYERS,
                 dropout: float = config.LSTM_DROPOUT,
                 spp_levels: List[int] = config.SPP_LEVELS):
        """
        Initialize LSTM-SPP network

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            spp_levels: SPP pyramid levels
        """
        super(LSTMSPPNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(levels=spp_levels)

        # Calculate SPP output size
        self.spp_output_size = hidden_size * sum(spp_levels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            Temporal features (batch_size, spp_output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Reshape for SPP: (batch_size, hidden_size, seq_len)
        lstm_out = lstm_out.transpose(1, 2)

        # Apply SPP
        spp_out = self.spp(lstm_out)

        # Apply dropout
        output = self.dropout(spp_out)

        return output


class TemporalBranch(nn.Module):
    """Complete temporal branch with LSTM-SPP"""

    def __init__(self,
                 input_size: int = 2,
                 hidden_size: int = config.LSTM_HIDDEN_SIZE,
                 num_layers: int = config.LSTM_LAYERS,
                 dropout: float = config.LSTM_DROPOUT,
                 spp_levels: List[int] = config.SPP_LEVELS,
                 output_size: int = 256):
        """
        Initialize temporal branch

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            spp_levels: SPP pyramid levels
            output_size: Output feature size
        """
        super(TemporalBranch, self).__init__()

        # LSTM-SPP network
        self.lstm_spp = LSTMSPPNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            spp_levels=spp_levels
        )

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.lstm_spp.spp_output_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, seq_len, input_size)

        Returns:
            Temporal features (batch_size, output_size)
        """
        # Extract temporal features
        temporal_features = self.lstm_spp(x)

        # Project to output size
        output = self.projection(temporal_features)

        return output


class ParallelLSTMBranch(nn.Module):
    """Parallel LSTM branch for dual-year processing"""

    def __init__(self,
                 input_size: int = 2,  # [total_log, net_flow_log]
                 hidden_size: int = config.LSTM_HIDDEN_SIZE,
                 num_layers: int = config.LSTM_LAYERS,
                 dropout: float = config.LSTM_DROPOUT,
                 output_size: int = 256):
        """
        Initialize parallel LSTM branch

        Args:
            input_size: Number of input features per year
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Output feature size
        """
        super(ParallelLSTMBranch, self).__init__()

        # Shared LSTM (processes both years with same weights)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Projection layer
        self.projection = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_2021, x_2024):
        """
        Forward pass for both years

        Args:
            x_2021: Input tensor for 2021 (batch_size, 7, 2)
            x_2024: Input tensor for 2024 (batch_size, 7, 2)

        Returns:
            h_2021: Features for 2021 (batch_size, output_size)
            h_2024: Features for 2024 (batch_size, output_size)
            diff: Difference features (batch_size, output_size)
        """
        # Process 2021
        _, (h_2021, _) = self.lstm(x_2021)  # h_2021: (num_layers, batch, hidden_size)
        h_2021 = h_2021[-1]  # Take last layer: (batch, hidden_size)
        h_2021 = self.projection(h_2021)  # (batch, output_size)
        h_2021 = self.dropout(h_2021)

        # Process 2024
        _, (h_2024, _) = self.lstm(x_2024)
        h_2024 = h_2024[-1]
        h_2024 = self.projection(h_2024)
        h_2024 = self.dropout(h_2024)

        # Compute difference (change pattern)
        diff = h_2024 - h_2021

        return h_2021, h_2024, diff


class ParallelSPPBranch(nn.Module):
    """Parallel SPP branch for dual-year processing"""

    def __init__(self,
                 input_size: int = 2,  # [total_log, net_flow_log]
                 spp_levels: List[int] = config.SPP_LEVELS,
                 output_size: int = 256):
        """
        Initialize parallel SPP branch

        Args:
            input_size: Number of input features per year
            spp_levels: SPP pyramid levels
            output_size: Output feature size
        """
        super(ParallelSPPBranch, self).__init__()

        # SPP layer
        self.spp = SpatialPyramidPooling(levels=spp_levels)

        # Calculate SPP output size: input_size * sum(levels)
        # For input_size=2, levels=[1,2,4]: output = 2 * (1+2+4) = 14
        spp_output_size = input_size * sum(spp_levels)

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(spp_output_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x_2021, x_2024):
        """
        Forward pass for both years

        Args:
            x_2021: Input tensor for 2021 (batch_size, 7, 2)
            x_2024: Input tensor for 2024 (batch_size, 7, 2)

        Returns:
            h_2021: Features for 2021 (batch_size, output_size)
            h_2024: Features for 2024 (batch_size, output_size)
            diff: Difference features (batch_size, output_size)
        """
        # Transpose for SPP: (batch, features, time)
        x_2021_t = x_2021.transpose(1, 2)  # (batch, 2, 7)
        x_2024_t = x_2024.transpose(1, 2)  # (batch, 2, 7)

        # Apply SPP
        spp_2021 = self.spp(x_2021_t)  # (batch, 14)
        spp_2024 = self.spp(x_2024_t)  # (batch, 14)

        # Project to output size
        h_2021 = self.projection(spp_2021)  # (batch, output_size)
        h_2024 = self.projection(spp_2024)  # (batch, output_size)

        # Compute difference
        diff = h_2024 - h_2021

        return h_2021, h_2024, diff


class ParallelTemporalBranch(nn.Module):
    """Complete parallel temporal branch combining LSTM and SPP"""

    def __init__(self,
                 input_size: int = 2,
                 hidden_size: int = config.LSTM_HIDDEN_SIZE,
                 num_layers: int = config.LSTM_LAYERS,
                 dropout: float = config.LSTM_DROPOUT,
                 spp_levels: List[int] = config.SPP_LEVELS,
                 output_size: int = 256):
        """
        Initialize parallel temporal branch

        Args:
            input_size: Number of input features per year
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            spp_levels: SPP pyramid levels
            output_size: Output feature size
        """
        super(ParallelTemporalBranch, self).__init__()

        # LSTM branch
        self.lstm_branch = ParallelLSTMBranch(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )

        # SPP branch
        self.spp_branch = ParallelSPPBranch(
            input_size=input_size,
            spp_levels=spp_levels,
            output_size=output_size
        )

    def forward(self, x_2021, x_2024):
        """
        Forward pass for both years

        Args:
            x_2021: Input tensor for 2021 (batch_size, 7, 2)
            x_2024: Input tensor for 2024 (batch_size, 7, 2)

        Returns:
            features: Stacked features (batch_size, 6, output_size)
                     [lstm_2021, lstm_2024, diff_lstm, spp_2021, spp_2024, diff_spp]
        """
        # LSTM branch
        lstm_2021, lstm_2024, diff_lstm = self.lstm_branch(x_2021, x_2024)

        # SPP branch
        spp_2021, spp_2024, diff_spp = self.spp_branch(x_2021, x_2024)

        # Stack all features: (batch, 6, output_size)
        features = torch.stack([
            lstm_2021, lstm_2024, diff_lstm,
            spp_2021, spp_2024, diff_spp
        ], dim=1)

        return features


if __name__ == "__main__":
    # Test temporal branch
    print("Testing Temporal Branch")

    batch_size = 8
    seq_len = 7  # 7 days (NEW)
    input_size = 2  # [total_log, net_flow_log]

    # Create dummy input
    x_2021 = torch.randn(batch_size, seq_len, input_size)
    x_2024 = torch.randn(batch_size, seq_len, input_size)

    # Test old model
    print("\n=== Old TemporalBranch ===")
    old_model = TemporalBranch()
    old_output = old_model(torch.randn(batch_size, 168, 2))
    print(f"Input shape: (batch, 168, 2)")
    print(f"Output shape: {old_output.shape}")

    # Test new parallel model
    print("\n=== New ParallelTemporalBranch ===")
    new_model = ParallelTemporalBranch()
    new_output = new_model(x_2021, x_2024)
    print(f"Input shape: (batch, 7, 2) x 2 years")
    print(f"Output shape: {new_output.shape}")
    print(f"Expected: (batch, 6, 256) - 6 features from LSTM+SPP")
    print(f"Model parameters: {sum(p.numel() for p in new_model.parameters())}")

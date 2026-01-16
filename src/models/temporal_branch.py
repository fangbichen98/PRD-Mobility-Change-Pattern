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
            # Calculate kernel size and stride for this level
            kernel_size = length // level
            stride = length // level

            # Apply max pooling
            pooled = F.max_pool1d(x, kernel_size=kernel_size, stride=stride)

            # Flatten and append
            pooled = pooled.view(batch_size, -1)
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


if __name__ == "__main__":
    # Test temporal branch
    print("Testing Temporal Branch")

    batch_size = 8
    seq_len = 168  # 7 days * 24 hours
    input_size = 2  # inflow + outflow

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_size)

    # Initialize model
    model = TemporalBranch()

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

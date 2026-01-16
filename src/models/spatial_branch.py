"""
Dynamic Graph branch: DySAT-Net for spatial relationship modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Tuple
import config


class StructuralAttentionLayer(nn.Module):
    """Structural attention layer for graph learning"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: int = 4,
                 dropout: float = 0.2):
        """
        Initialize structural attention layer

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super(StructuralAttentionLayer, self).__init__()

        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.output_size = out_channels * heads

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_dim)

        Returns:
            Updated node features (num_nodes, out_channels * heads)
        """
        return self.gat(x, edge_index, edge_attr)


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for dynamic graph learning"""

    def __init__(self, hidden_size: int, num_time_steps: int):
        """
        Initialize temporal attention layer

        Args:
            hidden_size: Hidden feature dimension
            num_time_steps: Number of time steps
        """
        super(TemporalAttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_time_steps = num_time_steps

        # Attention parameters
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Temporal node features (num_time_steps, num_nodes, hidden_size)

        Returns:
            Attended features (num_nodes, hidden_size)
        """
        num_time_steps, num_nodes, hidden_size = x.size()

        # Compute queries, keys, values
        Q = self.query(x)  # (num_time_steps, num_nodes, hidden_size)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        # Average over nodes to get time-level representations
        Q_time = Q.mean(dim=1)  # (num_time_steps, hidden_size)
        K_time = K.mean(dim=1)

        # Attention weights
        scores = torch.matmul(Q_time, K_time.transpose(0, 1))  # (num_time_steps, num_time_steps)
        scores = scores / (hidden_size ** 0.5)
        attention_weights = F.softmax(scores, dim=1)

        # Apply attention to values
        # Reshape for batch matrix multiplication
        V_reshaped = V.view(num_time_steps, -1)  # (num_time_steps, num_nodes * hidden_size)
        attended = torch.matmul(attention_weights, V_reshaped)  # (num_time_steps, num_nodes * hidden_size)

        # Take the last time step's attended features
        output = attended[-1].view(num_nodes, hidden_size)

        return output


class DySATNet(nn.Module):
    """Dynamic Self-Attention Network for temporal graphs"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = config.DYSAT_HIDDEN_SIZE,
                 num_layers: int = config.DYSAT_LAYERS,
                 heads: int = config.DYSAT_HEADS,
                 dropout: float = config.DYSAT_DROPOUT,
                 num_time_steps: int = None):
        """
        Initialize DySAT network

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden feature dimension
            num_layers: Number of structural attention layers
            heads: Number of attention heads
            dropout: Dropout rate
            num_time_steps: Number of time steps for temporal attention
        """
        super(DySATNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_time_steps = num_time_steps

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Structural attention layers
        self.structural_layers = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_size if i == 0 else hidden_size * heads
            out_channels = hidden_size

            self.structural_layers.append(
                StructuralAttentionLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads,
                    dropout=dropout
                )
            )

        # Temporal attention layer
        if num_time_steps is not None:
            self.temporal_attention = TemporalAttentionLayer(
                hidden_size=hidden_size * heads,
                num_time_steps=num_time_steps
            )
        else:
            self.temporal_attention = None

        self.output_size = hidden_size * heads

    def forward(self, x, edge_index, edge_attr=None, batch_time_steps=None):
        """
        Forward pass

        Args:
            x: Node features (num_nodes, input_size) or (num_nodes, time_steps, input_size)
            edge_index: Edge indices (2, num_edges) or list of edge indices for each time step
            edge_attr: Edge attributes
            batch_time_steps: Number of time steps in batch (for temporal attention)

        Returns:
            Node embeddings (num_nodes, output_size)
        """
        # Check if input is temporal (3D)
        is_temporal = len(x.shape) == 3

        if is_temporal:
            # Input is (num_nodes, time_steps, input_size)
            # Transpose to (time_steps, num_nodes, input_size) for processing
            num_nodes, num_time_steps, input_size = x.size()
            x = x.transpose(0, 1)  # Now (time_steps, num_nodes, input_size)

            temporal_embeddings = []

            for t in range(num_time_steps):
                x_t = x[t]  # (num_nodes, input_size)

                # Get edge index for this time step
                if isinstance(edge_index, list):
                    edge_index_t = edge_index[t]
                else:
                    edge_index_t = edge_index

                # Project input
                h = self.input_proj(x_t)

                # Apply structural attention layers
                for layer in self.structural_layers:
                    h = layer(h, edge_index_t, edge_attr)
                    h = F.elu(h)

                temporal_embeddings.append(h)

            # Stack temporal embeddings
            temporal_embeddings = torch.stack(temporal_embeddings, dim=0)  # (num_time_steps, num_nodes, hidden_size)

            # Apply temporal attention
            if self.temporal_attention is not None:
                output = self.temporal_attention(temporal_embeddings)
            else:
                # Simple average over time
                output = temporal_embeddings.mean(dim=0)

        else:
            # Static graph processing
            # Project input
            h = self.input_proj(x)

            # Apply structural attention layers
            for layer in self.structural_layers:
                h = layer(h, edge_index, edge_attr)
                h = F.elu(h)

            output = h

        return output


class SpatialBranch(nn.Module):
    """Complete spatial branch with DySAT"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = config.DYSAT_HIDDEN_SIZE,
                 num_layers: int = config.DYSAT_LAYERS,
                 heads: int = config.DYSAT_HEADS,
                 dropout: float = config.DYSAT_DROPOUT,
                 num_time_steps: int = None,
                 output_size: int = 256):
        """
        Initialize spatial branch

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden feature dimension
            num_layers: Number of structural attention layers
            heads: Number of attention heads
            dropout: Dropout rate
            num_time_steps: Number of time steps
            output_size: Output feature size
        """
        super(SpatialBranch, self).__init__()

        # DySAT network
        self.dysat = DySATNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            num_time_steps=num_time_steps
        )

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.dysat.output_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass

        Args:
            x: Node features (num_nodes, time_steps, features) for temporal DySAT
            edge_index: Edge indices
            edge_attr: Edge attributes

        Returns:
            Spatial features (num_nodes, output_size)
        """
        # Extract spatial features using DySAT with temporal attention
        # x should be (num_nodes, time_steps, features) for temporal processing
        spatial_features = self.dysat(x, edge_index, edge_attr)

        # Project to output size
        output = self.projection(spatial_features)

        return output


if __name__ == "__main__":
    # Test spatial branch
    print("Testing Spatial Branch")

    num_nodes = 100
    input_size = 168 * 2  # 168 hours * 2 features (inflow + outflow)
    num_edges = 500

    # Create dummy input
    x = torch.randn(num_nodes, input_size)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Initialize model
    model = SpatialBranch(input_size=input_size)

    # Forward pass
    output = model(x, edge_index)

    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

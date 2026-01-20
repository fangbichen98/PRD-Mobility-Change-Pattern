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

    def forward(self, x, edge_index, edge_attr=None, batch_time_steps=None, graphs=None):
        """
        Forward pass with support for dynamic graphs

        Args:
            x: Node features (num_nodes, input_size) or (num_nodes, time_steps, input_size)
            edge_index: Edge indices (2, num_edges) or list of edge indices for each time step
            edge_attr: Edge attributes
            batch_time_steps: Number of time steps in batch (for temporal attention)
            graphs: List of (edge_index, edge_attr) tuples for dynamic graphs (NEW)

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
                # NEW: Support dynamic graphs via graphs parameter
                if graphs is not None:
                    edge_index_t, edge_attr_t = graphs[t]
                elif isinstance(edge_index, list):
                    edge_index_t = edge_index[t]
                    edge_attr_t = edge_attr[t] if isinstance(edge_attr, list) else edge_attr
                else:
                    edge_index_t = edge_index
                    edge_attr_t = edge_attr

                # Project input
                h = self.input_proj(x_t)

                # Apply structural attention layers
                for layer in self.structural_layers:
                    h = layer(h, edge_index_t, edge_attr_t)
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


class DualYearDySAT(nn.Module):
    """Dual-year DySAT wrapper for parallel processing of 2021 and 2024"""

    def __init__(self,
                 input_size: int = 2,  # [total_log, net_flow_log]
                 hidden_size: int = config.DYSAT_HIDDEN_SIZE,
                 num_layers: int = config.DYSAT_LAYERS,
                 heads: int = config.DYSAT_HEADS,
                 dropout: float = config.DYSAT_DROPOUT,
                 num_time_steps: int = 7,  # 7 daily snapshots
                 output_size: int = 256):
        """
        Initialize dual-year DySAT

        Args:
            input_size: Input feature dimension per year
            hidden_size: Hidden feature dimension
            num_layers: Number of structural attention layers
            heads: Number of attention heads
            dropout: Dropout rate
            num_time_steps: Number of time steps (7 days)
            output_size: Output feature size
        """
        super(DualYearDySAT, self).__init__()

        # Shared DySAT network (processes both years with same weights)
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
            nn.Dropout(dropout)
        )

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, node_indices=None):
        """
        Forward pass for both years

        Args:
            x_2021: Node features for 2021 (num_nodes, 7, 2)
            x_2024: Node features for 2024 (num_nodes, 7, 2)
            graphs_2021: List of 7 (edge_index, edge_attr) tuples for 2021
            graphs_2024: List of 7 (edge_index, edge_attr) tuples for 2024
            node_indices: Optional node indices for subgraph extraction

        Returns:
            h_2021: Features for 2021 (num_nodes, output_size)
            h_2024: Features for 2024 (num_nodes, output_size)
            diff: Difference features (num_nodes, output_size)
        """
        # Process 2021 with dynamic graphs
        spatial_2021 = self.dysat(x_2021, edge_index=None, edge_attr=None, graphs=graphs_2021)
        h_2021 = self.projection(spatial_2021)

        # Process 2024 with dynamic graphs
        spatial_2024 = self.dysat(x_2024, edge_index=None, edge_attr=None, graphs=graphs_2024)
        h_2024 = self.projection(spatial_2024)

        # Compute difference (spatial change pattern)
        diff = h_2024 - h_2021

        # If node_indices provided, extract subgraph features
        if node_indices is not None:
            h_2021 = h_2021[node_indices]
            h_2024 = h_2024[node_indices]
            diff = diff[node_indices]

        return h_2021, h_2024, diff


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

    def forward(self, x, edge_index, edge_attr=None, graphs=None):
        """
        Forward pass

        Args:
            x: Node features (num_nodes, time_steps, features) for temporal DySAT
            edge_index: Edge indices
            edge_attr: Edge attributes
            graphs: List of (edge_index, edge_attr) tuples for dynamic graphs

        Returns:
            Spatial features (num_nodes, output_size)
        """
        # Extract spatial features using DySAT with temporal attention
        # x should be (num_nodes, time_steps, features) for temporal processing
        spatial_features = self.dysat(x, edge_index, edge_attr, graphs=graphs)

        # Project to output size
        output = self.projection(spatial_features)

        return output


if __name__ == "__main__":
    # Test spatial branch
    print("Testing Spatial Branch")

    num_nodes = 100
    num_time_steps = 7  # NEW: 7 daily snapshots
    input_size = 2  # [total_log, net_flow_log]
    num_edges = 500

    # Create dummy input for temporal processing
    x_2021 = torch.randn(num_nodes, num_time_steps, input_size)
    x_2024 = torch.randn(num_nodes, num_time_steps, input_size)

    # Create dummy dynamic graphs (7 snapshots per year)
    graphs_2021 = []
    graphs_2024 = []
    for t in range(num_time_steps):
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 1)
        graphs_2021.append((edge_index, edge_attr))
        graphs_2024.append((edge_index, edge_attr))

    # Test old model
    print("\n=== Old SpatialBranch ===")
    old_model = SpatialBranch(input_size=168*2, num_time_steps=None)
    old_x = torch.randn(num_nodes, 168*2)
    old_edge_index = torch.randint(0, num_nodes, (2, num_edges))
    old_output = old_model(old_x, old_edge_index)
    print(f"Input shape: (num_nodes, 168*2)")
    print(f"Output shape: {old_output.shape}")

    # Test new dual-year model
    print("\n=== New DualYearDySAT ===")
    new_model = DualYearDySAT(input_size=2, num_time_steps=7)
    h_2021, h_2024, diff = new_model(x_2021, x_2024, graphs_2021, graphs_2024)
    print(f"Input shape: (num_nodes, 7, 2) x 2 years")
    print(f"Output shapes:")
    print(f"  h_2021: {h_2021.shape}")
    print(f"  h_2024: {h_2024.shape}")
    print(f"  diff: {diff.shape}")
    print(f"Expected: (num_nodes, 256) for each")
    print(f"Model parameters: {sum(p.numel() for p in new_model.parameters())}")

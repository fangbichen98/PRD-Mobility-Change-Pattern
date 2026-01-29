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
"""
Simplified GAT-based spatial branch for static flow-only graphs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Tuple
import config


class SimplifiedDualYearGAT(nn.Module):
    """Simplified spatial branch using static flow-only graphs with GAT"""

    def __init__(self,
                 input_size: int = 2,  # [total_log, net_flow_log]
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.2,
                 output_size: int = 256):
        """
        Initialize simplified dual-year GAT

        Args:
            input_size: Input feature dimension per year (2 features)
            hidden_size: Hidden feature dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout rate
            output_size: Output feature size
        """
        super(SimplifiedDualYearGAT, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout
        self.output_size = output_size

        # GAT layers (process each day separately with static graph)
        # First layer: input_size -> hidden_size
        # Middle layers: hidden_size*heads -> hidden_size
        # Last layer: hidden_size*heads -> hidden_size
        self.gat_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer
                in_channels = input_size
            else:
                # Subsequent layers (concat heads from previous layer)
                in_channels = hidden_size * heads

            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=1  # Edge weights from flow graph
                )
            )

        # Output projection: (hidden_size * heads) -> output_size
        self.output_proj = nn.Linear(hidden_size * heads, output_size)
        self.dropout = nn.Dropout(dropout)

    def process_year(self, x, edge_index, edge_attr):
        """
        Process one year's data with static graph

        Args:
            x: Node features (num_nodes, 7, 2) - 7 days of flow features
            edge_index: Static graph edge indices (2, num_edges)
            edge_attr: Static graph edge weights (num_edges,)

        Returns:
            h: Aggregated node features (num_nodes, output_size)
        """
        num_nodes = x.size(0)
        num_days = x.size(1)  # 7 days

        # Process each day separately with the same static graph
        daily_embeddings = []

        for t in range(num_days):
            # Extract features for day t: (num_nodes, 2)
            x_t = x[:, t, :]

            h = x_t

            # Apply GAT layers
            for gat_layer in self.gat_layers:
                h = gat_layer(h, edge_index, edge_attr)
                h = F.elu(h)
                h = self.dropout(h)

            # h shape: (num_nodes, hidden_size * heads)
            daily_embeddings.append(h)

        # Stack daily embeddings: (7, num_nodes, hidden_size*heads)
        daily_embeddings = torch.stack(daily_embeddings, dim=0)

        # Temporal aggregation: average over 7 days
        # (7, num_nodes, hidden_size*heads) -> (num_nodes, hidden_size*heads)
        h_aggregated = daily_embeddings.mean(dim=0)

        # Project to output size: (num_nodes, hidden_size*heads) -> (num_nodes, output_size)
        h_out = self.output_proj(h_aggregated)

        return h_out

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, node_indices=None):
        """
        Forward pass for both years

        Args:
            x_2021: Node features for 2021 (num_nodes, 7, 2)
            x_2024: Node features for 2024 (num_nodes, 7, 2)
            graphs_2021: List with single (edge_index, edge_attr) tuple for 2021
            graphs_2024: List with single (edge_index, edge_attr) tuple for 2024
            node_indices: Optional node indices for batch extraction

        Returns:
            h_2021: Features for 2021 (batch_size, output_size)
            h_2024: Features for 2024 (batch_size, output_size)
            diff: Difference features (batch_size, output_size)
        """
        # Extract static graphs (single graph per year)
        edge_index_2021, edge_attr_2021 = graphs_2021[0]
        edge_index_2024, edge_attr_2024 = graphs_2024[0]

        # Ensure edge_attr is 2D: (num_edges,) -> (num_edges, 1)
        if edge_attr_2021.dim() == 1:
            edge_attr_2021 = edge_attr_2021.unsqueeze(-1)
        if edge_attr_2024.dim() == 1:
            edge_attr_2024 = edge_attr_2024.unsqueeze(-1)

        # Process 2021 with static flow graph
        h_2021 = self.process_year(x_2021, edge_index_2021, edge_attr_2021)

        # Process 2024 with static flow graph
        h_2024 = self.process_year(x_2024, edge_index_2024, edge_attr_2024)

        # Compute difference (spatial change pattern)
        diff = h_2024 - h_2021

        # Extract batch nodes if indices provided
        if node_indices is not None:
            h_2021 = h_2021[node_indices]
            h_2024 = h_2024[node_indices]
            diff = diff[node_indices]

        return h_2021, h_2024, diff


if __name__ == "__main__":
    """Test the simplified GAT model"""
    print("Testing SimplifiedDualYearGAT")
    print("=" * 80)

    # Test parameters
    num_nodes = 100
    num_edges = 500
    batch_size = 16

    # Create test data
    x_2021 = torch.randn(num_nodes, 7, 2)  # 7 days, 2 features
    x_2024 = torch.randn(num_nodes, 7, 2)

    # Create static graphs (single graph per year)
    edge_index_2021 = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_2021 = torch.rand(num_edges)  # Flow weights

    edge_index_2024 = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_2024 = torch.rand(num_edges)

    graphs_2021 = [(edge_index_2021, edge_attr_2021)]
    graphs_2024 = [(edge_index_2024, edge_attr_2024)]

    # Create model
    model = SimplifiedDualYearGAT(
        input_size=2,
        hidden_size=128,
        num_layers=3,
        heads=4,
        output_size=256
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nInput shapes:")
    print(f"  x_2021: {x_2021.shape}")
    print(f"  x_2024: {x_2024.shape}")
    print(f"  graphs_2021: 1 static graph with {num_edges} edges")
    print(f"  graphs_2024: 1 static graph with {num_edges} edges")

    # Test forward pass (full graph)
    print(f"\n=== Full Graph Forward Pass ===")
    h_2021, h_2024, diff = model(x_2021, x_2024, graphs_2021, graphs_2024)
    print(f"Output shapes:")
    print(f"  h_2021: {h_2021.shape}")
    print(f"  h_2024: {h_2024.shape}")
    print(f"  diff: {diff.shape}")
    print(f"Expected: ({num_nodes}, 256) for each")

    # Test forward pass (batch)
    print(f"\n=== Batch Forward Pass ===")
    node_indices = torch.randint(0, num_nodes, (batch_size,))
    h_2021_batch, h_2024_batch, diff_batch = model(
        x_2021, x_2024, graphs_2021, graphs_2024, node_indices=node_indices
    )
    print(f"Batch size: {batch_size}")
    print(f"Output shapes:")
    print(f"  h_2021: {h_2021_batch.shape}")
    print(f"  h_2024: {h_2024_batch.shape}")
    print(f"  diff: {diff_batch.shape}")
    print(f"Expected: ({batch_size}, 256) for each")

    print(f"\n✓ All tests passed!")

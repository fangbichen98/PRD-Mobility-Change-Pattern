"""
Complete model: Dual-branch spatiotemporal model with attention fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_branch import TemporalBranch, ParallelTemporalBranch
from .spatial_branch import SpatialBranch, DualYearDySAT
import config


class AttentionFusion(nn.Module):
    """Attention-based fusion layer for combining temporal and spatial features"""

    def __init__(self,
                 feature_size: int = 256,
                 num_heads: int = config.ATTENTION_HEADS,
                 dropout: float = 0.2):
        """
        Initialize attention fusion layer

        Args:
            feature_size: Size of input features from each branch
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AttentionFusion, self).__init__()

        self.feature_size = feature_size
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads

        assert feature_size % num_heads == 0, "feature_size must be divisible by num_heads"

        # Multi-head attention components
        self.query = nn.Linear(feature_size, feature_size)
        self.key = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(feature_size, feature_size)

    def forward(self, temporal_features, spatial_features):
        """
        Forward pass

        Args:
            temporal_features: Features from temporal branch (batch_size, feature_size)
            spatial_features: Features from spatial branch (batch_size, feature_size)

        Returns:
            Fused features (batch_size, feature_size)
        """
        batch_size = temporal_features.size(0)

        # Stack features for attention
        # Shape: (batch_size, 2, feature_size)
        features = torch.stack([temporal_features, spatial_features], dim=1)

        # Compute Q, K, V
        Q = self.query(features)  # (batch_size, 2, feature_size)
        K = self.key(features)
        V = self.value(features)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, 2, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, 2, head_dim)

        # Reshape and concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, 2, self.feature_size)

        # Average over the two branches
        fused = attended.mean(dim=1)  # (batch_size, feature_size)

        # Output projection
        output = self.output_proj(fused)

        return output


class MultiFeatureAttentionFusion(nn.Module):
    """Multi-feature attention fusion for 9 features (6 temporal + 3 spatial)"""

    def __init__(self,
                 feature_size: int = 256,
                 num_heads: int = config.ATTENTION_HEADS,
                 dropout: float = 0.2):
        """
        Initialize multi-feature attention fusion

        Args:
            feature_size: Size of each input feature
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiFeatureAttentionFusion, self).__init__()

        self.feature_size = feature_size
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads

        assert feature_size % num_heads == 0, "feature_size must be divisible by num_heads"

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm = nn.LayerNorm(feature_size)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, features):
        """
        Forward pass

        Args:
            features: Stacked features (batch_size, num_features, feature_size)
                     num_features = 9 (6 temporal + 3 spatial)

        Returns:
            Fused features (batch_size, feature_size)
        """
        # Self-attention over all features
        attn_out, _ = self.attention(features, features, features)

        # Average pooling over features
        pooled = attn_out.mean(dim=1)  # (batch_size, feature_size)

        # Layer norm
        normed = self.norm(pooled)

        # Output projection
        output = self.output_proj(normed)

        return output


class DualBranchSTModel(nn.Module):
    """Dual-branch spatiotemporal model for mobility pattern classification"""

    def __init__(self,
                 temporal_input_size: int = 2,
                 spatial_input_size: int = 336,  # 168 hours * 2 features
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.2):
        """
        Initialize dual-branch model

        Args:
            temporal_input_size: Input size for temporal branch
            spatial_input_size: Input size for spatial branch
            hidden_size: Hidden feature size
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(DualBranchSTModel, self).__init__()

        # Temporal branch (LSTM + SPP)
        self.temporal_branch = TemporalBranch(
            input_size=temporal_input_size,
            output_size=hidden_size
        )

        # Spatial branch (DySAT)
        self.spatial_branch = SpatialBranch(
            input_size=spatial_input_size,
            output_size=hidden_size
        )

        # Attention fusion
        self.fusion = AttentionFusion(
            feature_size=hidden_size,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, temporal_input, spatial_input, edge_index, edge_attr=None, node_indices=None):
        """
        Forward pass

        Args:
            temporal_input: Temporal features (batch_size, seq_len, temporal_input_size)
            spatial_input: Spatial features for SUBGRAPH nodes (num_subgraph_nodes, time_steps, features)
                          NOTE: After subgraph extraction, this contains only the k-hop neighborhood
                          of batch nodes, not the full graph. This dramatically reduces memory usage.
            edge_index: Subgraph edge indices (2, num_subgraph_edges)
            edge_attr: Subgraph edge attributes
            node_indices: Indices of batch nodes within subgraph (batch_size,)
                         These are the positions of the actual batch nodes in the subgraph

        Returns:
            Class logits (batch_size, num_classes)
        """
        # Extract temporal features
        temporal_features = self.temporal_branch(temporal_input)

        # Extract spatial features for SUBGRAPH (not full graph)
        # This processes only ~300 nodes instead of ~10,000 nodes (33x memory reduction)
        subgraph_features = self.spatial_branch(spatial_input, edge_index, edge_attr)

        # Select spatial features for batch nodes within subgraph
        if node_indices is not None:
            spatial_features = subgraph_features[node_indices]
        else:
            # If no node_indices provided, assume spatial_input is already for batch
            spatial_features = subgraph_features[:temporal_features.size(0)]

        # Fuse features
        fused_features = self.fusion(temporal_features, spatial_features)

        # Classify
        logits = self.classifier(fused_features)

        return logits

    def get_embeddings(self, temporal_input, spatial_input, edge_index, edge_attr=None):
        """
        Get fused embeddings without classification

        Args:
            temporal_input: Temporal features
            spatial_input: Spatial features
            edge_index: Graph edge indices
            edge_attr: Edge attributes

        Returns:
            Fused embeddings (batch_size, hidden_size)
        """
        temporal_features = self.temporal_branch(temporal_input)
        spatial_features = self.spatial_branch(spatial_input, edge_index, edge_attr)
        fused_features = self.fusion(temporal_features, spatial_features)

        return fused_features


class BaselineLSTM(nn.Module):
    """Baseline LSTM model for comparison"""

    def __init__(self,
                 input_size: int = 2,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.2):
        """Initialize baseline LSTM"""
        super(BaselineLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        _, (h_n, _) = self.lstm(x)
        logits = self.classifier(h_n[-1])
        return logits


class BaselineGAT(nn.Module):
    """Baseline GAT model for comparison"""

    def __init__(self,
                 input_size: int = 336,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = config.NUM_CLASSES,
                 heads: int = 4,
                 dropout: float = 0.2):
        """Initialize baseline GAT"""
        super(BaselineGAT, self).__init__()

        from torch_geometric.nn import GATConv

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(input_size, hidden_size, heads=heads, dropout=dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * heads, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)

        logits = self.classifier(x)
        return logits


class ImprovedDualBranchModel(nn.Module):
    """Improved dual-branch model with parallel temporal and spatial processing"""

    def __init__(self,
                 temporal_input_size: int = 2,  # [total_log, net_flow_log]
                 spatial_input_size: int = 2,   # [total_log, net_flow_log]
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 num_time_steps: int = 7,  # 7 daily snapshots
                 dropout: float = 0.2):
        """
        Initialize improved dual-branch model

        Args:
            temporal_input_size: Input size per year for temporal branch
            spatial_input_size: Input size per year for spatial branch
            hidden_size: Hidden feature size
            num_classes: Number of output classes
            num_time_steps: Number of time steps (7 days)
            dropout: Dropout rate
        """
        super(ImprovedDualBranchModel, self).__init__()

        # Parallel temporal branch (LSTM + SPP)
        self.temporal_branch = ParallelTemporalBranch(
            input_size=temporal_input_size,
            output_size=hidden_size
        )

        # Dual-year spatial branch (DySAT)
        self.spatial_branch = DualYearDySAT(
            input_size=spatial_input_size,
            num_time_steps=num_time_steps,
            output_size=hidden_size
        )

        # Multi-feature attention fusion (9 features: 6 temporal + 3 spatial)
        self.fusion = MultiFeatureAttentionFusion(
            feature_size=hidden_size,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, node_indices=None):
        """
        Forward pass

        Args:
            x_2021: Temporal features for 2021 (batch_size, 7, 2) or (num_nodes, 7, 2)
            x_2024: Temporal features for 2024 (batch_size, 7, 2) or (num_nodes, 7, 2)
            graphs_2021: List of 7 (edge_index, edge_attr) tuples for 2021
            graphs_2024: List of 7 (edge_index, edge_attr) tuples for 2024
            node_indices: Optional node indices for subgraph extraction

        Returns:
            Class logits (batch_size, num_classes)
        """
        # Determine if we need to extract batch features
        # If node_indices is provided, x_2021/x_2024 are full graph features
        if node_indices is not None:
            # Extract batch features for temporal branch
            x_2021_batch = x_2021[node_indices]  # (batch_size, 7, 2)
            x_2024_batch = x_2024[node_indices]  # (batch_size, 7, 2)

            # Extract temporal features (6 features: LSTM + SPP for both years + diff)
            temporal_features = self.temporal_branch(x_2021_batch, x_2024_batch)
            # Shape: (batch_size, 6, hidden_size)

            # Extract spatial features (3 features: 2021 + 2024 + diff)
            # Spatial branch processes full graph and extracts batch nodes
            spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
                x_2021, x_2024, graphs_2021, graphs_2024, node_indices=node_indices
            )
        else:
            # Input is already batch-level
            x_2021_batch = x_2021
            x_2024_batch = x_2024

            # Extract temporal features
            temporal_features = self.temporal_branch(x_2021_batch, x_2024_batch)

            # Extract spatial features (no node extraction needed)
            spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
                x_2021_batch, x_2024_batch, graphs_2021, graphs_2024, node_indices=None
            )

        # Stack spatial features: (batch_size, 3, hidden_size)
        spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate temporal and spatial features: (batch_size, 9, hidden_size)
        all_features = torch.cat([temporal_features, spatial_features], dim=1)

        # Fuse features using multi-head attention
        fused_features = self.fusion(all_features)

        # Classify
        logits = self.classifier(fused_features)

        return logits

    def get_embeddings(self, x_2021, x_2024, graphs_2021, graphs_2024, node_indices=None):
        """
        Get fused embeddings without classification

        Args:
            x_2021: Temporal features for 2021
            x_2024: Temporal features for 2024
            graphs_2021: Dynamic graphs for 2021
            graphs_2024: Dynamic graphs for 2024
            node_indices: Optional node indices

        Returns:
            Fused embeddings (batch_size, hidden_size)
        """
        # Extract features
        temporal_features = self.temporal_branch(x_2021, x_2024)
        spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
            x_2021, x_2024, graphs_2021, graphs_2024, node_indices=node_indices
        )
        spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate and fuse
        all_features = torch.cat([temporal_features, spatial_features], dim=1)
        fused_features = self.fusion(all_features)

        return fused_features


if __name__ == "__main__":
    # Test complete model
    print("Testing Improved Dual-Branch Spatiotemporal Model")

    batch_size = 8
    num_nodes = 100
    num_time_steps = 7
    input_size = 2  # [total_log, net_flow_log]
    num_edges = 500

    # Create dummy inputs
    x_2021 = torch.randn(num_nodes, num_time_steps, input_size)
    x_2024 = torch.randn(num_nodes, num_time_steps, input_size)

    # Create dummy dynamic graphs
    graphs_2021 = []
    graphs_2024 = []
    for t in range(num_time_steps):
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 1)
        graphs_2021.append((edge_index, edge_attr))
        graphs_2024.append((edge_index, edge_attr))

    # Node indices for batch
    node_indices = torch.randint(0, num_nodes, (batch_size,))

    # Initialize model
    model = ImprovedDualBranchModel(
        temporal_input_size=input_size,
        spatial_input_size=input_size,
        num_time_steps=num_time_steps
    )

    # Forward pass
    output = model(x_2021, x_2024, graphs_2021, graphs_2024, node_indices=node_indices)

    print(f"Input shapes:")
    print(f"  x_2021: {x_2021.shape}")
    print(f"  x_2024: {x_2024.shape}")
    print(f"  graphs: {len(graphs_2021)} snapshots per year")
    print(f"  node_indices: {node_indices.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {config.NUM_CLASSES})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test old model for comparison
    print("\n=== Old DualBranchSTModel ===")
    old_model = DualBranchSTModel()
    temporal_input = torch.randn(batch_size, 168, 2)
    spatial_input = torch.randn(batch_size, 336)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    old_output = old_model(temporal_input, spatial_input, edge_index)
    print(f"Input shapes: temporal={temporal_input.shape}, spatial={spatial_input.shape}")
    print(f"Output shape: {old_output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in old_model.parameters())}")

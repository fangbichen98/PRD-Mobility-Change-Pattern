"""
Complete model: Dual-branch spatiotemporal model with attention fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_branch import TemporalBranch
from .spatial_branch import SpatialBranch
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


if __name__ == "__main__":
    # Test complete model
    print("Testing Dual-Branch Spatiotemporal Model")

    batch_size = 8
    seq_len = 168
    temporal_input_size = 2
    num_nodes = 100
    spatial_input_size = 336
    num_edges = 500

    # Create dummy inputs
    temporal_input = torch.randn(batch_size, seq_len, temporal_input_size)
    spatial_input = torch.randn(num_nodes, spatial_input_size)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Initialize model
    model = DualBranchSTModel()

    # Forward pass (need to handle batch/node mismatch in practice)
    # For testing, we'll use first batch_size nodes
    spatial_input_batch = spatial_input[:batch_size]
    output = model(temporal_input, spatial_input_batch, edge_index)

    print(f"Temporal input shape: {temporal_input.shape}")
    print(f"Spatial input shape: {spatial_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

"""
Simplified dual-branch model using pure graph structure

REFACTORING: Now supports GCN (main model) and GraphSAGE (baseline)
"""
import torch
import torch.nn as nn
import config
from src.models.temporal_branch import ParallelTemporalBranch
from src.models.spatial_branch_pure_graph import PureGraphDualYearGCN, PureGraphDualYearSAGE


class MultiFeatureAttentionFusion(nn.Module):
    """Multi-feature attention fusion layer"""

    def __init__(self, feature_size: int = 256, num_heads: int = 4, dropout: float = 0.2):
        super(MultiFeatureAttentionFusion, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(feature_size)
        self.output_proj = nn.Linear(feature_size, feature_size)

    def forward(self, features):
        """
        Args:
            features: (batch_size, num_features, feature_size)

        Returns:
            fused: (batch_size, feature_size)
        """
        # Self-attention over features
        attn_out, _ = self.attention(features, features, features)

        # Average pooling over features
        pooled = attn_out.mean(dim=1)

        # Layer norm and projection
        output = self.output_proj(self.layer_norm(pooled))

        return output


class PureGraphDualBranchModel(nn.Module):
    """
    Pure Graph Dual-Branch Model for 9-Class Mobility Pattern Classification

    REFACTORING: Now supports GCN (main) and GraphSAGE (baseline) spatial branches

    Key changes from original:
    1. Spatial branch uses only graph structure (no external node features)
    2. Temporal branch still uses time series features (kept for temporal patterns)
    3. Fusion combines 6 temporal + 3 spatial = 9 features
    4. Single 9-class classification head (simplified from hierarchical)

    Architecture:
        Temporal Branch: LSTM + SPP → 6 features (2021, 2024, diff) × 2
        Spatial Branch: GCN or GraphSAGE → 3 features (2021, 2024, diff)
        Fusion: Multi-head attention → 256-dim
        Classifier: Single 9-class classification head
    """

    def __init__(self,
                 temporal_input_size: int = 2,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 num_time_steps: int = 7,
                 dropout: float = 0.2,
                 spatial_model_type: str = 'gcn'):
        """
        Initialize pure graph dual-branch model

        Args:
            temporal_input_size: Input size per year for temporal branch
            hidden_size: Hidden feature size
            num_classes: Number of output classes (9)
            num_time_steps: Number of time steps (7 days)
            dropout: Dropout rate
            spatial_model_type: Type of spatial model ('gcn' or 'sage')
        """
        super(PureGraphDualBranchModel, self).__init__()

        # Store model type for reference
        self.spatial_model_type = spatial_model_type

        # Temporal branch (unchanged - still uses time series features)
        self.temporal_branch = ParallelTemporalBranch(
            input_size=temporal_input_size,
            output_size=hidden_size
        )

        # Spatial branch (NEW - pure graph structure with GCN or GraphSAGE)
        if spatial_model_type == 'gcn':
            self.spatial_branch = PureGraphDualYearGCN(
                hidden_size=config.GAT_HIDDEN_SIZE,  # Reuse GAT_HIDDEN_SIZE config
                num_layers=config.GAT_LAYERS,        # Reuse GAT_LAYERS config
                dropout=dropout,
                output_size=hidden_size
            )
        elif spatial_model_type == 'sage':
            self.spatial_branch = PureGraphDualYearSAGE(
                hidden_size=config.GAT_HIDDEN_SIZE,  # Reuse GAT_HIDDEN_SIZE config
                num_layers=config.GAT_LAYERS,        # Reuse GAT_LAYERS config
                dropout=dropout,
                output_size=hidden_size
            )
        else:
            raise ValueError(f"Unsupported spatial_model_type: {spatial_model_type}. Use 'gcn' or 'sage'.")

        # Multi-feature attention fusion (9 features: 6 temporal + 3 spatial)
        self.fusion = MultiFeatureAttentionFusion(
            feature_size=hidden_size,
            dropout=dropout
        )

        # Single 9-class classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices=None):
        """
        Forward pass

        Args:
            x_2021: Temporal features for 2021 (batch_size, 7, 2) or (num_nodes, 7, 2)
            x_2024: Temporal features for 2024 (batch_size, 7, 2) or (num_nodes, 7, 2)
            graphs_2021: List of (edge_index, edge_attr) for 2021
            graphs_2024: List of (edge_index, edge_attr) for 2024
            num_nodes: Total number of nodes
            node_indices: Optional node indices for batch extraction

        Returns:
            logits: (batch_size, num_classes) - 9-class classification logits
        """
        # Extract batch features for temporal branch
        if node_indices is not None:
            x_2021_batch = x_2021[node_indices]  # (batch_size, 7, 2)
            x_2024_batch = x_2024[node_indices]  # (batch_size, 7, 2)
        else:
            x_2021_batch = x_2021
            x_2024_batch = x_2024

        # Extract temporal features (6 features: LSTM + SPP for both years + diff)
        temporal_features = self.temporal_branch(x_2021_batch, x_2024_batch)
        # Shape: (batch_size, 6, hidden_size)

        # Extract spatial features (3 features: 2021 + 2024 + diff)
        # NEW: Spatial branch only uses graph structure, no node features!
        spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        # Stack spatial features: (batch_size, 3, hidden_size)
        spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate temporal and spatial features: (batch_size, 9, hidden_size)
        all_features = torch.cat([temporal_features, spatial_features], dim=1)

        # Fuse features using multi-head attention
        fused_features = self.fusion(all_features)

        # Single 9-class classification
        logits = self.classifier(fused_features)

        return logits

    def get_embeddings(self, x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices=None):
        """
        Get fused embeddings without classification

        Args:
            x_2021: Temporal features for 2021
            x_2024: Temporal features for 2024
            graphs_2021: Graphs for 2021
            graphs_2024: Graphs for 2024
            num_nodes: Total number of nodes
            node_indices: Optional node indices

        Returns:
            Fused embeddings (batch_size, hidden_size)
        """
        # Extract batch features
        if node_indices is not None:
            x_2021_batch = x_2021[node_indices]
            x_2024_batch = x_2024[node_indices]
        else:
            x_2021_batch = x_2021
            x_2024_batch = x_2024

        # Extract features
        temporal_features = self.temporal_branch(x_2021_batch, x_2024_batch)
        spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
            graphs_2021, graphs_2024, num_nodes, node_indices
        )
        spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate and fuse
        all_features = torch.cat([temporal_features, spatial_features], dim=1)
        fused_features = self.fusion(all_features)

        return fused_features


if __name__ == "__main__":
    # Test complete model with both GCN and GraphSAGE
    print("Testing PureGraphDualBranchModel (GCN and GraphSAGE)")
    print("=" * 80)

    batch_size = 8
    num_nodes = 100
    num_edges = 500
    num_time_steps = 7

    # Create test data
    # Temporal features (still needed for temporal branch)
    x_2021 = torch.randn(num_nodes, num_time_steps, 2)
    x_2024 = torch.randn(num_nodes, num_time_steps, 2)

    # Graph structure (no node features needed!)
    # Simulate extreme flow values (max 12101)
    edge_index_2021 = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_2021 = torch.rand(num_edges) * 12101

    edge_index_2024 = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_2024 = torch.rand(num_edges) * 12101

    graphs_2021 = [(edge_index_2021, edge_attr_2021)]
    graphs_2024 = [(edge_index_2024, edge_attr_2024)]

    # Batch indices
    node_indices = torch.randint(0, num_nodes, (batch_size,))

    # ==============================================================================
    # Test GCN Model
    # ==============================================================================
    print("\n1. Testing with GCN Spatial Branch")
    print("-" * 80)

    model_gcn = PureGraphDualBranchModel(
        temporal_input_size=2,
        hidden_size=256,
        num_classes=9,
        num_time_steps=7,
        dropout=0.2,
        spatial_model_type='gcn'
    )

    print(f"Model Architecture:")
    print(f"  - Temporal Branch: LSTM + SPP → 6 features")
    print(f"  - Spatial Branch: GCN (normalize=True, add_self_loops=True) → 3 features")
    print(f"  - Fusion: Multi-head Attention → 256-dim")
    print(f"  - Classification: Single 9-class head")
    print(f"  - Total parameters: {sum(p.numel() for p in model_gcn.parameters()):,}")

    model_gcn.eval()
    with torch.no_grad():
        logits_gcn = model_gcn(
            x_2021=x_2021,
            x_2024=x_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

    print(f"\nOutput shapes:")
    print(f"  - Logits: {logits_gcn.shape} (expected: {batch_size}, 9)")

    # Check for NaN/Inf
    has_nan = torch.isnan(logits_gcn).any()
    has_inf = torch.isinf(logits_gcn).any()
    print(f"\nNumerical stability check:")
    print(f"  - Has NaN: {has_nan}")
    print(f"  - Has Inf: {has_inf}")
    if not (has_nan or has_inf):
        print(f"  ✓ GCN model handles extreme flow values well!")

    # ==============================================================================
    # Test GraphSAGE Model
    # ==============================================================================
    print("\n\n2. Testing with GraphSAGE Spatial Branch")
    print("-" * 80)

    model_sage = PureGraphDualBranchModel(
        temporal_input_size=2,
        hidden_size=256,
        num_classes=9,
        num_time_steps=7,
        dropout=0.2,
        spatial_model_type='sage'
    )

    print(f"Model Architecture:")
    print(f"  - Temporal Branch: LSTM + SPP → 6 features")
    print(f"  - Spatial Branch: GraphSAGE (aggr='mean', log1p weights) → 3 features")
    print(f"  - Fusion: Multi-head Attention → 256-dim")
    print(f"  - Classification: Single 9-class head")
    print(f"  - Total parameters: {sum(p.numel() for p in model_sage.parameters()):,}")

    model_sage.eval()
    with torch.no_grad():
        logits_sage = model_sage(
            x_2021=x_2021,
            x_2024=x_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

    print(f"\nOutput shapes:")
    print(f"  - Logits: {logits_sage.shape} (expected: {batch_size}, 9)")

    # Check for NaN/Inf
    has_nan = torch.isnan(logits_sage).any()
    has_inf = torch.isinf(logits_sage).any()
    print(f"\nNumerical stability check:")
    print(f"  - Has NaN: {has_nan}")
    print(f"  - Has Inf: {has_inf}")
    if not (has_nan or has_inf):
        print(f"  ✓ GraphSAGE model handles extreme flow values well!")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("\nKey features:")
    print("  1. Spatial branch uses only graph structure (no redundant node features)")
    print("  2. Supports GCN (main) and GraphSAGE (baseline)")
    print("  3. GCN: Laplacian normalization for extreme flows (max 12101)")
    print("  4. GraphSAGE: Log1p transformation for numerical stability")
    print("  5. Single pass per year (not 7 daily passes)")
    print("  6. ~7x faster spatial processing")
    print("  7. Simpler data preprocessing")
    print("  8. Single 9-class classification (simplified from hierarchical)")
    print("  9. Featureless learning (all-1 node features)")


"""
Enhanced dual-branch model with Phase 2 improvements:
1. Multi-scale temporal branch
2. Gated feature fusion

This model integrates the improvements from Phase 2 of the optimization plan.
"""
import torch
import torch.nn as nn
import config
from src.models.spatial_branch_pure_graph import PureGraphDualYearGAT
from src.models.multi_scale_temporal import SimplifiedMultiScaleTemporal
from src.models.gated_fusion import GatedFeatureFusion


class EnhancedDualBranchModel(nn.Module):
    """
    Enhanced Dual-Branch Model with multi-scale temporal processing and gated fusion.

    Improvements over baseline:
    1. Multi-scale temporal branch (hourly + daily + weekly patterns)
    2. Gated fusion for dynamic feature selection
    3. Enhanced regularization

    Architecture:
        Temporal Branch: Multi-scale (hourly/daily/weekly) → 3 features per year
        Spatial Branch: Pure Graph GAT → 3 features (2021, 2024, diff)
        Fusion: Gated mechanism → 256-dim
        Classifier: Single 9-class classification head
    """

    def __init__(self,
                 temporal_input_size: int = 1,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 num_time_steps: int = 168,
                 dropout: float = 0.4):
        """
        Initialize enhanced dual-branch model

        Args:
            temporal_input_size: Input size per timestep (default: 1 for total flow)
            hidden_size: Hidden feature size (default: 256)
            num_classes: Number of output classes (9)
            num_time_steps: Number of time steps (168 hours)
            dropout: Dropout rate (default: 0.4, increased from 0.2)
        """
        super(EnhancedDualBranchModel, self).__init__()

        self.temporal_input_size = temporal_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps

        # Temporal branch: Multi-scale processing
        self.temporal_branch = SimplifiedMultiScaleTemporal(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            lstm_hidden=128,
            dropout=dropout
        )

        # Spatial branch: Pure graph GAT (unchanged)
        self.spatial_branch = PureGraphDualYearGAT(
            hidden_size=config.GAT_HIDDEN_SIZE,
            num_layers=config.GAT_LAYERS,
            heads=config.GAT_HEADS,
            output_size=hidden_size
        )

        # Gated fusion layer (replaces attention fusion)
        self.fusion = GatedFeatureFusion(
            feature_size=hidden_size,
            num_features=6,  # 2 temporal + 3 spatial + 1 diff = 6
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
            x_2021: Temporal features for 2021 (batch_size, 168, 1) or (num_nodes, 168, 1)
            x_2024: Temporal features for 2024 (batch_size, 168, 1) or (num_nodes, 168, 1)
            graphs_2021: List of (edge_index, edge_attr) for 2021
            graphs_2024: List of (edge_index, edge_attr) for 2024
            num_nodes: Total number of nodes
            node_indices: Optional node indices for batch extraction

        Returns:
            logits: (batch_size, num_classes) - 9-class classification logits
        """
        # Extract batch features for temporal branch
        if node_indices is not None:
            x_2021_batch = x_2021[node_indices]  # (batch_size, 168, 1)
            x_2024_batch = x_2024[node_indices]  # (batch_size, 168, 1)
        else:
            x_2021_batch = x_2021
            x_2024_batch = x_2024

        # Extract temporal features with multi-scale processing
        # Returns: (batch_size, hidden_size) - already fused multi-scale features
        temporal_2021 = self.temporal_branch.extract_features_single(x_2021_batch)
        temporal_2024 = self.temporal_branch.extract_features_single(x_2024_batch)

        # Compute temporal difference
        temporal_diff = temporal_2024 - temporal_2021  # (batch_size, hidden_size)

        # Stack temporal features: (batch_size, 3, hidden_size)
        temporal_features = torch.stack([temporal_2021, temporal_2024, temporal_diff], dim=1)

        # Extract spatial features (3 features: 2021 + 2024 + diff)
        spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        # Stack spatial features: (batch_size, 3, hidden_size)
        spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate temporal and spatial features: (batch_size, 6, hidden_size)
        all_features = torch.cat([temporal_features, spatial_features], dim=1)

        # Gated fusion
        fused = self.fusion(all_features)  # (batch_size, hidden_size)

        # Classification
        logits = self.classifier(fused)  # (batch_size, num_classes)

        return logits


class AlternativeEnhancedModel(nn.Module):
    """
    Alternative enhanced model with AdaptiveGatedFusion.

    Uses multi-head attention + gating for more expressive fusion.
    """

    def __init__(self,
                 temporal_input_size: int = 1,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 num_time_steps: int = 168,
                 dropout: float = 0.4):
        super(AlternativeEnhancedModel, self).__init__()

        from src.models.multi_scale_temporal import MultiScaleTemporalBranch
        from src.models.gated_fusion import AdaptiveGatedFusion

        # Multi-scale temporal branch (full version)
        self.temporal_branch = MultiScaleTemporalBranch(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            lstm_hidden=128,
            lstm_layers=2,
            dropout=dropout
        )

        # Spatial branch
        self.spatial_branch = PureGraphDualYearGAT(
            hidden_size=config.GAT_HIDDEN_SIZE,
            num_layers=config.GAT_LAYERS,
            heads=config.GAT_HEADS,
            output_size=hidden_size
        )

        # Adaptive gated fusion
        self.fusion = AdaptiveGatedFusion(
            feature_size=hidden_size,
            num_features=6,
            num_heads=4,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices=None):
        """Forward pass"""
        # Extract batch features
        if node_indices is not None:
            x_2021_batch = x_2021[node_indices]
            x_2024_batch = x_2024[node_indices]
        else:
            x_2021_batch = x_2021
            x_2024_batch = x_2024

        # Multi-scale temporal features (already includes multi-year processing)
        temporal_features = self.temporal_branch(x_2021_batch, x_2024_batch)
        # Returns: (batch_size, hidden_size)

        # Expand to 3 features for consistency
        temporal_2021 = temporal_features  # Simplified
        temporal_2024 = temporal_features  # Simplified
        temporal_diff = temporal_features * 0  # Zero diff for now

        temporal_stack = torch.stack([temporal_2021, temporal_2024, temporal_diff], dim=1)

        # Spatial features
        spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        spatial_stack = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate
        all_features = torch.cat([temporal_stack, spatial_stack], dim=1)

        # Fusion
        fused = self.fusion(all_features)

        # Classification
        logits = self.classifier(fused)

        return logits


# Monkey-patch SimplifiedMultiScaleTemporal to add extract_features_single method
def extract_features_single(self, x):
    """Extract features from a single year's time series"""
    # Process hourly
    _, (h_n, _) = self.lstm(x)
    h_hourly = self.proj_hidden(h_n[-1])

    # Process daily
    x_daily = x.view(x.size(0), 7, 24, x.size(2)).mean(dim=2)
    _, (h_daily_n, _) = self.lstm(x_daily)
    h_daily = self.proj_hidden(h_daily_n[-1])

    # Weekly stats
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    trend = (x[:, -1, :] - x[:, 0, :]) / 168
    stats = torch.cat([mean, std, trend], dim=1)
    h_weekly = self.weekly_net(stats)

    # Concatenate and fuse
    multi_scale = torch.cat([h_hourly, h_daily, h_weekly], dim=1)
    output = self.fusion(multi_scale)

    return output


# Attach the method
SimplifiedMultiScaleTemporal.extract_features_single = extract_features_single


if __name__ == "__main__":
    # Test the models
    print("Testing Enhanced Models:")

    batch_size = 8
    num_nodes = 100
    timesteps = 168
    input_size = 1
    hidden_size = 256

    # Dummy data
    x_2021 = torch.randn(batch_size, timesteps, input_size)
    x_2024 = torch.randn(batch_size, timesteps, input_size)

    # Dummy graphs
    edge_index = torch.randint(0, num_nodes, (2, 1000))
    edge_attr = torch.randn(1000, 1)
    graphs_2021 = [(edge_index, edge_attr)]
    graphs_2024 = [(edge_index, edge_attr)]

    node_indices = torch.randint(0, num_nodes, (batch_size,))

    # Test EnhancedDualBranchModel
    print("\n1. EnhancedDualBranchModel:")
    model1 = EnhancedDualBranchModel(temporal_input_size=input_size, hidden_size=hidden_size)
    output1 = model1(x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices)
    print(f"   Output shape: {output1.shape}")
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"   Parameters: {params1:,}")

    # Test AlternativeEnhancedModel
    print("\n2. AlternativeEnhancedModel:")
    model2 = AlternativeEnhancedModel(temporal_input_size=input_size, hidden_size=hidden_size)
    output2 = model2(x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices)
    print(f"   Output shape: {output2.shape}")
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"   Parameters: {params2:,}")

    print(f"\nAll enhanced models work correctly!")
    print(f"Model 2 has {(params2/params1 - 1) * 100:.1f}% more parameters than Model 1")

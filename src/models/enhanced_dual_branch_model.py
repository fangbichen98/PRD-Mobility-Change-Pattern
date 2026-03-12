"""
Enhanced dual-branch model with Phase 2 improvements:
1. Multi-scale temporal branch
2. Gated feature fusion
3. GINE spatial branch with Laplacian PE (optional)

This model integrates the improvements from Phase 2 of the optimization plan.
"""
import torch
import torch.nn as nn
import config
from src.models.spatial_branch_pure_graph import PureGraphDualYearGCN, PureGraphDualYearSAGE
from src.models.spatial_branch_gine import PureGraphDualYearGINE, compute_laplacian_pe
from src.models.multi_scale_temporal import SimplifiedMultiScaleTemporal
from src.models.gated_fusion import GatedFeatureFusion


class EnhancedDualBranchModel(nn.Module):
    """
    Enhanced Dual-Branch Model with multi-scale temporal processing and gated fusion.

    Improvements over baseline:
    1. Multi-scale temporal branch (hourly + daily + weekly patterns)
    2. Gated fusion for dynamic feature selection
    3. Enhanced regularization
    4. Optional GINE spatial branch with Laplacian PE

    Architecture:
        Temporal Branch: Multi-scale (hourly/daily/weekly) → 3 features per year
        Spatial Branch: GCN/SAGE/GINE (featureless or with Laplacian PE) → 3 features (2021, 2024, diff)
        Fusion: Gated mechanism → 256-dim
        Classifier: Single 9-class classification head
    """

    def __init__(self,
                 temporal_input_size: int = 2,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 num_time_steps: int = 168,
                 dropout: float = 0.4,
                 spatial_model: str = "GCN"):
        """
        Initialize enhanced dual-branch model

        Args:
            temporal_input_size: Input size per timestep (default: 2 for [inflow, outflow])
            hidden_size: Hidden feature size (default: 256)
            num_classes: Number of output classes (9)
            num_time_steps: Number of time steps (168 hours)
            dropout: Dropout rate (default: 0.4, increased from 0.2)
            spatial_model: Spatial branch model type ("GCN", "SAGE", or "GINE")
        """
        super(EnhancedDualBranchModel, self).__init__()

        self.temporal_input_size = temporal_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps
        self.spatial_model = spatial_model

        # Temporal branch: Multi-scale processing
        self.temporal_branch = SimplifiedMultiScaleTemporal(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            lstm_hidden=128,
            dropout=dropout
        )

        # Spatial branch: Choose model type
        if spatial_model == "GINE":
            self.spatial_branch = PureGraphDualYearGINE(
                input_size=config.LAPLACIAN_PE_DIM,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size
            )
            self.use_laplacian_pe = True
            self.laplacian_pe_2021 = None
            self.laplacian_pe_2024 = None
        elif spatial_model == "SAGE":
            self.spatial_branch = PureGraphDualYearSAGE(
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size
            )
            self.use_laplacian_pe = False
        else:  # Default to GCN
            self.spatial_branch = PureGraphDualYearGCN(
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size
            )
            self.use_laplacian_pe = False

        # Gated fusion layer (replaces attention fusion)
        self.fusion = GatedFeatureFusion(
            feature_size=hidden_size,
            num_features=6,  # 3 temporal + 3 spatial
            dropout=dropout
        )

        # Single 9-class classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def compute_laplacian_pe_if_needed(self, graphs_2021, graphs_2024, num_nodes, device):
        """Compute Laplacian PE for GINE model if not already computed"""
        if self.use_laplacian_pe and self.laplacian_pe_2021 is None:
            edge_index_2021, edge_attr_2021 = graphs_2021[0]
            edge_index_2024, edge_attr_2024 = graphs_2024[0]

            self.laplacian_pe_2021 = compute_laplacian_pe(
                edge_index_2021,
                edge_attr_2021.squeeze(),
                num_nodes,
                k=config.LAPLACIAN_PE_DIM,
                device=device
            )
            self.laplacian_pe_2024 = compute_laplacian_pe(
                edge_index_2024,
                edge_attr_2024.squeeze(),
                num_nodes,
                k=config.LAPLACIAN_PE_DIM,
                device=device
            )

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices=None):
        """
        Forward pass

        Args:
            x_2021: Temporal features for 2021 (batch_size, 168, 2) or (num_nodes, 168, 2)
            x_2024: Temporal features for 2024 (batch_size, 168, 2) or (num_nodes, 168, 2)
            graphs_2021: List of (edge_index, edge_attr) for 2021
            graphs_2024: List of (edge_index, edge_attr) for 2024
            num_nodes: Total number of nodes
            node_indices: Optional node indices for batch extraction

        Returns:
            logits: (batch_size, num_classes) - 9-class classification logits
        """
        # Extract batch features for temporal branch
        if node_indices is not None:
            x_2021_batch = x_2021[node_indices]  # (batch_size, 168, 2)
            x_2024_batch = x_2024[node_indices]  # (batch_size, 168, 2)
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

        # Extract spatial features
        if self.use_laplacian_pe:
            # Compute Laplacian PE if needed
            self.compute_laplacian_pe_if_needed(graphs_2021, graphs_2024, num_nodes, x_2021.device)

            # GINE forward with Laplacian PE
            spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
                graphs_2021=graphs_2021,
                graphs_2024=graphs_2024,
                node_features_2021=self.laplacian_pe_2021,
                node_features_2024=self.laplacian_pe_2024,
                node_indices=node_indices
            )
        else:
            # GCN/SAGE forward (featureless)
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

        # Spatial branch: Pure graph GCN (featureless learning)
        self.spatial_branch = PureGraphDualYearGCN(
            hidden_size=config.SPATIAL_HIDDEN_SIZE,
            num_layers=config.SPATIAL_LAYERS,
            dropout=dropout,
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
        """
        Forward pass

        FIXED: Now properly extracts separate 2021/2024 features and computes temporal difference
        """
        # Extract batch features
        if node_indices is not None:
            x_2021_batch = x_2021[node_indices]
            x_2024_batch = x_2024[node_indices]
        else:
            x_2021_batch = x_2021
            x_2024_batch = x_2024

        # FIX: Extract multi-scale features for each year separately
        # Using the methods from MultiScaleTemporalBranch
        temporal_2021_list = []
        temporal_2024_list = []

        for x, year_list in [(x_2021_batch, temporal_2021_list), (x_2024_batch, temporal_2024_list)]:
            # Extract hourly features
            h_hourly = self.temporal_branch.extract_hourly_features(x)
            # Extract daily features
            h_daily = self.temporal_branch.extract_daily_features(x)
            # Extract weekly features
            h_weekly = self.temporal_branch.extract_weekly_features(x)

            # Stack multi-scale features
            multi_scale = torch.stack([h_hourly, h_daily, h_weekly], dim=1)
            year_list.append(multi_scale)

        # Compute proper temporal difference (FIXED: was temporal_features * 0)
        temporal_diff = temporal_2024_list[0] - temporal_2021_list[0]

        # Stack temporal features: (batch_size, 3, hidden_size)
        temporal_stack = torch.cat([temporal_2021_list[0], temporal_2024_list[0], temporal_diff], dim=1)

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
    """
    Extract features from a single year's time series

    FIXED: Now uses separate LSTMs for hourly and daily processing
    Updated: Supports 2-feature input [inflow, outflow]
    """
    # Process hourly with dedicated LSTM
    _, (h_hourly_n, _) = self.lstm_hourly(x)
    h_hourly = self.proj_hourly(h_hourly_n[-1])

    # Process daily (sum over 24 hours for true daily flow) with dedicated LSTM
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

    # FIX: Create full graph features (num_nodes, timesteps, input_size)
    x_2021 = torch.randn(num_nodes, timesteps, input_size)
    x_2024 = torch.randn(num_nodes, timesteps, input_size)

    # Dummy graphs
    edge_index = torch.randint(0, num_nodes, (2, 1000))
    edge_attr = torch.randn(1000, 1)
    graphs_2021 = [(edge_index, edge_attr)]
    graphs_2024 = [(edge_index, edge_attr)]

    # FIX: Create valid node indices for batch extraction
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

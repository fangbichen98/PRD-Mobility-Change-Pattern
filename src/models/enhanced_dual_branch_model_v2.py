"""
Enhanced dual-branch model to match checkpoint structure with MultiScaleGATStack.
This version is specifically for loading checkpoints that use gat_stack architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, List
import config


class MultiScaleGATStack(nn.Module):
    """
    Multi-scale GAT stack with scale projections and layer fusion.
    This matches the checkpoint structure: spatial_branch.gat_stack.gat_layers.X.*
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, num_scales: int = 3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = heads
        self.num_scales = num_scales

        # GAT layers for each scale
        self.gat_layers = nn.ModuleList()

        for layer_idx in range(num_layers):
            # Create GAT layer
            gat_conv = GATConv(
                in_channels=hidden_size if layer_idx > 0 else 3,  # Input: 3 structural features
                out_channels=hidden_size,
                heads=heads,
                dropout=dropout,
                edge_dim=1
            )

            # Scale projections for multi-scale processing
            scale_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size * heads, hidden_size),
                    nn.LayerNorm(hidden_size)
                ) for _ in range(num_scales)
            ])

            # Layer fusion
            layer_fusion = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )

            self.gat_layers.append(nn.ModuleDict({
                'gat_conv': gat_conv,
                'scale_projs': scale_projs,
                'layer_fusion': layer_fusion
            }))

        # Cross-layer attention
        self.cross_layer_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass

        Args:
            x: Node features (num_nodes, 3) - structural features
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, 1)

        Returns:
            h: Node embeddings (num_nodes, hidden_size)
        """
        h = x
        layer_outputs = []

        for layer_dict in self.gat_layers:
            gat_conv = layer_dict['gat_conv']
            scale_projs = layer_dict['scale_projs']
            layer_fusion = layer_dict['layer_fusion']

            # GAT convolution
            h_out = gat_conv(h, edge_index, edge_attr)

            # Multi-scale projections
            scale_outputs = []
            for proj in scale_projs:
                scale_out = proj(h_out)
                scale_outputs.append(scale_out)

            # Fuse scales
            if len(scale_outputs) > 1:
                h_fused = torch.stack(scale_outputs, dim=0).mean(dim=0)
            else:
                h_fused = scale_outputs[0]

            # Apply layer fusion
            h = layer_fusion(h_fused) + h  # Residual
            layer_outputs.append(h.unsqueeze(1))  # (num_nodes, 1, hidden_size)

        # Cross-layer attention
        h_stack = torch.cat(layer_outputs, dim=1)  # (num_nodes, num_layers, hidden_size)
        h_attn, _ = self.cross_layer_attn(h_stack, h_stack, h_stack)
        h_out = h_attn.mean(dim=1)  # (num_nodes, hidden_size)

        return h_out


class PureGraphDualYearGAT_v2(nn.Module):
    """
    Pure graph GAT with MultiScaleGATStack.
    Matches checkpoint: spatial_branch.gat_stack, spatial_branch.output_proj
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, output_size: int = 256):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = heads
        self.output_size = output_size

        # Multi-scale GAT stack
        self.gat_stack = MultiScaleGATStack(
            hidden_size=hidden_size,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )

        # Output projection with normalization and dropout
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cache
        self._cached_2021 = None
        self._cached_2024 = None
        self._cache_valid = False

    def compute_structural_features(self, edge_index, edge_attr, num_nodes):
        """Compute structural node features from graph"""
        device = edge_index.device

        if edge_attr.dim() == 2:
            edge_attr = edge_attr.squeeze(-1)

        in_degree = torch.zeros(num_nodes, device=device, dtype=edge_attr.dtype)
        out_degree = torch.zeros(num_nodes, device=device, dtype=edge_attr.dtype)

        src, dst = edge_index[0], edge_index[1]

        out_degree.scatter_add_(0, src, edge_attr)
        in_degree.scatter_add_(0, dst, edge_attr)

        total_degree = in_degree + out_degree

        in_degree_log = torch.log1p(in_degree)
        out_degree_log = torch.log1p(out_degree)
        total_degree_log = torch.log1p(total_degree)

        features = torch.stack([
            in_degree_log,
            out_degree_log,
            total_degree_log
        ], dim=1)

        return features

    def forward(self, graphs_2021, graphs_2024, num_nodes, node_indices=None):
        """
        Forward pass

        Args:
            graphs_2021: List of (edge_index, edge_attr) for 2021
            graphs_2024: List of (edge_index, edge_attr) for 2024
            num_nodes: Total number of nodes
            node_indices: Optional node indices for batch extraction

        Returns:
            h: (batch_size, output_size * 3) - [2021, 2024, diff]
        """
        edge_index_2021, edge_attr_2021 = graphs_2021[0]
        edge_index_2024, edge_attr_2024 = graphs_2024[0]

        edge_attr_2021 = edge_attr_2021.float()
        edge_attr_2024 = edge_attr_2024.float()

        # Compute structural features
        x_2021 = self.compute_structural_features(edge_index_2021, edge_attr_2021, num_nodes)
        x_2024 = self.compute_structural_features(edge_index_2024, edge_attr_2024, num_nodes)

        # Ensure edge_attr is 2D for GATConv
        if edge_attr_2021.dim() == 1:
            edge_attr_2021 = edge_attr_2021.unsqueeze(-1)
        if edge_attr_2024.dim() == 1:
            edge_attr_2024 = edge_attr_2024.unsqueeze(-1)

        # Process 2021
        h_2021 = self.gat_stack(x_2021, edge_index_2021, edge_attr_2021)
        h_2021 = self.output_proj(h_2021)

        # Process 2024
        h_2024 = self.gat_stack(x_2024, edge_index_2024, edge_attr_2024)
        h_2024 = self.output_proj(h_2024)

        # Compute difference
        diff = h_2024 - h_2021

        # Concatenate
        h_full = torch.cat([h_2021, h_2024, diff], dim=1)  # (num_nodes, output_size * 3)

        # Extract batch nodes if needed
        if node_indices is not None:
            h_batch = h_full[node_indices]
        else:
            h_batch = h_full

        return h_batch


class SimplifiedMultiScaleTemporal(nn.Module):
    """
    Simplified multi-scale temporal branch matching checkpoint structure:
    temporal_branch.lstm_hourly, lstm_daily, proj_hourly, proj_daily, weekly_net, fusion
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 256,
                 lstm_hidden: int = 128, dropout: float = 0.4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_hidden = lstm_hidden

        # Hourly LSTM (processes 168 time steps)
        self.lstm_hourly = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Daily LSTM (processes 7 time steps - one per day)
        self.lstm_daily = nn.LSTM(
            input_size=input_size * 24,  # 24 hours per day
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Projections
        self.proj_hourly = nn.Linear(lstm_hidden, hidden_size)
        self.proj_daily = nn.Linear(lstm_hidden, hidden_size)

        # Weekly network (processes aggregated weekly pattern)
        self.weekly_net = nn.Sequential(
            nn.Linear(input_size * 168, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features (batch_size, 168, input_size)

        Returns:
            h: (batch_size, hidden_size)
        """
        batch_size = x.size(0)

        # Hourly processing
        _, (h_hourly, _) = self.lstm_hourly(x)
        h_hourly = h_hourly[-1]  # (batch_size, lstm_hidden)
        h_hourly_proj = self.proj_hourly(h_hourly)  # (batch_size, hidden_size)

        # Daily processing - reshape to (batch_size, 7, 24*input_size)
        x_daily = x.view(batch_size, 7, -1)
        _, (h_daily, _) = self.lstm_daily(x_daily)
        h_daily = h_daily[-1]
        h_daily_proj = self.proj_daily(h_daily)

        # Weekly processing
        x_weekly = x.view(batch_size, -1)  # (batch_size, 168*input_size)
        h_weekly = self.weekly_net(x_weekly)

        # Concatenate and fuse
        h_concat = torch.cat([h_hourly_proj, h_daily_proj, h_weekly], dim=1)
        h_out = self.fusion(h_concat)

        return h_out


class GatedFeatureFusion(nn.Module):
    """
    Gated fusion mechanism matching checkpoint structure:
    fusion.gate_network, fusion.layer_norm, fusion.residual_proj, fusion.transform_network
    """

    def __init__(self, feature_size: int = 256, num_features: int = 6, dropout: float = 0.4):
        super().__init__()

        self.feature_size = feature_size
        self.num_features = num_features

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(feature_size * num_features, feature_size),
            nn.Sigmoid()
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(feature_size)

        # Residual projection
        self.residual_proj = nn.Linear(feature_size * num_features, feature_size)

        # Transform network
        self.transform_network = nn.Sequential(
            nn.Linear(feature_size * num_features, feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, features):
        """
        Forward pass

        Args:
            features: (batch_size, num_features, feature_size)

        Returns:
            h: (batch_size, feature_size)
        """
        batch_size = features.size(0)

        # Flatten
        features_flat = features.view(batch_size, -1)  # (batch_size, num_features * feature_size)

        # Compute gate
        gate = self.gate_network(features_flat)  # (batch_size, feature_size)

        # Residual
        residual = self.residual_proj(features_flat)

        # Transform
        transformed = self.transform_network(features_flat)

        # Apply gate and add residual
        h_out = gate * transformed + residual

        # Layer norm
        h_out = self.layer_norm(h_out)

        return h_out


class EnhancedDualBranchModel_v2(nn.Module):
    """
    Enhanced Dual-Branch Model matching checkpoint structure.
    """

    def __init__(self, temporal_input_size: int = 1, hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES, num_time_steps: int = 168,
                 dropout: float = 0.4):
        super().__init__()

        self.temporal_input_size = temporal_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps

        # Temporal branch
        self.temporal_branch = SimplifiedMultiScaleTemporal(
            input_size=temporal_input_size,
            hidden_size=hidden_size,
            lstm_hidden=128,
            dropout=dropout
        )

        # Spatial branch
        self.spatial_branch = PureGraphDualYearGAT_v2(
            hidden_size=config.GAT_HIDDEN_SIZE,
            num_layers=config.GAT_LAYERS,
            heads=config.GAT_HEADS,
            output_size=hidden_size
        )

        # Fusion
        self.fusion = GatedFeatureFusion(
            feature_size=hidden_size,
            num_features=2,  # 2 temporal features (hourly+daily fusion, weekly)
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
        """
        # Temporal branch
        h_temporal_2021 = self.temporal_branch(x_2021)
        h_temporal_2024 = self.temporal_branch(x_2024)

        # Spatial branch
        h_spatial = self.spatial_branch(graphs_2021, graphs_2024, num_nodes, node_indices)

        # Concatenate for fusion
        features = torch.stack([h_temporal_2021, h_temporal_2024], dim=1)

        # Fusion
        h_fused = self.fusion(features)

        # Classification
        logits = self.classifier(h_fused)

        return logits

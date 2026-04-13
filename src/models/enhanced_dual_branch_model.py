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
from src.models.spatial_branch_pure_graph import (
    PureGraphDualYearGCN,
    PureGraphDualYearSAGE,
    PureGraphDualYearWGCN,
    PureGraphDualYearGAT,
    PureGraphDualYearEvolveGCN,
    PureGraphDualYearMPNN,
)
from src.models.spatial_branch_gine import PureGraphDualYearGINE, compute_laplacian_pe
from src.models.multi_scale_temporal import (
    SimplifiedMultiScaleTemporal,
    SimplifiedMultiScaleTemporalGRU,
    SimplifiedMultiScaleTemporalTCN,
    SimplifiedMultiScaleTemporalTransformer,
    SimplifiedMultiScaleTemporalBiGRU,
    FullMultiScaleTemporalTransformer,
)
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

    RAW_TEMPORAL_GRAPH_STATS_DIM = 10

    def __init__(self,
                 temporal_input_size: int = 2,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 num_time_steps: int = 168,
                 dropout: float = 0.4,
                 spatial_model: str = "GCN",
                 temporal_model: str = "LSTM",
                 branch_ablation_mode: str = "full",
                 fusion_ablation_mode: str = "gated"):
        """
        Initialize enhanced dual-branch model

        Args:
            temporal_input_size: Input size per timestep (default: 2 for [inflow, outflow])
            hidden_size: Hidden feature size (default: 256)
            num_classes: Number of output classes (9)
            num_time_steps: Number of time steps (168 hours)
            dropout: Dropout rate (default: 0.4, increased from 0.2)
            spatial_model: Spatial branch model type ("GCN", "SAGE", "WGCN", "GAT", "EVOLVEGCN", or "GINE")
            temporal_model: Temporal branch model type ("LSTM", "GRU", "TCN", "TRANSFORMER", "TRANSFORMER_FULL", or "BIGRU")
            branch_ablation_mode: Branch ablation setting ("full", "temporal_only", or "spatial_only")
            fusion_ablation_mode: Fusion ablation setting ("gated", "mean", or "concat")
        """
        super(EnhancedDualBranchModel, self).__init__()

        self.temporal_input_size = temporal_input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model.upper()
        self.branch_ablation_mode = branch_ablation_mode
        self.fusion_ablation_mode = fusion_ablation_mode
        self.temporal_subscales = tuple(getattr(config, 'TEMPORAL_SUBSCALES', ('hourly', 'daily', 'weekly')))
        self.spatial_node_feature_mode = getattr(config, 'SPATIAL_NODE_FEATURE_MODE', 'ones')

        if self.branch_ablation_mode not in {'full', 'temporal_only', 'spatial_only'}:
            raise ValueError(
                f"Unsupported branch_ablation_mode={self.branch_ablation_mode}. "
                "Use 'full', 'temporal_only', or 'spatial_only'."
            )

        if self.fusion_ablation_mode not in {'gated', 'mean', 'concat'}:
            raise ValueError(
                f"Unsupported fusion_ablation_mode={self.fusion_ablation_mode}. "
                "Use 'gated', 'mean', or 'concat'."
            )

        if self.spatial_node_feature_mode not in {
            'ones', 'temporal_mean', 'annual_daily_mean', 'annual_daily_mean_2d',
            'raw_temporal_mean', 'raw_temporal_graph_stats', 'flow_wamd',
            'flow_degree_wamd', 'flow_wamd_v2'
        }:
            raise ValueError(
                f"Unsupported SPATIAL_NODE_FEATURE_MODE={self.spatial_node_feature_mode}. "
                "Use 'ones', 'temporal_mean', 'annual_daily_mean', 'annual_daily_mean_2d', "
                "'raw_temporal_mean', 'raw_temporal_graph_stats', 'flow_wamd', "
                "'flow_degree_wamd', or 'flow_wamd_v2'."
            )

        if self.temporal_model not in {'LSTM', 'GRU', 'TCN', 'TRANSFORMER', 'TRANSFORMER_FULL', 'BIGRU'}:
            raise ValueError(
                f"Unsupported temporal_model={self.temporal_model}. "
                "Use 'LSTM', 'GRU', 'TCN', 'TRANSFORMER', 'TRANSFORMER_FULL', or 'BIGRU'."
            )

        if self.temporal_model != 'TRANSFORMER' and self.temporal_subscales != ('hourly', 'daily', 'weekly'):
            raise ValueError(
                "TEMPORAL_SUBSCALES override is currently only supported when "
                "TEMPORAL_MODEL='TRANSFORMER'."
            )

        if self.spatial_node_feature_mode == 'raw_temporal_graph_stats':
            spatial_input_size = self.RAW_TEMPORAL_GRAPH_STATS_DIM
        elif self.spatial_node_feature_mode in {
            'temporal_mean', 'annual_daily_mean_2d', 'raw_temporal_mean', 'flow_wamd'
        }:
            spatial_input_size = temporal_input_size
        elif self.spatial_node_feature_mode == 'flow_degree_wamd':
            # GINE node features: Laplacian PE (8) + 6-dim fdw stats → handled in _build_gine_node_features
            # For non-GINE branches, raw_temporal_mean is used with temporal_input_size dims
            spatial_input_size = temporal_input_size
        else:
            # ones / annual_daily_mean both feed 1-dim node feature to spatial branch.
            spatial_input_size = 1

        # Temporal branch: Multi-scale processing
        _tfm = getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')
        if _tfm == 'total_wamd':
            daily_agg_mode = 'wamd'
        elif _tfm == 'flow_degree_wamd':
            daily_agg_mode = getattr(config, 'DAILY_AGG_MODE', 'sum') + '_fdw'  # 'sum_fdw' or 'mean_fdw'
        else:
            daily_agg_mode = 'sum'
        if self.temporal_model == 'GRU':
            self.temporal_branch = SimplifiedMultiScaleTemporalGRU(
                input_size=temporal_input_size,
                hidden_size=hidden_size,
                gru_hidden=config.LSTM_HIDDEN_SIZE,
                gru_layers=config.LSTM_LAYERS,
                dropout=dropout,
                daily_agg_mode=daily_agg_mode
            )
        elif self.temporal_model == 'TCN':
            self.temporal_branch = SimplifiedMultiScaleTemporalTCN(
                input_size=temporal_input_size,
                hidden_size=hidden_size,
                tcn_hidden=config.LSTM_HIDDEN_SIZE,
                tcn_layers=config.LSTM_LAYERS,
                dropout=dropout,
                daily_agg_mode=daily_agg_mode
            )
        elif self.temporal_model == 'TRANSFORMER':
            self.temporal_branch = SimplifiedMultiScaleTemporalTransformer(
                input_size=temporal_input_size,
                hidden_size=hidden_size,
                model_dim=config.LSTM_HIDDEN_SIZE,
                num_layers=config.LSTM_LAYERS,
                dropout=dropout,
                daily_agg_mode=daily_agg_mode,
                active_scales=self.temporal_subscales,
            )
        elif self.temporal_model == 'TRANSFORMER_FULL':
            self.temporal_branch = FullMultiScaleTemporalTransformer(
                input_size=temporal_input_size,
                hidden_size=hidden_size,
                model_dim=getattr(config, 'TRANSFORMER_FULL_MODEL_DIM', 256),
                hourly_layers=getattr(config, 'TRANSFORMER_FULL_HOURLY_LAYERS', 4),
                daily_layers=getattr(config, 'TRANSFORMER_FULL_DAILY_LAYERS', 3),
                nhead=getattr(config, 'TRANSFORMER_FULL_HEADS', 8),
                ff_multiplier=getattr(config, 'TRANSFORMER_FULL_FF_MULTIPLIER', 4),
                dropout=dropout,
                daily_agg_mode=daily_agg_mode
            )
        elif self.temporal_model == 'BIGRU':
            self.temporal_branch = SimplifiedMultiScaleTemporalBiGRU(
                input_size=temporal_input_size,
                hidden_size=hidden_size,
                gru_hidden=config.LSTM_HIDDEN_SIZE,
                gru_layers=config.LSTM_LAYERS,
                dropout=dropout,
                daily_agg_mode=daily_agg_mode
            )
        else:
            self.temporal_branch = SimplifiedMultiScaleTemporal(
                input_size=temporal_input_size,
                hidden_size=hidden_size,
                lstm_hidden=config.LSTM_HIDDEN_SIZE,
                lstm_layers=config.LSTM_LAYERS,
                dropout=dropout,
                daily_agg_mode=daily_agg_mode
            )

        # Spatial branch: Choose model type
        spatial_model = "GINE" if spatial_model == "GIN" else spatial_model

        self.gine_use_spatial_coords = (
            getattr(config, 'GINE_USE_SPATIAL_COORDS', False)
            and spatial_model in ('GINE', 'GIN')
        )

        if spatial_model == "GINE":
            gine_edge_feature_mode = getattr(config, 'GINE_EDGE_FEATURE_MODE', 'flow_only')
            if gine_edge_feature_mode == 'flow_distance_direction':
                edge_dim = 4
            elif gine_edge_feature_mode == 'flow_distribution':
                edge_dim = 3
            else:
                edge_dim = 1
            gine_input_size = config.LAPLACIAN_PE_DIM
            if self.spatial_node_feature_mode == 'raw_temporal_graph_stats':
                gine_input_size += self.RAW_TEMPORAL_GRAPH_STATS_DIM
            if self.spatial_node_feature_mode == 'raw_temporal_mean':
                gine_input_size += temporal_input_size
            if self.spatial_node_feature_mode == 'flow_wamd':
                gine_input_size += 2
            if self.spatial_node_feature_mode == 'flow_degree_wamd':
                gine_input_size += 6
            if self.spatial_node_feature_mode == 'flow_wamd_v2':
                gine_input_size += 4  # [flow21, wamd21, flow24, wamd24] log-transformed
            if self.gine_use_spatial_coords:
                gine_input_size += 2
            self.spatial_branch = PureGraphDualYearGINE(
                input_size=gine_input_size,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size,
                edge_dim=edge_dim
            )
            self.use_laplacian_pe = True
            self.laplacian_pe_2021 = None
            self.laplacian_pe_2024 = None
        elif spatial_model == "GAT":
            _gat_edge_mode = getattr(config, 'GINE_EDGE_FEATURE_MODE', 'flow_only')
            _gat_edge_dim = 4 if _gat_edge_mode == 'flow_distance_direction' else (3 if _gat_edge_mode == 'flow_distribution' else 1)
            self.spatial_branch = PureGraphDualYearGAT(
                input_size=spatial_input_size,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size,
                heads=1,
                edge_dim=_gat_edge_dim
            )
            self.use_laplacian_pe = False
        elif spatial_model == "SAGE":
            self.spatial_branch = PureGraphDualYearSAGE(
                input_size=spatial_input_size,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size
            )
            self.use_laplacian_pe = False
        elif spatial_model == "WGCN":
            self.spatial_branch = PureGraphDualYearWGCN(
                input_size=spatial_input_size,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size
            )
            self.use_laplacian_pe = False
        elif spatial_model == "EVOLVEGCN":
            self.spatial_branch = PureGraphDualYearEvolveGCN(
                input_size=spatial_input_size,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size
            )
            self.use_laplacian_pe = False
        elif spatial_model == "MPNN":
            _mpnn_edge_mode = getattr(config, 'GINE_EDGE_FEATURE_MODE', 'flow_only')
            _mpnn_edge_dim = 4 if _mpnn_edge_mode == 'flow_distance_direction' else (3 if _mpnn_edge_mode == 'flow_distribution' else 1)
            self.spatial_branch = PureGraphDualYearMPNN(
                input_size=spatial_input_size,
                hidden_size=config.SPATIAL_HIDDEN_SIZE,
                num_layers=config.SPATIAL_LAYERS,
                dropout=dropout,
                output_size=hidden_size,
                edge_dim=_mpnn_edge_dim
            )
            self.use_laplacian_pe = False
        else:  # Default to GCN
            self.spatial_branch = PureGraphDualYearGCN(
                input_size=spatial_input_size,
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

        classifier_input_size = hidden_size * 6 if self.fusion_ablation_mode == 'concat' else hidden_size

        # Single 9-class classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def register_node_coords(self, node_coords: torch.Tensor):
        """Register normalized spatial coordinates as a buffer for GINE node features."""
        mean = node_coords.mean(dim=0, keepdim=True)
        std = node_coords.std(dim=0, keepdim=True).clamp_min(1e-6)
        normalized = (node_coords - mean) / std
        self.register_buffer('_node_coords', normalized)

    def register_wamd_node_features(self, wamd_2021: torch.Tensor, wamd_2024: torch.Tensor):
        """Register precomputed wamd node features as log-transformed buffers.

        Args:
            wamd_2021: (num_nodes, 2) raw [total_w, wamd_w] for 2021
            wamd_2024: (num_nodes, 2) raw [total_w, wamd_w] for 2024
        """
        self.register_buffer('_wamd_node_features_2021', torch.log1p(wamd_2021))
        self.register_buffer('_wamd_node_features_2024', torch.log1p(wamd_2024))

    def register_fdw_node_features(self, fdw_2021: torch.Tensor, fdw_2024: torch.Tensor):
        """Register precomputed flow+degree+wamd node features as log-transformed buffers."""
        self.register_buffer('_fdw_node_features_2021', torch.log1p(fdw_2021))
        self.register_buffer('_fdw_node_features_2024', torch.log1p(fdw_2024))

    def register_flow_wamd_v2_node_features(self, fw2_2021: torch.Tensor, fw2_2024: torch.Tensor):
        """Register precomputed flow+wamd (no degree) node features as log-transformed buffers.

        Args:
            fw2_2021: (num_nodes, 2) raw [flow_w, wamd_w] for 2021
            fw2_2024: (num_nodes, 2) raw [flow_w, wamd_w] for 2024
        """
        self.register_buffer('_fw2_node_features_2021', torch.log1p(fw2_2021))
        self.register_buffer('_fw2_node_features_2024', torch.log1p(fw2_2024))

    @staticmethod
    def _mode_requires_raw_temporal(node_feature_mode: str) -> bool:
        return node_feature_mode in {'raw_temporal_mean', 'raw_temporal_graph_stats'}

    @staticmethod
    def _zscore_node_features(node_features: torch.Tensor) -> torch.Tensor:
        feature_mean = node_features.mean(dim=0, keepdim=True)
        feature_std = node_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (node_features - feature_mean) / feature_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _extract_primary_edge_weight(edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_edges = edge_index.size(1)

        if edge_attr is None:
            return torch.ones(num_edges, device=edge_index.device, dtype=torch.float32)

        edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            return edge_attr
        if edge_attr.dim() == 2:
            if edge_attr.size(0) == num_edges:
                return edge_attr[:, 0]
            if edge_attr.size(1) == num_edges:
                return edge_attr[0]

        return edge_attr.reshape(num_edges, -1)[:, 0]

    def _compute_raw_temporal_graph_stats(self, raw_x: torch.Tensor, graph, num_nodes: int) -> torch.Tensor:
        if raw_x is None:
            raise ValueError("raw_temporal_graph_stats requires raw_x_2021/raw_x_2024 tensors")

        hourly_mean = torch.log1p(raw_x.mean(dim=1))
        hourly_peak = torch.log1p(raw_x.max(dim=1).values)
        coeff_var = raw_x.std(dim=1, unbiased=False) / raw_x.mean(dim=1).clamp_min(1.0)

        edge_index, edge_attr = graph[0]
        edge_index = edge_index.long()
        edge_weight = self._extract_primary_edge_weight(edge_attr, edge_index)

        src = edge_index[0]
        dst = edge_index[1]

        out_degree = torch.bincount(src, minlength=num_nodes).float()
        in_degree = torch.bincount(dst, minlength=num_nodes).float()

        weighted_out = torch.zeros(num_nodes, device=edge_weight.device, dtype=torch.float32)
        weighted_in = torch.zeros(num_nodes, device=edge_weight.device, dtype=torch.float32)
        weighted_out.scatter_add_(0, src, edge_weight)
        weighted_in.scatter_add_(0, dst, edge_weight)

        avg_out_weight = weighted_out / out_degree.clamp_min(1.0)
        avg_in_weight = weighted_in / in_degree.clamp_min(1.0)

        graph_stats = torch.stack(
            [
                torch.log1p(out_degree),
                torch.log1p(in_degree),
                torch.log1p(avg_out_weight),
                torch.log1p(avg_in_weight),
            ],
            dim=1,
        )

        node_features = torch.cat([hourly_mean, hourly_peak, coeff_var, graph_stats], dim=1)
        return self._zscore_node_features(node_features)

    def _build_non_gine_node_features(self, x_2021, x_2024, raw_x_2021, raw_x_2024, graphs_2021, graphs_2024, num_nodes):
        spatial_node_features_2021 = None
        spatial_node_features_2024 = None

        if self.spatial_node_feature_mode == 'temporal_mean':
            spatial_node_features_2021 = x_2021.mean(dim=1)
            spatial_node_features_2024 = x_2024.mean(dim=1)
        elif self.spatial_node_feature_mode == 'annual_daily_mean':
            daily_total_2021 = x_2021.sum(dim=2).view(x_2021.shape[0], 7, 24).sum(dim=2)
            daily_total_2024 = x_2024.sum(dim=2).view(x_2024.shape[0], 7, 24).sum(dim=2)
            spatial_node_features_2021 = daily_total_2021.mean(dim=1, keepdim=True)
            spatial_node_features_2024 = daily_total_2024.mean(dim=1, keepdim=True)
        elif self.spatial_node_feature_mode == 'annual_daily_mean_2d':
            daily_inout_2021 = x_2021.view(x_2021.shape[0], 7, 24, x_2021.shape[2]).sum(dim=2)
            daily_inout_2024 = x_2024.view(x_2024.shape[0], 7, 24, x_2024.shape[2]).sum(dim=2)
            spatial_node_features_2021 = daily_inout_2021.mean(dim=1)
            spatial_node_features_2024 = daily_inout_2024.mean(dim=1)
        elif self.spatial_node_feature_mode == 'raw_temporal_mean':
            if raw_x_2021 is None or raw_x_2024 is None:
                raise ValueError("raw_temporal_mean requires raw_x_2021/raw_x_2024 tensors")
            spatial_node_features_2021 = raw_x_2021.mean(dim=1)
            spatial_node_features_2024 = raw_x_2024.mean(dim=1)
        elif self.spatial_node_feature_mode == 'raw_temporal_graph_stats':
            spatial_node_features_2021 = self._compute_raw_temporal_graph_stats(raw_x_2021, graphs_2021, num_nodes)
            spatial_node_features_2024 = self._compute_raw_temporal_graph_stats(raw_x_2024, graphs_2024, num_nodes)
        elif self.spatial_node_feature_mode == 'flow_wamd':
            if not hasattr(self, '_wamd_node_features_2021'):
                raise ValueError("flow_wamd mode requires register_wamd_node_features() to be called first")
            spatial_node_features_2021 = self._wamd_node_features_2021
            spatial_node_features_2024 = self._wamd_node_features_2024

        return spatial_node_features_2021, spatial_node_features_2024

    def _build_gine_node_features(self, graphs_2021, graphs_2024, num_nodes, raw_x_2021, raw_x_2024, device):
        self.compute_laplacian_pe_if_needed(graphs_2021, graphs_2024, num_nodes, device)

        features_2021 = self.laplacian_pe_2021
        features_2024 = self.laplacian_pe_2024

        if self.spatial_node_feature_mode == 'raw_temporal_graph_stats':
            raw_graph_stats_2021 = self._compute_raw_temporal_graph_stats(raw_x_2021, graphs_2021, num_nodes)
            raw_graph_stats_2024 = self._compute_raw_temporal_graph_stats(raw_x_2024, graphs_2024, num_nodes)
            features_2021 = torch.cat([features_2021, raw_graph_stats_2021], dim=1)
            features_2024 = torch.cat([features_2024, raw_graph_stats_2024], dim=1)
        elif self.spatial_node_feature_mode == 'raw_temporal_mean':
            # Use the flow change (delta) rather than absolute yearly means.
            # This separates responsibilities: temporal branch handles absolute patterns
            # via the full 168-step sequence; spatial branch propagates the change signal
            # through the graph. Both features_2021 and features_2024 receive the same
            # delta_mean so year differentiation falls on LapPE and edge OD flows.
            raw_mean_2021 = torch.log1p(raw_x_2021.mean(dim=1))  # (N, 2)
            raw_mean_2024 = torch.log1p(raw_x_2024.mean(dim=1))  # (N, 2)
            delta_mean = raw_mean_2024 - raw_mean_2021             # (N, 2)
            features_2021 = torch.cat([features_2021, delta_mean], dim=1)
            features_2024 = torch.cat([features_2024, delta_mean], dim=1)
        elif self.spatial_node_feature_mode == 'flow_wamd':
            if not hasattr(self, '_wamd_node_features_2021'):
                raise ValueError("flow_wamd mode requires register_wamd_node_features() to be called first")
            features_2021 = torch.cat([features_2021, self._wamd_node_features_2021.to(device)], dim=1)
            features_2024 = torch.cat([features_2024, self._wamd_node_features_2024.to(device)], dim=1)
        elif self.spatial_node_feature_mode == 'flow_degree_wamd':
            if not hasattr(self, '_fdw_node_features_2021'):
                raise ValueError("flow_degree_wamd mode requires register_fdw_node_features() to be called first")
            features_2021 = torch.cat([features_2021, self._fdw_node_features_2021.to(device)], dim=1)
            features_2024 = torch.cat([features_2024, self._fdw_node_features_2024.to(device)], dim=1)
        elif self.spatial_node_feature_mode == 'flow_wamd_v2':
            if not hasattr(self, '_fw2_node_features_2021'):
                raise ValueError("flow_wamd_v2 mode requires register_flow_wamd_v2_node_features() to be called first")
            features_2021 = torch.cat([features_2021, self._fw2_node_features_2021.to(device)], dim=1)
            features_2024 = torch.cat([features_2024, self._fw2_node_features_2024.to(device)], dim=1)

        if self.gine_use_spatial_coords and hasattr(self, '_node_coords'):
            coords = self._node_coords.to(device)
            features_2021 = torch.cat([features_2021, coords], dim=1)
            features_2024 = torch.cat([features_2024, coords], dim=1)

        return features_2021, features_2024

    def compute_laplacian_pe_if_needed(self, graphs_2021, graphs_2024, num_nodes, device):
        """Compute Laplacian PE for GINE model if not already computed.

        Results are cached to disk keyed by (num_nodes, topk_out, topk_in, pe_dim) so
        subsequent experiments with the same graph topology skip the expensive eigsh call.
        Cache lives in outputs/.laplacian_pe_cache/.
        """
        if not (self.use_laplacian_pe and self.laplacian_pe_2021 is None):
            return

        import os
        k = config.LAPLACIAN_PE_DIM
        topk_out = getattr(config, 'GRAPH_TOPK_OUT', 'none')
        topk_in  = getattr(config, 'GRAPH_TOPK_IN',  'none')
        cache_dir = "outputs/.laplacian_pe_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir,
            f"lpe_n{num_nodes}_k{k}_topk{topk_out}x{topk_in}.pt"
        )

        if os.path.exists(cache_path):
            print(f"[LapPE] Loading from cache: {cache_path}")
            cached = torch.load(cache_path, map_location='cpu')
            self.laplacian_pe_2021 = cached['pe_2021'].to(device)
            self.laplacian_pe_2024 = cached['pe_2024'].to(device)
            print(f"[LapPE] Loaded: {self.laplacian_pe_2021.shape}")
            return

        edge_index_2021, edge_attr_2021 = graphs_2021[0]
        edge_index_2024, edge_attr_2024 = graphs_2024[0]

        self.laplacian_pe_2021 = compute_laplacian_pe(
            edge_index_2021,
            edge_attr_2021.squeeze(),
            num_nodes,
            k=k,
            device=device
        )
        self.laplacian_pe_2024 = compute_laplacian_pe(
            edge_index_2024,
            edge_attr_2024.squeeze(),
            num_nodes,
            k=k,
            device=device
        )

        torch.save(
            {'pe_2021': self.laplacian_pe_2021.cpu(), 'pe_2024': self.laplacian_pe_2024.cpu()},
            cache_path
        )
        print(f"[LapPE] Saved cache: {cache_path}")

    def forward(self, x_2021, x_2024, graphs_2021, graphs_2024, num_nodes, node_indices=None,
                raw_x_2021=None, raw_x_2024=None):
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

        # Extract sub-scale features for both years before fusion.
        # This lets us compute temporal_diff at the intermediate representation level
        # (per time-scale delta) rather than subtracting nearly-identical fused vectors.
        h_hourly_21, h_daily_21, h_weekly_21 = self.temporal_branch.extract_subscale_features(x_2021_batch)
        h_hourly_24, h_daily_24, h_weekly_24 = self.temporal_branch.extract_subscale_features(x_2024_batch)

        temporal_2021 = self.temporal_branch.fusion(
            torch.cat([h_hourly_21, h_daily_21, h_weekly_21], dim=1)
        )
        temporal_2024 = self.temporal_branch.fusion(
            torch.cat([h_hourly_24, h_daily_24, h_weekly_24], dim=1)
        )
        # diff_fusion has independent weights from fusion to avoid gradient conflict
        # between absolute-pattern and change-pattern objectives.
        temporal_diff = self.temporal_branch.diff_fusion(
            torch.cat([
                h_hourly_24 - h_hourly_21,
                h_daily_24  - h_daily_21,
                h_weekly_24 - h_weekly_21,
            ], dim=1)
        )

        # Stack temporal features: (batch_size, 3, hidden_size)
        temporal_features = torch.stack([temporal_2021, temporal_2024, temporal_diff], dim=1)

        if self.branch_ablation_mode == 'spatial_only':
            temporal_features = temporal_features * 0.0

        if self._mode_requires_raw_temporal(self.spatial_node_feature_mode):
            if raw_x_2021 is None or raw_x_2024 is None:
                raise ValueError(
                    f"{self.spatial_node_feature_mode} requires raw_x_2021/raw_x_2024 tensors"
                )

        # Extract spatial features
        if self.branch_ablation_mode == 'temporal_only':
            spatial_features = temporal_features * 0.0
        elif self.use_laplacian_pe:
            node_features_2021, node_features_2024 = self._build_gine_node_features(
                graphs_2021=graphs_2021,
                graphs_2024=graphs_2024,
                num_nodes=num_nodes,
                raw_x_2021=raw_x_2021,
                raw_x_2024=raw_x_2024,
                device=x_2021.device,
            )

            spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
                graphs_2021=graphs_2021,
                graphs_2024=graphs_2024,
                node_features_2021=node_features_2021,
                node_features_2024=node_features_2024,
                node_indices=node_indices
            )

            # Stack spatial features: (batch_size, 3, hidden_size)
            spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)
        else:
            spatial_node_features_2021, spatial_node_features_2024 = self._build_non_gine_node_features(
                x_2021=x_2021,
                x_2024=x_2024,
                raw_x_2021=raw_x_2021,
                raw_x_2024=raw_x_2024,
                graphs_2021=graphs_2021,
                graphs_2024=graphs_2024,
                num_nodes=num_nodes,
            )

            spatial_2021, spatial_2024, spatial_diff = self.spatial_branch(
                graphs_2021=graphs_2021,
                graphs_2024=graphs_2024,
                num_nodes=num_nodes,
                node_indices=node_indices,
                node_features_2021=spatial_node_features_2021,
                node_features_2024=spatial_node_features_2024
            )

            # Stack spatial features: (batch_size, 3, hidden_size)
            spatial_features = torch.stack([spatial_2021, spatial_2024, spatial_diff], dim=1)

        # Concatenate temporal and spatial features: (batch_size, 6, hidden_size)
        all_features = torch.cat([temporal_features, spatial_features], dim=1)

        if self.fusion_ablation_mode == 'concat':
            fused = all_features.reshape(all_features.size(0), -1)
        elif self.fusion_ablation_mode == 'mean':
            fused = all_features.mean(dim=1)
        else:
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

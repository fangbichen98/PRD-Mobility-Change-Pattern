"""
Ablation study: Test model without SPP or DySAT components
"""
import torch
import logging
from src.models.dual_branch_model import DualBranchSTModel
from src.models.temporal_branch import TemporalBranch
from src.models.spatial_branch import SpatialBranch
import torch.nn as nn
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualBranchNoSPP(nn.Module):
    """Dual-branch model without SPP (only LSTM)"""

    def __init__(self,
                 temporal_input_size: int = 2,
                 spatial_input_size: int = 336,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.2):
        super(DualBranchNoSPP, self).__init__()

        # Temporal branch: LSTM only (no SPP)
        self.lstm = nn.LSTM(
            input_size=temporal_input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            dropout=dropout if config.LSTM_LAYERS > 1 else 0
        )

        self.temporal_proj = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Spatial branch (with DySAT)
        self.spatial_branch = SpatialBranch(
            input_size=spatial_input_size,
            output_size=hidden_size
        )

        # Fusion
        from src.models.dual_branch_model import AttentionFusion
        self.fusion = AttentionFusion(feature_size=hidden_size, dropout=dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, temporal_input, spatial_input, edge_index, edge_attr=None):
        # Temporal: LSTM only
        _, (h_n, _) = self.lstm(temporal_input)
        temporal_features = self.temporal_proj(h_n[-1])

        # Spatial: DySAT
        spatial_features = self.spatial_branch(spatial_input, edge_index, edge_attr)

        # Fuse and classify
        fused = self.fusion(temporal_features, spatial_features)
        logits = self.classifier(fused)

        return logits


class DualBranchNoDySAT(nn.Module):
    """Dual-branch model without DySAT (simple GCN)"""

    def __init__(self,
                 temporal_input_size: int = 2,
                 spatial_input_size: int = 336,
                 hidden_size: int = 256,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.2):
        super(DualBranchNoDySAT, self).__init__()

        # Temporal branch (with LSTM + SPP)
        self.temporal_branch = TemporalBranch(
            input_size=temporal_input_size,
            output_size=hidden_size
        )

        # Spatial branch: Simple GCN (no DySAT)
        from torch_geometric.nn import GCNConv

        self.gcn1 = GCNConv(spatial_input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion
        from src.models.dual_branch_model import AttentionFusion
        self.fusion = AttentionFusion(feature_size=hidden_size, dropout=dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, temporal_input, spatial_input, edge_index, edge_attr=None):
        # Temporal: LSTM + SPP
        temporal_features = self.temporal_branch(temporal_input)

        # Spatial: Simple GCN
        x = self.gcn1(spatial_input, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        spatial_features = self.spatial_proj(x)

        # Fuse and classify
        fused = self.fusion(temporal_features, spatial_features)
        logits = self.classifier(fused)

        return logits


def run_ablation_study(data, device='cuda'):
    """
    Run ablation study

    Args:
        data: Prepared data dictionary
        device: Device to use

    Returns:
        Results for ablation models
    """
    from train import train_model

    logger.info("=" * 50)
    logger.info("Ablation Study")
    logger.info("=" * 50)

    results = {}

    # Test without SPP
    logger.info("\nAblation 1: Without SPP")
    model_no_spp, results_no_spp, _ = train_model(data, 'dual_branch_no_spp', device)
    results['No SPP'] = results_no_spp

    # Test without DySAT
    logger.info("\nAblation 2: Without DySAT")
    model_no_dysat, results_no_dysat, _ = train_model(data, 'dual_branch_no_dysat', device)
    results['No DySAT'] = results_no_dysat

    return results


if __name__ == "__main__":
    from train import prepare_data

    # Prepare data
    data = prepare_data(year=2021, sample_size=1000000)

    # Run ablation study
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = run_ablation_study(data, device)

    # Print results
    logger.info("\nAblation Study Results:")
    logger.info("-" * 50)
    for model_name, res in results.items():
        logger.info(f"{model_name}: Acc={res['accuracy']:.4f}, F1={res['f1_macro']:.4f}")

"""
Training script with Multi-Scale Temporal Branch improvement.

Expected improvement: +3-5% accuracy (69% → 72-74%)
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from datetime import datetime
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F

import torch.nn.functional as F

import config
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Dynamically down-weights easy examples so training focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter. 0 = standard cross-entropy. Default: 2.
        weight: per-class weights tensor (same as CrossEntropyLoss weight). Default: None.
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        # ce shape: (N,)  — per-sample loss before reduction
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)                         # probability of the correct class
        focal = ((1.0 - pt) ** self.gamma) * ce     # down-weight easy examples
        return focal.mean()


# Import Enhanced model with Multi-Scale Temporal Branch
from src.models.enhanced_dual_branch_model import EnhancedDualBranchModel
from src.training.dataset_pure_graph import PureGraphDualYearDataset, PureGraphBatchCollator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


VALID_TEMPORAL_SUBSCALES = ('hourly', 'daily', 'weekly')


def parse_temporal_subscales(raw_value):
    """Parse comma-separated temporal subscales into a normalized tuple."""
    if raw_value is None:
        return tuple(getattr(config, 'TEMPORAL_SUBSCALES', VALID_TEMPORAL_SUBSCALES))

    normalized = []
    for token in str(raw_value).split(','):
        scale = token.strip().lower()
        if not scale:
            continue
        if scale not in VALID_TEMPORAL_SUBSCALES:
            raise ValueError(
                f"Unsupported temporal subscale '{scale}'. Use one of {VALID_TEMPORAL_SUBSCALES}."
            )
        if scale not in normalized:
            normalized.append(scale)

    if not normalized:
        raise ValueError("At least one temporal subscale must remain enabled.")

    return tuple(normalized)


def describe_temporal_subscales(scales):
    """Human-readable description for active temporal subscales."""
    normalized = tuple(scales)
    if normalized == ('hourly',):
        return 'Single-scale (168h hourly only)'
    return ' + '.join(scale.capitalize() for scale in normalized)


def parse_args():
    """Parse optional runtime overrides for controlled experiment sweeps."""
    parser = argparse.ArgumentParser(description="Train Multi-Scale Temporal + Graph model")
    parser.add_argument('--label-path', type=str, default=None,
                        help='Override label file path')
    parser.add_argument('--split-manifest', type=str, default=None,
                        help='Load a fixed train/val/test split manifest JSON')
    parser.add_argument('--export-split-manifest', type=str, default=None,
                        help='Export the resolved train/val/test split manifest JSON')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Optional per-class sampling for small-scale experiments')
    parser.add_argument('--spatial-model', type=str, choices=['GCN', 'SAGE', 'WGCN', 'GINE', 'GIN', 'GAT', 'EVOLVEGCN', 'MPNN'], default=None,
                        help='Override spatial branch model')
    parser.add_argument('--temporal-model', type=str, choices=['LSTM', 'GRU', 'TCN', 'TRANSFORMER', 'TRANSFORMER_FULL', 'BIGRU'], default=None,
                        help='Override temporal branch model')
    parser.add_argument('--temporal-layers', type=int, default=None,
                        help='Override temporal branch layer count for lightweight temporal models')
    parser.add_argument('--temporal-subscales', type=str, default=None,
                        help='Comma-separated temporal subscales for the light Transformer branch, e.g. hourly or hourly,daily,weekly')
    parser.add_argument('--no-temporal-log1p', action='store_true', default=False,
                        help='Disable log1p pre-processing on temporal branch inputs; use raw flow values instead')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='Override max number of epochs')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Override early stopping patience')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=None,
                        help='Override gradient accumulation steps')
    parser.add_argument('--scheduler-patience', type=int, default=None,
                        help='Override ReduceLROnPlateau patience')
    parser.add_argument('--scheduler-factor', type=float, default=None,
                        help='Override ReduceLROnPlateau factor')
    parser.add_argument('--run-tag', type=str, default=None,
                        help='Optional suffix for output directory naming')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable cached preprocessing data')
    parser.add_argument('--graph-topk-out', type=int, default=None,
                        help='Keep top-k outgoing edges per node in graph construction')
    parser.add_argument('--graph-topk-in', type=int, default=None,
                        help='Keep top-k incoming edges per node in graph construction')
    parser.add_argument('--graph-temporal-mode', type=str, choices=['static', 'daily'], default=None,
                        help='Graph temporal mode: static or daily discrete snapshots')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Override random seed for data sampling/splitting')
    parser.add_argument('--spatial-node-feature-mode', type=str,
                        choices=['ones', 'temporal_mean', 'annual_daily_mean', 'annual_daily_mean_2d',
                                 'raw_temporal_mean', 'raw_temporal_graph_stats', 'flow_wamd',
                                 'flow_degree_wamd', 'flow_wamd_v2'],
                        default=None,
                        help='Spatial node feature mode for non-GINE branches')
    parser.add_argument('--gine-edge-feature-mode', type=str,
                        choices=['flow_only', 'flow_distance_direction', 'flow_distribution'],
                        default=None,
                        help='Edge feature mode for GINE: flow only or flow+distance+direction')
    parser.add_argument('--gine-use-spatial-coords', action='store_true', default=None,
                        help='Concatenate normalized (lon, lat) into GINE node features')
    parser.add_argument('--no-gine-spatial-coords', action='store_true', default=False,
                        help='Disable spatial coords in GINE node features (overrides config)')
    parser.add_argument('--temporal-feature-mode', type=str,
                        choices=['inflow_outflow', 'total_wamd', 'flow_degree_wamd', 'flow_wamd_v2'], default=None,
                        help='Temporal feature mode')
    parser.add_argument('--daily-agg-mode', type=str, choices=['sum', 'mean'], default=None,
                        help='Daily aggregation mode for flow_degree_wamd: sum or mean (default: sum)')
    parser.add_argument('--branch-ablation-mode', type=str, choices=['full', 'temporal_only', 'spatial_only'], default='full',
                        help='Ablate temporal/spatial branches while keeping the same training pipeline')
    parser.add_argument('--fusion-ablation-mode', type=str, choices=['gated', 'mean', 'concat'], default='gated',
                        help='Ablate fusion mechanism by replacing gated fusion with simple mean pooling or concat')
    # Spatial architecture overrides
    parser.add_argument('--spatial-layers', type=int, default=None,
                        help='Override config.SPATIAL_LAYERS (number of GCN/GINE layers)')
    parser.add_argument('--spatial-hidden-size', type=int, default=None,
                        help='Override config.SPATIAL_HIDDEN_SIZE (hidden units per spatial layer)')
    parser.add_argument('--laplacian-pe-dim', type=int, default=None,
                        help='Override config.LAPLACIAN_PE_DIM (Laplacian PE dimension for GINE; 0 = disable PE)')
    # Loss function
    parser.add_argument('--focal-loss-gamma', type=float, default=None,
                        help='Use Focal Loss with this gamma value instead of CrossEntropy. '
                             'Recommended: 2.0. None (default) = standard weighted CrossEntropy.')
    return parser.parse_args()


def write_split_manifest(manifest_path, manifest_payload):
    """Write a split manifest JSON to disk."""
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest_payload, f, indent=2)


def build_split_manifest_payload(label_path, random_seed, dataset_grid_ids, split_indices):
    """Build a portable split manifest using grid IDs instead of dataset indices."""
    return {
        'label_path': label_path,
        'random_seed': int(random_seed),
        'num_samples': len(dataset_grid_ids),
        'splits': {
            split_name: [int(dataset_grid_ids[idx]) for idx in indices]
            for split_name, indices in split_indices.items()
        }
    }


def resolve_dataset_splits(dataset, label_path, split_manifest_path, export_manifest_paths, random_seed):
    """Create or load deterministic dataset splits and optionally export the manifest."""
    dataset_grid_ids = [int(grid_id) for grid_id in dataset.grid_ids]

    if split_manifest_path:
        with open(split_manifest_path, 'r') as f:
            split_manifest = json.load(f)

        split_grid_ids = split_manifest.get('splits', split_manifest)
        required_splits = ('train', 'val', 'test')
        missing_splits = [split_name for split_name in required_splits if split_name not in split_grid_ids]
        if missing_splits:
            raise ValueError(
                f"Split manifest missing required keys: {missing_splits}"
            )

        grid_id_to_index = {grid_id: idx for idx, grid_id in enumerate(dataset_grid_ids)}
        split_indices = {}
        used_grid_ids = []

        for split_name in required_splits:
            indices = []
            for grid_id in split_grid_ids[split_name]:
                normalized_grid_id = int(grid_id)
                if normalized_grid_id not in grid_id_to_index:
                    raise ValueError(
                        f"Grid ID {normalized_grid_id} in split '{split_name}' not found in current dataset"
                    )
                indices.append(grid_id_to_index[normalized_grid_id])
                used_grid_ids.append(normalized_grid_id)
            split_indices[split_name] = indices

        if len(set(used_grid_ids)) != len(used_grid_ids):
            raise ValueError("Split manifest contains duplicate grid IDs across train/val/test")

        missing_grid_ids = sorted(set(dataset_grid_ids) - set(used_grid_ids))
        if missing_grid_ids:
            raise ValueError(
                f"Split manifest does not cover all dataset samples; missing {len(missing_grid_ids)} grid IDs"
            )

        split_source = 'manifest'
        source_manifest_path = split_manifest_path
    else:
        train_size = int(config.TRAIN_SPLIT * len(dataset))
        val_size = int(config.VAL_SPLIT * len(dataset))
        test_size = len(dataset) - train_size - val_size

        shuffled_indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(random_seed)
        ).tolist()

        split_indices = {
            'train': shuffled_indices[:train_size],
            'val': shuffled_indices[train_size:train_size + val_size],
            'test': shuffled_indices[train_size + val_size:train_size + val_size + test_size],
        }
        split_source = 'seeded_random_split'
        source_manifest_path = None

    split_manifest_payload = build_split_manifest_payload(
        label_path=label_path,
        random_seed=random_seed,
        dataset_grid_ids=dataset_grid_ids,
        split_indices=split_indices
    )

    written_manifest_paths = []
    for export_manifest_path in export_manifest_paths:
        if export_manifest_path is None:
            continue
        write_split_manifest(export_manifest_path, split_manifest_payload)
        written_manifest_paths.append(export_manifest_path)

    split_subsets = {
        split_name: Subset(dataset, indices)
        for split_name, indices in split_indices.items()
    }
    split_counts = {split_name: len(indices) for split_name, indices in split_indices.items()}

    split_metadata = {
        'source': split_source,
        'source_manifest_path': source_manifest_path,
        'written_manifest_paths': written_manifest_paths,
        'counts': split_counts,
    }

    return split_subsets, split_metadata


def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=4, grad_clip_norm=1.0):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    correct = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        node_indices = batch['node_indices'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']

        # Get full temporal features
        all_temporal_2021 = batch['all_temporal_2021'].to(device)
        all_temporal_2024 = batch['all_temporal_2024'].to(device)
        all_raw_temporal_2021 = batch['all_raw_temporal_2021']
        all_raw_temporal_2024 = batch['all_raw_temporal_2024']
        if all_raw_temporal_2021 is not None:
            all_raw_temporal_2021 = all_raw_temporal_2021.to(device)
        if all_raw_temporal_2024 is not None:
            all_raw_temporal_2024 = all_raw_temporal_2024.to(device)

        # Move graphs to device
        graphs_2021 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2024']]

        # Forward pass
        logits = model(
            x_2021=all_temporal_2021,
            x_2024=all_temporal_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices,
            raw_x_2021=all_raw_temporal_2021,
            raw_x_2024=all_raw_temporal_2024
        )

        # Compute loss
        loss = criterion(logits, labels)
        normalized_loss = loss / accumulation_steps

        # Backward pass with gradient accumulation
        normalized_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Compute predictions
        pred = logits.argmax(dim=1)

        # Update metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct += (pred == labels).sum().item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Epoch [{batch_idx + 1}/{len(train_loader)}] "
                       f"Loss: {loss.item():.4f}")

    # Final gradient update if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    # Compute average metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_samples = 0

    # Collect predictions and labels
    all_preds = []
    all_labels = []

    for batch in data_loader:
        # Move data to device
        node_indices = batch['node_indices'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']

        # Get full temporal features
        all_temporal_2021 = batch['all_temporal_2021'].to(device)
        all_temporal_2024 = batch['all_temporal_2024'].to(device)
        all_raw_temporal_2021 = batch['all_raw_temporal_2021']
        all_raw_temporal_2024 = batch['all_raw_temporal_2024']
        if all_raw_temporal_2021 is not None:
            all_raw_temporal_2021 = all_raw_temporal_2021.to(device)
        if all_raw_temporal_2024 is not None:
            all_raw_temporal_2024 = all_raw_temporal_2024.to(device)

        # Move graphs to device
        graphs_2021 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2024']]

        # Forward pass
        logits = model(
            x_2021=all_temporal_2021,
            x_2024=all_temporal_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices,
            raw_x_2021=all_raw_temporal_2021,
            raw_x_2024=all_raw_temporal_2024
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Update metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute predictions
        pred = logits.argmax(dim=1)

        # Collect predictions
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * (all_preds == all_labels).sum() / total_samples
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'all_labels': all_labels,
        'all_preds': all_preds
    }


def main():
    """Main training function with Multi-Scale Temporal Branch"""
    args = parse_args()

    label_path = args.label_path if args.label_path else config.LABEL_PATH
    samples_per_class = args.samples_per_class
    spatial_model = args.spatial_model if args.spatial_model else getattr(config, 'SPATIAL_MODEL', 'GCN')
    temporal_model = args.temporal_model if args.temporal_model else getattr(config, 'TEMPORAL_MODEL', 'LSTM')
    temporal_model = temporal_model.upper()
    if spatial_model == 'GIN':
        spatial_model = 'GINE'
    num_epochs = args.num_epochs if args.num_epochs is not None else config.NUM_EPOCHS
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else config.EARLY_STOPPING_PATIENCE
    )
    learning_rate = args.learning_rate if args.learning_rate is not None else config.LEARNING_RATE
    batch_size = args.batch_size if args.batch_size is not None else config.BATCH_SIZE
    gradient_accumulation = (
        args.gradient_accumulation
        if args.gradient_accumulation is not None
        else config.GRADIENT_ACCUMULATION
    )
    scheduler_patience = (
        args.scheduler_patience
        if args.scheduler_patience is not None
        else config.SCHEDULER_PATIENCE
    )
    scheduler_factor = (
        args.scheduler_factor
        if args.scheduler_factor is not None
        else config.SCHEDULER_FACTOR
    )
    use_cache = not args.no_cache

    # Runtime graph sparsification overrides (direction 2).
    if args.graph_topk_out is not None:
        config.GRAPH_TOPK_OUT = args.graph_topk_out
    if args.graph_topk_in is not None:
        config.GRAPH_TOPK_IN = args.graph_topk_in
    if args.random_seed is not None:
        config.RANDOM_SEED = args.random_seed
    if args.spatial_node_feature_mode is not None:
        config.SPATIAL_NODE_FEATURE_MODE = args.spatial_node_feature_mode
    if args.gine_edge_feature_mode is not None:
        config.GINE_EDGE_FEATURE_MODE = args.gine_edge_feature_mode
    if args.gine_use_spatial_coords:
        config.GINE_USE_SPATIAL_COORDS = True
    if args.no_gine_spatial_coords:
        config.GINE_USE_SPATIAL_COORDS = False
    if args.temporal_feature_mode is not None:
        config.TEMPORAL_FEATURE_MODE = args.temporal_feature_mode
    if args.daily_agg_mode is not None:
        config.DAILY_AGG_MODE = args.daily_agg_mode
    if args.gine_use_spatial_coords:
        config.GINE_USE_SPATIAL_COORDS = True
    if args.no_gine_spatial_coords:
        config.GINE_USE_SPATIAL_COORDS = False
    if args.temporal_feature_mode is not None:
        config.TEMPORAL_FEATURE_MODE = args.temporal_feature_mode
    if args.daily_agg_mode is not None:
        config.DAILY_AGG_MODE = args.daily_agg_mode
    if args.graph_temporal_mode is not None:
        config.GRAPH_TEMPORAL_MODE = args.graph_temporal_mode
    if args.temporal_layers is not None:
        config.LSTM_LAYERS = args.temporal_layers
    config.TEMPORAL_SUBSCALES = parse_temporal_subscales(args.temporal_subscales)
    if args.no_temporal_log1p:
        config.TEMPORAL_LOG1P = False
    if args.spatial_layers is not None:
        config.SPATIAL_LAYERS = args.spatial_layers
    if args.spatial_hidden_size is not None:
        config.SPATIAL_HIDDEN_SIZE = args.spatial_hidden_size
    if args.laplacian_pe_dim is not None:
        config.LAPLACIAN_PE_DIM = args.laplacian_pe_dim

    focal_loss_gamma = args.focal_loss_gamma  # None → CrossEntropy, float → FocalLoss
    if args.spatial_layers is not None:
        config.SPATIAL_LAYERS = args.spatial_layers
    if args.spatial_hidden_size is not None:
        config.SPATIAL_HIDDEN_SIZE = args.spatial_hidden_size
    if args.laplacian_pe_dim is not None:
        config.LAPLACIAN_PE_DIM = args.laplacian_pe_dim

    focal_loss_gamma = args.focal_loss_gamma  # None → CrossEntropy, float → FocalLoss

    graph_topk_out = config.GRAPH_TOPK_OUT
    graph_topk_in = config.GRAPH_TOPK_IN
    topk_enabled = (graph_topk_out is not None) or (graph_topk_in is not None)
    active_gine_edge_feature_mode = (
        config.GINE_EDGE_FEATURE_MODE if spatial_model in ('GINE', 'GAT', 'MPNN') else 'flow_only'
    )
    temporal_subscales = tuple(config.TEMPORAL_SUBSCALES)
    temporal_subscale_desc = describe_temporal_subscales(temporal_subscales)
    temporal_log1p = getattr(config, 'TEMPORAL_LOG1P', True)

    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()

    logger.info("=" * 80)
    logger.info("Multi-Scale Temporal Branch Training")
    logger.info("=" * 80)
    logger.info(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"\nImprovement: Temporal Branch = {temporal_subscale_desc}")
    logger.info(f"Dataset: {label_path} with flow threshold {config.FLOW_THRESHOLD} and {config.TIME_STEPS} time steps")
    logger.info(
        f"Runtime overrides | spatial_model={spatial_model}, samples_per_class={samples_per_class}, "
        f"epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, "
        f"grad_accum={gradient_accumulation}, early_stop={early_stopping_patience}, use_cache={use_cache}"
    )
    logger.info(f"Temporal override | temporal_model={temporal_model}")
    logger.info(f"Temporal layer override | layers={config.LSTM_LAYERS}")
    logger.info(f"Temporal subscales | active={temporal_subscales}")
    logger.info(f"Temporal log1p | enabled={temporal_log1p}")
    logger.info(
        f"Graph overrides | topk_out={graph_topk_out}, topk_in={graph_topk_in}, topk_enabled={topk_enabled}"
    )
    logger.info(f"Graph temporal mode | mode={getattr(config, 'GRAPH_TEMPORAL_MODE', 'static')}")
    logger.info(f"Seed override | random_seed={config.RANDOM_SEED}")
    logger.info(f"Spatial node feature mode | mode={config.SPATIAL_NODE_FEATURE_MODE}")
    logger.info(f"GINE edge feature mode | mode={active_gine_edge_feature_mode}")
    logger.info(f"Temporal feature mode | mode={getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')}")
    if getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow') == 'flow_degree_wamd':
        logger.info(f"Daily agg mode | mode={getattr(config, 'DAILY_AGG_MODE', 'sum')}")
    logger.info(f"Temporal feature mode | mode={getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')}")
    if getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow') == 'flow_degree_wamd':
        logger.info(f"Daily agg mode | mode={getattr(config, 'DAILY_AGG_MODE', 'sum')}")
    logger.info(f"Split manifest | load={args.split_manifest}, export={args.export_split_manifest}")
    logger.info(f"Branch ablation mode | mode={args.branch_ablation_mode}")
    logger.info(f"Fusion ablation mode | mode={args.fusion_ablation_mode}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_stem = os.path.splitext(os.path.basename(label_path))[0]
    output_dir = f"outputs/multiscale_temporal_{timestamp}_{label_stem}"
    if args.run_tag:
        output_dir = f"{output_dir}_{args.run_tag}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)

    # Setup file logging
    file_handler = logging.FileHandler(f"{output_dir}/training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Log file: {output_dir}/training.log")

    output_split_manifest_path = f"{output_dir}/metrics/split_manifest.json"

    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading data")
    logger.info("=" * 80)

    data = prepare_dual_year_experiment_data(
        label_path=label_path,
        samples_per_class=samples_per_class,
        use_cache=use_cache,
        spatial_model=spatial_model,
        edge_feature_mode=active_gine_edge_feature_mode
    )

    logger.info(f"✓ Data loaded successfully")
    logger.info(f"  - Total grids: {len(data['labels'])}")
    logger.info(f"  - Feature shape: {list(data['change_features'].values())[0].shape}")
    logger.info(f"  - Graph 2021 edges: {data['graphs_2021'][0][0].shape[1]}")
    logger.info(f"  - Graph 2024 edges: {data['graphs_2024'][0][0].shape[1]}")
    logger.info(f"  - Graph edge feature mode: {data.get('edge_feature_mode', 'flow_only')}")

    # Prepare temporal features
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Preparing temporal features")
    logger.info("=" * 80)

    temporal_features_2021 = {}
    temporal_features_2024 = {}

    _tfm = getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')
    _temporal_log1p = getattr(config, 'TEMPORAL_LOG1P', True)
    if _temporal_log1p:
        # Use log1p-transformed features from change_features
        for grid_id, features in data['change_features'].items():
            if _tfm == 'flow_degree_wamd':
                # (168, 6) = [flow21, degree21, wamd21, flow24, degree24, wamd24]
                temporal_features_2021[grid_id] = features[:, [0, 1, 2]]  # (168, 3)
                temporal_features_2024[grid_id] = features[:, [3, 4, 5]]  # (168, 3)
            elif _tfm == 'flow_wamd_v2':
                # (168, 4) = [flow21, wamd21, flow24, wamd24]
                temporal_features_2021[grid_id] = features[:, [0, 1]]  # (168, 2)
                temporal_features_2024[grid_id] = features[:, [2, 3]]  # (168, 2)
            else:
                # inflow_outflow / total_wamd: (168, 4) = [feat0_2021, feat1_2021, feat0_2024, feat1_2024]
                temporal_features_2021[grid_id] = features[:, [0, 1]]  # (168, 2)
                temporal_features_2024[grid_id] = features[:, [2, 3]]  # (168, 2)
    else:
        # Use raw flows directly (no log1p) from data['flows_2021'] / data['flows_2024']
        for grid_id, raw21 in data['flows_2021'].items():
            if grid_id not in data['flows_2024']:
                continue
            raw24 = data['flows_2024'][grid_id]
            temporal_features_2021[grid_id] = raw21  # (168, 2) or (168, 3) depending on _tfm
            temporal_features_2024[grid_id] = raw24

    logger.info(f"✓ Temporal features prepared (log1p={'yes' if _temporal_log1p else 'no'})")

    # Create dataset
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Creating dataset")
    logger.info("=" * 80)

    grid_ids = list(data['labels'].keys())
    dataset = PureGraphDualYearDataset(
        temporal_features_2021=temporal_features_2021,
        temporal_features_2024=temporal_features_2024,
        labels=data['labels'],
        grid_ids=grid_ids
    )

    logger.info(f"✓ Dataset created: {len(dataset)} samples")

    split_export_paths = [output_split_manifest_path]
    if args.export_split_manifest:
        split_export_paths.append(args.export_split_manifest)

    split_subsets, split_metadata = resolve_dataset_splits(
        dataset=dataset,
        label_path=label_path,
        split_manifest_path=args.split_manifest,
        export_manifest_paths=split_export_paths,
        random_seed=config.RANDOM_SEED,
    )

    train_dataset = split_subsets['train']
    val_dataset = split_subsets['val']
    test_dataset = split_subsets['test']

    logger.info(f"  - Split Source: {split_metadata['source']}")
    if split_metadata['source_manifest_path']:
        logger.info(f"  - Loaded Split Manifest: {split_metadata['source_manifest_path']}")
    for written_manifest_path in split_metadata['written_manifest_paths']:
        logger.info(f"  - Wrote Split Manifest: {written_manifest_path}")
    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")

    # Prepare all temporal features as tensors
    num_nodes = len(data['grid_id_to_idx'])
    # Dynamically infer feature dim from the first sample
    # Dynamically infer feature dim from the first sample
    first_grid_id = next(iter(temporal_features_2021))
    feat_dim = torch.tensor(temporal_features_2021[first_grid_id]).shape[-1]
    # raw temporal always has the per-year dim (2 for inflow_outflow, 3 for flow_degree_wamd)
    raw_feat_dim = feat_dim
    feat_dim = torch.tensor(temporal_features_2021[first_grid_id]).shape[-1]
    # raw temporal always has the per-year dim (2 for inflow_outflow, 3 for flow_degree_wamd)
    raw_feat_dim = feat_dim

    all_temporal_2021 = torch.zeros(num_nodes, 168, feat_dim)
    all_temporal_2024 = torch.zeros(num_nodes, 168, feat_dim)
    all_raw_temporal_2021 = torch.zeros(num_nodes, 168, raw_feat_dim)
    all_raw_temporal_2024 = torch.zeros(num_nodes, 168, raw_feat_dim)
    all_raw_temporal_2021 = torch.zeros(num_nodes, 168, raw_feat_dim)
    all_raw_temporal_2024 = torch.zeros(num_nodes, 168, raw_feat_dim)

    for grid_id, idx in data['grid_id_to_idx'].items():
        if grid_id in temporal_features_2021:
            all_temporal_2021[idx] = torch.tensor(temporal_features_2021[grid_id], dtype=torch.float32)
            all_temporal_2024[idx] = torch.tensor(temporal_features_2024[grid_id], dtype=torch.float32)
        if grid_id in data['flows_2021']:
            raw_arr_2021 = data['flows_2021'][grid_id]
            raw_arr_2024 = data['flows_2024'][grid_id]
            # flows_2021/2024 shape may be (168,2) or (168,3) depending on mode
            if raw_arr_2021.shape[-1] == raw_feat_dim:
                all_raw_temporal_2021[idx] = torch.tensor(raw_arr_2021, dtype=torch.float32)
                all_raw_temporal_2024[idx] = torch.tensor(raw_arr_2024, dtype=torch.float32)
            else:
                # fallback: use only first raw_feat_dim columns
                all_raw_temporal_2021[idx] = torch.tensor(raw_arr_2021[:, :raw_feat_dim], dtype=torch.float32)
                all_raw_temporal_2024[idx] = torch.tensor(raw_arr_2024[:, :raw_feat_dim], dtype=torch.float32)
            raw_arr_2021 = data['flows_2021'][grid_id]
            raw_arr_2024 = data['flows_2024'][grid_id]
            # flows_2021/2024 shape may be (168,2) or (168,3) depending on mode
            if raw_arr_2021.shape[-1] == raw_feat_dim:
                all_raw_temporal_2021[idx] = torch.tensor(raw_arr_2021, dtype=torch.float32)
                all_raw_temporal_2024[idx] = torch.tensor(raw_arr_2024, dtype=torch.float32)
            else:
                # fallback: use only first raw_feat_dim columns
                all_raw_temporal_2021[idx] = torch.tensor(raw_arr_2021[:, :raw_feat_dim], dtype=torch.float32)
                all_raw_temporal_2024[idx] = torch.tensor(raw_arr_2024[:, :raw_feat_dim], dtype=torch.float32)

    # Create collator
    collator = PureGraphBatchCollator(
        graphs_2021=data['graphs_2021'],
        graphs_2024=data['graphs_2024'],
        grid_id_to_idx=data['grid_id_to_idx'],
        all_temporal_2021=all_temporal_2021,
        all_temporal_2024=all_temporal_2024,
        all_raw_temporal_2021=all_raw_temporal_2021,
        all_raw_temporal_2024=all_raw_temporal_2024
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    logger.info(f"✓ Data loaders created")

    # Create enhanced model with multi-scale temporal branch
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Creating Enhanced Model")
    logger.info("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info(f"Spatial model: {spatial_model}")
    logger.info(f"Temporal model: {temporal_model}")

    # Derive temporal_input_size from feature mode
    _tfm = getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')
    if _tfm == 'flow_degree_wamd':
        temporal_input_size = 3  # (168, 3) per year: [flow, degree, wamd]
    elif _tfm == 'flow_wamd_v2':
        temporal_input_size = 2  # (168, 2) per year: [flow, wamd]
    else:
        temporal_input_size = config.TEMPORAL_INPUT_SIZE  # default 2

    # Derive temporal_input_size from feature mode
    _tfm = getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')
    if _tfm == 'flow_degree_wamd':
        temporal_input_size = 3  # (168, 3) per year: [flow, degree, wamd]
    elif _tfm == 'flow_wamd_v2':
        temporal_input_size = 2  # (168, 2) per year: [flow, wamd]
    else:
        temporal_input_size = config.TEMPORAL_INPUT_SIZE  # default 2

    model = EnhancedDualBranchModel(
        temporal_input_size=temporal_input_size,
        hidden_size=config.FUSION_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        num_time_steps=config.TIME_STEPS,
        dropout=config.LSTM_DROPOUT,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        branch_ablation_mode=args.branch_ablation_mode,
        fusion_ablation_mode=args.fusion_ablation_mode
    )

    # Register spatial coordinates for GINE node features
    if getattr(model, 'gine_use_spatial_coords', False):
        node_coords = torch.zeros(num_nodes, 2, dtype=torch.float32)
        grid_id_to_idx = data['grid_id_to_idx']
        coord_lookup = data['metadata_df'].set_index('grid_id')[['lon', 'lat']]
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in coord_lookup.index:
                node_coords[idx, 0] = float(coord_lookup.at[grid_id, 'lon'])
                node_coords[idx, 1] = float(coord_lookup.at[grid_id, 'lat'])
        model.register_node_coords(node_coords)
        logger.info(f"  - Registered spatial coordinates for GINE node features ({node_coords.shape})")

    # Register wamd node features for flow_wamd spatial node feature mode
    if config.SPATIAL_NODE_FEATURE_MODE == 'flow_wamd':
        wamd_2021_dict = data.get('wamd_node_features_2021')
        wamd_2024_dict = data.get('wamd_node_features_2024')
        if wamd_2021_dict is None or wamd_2024_dict is None:
            raise ValueError("flow_wamd mode requires TEMPORAL_FEATURE_MODE='total_wamd' to precompute wamd features")
        grid_id_to_idx = data['grid_id_to_idx']
        wamd_2021 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        wamd_2024 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in wamd_2021_dict:
                wamd_2021[idx] = torch.from_numpy(wamd_2021_dict[grid_id])
            if grid_id in wamd_2024_dict:
                wamd_2024[idx] = torch.from_numpy(wamd_2024_dict[grid_id])
        model.register_wamd_node_features(wamd_2021, wamd_2024)
        logger.info(f"  - Registered wamd node features ({wamd_2021.shape})")

    # Register flow+degree+wamd node features for flow_degree_wamd spatial node feature mode
    if config.SPATIAL_NODE_FEATURE_MODE == 'flow_degree_wamd':
        fdw_2021_dict = data.get('fdw_node_features_2021')
        fdw_2024_dict = data.get('fdw_node_features_2024')
        if fdw_2021_dict is None or fdw_2024_dict is None:
            raise ValueError("flow_degree_wamd mode requires TEMPORAL_FEATURE_MODE='flow_degree_wamd' to precompute fdw features")
        grid_id_to_idx = data['grid_id_to_idx']
        fdw_2021 = torch.zeros(num_nodes, 3, dtype=torch.float32)
        fdw_2024 = torch.zeros(num_nodes, 3, dtype=torch.float32)
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in fdw_2021_dict:
                fdw_2021[idx] = torch.from_numpy(fdw_2021_dict[grid_id])
            if grid_id in fdw_2024_dict:
                fdw_2024[idx] = torch.from_numpy(fdw_2024_dict[grid_id])
        model.register_fdw_node_features(fdw_2021, fdw_2024)
        logger.info(f"  - Registered flow_degree_wamd node features ({fdw_2021.shape})")

    # Register flow+wamd (no degree) node features for flow_wamd_v2 spatial node feature mode
    if config.SPATIAL_NODE_FEATURE_MODE == 'flow_wamd_v2':
        fw2_2021_dict = data.get('fdw_node_features_2021')
        fw2_2024_dict = data.get('fdw_node_features_2024')
        if fw2_2021_dict is None or fw2_2024_dict is None:
            raise ValueError("flow_wamd_v2 mode requires TEMPORAL_FEATURE_MODE='flow_wamd_v2' to precompute features")
        grid_id_to_idx = data['grid_id_to_idx']
        fw2_2021 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        fw2_2024 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in fw2_2021_dict:
                fw2_2021[idx] = torch.from_numpy(fw2_2021_dict[grid_id])
            if grid_id in fw2_2024_dict:
                fw2_2024[idx] = torch.from_numpy(fw2_2024_dict[grid_id])
        model.register_flow_wamd_v2_node_features(fw2_2021, fw2_2024)
        logger.info(f"  - Registered flow_wamd_v2 node features ({fw2_2021.shape})")

    # Register wamd node features for flow_wamd spatial node feature mode
    if config.SPATIAL_NODE_FEATURE_MODE == 'flow_wamd':
        wamd_2021_dict = data.get('wamd_node_features_2021')
        wamd_2024_dict = data.get('wamd_node_features_2024')
        if wamd_2021_dict is None or wamd_2024_dict is None:
            raise ValueError("flow_wamd mode requires TEMPORAL_FEATURE_MODE='total_wamd' to precompute wamd features")
        grid_id_to_idx = data['grid_id_to_idx']
        wamd_2021 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        wamd_2024 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in wamd_2021_dict:
                wamd_2021[idx] = torch.from_numpy(wamd_2021_dict[grid_id])
            if grid_id in wamd_2024_dict:
                wamd_2024[idx] = torch.from_numpy(wamd_2024_dict[grid_id])
        model.register_wamd_node_features(wamd_2021, wamd_2024)
        logger.info(f"  - Registered wamd node features ({wamd_2021.shape})")

    # Register flow+degree+wamd node features for flow_degree_wamd spatial node feature mode
    if config.SPATIAL_NODE_FEATURE_MODE == 'flow_degree_wamd':
        fdw_2021_dict = data.get('fdw_node_features_2021')
        fdw_2024_dict = data.get('fdw_node_features_2024')
        if fdw_2021_dict is None or fdw_2024_dict is None:
            raise ValueError("flow_degree_wamd mode requires TEMPORAL_FEATURE_MODE='flow_degree_wamd' to precompute fdw features")
        grid_id_to_idx = data['grid_id_to_idx']
        fdw_2021 = torch.zeros(num_nodes, 3, dtype=torch.float32)
        fdw_2024 = torch.zeros(num_nodes, 3, dtype=torch.float32)
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in fdw_2021_dict:
                fdw_2021[idx] = torch.from_numpy(fdw_2021_dict[grid_id])
            if grid_id in fdw_2024_dict:
                fdw_2024[idx] = torch.from_numpy(fdw_2024_dict[grid_id])
        model.register_fdw_node_features(fdw_2021, fdw_2024)
        logger.info(f"  - Registered flow_degree_wamd node features ({fdw_2021.shape})")

    # Register flow+wamd (no degree) node features for flow_wamd_v2 spatial node feature mode
    if config.SPATIAL_NODE_FEATURE_MODE == 'flow_wamd_v2':
        fw2_2021_dict = data.get('fdw_node_features_2021')
        fw2_2024_dict = data.get('fdw_node_features_2024')
        if fw2_2021_dict is None or fw2_2024_dict is None:
            raise ValueError("flow_wamd_v2 mode requires TEMPORAL_FEATURE_MODE='flow_wamd_v2' to precompute features")
        grid_id_to_idx = data['grid_id_to_idx']
        fw2_2021 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        fw2_2024 = torch.zeros(num_nodes, 2, dtype=torch.float32)
        for grid_id, idx in grid_id_to_idx.items():
            if grid_id in fw2_2021_dict:
                fw2_2021[idx] = torch.from_numpy(fw2_2021_dict[grid_id])
            if grid_id in fw2_2024_dict:
                fw2_2024[idx] = torch.from_numpy(fw2_2024_dict[grid_id])
        model.register_flow_wamd_v2_node_features(fw2_2021, fw2_2024)
        logger.info(f"  - Registered flow_wamd_v2 node features ({fw2_2021.shape})")

    model = model.to(device)

    # Move graphs to device
    model.graphs_2021 = [(torch.from_numpy(edge_idx).to(device), torch.from_numpy(edge_attr).to(device))
                         for edge_idx, edge_attr in data['graphs_2021']]
    model.graphs_2024 = [(torch.from_numpy(edge_idx).to(device), torch.from_numpy(edge_attr).to(device))
                         for edge_idx, edge_attr in data['graphs_2024']]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✓ Enhanced model created")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Temporal subscales: {temporal_subscale_desc}")
    logger.info(f"  - Temporal branch model: {temporal_model}")
    logger.info(f"  - Spatial branch: {spatial_model}")
    if spatial_model == "GINE":
        logger.info(f"  - Laplacian PE dimension: {config.LAPLACIAN_PE_DIM}")
    logger.info(f"  - Branch ablation mode: {args.branch_ablation_mode}")
    logger.info(f"  - Fusion ablation mode: {args.fusion_ablation_mode}")

    # Create loss function
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Setting up training")
    logger.info("=" * 80)

    class_weights = data['class_weights'].to(device)
    if focal_loss_gamma is not None:
        criterion = FocalLoss(gamma=focal_loss_gamma, weight=class_weights)
        loss_desc = f"Focal Loss (gamma={focal_loss_gamma}, weighted)"
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss_desc = "Weighted CrossEntropy"
    if focal_loss_gamma is not None:
        criterion = FocalLoss(gamma=focal_loss_gamma, weight=class_weights)
        loss_desc = f"Focal Loss (gamma={focal_loss_gamma}, weighted)"
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss_desc = "Weighted CrossEntropy"

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        patience=scheduler_patience,
        factor=scheduler_factor,
        verbose=True
    )

    logger.info(f"✓ Training setup complete")
    logger.info(f"  - Optimizer: Adam (lr={learning_rate}, wd={config.WEIGHT_DECAY})")
    logger.info(
        f"  - Scheduler: ReduceLROnPlateau (patience={scheduler_patience}, "
        f"factor={scheduler_factor})"
    )
    logger.info(f"  - Loss: {loss_desc}")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Training")
    logger.info("=" * 80)

    best_accuracy = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader,
            criterion, optimizer, device, accumulation_steps=gradient_accumulation
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader,
            criterion, device
        )

        # Log metrics
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Train Accuracy: {train_metrics['accuracy']:.2f}%")

        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.2f}% | F1: {val_metrics['f1']:.4f}")

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  LR: {current_lr:.6f}")

        # Update scheduler
        scheduler.step(val_metrics['accuracy'])

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'f1': val_metrics['f1']
            }, f"{output_dir}/models/best_model.pth")

            logger.info(f"  ✓ New best model saved! (Acc: {best_accuracy:.2f}%)")
        else:
            patience_counter += 1

        logger.info(f"  Patience: {patience_counter}/{early_stopping_patience}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"\n✓ Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"\n✓ Training completed!")
    logger.info(f"  Best Accuracy: {best_accuracy:.2f}%")

    # Test
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Testing")
    logger.info("=" * 80)

    # Load best model
    checkpoint = torch.load(f"{output_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Evaluate on test set
    test_metrics = evaluate(
        model, test_loader,
        criterion, device
    )

    logger.info(f"\nTest Results:")
    logger.info(f"  - Accuracy: {test_metrics['accuracy']:.2f}%")
    logger.info(f"  - F1 Score: {test_metrics['f1']:.4f}")

    # Save test results with detailed configuration
    test_kappa = cohen_kappa_score(test_metrics['all_labels'], test_metrics['all_preds'])
    logger.info(f"  - Kappa: {test_kappa:.4f}")

    test_results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'test_kappa': float(test_kappa),
        'test_kappa': float(test_kappa),
        'improvement': 'Multi-Scale Temporal Branch',
        'data_config': {
            'flow_threshold': config.FLOW_THRESHOLD,
            'time_steps': config.TIME_STEPS,
            'time_steps_description': f'{config.TIME_STEPS // 24} days × 24 hours',
            'topk_enabled': bool(topk_enabled),
            'graph_topk_out': int(graph_topk_out) if graph_topk_out is not None else None,
            'graph_topk_in': int(graph_topk_in) if graph_topk_in is not None else None,
            'graph_edge_feature_mode': data.get('edge_feature_mode', 'flow_only')
        },
        'model_architecture': {
            'temporal_branch': {
                'type': f'{temporal_subscale_desc} + {temporal_model}',
                'temporal_model': temporal_model,
                'temporal_subscales': list(temporal_subscales),
                'temporal_log1p': temporal_log1p,
                'temporal_layers': config.LSTM_LAYERS,
                'lstm_hidden_size': config.LSTM_HIDDEN_SIZE,
                'lstm_dropout': config.LSTM_DROPOUT,
                'temporal_input_size': config.TEMPORAL_INPUT_SIZE
            },
            'spatial_branch': {
                'type': f'Pure Graph {spatial_model}',
                'spatial_layers': config.SPATIAL_LAYERS,
                'spatial_hidden_size': config.SPATIAL_HIDDEN_SIZE,
                'node_feature_mode': config.SPATIAL_NODE_FEATURE_MODE,
                'graph_temporal_mode': getattr(config, 'GRAPH_TEMPORAL_MODE', 'static'),
                'topk_enabled': bool(topk_enabled),
                'graph_topk_out': int(graph_topk_out) if graph_topk_out is not None else None,
                'graph_topk_in': int(graph_topk_in) if graph_topk_in is not None else None,
                'laplacian_pe_dim': config.LAPLACIAN_PE_DIM if spatial_model == 'GINE' else None,
                'edge_feature_mode': data.get('edge_feature_mode', 'flow_only')
            },
            'fusion': {
                'type': {
                    'gated': 'Gated Fusion',
                    'mean': 'Mean Pooling',
                    'concat': 'Concat + MLP'
                }[args.fusion_ablation_mode],
                'fusion_hidden_size': config.FUSION_HIDDEN_SIZE,
                'attention_heads': config.ATTENTION_HEADS,
                'branch_ablation_mode': args.branch_ablation_mode,
                'fusion_ablation_mode': args.fusion_ablation_mode
            },
            'output': {
                'num_classes': config.NUM_CLASSES
            }
        },
        'training_config': {
            'batch_size': batch_size,
            'gradient_accumulation': gradient_accumulation,
            'learning_rate': learning_rate,
            'weight_decay': config.WEIGHT_DECAY,
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'scheduler_patience': scheduler_patience,
            'scheduler_factor': scheduler_factor,
            'train_split': config.TRAIN_SPLIT,
            'val_split': config.VAL_SPLIT,
            'test_split': config.TEST_SPLIT,
            'random_seed': config.RANDOM_SEED,
            'split_source': split_metadata['source'],
            'split_manifest_loaded': split_metadata['source_manifest_path'],
            'split_manifest_written': split_metadata['written_manifest_paths'],
            'dropout': config.LSTM_DROPOUT,
            'loss_type': 'focal' if focal_loss_gamma is not None else 'cross_entropy',
            'focal_loss_gamma': focal_loss_gamma
        },
        'data_info': {
            'label_file': label_path,
            'label_file_hash': data.get('label_file_hash', 'N/A'),
            'total_samples': len(data['labels']),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'graph_2021_edges': int(data['graphs_2021'][0][0].shape[1]),
            'graph_2024_edges': int(data['graphs_2024'][0][0].shape[1]),
            'class_distribution': {
                f'class_{i+1}': int(data.get('class_distribution', {}).get(i, 0))
                for i in range(config.NUM_CLASSES)
            }
        }
    }

    with open(f"{output_dir}/metrics/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    # Save classification report
    report = classification_report(
        test_metrics['all_labels'],
        test_metrics['all_preds'],
        target_names=[f'Class {i+1}' for i in range(9)],
        zero_division=0
    )

    with open(f"{output_dir}/metrics/classification_report.txt", 'w') as f:
        f.write("9-Class Classification Report - Multi-Scale Temporal Branch\n")
        f.write("=" * 80 + "\n\n")

        # Write data information
        f.write("Data Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Label File: {data.get('label_file_name', 'N/A')}\n")
        f.write(f"  Label File Hash: {data.get('label_file_hash', 'N/A')}\n")
        f.write("\n")

        # Write class distribution
        f.write("Class Distribution:\n")
        f.write("-" * 80 + "\n")
        class_dist = data.get('class_distribution', {})
        for i in range(config.NUM_CLASSES):
            count = class_dist.get(i, 0)
            weight = data['class_weights'][i].item()
            f.write(f"  Class {i+1}: {count} samples (weight: {weight:.4f})\n")
        f.write("\n")

        # Write configuration
        f.write("Data Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Flow Threshold: {config.FLOW_THRESHOLD}\n")
        f.write(f"  Time Steps: {config.TIME_STEPS} ({config.TIME_STEPS // 24} days × 24 hours)\n")
        f.write(f"  Top-k Enabled: {topk_enabled}\n")
        f.write(f"  Graph Top-k Out: {graph_topk_out}\n")
        f.write(f"  Graph Top-k In: {graph_topk_in}\n")
        f.write(f"  Graph Edge Feature Mode: {data.get('edge_feature_mode', 'flow_only')}\n")
        f.write(f"  Graph 2021 Edges: {int(data['graphs_2021'][0][0].shape[1])}\n")
        f.write(f"  Graph 2024 Edges: {int(data['graphs_2024'][0][0].shape[1])}\n")
        f.write("\n")

        f.write("Model Architecture:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Temporal Branch:\n")
        f.write(f"    - Type: {temporal_subscale_desc} + {temporal_model}\n")
        f.write(f"    - Temporal Model: {temporal_model}\n")
        f.write(f"    - Temporal Subscales: {', '.join(temporal_subscales)}\n")
        f.write(f"    - Temporal Layers: {config.LSTM_LAYERS}\n")
        f.write(f"    - LSTM Hidden Size: {config.LSTM_HIDDEN_SIZE}\n")
        f.write(f"    - LSTM Dropout: {config.LSTM_DROPOUT}\n")
        f.write(f"    - Temporal Input Size: {config.TEMPORAL_INPUT_SIZE}\n")
        f.write(f"  Spatial Branch:\n")
        f.write(f"    - Type: Pure Graph {spatial_model}\n")
        f.write(f"    - SPATIAL Layers: {config.SPATIAL_LAYERS}\n")
        f.write(f"    - SPATIAL Hidden Size: {config.SPATIAL_HIDDEN_SIZE}\n")
        f.write(f"    - Node Feature Mode: {config.SPATIAL_NODE_FEATURE_MODE}\n")
        f.write(f"    - Graph Temporal Mode: {getattr(config, 'GRAPH_TEMPORAL_MODE', 'static')}\n")
        f.write(f"    - Top-k Enabled: {topk_enabled}\n")
        f.write(f"    - Graph Top-k Out: {graph_topk_out}\n")
        f.write(f"    - Graph Top-k In: {graph_topk_in}\n")
        f.write(f"    - Edge Feature Mode: {data.get('edge_feature_mode', 'flow_only')}\n")
        if spatial_model == "GINE":
            f.write(f"    - Laplacian PE Dimension: {config.LAPLACIAN_PE_DIM}\n")
        fusion_type_name = {
            'gated': 'Gated Fusion',
            'mean': 'Mean Pooling',
            'concat': 'Concat + MLP'
        }[args.fusion_ablation_mode]
        f.write(f"  Fusion:\n")
        f.write(f"    - Type: {fusion_type_name}\n")
        f.write(f"    - Fusion Hidden Size: {config.FUSION_HIDDEN_SIZE}\n")
        f.write(f"    - Attention Heads: {config.ATTENTION_HEADS}\n")
        f.write(f"    - Branch Ablation Mode: {args.branch_ablation_mode}\n")
        f.write(f"    - Fusion Ablation Mode: {args.fusion_ablation_mode}\n")
        f.write(f"  Output:\n")
        f.write(f"    - Num Classes: {config.NUM_CLASSES}\n")
        f.write("\n")

        f.write("Training Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Gradient Accumulation: {gradient_accumulation}\n")
        f.write(f"  Learning Rate: {learning_rate}\n")
        f.write(f"  Weight Decay: {config.WEIGHT_DECAY}\n")
        f.write(f"  Num Epochs: {num_epochs}\n")
        f.write(f"  Early Stopping Patience: {early_stopping_patience}\n")
        f.write(f"  Scheduler Patience: {scheduler_patience}\n")
        f.write(f"  Scheduler Factor: {scheduler_factor}\n")
        f.write(f"  Train/Val/Test Split: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}\n")
        f.write(f"  Random Seed: {config.RANDOM_SEED}\n")
        f.write(f"  Split Source: {split_metadata['source']}\n")
        f.write(f"  Loaded Split Manifest: {split_metadata['source_manifest_path']}\n")
        f.write(f"  Written Split Manifest(s): {', '.join(split_metadata['written_manifest_paths'])}\n")
        f.write(f"  Dropout: {config.LSTM_DROPOUT}\n")
        f.write(f"  Loss Type: {'focal' if focal_loss_gamma is not None else 'cross_entropy'}\n")
        if focal_loss_gamma is not None:
            f.write(f"  Focal Loss Gamma: {focal_loss_gamma}\n")
        f.write(f"  Loss Type: {'focal' if focal_loss_gamma is not None else 'cross_entropy'}\n")
        if focal_loss_gamma is not None:
            f.write(f"  Focal Loss Gamma: {focal_loss_gamma}\n")
        f.write("\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(test_metrics['all_labels'], test_metrics['all_preds'])
    np.save(f"{output_dir}/metrics/confusion_matrix.npy", cm)

    logger.info(f"\n✓ All results saved to {output_dir}/metrics/")
    logger.info("  - test_results.json")
    logger.info("  - classification_report.txt")
    logger.info("  - confusion_matrix.npy")

    # Calculate and log total training time
    end_time = time.time()
    end_datetime = datetime.now()
    total_time = end_time - start_time

    logger.info("\n" + "=" * 80)
    logger.info("Training Time Statistics")
    logger.info("=" * 80)
    logger.info(f"Start time:    {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:      {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time:    {format_time(total_time)} ({total_time:.2f} seconds)")
    logger.info(f"               ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
    logger.info("=" * 80)

    # Save timing information to file
    timing_info = {
        "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "total_time_seconds": total_time,
        "total_time_formatted": format_time(total_time),
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600
    }

    with open(f"{output_dir}/metrics/timing_info.json", 'w') as f:
        json.dump(timing_info, f, indent=2)

    logger.info(f"✓ Timing info saved to {output_dir}/metrics/timing_info.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

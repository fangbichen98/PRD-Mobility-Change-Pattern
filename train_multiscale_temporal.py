"""
Training script with Multi-Scale Temporal Branch improvement.

Expected improvement: +3-5% accuracy (69% → 72-74%)
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from datetime import datetime
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data

# Import Enhanced model with Multi-Scale Temporal Branch
from src.models.enhanced_dual_branch_model import EnhancedDualBranchModel
from src.training.dataset_pure_graph import PureGraphDualYearDataset, PureGraphBatchCollator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse optional runtime overrides for controlled experiment sweeps."""
    parser = argparse.ArgumentParser(description="Train Multi-Scale Temporal + Graph model")
    parser.add_argument('--label-path', type=str, default=None,
                        help='Override label file path')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='Optional per-class sampling for small-scale experiments')
    parser.add_argument('--spatial-model', type=str, choices=['GCN', 'SAGE', 'WGCN', 'GINE', 'GIN', 'GAT', 'EVOLVEGCN'], default=None,
                        help='Override spatial branch model')
    parser.add_argument('--temporal-model', type=str, choices=['LSTM', 'GRU', 'TCN', 'TRANSFORMER', 'TRANSFORMER_FULL', 'BIGRU'], default=None,
                        help='Override temporal branch model')
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
                        choices=['ones', 'temporal_mean', 'annual_daily_mean', 'annual_daily_mean_2d', 'raw_temporal_mean'],
                        default=None,
                        help='Spatial node feature mode for non-GINE branches')
    parser.add_argument('--branch-ablation-mode', type=str, choices=['full', 'temporal_only', 'spatial_only'], default='full',
                        help='Ablate temporal/spatial branches while keeping the same training pipeline')
    parser.add_argument('--fusion-ablation-mode', type=str, choices=['gated', 'mean', 'concat'], default='gated',
                        help='Ablate fusion mechanism by replacing gated fusion with simple mean pooling or concat')
    return parser.parse_args()


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
    if args.graph_temporal_mode is not None:
        config.GRAPH_TEMPORAL_MODE = args.graph_temporal_mode

    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()

    logger.info("=" * 80)
    logger.info("Multi-Scale Temporal Branch Training")
    logger.info("=" * 80)
    logger.info(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("\nImprovement: Multi-Scale Temporal Branch (hourly + daily + weekly)")
    logger.info(f"Dataset: {label_path} with flow threshold {config.FLOW_THRESHOLD} and {config.TIME_STEPS} time steps")
    logger.info(
        f"Runtime overrides | spatial_model={spatial_model}, samples_per_class={samples_per_class}, "
        f"epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, "
        f"grad_accum={gradient_accumulation}, early_stop={early_stopping_patience}, use_cache={use_cache}"
    )
    logger.info(f"Temporal override | temporal_model={temporal_model}")
    logger.info(
        f"Graph overrides | topk_out={config.GRAPH_TOPK_OUT}, topk_in={config.GRAPH_TOPK_IN}"
    )
    logger.info(f"Graph temporal mode | mode={getattr(config, 'GRAPH_TEMPORAL_MODE', 'static')}")
    logger.info(f"Seed override | random_seed={config.RANDOM_SEED}")
    logger.info(f"Spatial node feature mode | mode={config.SPATIAL_NODE_FEATURE_MODE}")
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

    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading data")
    logger.info("=" * 80)

    data = prepare_dual_year_experiment_data(
        label_path=label_path,
        samples_per_class=samples_per_class,
        use_cache=use_cache
    )

    logger.info(f"✓ Data loaded successfully")
    logger.info(f"  - Total grids: {len(data['labels'])}")
    logger.info(f"  - Feature shape: {list(data['change_features'].values())[0].shape}")
    logger.info(f"  - Graph 2021 edges: {data['graphs_2021'][0][0].shape[1]}")
    logger.info(f"  - Graph 2024 edges: {data['graphs_2024'][0][0].shape[1]}")

    # Prepare temporal features
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Preparing temporal features")
    logger.info("=" * 80)

    temporal_features_2021 = {}
    temporal_features_2024 = {}

    for grid_id, features in data['change_features'].items():
        # Features shape: (168, 4) = [inflow_2021_log, outflow_2021_log, inflow_2024_log, outflow_2024_log]
        temporal_features_2021[grid_id] = features[:, [0, 1]]  # (168, 2) - [inflow, outflow] for 2021
        temporal_features_2024[grid_id] = features[:, [2, 3]]  # (168, 2) - [inflow, outflow] for 2024

    logger.info(f"✓ Temporal features prepared")

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

    # Split dataset
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = int(config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    logger.info(f"  - Train: {len(train_dataset)} samples")
    logger.info(f"  - Val: {len(val_dataset)} samples")
    logger.info(f"  - Test: {len(test_dataset)} samples")

    # Prepare all temporal features as tensors
    num_nodes = len(data['grid_id_to_idx'])
    # 1. 动态获取特征维度（从数据字典中随便取一个样本看它的最后一维）
    first_grid_id = next(iter(temporal_features_2021))
    feat_dim = torch.tensor(temporal_features_2021[first_grid_id]).shape[-1] 
    # 这里 feat_dim 会自动变成 2

    # 2. 使用 feat_dim 初始化
    all_temporal_2021 = torch.zeros(num_nodes, 168, feat_dim)
    all_temporal_2024 = torch.zeros(num_nodes, 168, feat_dim)
    all_raw_temporal_2021 = torch.zeros(num_nodes, 168, feat_dim)
    all_raw_temporal_2024 = torch.zeros(num_nodes, 168, feat_dim)

    for grid_id, idx in data['grid_id_to_idx'].items():
        if grid_id in temporal_features_2021:
            # 现在维度匹配了，都是 (168, 2)
            all_temporal_2021[idx] = torch.tensor(temporal_features_2021[grid_id], dtype=torch.float32)
            all_temporal_2024[idx] = torch.tensor(temporal_features_2024[grid_id], dtype=torch.float32)
        if grid_id in data['flows_2021']:
            all_raw_temporal_2021[idx] = torch.tensor(data['flows_2021'][grid_id], dtype=torch.float32)
            all_raw_temporal_2024[idx] = torch.tensor(data['flows_2024'][grid_id], dtype=torch.float32)

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

    model = EnhancedDualBranchModel(
        temporal_input_size=config.TEMPORAL_INPUT_SIZE,
        hidden_size=config.FUSION_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        num_time_steps=config.TIME_STEPS,
        dropout=config.LSTM_DROPOUT,
        spatial_model=spatial_model,
        temporal_model=temporal_model,
        branch_ablation_mode=args.branch_ablation_mode,
        fusion_ablation_mode=args.fusion_ablation_mode
    )

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
    logger.info(f"  - Multi-scale temporal: Hourly + Daily + Weekly")
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
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    logger.info(f"  - Loss: Weighted cross-entropy")

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
    test_results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'improvement': 'Multi-Scale Temporal Branch',
        'data_config': {
            'flow_threshold': config.FLOW_THRESHOLD,
            'time_steps': config.TIME_STEPS,
            'time_steps_description': f'{config.TIME_STEPS // 24} days × 24 hours'
        },
        'model_architecture': {
            'temporal_branch': {
                'type': f'Multi-scale (hourly + daily + weekly) + {temporal_model}',
                'temporal_model': temporal_model,
                'lstm_layers': config.LSTM_LAYERS,
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
                'laplacian_pe_dim': config.LAPLACIAN_PE_DIM if spatial_model == 'GINE' else None
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
            'dropout': config.LSTM_DROPOUT
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
        f.write(f"  Graph 2021 Edges: {int(data['graphs_2021'][0][0].shape[1])}\n")
        f.write(f"  Graph 2024 Edges: {int(data['graphs_2024'][0][0].shape[1])}\n")
        f.write("\n")

        f.write("Model Architecture:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Temporal Branch:\n")
        f.write(f"    - Type: Multi-scale (hourly + daily + weekly) + {temporal_model}\n")
        f.write(f"    - Temporal Model: {temporal_model}\n")
        f.write(f"    - LSTM Layers: {config.LSTM_LAYERS}\n")
        f.write(f"    - LSTM Hidden Size: {config.LSTM_HIDDEN_SIZE}\n")
        f.write(f"    - LSTM Dropout: {config.LSTM_DROPOUT}\n")
        f.write(f"    - Temporal Input Size: {config.TEMPORAL_INPUT_SIZE}\n")
        f.write(f"  Spatial Branch:\n")
        f.write(f"    - Type: Pure Graph {spatial_model}\n")
        f.write(f"    - SPATIAL Layers: {config.SPATIAL_LAYERS}\n")
        f.write(f"    - SPATIAL Hidden Size: {config.SPATIAL_HIDDEN_SIZE}\n")
        f.write(f"    - Node Feature Mode: {config.SPATIAL_NODE_FEATURE_MODE}\n")
        f.write(f"    - Graph Temporal Mode: {getattr(config, 'GRAPH_TEMPORAL_MODE', 'static')}\n")
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
        f.write(f"  Dropout: {config.LSTM_DROPOUT}\n")
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

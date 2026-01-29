"""
Training script for pure graph-based hierarchical model
Simplified version that uses only graph structure for spatial branch
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data
from src.training.dataset_pure_graph import PureGraphDualYearDataset, PureGraphBatchCollator
from src.models.dual_branch_model_pure_graph import PureGraphDualBranchModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_labels_to_hierarchical(labels):
    """
    Convert 9-class labels to hierarchical labels

    Args:
        labels: Tensor of shape (batch_size,) with values 0-8

    Returns:
        intensity_labels: (batch_size,) with values 0-2
        direction_labels: (batch_size,) with values 0-2
    """
    intensity_labels = labels // 3  # 0, 1, 2
    direction_labels = labels % 3   # 0, 1, 2
    return intensity_labels, direction_labels


def combine_hierarchical_predictions(intensity_pred, direction_pred):
    """
    Combine hierarchical predictions into 9-class predictions

    Args:
        intensity_pred: (batch_size,) with values 0-2
        direction_pred: (batch_size,) with values 0-2

    Returns:
        combined_pred: (batch_size,) with values 0-8
    """
    return intensity_pred * 3 + direction_pred


def train_epoch_hierarchical(model, train_loader, criterion_intensity, criterion_direction,
                            criterion_direct, optimizer, device, accumulation_steps=4):
    """
    Train one epoch with hierarchical classification

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion_intensity: Loss function for intensity classification
        criterion_direction: Loss function for direction classification
        criterion_direct: Loss function for direct 9-class classification
        optimizer: Optimizer
        device: Device to use
        accumulation_steps: Number of steps for gradient accumulation

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    total_samples = 0

    # Metrics
    intensity_correct = 0
    direction_correct = 0
    hierarchical_correct = 0
    direct_correct = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        node_indices = batch['node_indices'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']

        # Get full temporal features
        all_temporal_2021 = batch['all_temporal_2021'].to(device)
        all_temporal_2024 = batch['all_temporal_2024'].to(device)

        # Move graphs to device (convert numpy arrays to tensors first)
        graphs_2021 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2024']]

        # Convert labels to hierarchical
        intensity_labels, direction_labels = convert_labels_to_hierarchical(labels)

        # Forward pass
        intensity_logits, direction_logits, direct_logits = model(
            x_2021=all_temporal_2021,
            x_2024=all_temporal_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        # Compute losses
        loss_intensity = criterion_intensity(intensity_logits, intensity_labels)
        loss_direction = criterion_direction(direction_logits, direction_labels)
        loss_direct = criterion_direct(direct_logits, labels)

        # Combined loss (weighted)
        loss = (loss_intensity + loss_direction + 0.5 * loss_direct) / 4

        # Backward pass with gradient accumulation
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Compute predictions
        intensity_pred = intensity_logits.argmax(dim=1)
        direction_pred = direction_logits.argmax(dim=1)
        hierarchical_pred = combine_hierarchical_predictions(intensity_pred, direction_pred)
        direct_pred = direct_logits.argmax(dim=1)

        # Update metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        intensity_correct += (intensity_pred == intensity_labels).sum().item()
        direction_correct += (direction_pred == direction_labels).sum().item()
        hierarchical_correct += (hierarchical_pred == labels).sum().item()
        direct_correct += (direct_pred == labels).sum().item()

        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Epoch [{batch_idx + 1}/{len(train_loader)}] "
                       f"Loss: {loss.item():.4f} "
                       f"(I:{loss_intensity.item():.3f} "
                       f"D:{loss_direction.item():.3f} "
                       f"9C:{loss_direct.item():.3f})")

    # Final gradient update if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Compute average metrics
    avg_loss = total_loss / total_samples
    intensity_acc = 100.0 * intensity_correct / total_samples
    direction_acc = 100.0 * direction_correct / total_samples
    hierarchical_acc = 100.0 * hierarchical_correct / total_samples
    direct_acc = 100.0 * direct_correct / total_samples

    return {
        'loss': avg_loss,
        'intensity_acc': intensity_acc,
        'direction_acc': direction_acc,
        'hierarchical_acc': hierarchical_acc,
        'direct_acc': direct_acc
    }


@torch.no_grad()
def evaluate_hierarchical(model, data_loader, criterion_intensity, criterion_direction,
                         criterion_direct, device):
    """
    Evaluate model with hierarchical classification

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    # Collect predictions and labels
    all_intensity_preds = []
    all_direction_preds = []
    all_hierarchical_preds = []
    all_direct_preds = []
    all_labels = []

    for batch in data_loader:
        # Move data to device
        node_indices = batch['node_indices'].to(device)
        labels = batch['labels'].to(device)
        num_nodes = batch['num_nodes']

        # Get full temporal features
        all_temporal_2021 = batch['all_temporal_2021'].to(device)
        all_temporal_2024 = batch['all_temporal_2024'].to(device)

        # Move graphs to device (convert numpy arrays to tensors first)
        graphs_2021 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2021']]
        graphs_2024 = [(torch.from_numpy(edge_idx).to(device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(device),
                        torch.from_numpy(edge_attr).to(device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(device))
                       for edge_idx, edge_attr in batch['graphs_2024']]

        # Convert labels
        intensity_labels, direction_labels = convert_labels_to_hierarchical(labels)

        # Forward pass
        intensity_logits, direction_logits, direct_logits = model(
            x_2021=all_temporal_2021,
            x_2024=all_temporal_2024,
            graphs_2021=graphs_2021,
            graphs_2024=graphs_2024,
            num_nodes=num_nodes,
            node_indices=node_indices
        )

        # Compute losses
        loss_intensity = criterion_intensity(intensity_logits, intensity_labels)
        loss_direction = criterion_direction(direction_logits, direction_labels)
        loss_direct = criterion_direct(direct_logits, labels)

        loss = (loss_intensity + loss_direction + 0.5 * loss_direct) / 4

        # Update metrics
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute predictions
        intensity_pred = intensity_logits.argmax(dim=1)
        direction_pred = direction_logits.argmax(dim=1)
        hierarchical_pred = combine_hierarchical_predictions(intensity_pred, direction_pred)
        direct_pred = direct_logits.argmax(dim=1)

        # Collect predictions
        all_intensity_preds.extend(intensity_pred.cpu().numpy())
        all_direction_preds.extend(direction_pred.cpu().numpy())
        all_hierarchical_preds.extend(hierarchical_pred.cpu().numpy())
        all_direct_preds.extend(direct_pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_intensity_preds = np.array(all_intensity_preds)
    all_direction_preds = np.array(all_direction_preds)
    all_hierarchical_preds = np.array(all_hierarchical_preds)
    all_direct_preds = np.array(all_direct_preds)
    all_labels = np.array(all_labels)

    # Convert labels to hierarchical
    all_intensity_labels = all_labels // 3
    all_direction_labels = all_labels % 3

    # Compute metrics
    avg_loss = total_loss / total_samples
    intensity_acc = 100.0 * (all_intensity_preds == all_intensity_labels).sum() / total_samples
    direction_acc = 100.0 * (all_direction_preds == all_direction_labels).sum() / total_samples
    hierarchical_acc = 100.0 * (all_hierarchical_preds == all_labels).sum() / total_samples
    direct_acc = 100.0 * (all_direct_preds == all_labels).sum() / total_samples

    # Compute F1 scores
    hierarchical_f1 = f1_score(all_labels, all_hierarchical_preds, average='macro', zero_division=0)
    direct_f1 = f1_score(all_labels, all_direct_preds, average='macro', zero_division=0)

    return {
        'loss': avg_loss,
        'intensity_acc': intensity_acc,
        'direction_acc': direction_acc,
        'hierarchical_acc': hierarchical_acc,
        'hierarchical_f1': hierarchical_f1,
        'direct_acc': direct_acc,
        'direct_f1': direct_f1,
        'all_labels': all_labels,
        'hierarchical_preds': all_hierarchical_preds,
        'direct_preds': all_direct_preds
    }


def main():
    """Main training function"""
    logger.info("=" * 80)
    logger.info("Pure Graph-Based Hierarchical Training")
    logger.info("=" * 80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/pure_graph_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/metrics", exist_ok=True)

    # Setup file logging
    file_handler = logging.FileHandler(f"{output_dir}/training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {output_dir}/training.log")

    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading data")
    logger.info("=" * 80)

    data = prepare_dual_year_experiment_data(
        label_path='data/labels.csv',
        samples_per_class=None,
        use_cache=True
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
        # features shape: (7, 4) = [total_2021, total_2024, net_2021, net_2024]
        # Extract temporal features for each year
        temporal_features_2021[grid_id] = features[:, [0, 2]]  # (7, 2) - [total, net] for 2021
        temporal_features_2024[grid_id] = features[:, [1, 3]]  # (7, 2) - [total, net] for 2024

    logger.info(f"✓ Temporal features prepared")
    logger.info(f"  - 2021 features: {len(temporal_features_2021)} grids")
    logger.info(f"  - 2024 features: {len(temporal_features_2024)} grids")

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
    all_temporal_2021 = torch.zeros(num_nodes, 7, 2)
    all_temporal_2024 = torch.zeros(num_nodes, 7, 2)

    for grid_id, idx in data['grid_id_to_idx'].items():
        if grid_id in temporal_features_2021:
            all_temporal_2021[idx] = torch.tensor(temporal_features_2021[grid_id], dtype=torch.float32)
            all_temporal_2024[idx] = torch.tensor(temporal_features_2024[grid_id], dtype=torch.float32)

    # Create collator
    collator = PureGraphBatchCollator(
        graphs_2021=data['graphs_2021'],
        graphs_2024=data['graphs_2024'],
        grid_id_to_idx=data['grid_id_to_idx'],
        all_temporal_2021=all_temporal_2021,
        all_temporal_2024=all_temporal_2024
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    logger.info(f"✓ Data loaders created")

    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Creating model")
    logger.info("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = PureGraphDualBranchModel(
        temporal_input_size=config.TEMPORAL_INPUT_SIZE,
        hidden_size=config.FUSION_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        num_time_steps=config.TIME_STEPS,
        dropout=0.2,
        use_hierarchical=True
    )

    model = model.to(device)

    # Move graphs to device (convert numpy arrays to tensors first)
    model.graphs_2021 = [(torch.from_numpy(edge_idx).to(device), torch.from_numpy(edge_attr).to(device))
                         for edge_idx, edge_attr in data['graphs_2021']]
    model.graphs_2024 = [(torch.from_numpy(edge_idx).to(device), torch.from_numpy(edge_attr).to(device))
                         for edge_idx, edge_attr in data['graphs_2024']]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✓ Model created")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Model type: Pure Graph-Based Dual-Branch")
    logger.info(f"  - Spatial branch: Uses only graph structure (no external node features)")

    # Create loss functions with class weights
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Setting up training")
    logger.info("=" * 80)

    class_weights = data['class_weights'].to(device)

    # Compute hierarchical weights
    intensity_weights = torch.zeros(3, device=device)
    direction_weights = torch.zeros(3, device=device)

    for i in range(3):
        intensity_weights[i] = (class_weights[i*3] + class_weights[i*3+1] + class_weights[i*3+2]) / 3
        direction_weights[i] = (class_weights[i] + class_weights[i+3] + class_weights[i+6]) / 3

    criterion_intensity = nn.CrossEntropyLoss(weight=intensity_weights)
    criterion_direction = nn.CrossEntropyLoss(weight=direction_weights)
    criterion_direct = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    logger.info(f"✓ Training setup complete")
    logger.info(f"  - Optimizer: Adam (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    logger.info(f"  - Scheduler: ReduceLROnPlateau (patience=5)")
    logger.info(f"  - Loss: Hierarchical (Intensity + Direction + 0.5*Direct)")
    logger.info(f"  - Gradient accumulation: 4 steps")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Training")
    logger.info("=" * 80)

    best_hierarchical_acc = 0
    best_direct_acc = 0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        # Train
        train_metrics = train_epoch_hierarchical(
            model, train_loader,
            criterion_intensity, criterion_direction, criterion_direct,
            optimizer, device, accumulation_steps=4
        )

        # Validate
        val_metrics = evaluate_hierarchical(
            model, val_loader,
            criterion_intensity, criterion_direction, criterion_direct,
            device
        )

        # Log metrics
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"    - Intensity: {train_metrics['intensity_acc']:.2f}%")
        logger.info(f"    - Direction: {train_metrics['direction_acc']:.2f}%")
        logger.info(f"    - Hierarchical (3×3): {train_metrics['hierarchical_acc']:.2f}%")
        logger.info(f"    - Direct (9-class): {train_metrics['direct_acc']:.2f}%")

        logger.info(f"  Val Accuracy:")
        logger.info(f"    - Intensity: {val_metrics['intensity_acc']:.2f}%")
        logger.info(f"    - Direction: {val_metrics['direction_acc']:.2f}%")
        logger.info(f"    - Hierarchical (3×3): {val_metrics['hierarchical_acc']:.2f}% | F1: {val_metrics['hierarchical_f1']:.4f}")
        logger.info(f"    - Direct (9-class): {val_metrics['direct_acc']:.2f}% | F1: {val_metrics['direct_f1']:.4f}")

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  LR: {current_lr:.6f}")

        # Update scheduler
        scheduler.step(val_metrics['hierarchical_acc'])

        # Save best model
        if val_metrics['hierarchical_acc'] > best_hierarchical_acc:
            best_hierarchical_acc = val_metrics['hierarchical_acc']
            best_direct_acc = val_metrics['direct_acc']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hierarchical_acc': best_hierarchical_acc,
                'direct_acc': best_direct_acc
            }, f"{output_dir}/models/best_model.pth")

            logger.info(f"  ✓ New best model saved!")
        else:
            patience_counter += 1

        logger.info(f"  Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"\n✓ Early stopping at epoch {epoch + 1}")
            break

    logger.info(f"\n✓ Training completed!")
    logger.info(f"  Best Hierarchical Acc: {best_hierarchical_acc:.2f}%")
    logger.info(f"  Best Direct Acc: {best_direct_acc:.2f}%")

    # Test
    logger.info("\n" + "=" * 80)
    logger.info("Step 7: Testing")
    logger.info("=" * 80)

    # Load best model
    checkpoint = torch.load(f"{output_dir}/models/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Evaluate on test set
    test_metrics = evaluate_hierarchical(
        model, test_loader,
        criterion_intensity, criterion_direction, criterion_direct,
        device
    )

    logger.info(f"\nTest Results:")
    logger.info(f"  - Intensity Acc: {test_metrics['intensity_acc']:.2f}%")
    logger.info(f"  - Direction Acc: {test_metrics['direction_acc']:.2f}%")
    logger.info(f"  - Hierarchical Acc: {test_metrics['hierarchical_acc']:.2f}% | F1: {test_metrics['hierarchical_f1']:.4f}")
    logger.info(f"  - Direct Acc: {test_metrics['direct_acc']:.2f}% | F1: {test_metrics['direct_f1']:.4f}")

    # Save test results
    test_results = {
        'test_intensity_acc': float(test_metrics['intensity_acc']),
        'test_direction_acc': float(test_metrics['direction_acc']),
        'test_hierarchical_acc': float(test_metrics['hierarchical_acc']),
        'test_hierarchical_f1': float(test_metrics['hierarchical_f1']),
        'test_direct_acc': float(test_metrics['direct_acc']),
        'test_direct_f1': float(test_metrics['direct_f1'])
    }

    with open(f"{output_dir}/metrics/test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    # Save classification reports
    hierarchical_report = classification_report(
        test_metrics['all_labels'],
        test_metrics['hierarchical_preds'],
        target_names=[f'Class {i+1}' for i in range(9)],
        zero_division=0
    )

    with open(f"{output_dir}/metrics/hierarchical_classification_report.txt", 'w') as f:
        f.write("Hierarchical Classification Report\n")
        f.write("=" * 80 + "\n")
        f.write(hierarchical_report)

    # Save confusion matrices
    hierarchical_cm = confusion_matrix(test_metrics['all_labels'], test_metrics['hierarchical_preds'])
    np.save(f"{output_dir}/metrics/hierarchical_confusion_matrix.npy", hierarchical_cm)

    logger.info(f"\n✓ All results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

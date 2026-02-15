"""
Training script for pure graph-based 9-class model
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


def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=4):
    """
    Train one epoch with 9-class classification

    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function for 9-class classification
        optimizer: Optimizer
        device: Device to use
        accumulation_steps: Number of steps for gradient accumulation

    Returns:
        Dictionary with training metrics
    """
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

        # Move graphs to device (convert numpy arrays to tensors first)
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
            node_indices=node_indices
        )

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass with gradient accumulation
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Compute average metrics
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model with 9-class classification

    Returns:
        Dictionary with evaluation metrics
    """
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

        # Move graphs to device (convert numpy arrays to tensors first)
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
            node_indices=node_indices
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
    """Main training function"""
    logger.info("=" * 80)
    logger.info("Pure Graph-Based 9-Class Training")
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
        # features shape: (7, 2) = [total_2021_log, total_2024_log]
        # Each year gets 1 feature (total flow only)
        temporal_features_2021[grid_id] = features[:, [0]]  # (7, 1) - total for 2021
        temporal_features_2024[grid_id] = features[:, [1]]  # (7, 1) - total for 2024

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
    all_temporal_2021 = torch.zeros(num_nodes, 7, 1)
    all_temporal_2024 = torch.zeros(num_nodes, 7, 1)

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
        dropout=0.2
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

    # Single loss function for 9-class classification
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    logger.info(f"✓ Training setup complete")
    logger.info(f"  - Optimizer: Adam (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    logger.info(f"  - Scheduler: ReduceLROnPlateau (patience=5)")
    logger.info(f"  - Loss: Single 9-class cross-entropy with class weights")
    logger.info(f"  - Gradient accumulation: 4 steps")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Training")
    logger.info("=" * 80)

    best_accuracy = 0
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        # Train
        train_metrics = train_epoch(
            model, train_loader,
            criterion, optimizer, device, accumulation_steps=4
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
                'accuracy': best_accuracy
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

    # Save test results
    test_results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1'])
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
        f.write("9-Class Classification Report\n")
        f.write("=" * 80 + "\n")
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(test_metrics['all_labels'], test_metrics['all_preds'])
    np.save(f"{output_dir}/metrics/confusion_matrix.npy", cm)

    logger.info(f"\n✓ All results saved to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

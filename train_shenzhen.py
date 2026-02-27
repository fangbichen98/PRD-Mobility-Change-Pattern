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
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data

# Adjust flow threshold for Shenzhen data (reduce graph size)
# Shenzhen has much more flow data, so we need higher threshold
# Original labels (3051万 OD): threshold 40 works (~73K-96K edges)
# Sorted labels (4047万 OD): need much higher threshold
# Threshold 50: 119K-163K edges (OOM)
# Threshold 60: 91K-133K edges (OOM)
# Threshold 70: should be ~70K-100K edges
config.FLOW_THRESHOLD = 50.0
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


def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """Main training function"""
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()

    logger.info("=" * 80)
    logger.info("Pure Graph-Based 9-Class Training - Shenzhen Internal Mobility")
    logger.info("=" * 80)
    logger.info(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/shenzhen_pure_graph_{timestamp}"
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
        label_path='data/labels_shenzhen_entropy.csv',
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
        # features shape: (168, 2) = [total_2021_log, total_2024_log] (hourly snapshots)
        # Each year gets 1 feature (total flow only)
        temporal_features_2021[grid_id] = features[:, [0]]  # (168, 1) - total for 2021
        temporal_features_2024[grid_id] = features[:, [1]]  # (168, 1) - total for 2024

    logger.info(f"✓ Temporal features prepared")
    logger.info(f"  - 2021 features: {len(temporal_features_2021)} grids")
    logger.info(f"  - 2024 features: {len(temporal_features_2024)} grids")
    logger.info(f"  - Feature shape: (168, 1) - 168 hourly snapshots")

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
    all_temporal_2021 = torch.zeros(num_nodes, 168, 1)
    all_temporal_2024 = torch.zeros(num_nodes, 168, 1)

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

    # Save test results with configuration
    test_results = {
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1': float(test_metrics['f1']),
        'data_config': {
            'flow_threshold': config.FLOW_THRESHOLD,
            'time_steps': config.TIME_STEPS,
            'time_steps_description': f'{config.TIME_STEPS // 24} days × 24 hours'
        },
        'model_architecture': {
            'temporal_branch': {
                'lstm_layers': config.LSTM_LAYERS,
                'lstm_hidden_size': config.LSTM_HIDDEN_SIZE,
                'lstm_dropout': config.LSTM_DROPOUT,
                'temporal_input_size': config.TEMPORAL_INPUT_SIZE
            },
            'spatial_branch': {
                'gat_layers': config.GAT_LAYERS,
                'gat_hidden_size': config.GAT_HIDDEN_SIZE,
                'gat_heads': config.GAT_HEADS
            },
            'fusion': {
                'fusion_hidden_size': config.FUSION_HIDDEN_SIZE,
                'attention_heads': config.ATTENTION_HEADS
            },
            'output': {
                'num_classes': config.NUM_CLASSES
            }
        },
        'training_config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY,
            'num_epochs': config.NUM_EPOCHS,
            'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
            'train_split': config.TRAIN_SPLIT,
            'val_split': config.VAL_SPLIT,
            'test_split': config.TEST_SPLIT,
            'random_seed': config.RANDOM_SEED
        },
        'data_info': {
            'label_file': data.get('label_file_name', 'N/A'),
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
        f.write("9-Class Classification Report - Shenzhen Internal Mobility\n")
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
        f.write(f"    - LSTM Layers: {config.LSTM_LAYERS}\n")
        f.write(f"    - LSTM Hidden Size: {config.LSTM_HIDDEN_SIZE}\n")
        f.write(f"    - LSTM Dropout: {config.LSTM_DROPOUT}\n")
        f.write(f"    - Temporal Input Size: {config.TEMPORAL_INPUT_SIZE}\n")
        f.write(f"  Spatial Branch:\n")
        f.write(f"    - GAT Layers: {config.GAT_LAYERS}\n")
        f.write(f"    - GAT Hidden Size: {config.GAT_HIDDEN_SIZE}\n")
        f.write(f"    - GAT Heads: {config.GAT_HEADS}\n")
        f.write(f"  Fusion:\n")
        f.write(f"    - Fusion Hidden Size: {config.FUSION_HIDDEN_SIZE}\n")
        f.write(f"    - Attention Heads: {config.ATTENTION_HEADS}\n")
        f.write(f"  Output:\n")
        f.write(f"    - Num Classes: {config.NUM_CLASSES}\n")
        f.write("\n")

        f.write("Training Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {config.LEARNING_RATE}\n")
        f.write(f"  Weight Decay: {config.WEIGHT_DECAY}\n")
        f.write(f"  Num Epochs: {config.NUM_EPOCHS}\n")
        f.write(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}\n")
        f.write(f"  Train/Val/Test Split: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}\n")
        f.write(f"  Random Seed: {config.RANDOM_SEED}\n")
        f.write("\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    # Save confusion matrix
    cm = confusion_matrix(test_metrics['all_labels'], test_metrics['all_preds'])
    np.save(f"{output_dir}/metrics/confusion_matrix.npy", cm)

    logger.info(f"\n✓ All results saved to {output_dir}")
    logger.info("=" * 80)

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

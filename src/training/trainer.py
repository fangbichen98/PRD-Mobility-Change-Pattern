"""
Training pipeline for mobility pattern classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
from datetime import datetime

import config
from src.models.dual_branch_model import DualBranchSTModel, BaselineLSTM, BaselineGAT
from src.training.dataset import MobilityDataset, GraphBatchCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for mobility pattern classification"""

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=config.LEARNING_RATE,
                 weight_decay=config.WEIGHT_DECAY,
                 log_dir=config.LOG_DIR,
                 checkpoint_dir=config.CHECKPOINT_DIR):
        """
        Initialize trainer

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay
            log_dir: Directory for tensorboard logs
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )

        # Logging
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f'run_{timestamp}'))
        self.checkpoint_dir = checkpoint_dir

        # Training state
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0

        # Training history
        self.train_losses = []
        self.train_accs = []
        self.train_f1s = []
        self.val_losses = []
        self.val_accs = []
        self.val_f1s = []

        logger.info(f"Trainer initialized on device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch in pbar:
            # Move data to device
            temporal = batch['temporal'].to(self.device)
            spatial = batch['spatial'].to(self.device)
            labels = batch['labels'].to(self.device)
            edge_index = batch['edge_index'].to(self.device)
            edge_attr = batch['edge_attr'].to(self.device) if batch['edge_attr'] is not None else None
            node_indices = batch['node_indices'].to(self.device) if batch['node_indices'] is not None else None
            all_spatial_features = batch['all_spatial_features'].to(self.device) if batch['all_spatial_features'] is not None else spatial

            # Forward pass
            self.optimizer.zero_grad()

            if isinstance(self.model, (BaselineLSTM,)):
                logits = self.model(temporal)
            elif isinstance(self.model, (BaselineGAT,)):
                # For GAT, use all spatial features
                all_logits = self.model(all_spatial_features, edge_index, edge_attr)
                # Select batch nodes
                if node_indices is not None:
                    logits = all_logits[node_indices]
                else:
                    logits = all_logits[:len(labels)]
            else:
                # Dual-branch model
                logits = self.model(temporal, all_spatial_features, edge_index, edge_attr, node_indices)

            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, accuracy, f1

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

            for batch in pbar:
                # Move data to device
                temporal = batch['temporal'].to(self.device)
                spatial = batch['spatial'].to(self.device)
                labels = batch['labels'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device) if batch['edge_attr'] is not None else None
                node_indices = batch['node_indices'].to(self.device) if batch['node_indices'] is not None else None
                all_spatial_features = batch['all_spatial_features'].to(self.device) if batch['all_spatial_features'] is not None else spatial

                # Forward pass
                if isinstance(self.model, (BaselineLSTM,)):
                    logits = self.model(temporal)
                elif isinstance(self.model, (BaselineGAT,)):
                    # For GAT, use all spatial features
                    all_logits = self.model(all_spatial_features, edge_index, edge_attr)
                    # Select batch nodes
                    if node_indices is not None:
                        logits = all_logits[node_indices]
                    else:
                        logits = all_logits[:len(labels)]
                else:
                    logits = self.model(temporal, all_spatial_features, edge_index, edge_attr, node_indices)

                loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': loss.item()})

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        # Per-class F1 scores
        f1_per_class = f1_score(all_labels, all_preds, average=None)

        return avg_loss, accuracy, f1, f1_per_class, all_preds, all_labels

    def train(self, num_epochs=config.NUM_EPOCHS, early_stopping_patience=config.EARLY_STOPPING_PATIENCE):
        """
        Train model

        Args:
            num_epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_f1, val_f1_per_class, val_preds, val_labels = self.validate(epoch)

            # Record history
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_f1s.append(train_f1)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)

            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('F1/train', train_f1, epoch)
            self.writer.add_scalar('F1/val', val_f1, epoch)

            # Log per-class F1 scores
            for i, f1_score_class in enumerate(val_f1_per_class):
                self.writer.add_scalar(f'F1_Class/class_{i+1}', f1_score_class, epoch)

            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                self.epochs_without_improvement = 0

                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'val_f1_per_class': val_f1_per_class.tolist()
                }, checkpoint_path)

                logger.info(f"  Saved best model (Acc: {val_acc:.4f}, F1: {val_f1:.4f})")

                # Save confusion matrix
                cm = confusion_matrix(val_labels, val_preds)
                np.save(os.path.join(self.checkpoint_dir, 'confusion_matrix.npy'), cm)

            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, checkpoint_path)

        logger.info("Training completed")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")

        self.writer.close()

        return self.best_val_acc, self.best_val_f1


def create_data_loaders(dataset, edge_index, edge_attr, grid_id_to_idx,
                       batch_size=config.BATCH_SIZE,
                       val_split=config.VAL_SPLIT,
                       test_split=config.TEST_SPLIT):
    """
    Create train/val/test data loaders

    Args:
        dataset: MobilityDataset
        edge_index: Graph edge indices
        edge_attr: Edge attributes
        grid_id_to_idx: Grid ID to node index mapping
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio

    Returns:
        train_loader, val_loader, test_loader
    """
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    logger.info(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Build full spatial feature matrix for all nodes
    num_nodes = len(grid_id_to_idx)
    spatial_dim = dataset.grid_flows[dataset.grid_ids[0]].shape[0] * dataset.grid_flows[dataset.grid_ids[0]].shape[1]
    all_spatial_features = torch.zeros(num_nodes, spatial_dim)

    for grid_id, node_idx in grid_id_to_idx.items():
        if grid_id in dataset.grid_flows:
            all_spatial_features[node_idx] = torch.FloatTensor(dataset.grid_flows[grid_id].flatten())

    # Create collator
    collator = GraphBatchCollator(edge_index, edge_attr, grid_id_to_idx, all_spatial_features)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # Set to 0 for Windows compatibility
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

    return train_loader, val_loader, test_loader

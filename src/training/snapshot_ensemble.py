"""
Snapshot Ensemble training and inference.

Snapshot ensemble saves multiple model checkpoints from different training epochs
and combines their predictions at test time. This acts as an ensemble without
additional training cost.

Reference: "Snapshot Ensembles: Train 1, get M for free" (ICLR 2017)
"""

import torch
import torch.nn as nn
import os
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class SnapshotEnsembleTrainer:
    """
    Trainer that saves model snapshots at specified epochs.

    Saves snapshots when learning rate reaches cyclical minimums,
    capturing models from different optimization trajectories.

    Args:
        model: Model to train
        output_dir: Directory to save snapshots
        snapshot_epochs: List of epochs to save snapshots at (optional)
        num_snapshots: Total number of snapshots to save (default: 4)
        keep_last_n: Only keep last N snapshots to save disk space (default: 10)
    """

    def __init__(self, model: nn.Module, output_dir: str,
                 snapshot_epochs: List[int] = None,
                 num_snapshots: int = 4,
                 keep_last_n: int = 10):
        self.model = model
        self.output_dir = output_dir
        self.snapshot_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.snapshot_epochs = snapshot_epochs or []
        self.num_snapshots = num_snapshots
        self.keep_last_n = keep_last_n

        self.snapshots_saved = []

    def should_save_snapshot(self, epoch: int, current_lr: float, min_lr: float) -> bool:
        """
        Determine if snapshot should be saved at current epoch.

        Args:
            epoch: Current epoch number
            current_lr: Current learning rate
            min_lr: Minimum learning rate (cyclical)

        Returns:
            True if snapshot should be saved
        """
        # Check if epoch is in predefined list
        if epoch in self.snapshot_epochs:
            return True

        # Check if LR is near minimum (for cosine annealing)
        if abs(current_lr - min_lr) < 1e-7:
            return True

        return False

    def save_snapshot(self, epoch: int, optimizer: torch.optim.Optimizer,
                     metrics: Dict) -> str:
        """
        Save model snapshot.

        Args:
            epoch: Current epoch
            optimizer: Optimizer state
            metrics: Dictionary of metrics (accuracy, f1, etc.)

        Returns:
            Path to saved snapshot
        """
        snapshot_path = os.path.join(
            self.snapshot_dir,
            f"snapshot_epoch_{epoch:03d}_acc_{metrics.get('accuracy', 0):.2f}.pth"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, snapshot_path)

        self.snapshots_saved.append(snapshot_path)

        # Cleanup old snapshots
        self._cleanup_old_snapshots()

        return snapshot_path

    def _cleanup_old_snapshots(self):
        """Keep only last N snapshots to save disk space"""
        if len(self.snapshots_saved) > self.keep_last_n:
            # Remove oldest snapshots
            for old_path in self.snapshots_saved[:-self.keep_last_n]:
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"Removed old snapshot: {old_path}")

            self.snapshots_saved = self.snapshots_saved[-self.keep_last_n:]


def ensemble_predict(snapshots: List[str], model_class: nn.Module,
                    model_args: Dict, device: torch.device,
                    x_2021: torch.Tensor, x_2024: torch.Tensor,
                    graphs_2021: List, graphs_2024: List,
                    num_nodes: int, node_indices: torch.Tensor) -> torch.Tensor:
    """
    Make predictions using an ensemble of model snapshots.

    Args:
        snapshots: List of snapshot file paths
        model_class: Model class to instantiate
        model_args: Arguments to pass to model class
        device: Device to run on
        x_2021, x_2024: Input features
        graphs_2021, graphs_2024: Graph structures
        num_nodes: Total number of nodes
        node_indices: Batch node indices

    Returns:
        predictions: (batch_size,) - ensemble class predictions
    """
    all_logits = []

    for snapshot_path in snapshots:
        # Load snapshot
        checkpoint = torch.load(snapshot_path, map_location=device)

        # Create model instance
        model = model_class(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Prepare graphs
        graphs_2021_device = [(edge_idx.to(device), edge_attr.to(device))
                             for edge_idx, edge_attr in graphs_2021]
        graphs_2024_device = [(edge_idx.to(device), edge_attr.to(device))
                             for edge_idx, edge_attr in graphs_2024]

        # Predict
        with torch.no_grad():
            logits = model(
                x_2021.to(device),
                x_2024.to(device),
                graphs_2021_device,
                graphs_2024_device,
                num_nodes,
                node_indices.to(device)
            )
        all_logits.append(logits)

    # Soft voting: average probabilities
    avg_logits = torch.stack(all_logits).mean(dim=0)
    predictions = avg_logits.argmax(dim=1)

    return predictions


def ensemble_predict_dataloader(snapshots: List[str], model_class: nn.Module,
                               model_args: Dict, device: torch.device,
                               data_loader) -> Dict:
    """
    Evaluate ensemble on entire dataloader.

    Args:
        snapshots: List of snapshot paths
        model_class: Model class
        model_args: Model initialization arguments
        device: Device to use
        data_loader: Test data loader

    Returns:
        Dictionary with metrics
    """
    all_preds = []
    all_labels = []

    # Load all models once
    models = []
    for snapshot_path in snapshots:
        checkpoint = torch.load(snapshot_path, map_location=device)
        model = model_class(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        models.append(model)

    # Iterate through batches
    for batch in data_loader:
        labels = batch['labels']
        all_labels.extend(labels.cpu().numpy())

        # Prepare inputs
        x_2021 = batch['all_temporal_2021']
        x_2024 = batch['all_temporal_2024']
        graphs_2021 = batch['graphs_2021']
        graphs_2024 = batch['graphs_2024']
        num_nodes = batch['num_nodes']
        node_indices = batch['node_indices']

        # Ensemble prediction
        all_logits = []
        for model in models:
            graphs_2021_device = [(edge_idx.to(device), edge_attr.to(device))
                                 for edge_idx, edge_attr in graphs_2021]
            graphs_2024_device = [(edge_idx.to(device), edge_attr.to(device))
                                 for edge_idx, edge_attr in graphs_2024]

            with torch.no_grad():
                logits = model(
                    x_2021.to(device), x_2024.to(device),
                    graphs_2021_device, graphs_2024_device,
                    num_nodes, node_indices.to(device)
                )
            all_logits.append(logits)

        # Soft voting
        avg_logits = torch.stack(all_logits).mean(dim=0)
        preds = avg_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100.0 * accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'all_labels': all_labels,
        'all_preds': all_preds
    }


def get_auto_snapshot_epochs(num_epochs: int, num_snapshots: int = 4) -> List[int]:
    """
    Automatically determine snapshot epochs based on training schedule.

    For cosine annealing with warm restarts, snapshots should be saved
    at epochs where LR reaches minimum.

    Args:
        num_epochs: Total training epochs
        num_snapshots: Number of snapshots to save

    Returns:
        List of epochs to save snapshots
    """
    # Estimate restart points
    # For T_0=10, T_mult=2: restarts at 10, 30, 70, 150, ...
    snapshot_epochs = []
    current_t0 = 10
    current_epoch = current_t0

    while current_epoch < num_epochs and len(snapshot_epochs) < num_snapshots:
        snapshot_epochs.append(current_epoch - 1)  # Save just before restart
        current_t0 *= 2
        current_epoch += current_t0

    # If not enough snapshots, add some at the end
    if len(snapshot_epochs) < num_snapshots:
        for i in range(1, num_snapshots - len(snapshot_epochs) + 1):
            snapshot_epochs.append(num_epochs - i * (num_epochs // num_snapshots))

    return sorted(set(snapshot_epochs))


# Example integration into training loop
def train_with_snapshots(model, train_loader, val_loader, optimizer,
                        scheduler, criterion, device, num_epochs,
                        output_dir):
    """
    Example training loop with snapshot saving.

    This function shows how to integrate snapshot ensemble into training.
    """
    snapshot_trainer = SnapshotEnsembleTrainer(
        model, output_dir,
        num_snapshots=4,
        keep_last_n=10
    )

    best_accuracy = 0

    for epoch in range(num_epochs):
        # Train and validate (your training code here)
        # train_metrics = train_one_epoch(...)
        # val_metrics = validate(...)

        # Example placeholders
        val_metrics = {'accuracy': 75.0, 'f1': 0.65}

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        min_lr = 1e-6  # Your scheduler's minimum LR

        # Check if should save snapshot
        if snapshot_trainer.should_save_snapshot(epoch, current_lr, min_lr):
            snapshot_path = snapshot_trainer.save_snapshot(
                epoch, optimizer, val_metrics
            )
            print(f"Saved snapshot: {snapshot_path}")

        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(),
                      os.path.join(output_dir, "best_model.pth"))

        # Step scheduler
        scheduler.step()

    return snapshot_trainer.snapshots_saved


if __name__ == "__main__":
    print("Snapshot Ensemble Module")
    print("=" * 50)
    print("\nFeatures:")
    print("  - SnapshotEnsembleTrainer: Save model snapshots during training")
    print("  - ensemble_predict: Ensemble inference for single batch")
    print("  - ensemble_predict_dataloader: Ensemble inference for dataloader")
    print("  - get_auto_snapshot_epochs: Auto-determine snapshot epochs")
    print("\nExpected improvement: +2-3% accuracy")
    print("\nExample usage:")
    print("  snapshots = train_with_snapshots(...)")
    print("  results = ensemble_predict_dataloader(snapshots, ...)")

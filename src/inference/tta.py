"""
Test-Time Augmentation (TTA) for improved prediction robustness.

Performs multiple predictions with augmented inputs and aggregates the results.
This can improve accuracy by 1-2% through variance reduction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class TestTimeAugmentation:
    """
    Test-Time Augmentation for time-series graph models.

    Performs multiple predictions with different noise augmentations
    and aggregates results using soft voting.

    Args:
        model: The trained model to use for prediction
        device: Device to run inference on
        n_augment: Number of augmentations to perform (default: 5)
        noise_level: Standard deviation of Gaussian noise (default: 0.05 = 5%)
        random_state: Random seed for reproducibility
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 n_augment: int = 5, noise_level: float = 0.05,
                 random_state: int = 42):
        self.model = model
        self.device = device
        self.n_augment = n_augment
        self.noise_level = noise_level
        self.rng = np.random.RandomState(random_state)

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to input tensor.

        Args:
            x: Input tensor of shape (batch, timesteps, features)

        Returns:
            Noisy tensor of same shape
        """
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

    @torch.no_grad()
    def predict_single(self, x_2021: torch.Tensor, x_2024: torch.Tensor,
                      graphs_2021: List, graphs_2024: List,
                      num_nodes: int, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Make a single prediction.

        Args:
            x_2021: 2021 features (batch, 168, 1)
            x_2024: 2024 features (batch, 168, 1)
            graphs_2021: 2021 graph edges
            graphs_2024: 2024 graph edges
            num_nodes: Total number of nodes
            node_indices: Node indices for batch

        Returns:
            logits: (batch, num_classes)
        """
        self.model.eval()

        # Ensure graphs are on correct device (handle both tensors and numpy arrays)
        graphs_2021_device = [(torch.from_numpy(edge_idx).to(self.device) if isinstance(edge_idx, np.ndarray) else
                               (edge_idx.to(self.device) if hasattr(edge_idx, 'to') else edge_idx),
                               torch.from_numpy(edge_attr).to(self.device) if isinstance(edge_attr, np.ndarray) else
                               (edge_attr.to(self.device) if hasattr(edge_attr, 'to') else edge_attr))
                             for edge_idx, edge_attr in graphs_2021]
        graphs_2024_device = [(torch.from_numpy(edge_idx).to(self.device) if isinstance(edge_idx, np.ndarray) else
                               (edge_idx.to(self.device) if hasattr(edge_idx, 'to') else edge_idx),
                               torch.from_numpy(edge_attr).to(self.device) if isinstance(edge_attr, np.ndarray) else
                               (edge_attr.to(self.device) if hasattr(edge_attr, 'to') else edge_attr))
                             for edge_idx, edge_attr in graphs_2024]

        # Forward pass
        logits = self.model(
            x_2021.to(self.device),
            x_2024.to(self.device),
            graphs_2021_device,
            graphs_2024_device,
            num_nodes,
            node_indices.to(self.device)
        )

        return logits

    @torch.no_grad()
    def predict(self, x_2021: torch.Tensor, x_2024: torch.Tensor,
               graphs_2021: List, graphs_2024: List,
               num_nodes: int, node_indices: torch.Tensor,
               return_probs: bool = False) -> torch.Tensor:
        """
        Make prediction with test-time augmentation.

        Performs n_augment predictions with different noise and aggregates.

        Args:
            x_2021: 2021 features (batch, 168, 1)
            x_2024: 2024 features (batch, 168, 1)
            graphs_2021: 2021 graph edges
            graphs_2024: 2024 graph edges
            num_nodes: Total number of nodes
            node_indices: Node indices for batch
            return_probs: If True, return probabilities; if False, return class predictions

        Returns:
            predictions: (batch,) if return_probs=False, (batch, num_classes) otherwise
        """
        self.model.eval()

        all_logits = []

        # Original prediction (no noise)
        logits_orig = self.predict_single(
            x_2021, x_2024,
            graphs_2021, graphs_2024,
            num_nodes, node_indices
        )
        all_logits.append(logits_orig)

        # Augmented predictions
        for _ in range(self.n_augment - 1):
            # Add noise to inputs
            x_2021_noisy = self.add_noise(x_2021)
            x_2024_noisy = self.add_noise(x_2024)

            # Predict with noisy inputs
            logits_noisy = self.predict_single(
                x_2021_noisy, x_2024_noisy,
                graphs_2021, graphs_2024,
                num_nodes, node_indices
            )
            all_logits.append(logits_noisy)

        # Average predictions (soft voting)
        avg_logits = torch.stack(all_logits).mean(dim=0)  # (batch, num_classes)

        if return_probs:
            # Return probabilities
            probs = torch.softmax(avg_logits, dim=1)
            return probs
        else:
            # Return class predictions
            predictions = avg_logits.argmax(dim=1)  # (batch,)
            return predictions


class AdvancedTTA:
    """
    Advanced Test-Time Augmentation with multiple augmentation strategies.

    Combines:
    1. Gaussian noise augmentation
    2. Time shift augmentation
    3. Scaling augmentation

    Provides more diverse augmentations for better robustness.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 n_augment: int = 7, noise_level: float = 0.05,
                 shift_max: int = 2, scale_range: Tuple[float, float] = (0.95, 1.05)):
        self.model = model
        self.device = device
        self.n_augment = n_augment
        self.noise_level = noise_level
        self.shift_max = shift_max
        self.scale_range = scale_range

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random time shift"""
        shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()

        if shift == 0:
            return x

        if shift > 0:
            # Shift forward
            x_shifted = torch.zeros_like(x)
            x_shifted[:, shift:, :] = x[:, :-shift, :]
            x_shifted[:, :shift, :] = x[:, 0:1, :].expand(-1, shift, -1)
        else:
            # Shift backward
            shift = abs(shift)
            x_shifted = torch.zeros_like(x)
            x_shifted[:, :-shift, :] = x[:, shift:, :]
            x_shifted[:, -shift:, :] = x[:, -1:, :].expand(-1, shift, -1)

        return x_shifted

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random scaling"""
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        return x * scale

    @torch.no_grad()
    def predict(self, x_2021: torch.Tensor, x_2024: torch.Tensor,
               graphs_2021: List, graphs_2024: List,
               num_nodes: int, node_indices: torch.Tensor) -> torch.Tensor:
        """Advanced TTA prediction"""
        self.model.eval()

        all_logits = []

        # Original
        logits = self._predict_single(x_2021, x_2024, graphs_2021, graphs_2024,
                                      num_nodes, node_indices)
        all_logits.append(logits)

        # Noise augmentations
        for _ in range(2):
            x_2021_noisy = x_2021 + torch.randn_like(x_2021) * self.noise_level
            x_2024_noisy = x_2024 + torch.randn_like(x_2024) * self.noise_level
            logits = self._predict_single(x_2021_noisy, x_2024_noisy,
                                         graphs_2021, graphs_2024,
                                         num_nodes, node_indices)
            all_logits.append(logits)

        # Time shift augmentations
        for _ in range(2):
            x_2021_shifted = self.time_shift(x_2021)
            x_2024_shifted = self.time_shift(x_2024)
            logits = self._predict_single(x_2021_shifted, x_2024_shifted,
                                         graphs_2021, graphs_2024,
                                         num_nodes, node_indices)
            all_logits.append(logits)

        # Scaling augmentations
        for _ in range(2):
            scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
            x_2021_scaled = x_2021 * scale_factor
            x_2024_scaled = x_2024 * scale_factor
            logits = self._predict_single(x_2021_scaled, x_2024_scaled,
                                         graphs_2021, graphs_2024,
                                         num_nodes, node_indices)
            all_logits.append(logits)

        # Average all predictions
        avg_logits = torch.stack(all_logits).mean(dim=0)
        predictions = avg_logits.argmax(dim=1)

        return predictions

    @torch.no_grad()
    def _predict_single(self, x_2021, x_2024, graphs_2021, graphs_2024,
                       num_nodes, node_indices):
        """Single prediction helper"""
        graphs_2021_device = [(torch.from_numpy(edge_idx).to(self.device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(self.device),
                               torch.from_numpy(edge_attr).to(self.device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(self.device))
                             for edge_idx, edge_attr in graphs_2021]
        graphs_2024_device = [(torch.from_numpy(edge_idx).to(self.device) if isinstance(edge_idx, np.ndarray) else edge_idx.to(self.device),
                               torch.from_numpy(edge_attr).to(self.device) if isinstance(edge_attr, np.ndarray) else edge_attr.to(self.device))
                             for edge_idx, edge_attr in graphs_2024]

        logits = self.model(
            x_2021.to(self.device), x_2024.to(self.device),
            graphs_2021_device, graphs_2024_device,
            num_nodes, node_indices.to(self.device)
        )
        return logits


def evaluate_with_tta(model: nn.Module, data_loader, device: torch.device,
                     n_augment: int = 5) -> dict:
    """
    Evaluate model with test-time augmentation.

    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device to use
        n_augment: Number of TTA augmentations

    Returns:
        Dictionary with metrics
    """
    tta = TestTimeAugmentation(model, device, n_augment=n_augment)

    all_preds = []
    all_labels = []

    for batch in data_loader:
        labels = batch['labels'].to(device)
        all_labels.extend(labels.cpu().numpy())

        # TTA prediction
        preds = tta.predict(
            batch['all_temporal_2021'],
            batch['all_temporal_2024'],
            batch['graphs_2021'],
            batch['graphs_2024'],
            batch['num_nodes'],
            batch['node_indices']
        )

        all_preds.extend(preds.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from sklearn.metrics import f1_score, accuracy_score

    accuracy = 100.0 * accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'all_labels': all_labels,
        'all_preds': all_preds
    }


# Example usage
if __name__ == "__main__":
    print("Test-Time Augmentation Module")
    print("=" * 50)
    print("\nThis module provides TTA functionality:")
    print("  - TestTimeAugmentation: Basic noise-based TTA")
    print("  - AdvancedTTA: Multi-strategy TTA (noise, shift, scale)")
    print("  - evaluate_with_tta: Evaluate dataloader with TTA")
    print("\nExpected improvement: +1-2% accuracy")

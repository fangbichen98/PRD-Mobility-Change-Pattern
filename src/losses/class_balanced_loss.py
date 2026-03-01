"""
Class-Balanced Focal Loss for handling extreme class imbalance.

Combines:
1. Class-Balanced Loss (CVPR 2019): Effective number of samples weighting
2. Focal Loss (ICCV 2017): Down-weights easy examples

Reference:
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
- Focal Loss: https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for imbalanced classification.

    Args:
        samples_per_class (list): Number of samples for each class
        num_classes (int): Total number of classes
        beta (float): Hyperparameter for effective number formula (default: 0.9999)
                     Higher values (0.9999) work better for extreme imbalance
        gamma (float): Focusing parameter for focal loss (default: 2.0)
                     gamma=0 is equivalent to CE loss, gamma=2 is standard
        label_smoothing (float): Label smoothing factor (default: 0.1)
    """

    def __init__(self, samples_per_class, num_classes=9, beta=0.9999, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # Compute effective number of samples (Class-Balanced Loss)
        # Effective number = (1 - beta^n) / (1 - beta)
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)

        # Normalize weights
        weights = weights / np.sum(weights) * num_classes

        self.register_buffer('weights', torch.FloatTensor(weights))

    def forward(self, logits, labels):
        """
        Compute class-balanced focal loss.

        Args:
            logits: (batch_size, num_classes) raw model outputs
            labels: (batch_size,) ground truth class indices

        Returns:
            loss: scalar tensor
        """
        # Convert labels to one-hot with smoothing
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        # Apply label smoothing
        if self.label_smoothing > 0:
            one_hot_labels = one_hot_labels * (1 - self.label_smoothing) + \
                           self.label_smoothing / self.num_classes

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Compute weighted cross-entropy with class-balanced weights
        ce_loss = -one_hot_labels * log_probs  # (batch, num_classes)

        # Apply class weights
        weighted_loss = ce_loss * self.weights.unsqueeze(0)  # Broadcast weights

        # Sum over classes (after weighting)
        ce_loss = weighted_loss.sum(dim=1)  # (batch,)

        # Compute p_t (probability of true class) for focal term
        probs = F.softmax(logits, dim=1)
        p_t = probs[range(len(labels)), labels]

        # Focal loss: (1 - p_t)^gamma * CE
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    Simplified Class-Balanced Loss without focal term.

    Use this if focal loss causes training instability.
    """

    def __init__(self, samples_per_class, num_classes=9, beta=0.9999, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes

        self.register_buffer('weights', torch.FloatTensor(weights))

    def forward(self, logits, labels):
        """Compute class-balanced cross-entropy loss."""
        # One-hot with smoothing
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        if self.label_smoothing > 0:
            one_hot_labels = one_hot_labels * (1 - self.label_smoothing) + \
                           self.label_smoothing / self.num_classes

        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = -(one_hot_labels * log_probs).sum(dim=1)

        # Apply weights
        weights = self.weights[labels]
        weighted_loss = weights * ce_loss

        return weighted_loss.mean()


def get_class_balanced_loss(samples_per_class, use_focal=True, **kwargs):
    """
    Factory function to create class-balanced loss.

    Args:
        samples_per_class: List of sample counts per class
        use_focal: If True, use focal loss; otherwise use standard CB loss
        **kwargs: Additional arguments (beta, gamma, label_smoothing)

    Returns:
        Loss function
    """
    if use_focal:
        return ClassBalancedFocalLoss(samples_per_class, **kwargs)
    else:
        return ClassBalancedLoss(samples_per_class, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test with the actual class distribution from the project
    samples_per_class = [857, 4, 5, 1241, 57, 188, 1384, 222, 42]

    # Create loss functions
    loss_focal = ClassBalancedFocalLoss(samples_per_class, num_classes=9, beta=0.9999, gamma=2.0)
    loss_standard = ClassBalancedLoss(samples_per_class, num_classes=9, beta=0.9999)

    # Test with dummy data
    batch_size = 16
    num_classes = 9

    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Compute losses
    loss1 = loss_focal(logits, labels)
    loss2 = loss_standard(logits, labels)

    print(f"Class-Balanced Focal Loss: {loss1.item():.4f}")
    print(f"Class-Balanced Standard Loss: {loss2.item():.4f}")
    print(f"Effective weights: {loss_focal.weights.numpy()}")

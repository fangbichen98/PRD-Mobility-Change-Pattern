"""
Data augmentation techniques for time-series mobility data.

Focuses on handling extreme class imbalance through synthetic sample generation.
"""

import numpy as np
from typing import Tuple
import random


class TemporalSMOTE:
    """
    Temporal SMOTE (Synthetic Minority Over-sampling Technique) for time-series.

    Generates synthetic samples for minority classes by:
    1. Selecting a random sample from the minority class
    2. Adding controlled Gaussian noise to create a new sample
    3. Preserving temporal structure and relationships

    This is simpler than traditional SMOTE's interpolation approach,
    but works better for high-dimensional time-series data.
    """

    def __init__(self, noise_ratio=0.1, random_state=42):
        """
        Args:
            noise_ratio (float): Standard deviation of noise as ratio of feature std (default: 0.1 = 10%)
            random_state (int): Random seed for reproducibility
        """
        self.noise_ratio = noise_ratio
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, features_2021, features_2024, labels, target_samples=200):
        """
        Oversample minority classes to reach target_samples.

        Args:
            features_2021: (n_samples, 168, 1) features from 2021
            features_2024: (n_samples, 168, 1) features from 2024
            labels: (n_samples,) class labels
            target_samples: Minimum number of samples per class (default: 200)

        Returns:
            features_2021_aug: (n_samples_augmented, 168, 1)
            features_2024_aug: (n_samples_augmented, 168, 1)
            labels_aug: (n_samples_augmented,)
        """
        unique, counts = np.unique(labels, return_counts=True)
        minority_classes = unique[counts < target_samples]

        print(f"TemporalSMOTE: Found {len(minority_classes)} minority classes out of {len(unique)} total")
        for cls in minority_classes:
            cls_count = np.sum(labels == cls)
            print(f"  Class {cls}: {cls_count} -> {target_samples} (adding {target_samples - cls_count} samples)")

        # Ensure 3D arrays
        if features_2021.ndim == 2:
            features_2021 = np.expand_dims(features_2021, axis=-1)
        if features_2024.ndim == 2:
            features_2024 = np.expand_dims(features_2024, axis=-1)

        # Lists to collect new samples
        new_samples_2021 = []
        new_samples_2024 = []
        new_labels = list(labels)  # Start with original labels

        # Generate synthetic samples for each minority class
        for cls in minority_classes:
            cls_indices = np.where(labels == cls)[0]
            cls_features_2021 = features_2021[cls_indices]
            cls_features_2024 = features_2024[cls_indices]

            n_current = len(cls_indices)
            n_needed = target_samples - n_current

            # Compute noise level from class features
            all_cls_features = np.concatenate([cls_features_2021, cls_features_2024], axis=0)
            noise_std = np.std(all_cls_features) * self.noise_ratio

            # Generate synthetic samples
            for _ in range(n_needed):
                # Randomly select a base sample
                idx = self.rng.choice(n_current)
                sample_2021 = cls_features_2021[idx].copy()
                sample_2024 = cls_features_2024[idx].copy()

                # Add Gaussian noise
                noisy_2021 = sample_2021 + self.rng.normal(0, noise_std, sample_2021.shape)
                noisy_2024 = sample_2024 + self.rng.normal(0, noise_std, sample_2024.shape)

                # Clip to prevent extreme values (safety check)
                noisy_2021 = np.clip(noisy_2021, -10, 10)
                noisy_2024 = np.clip(noisy_2024, -10, 10)

                # Add to lists - expand dims to make them (1, 168, 1)
                new_samples_2021.append(np.expand_dims(noisy_2021, axis=0))
                new_samples_2024.append(np.expand_dims(noisy_2024, axis=0))
                new_labels.append(cls)

        # Concatenate all new samples
        if new_samples_2021:
            new_features_2021 = np.concatenate(new_samples_2021, axis=0)
            new_features_2024 = np.concatenate(new_samples_2024, axis=0)

            # Concatenate with original data
            features_2021_aug = np.concatenate([features_2021, new_features_2021], axis=0)
            features_2024_aug = np.concatenate([features_2024, new_features_2024], axis=0)
        else:
            features_2021_aug = features_2021
            features_2024_aug = features_2024

        labels_aug = np.array(new_labels)

        print(f"TemporalSMOTE: Original {len(labels)} -> Augmented {len(labels_aug)} samples")

        return features_2021_aug, features_2024_aug, labels_aug


class MixupAugmentation:
    """
    Mixup augmentation for time-series.

    Mixup: x = lambda * x_i + (1 - lambda) * x_j
    where lambda ~ Beta(alpha, alpha)

    Reference: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha=0.2, random_state=42):
        """
        Args:
            alpha (float): Beta distribution parameter (default: 0.2)
                         Lower values = more mixing between diverse samples
            random_state (int): Random seed
        """
        self.alpha = alpha
        self.rng = np.random.RandomState(random_state)

    def __call__(self, features_2021, features_2024, labels):
        """
        Apply mixup augmentation.

        Args:
            features_2021: (n_samples, 7, 1)
            features_2024: (n_samples, 7, 1)
            labels: (n_samples,)

        Returns:
            Mixed features and labels (same shape)
        """
        n_samples = len(labels)

        # Sample lambda from Beta distribution
        lam = self.rng.beta(self.alpha, self.alpha, n_samples)

        # Shuffle indices for mixing
        indices = self.rng.permutation(n_samples)

        # Mix features
        features_2021_mixed = lam[:, None, None] * features_2021 + \
                             (1 - lam[:, None, None]) * features_2021[indices]
        features_2024_mixed = lam[:, None, None] * features_2024 + \
                             (1 - lam[:, None, None]) * features_2024[indices]

        # Mix labels (for soft labels in loss)
        labels_a = labels
        labels_b = labels[indices]

        return features_2021_mixed, features_2024_mixed, (labels_a, labels_b, lam)


class TimeShiftAugmentation:
    """
    Time shift augmentation for time-series.

    Randomly shifts the time series by a few timesteps.
    Useful for making the model robust to temporal misalignments.
    """

    def __init__(self, max_shift=2, random_state=42):
        """
        Args:
            max_shift (int): Maximum number of timesteps to shift (default: 2)
            random_state (int): Random seed
        """
        self.max_shift = max_shift
        self.rng = np.random.RandomState(random_state)

    def __call__(self, features):
        """
        Apply random time shift.

        Args:
            features: (n_samples, timesteps, features)

        Returns:
            Shifted features: same shape
        """
        n_samples, timesteps, n_features = features.shape
        shifted = np.zeros_like(features)

        for i in range(n_samples):
            shift = self.rng.randint(-self.max_shift, self.max_shift + 1)

            if shift == 0:
                shifted[i] = features[i]
            elif shift > 0:
                # Shift forward (earlier timesteps become zero)
                shifted[i, shift:, :] = features[i, :-shift, :]
                # Pad with repeated values
                shifted[i, :shift, :] = features[i, 0:1, :]
            else:
                # Shift backward (later timesteps become zero)
                shifted[i, :shift, :] = features[i, -shift:, :]
                # Pad with repeated values
                shifted[i, shift:, :] = features[i, -1:, :]

        return shifted


def augment_minority_classes(features_2021, features_2024, labels,
                            method='smote', target_samples=200, **kwargs):
    """
    High-level function for minority class augmentation.

    Args:
        features_2021, features_2024: Input features
        labels: Class labels
        method: 'smote' (default), 'mixup', 'timeshift', or None
        target_samples: Target samples per class for SMOTE
        **kwargs: Additional arguments for augmentation method

    Returns:
        Augmented features and labels
    """
    if method is None:
        return features_2021, features_2024, labels

    if method == 'smote':
        augmentor = TemporalSMOTE(**kwargs)
        return augmentor.fit_resample(features_2021, features_2024, labels, target_samples)

    elif method == 'mixup':
        augmentor = MixupAugmentation(**kwargs)
        return augmentor(features_2021, features_2024, labels)

    elif method == 'timeshift':
        augmentor = TimeShiftAugmentation(**kwargs)
        features_2021_aug = augmentor(features_2021)
        features_2024_aug = augmentor(features_2024)
        return features_2021_aug, features_2024_aug, labels

    else:
        raise ValueError(f"Unknown augmentation method: {method}")


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    n_samples = 100
    timesteps = 7
    features_2021 = np.random.randn(n_samples, timesteps, 1)
    features_2024 = np.random.randn(n_samples, timesteps, 1)

    # Create imbalanced labels (similar to actual problem)
    labels = np.array([0] * 40 + [1] * 2 + [2] * 3 + [3] * 35 + [4] * 20)

    print("Original class distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")

    # Apply SMOTE
    smote = TemporalSMOTE(noise_ratio=0.1)
    f1, f2, l = smote.fit_resample(features_2021, features_2024, labels, target_samples=50)

    print("\nAugmented class distribution:")
    unique, counts = np.unique(l, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")

    print(f"\nOriginal shape: {features_2021.shape}")
    print(f"Augmented shape: {f1.shape}")

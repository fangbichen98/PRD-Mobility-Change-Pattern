"""
Gated feature fusion for dynamic feature selection.

Uses a gating mechanism to learn the importance of different features,
allowing the model to dynamically select which features to emphasize.

This is more interpretable and potentially more effective than simple
attention mechanisms for feature fusion.
"""

import torch
import torch.nn as nn


class GatedFeatureFusion(nn.Module):
    """
    Gated feature fusion layer.

    Learns a gate for each input feature, controlling how much each feature
    contributes to the final representation.

    Args:
        feature_size: Size of each feature vector (default: 256)
        num_features: Number of features to fuse (default: 9)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self, feature_size=256, num_features=9, dropout=0.3):
        super().__init__()

        self.feature_size = feature_size
        self.num_features = num_features

        # Gate network: outputs gating coefficients for each feature
        self.gate_network = nn.Sequential(
            nn.Linear(feature_size * num_features, feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, num_features),
            nn.Sigmoid()  # Gates in [0, 1]
        )

        # Feature transformation network
        self.transform_network = nn.Sequential(
            nn.Linear(feature_size * num_features, feature_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size * 2, feature_size)
        )

        # Residual connection
        self.residual_proj = nn.Linear(feature_size * num_features, feature_size)

        # Normalization
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, features):
        """
        Forward pass with gated fusion.

        Args:
            features: (batch, num_features, feature_size)

        Returns:
            output: (batch, feature_size)
        """
        batch_size = features.size(0)

        # Flatten features
        flat_features = features.view(batch_size, -1)  # (batch, num_features * feature_size)

        # Compute gates
        gates = self.gate_network(flat_features)  # (batch, num_features)

        # Reshape gates for broadcasting
        gates = gates.unsqueeze(-1)  # (batch, num_features, 1)

        # Apply gates to features
        gated_features = features * gates  # (batch, num_features, feature_size)

        # Flatten gated features
        gated_flat = gated_features.view(batch_size, -1)  # (batch, num_features * feature_size)

        # Transform gated features
        transformed = self.transform_network(gated_flat)  # (batch, feature_size)

        # Residual connection
        residual = self.residual_proj(flat_features)  # (batch, feature_size)

        # Combine with residual
        output = transformed + residual

        # Layer normalization
        output = self.layer_norm(output)

        return output


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive gated fusion with multi-head attention.

    Combines gating mechanism with multi-head attention for more expressive fusion.
    """

    def __init__(self, feature_size=256, num_features=9, num_heads=4, dropout=0.3):
        super().__init__()

        self.feature_size = feature_size
        self.num_features = num_features

        # Multi-head attention for feature interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(feature_size, feature_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size // 4, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_size, feature_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size * 2, feature_size)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feature_size)

    def forward(self, features):
        """
        Forward pass with adaptive gated fusion.

        Args:
            features: (batch, num_features, feature_size)

        Returns:
            output: (batch, feature_size)
        """
        # Multi-head attention
        attended, _ = self.attention(features, features, features)
        # attended: (batch, num_features, feature_size)

        # Compute gate for each feature
        gates = self.gate_network(attended)  # (batch, num_features, 1)

        # Apply gates
        gated = attended * gates  # (batch, num_features, feature_size)

        # Aggregate by summing gated features
        aggregated = gated.sum(dim=1)  # (batch, feature_size)

        # Output projection
        output = self.output_proj(aggregated)

        # Layer norm
        output = self.layer_norm(output)

        return output


class DynamicWeightFusion(nn.Module):
    """
    Dynamic weight-based fusion.

    Learns importance weights for each feature dynamically based on the input.
    Simpler than gated fusion but still adaptive.
    """

    def __init__(self, feature_size=256, num_features=9, dropout=0.3):
        super().__init__()

        self.num_features = num_features

        # Network to compute dynamic weights
        self.weight_network = nn.Sequential(
            nn.Linear(feature_size * num_features, feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, num_features),
            nn.Softmax(dim=1)  # Weights sum to 1
        )

        # Feature transformation
        self.transform = nn.Linear(feature_size * num_features, feature_size)

    def forward(self, features):
        """
        Forward pass with dynamic weight fusion.

        Args:
            features: (batch, num_features, feature_size)

        Returns:
            output: (batch, feature_size)
        """
        batch_size = features.size(0)

        # Flatten features
        flat = features.view(batch_size, -1)  # (batch, num_features * feature_size)

        # Compute dynamic weights
        weights = self.weight_network(flat)  # (batch, num_features)

        # Reshape for broadcasting
        weights = weights.unsqueeze(-1)  # (batch, num_features, 1)

        # Apply weights
        weighted = features * weights  # (batch, num_features, feature_size)

        # Sum over features
        summed = weighted.sum(dim=1)  # (batch, feature_size)

        # Transform
        output = self.transform(flat) + summed  # Include direct connection

        return output


# Test the modules
if __name__ == "__main__":
    # Test with dummy data
    batch_size = 8
    num_features = 9
    feature_size = 256

    features = torch.randn(batch_size, num_features, feature_size)

    # Test GatedFeatureFusion
    print("Testing GatedFeatureFusion:")
    model1 = GatedFeatureFusion(feature_size=feature_size, num_features=num_features)
    output1 = model1(features)
    print(f"  Input shape: {features.shape}")
    print(f"  Output shape: {output1.shape}")
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"  Parameters: {params1:,}")

    # Test AdaptiveGatedFusion
    print("\nTesting AdaptiveGatedFusion:")
    model2 = AdaptiveGatedFusion(feature_size=feature_size, num_features=num_features)
    output2 = model2(features)
    print(f"  Output shape: {output2.shape}")
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"  Parameters: {params2:,}")

    # Test DynamicWeightFusion
    print("\nTesting DynamicWeightFusion:")
    model3 = DynamicWeightFusion(feature_size=feature_size, num_features=num_features)
    output3 = model3(features)
    print(f"  Output shape: {output3.shape}")
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"  Parameters: {params3:,}")

    print(f"\nAll fusion modules work correctly!")

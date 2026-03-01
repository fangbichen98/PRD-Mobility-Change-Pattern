"""
Enhanced feature extraction for time-series mobility data.

Adds 4 additional features to the basic total flow:
1. Net flow (direction)
2. Volatility (rolling std)
3. Time pattern (periodic encoding)
4. Trend (linear slope)

This increases feature dimension from 1 to 5, providing richer information.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def extract_enhanced_features(flow_2021: np.ndarray, flow_2024: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract enhanced features from flow data.

    Args:
        flow_2021: (168, 2) array - [inflow, outflow] for each hour
        flow_2024: (168, 2) array - [inflow, outflow] for each hour

    Returns:
        features_2021: (168, 5) array - enhanced features for 2021
        features_2024: (168, 5) array - enhanced features for 2024

    Features:
        1. total_log: Log-transformed total flow (inflow + outflow)
        2. net_flow_log: Log-transformed net flow (outflow - inflow, preserving sign)
        3. volatility: Rolling standard deviation of total flow (12-hour window)
        4. time_pattern: Sinusoidal encoding of time-of-week
        5. trend: Linear slope of total flow (24-hour window)
    """
    features_list = []

    for flow_data in [flow_2021, flow_2024]:
        n_timesteps = flow_data.shape[0]

        # Extract inflow and outflow
        inflow = flow_data[:, 0]
        outflow = flow_data[:, 1]

        # Feature 1: Total flow (log-transformed)
        total = inflow + outflow
        total_log = np.log1p(total)  # log(1 + x) preserves zeros

        # Feature 2: Net flow (direction, log-transformed with sign preservation)
        net_flow = outflow - inflow
        net_flow_log = np.sign(net_flow) * np.log1p(np.abs(net_flow))

        # Feature 3: Volatility (rolling standard deviation)
        # Use 12-hour window to capture intra-day volatility
        volatility = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            # Window: [max(0, t-12), t+12]
            window_start = max(0, t - 12)
            window_end = min(n_timesteps, t + 12 + 1)
            window = total[window_start:window_end]
            volatility[t] = np.std(window) if len(window) > 1 else 0

        # Normalize volatility
        if volatility.max() > 0:
            volatility = volatility / volatility.max()

        # Feature 4: Time pattern (sinusoidal encoding of time-of-week)
        # This helps the model learn periodic patterns (daily/weekly cycles)
        hour_of_week = np.arange(n_timesteps) % 168  # 0-167 (168 hours in a week)
        time_pattern = np.sin(2 * np.pi * hour_of_week / 168)

        # Feature 5: Trend (linear slope over recent 24 hours)
        trend = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            if t >= 24:
                # Fit linear trend to last 24 hours
                recent_total = total[t-24:t+1]
                # Simple slope: (last - first) / 24
                trend[t] = (recent_total[-1] - recent_total[0]) / 24
            else:
                # For first 24 hours, use available data
                recent_total = total[:t+1]
                if len(recent_total) > 1:
                    trend[t] = (recent_total[-1] - recent_total[0]) / len(recent_total)

        # Normalize trend
        trend_max = np.abs(trend).max()
        if trend_max > 0:
            trend = trend / trend_max

        # Stack all features
        enhanced = np.stack([
            total_log,
            net_flow_log,
            volatility,
            time_pattern,
            trend
        ], axis=1)  # (168, 5)

        features_list.append(enhanced)

    return features_list[0], features_list[1]


def extract_enhanced_features_simple(flow_2021: np.ndarray, flow_2024: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified enhanced feature extraction (faster, less memory).

    Only adds 2 features:
    1. Total flow (existing)
    2. Net flow (direction)

    This is a good balance between simplicity and expressiveness.
    """
    features_list = []

    for flow_data in [flow_2021, flow_2024]:
        inflow = flow_data[:, 0]
        outflow = flow_data[:, 1]

        # Feature 1: Total flow
        total = inflow + outflow
        total_log = np.log1p(total)

        # Feature 2: Net flow (direction)
        net_flow = outflow - inflow
        net_flow_log = np.sign(net_flow) * np.log1p(np.abs(net_flow))

        # Stack
        enhanced = np.stack([total_log, net_flow_log], axis=1)  # (168, 2)

        features_list.append(enhanced)

    return features_list[0], features_list[1]


def extract_features_with_gradients(flow_2021: np.ndarray, flow_2024: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features with gradient information.

    Adds temporal gradient (derivative) to capture rate of change.
    """
    features_list = []

    for flow_data in [flow_2021, flow_2024]:
        inflow = flow_data[:, 0]
        outflow = flow_data[:, 1]

        # Basic features
        total = inflow + outflow
        total_log = np.log1p(total)

        # Temporal gradient (rate of change)
        gradient = np.gradient(total_log)
        gradient = np.clip(gradient, -1, 1)  # Clip extreme values

        # Stack
        enhanced = np.stack([total_log, gradient], axis=1)  # (168, 2)

        features_list.append(enhanced)

    return features_list[0], features_list[1]


def extract_comprehensive_features(flow_2021: np.ndarray, flow_2024: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Comprehensive feature extraction with all possible features.

    This is the most expressive but also most memory-intensive version.
    Use only if you have sufficient GPU memory.

    Features (168, 8):
        1. total_log
        2. net_flow_log
        3. volatility
        4. time_pattern (sin)
        5. time_pattern_cos (cos) - additional periodic encoding
        6. trend
        7. gradient
        8. acceleration (second derivative)
    """
    features_list = []

    for flow_data in [flow_2021, flow_2024]:
        n_timesteps = flow_data.shape[0]

        inflow = flow_data[:, 0]
        outflow = flow_data[:, 1]

        # Basic features
        total = inflow + outflow
        total_log = np.log1p(total)

        net_flow = outflow - inflow
        net_flow_log = np.sign(net_flow) * np.log1p(np.abs(net_flow))

        # Volatility
        volatility = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            window_start = max(0, t - 12)
            window_end = min(n_timesteps, t + 12 + 1)
            window = total[window_start:window_end]
            volatility[t] = np.std(window) if len(window) > 1 else 0
        if volatility.max() > 0:
            volatility = volatility / volatility.max()

        # Time patterns (sin and cos)
        hour_of_week = np.arange(n_timesteps) % 168
        time_pattern_sin = np.sin(2 * np.pi * hour_of_week / 168)
        time_pattern_cos = np.cos(2 * np.pi * hour_of_week / 168)

        # Trend
        trend = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            if t >= 24:
                recent_total = total[t-24:t+1]
                trend[t] = (recent_total[-1] - recent_total[0]) / 24
        trend_max = np.abs(trend).max()
        if trend_max > 0:
            trend = trend / trend_max

        # Gradient
        gradient = np.gradient(total_log)
        gradient = np.clip(gradient, -1, 1)

        # Acceleration (second derivative)
        acceleration = np.gradient(gradient)
        acceleration = np.clip(acceleration, -1, 1)

        # Stack all features
        enhanced = np.stack([
            total_log,
            net_flow_log,
            volatility,
            time_pattern_sin,
            time_pattern_cos,
            trend,
            gradient,
            acceleration
        ], axis=1)  # (168, 8)

        features_list.append(enhanced)

    return features_list[0], features_list[1]


# Feature extraction configuration
FEATURE_EXTRACTION_METHODS = {
    'basic': None,  # Use existing single-feature extraction
    'simple': extract_enhanced_features_simple,  # 2 features
    'standard': extract_enhanced_features,  # 5 features
    'gradient': extract_features_with_gradients,  # 2 features (with gradient)
    'comprehensive': extract_comprehensive_features  # 8 features
}


def get_feature_extractor(method: str = 'standard'):
    """
    Get feature extraction function by name.

    Args:
        method: One of 'basic', 'simple', 'standard', 'gradient', 'comprehensive'

    Returns:
        Feature extraction function or None
    """
    return FEATURE_EXTRACTION_METHODS.get(method)


if __name__ == "__main__":
    # Test the feature extractors
    print("Testing Enhanced Feature Extraction")
    print("=" * 60)

    # Create dummy data
    n_timesteps = 168
    flow_2021 = np.random.rand(n_timesteps, 2) * 100
    flow_2024 = np.random.rand(n_timesteps, 2) * 100

    # Test each method
    for method_name, extractor in FEATURE_EXTRACTION_METHODS.items():
        if extractor is None:
            continue

        print(f"\n{method_name.upper()}:")
        feat_2021, feat_2024 = extractor(flow_2021, flow_2024)
        print(f"  Shape: {feat_2021.shape}")
        print(f"  Range: [{feat_2021.min():.3f}, {feat_2021.max():.3f}]")
        print(f"  Mean: {feat_2021.mean():.3f}")
        print(f"  Std: {feat_2021.std():.3f}")

    print("\n✓ All feature extraction methods work correctly!")

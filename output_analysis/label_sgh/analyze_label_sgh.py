#!/usr/bin/env python3
"""
Comprehensive Visualization and Analysis for label_sgh Model Results

This script analyzes the label_sgh.csv data and v4 model results to create:
1. Geographic distribution maps of labels
2. Confusion matrix visualization
3. Class distribution analysis
4. Model performance metrics
5. City-wise and area-wise analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import json
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Class names mapping
CLASS_NAMES = {
    1: "Stable-Balanced",
    2: "Growth-Balanced",
    3: "Decline-Balanced",
    4: "Stable-Aggregation",
    5: "Growth-Aggregation",
    6: "Decline-Aggregation",
    7: "Stable-Diffusion",
    8: "Growth-Diffusion",
    9: "Decline-Diffusion"
}

CLASS_COLORS = {
    1: '#4e79a7',  # Blue
    2: '#59a14f',  # Green
    3: '#e15759',  # Red
    4: '#76b7b2',  # Teal
    5: '#edc948',  # Yellow
    6: '#b07aa1',  # Purple
    7: '#ff9da7',  # Pink
    8: '#9c755f',  # Brown
    9: '#bab0ac'   # Gray
}

INTENSITY_LABELS = {0: 'Stable', 1: 'Growth', 2: 'Decline'}
DIRECTION_LABELS = {0: 'Balanced', 1: 'Aggregation', 2: 'Diffusion'}


def load_data():
    """Load all required data files."""
    base_path = Path('/root/workspace/Graph_Deep_Learning/20251001-PRD_18-21-24-mobility_change_pattern/analysis/PRD-Mobility-Change-Pattern')

    # Load labels
    labels_df = pd.read_csv(base_path / 'data/label_sgh.csv')
    print(f"✓ Loaded {len(labels_df)} labeled grids")

    # Load grid metadata
    metadata_df = pd.read_csv(base_path / 'data/grid_metadata/sgh_grid_metadata.csv')
    print(f"✓ Loaded {len(metadata_df)} grid metadata entries")

    # Merge labels with metadata
    merged_df = labels_df.merge(metadata_df[['grid_id', 'lon', 'lat', 'city_name', 'area_name']],
                                on='grid_id', how='left')
    print(f"✓ Merged data: {len(merged_df)} grids with geographic info")

    # Load model results
    results_path = base_path / 'outputs/dual_branch_v4_larger_capacity_20260311_133946/metrics'
    with open(results_path / 'test_results.json', 'r') as f:
        results = json.load(f)

    # Load confusion matrix
    cm = np.load(results_path / 'confusion_matrix.npy')

    return merged_df, results, cm


def plot_geographic_distribution(df, output_dir):
    """Create geographic distribution maps of labels."""
    print("\n📍 Creating geographic distribution maps...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Overall label distribution
    ax = axes[0, 0]
    for label in range(1, 10):
        data = df[df['label'] == label]
        ax.scatter(data['lon'], data['lat'], c=CLASS_COLORS[label],
                  label=CLASS_NAMES[label], s=10, alpha=0.6, edgecolors='none')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Geographic Distribution of All 9 Mobility Patterns', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 2. Flow Intensity pattern (rows of the 3x3 matrix)
    ax = axes[0, 1]
    intensity_colors = {'Stable': CLASS_COLORS[1], 'Growth': CLASS_COLORS[2], 'Decline': CLASS_COLORS[3]}
    for intensity, name in INTENSITY_LABELS.items():
        # Labels with this intensity: [1,2,3], [4,5,6], [7,8,9] for Stable, Growth, Decline
        label_range = [(intensity * 3 + 1), (intensity * 3 + 2), (intensity * 3 + 3)]
        data = df[df['label'].isin(label_range)]
        ax.scatter(data['lon'], data['lat'], c=intensity_colors[name],
                  label=name, s=15, alpha=0.6, edgecolors='none')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Flow Intensity Patterns', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 3. Spatial Direction pattern (columns of the 3x3 matrix)
    ax = axes[1, 0]
    direction_colors = {'Balanced': CLASS_COLORS[1], 'Aggregation': CLASS_COLORS[4], 'Diffusion': CLASS_COLORS[7]}
    for direction, name in DIRECTION_LABELS.items():
        # Labels with this direction: [1,4,7], [2,5,8], [3,6,9] for Balanced, Aggregation, Diffusion
        label_range = [direction + 1, direction + 4, direction + 7]
        data = df[df['label'].isin(label_range)]
        ax.scatter(data['lon'], data['lat'], c=direction_colors[name],
                  label=name, s=15, alpha=0.6, edgecolors='none')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Spatial Direction Patterns', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 4. City-wise distribution
    ax = axes[1, 1]
    city_counts = df.groupby('city_name')['label'].count().sort_values(ascending=False)
    cities = city_counts.index[:10]
    colors = plt.cm.Set3(np.linspace(0, 1, len(cities)))
    ax.barh(range(len(cities)), city_counts.values[:10], color=colors)
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels(cities)
    ax.set_xlabel('Number of Grids', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Cities by Grid Count', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'geographic_distribution.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: geographic_distribution.png")
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    """Create confusion matrix visualization."""
    print("\n📊 Creating confusion matrix visualization...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 1. Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=[f'C{i}' for i in range(1, 10)],
                yticklabels=[f'C{i}' for i in range(1, 10)])
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')

    # 2. Normalized by row (recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2,
                xticklabels=[f'C{i}' for i in range(1, 10)],
                yticklabels=[f'C{i}' for i in range(1, 10)])
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized - Recall)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: confusion_matrix.png")
    plt.close()


def plot_class_distribution(df, results, output_dir):
    """Create class distribution analysis."""
    print("\n📈 Creating class distribution analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. Overall class distribution
    ax = axes[0, 0]
    class_counts = df['label'].value_counts().sort_index()
    colors = [CLASS_COLORS[i] for i in class_counts.index]
    bars = ax.bar(range(1, 10), class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f'Class {i}\n{CLASS_NAMES[i]}' for i in range(1, 10)], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution (Overall)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Intensity distribution
    ax = axes[0, 1]
    intensity_counts = {
        'Stable': ((df['label'] - 1) // 3 == 0).sum(),
        'Growth': ((df['label'] - 1) // 3 == 1).sum(),
        'Decline': ((df['label'] - 1) // 3 == 2).sum()
    }
    colors_intensity = [CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]]
    wedges, texts, autotexts = ax.pie(intensity_counts.values(), labels=intensity_counts.keys(),
                                       autopct='%1.1f%%', colors=colors_intensity,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    ax.set_title('Flow Intensity Distribution', fontsize=14, fontweight='bold')

    # 3. Direction distribution
    ax = axes[1, 0]
    direction_counts = {
        'Balanced': ((df['label'] - 1) % 3 == 0).sum(),
        'Aggregation': ((df['label'] - 1) % 3 == 1).sum(),
        'Diffusion': ((df['label'] - 1) % 3 == 2).sum()
    }
    colors_direction = [CLASS_COLORS[1], CLASS_COLORS[4], CLASS_COLORS[7]]
    wedges, texts, autotexts = ax.pie(direction_counts.values(), labels=direction_counts.keys(),
                                       autopct='%1.1f%%', colors=colors_direction,
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    ax.set_title('Spatial Direction Distribution', fontsize=14, fontweight='bold')

    # 4. Class weights (from model training)
    ax = axes[1, 1]
    class_dist = results['data_info']['class_distribution']
    weights = [class_dist[f'class_{i}'] for i in range(1, 10)]
    # Calculate inverse weights as used in training
    total = sum(weights)
    class_weights = [total / (9 * w) if w > 0 else 0 for w in weights]

    bars = ax.bar(range(1, 10), class_weights, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f'C{i}' for i in range(1, 10)], fontsize=10)
    ax.set_ylabel('Class Weight', fontsize=12, fontweight='bold')
    ax.set_title('Class Weights (Inverse Frequency)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add weight labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'class_distribution.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: class_distribution.png")
    plt.close()


def plot_city_analysis(df, output_dir):
    """Create city-wise analysis."""
    print("\n🏙️  Creating city-wise analysis...")

    # Get top 15 cities by count
    top_cities = df.groupby('city_name').size().sort_values(ascending=False).head(15).index
    df_top = df[df['city_name'].isin(top_cities)]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Stacked bar: class distribution by city
    ax = axes[0, 0]
    city_class_counts = pd.crosstab(df_top['city_name'], df_top['label'])
    city_class_counts = city_class_counts.loc[top_cities]  # Sort by total count

    bottom = np.zeros(len(city_class_counts))
    for label in range(1, 10):
        counts = city_class_counts[label].values
        ax.barh(range(len(city_class_counts)), counts, left=bottom,
               label=CLASS_NAMES[label], color=CLASS_COLORS[label], edgecolor='white', linewidth=0.5)
        bottom += counts

    ax.set_yticks(range(len(city_class_counts)))
    ax.set_yticklabels(city_class_counts.index, fontsize=10)
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution by City (Top 15)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=7, ncol=2)
    ax.grid(axis='x', alpha=0.3)

    # 2. Geographic map of top cities
    ax = axes[0, 1]
    for i, city in enumerate(top_cities[:10]):
        data = df_top[df_top['city_name'] == city]
        ax.scatter(data['lon'], data['lat'], label=city, s=20, alpha=0.6, edgecolors='none')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Geographic Distribution of Top 10 Cities', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 3. Intensity pattern by city
    ax = axes[1, 0]
    city_intensity = pd.DataFrame()
    for city in top_cities:
        city_data = df_top[df_top['city_name'] == city]
        intensity_counts = {
            'Stable': ((city_data['label'] - 1) // 3 == 0).sum(),
            'Growth': ((city_data['label'] - 1) // 3 == 1).sum(),
            'Decline': ((city_data['label'] - 1) // 3 == 2).sum()
        }
        total = sum(intensity_counts.values())
        intensity_counts = {k: v/total*100 for k, v in intensity_counts.items()}
        city_intensity = pd.concat([city_intensity, pd.DataFrame([intensity_counts])], ignore_index=True)

    city_intensity.index = top_cities
    city_intensity.plot(kind='barh', stacked=True, color=[CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[3]],
                       ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Flow Intensity Pattern by City (%)', fontsize=14, fontweight='bold')
    ax.legend(title='Intensity', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 4. Direction pattern by city
    ax = axes[1, 1]
    city_direction = pd.DataFrame()
    for city in top_cities:
        city_data = df_top[df_top['city_name'] == city]
        direction_counts = {
            'Balanced': ((city_data['label'] - 1) % 3 == 0).sum(),
            'Aggregation': ((city_data['label'] - 1) % 3 == 1).sum(),
            'Diffusion': ((city_data['label'] - 1) % 3 == 2).sum()
        }
        total = sum(direction_counts.values())
        direction_counts = {k: v/total*100 for k, v in direction_counts.items()}
        city_direction = pd.concat([city_direction, pd.DataFrame([direction_counts])], ignore_index=True)

    city_direction.index = top_cities
    city_direction.plot(kind='barh', stacked=True, color=[CLASS_COLORS[1], CLASS_COLORS[4], CLASS_COLORS[7]],
                       ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Spatial Direction Pattern by City (%)', fontsize=14, fontweight='bold')
    ax.legend(title='Direction', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'city_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'city_analysis.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: city_analysis.png")
    plt.close()


def plot_model_performance(results, output_dir):
    """Create model performance visualization."""
    print("\n🎯 Creating model performance visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. Baseline comparison
    ax = axes[0, 0]
    models = ['LSTM\nBaseline', 'v1\n(128)', 'v2\n(lr=0.0002)', 'v4\n(192)']
    accuracies = [
        results['baseline_comparison']['lstm_baseline'],
        results['baseline_comparison']['v1_baseline'],
        results['baseline_comparison']['v2_improved'],
        results['baseline_comparison']['v4_larger_capacity']
    ]
    colors_bars = ['#7fb06f', '#7d9f9a', '#d9a4a7', '#f0a500']
    bars = ax.bar(models, accuracies, color=colors_bars, edgecolor='black', linewidth=2)

    # Add improvement annotations
    baseline = accuracies[0]
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if i > 0:
            improvement = acc - accuracies[i-1]
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f'+{improvement:.2f}%', ha='center', fontsize=10,
                   fontweight='bold', color='green' if improvement > 0 else 'red')

    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Version Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(68, 71)
    ax.grid(axis='y', alpha=0.3)

    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 0.5,
               f'{acc:.2f}%', ha='center', va='top', fontsize=10, fontweight='bold', color='white')

    # 2. Architecture parameters
    ax = axes[0, 1]
    arch = results['model_architecture']
    params = {
        'LSTM\nHidden': arch['temporal_branch']['hidden_size'],
        'GCN\nHidden': arch['spatial_branch']['hidden_size'],
        'Fusion\nHidden': arch['fusion']['hidden_size'],
        'Total\nParams': results.get('total_params', 4.66)  # Approximate
    }
    bars = ax.bar(range(len(params)), list(params.values()),
                  color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'],
                  edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(list(params.keys()), fontsize=11)
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Architecture Parameters (v4)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, params.values())):
        if i < 3:
            label = f'{val}'
        else:
            label = f'{val:.1f}M'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (max(params.values()) * 0.02),
               label, ha='center', fontsize=11, fontweight='bold')

    # 3. Data split
    ax = axes[1, 0]
    split = results['training_config']['train_val_test_split']
    sizes = [s * 100 for s in split]
    labels = [f'Train\n{sizes[0]:.0f}%', f'Val\n{sizes[1]:.0f}%', f'Test\n{sizes[2]:.0f}%']
    colors_split = ['#3498db', '#e74c3c', '#2ecc71']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_split,
                                       startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'},
                                       autopct=lambda p: f'{p:.1f}%%' if p > 5 else '')
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    ax.set_title('Data Split Configuration', fontsize=14, fontweight='bold')

    # 4. Training configuration summary
    ax = axes[1, 1]
    ax.axis('off')
    config_text = f"""
    Training Configuration Summary

    📊 Model: v4 (Larger Capacity)
    ├─ Test Accuracy: {results['test_accuracy']:.2f}%
    ├─ Test F1 Score: {results['test_f1']:.4f}
    ├─ Best Val Accuracy: {results['best_val_accuracy']:.2f}%

    ⚙️  Hyperparameters
    ├─ Batch Size: {results['training_config']['batch_size']}
    ├─ Learning Rate: {results['training_config']['learning_rate']}
    ├─ Weight Decay: {results['training_config']['weight_decay']}
    ├─ Max Epochs: {results['training_config']['num_epochs']}
    ├─ Early Stopping: {results['training_config']['early_stopping_patience']} epochs
    ├─ Random Seed: {results['training_config']['random_seed']}

    📈 Architecture
    ├─ Temporal: LSTM (hidden={arch['temporal_branch']['hidden_size']}, layers={arch['temporal_branch']['num_layers']})
    ├─ Spatial: GCN (hidden={arch['spatial_branch']['hidden_size']}, layers={arch['spatial_branch']['num_layers']})
    ├─ Fusion: Gated (hidden={arch['fusion']['hidden_size']})
    ├─ Classifier: MLP ({' → '.join(map(str, arch['classifier']['layers']))})

    📁 Dataset Info
    ├─ Label File: {results['data_info']['label_file']}
    ├─ Total Samples: {results['data_info']['total_samples']:,}
    ├─ Train/Val/Test: {results['data_info']['train_samples']}/{results['data_info']['val_samples']}/{results['data_info']['test_samples']}
    ├─ Graph 2021 Edges: {results['data_info']['graph_2021_edges']:,}
    ├─ Graph 2024 Edges: {results['data_info']['graph_2024_edges']:,}
    """
    ax.text(0.1, 0.5, config_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'model_performance.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: model_performance.png")
    plt.close()


def plot_per_class_metrics(results, output_dir):
    """Create per-class performance metrics."""
    print("\n📉 Creating per-class metrics visualization...")

    # Parse classification report
    report_file = Path('/root/workspace/Graph_Deep_Learning/20251001-PRD_18-21-24-mobility_change_pattern/analysis/PRD-Mobility-Change-Pattern/outputs/dual_branch_v4_larger_capacity_20260311_133946/metrics/classification_report.txt')

    # Extract metrics from report file
    import re
    with open(report_file, 'r') as f:
        content = f.read()

    # Parse precision, recall, f1 for each class
    class_metrics = {}
    for i in range(1, 10):
        # Find the line for each class
        pattern = rf'Class {i}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
        match = re.search(pattern, content)
        if match:
            class_metrics[i] = {
                'precision': float(match.group(1)),
                'recall': float(match.group(2)),
                'f1': float(match.group(3)),
                'support': int(match.group(4))
            }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. Precision, Recall, F1 by class
    ax = axes[0, 0]
    classes = list(class_metrics.keys())
    precision = [class_metrics[c]['precision'] for c in classes]
    recall = [class_metrics[c]['recall'] for c in classes]
    f1 = [class_metrics[c]['f1'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black', linewidth=1)
    ax.bar(x, recall, width, label='Recall', color='#e74c3c', edgecolor='black', linewidth=1)
    ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71', edgecolor='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in classes], fontsize=10)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # 2. Support by class
    ax = axes[0, 1]
    support = [class_metrics[c]['support'] for c in classes]
    colors_support = [CLASS_COLORS[c] for c in classes]
    bars = ax.bar(classes, support, color=colors_support, edgecolor='black', linewidth=1.5)
    ax.set_xticks(classes)
    ax.set_xticklabels([f'C{i}' for i in classes], fontsize=10)
    ax.set_ylabel('Support (Test Samples)', fontsize=12, fontweight='bold')
    ax.set_title('Test Set Support by Class', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Heatmap of metrics
    ax = axes[1, 0]
    metrics_matrix = np.array([[class_metrics[c]['precision'], class_metrics[c]['recall'], class_metrics[c]['f1']]
                              for c in classes])
    im = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['Precision', 'Recall', 'F1'])
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels([f'Class {i}' for i in classes])
    ax.set_title('Metrics Heatmap', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(classes)):
        for j in range(3):
            text = ax.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Score')

    # 4. Performance ranking by F1
    ax = axes[1, 1]
    sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    sorted_f1 = [m['f1'] for _, m in sorted_classes]
    sorted_labels = [f"C{c}\n{CLASS_NAMES[c][:15]}" for c, _ in sorted_classes]
    sorted_colors = [CLASS_COLORS[c] for c, _ in sorted_classes]

    bars = ax.barh(range(len(sorted_classes)), sorted_f1, color=sorted_colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels(sorted_labels, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Classes Ranked by F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0.65, color='red', linestyle='--', alpha=0.7, label='Macro Avg')
    ax.legend(fontsize=10)

    # Add F1 labels
    for bar, f1 in zip(bars, sorted_f1):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
               f'{f1:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'per_class_metrics.pdf', bbox_inches='tight')
    print(f"  ✓ Saved: per_class_metrics.png")
    plt.close()


def create_summary_report(df, results, output_dir):
    """Create a summary text report."""
    print("\n📝 Creating summary report...")

    report_path = output_dir / 'analysis_summary.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LABEL_SGH MODEL ANALYSIS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("1. DATA OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Labeled Grids: {len(df):,}\n")
        f.write(f"Date: 2026-03-11\n")
        f.write(f"Model Version: v4 (Larger Capacity)\n\n")

        f.write("Class Distribution:\n")
        class_counts = df['label'].value_counts().sort_index()
        for label in range(1, 10):
            count = class_counts.get(label, 0)
            pct = count / len(df) * 100
            f.write(f"  Class {label} ({CLASS_NAMES[label]}): {count:4d} ({pct:5.2f}%)\n")

        f.write("\nIntensity Distribution:\n")
        for intensity, name in INTENSITY_LABELS.items():
            count = ((df['label'] - 1) // 3 == intensity).sum()
            pct = count / len(df) * 100
            f.write(f"  {name:12s}: {count:4d} ({pct:5.2f}%)\n")

        f.write("\nDirection Distribution:\n")
        for direction, name in DIRECTION_LABELS.items():
            count = ((df['label'] - 1) % 3 == direction).sum()
            pct = count / len(df) * 100
            f.write(f"  {name:12s}: {count:4d} ({pct:5.2f}%)\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("2. GEOGRAPHIC DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        f.write(f"Longitude Range: {df['lon'].min():.4f} to {df['lon'].max():.4f}\n")
        f.write(f"Latitude Range:  {df['lat'].min():.4f} to {df['lat'].max():.4f}\n\n")

        f.write("Top 10 Cities by Grid Count:\n")
        city_counts = df.groupby('city_name').size().sort_values(ascending=False).head(10)
        for i, (city, count) in enumerate(city_counts.items(), 1):
            pct = count / len(df) * 100
            f.write(f"  {i:2d}. {city:20s}: {count:4d} ({pct:5.2f}%)\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("3. MODEL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"Test Accuracy:     {results['test_accuracy']:.2f}%\n")
        f.write(f"Test F1 (macro):   {results['test_f1']:.4f}\n")
        f.write(f"Best Val Accuracy: {results['best_val_accuracy']:.2f}%\n\n")

        f.write("Baseline Comparison:\n")
        f.write(f"  LSTM Baseline:       {results['baseline_comparison']['lstm_baseline']:.2f}%\n")
        f.write(f"  v1 (hidden=128):     {results['baseline_comparison']['v1_baseline']:.2f}%\n")
        f.write(f"  v2 (lr=0.0002):      {results['baseline_comparison']['v2_improved']:.2f}%\n")
        f.write(f"  v4 (hidden=192):     {results['baseline_comparison']['v4_larger_capacity']:.2f}%\n")
        f.write(f"  Improvement over v1: +{results['baseline_comparison']['improvement_over_v1']:.2f}%\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("4. MODEL ARCHITECTURE\n")
        f.write("-"*80 + "\n")
        arch = results['model_architecture']
        f.write(f"Temporal Branch: {arch['temporal_branch']['type']}\n")
        f.write(f"  Hidden Size: {arch['temporal_branch']['hidden_size']}\n")
        f.write(f"  Num Layers:  {arch['temporal_branch']['num_layers']}\n")
        f.write(f"  Dropout:     {arch['temporal_branch']['dropout']}\n\n")

        f.write(f"Spatial Branch: {arch['spatial_branch']['type']}\n")
        f.write(f"  Hidden Size: {arch['spatial_branch']['hidden_size']}\n")
        f.write(f"  Num Layers:  {arch['spatial_branch']['num_layers']}\n")
        f.write(f"  Dropout:     {arch['spatial_branch']['dropout']}\n\n")

        f.write(f"Fusion: {arch['fusion']['type']}\n")
        f.write(f"  Num Features: {arch['fusion']['num_features']}\n")
        f.write(f"  Hidden Size:  {arch['fusion']['hidden_size']}\n")
        f.write(f"  Dropout:      {arch['fusion']['dropout']}\n\n")

        f.write(f"Classifier: {arch['classifier']['type']}\n")
        f.write(f"  Layers: {' → '.join(map(str, arch['classifier']['layers']))}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("5. TRAINING CONFIGURATION\n")
        f.write("-"*80 + "\n")
        config = results['training_config']
        f.write(f"Batch Size:          {config['batch_size']}\n")
        f.write(f"Learning Rate:       {config['learning_rate']}\n")
        f.write(f"Weight Decay:        {config['weight_decay']}\n")
        f.write(f"Max Epochs:          {config['num_epochs']}\n")
        f.write(f"Early Stop Patience: {config['early_stopping_patience']}\n")
        f.write(f"Random Seed:         {config['random_seed']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"  ✓ Saved: analysis_summary.txt")


def main():
    """Main function to run all analyses."""
    print("="*80)
    print("LABEL_SGH MODEL ANALYSIS AND VISUALIZATION")
    print("="*80)

    # Setup output directory
    output_dir = Path('/root/workspace/Graph_Deep_Learning/20251001-PRD_18-21-24-mobility_change_pattern/analysis/PRD-Mobility-Change-Pattern/output_analysis/label_sgh')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Load data
    print("\n📊 Loading data...")
    df, results, cm = load_data()

    # Create all visualizations
    plot_geographic_distribution(df, output_dir)
    plot_confusion_matrix(cm, output_dir)
    plot_class_distribution(df, results, output_dir)
    plot_city_analysis(df, output_dir)
    plot_model_performance(results, output_dir)
    plot_per_class_metrics(results, output_dir)
    create_summary_report(df, results, output_dir)

    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated files in {output_dir}:")
    print("  📊 geographic_distribution.png/pdf")
    print("  📊 confusion_matrix.png/pdf")
    print("  📊 class_distribution.png/pdf")
    print("  📊 city_analysis.png/pdf")
    print("  📊 model_performance.png/pdf")
    print("  📊 per_class_metrics.png/pdf")
    print("  📝 analysis_summary.txt")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

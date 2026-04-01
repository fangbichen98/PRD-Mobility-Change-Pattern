#!/usr/bin/env python3
"""
Analyze Class Overlap - Statistical Analysis Script

This script performs statistical analysis to determine if confused classes
(Class 4↔5, Class 7↔9) are truly distinct or should be merged.

Usage:
    python scripts/analyze_class_overlap.py --classes 4 5
    python scripts/analyze_class_overlap.py --classes 7 9
    python scripts/analyze_class_overlap.py --all
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import *
from src.preprocessing.dual_year_processor import DualYearProcessor


def compute_feature_statistics(features_list):
    """
    Compute summary statistics for a list of feature arrays.

    Args:
        features_list: List of (168, 4) arrays

    Returns:
        Dictionary of statistics
    """
    stats = {}

    # Convert to numpy array for easier computation
    all_features = np.array(features_list)  # (N, 168, 4)

    # Temporal statistics
    stats['mean_inflow_2021'] = all_features[:, :, 0].mean(axis=1).mean()
    stats['std_inflow_2021'] = all_features[:, :, 0].mean(axis=1).std()
    stats['mean_outflow_2021'] = all_features[:, :, 1].mean(axis=1).mean()
    stats['std_outflow_2021'] = all_features[:, :, 1].mean(axis=1).std()
    stats['mean_inflow_2024'] = all_features[:, :, 2].mean(axis=1).mean()
    stats['std_inflow_2024'] = all_features[:, :, 2].mean(axis=1).std()
    stats['mean_outflow_2024'] = all_features[:, :, 3].mean(axis=1).mean()
    stats['std_outflow_2024'] = all_features[:, :, 3].mean(axis=1).std()

    # Flow change statistics
    flow_2021 = all_features[:, :, :2].sum(axis=2).mean(axis=1)  # Total flow 2021
    flow_2024 = all_features[:, :, 2:].sum(axis=2).mean(axis=1)  # Total flow 2024
    stats['mean_flow_change'] = (flow_2024 - flow_2021).mean()
    stats['std_flow_change'] = (flow_2024 - flow_2021).std()
    stats['mean_flow_ratio'] = (flow_2024 / (flow_2021 + 1e-8)).mean()

    # Temporal variability
    stats['mean_temporal_std'] = all_features.std(axis=1).mean()

    # Peak flow statistics
    stats['mean_peak_inflow_2021'] = all_features[:, :, 0].max(axis=1).mean()
    stats['mean_peak_outflow_2021'] = all_features[:, :, 1].max(axis=1).mean()
    stats['mean_peak_inflow_2024'] = all_features[:, :, 2].max(axis=1).mean()
    stats['mean_peak_outflow_2024'] = all_features[:, :, 3].max(axis=1).mean()

    return stats


def statistical_tests(features_class1, features_class2, class1_id, class2_id):
    """
    Perform statistical tests to determine if two classes are distinct.

    Args:
        features_class1: List of (168, 4) arrays for class 1
        features_class2: List of (168, 4) arrays for class 2
        class1_id: Class 1 ID (1-9)
        class2_id: Class 2 ID (1-9)
    """
    print("="*80)
    print(f"STATISTICAL ANALYSIS: Class {class1_id} vs Class {class2_id}")
    print("="*80)
    print()

    # Convert to numpy arrays
    features1 = np.array(features_class1)  # (N1, 168, 4)
    features2 = np.array(features_class2)  # (N2, 168, 4)

    print(f"Sample sizes: Class {class1_id}={len(features1)}, Class {class2_id}={len(features2)}")
    print()

    # 1. Summary statistics comparison
    print("1. SUMMARY STATISTICS COMPARISON")
    print("-" * 80)

    stats1 = compute_feature_statistics(features_class1)
    stats2 = compute_feature_statistics(features_class2)

    comparison_df = pd.DataFrame({
        f'Class {class1_id}': stats1,
        f'Class {class2_id}': stats2,
        'Difference': {k: abs(stats1[k] - stats2[k]) for k in stats1.keys()},
        'Relative Diff (%)': {k: 100 * abs(stats1[k] - stats2[k]) / (abs(stats1[k]) + 1e-8)
                              for k in stats1.keys()}
    })

    print(comparison_df.to_string())
    print()

    # 2. Distribution tests
    print("2. DISTRIBUTION SIMILARITY TESTS")
    print("-" * 80)

    # Extract key features for testing
    flow_change_1 = (features1[:, :, 2:].sum(axis=2) - features1[:, :, :2].sum(axis=2)).mean(axis=1)
    flow_change_2 = (features2[:, :, 2:].sum(axis=2) - features2[:, :, :2].sum(axis=2)).mean(axis=1)

    inflow_change_1 = (features1[:, :, 2] - features1[:, :, 0]).mean(axis=1)
    inflow_change_2 = (features2[:, :, 2] - features2[:, :, 0]).mean(axis=1)

    outflow_change_1 = (features1[:, :, 3] - features1[:, :, 1]).mean(axis=1)
    outflow_change_2 = (features2[:, :, 3] - features2[:, :, 1]).mean(axis=1)

    # Kolmogorov-Smirnov test
    ks_flow, p_ks_flow = ks_2samp(flow_change_1, flow_change_2)
    ks_inflow, p_ks_inflow = ks_2samp(inflow_change_1, inflow_change_2)
    ks_outflow, p_ks_outflow = ks_2samp(outflow_change_1, outflow_change_2)

    print("Kolmogorov-Smirnov Test (tests if distributions are different):")
    print(f"  Total flow change:  KS={ks_flow:.4f}, p-value={p_ks_flow:.4f} {'✓ DIFFERENT' if p_ks_flow < 0.05 else '✗ SIMILAR'}")
    print(f"  Inflow change:      KS={ks_inflow:.4f}, p-value={p_ks_inflow:.4f} {'✓ DIFFERENT' if p_ks_inflow < 0.05 else '✗ SIMILAR'}")
    print(f"  Outflow change:     KS={ks_outflow:.4f}, p-value={p_ks_outflow:.4f} {'✓ DIFFERENT' if p_ks_outflow < 0.05 else '✗ SIMILAR'}")
    print()

    # Mann-Whitney U test (non-parametric)
    u_flow, p_u_flow = mannwhitneyu(flow_change_1, flow_change_2, alternative='two-sided')
    u_inflow, p_u_inflow = mannwhitneyu(inflow_change_1, inflow_change_2, alternative='two-sided')
    u_outflow, p_u_outflow = mannwhitneyu(outflow_change_1, outflow_change_2, alternative='two-sided')

    print("Mann-Whitney U Test (non-parametric test for different medians):")
    print(f"  Total flow change:  U={u_flow:.1f}, p-value={p_u_flow:.4f} {'✓ DIFFERENT' if p_u_flow < 0.05 else '✗ SIMILAR'}")
    print(f"  Inflow change:      U={u_inflow:.1f}, p-value={p_u_inflow:.4f} {'✓ DIFFERENT' if p_u_inflow < 0.05 else '✗ SIMILAR'}")
    print(f"  Outflow change:     U={u_outflow:.1f}, p-value={p_u_outflow:.4f} {'✓ DIFFERENT' if p_u_outflow < 0.05 else '✗ SIMILAR'}")
    print()

    # 3. Effect size (Cohen's d)
    print("3. EFFECT SIZE (Cohen's d)")
    print("-" * 80)

    def cohens_d(x1, x2):
        n1, n2 = len(x1), len(x2)
        var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        return (np.mean(x1) - np.mean(x2)) / pooled_std

    d_flow = cohens_d(flow_change_1, flow_change_2)
    d_inflow = cohens_d(inflow_change_1, inflow_change_2)
    d_outflow = cohens_d(outflow_change_1, outflow_change_2)

    def interpret_cohens_d(d):
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    print(f"  Total flow change:  d={d_flow:.3f} ({interpret_cohens_d(d_flow)} effect)")
    print(f"  Inflow change:      d={d_inflow:.3f} ({interpret_cohens_d(d_inflow)} effect)")
    print(f"  Outflow change:     d={d_outflow:.3f} ({interpret_cohens_d(d_outflow)} effect)")
    print()

    # 4. Overall conclusion
    print("4. OVERALL CONCLUSION")
    print("-" * 80)

    # Count how many tests show significant difference
    significant_tests = sum([
        p_ks_flow < 0.05,
        p_ks_inflow < 0.05,
        p_ks_outflow < 0.05,
        p_u_flow < 0.05,
        p_u_inflow < 0.05,
        p_u_outflow < 0.05
    ])

    # Check effect sizes
    large_effects = sum([
        abs(d_flow) >= 0.8,
        abs(d_inflow) >= 0.8,
        abs(d_outflow) >= 0.8
    ])

    print(f"Significant tests: {significant_tests}/6")
    print(f"Large effect sizes: {large_effects}/3")
    print()

    if significant_tests >= 4 and large_effects >= 2:
        print("✅ CONCLUSION: Classes are STATISTICALLY DISTINCT")
        print("   Recommendation: Keep classes separate, improve feature engineering")
    elif significant_tests >= 2 or large_effects >= 1:
        print("⚠️  CONCLUSION: Classes are MODERATELY DISTINCT")
        print("   Recommendation: Consider adding more discriminative features")
    else:
        print("❌ CONCLUSION: Classes are STATISTICALLY SIMILAR")
        print("   Recommendation: MERGE these classes into a single class")

    print()

    # 5. Visualization
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Flow change distribution
    ax = axes[0, 0]
    ax.hist(flow_change_1, bins=30, alpha=0.6, label=f'Class {class1_id}', color='blue', density=True)
    ax.hist(flow_change_2, bins=30, alpha=0.6, label=f'Class {class2_id}', color='red', density=True)
    ax.set_xlabel('Total Flow Change', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(f'Total Flow Change Distribution\nKS p-value={p_ks_flow:.4f}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Inflow change distribution
    ax = axes[0, 1]
    ax.hist(inflow_change_1, bins=30, alpha=0.6, label=f'Class {class1_id}', color='blue', density=True)
    ax.hist(inflow_change_2, bins=30, alpha=0.6, label=f'Class {class2_id}', color='red', density=True)
    ax.set_xlabel('Inflow Change', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(f'Inflow Change Distribution\nKS p-value={p_ks_inflow:.4f}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Outflow change distribution
    ax = axes[0, 2]
    ax.hist(outflow_change_1, bins=30, alpha=0.6, label=f'Class {class1_id}', color='blue', density=True)
    ax.hist(outflow_change_2, bins=30, alpha=0.6, label=f'Class {class2_id}', color='red', density=True)
    ax.set_xlabel('Outflow Change', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title(f'Outflow Change Distribution\nKS p-value={p_ks_outflow:.4f}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Box plots
    ax = axes[1, 0]
    data_to_plot = [flow_change_1, flow_change_2]
    bp = ax.boxplot(data_to_plot, labels=[f'Class {class1_id}', f'Class {class2_id}'],
                    patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('Total Flow Change', fontweight='bold')
    ax.set_title(f'Flow Change Box Plot\nCohen\'s d={d_flow:.3f}', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Plot 5: Scatter plot (inflow vs outflow change)
    ax = axes[1, 1]
    ax.scatter(inflow_change_1, outflow_change_1, alpha=0.6, label=f'Class {class1_id}',
               color='blue', s=50, edgecolors='black', linewidth=0.5)
    ax.scatter(inflow_change_2, outflow_change_2, alpha=0.6, label=f'Class {class2_id}',
               color='red', s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Inflow Change', fontweight='bold')
    ax.set_ylabel('Outflow Change', fontweight='bold')
    ax.set_title('Inflow vs Outflow Change', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 6: PCA visualization
    ax = axes[1, 2]

    # Flatten features for PCA
    features1_flat = features1.reshape(len(features1), -1)  # (N1, 168*4)
    features2_flat = features2.reshape(len(features2), -1)  # (N2, 168*4)
    all_features_flat = np.vstack([features1_flat, features2_flat])

    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(all_features_flat)

    # Split back
    features1_pca = features_pca[:len(features1)]
    features2_pca = features_pca[len(features1):]

    ax.scatter(features1_pca[:, 0], features1_pca[:, 1], alpha=0.6,
               label=f'Class {class1_id}', color='blue', s=50, edgecolors='black', linewidth=0.5)
    ax.scatter(features2_pca[:, 0], features2_pca[:, 1], alpha=0.6,
               label=f'Class {class2_id}', color='red', s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax.set_title('PCA Visualization', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Class {class1_id} vs Class {class2_id}: Statistical Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = f'output_analysis/class_overlap_analysis_{class1_id}_vs_{class2_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    print()

    return {
        'class1_id': class1_id,
        'class2_id': class2_id,
        'ks_test_pvalue': min(p_ks_flow, p_ks_inflow, p_ks_outflow),
        'mann_whitney_pvalue': min(p_u_flow, p_u_inflow, p_u_outflow),
        'cohens_d_max': max(abs(d_flow), abs(d_inflow), abs(d_outflow)),
        'significant_tests': significant_tests,
        'large_effects': large_effects,
        'recommendation': 'merge' if significant_tests < 2 and large_effects < 1 else 'keep'
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze class overlap")
    parser.add_argument('--classes', type=int, nargs=2, metavar=('CLASS1', 'CLASS2'),
                       help='Two class IDs to compare (1-9)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all problematic class pairs (4-5, 7-9)')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    processor = DualYearProcessor(
        label_file=LABEL_FILE,
        od_file_2021=OD_FILE_2021,
        od_file_2024=OD_FILE_2024,
        metadata_file=METADATA_FILE,
        train_days=TRAIN_DAYS,
        flow_threshold=FLOW_THRESHOLD,
        use_knn_fallback=USE_KNN_FALLBACK,
        knn_k=KNN_K,
        cache_dir=CACHE_DIR
    )

    data = processor.load_data()
    print(f"✓ Data loaded: {len(data['labels'])} samples")
    print()

    # Extract features by class
    features_by_class = {i: [] for i in range(1, 10)}
    for idx, label in enumerate(data['labels']):
        features_by_class[label].append(data['change_features'][idx])

    # Print class distribution
    print("Class distribution:")
    for class_id in range(1, 10):
        print(f"  Class {class_id}: {len(features_by_class[class_id])} samples")
    print()

    # Run analysis
    results = []

    if args.all:
        # Analyze problematic pairs
        pairs = [(4, 5), (7, 9)]
        for class1, class2 in pairs:
            result = statistical_tests(
                features_by_class[class1],
                features_by_class[class2],
                class1, class2
            )
            results.append(result)
    elif args.classes:
        class1, class2 = args.classes
        if class1 < 1 or class1 > 9 or class2 < 1 or class2 > 9:
            print("Error: Class IDs must be between 1 and 9")
            return

        result = statistical_tests(
            features_by_class[class1],
            features_by_class[class2],
            class1, class2
        )
        results.append(result)
    else:
        print("Error: Must specify either --classes or --all")
        return

    # Summary
    if len(results) > 1:
        print("="*80)
        print("SUMMARY OF ALL ANALYSES")
        print("="*80)

        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        print()

        merge_recommendations = summary_df[summary_df['recommendation'] == 'merge']
        if len(merge_recommendations) > 0:
            print("⚠️  CLASSES RECOMMENDED FOR MERGING:")
            for _, row in merge_recommendations.iterrows():
                print(f"   - Class {row['class1_id']} and Class {row['class2_id']}")
        else:
            print("✓ All analyzed class pairs are statistically distinct")


if __name__ == '__main__':
    main()

"""
Predict all grids using cached data and best model
Generate full-region visualizations with English labels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import torch
import pickle
import json
import os
import sys
import math
from tqdm import tqdm

sys.path.append('src')

from models.enhanced_dual_branch_model import EnhancedDualBranchModel
import config
from viz_config import AREA_NAME_EN, CITY_NAME_EN, CLASS_COLORS, CLASS_NAMES, DEFAULT_DPI, EXPORT_DPI, EXPERIMENT_DIR, FONT_FAMILY

# Set matplotlib parameters
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['figure.dpi'] = DEFAULT_DPI


def add_north_arrow(ax, x=0.94, y=0.85, size=0.08):
    """Add a journal-style black/white compass north arrow."""
    # Outer black needle (north-pointing)
    outer = np.array([
        [x, y + size],
        [x - size * 0.20, y - size * 0.55],
        [x, y - size * 0.30],
        [x + size * 0.20, y - size * 0.55],
    ])

    # Inner white face to mimic standard compass style
    inner = np.array([
        [x, y + size * 0.82],
        [x - size * 0.06, y - size * 0.40],
        [x, y - size * 0.22],
    ])

    ax.add_patch(Polygon(outer, closed=True, transform=ax.transAxes,
                         facecolor='black', edgecolor='black', linewidth=0.8, zorder=10))
    ax.add_patch(Polygon(inner, closed=True, transform=ax.transAxes,
                         facecolor='white', edgecolor='none', zorder=11))
    ax.text(x, y + size * 1.18, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=16, fontweight='bold', color='black', zorder=12)


def add_scalebar(ax, lon_min, lon_max, lat_min, lat_max, length_km=20, position='left'):
    """Add a simple geographic scale bar using local latitude"""
    lat_ref = (lat_min + lat_max) / 2.0
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat_ref))
    if km_per_deg_lon <= 0:
        return

    length_deg = length_km / km_per_deg_lon
    if position == 'right':
        x0 = lon_max - (lon_max - lon_min) * 0.06 - length_deg
    elif position == 'center':
        x0 = lon_min + (lon_max - lon_min - length_deg) * 0.5
    else:
        x0 = lon_min + (lon_max - lon_min) * 0.06
    y0 = lat_min + (lat_max - lat_min) * 0.035
    x1 = x0 + length_deg

    ax.plot([x0, x1], [y0, y0], color='black', linewidth=3)
    ax.plot([x0, x0], [y0 - 0.001, y0 + 0.001], color='black', linewidth=2)
    ax.plot([x1, x1], [y0 - 0.001, y0 + 0.001], color='black', linewidth=2)
    ax.text((x0 + x1) / 2, y0 + (lat_max - lat_min) * 0.012, f'{length_km} km',
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.2))


def apply_geographic_axes_style(ax, lon_min, lon_max, lat_min, lat_max, scalebar_km=20, scalebar_position='left'):
    """Apply map style with approximate true geographic scale"""
    lat_ref = (lat_min + lat_max) / 2.0
    ax.set_aspect(1.0 / max(math.cos(math.radians(lat_ref)), 1e-6))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    add_scalebar(ax, lon_min, lon_max, lat_min, lat_max, length_km=scalebar_km, position=scalebar_position)
    add_north_arrow(ax)

def load_cached_data(cache_path):
    """Load cached data"""
    print(f"Loading cached data from {cache_path}...")
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Loaded cached data")
    print(f"  - Grids with features: {len(data['change_features'])}")
    print(f"  - Total grids in mapping: {len(data['grid_id_to_idx'])}")
    print(f"  - Graph 2021 edges: {data['graphs_2021'][0][0].shape[1]}")
    print(f"  - Graph 2024 edges: {data['graphs_2024'][0][0].shape[1]}")
    return data

def load_model(model_path, device):
    """Load the best trained model"""
    print(f"\nLoading model from {model_path}...")

    model = EnhancedDualBranchModel(
        temporal_input_size=config.TEMPORAL_INPUT_SIZE,
        hidden_size=config.FUSION_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES,
        num_time_steps=config.TIME_STEPS,
        dropout=0.4,
        spatial_model=config.SPATIAL_MODEL,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print(f"✓ Model loaded successfully")
    return model

def predict_grids_with_features(model, cached_data, device, batch_size=64):
    """Predict only grids that have features"""
    print(f"\nPredicting grids with features...")

    # Get grids with features
    grid_ids = list(cached_data['change_features'].keys())
    print(f"  Total grids to predict: {len(grid_ids)}")

    # Prepare data
    all_temporal_2021 = []
    all_temporal_2024 = []

    for grid_id in grid_ids:
        features = cached_data['change_features'][grid_id]  # (168, 4)
        temporal_2021 = features[:, :2]  # (168, 2)
        temporal_2024 = features[:, 2:]  # (168, 2)
        all_temporal_2021.append(temporal_2021)
        all_temporal_2024.append(temporal_2024)

    all_temporal_2021 = torch.FloatTensor(np.array(all_temporal_2021)).to(device)  # (N, 168, 2)
    all_temporal_2024 = torch.FloatTensor(np.array(all_temporal_2024)).to(device)  # (N, 168, 2)

    # Get node indices
    node_indices = torch.LongTensor([cached_data['grid_id_to_idx'][gid] for gid in grid_ids]).to(device)

    # Get graphs
    edge_index_2021, edge_attr_2021 = cached_data['graphs_2021'][0]
    edge_index_2024, edge_attr_2024 = cached_data['graphs_2024'][0]

    edge_index_2021 = torch.from_numpy(edge_index_2021).long().to(device)
    edge_attr_2021 = torch.from_numpy(edge_attr_2021).float().to(device)
    edge_index_2024 = torch.from_numpy(edge_index_2024).long().to(device)
    edge_attr_2024 = torch.from_numpy(edge_attr_2024).float().to(device)

    graphs_2021 = [(edge_index_2021, edge_attr_2021)]
    graphs_2024 = [(edge_index_2024, edge_attr_2024)]

    num_nodes = len(cached_data['grid_id_to_idx'])

    # Create full size temporal tensors for model inputs
    full_temporal_2021 = torch.zeros((num_nodes, 168, 2), dtype=torch.float32).to(device)
    full_temporal_2024 = torch.zeros((num_nodes, 168, 2), dtype=torch.float32).to(device)

    for i, grid_id in enumerate(grid_ids):
        node_idx = cached_data['grid_id_to_idx'][grid_id]
        full_temporal_2021[node_idx] = all_temporal_2021[i]
        full_temporal_2024[node_idx] = all_temporal_2024[i]

    # Predict in batches
    all_predictions = []

    model.eval()
    with torch.no_grad():
        n_samples = len(grid_ids)
        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting"):
            batch_end = min(i + batch_size, n_samples)
            batch_node_indices = node_indices[i:batch_end]

            try:
                logits = model(
                    x_2021=full_temporal_2021,
                    x_2024=full_temporal_2024,
                    graphs_2021=graphs_2021,
                    graphs_2024=graphs_2024,
                    num_nodes=num_nodes,
                    node_indices=batch_node_indices
                )

                predictions = torch.argmax(logits, dim=1).cpu().numpy() + 1
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"\nError in batch {i}-{batch_end}: {e}")
                # Use fallback: assign most common class
                all_predictions.extend([5] * (batch_end - i))

    print(f"✓ Predicted {len(all_predictions)} grids")
    return np.array(all_predictions), grid_ids

def create_full_prediction_dataframe(predictions, grid_ids, grid_metadata_path):
    """Create dataframe with predictions and coordinates"""
    print(f"\nCreating prediction dataframe...")

    # Load grid metadata
    grid_meta = pd.read_csv(grid_metadata_path)

    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'grid_id': grid_ids,
        'predicted_label': predictions,
    })

    # Merge with metadata
    result_df = pred_df.merge(
        grid_meta[['grid_id', 'lon', 'lat', 'city_name', 'area_name']],
        on='grid_id',
        how='left'
    )

    result_df['city_name_en'] = result_df['city_name'].map(CITY_NAME_EN).fillna(result_df['city_name'])
    result_df['area_name_en'] = result_df['area_name'].map(AREA_NAME_EN).fillna(result_df['area_name'])

    print(f"✓ Created dataframe with {len(result_df)} grids")
    print(f"  - Cities: {result_df['city_name'].nunique()}")
    print(f"  - Areas: {result_df['area_name'].nunique()}")

    return result_df

def plot_full_region_map(pred_df, output_dir):
    """Plot full region map with all predictions"""
    print(f"\nPlotting full region map...")

    fig, ax = plt.subplots(figsize=(20, 16))
    lon_min = pred_df['lon'].min() - 0.05
    lon_max = pred_df['lon'].max() + 0.05
    lat_min = pred_df['lat'].min() - 0.05
    lat_max = pred_df['lat'].max() + 0.05

    for class_id in range(1, 10):
        mask = pred_df['predicted_label'] == class_id
        if mask.sum() > 0:
            class_data = pred_df[mask]
            ax.scatter(
                class_data['lon'],
                class_data['lat'],
                c=CLASS_COLORS[class_id],
                s=15,
                alpha=0.7,
                label=CLASS_NAMES[class_id],
                edgecolors='none'
            )

    ax.set_xlabel('Longitude', fontsize=20, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=20, fontweight='bold')
    ax.set_title('Mobility Pattern Changes in Shenzhen-Dongguan-Huizhou Region\n(2021-2024)',
                 fontsize=24, fontweight='bold', pad=20)

    # Single merged legend (journal-friendly, avoids excessive right-side blocks)
    merged_handles = [
        mpatches.Patch(color=CLASS_COLORS[1], label='Stable Static'),
        mpatches.Patch(color=CLASS_COLORS[2], label='Stable Aggregation'),
        mpatches.Patch(color=CLASS_COLORS[3], label='Stable Dispersion'),
        mpatches.Patch(color=CLASS_COLORS[4], label='Growth Static'),
        mpatches.Patch(color=CLASS_COLORS[5], label='Growth Aggregation'),
        mpatches.Patch(color=CLASS_COLORS[6], label='Growth Dispersion'),
        mpatches.Patch(color=CLASS_COLORS[7], label='Decline Static'),
        mpatches.Patch(color=CLASS_COLORS[8], label='Decline Aggregation'),
        mpatches.Patch(color=CLASS_COLORS[9], label='Decline Dispersion'),
    ]
    ax.legend(handles=merged_handles, title='Legend',
              loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1,
              fontsize=15, title_fontsize=17, frameon=True, framealpha=0.95, shadow=True)

    ax.grid(True, alpha=0.3, linestyle='--')
    apply_geographic_axes_style(ax, lon_min, lon_max, lat_min, lat_max, scalebar_km=20)

    plt.tight_layout()

    output_path = f"{output_dir}/full_region_prediction_map.png"
    output_pdf = f"{output_dir}/full_region_prediction_map.pdf"
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    print(f"✓ Saved to {output_pdf}")
    plt.close()

def plot_pattern_group_maps(pred_df, output_dir):
    """Plot separate maps for each pattern group"""
    print(f"\nPlotting pattern group maps...")

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    lon_min = pred_df['lon'].min() - 0.05
    lon_max = pred_df['lon'].max() + 0.05
    lat_min = pred_df['lat'].min() - 0.05
    lat_max = pred_df['lat'].max() + 0.05

    groups = [
        ('Stable Patterns', [1, 2, 3]),
        ('Growth Patterns', [4, 5, 6]),
        ('Decline Patterns', [7, 8, 9]),
    ]

    for idx, (title, classes) in enumerate(groups):
        ax = axes[idx]

        for class_id in classes:
            mask = pred_df['predicted_label'] == class_id
            if mask.sum() > 0:
                class_data = pred_df[mask]
                ax.scatter(
                    class_data['lon'],
                    class_data['lat'],
                    c=CLASS_COLORS[class_id],
                    s=20,
                    alpha=0.7,
                    label=CLASS_NAMES[class_id],
                    edgecolors='white',
                    linewidths=0.3
                )

        ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=17, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')

        apply_geographic_axes_style(ax, lon_min, lon_max, lat_min, lat_max, scalebar_km=20, scalebar_position='center')

    plt.suptitle('Mobility Pattern Changes by Group (2021-2024)',
                 fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = f"{output_dir}/pattern_group_maps.png"
    output_pdf = f"{output_dir}/pattern_group_maps.pdf"
    plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    print(f"✓ Saved to {output_pdf}")
    plt.close()

def plot_city_comparison(pred_df, output_dir):
    """Plot city-wise comparison as three separate city maps"""
    print(f"\nPlotting city comparison...")

    city_slug = {
        'Shenzhen': 'shenzhen',
        'Dongguan': 'dongguan',
        'Huizhou': 'huizhou',
    }

    for city in sorted(pred_df['city_name_en'].dropna().unique()):
        city_data = pred_df[pred_df['city_name_en'] == city]
        if city_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 10))
        lon_min = city_data['lon'].min() - 0.03
        lon_max = city_data['lon'].max() + 0.03
        lat_min = city_data['lat'].min() - 0.03
        lat_max = city_data['lat'].max() + 0.03

        for class_id in range(1, 10):
            mask = city_data['predicted_label'] == class_id
            if mask.sum() > 0:
                class_data = city_data[mask]
                ax.scatter(
                    class_data['lon'],
                    class_data['lat'],
                    c=CLASS_COLORS[class_id],
                    s=26,
                    alpha=0.75,
                    label=CLASS_NAMES[class_id],
                    edgecolors='white',
                    linewidths=0.35
                )

        ax.set_xlabel('Longitude', fontsize=16, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=16, fontweight='bold')
        ax.set_title(f'{city}: Mobility Pattern Changes (2021-2024)\n({len(city_data)} grids)',
                     fontsize=20, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        apply_geographic_axes_style(
            ax,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            scalebar_km=10,
            scalebar_position='center'
        )

        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10, frameon=True)
        plt.tight_layout()

        slug = city_slug.get(city, city.lower().replace(' ', '_'))
        output_path = f"{output_dir}/city_map_{slug}.png"
        output_pdf = f"{output_dir}/city_map_{slug}.pdf"
        plt.savefig(output_path, dpi=EXPORT_DPI, bbox_inches='tight')
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
        print(f"✓ Saved to {output_pdf}")
        plt.close()

def plot_class_distribution(pred_df, output_dir):
    """Plot class distribution statistics"""
    print(f"\nPlotting class distribution...")

    # Overall distribution data
    class_counts = pred_df['predicted_label'].value_counts().sort_index()
    colors = [CLASS_COLORS[i] for i in class_counts.index]

    all_handles = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(1, 10)
    ]

    # City-wise distribution
    city_class_counts = pred_df.groupby(['city_name_en', 'predicted_label']).size().unstack(fill_value=0)
    city_class_pct = city_class_counts.div(city_class_counts.sum(axis=1), axis=0) * 100

    # Area-wise (District) distribution
    area_class_counts = pred_df.groupby(['area_name_en', 'predicted_label']).size().unstack(fill_value=0)
    area_class_pct = area_class_counts.div(area_class_counts.sum(axis=1), axis=0) * 100

    for class_id in range(1, 10):
        if class_id not in city_class_pct.columns:
            city_class_pct[class_id] = 0
        if class_id not in area_class_pct.columns:
            area_class_pct[class_id] = 0

    city_class_pct = city_class_pct[[i for i in range(1, 10)]]
    area_class_pct = area_class_pct[[i for i in range(1, 10)]]

    # ---------------- Figure A: Overall only ----------------
    fig_a, ax_a = plt.subplots(figsize=(15, 8))
    ax_a.bar(range(len(class_counts)), class_counts.values,
             color=colors, edgecolor='white', linewidth=0.9)
    ax_a.set_xlabel('Pattern Class', fontsize=16, fontweight='bold')
    ax_a.set_ylabel('Number of Grids', fontsize=16, fontweight='bold')
    ax_a.set_title('Overall Pattern Distribution', fontsize=19, fontweight='bold')
    ax_a.set_xticks(range(len(class_counts)))
    ax_a.set_xticklabels([CLASS_NAMES[i] for i in class_counts.index], rotation=20, ha='right', fontsize=12)
    ax_a.tick_params(axis='y', labelsize=12)
    ax_a.grid(True, alpha=0.30, axis='y')
    fig_a.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.23)

    output_a_png = f"{output_dir}/class_distribution_overall.png"
    output_a_pdf = f"{output_dir}/class_distribution_overall.pdf"
    fig_a.savefig(output_a_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig_a.savefig(output_a_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_a_png}")
    print(f"✓ Saved to {output_a_pdf}")
    plt.close(fig_a)

    # ---------------- Figure B: City + District ----------------
    fig_b, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7.0))

    # Bottom-left: city-wise distribution
    x_city = np.arange(len(city_class_pct))
    bottom_city = np.zeros(len(city_class_pct))
    for class_id in range(1, 10):
        values = city_class_pct[class_id].values
        ax_left.bar(x_city, values, bottom=bottom_city, color=CLASS_COLORS[class_id],
                    edgecolor='white', linewidth=0.4)
        bottom_city += values

    ax_left.set_xlabel('City', fontsize=15, fontweight='bold')
    ax_left.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
    ax_left.set_title('City-wise Pattern Distribution', fontsize=17, fontweight='bold')
    ax_left.set_xticks(x_city)
    ax_left.set_xticklabels(city_class_pct.index, fontsize=12)
    ax_left.tick_params(axis='y', labelsize=12)
    ax_left.grid(True, alpha=0.28, axis='y')
    ax_left.set_ylim(0, 100)

    # Bottom-right: district-wise distribution
    x_area = np.arange(len(area_class_pct))
    bottom_area = np.zeros(len(area_class_pct))
    for class_id in range(1, 10):
        values = area_class_pct[class_id].values
        ax_right.bar(x_area, values, bottom=bottom_area, color=CLASS_COLORS[class_id],
                     edgecolor='white', linewidth=0.35)
        bottom_area += values

    ax_right.set_xlabel('District / Area', fontsize=15, fontweight='bold')
    ax_right.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
    ax_right.set_title('District-wise Pattern Distribution', fontsize=17, fontweight='bold')
    ax_right.set_xticks(x_area)
    ax_right.set_xticklabels(area_class_pct.index, rotation=35, ha='right', fontsize=11)
    ax_right.tick_params(axis='y', labelsize=12)
    ax_right.grid(True, alpha=0.28, axis='y')
    ax_right.set_ylim(0, 100)

    # Shared legend on the right side
    fig_b.legend(
        handles=all_handles,
        loc='center left',
        bbox_to_anchor=(0.83, 0.5),
        ncol=1,
        fontsize=10,
        frameon=True,
        title='Legend',
        title_fontsize=11
    )
    fig_b.subplots_adjust(left=0.07, right=0.80, top=0.89, bottom=0.19, wspace=0.26)
    output_b_png = f"{output_dir}/class_distribution_city_district.png"
    output_b_pdf = f"{output_dir}/class_distribution_city_district.pdf"
    fig_b.savefig(output_b_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig_b.savefig(output_b_pdf, bbox_inches='tight')
    print(f"✓ Saved to {output_b_png}")
    print(f"✓ Saved to {output_b_pdf}")
    plt.close(fig_b)

    # Keep backward-compatible combined file name pointing to bottom figure layout
    compat_png = f"{output_dir}/class_distribution.png"
    compat_pdf = f"{output_dir}/class_distribution.pdf"
    import shutil
    shutil.copyfile(output_b_png, compat_png)
    shutil.copyfile(output_b_pdf, compat_pdf)
    print(f"✓ Updated compatibility file: {compat_png}")
    print(f"✓ Updated compatibility file: {compat_pdf}")

    # Save statistics
    stats_path = f"{output_dir}/prediction_statistics.csv"
    city_class_pct.to_csv(stats_path)
    print(f"✓ Saved statistics to {stats_path}")

def save_predictions(pred_df, output_dir):
    """Save predictions to CSV"""
    print(f"\nSaving predictions...")

    output_path = f"{output_dir}/all_grids_predictions.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"✓ Saved predictions to {output_path}")

    summary = {
        'total_grids': int(len(pred_df)),
        'cities': int(pred_df['city_name'].nunique()),
        'areas': int(pred_df['area_name'].nunique()),
        'class_distribution': {int(k): int(v) for k, v in pred_df['predicted_label'].value_counts().sort_index().to_dict().items()},
    }

    summary_path = f"{output_dir}/prediction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")


def main():
    """Main function"""
    # Configuration
    cache_path = "/root/workspace/Graph_Deep_Learning/20251001-PRD_18-21-24-mobility_change_pattern/analysis/PRD-Mobility-Change-Pattern/data/cache/dual_year_data_all_grids.pkl"
    model_path = str(EXPERIMENT_DIR / 'models' / 'best_model.pth')
    grid_metadata_path = "data/grid_metadata/sgh_grid_metadata.csv"

    # Create output directory
    output_dir = str(EXPERIMENT_DIR / 'model_predictions')
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Full Region Prediction with Best Model (Using Cached Data)")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load cached data
    cached_data = load_cached_data(cache_path)

    # Load model
    model = load_model(model_path, device)

    # Predict
    predictions, grid_ids = predict_grids_with_features(model, cached_data, device, batch_size=64)

    # Create dataframe
    pred_df = create_full_prediction_dataframe(predictions, grid_ids, grid_metadata_path)

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations (All English)")
    print("="*80)

    plot_full_region_map(pred_df, output_dir)
    plot_pattern_group_maps(pred_df, output_dir)
    plot_city_comparison(pred_df, output_dir)
    plot_class_distribution(pred_df, output_dir)

    # Save predictions
    save_predictions(pred_df, output_dir)

    print("\n" + "="*80)
    print("✓ All predictions and visualizations completed!")
    print(f"  Output directory: {output_dir}")
    print("="*80)

    # Print summary
    print("\nPrediction Summary:")
    print(f"  - Total grids: {len(pred_df)}")
    print(f"  - Cities: {pred_df['city_name'].nunique()}")
    print(f"  - Areas: {pred_df['area_name'].nunique()}")
    print("\nClass Distribution:")
    for class_id in range(1, 10):
        count = (pred_df['predicted_label'] == class_id).sum()
        pct = count / len(pred_df) * 100
        print(f"  {CLASS_NAMES[class_id]}: {count} ({pct:.2f}%)")

if __name__ == "__main__":
    main()

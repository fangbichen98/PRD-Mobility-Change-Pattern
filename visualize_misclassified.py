"""
Extract misclassified nodes from phase41c delta_stats model and visualize.
Lightweight version that reuses training data pipeline.
Usage: python visualize_misclassified.py
"""
import sys, os, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.models.enhanced_dual_branch_model import EnhancedDualBranchModel
from src.training.dataset_pure_graph import PureGraphDualYearDataset, PureGraphBatchCollator
from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str,
        default='outputs/multiscale_temporal_20260404_202806_sampled_labels_spc250_seed202_reconstructed_phase41c_delta_stats')
    parser.add_argument('--output-dir', type=str, default='outputs/misclassified_analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load model
    print("\n--- Loading model ---")
    model_path = os.path.join(args.model_dir, 'models/best_model.pth')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config.SPATIAL_LAYERS = 3  # Phase41c trained with 3 spatial layers

    model = EnhancedDualBranchModel(
        temporal_input_size=2, hidden_size=config.FUSION_HIDDEN_SIZE,
        num_classes=config.NUM_CLASSES, num_time_steps=config.TIME_STEPS,
        dropout=config.LSTM_DROPOUT, spatial_model='GINE',
        temporal_model='TRANSFORMER', use_delta_stats=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Model: epoch={checkpoint['epoch']}, val_acc={checkpoint['accuracy']:.4f}")

    # 2. Load data (reuse training pipeline)
    print("\n--- Loading data ---")
    label_path = 'data/sampled_labels_spc250_seed202_reconstructed.csv'
    data = prepare_dual_year_experiment_data(
        label_path=label_path, samples_per_class=None,
        use_cache=True, spatial_model='GINE',
        edge_feature_mode='flow_distance_direction'
    )

    # Prepare temporal features (same as train)
    temporal_features_2021, temporal_features_2024 = {}, {}
    for grid_id, features in data['change_features'].items():
        temporal_features_2021[grid_id] = features[:, [0, 1]]
        temporal_features_2024[grid_id] = features[:, [2, 3]]

    grid_ids = list(data['labels'].keys())
    dataset = PureGraphDualYearDataset(
        temporal_features_2021=temporal_features_2021,
        temporal_features_2024=temporal_features_2024,
        labels=data['labels'], grid_ids=grid_ids
    )

    # Resolve splits from saved manifest
    split_path = os.path.join(args.model_dir, 'metrics/split_manifest.json')
    with open(split_path) as f:
        manifest = json.load(f)
    test_grid_ids_manifest = manifest['splits']['test']

    dataset_grid_ids = [int(g) for g in dataset.grid_ids]
    gid_to_idx = {g: i for i, g in enumerate(dataset_grid_ids)}
    test_indices = [gid_to_idx[int(g)] for g in test_grid_ids_manifest]
    test_dataset = Subset(dataset, test_indices)
    print(f"Test samples: {len(test_dataset)}")

    # Prepare all temporal tensors for collator
    num_nodes = len(data['grid_id_to_idx'])
    feat_dim = torch.tensor(temporal_features_2021[grid_ids[0]]).shape[-1]
    all_temporal_2021 = torch.zeros(num_nodes, 168, feat_dim)
    all_temporal_2024 = torch.zeros(num_nodes, 168, feat_dim)
    all_raw_temporal_2021 = torch.zeros(num_nodes, 168, feat_dim)
    all_raw_temporal_2024 = torch.zeros(num_nodes, 168, feat_dim)

    for gid, idx in data['grid_id_to_idx'].items():
        if gid in temporal_features_2021:
            all_temporal_2021[idx] = torch.tensor(temporal_features_2021[gid], dtype=torch.float32)
            all_temporal_2024[idx] = torch.tensor(temporal_features_2024[gid], dtype=torch.float32)
        if gid in data['flows_2021']:
            raw_arr_2021 = data['flows_2021'][gid]
            raw_arr_2024 = data['flows_2024'][gid]
            all_raw_temporal_2021[idx] = torch.tensor(raw_arr_2021[:, :feat_dim], dtype=torch.float32)
            all_raw_temporal_2024[idx] = torch.tensor(raw_arr_2024[:, :feat_dim], dtype=torch.float32)

    # Register node coords for GINE
    meta = pd.read_csv(config.GRID_METADATA_PATH)
    coords = torch.tensor(meta[['lon', 'lat']].values, dtype=torch.float32)
    coord_lookup = dict(zip(meta['grid_id'], range(len(meta))))
    model.register_node_coords(coords)

    collator = PureGraphBatchCollator(
        graphs_2021=data['graphs_2021'], graphs_2024=data['graphs_2024'],
        grid_id_to_idx=data['grid_id_to_idx'],
        all_temporal_2021=all_temporal_2021, all_temporal_2024=all_temporal_2024,
        all_raw_temporal_2021=all_raw_temporal_2021, all_raw_temporal_2024=all_raw_temporal_2024
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=collator, num_workers=0)

    # 3. Run inference
    print("\n--- Running inference ---")
    all_grid_ids_list, all_true, all_pred = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            node_indices = batch['node_indices'].to(device)
            labels = batch['labels'].to(device)
            num_nodes_batch = batch['num_nodes']
            batch_grid_ids = batch['grid_ids']

            at21 = batch['all_temporal_2021'].to(device)
            at24 = batch['all_temporal_2024'].to(device)
            ar21 = batch['all_raw_temporal_2021']
            ar24 = batch['all_raw_temporal_2024']
            if ar21 is not None: ar21 = ar21.to(device)
            if ar24 is not None: ar24 = ar24.to(device)

            def _to_device(graphs):
                return [(torch.from_numpy(e).to(device) if isinstance(e, np.ndarray) else e.to(device),
                         torch.from_numpy(a).to(device) if isinstance(a, np.ndarray) else a.to(device))
                        for e, a in graphs]

            logits = model(
                x_2021=at21, x_2024=at24,
                graphs_2021=_to_device(batch['graphs_2021']),
                graphs_2024=_to_device(batch['graphs_2024']),
                num_nodes=num_nodes_batch, node_indices=node_indices,
                raw_x_2021=ar21, raw_x_2024=ar24,
            )

            preds = logits.argmax(dim=1)
            all_grid_ids_list.extend(batch_grid_ids)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    acc = (all_true == all_pred).mean() * 100
    print(f"Test Accuracy: {acc:.2f}% ({(all_true == all_pred).sum()}/{len(all_true)})")

    # 4. Load coordinates
    coord_map = dict(zip(meta['grid_id'], zip(meta['lon'], meta['lat'])))

    # 5. Build DataFrame
    results = pd.DataFrame({
        'grid_id': all_grid_ids_list,
        'true_label': all_true + 1,
        'pred_label': all_pred + 1,
        'correct': all_true == all_pred,
    })
    results['lon'] = results['grid_id'].map(lambda g: coord_map.get(g, (np.nan, np.nan))[0])
    results['lat'] = results['grid_id'].map(lambda g: coord_map.get(g, (np.nan, np.nan))[1])

    misclassified = results[~results['correct']].copy()
    print(f"Misclassified: {len(misclassified)} nodes")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=list(range(9)))

    # Per-class stats
    print("\nPer-class misclassification:")
    for cls in range(1, 10):
        total = (results['true_label'] == cls).sum()
        wrong = (misclassified['true_label'] == cls).sum()
        print(f"  Class {cls}: {wrong}/{total} ({wrong/total*100:.0f}%)")

    # Top confusion pairs
    pairs = []
    for i in range(9):
        for j in range(9):
            if i != j and cm[i][j] > 0:
                pairs.append((cm[i][j], i+1, j+1))
    pairs.sort(reverse=True)
    print("\nTop confusion pairs (true -> pred):")
    for count, t, p in pairs[:10]:
        print(f"  Class {t} -> Class {p}: {count}")

    # Save CSV
    results.to_csv(os.path.join(args.output_dir, 'all_test_results.csv'), index=False)
    misclassified.to_csv(os.path.join(args.output_dir, 'misclassified_nodes.csv'), index=False)

    # 6. Visualize
    print("\n--- Generating visualizations ---")
    visualize(results, misclassified, cm, args.output_dir)


def visualize(results, misclassified, cm, output_dir):
    class_names = [f'C{i}' for i in range(1, 10)]

    # Fig 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (phase41c + delta_stats)', fontsize=14)
    for i in range(9):
        for j in range(9):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', color=color, fontsize=10)
    plt.colorbar(im, ax=ax, label='Recall')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Fig 2: Spatial map
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    valid = results.dropna(subset=['lon', 'lat'])
    sc = axes[0].scatter(valid['lon'], valid['lat'], c=valid['true_label'],
                          cmap='tab10', s=12, alpha=0.5, edgecolors='none', vmin=1, vmax=9)
    axes[0].set_title(f'All Test Nodes ({len(valid)}), Color=True Label', fontsize=13)
    axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')
    plt.colorbar(sc, ax=axes[0], label='True Class', ticks=range(1, 10))

    cor = results[results['correct']].dropna(subset=['lon', 'lat'])
    mis = misclassified.dropna(subset=['lon', 'lat'])
    axes[1].scatter(cor['lon'], cor['lat'], c=cor['true_label'],
                     cmap='tab10', s=5, alpha=0.2, edgecolors='none', vmin=1, vmax=9)
    if len(mis) > 0:
        sc2 = axes[1].scatter(mis['lon'], mis['lat'], c=mis['true_label'],
                               cmap='tab10', s=50, alpha=0.9, edgecolors='red',
                               linewidths=1.2, marker='x', vmin=1, vmax=9, zorder=5)
    axes[1].set_title(f'Misclassified ({len(mis)} X marks) over Correct (dots)', fontsize=13)
    axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
    plt.colorbar(sc2, ax=axes[1], label='True Class', ticks=range(1, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_misclassified.png'), dpi=150)
    plt.close()

    # Fig 3: Per-class bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(9)
    w = 0.35
    correct_counts = [(results['true_label'] == c).sum() - (misclassified['true_label'] == c).sum() for c in range(1, 10)]
    wrong_counts = [(misclassified['true_label'] == c).sum() for c in range(1, 10)]
    b1 = ax.bar(x - w/2, correct_counts, w, label='Correct', color='#4CAF50')
    b2 = ax.bar(x + w/2, wrong_counts, w, label='Wrong', color='#F44336')
    for bar in b1:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{int(bar.get_height())}', ha='center', fontsize=9)
    for bar in b2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{int(bar.get_height())}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_xlabel('True Class'); ax.set_ylabel('Count')
    ax.set_title('Per-Class: Correct vs Misclassified', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_bar.png'), dpi=150)
    plt.close()

    # Fig 4: Top confusion pairs
    pairs = []
    for i in range(9):
        for j in range(9):
            if i != j and cm[i][j] > 0:
                pairs.append((cm[i][j], f'C{i+1}', f'C{j+1}'))
    pairs.sort(reverse=True)
    top = pairs[:8]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f'{t} -> {p}' for _, t, p in top]
    counts = [c for c, _, _ in top]
    colors = ['#FF6B6B' if c >= 5 else '#FFA07A' if c >= 3 else '#FFD700' for c in counts]
    ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    for i, (c, _, _) in enumerate(top):
        ax.text(c + 0.1, i, str(c), va='center', fontsize=10)
    ax.set_xlabel('Count'); ax.set_title('Top Confusion Pairs (True -> Predicted)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_confusion_pairs.png'), dpi=150)
    plt.close()

    print(f"Saved 4 figures to {output_dir}/")


if __name__ == '__main__':
    main()

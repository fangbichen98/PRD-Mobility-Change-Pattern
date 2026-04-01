"""
Baseline ML classifiers for mobility change pattern classification.

Models: XGBoost, Random Forest, Logistic Regression, SVM (RBF)
Features: hand-crafted statistics from raw OD flow data (2021 + 2024)
Protocol: same spc250 frozen split as phase41c for fair comparison

Usage:
    python train_baseline_ml.py
    python train_baseline_ml.py --label-path data/sampled_labels_spc250_seed202_reconstructed.csv \
        --split-manifest data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json
"""

import os
import json
import argparse
import logging
import hashlib
import pickle
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in km."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def extract_features_for_grid(grid_id, od_df, coord_lookup, num_hours=168):
    """
    Extract features for a single grid from one year's OD data.

    Temporal (per grid):
      - 周总: total_in, total_out, total_flow
      - 日均: daily_mean_in, daily_mean_out, daily_mean_total
      - 小时均: hourly_mean_in, hourly_mean_out, hourly_mean_total

    Spatial (per grid):
      - wamd_out: o点出行加权平均距离 (km)
      - wamd_in:  d点来源加权平均距离 (km)
      - out_degree: 唯一目的地数（出度）
      - in_degree:  唯一来源数（入度）

    Returns a dict of 13 features.
    """
    feats = {}

    inflow_df  = od_df[od_df['d_grid_500'] == grid_id]
    outflow_df = od_df[od_df['o_grid_500'] == grid_id]

    # ---- hourly series ----
    min_date = od_df['date_dt'].min()
    hourly_in  = np.zeros(num_hours, dtype=np.float32)
    hourly_out = np.zeros(num_hours, dtype=np.float32)

    if len(inflow_df) > 0:
        tmp = inflow_df.copy()
        tmp['hidx'] = (tmp['date_dt'] - min_date).dt.days * 24 + tmp['time']
        for h, v in tmp.groupby('hidx')['num_total'].sum().items():
            if 0 <= h < num_hours:
                hourly_in[h] = v

    if len(outflow_df) > 0:
        tmp = outflow_df.copy()
        tmp['hidx'] = (tmp['date_dt'] - min_date).dt.days * 24 + tmp['time']
        for h, v in tmp.groupby('hidx')['num_total'].sum().items():
            if 0 <= h < num_hours:
                hourly_out[h] = v

    hourly_total = hourly_in + hourly_out

    # ---- 周总 ----
    feats['total_in']    = float(hourly_in.sum())
    feats['total_out']   = float(hourly_out.sum())
    feats['total_flow']  = float(hourly_total.sum())

    # ---- 日均 ----
    daily_in    = hourly_in.reshape(7, 24).sum(axis=1)
    daily_out   = hourly_out.reshape(7, 24).sum(axis=1)
    daily_total = hourly_total.reshape(7, 24).sum(axis=1)
    feats['daily_mean_in']    = float(daily_in.mean())
    feats['daily_mean_out']   = float(daily_out.mean())
    feats['daily_mean_total'] = float(daily_total.mean())

    # ---- 小时均 ----
    feats['hourly_mean_in']    = float(hourly_in.mean())
    feats['hourly_mean_out']   = float(hourly_out.mean())
    feats['hourly_mean_total'] = float(hourly_total.mean())

    # ---- 空间 OD: outflow (o点) ----
    lon, lat = coord_lookup.get(grid_id, (None, None))
    if lon is not None and len(outflow_df) > 0:
        dests = outflow_df[['d_grid_500', 'num_total']].copy()
        dests = dests[dests['d_grid_500'].isin(coord_lookup)]
        if len(dests) > 0:
            dst_coords = np.array([coord_lookup[d] for d in dests['d_grid_500']])
            dists = haversine_km(lon, lat, dst_coords[:, 0], dst_coords[:, 1])
            flows = dests['num_total'].values.astype(np.float32)
            feats['wamd_out']   = float((flows * dists).sum() / (flows.sum() + 1e-6))
            feats['out_degree'] = float(dests['d_grid_500'].nunique())
        else:
            feats['wamd_out']   = 0.0
            feats['out_degree'] = 0.0
    else:
        feats['wamd_out']   = 0.0
        feats['out_degree'] = 0.0

    # ---- 空间 OD: inflow (d点) ----
    if lon is not None and len(inflow_df) > 0:
        srcs = inflow_df[['o_grid_500', 'num_total']].copy()
        srcs = srcs[srcs['o_grid_500'].isin(coord_lookup)]
        if len(srcs) > 0:
            src_coords = np.array([coord_lookup[s] for s in srcs['o_grid_500']])
            dists = haversine_km(lon, lat, src_coords[:, 0], src_coords[:, 1])
            flows = srcs['num_total'].values.astype(np.float32)
            feats['wamd_in']   = float((flows * dists).sum() / (flows.sum() + 1e-6))
            feats['in_degree'] = float(srcs['o_grid_500'].nunique())
        else:
            feats['wamd_in']   = 0.0
            feats['in_degree'] = 0.0
    else:
        feats['wamd_in']   = 0.0
        feats['in_degree'] = 0.0

    return feats


def build_feature_matrix(grid_ids, od_2021, od_2024, coord_lookup, cache_path=None):
    """
    Build (N, D) feature matrix for all grid_ids.
    Features = [year2021_feats | year2024_feats | delta_feats]
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading feature cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logger.info(f"Extracting features for {len(grid_ids)} grids ...")

    rows_2021, rows_2024 = [], []
    feat_keys = None

    for gid in tqdm(grid_ids, desc="Feature extraction"):
        f21 = extract_features_for_grid(gid, od_2021, coord_lookup)
        f24 = extract_features_for_grid(gid, od_2024, coord_lookup)
        if feat_keys is None:
            feat_keys = list(f21.keys())
        rows_2021.append([f21[k] for k in feat_keys])
        rows_2024.append([f24[k] for k in feat_keys])

    arr_2021 = np.array(rows_2021, dtype=np.float32)
    arr_2024 = np.array(rows_2024, dtype=np.float32)

    X = np.concatenate([arr_2021, arr_2024], axis=1)
    col_names = (
        [f"y21_{k}" for k in feat_keys] +
        [f"y24_{k}" for k in feat_keys]
    )

    result = (X, col_names)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Feature cache saved: {cache_path}")

    return result


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_od_chunked(path, usecols=None, chunksize=500_000):
    logger.info(f"Loading OD data: {path}")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        chunk['date_dt'] = pd.to_datetime(chunk['date_dt'], format='%Y%m%d')
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Loaded {len(df):,} rows")
    return df


def load_coord_lookup(metadata_path):
    meta = pd.read_csv(metadata_path)
    return {row['grid_id']: (row['lon'], row['lat']) for _, row in meta.iterrows()}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix: {out_path}")


def plot_baseline_comparison(results, out_path):
    """
    Line chart comparing accuracy and macro F1 across all baseline models.
    """
    model_names = [r['model'] for r in results]
    accs  = [r['test_accuracy'] for r in results]
    f1s   = [r['test_f1'] for r in results]

    # Add phase41c reference
    ref_acc = 73.33
    ref_f1  = 0.7340

    x = np.arange(len(model_names))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, accs, marker='o', linewidth=2, markersize=7, label='Accuracy (%)', color='#2196F3')
    ax.plot(x, [f * 100 for f in f1s], marker='s', linewidth=2, markersize=7,
            label='Macro F1 (×100)', color='#FF9800', linestyle='--')

    # Reference lines for phase41c
    ax.axhline(ref_acc, color='#2196F3', linewidth=1.2, linestyle=':', alpha=0.7,
               label=f'Phase41c Acc ({ref_acc:.1f}%)')
    ax.axhline(ref_f1 * 100, color='#FF9800', linewidth=1.2, linestyle=':', alpha=0.7,
               label=f'Phase41c F1 ({ref_f1*100:.1f})')

    # Annotate values
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.annotate(f'{a:.1f}', (x[i], a), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9, color='#2196F3')
        ax.annotate(f'{f*100:.1f}', (x[i], f * 100), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=9, color='#FF9800')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Baseline ML Models vs Phase41c (TRANSFORMER+GINE)', fontsize=12, pad=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison chart: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-path', type=str,
                        default='data/sampled_labels_spc250_seed202_reconstructed.csv')
    parser.add_argument('--split-manifest', type=str,
                        default='data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (auto-generated if not set)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore feature cache and recompute')
    return parser.parse_args()


def main():
    args = parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    label_base = os.path.splitext(os.path.basename(args.label_path))[0]
    out_dir = args.output_dir or f"outputs/baseline_ml_{ts}_{label_base}"
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # ---- load labels ----
    labels_df = pd.read_csv(args.label_path)
    grid_label = dict(zip(labels_df['grid_id'], labels_df['label_idx']))
    all_grid_ids = list(grid_label.keys())
    logger.info(f"Loaded {len(all_grid_ids)} labeled grids from {args.label_path}")

    # ---- load split ----
    with open(args.split_manifest) as f:
        manifest = json.load(f)
    splits = manifest['splits']
    train_ids = [g for g in splits['train'] if g in grid_label]
    val_ids   = [g for g in splits['val']   if g in grid_label]
    test_ids  = [g for g in splits['test']  if g in grid_label]
    logger.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ---- load OD data ----
    usecols = ['date_dt', 'time', 'o_grid_500', 'd_grid_500', 'num_total']
    od_2021 = load_od_chunked(config.OD_2021_PATH, usecols=usecols)
    od_2024 = load_od_chunked(config.OD_2024_PATH, usecols=usecols)

    # ---- load coordinates ----
    coord_lookup = load_coord_lookup(config.GRID_METADATA_PATH)
    logger.info(f"Loaded coordinates for {len(coord_lookup)} grids")

    # ---- build features ----
    label_hash = hashlib.md5(args.label_path.encode()).hexdigest()[:8]
    cache_path = None if args.no_cache else f"data/cache/baseline_ml_features_{label_hash}_v2.pkl"

    X_all, col_names = build_feature_matrix(
        all_grid_ids, od_2021, od_2024, coord_lookup, cache_path=cache_path
    )
    gid_to_idx = {gid: i for i, gid in enumerate(all_grid_ids)}

    def get_Xy(ids):
        idx = [gid_to_idx[g] for g in ids]
        X = X_all[idx]
        y = np.array([grid_label[g] for g in ids])
        return X, y

    X_train, y_train = get_Xy(train_ids)
    X_val,   y_val   = get_Xy(val_ids)
    X_test,  y_test  = get_Xy(test_ids)

    # combine train+val for final fit (same as deep model which uses val for early stopping only)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    logger.info(f"Feature matrix shape: {X_train.shape[1]} dims")

    # ---- scale ----
    scaler = StandardScaler()
    X_train_s    = scaler.fit_transform(X_train)
    X_trainval_s = scaler.transform(X_trainval)
    X_test_s     = scaler.transform(X_test)

    # ---- models ----
    n_classes = len(set(grid_label.values()))
    class_names = [str(i) for i in range(n_classes)]

    models = [
        ('XGBoost', xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, n_jobs=-1, verbosity=0
        )),
        ('Random Forest', RandomForestClassifier(
            n_estimators=300, max_depth=None,
            random_state=42, n_jobs=-1
        )),
        ('Logistic Regression', LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        )),
        ('SVM (RBF)', SVC(
            kernel='rbf', C=1.0, gamma='scale',
            decision_function_shape='ovr', random_state=42
        )),
    ]

    all_results = []

    for name, clf in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*60}")

        # tree models don't need scaling; linear/SVM do
        needs_scale = name in ('Logistic Regression', 'SVM (RBF)')
        Xtr = X_trainval_s if needs_scale else X_trainval
        Xte = X_test_s     if needs_scale else X_test

        clf.fit(Xtr, y_trainval)
        y_pred = clf.predict(Xte)

        acc = accuracy_score(y_test, y_pred) * 100
        f1  = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"  Test Accuracy: {acc:.2f}%")
        logger.info(f"  Macro F1:      {f1:.4f}")

        # confusion matrix plot
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        cm_path = os.path.join(out_dir, f"confusion_matrix_{safe_name}.png")
        plot_confusion_matrix(cm, class_names, f"{name} — Acc {acc:.1f}% / F1 {f1:.4f}", cm_path)

        result = {
            'model': name,
            'test_accuracy': round(acc, 4),
            'test_f1': round(f1, 6),
            'per_class_f1': {k: round(v['f1-score'], 4)
                             for k, v in report.items()
                             if k not in ('accuracy', 'macro avg', 'weighted avg')},
            'confusion_matrix': cm.tolist(),
        }
        all_results.append(result)

    # ---- comparison chart ----
    chart_path = os.path.join(out_dir, 'baseline_comparison.png')
    plot_baseline_comparison(all_results, chart_path)

    # ---- save JSON ----
    summary = {
        'timestamp': ts,
        'label_file': args.label_path,
        'split_manifest': args.split_manifest,
        'train_samples': len(X_trainval),
        'test_samples': len(X_test),
        'feature_dim': int(X_all.shape[1]),
        'reference_phase41c': {'test_accuracy': 73.33, 'test_f1': 0.7340},
        'results': all_results,
    }
    json_path = os.path.join(out_dir, 'baseline_ml_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved: {json_path}")

    # ---- print summary table ----
    logger.info("\n" + "=" * 55)
    logger.info(f"{'Model':<25} {'Acc (%)':>10} {'Macro F1':>10}")
    logger.info("=" * 55)
    for r in all_results:
        logger.info(f"{r['model']:<25} {r['test_accuracy']:>10.2f} {r['test_f1']:>10.4f}")
    logger.info("-" * 55)
    logger.info(f"{'Phase41c (reference)':<25} {'73.33':>10} {'0.7340':>10}")
    logger.info("=" * 55)


if __name__ == '__main__':
    main()

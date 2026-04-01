"""
Build a full-region features cache for visualization.

This generates change_features + flows for ALL grids that appear in the OD data
(not just the labeled training grids), so predict_with_cache.py can produce
full-region prediction maps.

Output: data/cache/features/dual_year_features_all_grids.pkl
  - change_features: {grid_id: (168, 4)} for every grid with flow in either year
  - flows_2021 / flows_2024: raw hourly flows (168, 2)
  - grid_id_to_idx: from the matching graph variant cache (passed via --graph-cache)
  - is_full_grid_cache: True  (flag for resolver)
"""

import sys
import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_od_year(data_path: str, chunksize: int = 500_000) -> pd.DataFrame:
    logger.info(f"Loading {data_path} ...")
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        chunk['date_dt'] = pd.to_datetime(chunk['date_dt'], format='%Y%m%d')
        chunk = chunk[chunk['time'].between(0, 23)]
        chunk = chunk[chunk['num_total'] > 0]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Loaded {len(df):,} records")
    return df


def filter_training_period(df: pd.DataFrame, train_days: int = 7) -> pd.DataFrame:
    min_date = df['date_dt'].min()
    cutoff = min_date + pd.Timedelta(days=train_days)
    return df[df['date_dt'] < cutoff].copy()


def aggregate_flows_vectorized(od_df: pd.DataFrame, grid_ids_set: set,
                               num_hours: int = 168) -> dict:
    """
    Vectorized aggregation: build a (N_grids, 168, 2) array in one pass.
    Returns {grid_id: (168, 2)}.
    """
    logger.info(f"Aggregating flows for {len(grid_ids_set):,} grids (vectorized) ...")

    min_date = od_df['date_dt'].min()
    hour_idx = ((od_df['date_dt'] - min_date).dt.days * 24 + od_df['time']).clip(0, num_hours - 1).values

    # Assign a dense integer index to each grid_id
    grid_ids_list = sorted(grid_ids_set)
    gid_to_i = {g: i for i, g in enumerate(grid_ids_list)}
    N = len(grid_ids_list)

    # (N, 168, 2)  channel 0 = inflow, channel 1 = outflow
    arr = np.zeros((N, num_hours, 2), dtype=np.float32)

    flow_vals = od_df['num_total'].values.astype(np.float32)

    # inflow: destination grid
    dst = od_df['d_grid_500'].values
    mask_in = np.isin(dst, grid_ids_list)
    dst_f = dst[mask_in]
    h_in  = hour_idx[mask_in]
    v_in  = flow_vals[mask_in]
    gi_in = np.array([gid_to_i[g] for g in dst_f], dtype=np.int32)
    np.add.at(arr, (gi_in, h_in, 0), v_in)

    # outflow: origin grid
    src = od_df['o_grid_500'].values
    mask_out = np.isin(src, grid_ids_list)
    src_f = src[mask_out]
    h_out = hour_idx[mask_out]
    v_out = flow_vals[mask_out]
    gi_out = np.array([gid_to_i[g] for g in src_f], dtype=np.int32)
    np.add.at(arr, (gi_out, h_out, 1), v_out)

    grid_flows = {g: arr[i] for i, g in enumerate(grid_ids_list)}
    logger.info(f"  Done. {len(grid_flows):,} grids aggregated.")
    return grid_flows


def compute_change_features(flows_2021: dict, flows_2024: dict) -> dict:
    """Compute log-transformed (168,4) features for grids present in either year."""
    all_grids = set(flows_2021.keys()) | set(flows_2024.keys())
    zero_arr = np.zeros((config.TIME_STEPS, 2), dtype=np.float32)
    change_features = {}
    for grid_id in all_grids:
        f21 = flows_2021.get(grid_id, zero_arr)
        f24 = flows_2024.get(grid_id, zero_arr)
        combined = np.stack([
            np.log1p(f21[:, 0]),
            np.log1p(f21[:, 1]),
            np.log1p(f24[:, 0]),
            np.log1p(f24[:, 1]),
        ], axis=1)  # (168, 4)
        change_features[grid_id] = combined
    return change_features


def main():
    parser = argparse.ArgumentParser(description='Build full-grid features cache for visualization')
    parser.add_argument('--graph-cache', required=True,
                        help='Path to graph variant cache (to copy grid_id_to_idx)')
    parser.add_argument('--train-feature-cache', default=None,
                        help='Path to training feature cache — used to embed the correct '
                             'train_flow_grid_ids so spatial raw-feature masking at inference '
                             'matches training exactly.')
    parser.add_argument('--out', default='data/cache/features/dual_year_features_all_grids.pkl',
                        help='Output path')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load graph cache to get valid grid set and grid_id_to_idx
    logger.info(f"Loading graph cache: {args.graph_cache}")
    with open(args.graph_cache, 'rb') as f:
        graph_data = pickle.load(f)
    grid_id_to_idx = graph_data['grid_id_to_idx']
    valid_grid_ids = set(grid_id_to_idx.keys())
    logger.info(f"  Valid grids from graph cache: {len(valid_grid_ids):,}")

    num_hours = config.TRAIN_DAYS * 24  # 168

    od_2021_path = getattr(config, 'OD_2021_PATH', 'data/2021_sgh_week.csv')
    od_2024_path = getattr(config, 'OD_2024_PATH', 'data/2024_sgh_week.csv')
    # Phase31 experiment used sgh data; fall back to sgh if config points elsewhere
    if not os.path.exists(od_2021_path):
        od_2021_path = 'data/2021_sgh_week.csv'
    if not os.path.exists(od_2024_path):
        od_2024_path = 'data/2024_sgh_week.csv'
    logger.info(f"OD 2021: {od_2021_path}")
    logger.info(f"OD 2024: {od_2024_path}")

    # Load and process 2021
    od_2021 = load_od_year(od_2021_path)
    od_2021 = filter_training_period(od_2021, config.TRAIN_DAYS)
    od_2021 = od_2021[od_2021['o_grid_500'].isin(valid_grid_ids) &
                      od_2021['d_grid_500'].isin(valid_grid_ids)]
    grids_2021 = set(od_2021['o_grid_500'].unique()) | set(od_2021['d_grid_500'].unique())
    logger.info(f"2021: {len(od_2021):,} records, {len(grids_2021):,} unique grids")

    # Load and process 2024
    od_2024 = load_od_year(od_2024_path)
    od_2024 = filter_training_period(od_2024, config.TRAIN_DAYS)
    od_2024 = od_2024[od_2024['o_grid_500'].isin(valid_grid_ids) &
                      od_2024['d_grid_500'].isin(valid_grid_ids)]
    grids_2024 = set(od_2024['o_grid_500'].unique()) | set(od_2024['d_grid_500'].unique())
    logger.info(f"2024: {len(od_2024):,} records, {len(grids_2024):,} unique grids")

    all_grids = sorted(grids_2021 | grids_2024)
    logger.info(f"Total grids with flow in either year: {len(all_grids):,}")

    # Aggregate flows
    flows_2021 = aggregate_flows_vectorized(od_2021, grids_2021 & valid_grid_ids, num_hours)
    flows_2024 = aggregate_flows_vectorized(od_2024, grids_2024 & valid_grid_ids, num_hours)

    # Compute change features
    logger.info("Computing change features ...")
    change_features = compute_change_features(flows_2021, flows_2024)
    logger.info(f"  change_features: {len(change_features):,} grids, shape {next(iter(change_features.values())).shape}")

    # Load train_flow_grid_ids from training feature cache if provided.
    # This is CRITICAL for models using raw_temporal_mean node features:
    # at training time only the labeled grids had non-zero spatial node features,
    # so we must reproduce that exact mask at inference time.
    train_flow_grid_ids = None
    train_flow_label_hash = None
    if args.train_feature_cache:
        logger.info(f"Loading train_flow_grid_ids from: {args.train_feature_cache}")
        with open(args.train_feature_cache, 'rb') as f:
            train_feat = pickle.load(f)
        train_flow_grid_ids = train_feat.get('train_flow_grid_ids',
                                             sorted(train_feat.get('flows_2021', {}).keys()))
        train_flow_label_hash = train_feat.get('label_file_hash') or train_feat.get('train_flow_label_hash')
        logger.info(f"  train_flow_grid_ids: {len(train_flow_grid_ids):,} grids")
        logger.info(f"  label_file_hash: {train_flow_label_hash}")
    else:
        logger.warning("--train-feature-cache not provided; train_flow_grid_ids will be absent. "
                       "Spatial raw-feature masking may not match training for raw_temporal_mean models.")

    cache = {
        'change_features':          change_features,
        'flows_2021':               flows_2021,
        'flows_2024':               flows_2024,
        'is_full_grid_cache':       True,
        'train_flow_grid_ids':      train_flow_grid_ids,   # None if not provided
        'train_flow_label_hash':    train_flow_label_hash,
        'label_file_hash':          train_flow_label_hash,
        'train_flow_total_samples': len(train_flow_grid_ids) if train_flow_grid_ids else None,
    }

    logger.info(f"Saving to {args.out} ...")
    with open(args.out, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Done.")


if __name__ == '__main__':
    main()

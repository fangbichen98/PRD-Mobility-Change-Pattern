import pandas as pd
import numpy as np
import pickle
import os
import glob
import json
from tqdm import tqdm


def load_pickle(path):
    with open(path, 'rb') as f:
        import torch
        return pickle.load(f)


def load_experiment_manifest(experiment_dir):
    manifest_path = os.path.join(experiment_dir, 'metrics', 'test_results.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_train_flow_cache():
    explicit_cache = os.environ.get('TRAIN_FLOW_CACHE_PATH')
    if explicit_cache:
        return explicit_cache

    experiment_dir = os.environ.get('VIS_EXPERIMENT_DIR') or os.environ.get('EXPERIMENT_DIR')
    if not experiment_dir:
        raise RuntimeError(
            "TRAIN_FLOW_CACHE_PATH is not set and no EXPERIMENT_DIR/VIS_EXPERIMENT_DIR was provided. "
            "Refusing to guess a training-flow cache because that can silently mix experiments."
        )

    manifest = load_experiment_manifest(experiment_dir)
    expected = manifest.get('data_info', {})
    expected_hash = expected.get('label_file_hash')
    expected_total = expected.get('total_samples')

    best_path = None
    best_score = (-1, -1)
    for path in sorted(glob.glob('data/cache/features/dual_year_features_*.pkl')):
        try:
            data = load_pickle(path)
        except Exception:
            continue

        if expected_hash and data.get('label_file_hash') != expected_hash:
            continue

        label_count = len(data.get('labels', {}))
        if expected_total is not None and label_count != int(expected_total):
            continue

        train_flow_grid_ids = data.get('train_flow_grid_ids')
        has_explicit_train_mask = train_flow_grid_ids is not None
        if train_flow_grid_ids is None:
            train_flow_grid_ids = sorted(
                set(data.get('flows_2021', {}).keys()) |
                set(data.get('flows_2024', {}).keys())
            )
        train_flow_count = len(train_flow_grid_ids)
        score = (
            1 if has_explicit_train_mask else 0,
            1 if train_flow_count > 0 else 0,
            train_flow_count,
        )
        if score > best_score:
            best_score = score
            best_path = path

    if best_path is None:
        raise FileNotFoundError(
            f"Could not resolve a training-flow cache for experiment {experiment_dir} "
            f"(label_file_hash={expected_hash}, total_samples={expected_total})."
        )

    print(
        f"Auto-selected training-flow cache from experiment manifest: {best_path} "
        f"(label_file_hash={expected_hash}, total_samples={expected_total})"
    )
    return best_path

def process_year(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, usecols=['o_grid_500', 'd_grid_500', 'date_dt', 'time', 'num_total'])
    
    print("Converting dates...")
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    min_date = df['date_dt'].min()
    df['day_idx'] = (df['date_dt'] - min_date).dt.days
    df['hour_idx'] = df['day_idx'] * 24 + df['time']
    
    # filter to only 7 days
    df = df[df['hour_idx'] < 168]
    
    print("Grouping inflows...")
    inflow = df.groupby(['d_grid_500', 'hour_idx'])['num_total'].sum().reset_index()
    print("Grouping outflows...")
    outflow = df.groupby(['o_grid_500', 'hour_idx'])['num_total'].sum().reset_index()
    
    # Preallocate an array of zeros for all grids in this data
    all_grids = set(df['o_grid_500']).union(set(df['d_grid_500']))
    flows = {g: np.zeros((168, 2), dtype=np.float32) for g in all_grids}
    
    print("Populating arrays...")
    for row in tqdm(inflow.itertuples(), total=len(inflow), desc="Inflow"):
        flows[row.d_grid_500][row.hour_idx, 0] = row.num_total
        
    for row in tqdm(outflow.itertuples(), total=len(outflow), desc="Outflow"):
        flows[row.o_grid_500][row.hour_idx, 1] = row.num_total
        
    return flows, all_grids

print("Processing 2021...")
flows_2021, grids_2021 = process_year('data/2021_sgh_week.csv')
print("Processing 2024...")
flows_2024, grids_2024 = process_year('data/2024_sgh_week.csv')

default_template_cache = (
    'data/cache/dual_year_data_all_grids_correct.pkl'
    if os.path.exists('data/cache/dual_year_data_all_grids_correct.pkl')
    else 'data/cache/dual_year_data_a758cd48f656.pkl'
)
template_cache_file = os.environ.get('FULLGRID_TEMPLATE_CACHE', default_template_cache)
train_flow_cache_file = resolve_train_flow_cache()
output_cache_file = os.environ.get('FULLGRID_OUTPUT_CACHE', template_cache_file)

print(f"Loading template cache from {template_cache_file}...")
data = load_pickle(template_cache_file)

print(f"Loading training-flow cache from {train_flow_cache_file}...")
train_flow_data = load_pickle(train_flow_cache_file)

train_flow_grid_ids = train_flow_data.get('train_flow_grid_ids')
if train_flow_grid_ids is None:
    train_flow_grid_ids = sorted(
        set(train_flow_data.get('flows_2021', {}).keys()) |
        set(train_flow_data.get('flows_2024', {}).keys())
    )
print(f"Training-flow grids preserved from original cache: {len(train_flow_grid_ids)}")

# The set of target grids is the ones in the graph mapping
valid_grids = list(data['grid_id_to_idx'].keys())
print(f"Target valid grids: {len(valid_grids)}")

change_features = {}
for grid in tqdm(valid_grids, desc="Building change features"):
    flow21 = flows_2021.get(grid, np.zeros((168, 2), dtype=np.float32))
    flow24 = flows_2024.get(grid, np.zeros((168, 2), dtype=np.float32))
    
    in21 = np.log1p(flow21[:, 0])
    out21 = np.log1p(flow21[:, 1])
    in24 = np.log1p(flow24[:, 0])
    out24 = np.log1p(flow24[:, 1])
    
    comb = np.stack([in21, out21, in24, out24], axis=1) # (168, 4)
    change_features[grid] = comb

# update data
data['change_features'] = change_features
data['flows_2021'] = {g: flows_2021.get(g, np.zeros((168, 2), dtype=np.float32)) for g in valid_grids}
data['flows_2024'] = {g: flows_2024.get(g, np.zeros((168, 2), dtype=np.float32)) for g in valid_grids}
data['train_flow_grid_ids'] = train_flow_grid_ids
data['train_flow_cache_file'] = train_flow_cache_file
data['train_flow_label_file'] = train_flow_data.get('label_file_name')
data['train_flow_label_hash'] = train_flow_data.get('label_file_hash')
data['train_flow_total_samples'] = len(train_flow_data.get('labels', {}))

print(f"Saving to {output_cache_file}...")
with open(output_cache_file, 'wb') as f:
    pickle.dump(data, f)
print("Done!")

"""
Dual-year data processor for mobility pattern change classification
Processes both 2021 and 2024 data to capture temporal changes
"""
import pandas as pd
import numpy as np
import torch
import logging
from tqdm import tqdm
import config

logger = logging.getLogger(__name__)


class DualYearDataProcessor:
    """Process and compare mobility data from two years"""

    def __init__(self, year1=2021, year2=2024):
        """
        Initialize dual-year processor

        Args:
            year1: First year (baseline)
            year2: Second year (comparison)
        """
        self.year1 = year1
        self.year2 = year2
        self.data_path_1 = f'data/{year1}.csv'
        self.data_path_2 = f'data/{year2}.csv'

    def load_year_data(self, year, sampled_grid_ids, chunksize=500000):
        """
        Load OD flow data for a specific year

        Args:
            year: Year to load
            sampled_grid_ids: Set of grid IDs to filter
            chunksize: Chunk size for reading

        Returns:
            DataFrame with OD flow data
        """
        data_path = f'data/{year}.csv'
        logger.info(f"Loading OD flow data for {year}...")

        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=chunksize):
            # Filter for sampled grids
            chunk = chunk[
                chunk['o_grid_500'].isin(sampled_grid_ids) |
                chunk['d_grid_500'].isin(sampled_grid_ids)
            ]

            if len(chunk) > 0:
                # Convert date
                chunk['date_dt'] = pd.to_datetime(chunk['date_dt'], format='%Y%m%d')

                # Validate time
                chunk = chunk[chunk['time'].between(0, 23)]

                # Validate num_total
                chunk = chunk[chunk['num_total'] > 0]

                chunks.append(chunk)

        od_df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(od_df)} OD records for {year}")

        return od_df

    def filter_training_period(self, od_df, train_days=7):
        """
        Filter data to training period (first N days)

        Args:
            od_df: OD flow DataFrame
            train_days: Number of days to use

        Returns:
            Filtered DataFrame
        """
        # Get date range
        min_date = od_df['date_dt'].min()
        max_date = min_date + pd.Timedelta(days=train_days)

        # Filter
        filtered_df = od_df[
            (od_df['date_dt'] >= min_date) &
            (od_df['date_dt'] < max_date)
        ].copy()

        logger.info(f"Filtering data from {min_date} to {max_date}")
        logger.info(f"Training data: {len(filtered_df)} records covering {train_days * 24} hours")

        return filtered_df

    def normalize_flow(self, od_df):
        """
        Normalize flow values using z-score normalization

        Args:
            od_df: OD flow DataFrame

        Returns:
            Normalized DataFrame and normalization parameters
        """
        mean_flow = od_df['num_total'].mean()
        std_flow = od_df['num_total'].std()

        od_df['num_total_normalized'] = (od_df['num_total'] - mean_flow) / std_flow

        logger.info(f"Normalized flow data: mean={mean_flow:.2f}, std={std_flow:.2f}")

        return od_df, {'mean': mean_flow, 'std': std_flow}

    def build_temporal_features(self, od_df):
        """
        Build temporal features from OD data

        Args:
            od_df: OD flow DataFrame

        Returns:
            DataFrame with temporal features
        """
        logger.info("Building temporal features")

        # Extract temporal features
        od_df['hour'] = od_df['time']
        od_df['day_of_week'] = od_df['date_dt'].dt.dayofweek
        od_df['is_weekend'] = od_df['day_of_week'].isin([5, 6]).astype(int)

        return od_df

    def aggregate_grid_flows(self, od_df, grid_ids, use_raw=False):
        """
        Aggregate inflow and outflow for each grid over time

        Args:
            od_df: OD flow DataFrame
            grid_ids: List of grid IDs to aggregate
            use_raw: If True, use raw num_total instead of normalized values

        Returns:
            Dictionary mapping grid_id to temporal flow array (time_steps, 2)
        """
        logger.info("Aggregating grid flows")

        # Create time index (assuming 7 days * 24 hours = 168 time steps)
        time_steps = config.TRAIN_DAYS * 24

        grid_flows = {}

        # Choose which column to use
        flow_column = 'num_total' if use_raw else 'num_total_normalized'

        for grid_id in tqdm(grid_ids, desc="Processing grids"):
            # Initialize flow array
            flow_array = np.zeros((time_steps, 2))  # (time, [inflow, outflow])

            # Get inflow (this grid as destination)
            inflow = od_df[od_df['d_grid_500'] == grid_id].groupby('time')[flow_column].sum()

            # Get outflow (this grid as origin)
            outflow = od_df[od_df['o_grid_500'] == grid_id].groupby('time')[flow_column].sum()

            # Fill flow array
            for t in range(time_steps):
                if t in inflow.index:
                    flow_array[t, 0] = inflow[t]
                if t in outflow.index:
                    flow_array[t, 1] = outflow[t]

            grid_flows[grid_id] = flow_array

        return grid_flows

    def compute_temporal_change_features(self, flows_2021_raw, flows_2024_raw,
                                        flows_2021_norm, flows_2024_norm):
        """
        Compute temporal change features between two years

        CRITICAL FIX: Compute relative change on RAW values, then normalize the result.
        This avoids the mathematical error of computing ratios on Z-scores.

        Args:
            flows_2021_raw: Raw grid flows for 2021 {grid_id: array(168, 2)}
            flows_2024_raw: Raw grid flows for 2024 {grid_id: array(168, 2)}
            flows_2021_norm: Normalized grid flows for 2021 {grid_id: array(168, 2)}
            flows_2024_norm: Normalized grid flows for 2024 {grid_id: array(168, 2)}

        Returns:
            Dictionary with change features for each grid
        """
        logger.info("Computing temporal change features (using raw values for relative change)")

        change_features = {}
        all_rel_changes = []  # Collect all relative changes for global normalization

        # First pass: compute relative changes and collect for global statistics
        temp_rel_changes = {}
        for grid_id in flows_2021_raw.keys():
            if grid_id not in flows_2024_raw:
                logger.warning(f"Grid {grid_id} not found in 2024 data, skipping")
                continue

            # Get raw flows for computing relative change
            flow_2021_raw = flows_2021_raw[grid_id]  # (168, 2)
            flow_2024_raw = flows_2024_raw[grid_id]  # (168, 2)

            # Relative change (on RAW values to avoid Z-score division issues)
            epsilon = 1e-6
            rel_change_raw = (flow_2024_raw - flow_2021_raw) / (flow_2021_raw + epsilon)  # (168, 2)

            # Clip extreme values BEFORE normalization
            rel_change_raw = np.clip(rel_change_raw, -10, 10)

            temp_rel_changes[grid_id] = rel_change_raw
            all_rel_changes.append(rel_change_raw)

        # Compute global statistics for relative change normalization
        all_rel_changes = np.concatenate(all_rel_changes, axis=0)
        rel_change_mean = all_rel_changes.mean()
        rel_change_std = all_rel_changes.std() + 1e-6

        logger.info(f"Relative change statistics - Mean: {rel_change_mean:.4f}, Std: {rel_change_std:.4f}")

        # Second pass: normalize and create final features
        for grid_id in flows_2021_raw.keys():
            if grid_id not in flows_2024_raw:
                continue

            # Get normalized flows for direct features
            flow_2021_norm = flows_2021_norm[grid_id]  # (168, 2)
            flow_2024_norm = flows_2024_norm[grid_id]  # (168, 2)

            # 1. Absolute difference (on normalized values)
            diff_norm = flow_2024_norm - flow_2021_norm  # (168, 2)

            # 2. Normalize relative change using GLOBAL statistics
            rel_change_norm = (temp_rel_changes[grid_id] - rel_change_mean) / rel_change_std

            # 3. Total flow volumes (normalized)
            total_2021_norm = flow_2021_norm.sum(axis=1, keepdims=True)  # (168, 1)
            total_2024_norm = flow_2024_norm.sum(axis=1, keepdims=True)  # (168, 1)

            # 4. Concatenate: [2021_flow_norm, 2024_flow_norm, diff_norm, rel_change_norm, total_2021_norm, total_2024_norm]
            # Shape: (168, 10) = (168, 2+2+2+2+1+1)
            combined = np.concatenate([
                flow_2021_norm,      # (168, 2) - normalized
                flow_2024_norm,      # (168, 2) - normalized
                diff_norm,           # (168, 2) - normalized difference
                rel_change_norm,     # (168, 2) - normalized relative change (computed from raw)
                total_2021_norm,     # (168, 1) - normalized total
                total_2024_norm      # (168, 1) - normalized total
            ], axis=1)

            change_features[grid_id] = combined

        logger.info(f"Computed change features for {len(change_features)} grids")

        return change_features

    def prepare_dual_year_data(self, sampled_grid_ids, valid_grid_ids):
        """
        Prepare data from both years for training

        Args:
            sampled_grid_ids: Set of grid IDs to process
            valid_grid_ids: Set of all valid grid IDs

        Returns:
            Dictionary with processed data from both years
        """
        logger.info("=" * 80)
        logger.info("Preparing Dual-Year Data (2021 vs 2024)")
        logger.info("=" * 80)

        # Load 2021 data
        od_2021 = self.load_year_data(self.year1, sampled_grid_ids)
        od_2021 = self.filter_training_period(od_2021, config.TRAIN_DAYS)
        od_2021 = od_2021[od_2021['o_grid_500'].isin(valid_grid_ids) &
                          od_2021['d_grid_500'].isin(valid_grid_ids)]
        od_2021, norm_params_2021 = self.normalize_flow(od_2021)
        od_2021 = self.build_temporal_features(od_2021)

        # Load 2024 data
        od_2024 = self.load_year_data(self.year2, sampled_grid_ids)
        od_2024 = self.filter_training_period(od_2024, config.TRAIN_DAYS)
        od_2024 = od_2024[od_2024['o_grid_500'].isin(valid_grid_ids) &
                          od_2024['d_grid_500'].isin(valid_grid_ids)]
        od_2024, norm_params_2024 = self.normalize_flow(od_2024)
        od_2024 = self.build_temporal_features(od_2024)

        # Aggregate flows for each year (both raw and normalized)
        labeled_grid_ids = list(sampled_grid_ids)

        # Raw flows for computing relative change
        flows_2021_raw = self.aggregate_grid_flows(od_2021, labeled_grid_ids, use_raw=True)
        flows_2024_raw = self.aggregate_grid_flows(od_2024, labeled_grid_ids, use_raw=True)

        # Normalized flows for model input
        flows_2021_norm = self.aggregate_grid_flows(od_2021, labeled_grid_ids, use_raw=False)
        flows_2024_norm = self.aggregate_grid_flows(od_2024, labeled_grid_ids, use_raw=False)

        # Compute change features (using both raw and normalized)
        change_features = self.compute_temporal_change_features(
            flows_2021_raw, flows_2024_raw,
            flows_2021_norm, flows_2024_norm
        )

        logger.info(f"\nDual-year data preparation completed:")
        logger.info(f"  - 2021 OD records: {len(od_2021)}")
        logger.info(f"  - 2024 OD records: {len(od_2024)}")
        logger.info(f"  - Grids with change features: {len(change_features)}")

        return {
            'od_2021': od_2021,
            'od_2024': od_2024,
            'flows_2021': flows_2021_norm,  # Return normalized for consistency
            'flows_2024': flows_2024_norm,
            'change_features': change_features,
            'norm_params_2021': norm_params_2021,
            'norm_params_2024': norm_params_2024
        }


def prepare_dual_year_experiment_data(label_path, samples_per_class=None, use_cache=True, cache_dir='data/cache'):
    """
    Prepare complete dataset for dual-year experiment with caching support

    Args:
        label_path: Path to label file
        samples_per_class: Number of samples per class (None = use all samples)
        use_cache: Whether to use cached data if available
        cache_dir: Directory to store cached data

    Returns:
        Dictionary with all prepared data including class_weights
    """
    import os
    import pickle
    import hashlib
    from src.preprocessing.data_processor import GridMetadataProcessor
    from src.preprocessing.graph_builder import SpatialGraphBuilder

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache key based on label_path and samples_per_class
    label_basename = os.path.basename(label_path)
    cache_key = f"{label_basename}_samples_{samples_per_class}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"dual_year_data_{cache_hash}.pkl")
    cache_info_file = os.path.join(cache_dir, f"dual_year_data_{cache_hash}_info.txt")

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        logger.info("=" * 80)
        logger.info("Loading Preprocessed Data from Cache")
        logger.info("=" * 80)
        logger.info(f"Cache file: {cache_file}")

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Display cache info
            if os.path.exists(cache_info_file):
                with open(cache_info_file, 'r') as f:
                    logger.info(f.read())

            logger.info("✓ Successfully loaded cached data!")
            logger.info(f"  - Total grids: {len(data['labels'])}")
            logger.info(f"  - Graph edges: {data['edge_index'].shape[1]}")
            logger.info("=" * 80)
            return data

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            logger.info("Proceeding with fresh data preparation...")

    logger.info("=" * 80)
    logger.info("Dual-Year Experiment Data Preparation")
    logger.info("=" * 80)
    logger.info(f"Label file: {label_path}")
    logger.info(f"Samples per class: {samples_per_class if samples_per_class else 'ALL'}")
    logger.info("")

    # Load metadata
    logger.info("Loading grid metadata...")
    metadata_processor = GridMetadataProcessor()
    metadata_df = metadata_processor.load_and_validate()
    valid_grid_ids = metadata_processor.get_valid_grid_ids(metadata_df)

    # Load labels
    logger.info(f"Loading labels from {label_path}...")
    label_df = pd.read_csv(label_path)
    label_df = label_df[label_df['grid_id'].isin(valid_grid_ids)]
    label_df = label_df[label_df['label'].between(1, 9)]

    # Sample or use all labels
    if samples_per_class is not None:
        logger.info(f"Sampling {samples_per_class} samples per class...")
        sampled_dfs = []
        for label in range(1, config.NUM_CLASSES + 1):
            class_df = label_df[label_df['label'] == label]
            if len(class_df) >= samples_per_class:
                sampled = class_df.sample(n=samples_per_class, random_state=config.RANDOM_SEED)
            else:
                logger.warning(f"Class {label} has only {len(class_df)} samples, using all")
                sampled = class_df

            sampled_dfs.append(sampled)

        label_df = pd.concat(sampled_dfs, ignore_index=True)
    else:
        logger.info("Using all available labels (no sampling)")

    label_df['label_idx'] = label_df['label'] - 1

    labels = dict(zip(label_df['grid_id'], label_df['label_idx']))
    sampled_grid_ids = set(labels.keys())

    # Compute class weights for imbalanced data
    class_counts = label_df['label_idx'].value_counts().sort_index()
    total_samples = len(label_df)
    class_weights = torch.FloatTensor([
        total_samples / (config.NUM_CLASSES * class_counts[i])
        for i in range(config.NUM_CLASSES)
    ])

    logger.info(f"Total samples: {len(labels)} grids across {config.NUM_CLASSES} classes")
    logger.info("Class distribution:")
    for i in range(config.NUM_CLASSES):
        count = class_counts.get(i, 0)
        weight = class_weights[i].item()
        logger.info(f"  Class {i+1}: {count} samples (weight: {weight:.4f})")

    # Prepare dual-year data
    dual_year_processor = DualYearDataProcessor(year1=2021, year2=2024)
    dual_year_data = dual_year_processor.prepare_dual_year_data(sampled_grid_ids, valid_grid_ids)

    # Build spatial graph (using 2024 data for current spatial structure)
    logger.info("Building spatial graph...")
    metadata_sampled = metadata_df[metadata_df['grid_id'].isin(sampled_grid_ids)].copy()
    graph_builder = SpatialGraphBuilder(metadata_sampled, k_neighbors=8)
    edge_index, edge_weights = graph_builder.build_hybrid_graph(dual_year_data['od_2024'])

    # Create grid_id to index mapping
    # CRITICAL FIX: Use the mapping from graph_builder to ensure consistency with edge_index
    grid_id_to_idx = graph_builder.grid_id_to_idx

    logger.info(f"\nComplete data preparation:")
    logger.info(f"  - Total grids: {len(labels)}")
    logger.info(f"  - Graph edges: {edge_index.shape[1]}")
    logger.info(f"  - Feature dimension: {list(dual_year_data['change_features'].values())[0].shape}")

    data = {
        'metadata_df': metadata_sampled,
        'labels': labels,
        'change_features': dual_year_data['change_features'],
        'flows_2021': dual_year_data['flows_2021'],
        'flows_2024': dual_year_data['flows_2024'],
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'grid_id_to_idx': grid_id_to_idx,
        'norm_params_2021': dual_year_data['norm_params_2021'],
        'norm_params_2024': dual_year_data['norm_params_2024'],
        'label_df': label_df,
        'class_weights': class_weights
    }

    # Save to cache
    if use_cache:
        logger.info(f"\nSaving preprocessed data to cache...")
        logger.info(f"Cache file: {cache_file}")

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save cache info
            with open(cache_info_file, 'w') as f:
                f.write(f"Cache Information:\n")
                f.write(f"  Label file: {label_path}\n")
                f.write(f"  Samples per class: {samples_per_class if samples_per_class else 'ALL'}\n")
                f.write(f"  Total grids: {len(labels)}\n")
                f.write(f"  Graph edges: {edge_index.shape[1]}\n")
                f.write(f"  Feature shape: {list(dual_year_data['change_features'].values())[0].shape}\n")
                f.write(f"  Class distribution:\n")
                for i in range(config.NUM_CLASSES):
                    count = class_counts.get(i, 0)
                    weight = class_weights[i].item()
                    f.write(f"    Class {i+1}: {count} samples (weight: {weight:.4f})\n")

            logger.info("✓ Cache saved successfully!")
            logger.info(f"  Next run will load from cache in ~1 second")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            logger.info("Continuing without cache...")

    return data


if __name__ == "__main__":
    # Test dual-year data processing
    logging.basicConfig(level=logging.INFO)

    data = prepare_dual_year_experiment_data(
        label_path='data/labels_1w.csv',
        samples_per_class=10  # Small sample for testing
    )

    print("\nData preparation successful!")
    print(f"Change features shape: {list(data['change_features'].values())[0].shape}")
    print(f"Number of grids: {len(data['labels'])}")

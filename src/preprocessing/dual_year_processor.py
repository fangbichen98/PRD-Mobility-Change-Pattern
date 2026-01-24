"""
Dual-year data processor for mobility pattern change classification
Processes both 2021 and 2024 data to capture temporal changes
"""
import pandas as pd
import numpy as np
import torch
import logging
import os
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
        NEW: Aggregates to daily snapshots (7 days) instead of hourly (168 hours)

        Args:
            od_df: OD flow DataFrame
            grid_ids: List of grid IDs to aggregate
            use_raw: If True, use raw num_total instead of normalized values

        Returns:
            Dictionary mapping grid_id to temporal flow array (7, 2) - 7 days, [inflow, outflow]
        """
        logger.info("Aggregating grid flows to daily snapshots")

        # Number of days
        num_days = config.TRAIN_DAYS  # 7 days

        grid_flows = {}

        # Choose which column to use
        flow_column = 'num_total' if use_raw else 'num_total_normalized'

        for grid_id in tqdm(grid_ids, desc="Processing grids"):
            # Initialize flow array for daily aggregation
            daily_flow_array = np.zeros((num_days, 2))  # (7, [inflow, outflow])

            # Get all inflow records (this grid as destination)
            inflow_df = od_df[od_df['d_grid_500'] == grid_id].copy()

            # Get all outflow records (this grid as origin)
            outflow_df = od_df[od_df['o_grid_500'] == grid_id].copy()

            # Aggregate by day
            if len(inflow_df) > 0:
                # Group by date and sum across all hours
                inflow_df['day_idx'] = (inflow_df['date_dt'] - inflow_df['date_dt'].min()).dt.days
                daily_inflow = inflow_df.groupby('day_idx')[flow_column].sum()

                for day_idx, flow_val in daily_inflow.items():
                    if 0 <= day_idx < num_days:
                        daily_flow_array[day_idx, 0] = flow_val

            if len(outflow_df) > 0:
                # Group by date and sum across all hours
                outflow_df['day_idx'] = (outflow_df['date_dt'] - outflow_df['date_dt'].min()).dt.days
                daily_outflow = outflow_df.groupby('day_idx')[flow_column].sum()

                for day_idx, flow_val in daily_outflow.items():
                    if 0 <= day_idx < num_days:
                        daily_flow_array[day_idx, 1] = flow_val

            grid_flows[grid_id] = daily_flow_array

        logger.info(f"Aggregated flows to {num_days} daily snapshots per grid")
        return grid_flows

    def log_transform_features(self, total, net_flow):
        """
        Apply log transformation to preserve magnitude information

        Args:
            total: Total flow (inflow + outflow)
            net_flow: Net flow (outflow - inflow)

        Returns:
            total_log: Log-transformed total flow
            net_flow_log: Signed log-transformed net flow
        """
        # Total: direct log transformation (always positive)
        total_log = np.log1p(total)  # log(1 + x)

        # Net flow: preserve sign with log transformation
        net_flow_log = np.sign(net_flow) * np.log1p(np.abs(net_flow))

        return total_log, net_flow_log

    def compute_temporal_change_features(self, flows_2021_raw, flows_2024_raw,
                                        flows_2021_norm, flows_2024_norm,
                                        ellipse_data=None):
        """
        Compute temporal change features between two years
        NEW: Uses log transformation to preserve magnitude information
        NEW: Optionally integrates ellipse features for direction modeling

        Args:
            flows_2021_raw: Raw grid flows for 2021 {grid_id: array(7, 2)}
            flows_2024_raw: Raw grid flows for 2024 {grid_id: array(7, 2)}
            flows_2021_norm: Not used (kept for compatibility)
            flows_2024_norm: Not used (kept for compatibility)
            ellipse_data: Ellipse data dictionary (from JSON), optional

        Returns:
            Dictionary with change features for each grid
            Shape: (7, 4) without ellipse or (7, 8) with ellipse
            Without: [2021_total_log, 2024_total_log, 2021_net_flow_log, 2024_net_flow_log]
            With: [..., eccentricity_2021, log_area_2021, eccentricity_2024, log_area_2024]
        """
        logger.info("Computing temporal change features with log transformation")

        # If ellipse data provided, load feature extraction function
        if ellipse_data is not None:
            from .ellipse_features import compute_ellipse_features_dual_year
            logger.info("Using ellipse features for direction modeling")

        change_features = {}
        grids_with_ellipse = 0
        grids_without_ellipse = 0

        for grid_id in flows_2021_raw.keys():
            if grid_id not in flows_2024_raw:
                logger.warning(f"Grid {grid_id} not found in 2024 data, skipping")
                continue

            # Get raw flows (7, 2) - [inflow, outflow]
            flow_2021_raw = flows_2021_raw[grid_id]  # (7, 2)
            flow_2024_raw = flows_2024_raw[grid_id]  # (7, 2)

            # Compute total and net_flow for each year
            # Total = inflow + outflow (flow intensity)
            total_2021 = flow_2021_raw[:, 0] + flow_2021_raw[:, 1]  # (7,)
            total_2024 = flow_2024_raw[:, 0] + flow_2024_raw[:, 1]  # (7,)

            # Net flow = outflow - inflow (spatial direction: positive=diffusion, negative=aggregation)
            net_flow_2021 = flow_2021_raw[:, 1] - flow_2021_raw[:, 0]  # (7,)
            net_flow_2024 = flow_2024_raw[:, 1] - flow_2024_raw[:, 0]  # (7,)

            # Apply log transformation to preserve magnitude
            total_2021_log, net_flow_2021_log = self.log_transform_features(total_2021, net_flow_2021)
            total_2024_log, net_flow_2024_log = self.log_transform_features(total_2024, net_flow_2024)

            # Extract ellipse features if available
            if ellipse_data is not None:
                ellipse_feats = compute_ellipse_features_dual_year(grid_id, ellipse_data)

                if ellipse_feats is not None:
                    grids_with_ellipse += 1
                    # Ellipse features broadcast to 7 days: (4,) -> (7, 4)
                    ellipse_array = np.array([
                        ellipse_feats['eccentricity_2021'],
                        ellipse_feats['log_area_2021'],
                        ellipse_feats['eccentricity_2024'],
                        ellipse_feats['log_area_2024'],
                    ])  # (4,)
                    ellipse_array_7d = np.tile(ellipse_array, (7, 1))  # (7, 4)

                    # Concatenate: (7, 4) + (7, 4) = (7, 8)
                    combined = np.concatenate([
                        np.stack([total_2021_log, total_2024_log,
                                 net_flow_2021_log, net_flow_2024_log], axis=1),  # (7, 4)
                        ellipse_array_7d  # (7, 4)
                    ], axis=1)
                else:
                    grids_without_ellipse += 1
                    # If this grid has no ellipse data, use only flow features
                    combined = np.stack([
                        total_2021_log, total_2024_log,
                        net_flow_2021_log, net_flow_2024_log
                    ], axis=1)  # (7, 4)
            else:
                # No ellipse data provided, use only flow features
                combined = np.stack([
                    total_2021_log, total_2024_log,
                    net_flow_2021_log, net_flow_2024_log
                ], axis=1)  # (7, 4)

            change_features[grid_id] = combined

        logger.info(f"Computed change features for {len(change_features)} grids")
        if ellipse_data is not None:
            logger.info(f"  - Grids with ellipse features: {grids_with_ellipse}")
            logger.info(f"  - Grids without ellipse features: {grids_without_ellipse}")
            logger.info(f"Feature shape per grid: (7, 8) = [total_2021, total_2024, net_2021, net_2024, ecc_2021, area_2021, ecc_2024, area_2024]")
        else:
            logger.info(f"Feature shape per grid: (7, 4) = [2021_total_log, 2024_total_log, 2021_net_flow_log, 2024_net_flow_log]")

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

        # Load ellipse data if available
        ellipse_data = None
        ellipse_path = 'data/ellipses.json'
        if os.path.exists(ellipse_path):
            logger.info(f"Loading ellipse data from {ellipse_path}")
            from .ellipse_features import load_ellipse_data
            ellipse_data = load_ellipse_data(ellipse_path)
            logger.info(f"✓ Loaded ellipse data for years: {list(ellipse_data['years'].keys())}")
            logger.info(f"  - 2021: {len(ellipse_data['years']['2021'])} grids")
            logger.info(f"  - 2024: {len(ellipse_data['years']['2024'])} grids")
        else:
            logger.warning(f"Ellipse data not found at {ellipse_path}, using flow features only")

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

        # Compute change features (using both raw and normalized, plus ellipse data)
        change_features = self.compute_temporal_change_features(
            flows_2021_raw, flows_2024_raw,
            flows_2021_norm, flows_2024_norm,
            ellipse_data=ellipse_data
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
    NEW: Includes dynamic graph snapshots for both years

    Args:
        label_path: Path to label file
        samples_per_class: Number of samples per class (None = use all samples)
        use_cache: Whether to use cached data if available
        cache_dir: Directory to store cached data

    Returns:
        Dictionary with all prepared data including class_weights and dynamic graphs
    """
    import os
    import pickle
    import hashlib
    from src.preprocessing.data_processor import GridMetadataProcessor
    from src.preprocessing.graph_builder import SpatialGraphBuilder, DynamicGraphBuilder

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache key with file content hash and modification times
    # Strategy: Use content hash for label file (small), mtime for OD data files (large)

    # 1. Label file content hash (small file, use content hash for 100% accuracy)
    with open(label_path, 'rb') as f:
        label_content_hash = hashlib.md5(f.read()).hexdigest()[:8]

    # 2. OD data file modification times (large files, use mtime for efficiency)
    data_2021_path = 'data/2021.csv'
    data_2024_path = 'data/2024.csv'

    if os.path.exists(data_2021_path):
        data_2021_mtime = int(os.path.getmtime(data_2021_path))
    else:
        data_2021_mtime = 0

    if os.path.exists(data_2024_path):
        data_2024_mtime = int(os.path.getmtime(data_2024_path))
    else:
        data_2024_mtime = 0

    # 3. Generate cache key
    label_basename = os.path.basename(label_path)
    cache_key = f"{label_basename}_{label_content_hash}_samples_{samples_per_class}_data_{data_2021_mtime}_{data_2024_mtime}_v3"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]  # Use 12-char hash for better uniqueness

    cache_file = os.path.join(cache_dir, f"dual_year_data_{cache_hash}.pkl")
    cache_info_file = os.path.join(cache_dir, f"dual_year_data_{cache_hash}_info.txt")

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        logger.info("=" * 80)
        logger.info("Loading Preprocessed Data from Cache")
        logger.info("=" * 80)
        logger.info(f"Cache file: {cache_file}")

        try:
            import time
            load_start = time.time()

            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            load_time = time.time() - load_start

            # Validate cache integrity
            required_keys = ['labels', 'change_features', 'graphs_2021', 'graphs_2024', 'class_weights']
            missing_keys = [k for k in required_keys if k not in data]
            if missing_keys:
                logger.warning(f"Cache is missing keys: {missing_keys}")
                logger.info("Regenerating cache...")
                # Continue to regenerate cache
            else:
                # Display cache info
                if os.path.exists(cache_info_file):
                    with open(cache_info_file, 'r') as f:
                        logger.info(f.read())

                logger.info("✓ Successfully loaded cached data!")
                logger.info(f"  - Total grids: {len(data['labels'])}")
                logger.info(f"  - Graphs 2021: {len(data['graphs_2021'])} daily snapshots")
                logger.info(f"  - Graphs 2024: {len(data['graphs_2024'])} daily snapshots")
                logger.info(f"  - Load time: {load_time:.2f} seconds")
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

    # Build spatial graph builder
    logger.info("Building spatial graphs...")
    metadata_sampled = metadata_df[metadata_df['grid_id'].isin(sampled_grid_ids)].copy()
    graph_builder = SpatialGraphBuilder(metadata_sampled, k_neighbors=8)

    # Build dynamic graphs for both years (7 daily snapshots each)
    dynamic_graph_builder = DynamicGraphBuilder(graph_builder, time_window=24)
    graphs_2021 = dynamic_graph_builder.build_daily_graphs(dual_year_data['od_2021'], num_days=7)
    graphs_2024 = dynamic_graph_builder.build_daily_graphs(dual_year_data['od_2024'], num_days=7)

    # Also build a static graph for compatibility (using 2024 data)
    edge_index, edge_weights = graph_builder.build_hybrid_graph(dual_year_data['od_2024'])

    # Create grid_id to index mapping
    grid_id_to_idx = graph_builder.grid_id_to_idx

    logger.info(f"\nComplete data preparation:")
    logger.info(f"  - Total grids: {len(labels)}")
    logger.info(f"  - Static graph edges: {edge_index.shape[1]}")
    logger.info(f"  - Dynamic graphs 2021: {len(graphs_2021)} daily snapshots")
    logger.info(f"  - Dynamic graphs 2024: {len(graphs_2024)} daily snapshots")
    logger.info(f"  - Feature dimension: {list(dual_year_data['change_features'].values())[0].shape}")

    data = {
        'metadata_df': metadata_sampled,
        'labels': labels,
        'change_features': dual_year_data['change_features'],
        'flows_2021': dual_year_data['flows_2021'],
        'flows_2024': dual_year_data['flows_2024'],
        'edge_index': edge_index,  # Static graph for compatibility
        'edge_weights': edge_weights,
        'graphs_2021': graphs_2021,  # NEW: Dynamic graphs for 2021
        'graphs_2024': graphs_2024,  # NEW: Dynamic graphs for 2024
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
                f.write(f"  Label file hash: {label_content_hash}\n")
                f.write(f"  Data 2021 mtime: {data_2021_mtime}\n")
                f.write(f"  Data 2024 mtime: {data_2024_mtime}\n")
                f.write(f"  Samples per class: {samples_per_class if samples_per_class else 'ALL'}\n")
                f.write(f"  Total grids: {len(labels)}\n")
                f.write(f"  Static graph edges: {edge_index.shape[1]}\n")
                f.write(f"  Dynamic graphs 2021: {len(graphs_2021)} daily snapshots\n")
                f.write(f"  Dynamic graphs 2024: {len(graphs_2024)} daily snapshots\n")
                f.write(f"  Feature shape: {list(dual_year_data['change_features'].values())[0].shape}\n")
                f.write(f"  Class distribution:\n")
                for i in range(config.NUM_CLASSES):
                    count = class_counts.get(i, 0)
                    weight = class_weights[i].item()
                    f.write(f"    Class {i+1}: {count} samples (weight: {weight:.4f})\n")

            # Get cache file size
            cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)

            logger.info("✓ Cache saved successfully!")
            logger.info(f"  Cache size: {cache_size_mb:.2f} MB")
            logger.info(f"  Cache location: {os.path.abspath(cache_file)}")
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

"""
Dual-year data processor for mobility pattern change classification
Processes both 2021 and 2024 data to capture temporal changes
"""
import pandas as pd
import numpy as np
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

    def aggregate_grid_flows(self, od_df, grid_ids):
        """
        Aggregate inflow and outflow for each grid over time

        Args:
            od_df: OD flow DataFrame
            grid_ids: List of grid IDs to aggregate

        Returns:
            Dictionary mapping grid_id to temporal flow array (time_steps, 2)
        """
        logger.info("Aggregating grid flows")

        # Create time index (assuming 7 days * 24 hours = 168 time steps)
        time_steps = config.TRAIN_DAYS * 24

        grid_flows = {}

        for grid_id in tqdm(grid_ids, desc="Processing grids"):
            # Initialize flow array
            flow_array = np.zeros((time_steps, 2))  # (time, [inflow, outflow])

            # Get inflow (this grid as destination)
            inflow = od_df[od_df['d_grid_500'] == grid_id].groupby('time')['num_total_normalized'].sum()

            # Get outflow (this grid as origin)
            outflow = od_df[od_df['o_grid_500'] == grid_id].groupby('time')['num_total_normalized'].sum()

            # Fill flow array
            for t in range(time_steps):
                if t in inflow.index:
                    flow_array[t, 0] = inflow[t]
                if t in outflow.index:
                    flow_array[t, 1] = outflow[t]

            grid_flows[grid_id] = flow_array

        return grid_flows

    def compute_temporal_change_features(self, flows_2021, flows_2024):
        """
        Compute temporal change features between two years

        Args:
            flows_2021: Grid flows for 2021 {grid_id: array(168, 2)}
            flows_2024: Grid flows for 2024 {grid_id: array(168, 2)}

        Returns:
            Dictionary with change features for each grid
        """
        logger.info("Computing temporal change features")

        change_features = {}

        for grid_id in flows_2021.keys():
            if grid_id not in flows_2024:
                logger.warning(f"Grid {grid_id} not found in 2024 data, skipping")
                continue

            flow_2021 = flows_2021[grid_id]  # (168, 2)
            flow_2024 = flows_2024[grid_id]  # (168, 2)

            # Compute change features
            # 1. Absolute difference
            diff = flow_2024 - flow_2021  # (168, 2)

            # 2. Relative change (avoid division by zero)
            epsilon = 1e-6
            rel_change = diff / (np.abs(flow_2021) + epsilon)  # (168, 2)

            # 3. Total flow volumes
            total_2021 = flow_2021.sum(axis=1, keepdims=True)  # (168, 1)
            total_2024 = flow_2024.sum(axis=1, keepdims=True)  # (168, 1)

            # 4. Concatenate: [2021_flow, 2024_flow, diff, rel_change, total_2021, total_2024]
            # Shape: (168, 10) = (168, 2+2+2+2+1+1)
            combined = np.concatenate([
                flow_2021,      # (168, 2)
                flow_2024,      # (168, 2)
                diff,           # (168, 2)
                rel_change,     # (168, 2)
                total_2021,     # (168, 1)
                total_2024      # (168, 1)
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

        # Aggregate flows for each year
        labeled_grid_ids = list(sampled_grid_ids)
        flows_2021 = self.aggregate_grid_flows(od_2021, labeled_grid_ids)
        flows_2024 = self.aggregate_grid_flows(od_2024, labeled_grid_ids)

        # Compute change features
        change_features = self.compute_temporal_change_features(flows_2021, flows_2024)

        logger.info(f"\nDual-year data preparation completed:")
        logger.info(f"  - 2021 OD records: {len(od_2021)}")
        logger.info(f"  - 2024 OD records: {len(od_2024)}")
        logger.info(f"  - Grids with change features: {len(change_features)}")

        return {
            'od_2021': od_2021,
            'od_2024': od_2024,
            'flows_2021': flows_2021,
            'flows_2024': flows_2024,
            'change_features': change_features,
            'norm_params_2021': norm_params_2021,
            'norm_params_2024': norm_params_2024
        }


def prepare_dual_year_experiment_data(label_path, samples_per_class=100):
    """
    Prepare complete dataset for dual-year experiment

    Args:
        label_path: Path to label file
        samples_per_class: Number of samples per class

    Returns:
        Dictionary with all prepared data
    """
    from src.preprocessing.data_processor import GridMetadataProcessor
    from src.preprocessing.graph_builder import SpatialGraphBuilder

    logger.info("=" * 80)
    logger.info("Dual-Year Experiment Data Preparation")
    logger.info("=" * 80)

    # Load metadata
    logger.info("Loading grid metadata...")
    metadata_processor = GridMetadataProcessor()
    metadata_df = metadata_processor.load_and_validate()
    valid_grid_ids = metadata_processor.get_valid_grid_ids(metadata_df)

    # Load and sample labels
    logger.info(f"Loading labels from {label_path}...")
    label_df = pd.read_csv(label_path)
    label_df = label_df[label_df['grid_id'].isin(valid_grid_ids)]
    label_df = label_df[label_df['label'].between(1, 9)]

    # Sample balanced labels
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
    label_df['label_idx'] = label_df['label'] - 1

    labels = dict(zip(label_df['grid_id'], label_df['label_idx']))
    sampled_grid_ids = set(labels.keys())

    logger.info(f"Sampled {len(labels)} grids across {config.NUM_CLASSES} classes")

    # Prepare dual-year data
    dual_year_processor = DualYearDataProcessor(year1=2021, year2=2024)
    dual_year_data = dual_year_processor.prepare_dual_year_data(sampled_grid_ids, valid_grid_ids)

    # Build spatial graph (using 2024 data for current spatial structure)
    logger.info("Building spatial graph...")
    metadata_sampled = metadata_df[metadata_df['grid_id'].isin(sampled_grid_ids)].copy()
    graph_builder = SpatialGraphBuilder(metadata_sampled, k_neighbors=8)
    edge_index, edge_weights = graph_builder.build_hybrid_graph(dual_year_data['od_2024'])

    # Create grid_id to index mapping
    labeled_grid_ids = list(labels.keys())
    grid_id_to_idx = {gid: idx for idx, gid in enumerate(labeled_grid_ids)}

    logger.info(f"\nComplete data preparation:")
    logger.info(f"  - Total grids: {len(labels)}")
    logger.info(f"  - Graph edges: {edge_index.shape[1]}")
    logger.info(f"  - Feature dimension: {list(dual_year_data['change_features'].values())[0].shape}")

    return {
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
        'label_df': label_df
    }


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

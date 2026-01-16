"""
Data preprocessing utilities for mobility analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import logging
from tqdm import tqdm
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ODFlowProcessor:
    """Process Origin-Destination flow data"""

    def __init__(self, year: int):
        self.year = year
        self.data_path = config.OD_2021_PATH if year == 2021 else config.OD_2024_PATH

    def load_and_preprocess(self, nrows: int = None) -> pd.DataFrame:
        """
        Load and preprocess OD flow data

        Args:
            nrows: Number of rows to load (None for all)

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Loading OD flow data from {self.data_path}")

        # Load data in chunks due to large file size
        chunks = []
        chunksize = 100000

        for chunk in tqdm(pd.read_csv(self.data_path, chunksize=chunksize, nrows=nrows),
                         desc=f"Loading {self.year} data"):
            # Convert date_dt to datetime
            chunk['date_dt'] = pd.to_datetime(chunk['date_dt'], format='%Y%m%d')

            # Validate time field (0-23)
            if not chunk['time'].between(0, 23).all():
                logger.warning(f"Invalid time values found in {self.year} data")
                chunk = chunk[chunk['time'].between(0, 23)]

            # Validate num_total (should be positive)
            chunk = chunk[chunk['num_total'] > 0]

            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df)} records from {self.year}")

        return df

    def filter_training_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to first 7 days (168 hours) for training

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        # Get the earliest date
        min_date = df['date_dt'].min()
        max_train_date = min_date + timedelta(days=config.TRAIN_DAYS)

        logger.info(f"Filtering data from {min_date} to {max_train_date}")
        df_train = df[df['date_dt'] < max_train_date].copy()

        logger.info(f"Training data: {len(df_train)} records covering {config.TRAIN_HOURS} hours")
        return df_train

    def validate_grid_ids(self, df: pd.DataFrame, valid_grid_ids: set) -> pd.DataFrame:
        """
        Validate that grid IDs exist in metadata

        Args:
            df: Input DataFrame
            valid_grid_ids: Set of valid grid IDs from metadata

        Returns:
            Validated DataFrame
        """
        initial_count = len(df)

        # Filter records with valid grid IDs
        df = df[df['o_grid_500'].isin(valid_grid_ids) &
                df['d_grid_500'].isin(valid_grid_ids)].copy()

        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} records with invalid grid IDs")

        return df

    def normalize_flow(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize num_total using Z-score standardization

        Args:
            df: Input DataFrame

        Returns:
            Normalized DataFrame and normalization parameters
        """
        mean_val = df['num_total'].mean()
        std_val = df['num_total'].std()

        df['num_total_normalized'] = (df['num_total'] - mean_val) / std_val

        norm_params = {
            'mean': mean_val,
            'std': std_val
        }

        logger.info(f"Normalized flow data: mean={mean_val:.2f}, std={std_val:.2f}")
        return df, norm_params


class GridMetadataProcessor:
    """Process grid metadata"""

    def __init__(self):
        self.metadata_path = config.GRID_METADATA_PATH

    def load_and_validate(self) -> pd.DataFrame:
        """
        Load and validate grid metadata

        Returns:
            Validated metadata DataFrame
        """
        logger.info(f"Loading grid metadata from {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)

        # Check required fields
        required_fields = ['grid_id', 'lon', 'lat']
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate coordinates
        lon_valid = df['lon'].between(*config.LON_RANGE)
        lat_valid = df['lat'].between(*config.LAT_RANGE)

        invalid_coords = ~(lon_valid & lat_valid)
        if invalid_coords.any():
            logger.warning(f"Found {invalid_coords.sum()} records with invalid coordinates")
            df = df[~invalid_coords].copy()

        # Remove duplicates based on grid_id
        df = df.drop_duplicates(subset=['grid_id'], keep='first')

        logger.info(f"Loaded {len(df)} valid grid cells")
        return df

    def get_valid_grid_ids(self, df: pd.DataFrame) -> set:
        """Get set of valid grid IDs"""
        return set(df['grid_id'].values)


class LabelProcessor:
    """Process label data"""

    def __init__(self):
        self.label_path = config.LABEL_PATH

    def load_and_validate(self, valid_grid_ids: set) -> pd.DataFrame:
        """
        Load and validate label data

        Args:
            valid_grid_ids: Set of valid grid IDs from metadata

        Returns:
            Validated label DataFrame
        """
        logger.info(f"Loading label data from {self.label_path}")
        df = pd.read_csv(self.label_path)

        # Check required fields
        required_fields = ['grid_id', 'label']
        missing_fields = [f for f in required_fields if f not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate label range (1-9)
        label_valid = df['label'].between(*config.LABEL_RANGE)
        if not label_valid.all():
            invalid_count = (~label_valid).sum()
            logger.warning(f"Found {invalid_count} records with invalid labels")
            df = df[label_valid].copy()

        # Validate grid_id exists in metadata
        grid_valid = df['grid_id'].isin(valid_grid_ids)
        if not grid_valid.all():
            invalid_count = (~grid_valid).sum()
            logger.warning(f"Found {invalid_count} records with grid_ids not in metadata")
            df = df[grid_valid].copy()

        # Convert labels to 0-indexed (0-8) for model training
        df['label_idx'] = df['label'] - 1

        logger.info(f"Loaded {len(df)} valid labeled grid cells")
        logger.info(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

        return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal features from OD flow data

    Args:
        df: OD flow DataFrame

    Returns:
        DataFrame with temporal features
    """
    logger.info("Building temporal features")

    # Create datetime column
    df['datetime'] = df['date_dt'] + pd.to_timedelta(df['time'], unit='h')

    # Add temporal features
    df['hour'] = df['time']
    df['day_of_week'] = df['date_dt'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    return df


def aggregate_grid_flows(df: pd.DataFrame, grid_ids: List[int]) -> Dict[int, np.ndarray]:
    """
    Aggregate flows for each grid cell over time

    Args:
        df: OD flow DataFrame with temporal features
        grid_ids: List of grid IDs to process

    Returns:
        Dictionary mapping grid_id to temporal flow array (shape: [time_steps, features])
    """
    logger.info("Aggregating grid flows")

    # Create time index (0 to 167 for 168 hours)
    df['time_idx'] = (df['date_dt'] - df['date_dt'].min()).dt.days * 24 + df['hour']

    grid_flows = {}

    for grid_id in tqdm(grid_ids, desc="Processing grids"):
        # Get outflows (origin)
        outflows = df[df['o_grid_500'] == grid_id].groupby('time_idx')['num_total_normalized'].sum()

        # Get inflows (destination)
        inflows = df[df['d_grid_500'] == grid_id].groupby('time_idx')['num_total_normalized'].sum()

        # Create full time series (168 hours)
        time_series = np.zeros((config.TRAIN_HOURS, 2))

        for t in range(config.TRAIN_HOURS):
            time_series[t, 0] = outflows.get(t, 0)  # Outflow
            time_series[t, 1] = inflows.get(t, 0)   # Inflow

        grid_flows[grid_id] = time_series

    return grid_flows


if __name__ == "__main__":
    # Test preprocessing pipeline
    logger.info("Testing preprocessing pipeline")

    # Load metadata
    metadata_processor = GridMetadataProcessor()
    metadata_df = metadata_processor.load_and_validate()
    valid_grid_ids = metadata_processor.get_valid_grid_ids(metadata_df)

    # Load labels
    label_processor = LabelProcessor()
    label_df = label_processor.load_and_validate(valid_grid_ids)

    # Load OD flow data (sample)
    od_processor = ODFlowProcessor(2021)
    od_df = od_processor.load_and_preprocess(nrows=1000000)  # Load 1M rows for testing
    od_df = od_processor.filter_training_period(od_df)
    od_df = od_processor.validate_grid_ids(od_df, valid_grid_ids)
    od_df, norm_params = od_processor.normalize_flow(od_df)

    logger.info("Preprocessing test completed successfully")

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
        self.data_path_1 = config.OD_2021_PATH
        self.data_path_2 = config.OD_2024_PATH

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
        # Use config paths based on year
        if year == self.year1:
            data_path = self.data_path_1
        elif year == self.year2:
            data_path = self.data_path_2
        else:
            raise ValueError(f"Year {year} not recognized. Expected {self.year1} or {self.year2}.")

        logger.info(f"Loading OD flow data for {year} from {data_path}...")
        logger.info(f"CRITICAL FIX: Loading GLOBAL OD data (all {len(sampled_grid_ids)}+ grids)")
        logger.info(f"  → Preserving complete graph structure for GNN message passing")
        logger.info(f"  → NOT filtering edges by sampled_grid_ids")

        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=chunksize):
            # CRITICAL FIX: Don't filter edges! Load all global OD data.
            # This preserves the complete graph structure for GNN message passing.
            # GNN needs the full graph topology, not just edges connected to labeled nodes.
            # Only validate date, time, and flow > 0.
            pass  # No edge filtering - keep all edges!

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
        [DEPRECATED] Z-score normalization removed to prevent NaN errors.

        CRITICAL FIX: Z-score normalization produces negative values, which cause
        NaN when combined with log1p transformation. We now use raw flows directly
        and apply log1p only during feature aggregation.

        Args:
            od_df: OD flow DataFrame

        Returns:
            Original DataFrame (no normalization) and empty dict
        """
        logger.info("WARNING: normalize_flow() deprecated due to NaN risk.")
        logger.info("Using raw num_total values directly. log1p will be applied during aggregation.")
        return od_df, {}

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
        NEW: Aggregates to hourly snapshots (168 hours) for fine-grained temporal patterns

        Args:
            od_df: OD flow DataFrame
            grid_ids: List of grid IDs to aggregate
            use_raw: If True, use raw num_total instead of normalized values

        Returns:
            Dictionary mapping grid_id to temporal flow array (168, 2) - 168 hours, [inflow, outflow]
        """
        logger.info("Aggregating grid flows to hourly snapshots")

        # Number of hours
        num_hours = config.TRAIN_DAYS * 24  # 7 days * 24 hours = 168 hours

        grid_flows = {}

        # Choose which column to use
        flow_column = 'num_total' if use_raw else 'num_total_normalized'

        for grid_id in tqdm(grid_ids, desc="Processing grids"):
            # Initialize flow array for hourly aggregation
            hourly_flow_array = np.zeros((num_hours, 2))  # (168, [inflow, outflow])

            # Get all inflow records (this grid as destination)
            inflow_df = od_df[od_df['d_grid_500'] == grid_id].copy()

            # Get all outflow records (this grid as origin)
            outflow_df = od_df[od_df['o_grid_500'] == grid_id].copy()

            # Calculate hour index for each record
            # hour_idx = day * 24 + hour
            if len(inflow_df) > 0:
                inflow_df['day_idx'] = (inflow_df['date_dt'] - inflow_df['date_dt'].min()).dt.days
                inflow_df['hour_idx'] = inflow_df['day_idx'] * 24 + inflow_df['time']
                hourly_inflow = inflow_df.groupby('hour_idx')[flow_column].sum()

                for hour_idx, flow_val in hourly_inflow.items():
                    if 0 <= hour_idx < num_hours:
                        hourly_flow_array[hour_idx, 0] = flow_val

            if len(outflow_df) > 0:
                outflow_df['day_idx'] = (outflow_df['date_dt'] - outflow_df['date_dt'].min()).dt.days
                outflow_df['hour_idx'] = outflow_df['day_idx'] * 24 + outflow_df['time']
                hourly_outflow = outflow_df.groupby('hour_idx')[flow_column].sum()

                for hour_idx, flow_val in hourly_outflow.items():
                    if 0 <= hour_idx < num_hours:
                        hourly_flow_array[hour_idx, 1] = flow_val

            grid_flows[grid_id] = hourly_flow_array

        logger.info(f"Aggregated flows to {num_hours} hourly snapshots per grid")
        return grid_flows

    def aggregate_grid_flows_with_wamd(self, od_df, grid_ids, coord_lookup):
        """
        Aggregate total flow and weighted-average OD distance (wamd) per grid per hour.

        Args:
            od_df: OD flow DataFrame (with hour_idx pre-computed)
            grid_ids: List of grid IDs to aggregate
            coord_lookup: {grid_id: (lon, lat)} for haversine distance computation

        Returns:
            {grid_id: np.ndarray (168, 2)} where col0=total_h (raw), col1=wamd_h (km, raw)
        """
        from src.preprocessing.graph_builder import SpatialGraphBuilder

        logger.info("Aggregating grid flows with wamd (total flow + weighted-average OD distance)")
        num_hours = config.TRAIN_DAYS * 24

        # Pre-compute hour_idx once on the full dataframe
        min_date = od_df['date_dt'].min()
        od_df = od_df.copy()
        od_df['hour_idx'] = (od_df['date_dt'] - min_date).dt.days * 24 + od_df['time']

        grid_flows = {}
        for grid_id in tqdm(grid_ids, desc="Processing grids (wamd)"):
            hourly_array = np.zeros((num_hours, 2), dtype=np.float32)

            grid_lon, grid_lat = coord_lookup.get(grid_id, (0.0, 0.0))

            # --- inflow side (this grid is destination) ---
            inflow_df = od_df[od_df['d_grid_500'] == grid_id]
            if len(inflow_df) > 0:
                inflow_df = inflow_df.copy()
                o_lons = inflow_df['o_grid_500'].map(lambda g: coord_lookup.get(g, (0.0, 0.0))[0]).values
                o_lats = inflow_df['o_grid_500'].map(lambda g: coord_lookup.get(g, (0.0, 0.0))[1]).values
                src_coords = np.stack([o_lons, o_lats], axis=1).astype(np.float32)
                dst_coords = np.full_like(src_coords, [grid_lon, grid_lat])
                dists = SpatialGraphBuilder._haversine_distance_km(src_coords, dst_coords)
                inflow_df = inflow_df.assign(dist_km=dists, weighted_dist=inflow_df['num_total'].values * dists)
                in_total = inflow_df.groupby('hour_idx')['num_total'].sum()
                in_wdist = inflow_df.groupby('hour_idx')['weighted_dist'].sum()
            else:
                in_total = pd.Series(dtype=float)
                in_wdist = pd.Series(dtype=float)

            # --- outflow side (this grid is origin) ---
            outflow_df = od_df[od_df['o_grid_500'] == grid_id]
            if len(outflow_df) > 0:
                outflow_df = outflow_df.copy()
                d_lons = outflow_df['d_grid_500'].map(lambda g: coord_lookup.get(g, (0.0, 0.0))[0]).values
                d_lats = outflow_df['d_grid_500'].map(lambda g: coord_lookup.get(g, (0.0, 0.0))[1]).values
                dst_coords2 = np.stack([d_lons, d_lats], axis=1).astype(np.float32)
                src_coords2 = np.full_like(dst_coords2, [grid_lon, grid_lat])
                dists2 = SpatialGraphBuilder._haversine_distance_km(src_coords2, dst_coords2)
                outflow_df = outflow_df.assign(dist_km=dists2, weighted_dist=outflow_df['num_total'].values * dists2)
                out_total = outflow_df.groupby('hour_idx')['num_total'].sum()
                out_wdist = outflow_df.groupby('hour_idx')['weighted_dist'].sum()
            else:
                out_total = pd.Series(dtype=float)
                out_wdist = pd.Series(dtype=float)

            # --- combine into hourly array ---
            for h in range(num_hours):
                t = float(in_total.get(h, 0.0)) + float(out_total.get(h, 0.0))
                wd = float(in_wdist.get(h, 0.0)) + float(out_wdist.get(h, 0.0))
                hourly_array[h, 0] = t
                hourly_array[h, 1] = wd / t if t > 1e-6 else 0.0

            grid_flows[grid_id] = hourly_array

        logger.info(f"Aggregated wamd flows to {num_hours} hourly snapshots per grid")
        return grid_flows

    def compute_temporal_change_features_wamd(self, flows_2021_raw, flows_2024_raw):
        """
        Build (168, 4) temporal features from total_wamd flows.

        Args:
            flows_2021_raw: {grid_id: (168, 2)} raw [total_h, wamd_h]
            flows_2024_raw: {grid_id: (168, 2)} raw [total_h, wamd_h]

        Returns:
            {grid_id: np.ndarray (168, 4)}
            = [log(1+total_2021), log(1+wamd_2021), log(1+total_2024), log(1+wamd_2024)]
        """
        logger.info("Computing temporal change features (total_wamd mode)")
        change_features = {}
        for grid_id, f21 in flows_2021_raw.items():
            if grid_id not in flows_2024_raw:
                logger.warning(f"Grid {grid_id} not found in 2024 data, skipping")
                continue
            f24 = flows_2024_raw[grid_id]
            combined = np.stack([
                np.log1p(f21[:, 0]),  # total_2021_log
                np.log1p(f21[:, 1]),  # wamd_2021_log
                np.log1p(f24[:, 0]),  # total_2024_log
                np.log1p(f24[:, 1]),  # wamd_2024_log
            ], axis=1)  # (168, 4)
            change_features[grid_id] = combined
        logger.info(f"Computed wamd change features for {len(change_features)} grids")
        logger.info("Feature shape per grid: (168, 4) = [total_2021_log, wamd_2021_log, total_2024_log, wamd_2024_log]")
        return change_features

    def aggregate_grid_flows_with_degree_wamd(self, od_df, grid_ids, coord_lookup):
        """
        Aggregate (168, 3) = [total_flow_h, degree_h, wamd_h] per grid per hour.

        - total_flow_h: inflow + outflow at hour h (raw num_total)
        - degree_h:     number of unique neighbours (union of origins and destinations) at hour h
        - wamd_h:       flow-weighted average OD distance (km) at hour h

        Args:
            od_df:        OD DataFrame with date_dt, time, o_grid_500, d_grid_500, num_total
            grid_ids:     list of grid IDs to process
            coord_lookup: {grid_id: (lon, lat)}

        Returns:
            {grid_id: np.ndarray (168, 3)}  raw (NOT log-transformed)
        """
        from src.preprocessing.graph_builder import SpatialGraphBuilder

        logger.info("Aggregating grid flows with degree+wamd (flow / degree / wamd per hour)")
        num_hours = config.TRAIN_DAYS * 24

        min_date = od_df['date_dt'].min()
        od_df = od_df.copy()
        od_df['hour_idx'] = (od_df['date_dt'] - min_date).dt.days * 24 + od_df['time']

        grid_flows = {}
        for grid_id in tqdm(grid_ids, desc="Processing grids (degree+wamd)"):
            hourly_array = np.zeros((num_hours, 3), dtype=np.float32)

            grid_lon, grid_lat = coord_lookup.get(grid_id, (0.0, 0.0))

            # --- inflow side ---
            inflow_df = od_df[od_df['d_grid_500'] == grid_id]
            if len(inflow_df) > 0:
                inflow_df = inflow_df.copy()
                o_lons = inflow_df['o_grid_500'].map(
                    lambda g: coord_lookup.get(g, (0.0, 0.0))[0]).values
                o_lats = inflow_df['o_grid_500'].map(
                    lambda g: coord_lookup.get(g, (0.0, 0.0))[1]).values
                src_coords = np.stack([o_lons, o_lats], axis=1).astype(np.float32)
                dst_coords = np.full_like(src_coords, [grid_lon, grid_lat])
                dists = SpatialGraphBuilder._haversine_distance_km(src_coords, dst_coords)
                inflow_df = inflow_df.assign(
                    dist_km=dists,
                    weighted_dist=inflow_df['num_total'].values * dists
                )
                in_total  = inflow_df.groupby('hour_idx')['num_total'].sum()
                in_wdist  = inflow_df.groupby('hour_idx')['weighted_dist'].sum()
                in_degree = inflow_df.groupby('hour_idx')['o_grid_500'].nunique()
            else:
                in_total  = pd.Series(dtype=float)
                in_wdist  = pd.Series(dtype=float)
                in_degree = pd.Series(dtype=float)

            # --- outflow side ---
            outflow_df = od_df[od_df['o_grid_500'] == grid_id]
            if len(outflow_df) > 0:
                outflow_df = outflow_df.copy()
                d_lons = outflow_df['d_grid_500'].map(
                    lambda g: coord_lookup.get(g, (0.0, 0.0))[0]).values
                d_lats = outflow_df['d_grid_500'].map(
                    lambda g: coord_lookup.get(g, (0.0, 0.0))[1]).values
                dst_coords2 = np.stack([d_lons, d_lats], axis=1).astype(np.float32)
                src_coords2 = np.full_like(dst_coords2, [grid_lon, grid_lat])
                dists2 = SpatialGraphBuilder._haversine_distance_km(src_coords2, dst_coords2)
                outflow_df = outflow_df.assign(
                    dist_km=dists2,
                    weighted_dist=outflow_df['num_total'].values * dists2
                )
                out_total  = outflow_df.groupby('hour_idx')['num_total'].sum()
                out_wdist  = outflow_df.groupby('hour_idx')['weighted_dist'].sum()
                out_degree = outflow_df.groupby('hour_idx')['d_grid_500'].nunique()
            else:
                out_total  = pd.Series(dtype=float)
                out_wdist  = pd.Series(dtype=float)
                out_degree = pd.Series(dtype=float)

            # --- combine ---
            for h in range(num_hours):
                t  = float(in_total.get(h, 0.0))  + float(out_total.get(h, 0.0))
                wd = float(in_wdist.get(h, 0.0))  + float(out_wdist.get(h, 0.0))
                d  = float(in_degree.get(h, 0.0)) + float(out_degree.get(h, 0.0))
                hourly_array[h, 0] = t
                hourly_array[h, 1] = d
                hourly_array[h, 2] = wd / t if t > 1e-6 else 0.0

            grid_flows[grid_id] = hourly_array

        logger.info(f"Aggregated degree+wamd flows to {num_hours} hourly snapshots per grid")
        return grid_flows

    def compute_temporal_change_features_flow_wamd_v2(self, flows_2021_raw, flows_2024_raw):
        """
        Build (168, 4) temporal features from flow+wamd hourly arrays (degree dropped).

        Args:
            flows_2021_raw: {grid_id: (168, 3)} raw [total_h, degree_h, wamd_h]
            flows_2024_raw: {grid_id: (168, 3)} raw [total_h, degree_h, wamd_h]

        Returns:
            {grid_id: np.ndarray (168, 4)}
            = [log(1+flow_2021), log(1+wamd_2021), log(1+flow_2024), log(1+wamd_2024)]
        """
        logger.info("Computing temporal change features (flow_wamd_v2 mode, degree dropped)")
        change_features = {}
        for grid_id, f21 in flows_2021_raw.items():
            if grid_id not in flows_2024_raw:
                logger.warning(f"Grid {grid_id} not found in 2024 data, skipping")
                continue
            f24 = flows_2024_raw[grid_id]
            combined = np.stack([
                np.log1p(f21[:, 0]),  # flow_2021_log
                np.log1p(f21[:, 2]),  # wamd_2021_log  (col 1 = degree, skipped)
                np.log1p(f24[:, 0]),  # flow_2024_log
                np.log1p(f24[:, 2]),  # wamd_2024_log
            ], axis=1)  # (168, 4)
            change_features[grid_id] = combined
        logger.info(f"Computed flow_wamd_v2 change features for {len(change_features)} grids")
        logger.info("Feature shape per grid: (168, 4) = [flow21_log, wamd21_log, flow24_log, wamd24_log]")
        return change_features

    @staticmethod
    def compute_flow_wamd_v2_node_features(flows_2021_raw, flows_2024_raw):
        """
        Compute weekly [flow, wamd] per node (degree dropped) for GINE spatial node features.

        Returns:
            node_feats_2021: {grid_id: np.ndarray (2,)} = [flow_w, wamd_w]
            node_feats_2024: {grid_id: np.ndarray (2,)} = [flow_w, wamd_w]
        """
        def _weekly(arr):
            total_h = arr[:, 0]
            wamd_h  = arr[:, 2]
            flow_w  = float(total_h.sum())
            wamd_w  = float((total_h * wamd_h).sum() / max(flow_w, 1e-6))
            return np.array([flow_w, wamd_w], dtype=np.float32)

        node_feats_2021 = {gid: _weekly(arr) for gid, arr in flows_2021_raw.items()}
        node_feats_2024 = {gid: _weekly(arr) for gid, arr in flows_2024_raw.items()}
        return node_feats_2021, node_feats_2024

    def compute_temporal_change_features_flow_degree_wamd(self, flows_2021_raw, flows_2024_raw):
        """
        Build (168, 6) temporal features from flow+degree+wamd hourly arrays.

        Args:
            flows_2021_raw: {grid_id: (168, 3)} raw [total_h, degree_h, wamd_h]
            flows_2024_raw: {grid_id: (168, 3)} raw [total_h, degree_h, wamd_h]

        Returns:
            {grid_id: np.ndarray (168, 6)}
            = [log(1+flow_2021), log(1+degree_2021), log(1+wamd_2021),
               log(1+flow_2024), log(1+degree_2024), log(1+wamd_2024)]
        """
        logger.info("Computing temporal change features (flow_degree_wamd mode)")
        change_features = {}
        for grid_id, f21 in flows_2021_raw.items():
            if grid_id not in flows_2024_raw:
                logger.warning(f"Grid {grid_id} not found in 2024 data, skipping")
                continue
            f24 = flows_2024_raw[grid_id]
            combined = np.stack([
                np.log1p(f21[:, 0]),  # flow_2021_log
                np.log1p(f21[:, 1]),  # degree_2021_log
                np.log1p(f21[:, 2]),  # wamd_2021_log
                np.log1p(f24[:, 0]),  # flow_2024_log
                np.log1p(f24[:, 1]),  # degree_2024_log
                np.log1p(f24[:, 2]),  # wamd_2024_log
            ], axis=1)  # (168, 6)
            change_features[grid_id] = combined
        logger.info(f"Computed flow_degree_wamd change features for {len(change_features)} grids")
        logger.info("Feature shape per grid: (168, 6) = "
                    "[flow21_log, degree21_log, wamd21_log, flow24_log, degree24_log, wamd24_log]")
        return change_features

    @staticmethod
    def compute_flow_degree_wamd_node_features(flows_2021_raw, flows_2024_raw):
        """
        Compute weekly [flow, degree, wamd] per node for GINE spatial node features.

        Args:
            flows_2021_raw: {grid_id: (168, 3)} raw [total_h, degree_h, wamd_h]
            flows_2024_raw: {grid_id: (168, 3)} raw [total_h, degree_h, wamd_h]

        Returns:
            node_feats_2021: {grid_id: np.ndarray (3,)} = [flow_w, degree_w, wamd_w]  raw
            node_feats_2024: {grid_id: np.ndarray (3,)} = [flow_w, degree_w, wamd_w]  raw
        """
        def _weekly(arr):
            total_h  = arr[:, 0]
            degree_h = arr[:, 1]
            wamd_h   = arr[:, 2]
            flow_w   = float(total_h.sum())
            degree_w = float(degree_h.sum())
            wamd_w   = float((total_h * wamd_h).sum() / max(flow_w, 1e-6))
            return np.array([flow_w, degree_w, wamd_w], dtype=np.float32)

        node_feats_2021 = {gid: _weekly(arr) for gid, arr in flows_2021_raw.items()}
        node_feats_2024 = {gid: _weekly(arr) for gid, arr in flows_2024_raw.items()}
        return node_feats_2021, node_feats_2024

    @staticmethod
    def compute_wamd_node_features(flows_raw):
        """
        Compute weekly total flow and wamd per node for spatial branch node features.

        Args:
            flows_raw: {grid_id: (168, 2)} raw [total_h, wamd_h]

        Returns:
            {grid_id: np.ndarray (2,)} = [total_w_raw, wamd_w_raw] (NOT log-transformed)
        """
        result = {}
        for grid_id, arr in flows_raw.items():
            total_h = arr[:, 0]  # (168,)
            wamd_h = arr[:, 1]   # (168,)
            total_w = float(total_h.sum())
            if total_w > 1e-6:
                wamd_w = float((total_h * wamd_h).sum() / total_w)
            else:
                wamd_w = 0.0
            result[grid_id] = np.array([total_w, wamd_w], dtype=np.float32)
        return result


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
            flows_2021_raw: Raw grid flows for 2021 {grid_id: array(168, 2)}
            flows_2024_raw: Raw grid flows for 2024 {grid_id: array(168, 2)}
            flows_2021_norm: Not used (kept for compatibility)
            flows_2024_norm: Not used (kept for compatibility)
            ellipse_data: Ellipse data dictionary (from JSON), optional

        Returns:
            Dictionary with change features for each grid
            Shape: (168, 4) without ellipse or (168, 8) with ellipse
            Without: [inflow_2021_log, outflow_2021_log, inflow_2024_log, outflow_2024_log]
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

            # Get raw flows (168, 2) - [inflow, outflow]
            flow_2021_raw = flows_2021_raw[grid_id]  # (168, 2)
            flow_2024_raw = flows_2024_raw[grid_id]  # (168, 2)

            # NEW: Use inflow and outflow separately (preserves directional information)
            # This captures both flow intensity and spatial direction

            # Extract inflow and outflow for each year
            # Shape: (168, 2) = [inflow, outflow]
            inflow_2021 = flow_2021_raw[:, 0]  # (7,)
            outflow_2021 = flow_2021_raw[:, 1]  # (7,)
            inflow_2024 = flow_2024_raw[:, 0]  # (7,)
            outflow_2024 = flow_2024_raw[:, 1]  # (7,)

            # Apply log transformation to preserve magnitude
            inflow_2021_log = np.log1p(inflow_2021)
            outflow_2021_log = np.log1p(outflow_2021)
            inflow_2024_log = np.log1p(inflow_2024)
            outflow_2024_log = np.log1p(outflow_2024)

            # Stack features: [inflow_2021, outflow_2021, inflow_2024, outflow_2024]
            # Shape: (168, 4) = [inflow_2021_log, outflow_2021_log, inflow_2024_log, outflow_2024_log]
            combined = np.stack([
                inflow_2021_log,
                outflow_2021_log,
                inflow_2024_log,
                outflow_2024_log
            ], axis=1)  # (168, 4)

            change_features[grid_id] = combined

        logger.info(f"Computed change features for {len(change_features)} grids")
        if ellipse_data is not None:
            logger.info(f"  - Grids with ellipse features: {grids_with_ellipse}")
            logger.info(f"  - Grids without ellipse features: {grids_without_ellipse}")
            logger.info(f"Feature shape per grid: (168, 8) = [inflow_2021, outflow_2021, inflow_2024, outflow_2024, ecc_2021, area_2021, ecc_2024, area_2024]")
        else:
            logger.info(f"Feature shape per grid: (168, 4) = [inflow_2021_log, outflow_2021_log, inflow_2024_log, outflow_2024_log]")

        return change_features

    def prepare_dual_year_data(self, sampled_grid_ids, valid_grid_ids, coord_lookup=None):
        """
        Prepare data from both years for training

        Args:
            sampled_grid_ids: Set of grid IDs to process
            valid_grid_ids: Set of all valid grid IDs
            coord_lookup: {grid_id: (lon, lat)} required for TEMPORAL_FEATURE_MODE="total_wamd"

        Returns:
            Dictionary with processed data from both years
        """
        logger.info("=" * 80)
        logger.info("Preparing Dual-Year Data (2021 vs 2024)")
        logger.info("=" * 80)

        temporal_feature_mode = getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')
        logger.info(f"Temporal feature mode: {temporal_feature_mode}")

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

        labeled_grid_ids = list(sampled_grid_ids)

        if temporal_feature_mode == 'total_wamd':
            if coord_lookup is None:
                raise ValueError("coord_lookup is required for TEMPORAL_FEATURE_MODE='total_wamd'")
            logger.info(f"Aggregating total+wamd flows for {len(labeled_grid_ids)} labeled grids")
            flows_2021_raw = self.aggregate_grid_flows_with_wamd(od_2021, labeled_grid_ids, coord_lookup)
            flows_2024_raw = self.aggregate_grid_flows_with_wamd(od_2024, labeled_grid_ids, coord_lookup)
            change_features = self.compute_temporal_change_features_wamd(flows_2021_raw, flows_2024_raw)
            wamd_node_features_2021 = self.compute_wamd_node_features(flows_2021_raw)
            wamd_node_features_2024 = self.compute_wamd_node_features(flows_2024_raw)
            fdw_node_features_2021 = None
            fdw_node_features_2024 = None
        elif temporal_feature_mode == 'flow_degree_wamd':
            if coord_lookup is None:
                raise ValueError("coord_lookup is required for TEMPORAL_FEATURE_MODE='flow_degree_wamd'")
            logger.info(f"Aggregating flow+degree+wamd for {len(labeled_grid_ids)} labeled grids")
            flows_2021_raw = self.aggregate_grid_flows_with_degree_wamd(od_2021, labeled_grid_ids, coord_lookup)
            flows_2024_raw = self.aggregate_grid_flows_with_degree_wamd(od_2024, labeled_grid_ids, coord_lookup)
            change_features = self.compute_temporal_change_features_flow_degree_wamd(flows_2021_raw, flows_2024_raw)
            fdw_node_features_2021, fdw_node_features_2024 = \
                self.compute_flow_degree_wamd_node_features(flows_2021_raw, flows_2024_raw)
            wamd_node_features_2021 = None
            wamd_node_features_2024 = None
        elif temporal_feature_mode == 'flow_wamd_v2':
            if coord_lookup is None:
                raise ValueError("coord_lookup is required for TEMPORAL_FEATURE_MODE='flow_wamd_v2'")
            logger.info(f"Aggregating flow+wamd (no degree) for {len(labeled_grid_ids)} labeled grids")
            flows_2021_raw = self.aggregate_grid_flows_with_degree_wamd(od_2021, labeled_grid_ids, coord_lookup)
            flows_2024_raw = self.aggregate_grid_flows_with_degree_wamd(od_2024, labeled_grid_ids, coord_lookup)
            change_features = self.compute_temporal_change_features_flow_wamd_v2(flows_2021_raw, flows_2024_raw)
            fdw_node_features_2021, fdw_node_features_2024 = \
                self.compute_flow_wamd_v2_node_features(flows_2021_raw, flows_2024_raw)
            wamd_node_features_2021 = None
            wamd_node_features_2024 = None
        else:
            logger.info(f"Aggregating RAW flows for {len(labeled_grid_ids)} labeled grids")
            logger.info("  → Using raw num_total (absolute trip counts)")
            flows_2021_raw = self.aggregate_grid_flows(od_2021, labeled_grid_ids, use_raw=True)
            flows_2024_raw = self.aggregate_grid_flows(od_2024, labeled_grid_ids, use_raw=True)
            change_features = self.compute_temporal_change_features(
                flows_2021_raw, flows_2024_raw,
                flows_2021_raw, flows_2024_raw,
                ellipse_data=None
            )
            wamd_node_features_2021 = None
            wamd_node_features_2024 = None
            fdw_node_features_2021 = None
            fdw_node_features_2024 = None

        logger.info(f"\nDual-year data preparation completed:")
        logger.info(f"  - 2021 OD records: {len(od_2021)}")
        logger.info(f"  - 2024 OD records: {len(od_2024)}")
        logger.info(f"  - Grids with change features: {len(change_features)}")

        return {
            'od_2021': od_2021,
            'od_2024': od_2024,
            'flows_2021': flows_2021_raw,
            'flows_2024': flows_2024_raw,
            'change_features': change_features,
            'norm_params_2021': {},
            'norm_params_2024': {},
            'wamd_node_features_2021': wamd_node_features_2021,
            'wamd_node_features_2024': wamd_node_features_2024,
            'fdw_node_features_2021': fdw_node_features_2021,
            'fdw_node_features_2024': fdw_node_features_2024,
        }

    def extract_features_for_grids(self, od_data, grid_ids):
        """
        Extract temporal features for specific grid IDs

        Args:
            od_data: OD flow DataFrame (already processed with temporal features)
            grid_ids: List of grid IDs to extract features for

        Returns:
            Dictionary mapping grid_id to feature array
        """
        logger.info(f"Extracting features for {len(grid_ids)} grid nodes...")

        features_dict = {}

        # Get unique time indices from od_data
        if 'time_idx' in od_data.columns:
            time_indices = od_data['time_idx'].unique()
        else:
            # If no time_idx, assume 7 days with 24 hours each
            time_indices = range(168)

        for grid_id in grid_ids:
            # Filter OD data for this grid as origin or destination
            grid_od = od_data[
                (od_data['o_grid_500'] == grid_id) |
                (od_data['d_grid_500'] == grid_id)
            ]

            if len(grid_od) == 0:
                # No OD data for this grid - use zero features
                logger.warning(f"  No OD data found for grid {grid_id}, using zero features")
                features_dict[grid_id] = np.zeros((len(time_indices), 1))
                continue

            # Aggregate by time to get total flow per timestep
            if len(time_indices) > 0:
                # Get inflow and outflow for each time step
                time_series = []
                for t in time_indices:
                    t_data = grid_od[grid_od['time_idx'] == t] if 'time_idx' in od_data.columns else grid_od

                    # Sum num_total as inflow + outflow
                    if 'num_total' in t_data.columns:
                        inflow = t_data[t_data['d_grid_500'] == grid_id]['num_total'].sum()
                        outflow = t_data[t_data['o_grid_500'] == grid_id]['num_total'].sum()
                    else:
                        inflow = 0
                        outflow = 0

                    total = inflow + outflow
                    time_series.append([total])

                features_dict[grid_id] = np.array(time_series)
            else:
                # No temporal dimension, use aggregated total
                if 'num_total' in grid_od.columns:
                    total_flow = grid_od['num_total'].sum()
                    features_dict[grid_id] = np.array([[total_flow]])
                else:
                    features_dict[grid_id] = np.zeros((1, 1))

        logger.info(f"  ✓ Extracted features for {len(features_dict)} grids")
        return features_dict


def _build_daily_graph_sequence_from_static(
        od_df: pd.DataFrame,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
        grid_id_to_idx: dict,
    train_days: int,
    edge_attr_template: np.ndarray = None) -> list:
    """
    Build daily discrete graph snapshots using fixed topology and dynamic edge weights.

    Dynamic strategy:
    - Keep edge_index fixed (derived from the static graph/top-k graph).
    - Recompute daily edge weights for OD edges.
    - Preserve static weights for non-OD fallback edges (e.g., KNN edges).
    - Preserve static self-loop weights when a daily edge weight is zero.
    """
    num_edges = edge_index.shape[1]
    edge_weights = edge_weights.astype(np.float32)

    edge_pos = {(int(edge_index[0, i]), int(edge_index[1, i])): i for i in range(num_edges)}
    self_loop_mask = edge_index[0] == edge_index[1]

    od_df = od_df.copy()
    if not np.issubdtype(od_df['date_dt'].dtype, np.datetime64):
        od_df['date_dt'] = pd.to_datetime(od_df['date_dt'])

    min_date = od_df['date_dt'].min()
    od_df['day_idx'] = (od_df['date_dt'] - min_date).dt.days
    od_df = od_df[(od_df['day_idx'] >= 0) & (od_df['day_idx'] < train_days)]

    grouped = od_df.groupby(['day_idx', 'o_grid_500', 'd_grid_500'])['num_total'].sum().reset_index()

    day_to_updates = {day: [] for day in range(train_days)}
    od_edge_positions = set()

    for row in grouped.itertuples(index=False):
        if row.o_grid_500 not in grid_id_to_idx or row.d_grid_500 not in grid_id_to_idx:
            continue
        src = grid_id_to_idx[row.o_grid_500]
        dst = grid_id_to_idx[row.d_grid_500]
        pos = edge_pos.get((src, dst))
        if pos is None:
            continue
        day_to_updates[int(row.day_idx)].append((pos, float(row.num_total)))
        od_edge_positions.add(pos)

    non_od_mask = np.ones(num_edges, dtype=bool)
    if od_edge_positions:
        non_od_mask[list(od_edge_positions)] = False

    graphs = []
    for day in range(train_days):
        day_weights = np.zeros(num_edges, dtype=np.float32)
        for pos, val in day_to_updates[day]:
            day_weights[pos] = val

        # Keep static weights for non-OD fallback edges (e.g., KNN fallback edges).
        day_weights[non_od_mask] = edge_weights[non_od_mask]

        # Keep self-loop weights stable when daily OD is absent.
        zero_self_loop_mask = self_loop_mask & (day_weights <= 0)
        day_weights[zero_self_loop_mask] = edge_weights[zero_self_loop_mask]

        if edge_attr_template is None or edge_attr_template.ndim == 1:
            day_edge_attr = day_weights
        else:
            day_edge_attr = edge_attr_template.copy()
            day_edge_attr[:, 0] = day_weights

        graphs.append((edge_index.copy(), day_edge_attr))

    return graphs


def _resolve_edge_feature_mode(spatial_model: str, edge_feature_mode: str = None) -> str:
    """Resolve effective graph edge feature mode for the current experiment."""
    spatial_model = (spatial_model or getattr(config, 'SPATIAL_MODEL', 'GCN')).upper()
    configured_mode = edge_feature_mode or getattr(config, 'GINE_EDGE_FEATURE_MODE', 'flow_only')

    if spatial_model not in ('GINE', 'GAT', 'MPNN'):
        return 'flow_only'

    if configured_mode not in {'flow_only', 'flow_distance_direction', 'flow_distribution'}:
        raise ValueError(
            f"Unsupported GINE edge feature mode: {configured_mode}. "
            "Use 'flow_only', 'flow_distance_direction', or 'flow_distribution'."
        )

    return configured_mode


def _ensure_scalar_edge_weights(edge_weights: np.ndarray) -> np.ndarray:
    """Normalize cached base edge weights to a 1-d flow vector."""
    edge_weights = np.asarray(edge_weights)
    if edge_weights.ndim == 1:
        return edge_weights.astype(np.float32, copy=False)
    return edge_weights[:, 0].astype(np.float32, copy=False)


def prepare_dual_year_experiment_data(
    label_path,
    samples_per_class=None,
    use_cache=True,
    cache_dir='data/cache',
    spatial_model=None,
    edge_feature_mode=None):
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
    from src.preprocessing.graph_builder import SpatialGraphBuilder

    # Two-level cache layout:
    # 1) feature cache: labels + temporal features
    # 2) graph cache: graph structure (with base graph reuse for top-k variants)
    feature_cache_dir = os.path.join(cache_dir, 'features')
    graph_cache_dir = os.path.join(cache_dir, 'graphs')
    os.makedirs(feature_cache_dir, exist_ok=True)
    os.makedirs(graph_cache_dir, exist_ok=True)

    # Generate cache key with file content hash and modification times
    # Strategy: Use content hash for label file (small), mtime for OD data files (large)

    # 1. Label file content hash (small file, use content hash for 100% accuracy)
    with open(label_path, 'rb') as f:
        label_content_hash = hashlib.md5(f.read()).hexdigest()[:8]

    # 2. OD data file modification times (large files, use mtime for efficiency)
    data_2021_path = config.OD_2021_PATH
    data_2024_path = config.OD_2024_PATH

    if os.path.exists(data_2021_path):
        data_2021_mtime = int(os.path.getmtime(data_2021_path))
    else:
        data_2021_mtime = 0

    if os.path.exists(data_2024_path):
        data_2024_mtime = int(os.path.getmtime(data_2024_path))
    else:
        data_2024_mtime = 0

    metadata_path = config.GRID_METADATA_PATH
    metadata_mtime = int(os.path.getmtime(metadata_path)) if os.path.exists(metadata_path) else 0

    # Feature cache key excludes graph hyperparameters so top-k sweeps can reuse temporal preprocessing.
    label_basename = os.path.basename(label_path)
    feature_cache_key = (
        f"{label_basename}_{label_content_hash}_samples_{samples_per_class}_"
        f"seed_{config.RANDOM_SEED}_days_{config.TRAIN_DAYS}_"
        f"data_{data_2021_mtime}_{data_2024_mtime}_"
        f"temporal_mode_{getattr(config, 'TEMPORAL_FEATURE_MODE', 'inflow_outflow')}_v3"
    )
    feature_cache_hash = hashlib.md5(feature_cache_key.encode()).hexdigest()[:12]
    feature_cache_file = os.path.join(feature_cache_dir, f"dual_year_features_{feature_cache_hash}.pkl")

    # Graph base cache key excludes top-k; derived top-k graphs are generated from this base cache.
    graph_base_key = (
        f"meta_{metadata_mtime}_days_{config.TRAIN_DAYS}_"
        f"data_{data_2021_mtime}_{data_2024_mtime}_threshold_{config.FLOW_THRESHOLD}_base_v2"
    )
    graph_base_hash = hashlib.md5(graph_base_key.encode()).hexdigest()[:12]
    graph_base_cache_file = os.path.join(graph_cache_dir, f"dual_year_graph_base_{graph_base_hash}.pkl")

    graph_temporal_mode = getattr(config, 'GRAPH_TEMPORAL_MODE', 'static')
    effective_spatial_model = (spatial_model or getattr(config, 'SPATIAL_MODEL', 'GCN')).upper()
    if effective_spatial_model == 'GIN':
        effective_spatial_model = 'GINE'
    effective_edge_feature_mode = _resolve_edge_feature_mode(effective_spatial_model, edge_feature_mode)
    graph_variant_key = (
        f"base_{graph_base_hash}_topkout_{config.GRAPH_TOPK_OUT}_topkin_{config.GRAPH_TOPK_IN}_"
        f"temporal_{graph_temporal_mode}_spatial_{effective_spatial_model}_"
        f"edgefeat_{effective_edge_feature_mode}_v5"
    )
    graph_variant_hash = hashlib.md5(graph_variant_key.encode()).hexdigest()[:12]
    graph_variant_cache_file = os.path.join(graph_cache_dir, f"dual_year_graph_variant_{graph_variant_hash}.pkl")

    feature_data = None

    logger.info("=" * 80)
    logger.info("Dual-Year Experiment Data Preparation")
    logger.info("=" * 80)
    logger.info(f"Label file: {label_path}")
    logger.info(f"Samples per class: {samples_per_class if samples_per_class else 'ALL'}")
    logger.info("")

    # Load metadata (fast enough to do every run; reused by both cache levels)
    logger.info("Loading grid metadata...")
    metadata_processor = GridMetadataProcessor()
    metadata_df = metadata_processor.load_and_validate()
    valid_grid_ids = metadata_processor.get_valid_grid_ids(metadata_df)

    # ----------------------------
    # Level-1 cache: feature data
    # ----------------------------
    if use_cache and os.path.exists(feature_cache_file):
        logger.info("=" * 80)
        logger.info("Loading Feature Cache (Level-1)")
        logger.info("=" * 80)
        logger.info(f"Feature cache file: {feature_cache_file}")
        try:
            with open(feature_cache_file, 'rb') as f:
                feature_data = pickle.load(f)
            logger.info("✓ Feature cache hit")
            logger.info(f"  - Total labeled grids: {len(feature_data['labels'])}")
        except Exception as e:
            logger.warning(f"Failed to load feature cache: {e}")
            feature_data = None

    if feature_data is None:
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

        # Prepare dual-year temporal data
        dual_year_processor = DualYearDataProcessor(year1=2021, year2=2024)
        coord_lookup = {
            row.grid_id: (float(row.lon), float(row.lat))
            for row in metadata_df[['grid_id', 'lon', 'lat']].itertuples(index=False)
        }
        dual_year_data = dual_year_processor.prepare_dual_year_data(
            sampled_grid_ids, valid_grid_ids, coord_lookup=coord_lookup
        )

        feature_data = {
            'labels': labels,
            'change_features': dual_year_data['change_features'],
            'flows_2021': dual_year_data['flows_2021'],
            'flows_2024': dual_year_data['flows_2024'],
            'train_flow_grid_ids': sorted(dual_year_data['flows_2021'].keys()),
            'norm_params_2021': dual_year_data['norm_params_2021'],
            'norm_params_2024': dual_year_data['norm_params_2024'],
            'wamd_node_features_2021': dual_year_data['wamd_node_features_2021'],
            'wamd_node_features_2024': dual_year_data['wamd_node_features_2024'],
            'fdw_node_features_2021': dual_year_data.get('fdw_node_features_2021'),
            'fdw_node_features_2024': dual_year_data.get('fdw_node_features_2024'),
            'label_df': label_df,
            'class_weights': class_weights,
            'label_file_name': label_path,
            'label_file_hash': label_content_hash,
            'class_distribution': class_counts.to_dict()
        }

        if use_cache:
            try:
                with open(feature_cache_file, 'wb') as f:
                    pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("✓ Feature cache saved")
                logger.info(f"  - Path: {feature_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save feature cache: {e}")

    labels = feature_data['labels']
    class_weights = feature_data['class_weights']

    # -------------------------
    # Level-2 cache: graph data
    # -------------------------
    logger.info("Building/loading spatial graphs...")
    logger.info(f"CRITICAL FIX: Using GLOBAL metadata (all {len(metadata_df)} grids)")

    graph_builder = SpatialGraphBuilder(metadata_df, k_neighbors=8)

    graph_cache_data = None
    if use_cache and os.path.exists(graph_variant_cache_file):
        logger.info("=" * 80)
        logger.info("Loading Graph Cache (Level-2 variant)")
        logger.info("=" * 80)
        logger.info(f"Graph cache file: {graph_variant_cache_file}")
        try:
            with open(graph_variant_cache_file, 'rb') as f:
                graph_cache_data = pickle.load(f)
            logger.info("✓ Graph variant cache hit")
        except Exception as e:
            logger.warning(f"Failed to load graph variant cache: {e}")
            graph_cache_data = None

    if graph_cache_data is None:
        base_graph_data = None
        if use_cache and os.path.exists(graph_base_cache_file):
            logger.info("=" * 80)
            logger.info("Loading Graph Base Cache")
            logger.info("=" * 80)
            logger.info(f"Graph base cache file: {graph_base_cache_file}")
            try:
                with open(graph_base_cache_file, 'rb') as f:
                    base_graph_data = pickle.load(f)
                base_graph_data['edge_weights_2021_base'] = _ensure_scalar_edge_weights(
                    base_graph_data['edge_weights_2021_base']
                )
                base_graph_data['edge_weights_2024_base'] = _ensure_scalar_edge_weights(
                    base_graph_data['edge_weights_2024_base']
                )
                logger.info("✓ Graph base cache hit")
            except Exception as e:
                logger.warning(f"Failed to load graph base cache: {e}")
                base_graph_data = None

        if base_graph_data is None:
            logger.info("Graph base cache miss: building base graphs from OD data (no top-k)")
            logger.info("  → This is done once per data version; top-k variants will reuse it")

            dual_year_processor_for_graph = DualYearDataProcessor(year1=2021, year2=2024)

            # For base graph construction, labels are irrelevant; we only need global OD and valid nodes.
            od_2021 = dual_year_processor_for_graph.load_year_data(2021, valid_grid_ids)
            od_2021 = dual_year_processor_for_graph.filter_training_period(od_2021, config.TRAIN_DAYS)
            od_2021 = od_2021[od_2021['o_grid_500'].isin(valid_grid_ids) &
                              od_2021['d_grid_500'].isin(valid_grid_ids)]

            od_2024 = dual_year_processor_for_graph.load_year_data(2024, valid_grid_ids)
            od_2024 = dual_year_processor_for_graph.filter_training_period(od_2024, config.TRAIN_DAYS)
            od_2024 = od_2024[od_2024['o_grid_500'].isin(valid_grid_ids) &
                              od_2024['d_grid_500'].isin(valid_grid_ids)]

            # Prune nodes with no flow in either year to keep the graph compact.
            # Nodes absent from both OD datasets carry no signal and inflate graph size.
            nodes_2021 = set(od_2021['o_grid_500']) | set(od_2021['d_grid_500'])
            nodes_2024 = set(od_2024['o_grid_500']) | set(od_2024['d_grid_500'])
            active_grid_ids = nodes_2021 & nodes_2024
            before = len(valid_grid_ids)
            after = len(active_grid_ids)
            logger.info(f"Active-node filter: {before} metadata nodes → {after} nodes active in both years "
                        f"({before - after} pruned, {100*(before-after)/before:.1f}%)")
            # Rebuild graph_builder restricted to active nodes so node indices are compact.
            active_metadata_df = metadata_df[metadata_df['grid_id'].isin(active_grid_ids)].copy()
            graph_builder = SpatialGraphBuilder(active_metadata_df, k_neighbors=8)
            od_2021 = od_2021[od_2021['o_grid_500'].isin(active_grid_ids) &
                              od_2021['d_grid_500'].isin(active_grid_ids)]
            od_2024 = od_2024[od_2024['o_grid_500'].isin(active_grid_ids) &
                              od_2024['d_grid_500'].isin(active_grid_ids)]

            # Build base graphs without top-k; this cache is reused for all top-k variants.
            edge_index_2021_base, edge_weights_2021_base = graph_builder.build_flow_graph(
                od_2021,
                threshold=config.FLOW_THRESHOLD,
                include_neighbors=False,
                topk_out=None,
                topk_in=None
            )
            edge_index_2024_base, edge_weights_2024_base = graph_builder.build_flow_graph(
                od_2024,
                threshold=config.FLOW_THRESHOLD,
                include_neighbors=False,
                topk_out=None,
                topk_in=None
            )

            base_graph_data = {
                'edge_index_2021_base': edge_index_2021_base,
                'edge_weights_2021_base': edge_weights_2021_base,
                'edge_index_2024_base': edge_index_2024_base,
                'edge_weights_2024_base': edge_weights_2024_base,
                'grid_id_to_idx': graph_builder.grid_id_to_idx
            }

            if use_cache:
                try:
                    with open(graph_base_cache_file, 'wb') as f:
                        pickle.dump(base_graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info("✓ Graph base cache saved")
                    logger.info(f"  - Path: {graph_base_cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to save graph base cache: {e}")

        # Derive requested top-k variant from base graph.
        if config.GRAPH_TOPK_OUT is None and config.GRAPH_TOPK_IN is None:
            edge_index_2021 = base_graph_data['edge_index_2021_base']
            edge_weights_2021 = base_graph_data['edge_weights_2021_base']
            edge_index_2024 = base_graph_data['edge_index_2024_base']
            edge_weights_2024 = base_graph_data['edge_weights_2024_base']
        else:
            logger.info(f"Deriving top-k variant from base graph: out={config.GRAPH_TOPK_OUT}, in={config.GRAPH_TOPK_IN}")
            edge_index_2021, edge_weights_2021 = graph_builder._apply_nodewise_topk(
                base_graph_data['edge_index_2021_base'],
                base_graph_data['edge_weights_2021_base'],
                topk_out=config.GRAPH_TOPK_OUT,
                topk_in=config.GRAPH_TOPK_IN
            )
            edge_index_2024, edge_weights_2024 = graph_builder._apply_nodewise_topk(
                base_graph_data['edge_index_2024_base'],
                base_graph_data['edge_weights_2024_base'],
                topk_out=config.GRAPH_TOPK_OUT,
                topk_in=config.GRAPH_TOPK_IN
            )

        edge_attr_2021 = graph_builder.build_edge_attr(
            edge_index_2021,
            edge_weights_2021,
            mode=effective_edge_feature_mode
        )
        edge_attr_2024 = graph_builder.build_edge_attr(
            edge_index_2024,
            edge_weights_2024,
            mode=effective_edge_feature_mode
        )

        if graph_temporal_mode == 'daily':
            logger.info("Building daily discrete graph snapshots (fixed topology + dynamic edge weights)")

            dual_year_processor_for_graph = DualYearDataProcessor(year1=2021, year2=2024)

            od_2021_daily = dual_year_processor_for_graph.load_year_data(2021, valid_grid_ids)
            od_2021_daily = dual_year_processor_for_graph.filter_training_period(od_2021_daily, config.TRAIN_DAYS)
            od_2021_daily = od_2021_daily[
                od_2021_daily['o_grid_500'].isin(valid_grid_ids) &
                od_2021_daily['d_grid_500'].isin(valid_grid_ids)
            ]

            od_2024_daily = dual_year_processor_for_graph.load_year_data(2024, valid_grid_ids)
            od_2024_daily = dual_year_processor_for_graph.filter_training_period(od_2024_daily, config.TRAIN_DAYS)
            od_2024_daily = od_2024_daily[
                od_2024_daily['o_grid_500'].isin(valid_grid_ids) &
                od_2024_daily['d_grid_500'].isin(valid_grid_ids)
            ]

            graphs_2021 = _build_daily_graph_sequence_from_static(
                od_df=od_2021_daily,
                edge_index=edge_index_2021,
                edge_weights=edge_weights_2021,
                grid_id_to_idx=base_graph_data['grid_id_to_idx'],
                train_days=config.TRAIN_DAYS,
                edge_attr_template=edge_attr_2021
            )
            graphs_2024 = _build_daily_graph_sequence_from_static(
                od_df=od_2024_daily,
                edge_index=edge_index_2024,
                edge_weights=edge_weights_2024,
                grid_id_to_idx=base_graph_data['grid_id_to_idx'],
                train_days=config.TRAIN_DAYS,
                edge_attr_template=edge_attr_2024
            )
        else:
            graphs_2021 = [(edge_index_2021, edge_attr_2021)]
            graphs_2024 = [(edge_index_2024, edge_attr_2024)]

        graph_cache_data = {
            'graphs_2021': graphs_2021,
            'graphs_2024': graphs_2024,
            'edge_index': edge_index_2024,
            'edge_weights': edge_weights_2024,
            'grid_id_to_idx': base_graph_data['grid_id_to_idx'],
            'graph_temporal_mode': graph_temporal_mode,
            'edge_feature_mode': effective_edge_feature_mode
        }

        if use_cache:
            try:
                with open(graph_variant_cache_file, 'wb') as f:
                    pickle.dump(graph_cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("✓ Graph variant cache saved")
                logger.info(f"  - Path: {graph_variant_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save graph variant cache: {e}")

    graphs_2021 = graph_cache_data['graphs_2021']
    graphs_2024 = graph_cache_data['graphs_2024']
    edge_index = graph_cache_data['edge_index']
    edge_weights = graph_cache_data['edge_weights']
    grid_id_to_idx = graph_cache_data['grid_id_to_idx']

    logger.info(f"\n✓ CRITICAL FIX APPLIED:")
    logger.info(f"  - Global graph nodes: {len(grid_id_to_idx)} (from {len(metadata_df)} metadata)")
    logger.info(f"  - Labeled nodes for training: {len(labels)}")
    logger.info(f"  - Unlabeled nodes in graph: {len(grid_id_to_idx) - len(labels)}")
    logger.info(f"  - Graph temporal mode: {graph_temporal_mode}")
    logger.info(f"  - Graph edge feature mode: {graph_cache_data.get('edge_feature_mode', 'flow_only')}")
    logger.info(f"  - Graph snapshots per year: {len(graphs_2021)}")
    logger.info(f"  - Graph 2021 edges (snapshot0): {graphs_2021[0][0].shape[1]}")
    logger.info(f"  - Graph 2024 edges (snapshot0): {graphs_2024[0][0].shape[1]}")
    logger.info(f"\nArchitecture:")
    logger.info(f"  - Temporal Branch (LSTM): Uses {len(labels)} labeled nodes with (168, 2) features")
    logger.info(f"  - Spatial Branch (GCN/SAGE): Uses {len(grid_id_to_idx)} nodes with all-1 features")
    logger.info(f"  - Feature dimension per labeled node: {list(feature_data['change_features'].values())[0].shape}")

    data = {
        'metadata_df': metadata_df,
        'labels': feature_data['labels'],
        'change_features': feature_data['change_features'],
        'flows_2021': feature_data['flows_2021'],
        'flows_2024': feature_data['flows_2024'],
        'train_flow_grid_ids': feature_data.get('train_flow_grid_ids', sorted(feature_data['flows_2021'].keys())),
        'edge_index': edge_index,
        'edge_weights': edge_weights,
        'graphs_2021': graphs_2021,
        'graphs_2024': graphs_2024,
        'edge_feature_mode': graph_cache_data.get('edge_feature_mode', 'flow_only'),
        'grid_id_to_idx': grid_id_to_idx,
        'norm_params_2021': feature_data['norm_params_2021'],
        'norm_params_2024': feature_data['norm_params_2024'],
        'wamd_node_features_2021': feature_data.get('wamd_node_features_2021'),
        'wamd_node_features_2024': feature_data.get('wamd_node_features_2024'),
        'fdw_node_features_2021': feature_data.get('fdw_node_features_2021'),
        'fdw_node_features_2024': feature_data.get('fdw_node_features_2024'),
        'label_df': feature_data['label_df'],
        'class_weights': feature_data['class_weights'],
        'label_file_name': feature_data['label_file_name'],
        'label_file_hash': feature_data['label_file_hash'],
        'class_distribution': feature_data['class_distribution']
    }

    logger.info("\nTwo-level cache summary:")
    logger.info(f"  - Feature cache: {feature_cache_file}")
    logger.info(f"  - Graph base cache: {graph_base_cache_file}")
    logger.info(f"  - Graph variant cache: {graph_variant_cache_file}")

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

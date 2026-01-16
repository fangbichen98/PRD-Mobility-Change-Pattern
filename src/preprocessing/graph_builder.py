"""
Graph construction utilities for spatial relationship modeling
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class SpatialGraphBuilder:
    """Build spatial graphs from grid metadata"""

    def __init__(self, metadata_df: pd.DataFrame, k_neighbors: int = 8):
        """
        Initialize graph builder

        Args:
            metadata_df: Grid metadata DataFrame
            k_neighbors: Number of nearest neighbors for graph construction
        """
        self.metadata_df = metadata_df
        self.k_neighbors = k_neighbors
        self.grid_id_to_idx = {gid: idx for idx, gid in enumerate(metadata_df['grid_id'])}
        self.idx_to_grid_id = {idx: gid for gid, idx in self.grid_id_to_idx.items()}

    def build_knn_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-nearest neighbor graph based on spatial proximity

        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_weights: Edge weights based on distance
        """
        logger.info(f"Building KNN graph with k={self.k_neighbors}")

        # Extract coordinates
        coords = self.metadata_df[['lon', 'lat']].values

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(coords)

        # Find k nearest neighbors for each node
        distances, indices = tree.query(coords, k=self.k_neighbors + 1)

        # Build edge list (exclude self-loops)
        edge_list = []
        edge_weights = []

        for i in range(len(coords)):
            for j, dist in zip(indices[i, 1:], distances[i, 1:]):  # Skip first (self)
                edge_list.append([i, j])
                # Use inverse distance as weight (closer = stronger connection)
                edge_weights.append(1.0 / (dist + 1e-6))

        edge_index = np.array(edge_list).T
        edge_weights = np.array(edge_weights)

        logger.info(f"Created graph with {len(coords)} nodes and {len(edge_list)} edges")

        return edge_index, edge_weights

    def build_flow_graph(self, od_df: pd.DataFrame, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on OD flow patterns

        Args:
            od_df: OD flow DataFrame
            threshold: Minimum normalized flow to create edge

        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_weights: Edge weights based on flow volume
        """
        logger.info(f"Building flow graph with threshold={threshold}")

        # Aggregate flows between grid pairs
        flow_agg = od_df.groupby(['o_grid_500', 'd_grid_500'])['num_total_normalized'].sum().reset_index()
        flow_agg = flow_agg[flow_agg['num_total_normalized'] >= threshold]

        # Convert grid IDs to indices
        edge_list = []
        edge_weights = []

        for _, row in flow_agg.iterrows():
            o_grid = row['o_grid_500']
            d_grid = row['d_grid_500']

            if o_grid in self.grid_id_to_idx and d_grid in self.grid_id_to_idx:
                o_idx = self.grid_id_to_idx[o_grid]
                d_idx = self.grid_id_to_idx[d_grid]

                edge_list.append([o_idx, d_idx])
                edge_weights.append(row['num_total_normalized'])

        edge_index = np.array(edge_list).T if edge_list else np.zeros((2, 0))
        edge_weights = np.array(edge_weights) if edge_weights else np.array([])

        logger.info(f"Created flow graph with {len(edge_list)} edges")

        return edge_index, edge_weights

    def build_hybrid_graph(self, od_df: pd.DataFrame,
                          spatial_weight: float = 0.5,
                          flow_weight: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build hybrid graph combining spatial proximity and flow patterns

        Args:
            od_df: OD flow DataFrame
            spatial_weight: Weight for spatial edges
            flow_weight: Weight for flow edges

        Returns:
            edge_index: Combined edge indices
            edge_weights: Combined edge weights
        """
        logger.info("Building hybrid graph")

        # Build spatial graph
        spatial_edges, spatial_weights = self.build_knn_graph()

        # Build flow graph
        flow_edges, flow_weights = self.build_flow_graph(od_df)

        # Normalize weights
        if len(spatial_weights) > 0:
            spatial_weights = spatial_weights / spatial_weights.max() * spatial_weight

        if len(flow_weights) > 0:
            flow_weights = flow_weights / flow_weights.max() * flow_weight

        # Combine edges
        edge_index = np.concatenate([spatial_edges, flow_edges], axis=1)
        edge_weights = np.concatenate([spatial_weights, flow_weights])

        logger.info(f"Hybrid graph: {edge_index.shape[1]} total edges")

        return edge_index, edge_weights


class DynamicGraphBuilder:
    """Build dynamic graphs for temporal modeling"""

    def __init__(self, graph_builder: SpatialGraphBuilder, time_window: int = 24):
        """
        Initialize dynamic graph builder

        Args:
            graph_builder: Spatial graph builder
            time_window: Time window size for dynamic graphs
        """
        self.graph_builder = graph_builder
        self.time_window = time_window

    def build_temporal_graphs(self, od_df: pd.DataFrame,
                             num_time_steps: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build sequence of graphs over time

        Args:
            od_df: OD flow DataFrame with time_idx
            num_time_steps: Number of time steps

        Returns:
            List of (edge_index, edge_weights) for each time window
        """
        logger.info(f"Building temporal graphs with {num_time_steps} time steps")

        graphs = []

        for t in range(0, num_time_steps, self.time_window):
            # Get data for current time window
            window_end = min(t + self.time_window, num_time_steps)
            window_df = od_df[od_df['time_idx'].between(t, window_end - 1)]

            # Build graph for this window
            edge_index, edge_weights = self.graph_builder.build_hybrid_graph(window_df)

            graphs.append((edge_index, edge_weights))

        logger.info(f"Created {len(graphs)} temporal graphs")

        return graphs


def create_pyg_data(node_features: np.ndarray,
                   edge_index: np.ndarray,
                   edge_weights: np.ndarray,
                   labels: np.ndarray = None) -> Data:
    """
    Create PyTorch Geometric Data object

    Args:
        node_features: Node feature matrix (num_nodes, num_features)
        edge_index: Edge indices (2, num_edges)
        edge_weights: Edge weights (num_edges,)
        labels: Node labels (num_nodes,)

    Returns:
        PyTorch Geometric Data object
    """
    data = Data(
        x=torch.FloatTensor(node_features),
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_weights).unsqueeze(1)
    )

    if labels is not None:
        data.y = torch.LongTensor(labels)

    return data

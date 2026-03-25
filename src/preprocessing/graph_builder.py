"""
Graph construction utilities for spatial relationship modeling
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Optional
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
        self.coord_by_grid_id = {
            gid: (float(lon), float(lat))
            for gid, lon, lat in metadata_df[['grid_id', 'lon', 'lat']].itertuples(index=False, name=None)
        }
        self.node_coords = None
        self._refresh_node_coords()

    def _refresh_node_coords(self):
        """Refresh node coordinates to align with the current node index mapping."""
        self.node_coords = np.zeros((len(self.grid_id_to_idx), 2), dtype=np.float32)
        for idx, grid_id in self.idx_to_grid_id.items():
            lon, lat = self.coord_by_grid_id[grid_id]
            self.node_coords[idx] = (lon, lat)

    @staticmethod
    def _haversine_distance_km(src_coords: np.ndarray, dst_coords: np.ndarray) -> np.ndarray:
        """Compute great-circle distance in kilometers for batched lon/lat coordinates."""
        if src_coords.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)

        lon1 = np.radians(src_coords[:, 0])
        lat1 = np.radians(src_coords[:, 1])
        lon2 = np.radians(dst_coords[:, 0])
        lat2 = np.radians(dst_coords[:, 1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
        return (6371.0 * c).astype(np.float32)

    def build_edge_attr(
        self,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
        mode: str = 'flow_only'
    ) -> np.ndarray:
        """Build edge attributes from scalar weights and grid geometry."""
        edge_weights = edge_weights.astype(np.float32, copy=False)

        if mode == 'flow_only':
            return edge_weights

        if mode != 'flow_distance_direction':
            raise ValueError(f"Unsupported edge feature mode: {mode}")

        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return np.zeros((0, 4), dtype=np.float32)

        src_coords = self.node_coords[edge_index[0]]
        dst_coords = self.node_coords[edge_index[1]]

        distances_km = self._haversine_distance_km(src_coords, dst_coords)
        max_distance = float(distances_km.max()) if distances_km.size > 0 else 0.0
        if max_distance > 0.0:
            normalized_distance = distances_km / max_distance
        else:
            normalized_distance = np.zeros_like(distances_km, dtype=np.float32)

        delta_lon = dst_coords[:, 0] - src_coords[:, 0]
        delta_lat = dst_coords[:, 1] - src_coords[:, 1]
        direction_norm = np.sqrt(delta_lon ** 2 + delta_lat ** 2)
        safe_norm = np.where(direction_norm > 0.0, direction_norm, 1.0)
        direction_cos = (delta_lon / safe_norm).astype(np.float32)
        direction_sin = (delta_lat / safe_norm).astype(np.float32)
        direction_cos[direction_norm <= 0.0] = 0.0
        direction_sin[direction_norm <= 0.0] = 0.0

        return np.stack(
            [
                edge_weights,
                normalized_distance.astype(np.float32),
                direction_cos,
                direction_sin,
            ],
            axis=1,
        )

    @staticmethod
    def add_self_loops_intelligently(edge_index: np.ndarray, edge_attr: np.ndarray,
                                     num_nodes: int, weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        INTELLIGENT SELF-LOOP SUPPLEMENTATION

        CRITICAL FIX: Preserve real self-loop flows from OD data!
        - OD data contains REAL self-loop edges (o_grid == d_grid) with actual trip volumes
        - These are CRITICAL business features (e.g., 18.5M self-loop trips in 2021)
        - We MUST preserve these real flows, NOT replace them with constant 1.0

        Logic:
        1. Check each node: does it already have a self-loop edge in edge_index?
        2. If YES (node has real self-loop): PRESERVE it! Don't add another one.
        3. If NO (node lacks self-loop): ADD artificial self-loop with weight=1.0

        Args:
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge weights (num_edges,)
            num_nodes: Total number of nodes
            weight: Weight for artificial self-loops (default: 1.0)

        Returns:
            edge_index_with_loops: Edge indices with intelligently added self-loops
            edge_attr_with_loops: Edge weights with preserved real flows + artificial weights
        """
        logger.info(f"Intelligently adding self-loops (preserving real OD self-loop flows)")

        # Identify which nodes already have self-loops
        nodes_with_self_loops = set()
        if len(edge_index) > 0 and edge_index.shape[1] > 0:
            # Self-loop: source index == target index
            for i in range(edge_index.shape[1]):
                if edge_index[0, i] == edge_index[1, i]:
                    nodes_with_self_loops.add(edge_index[0, i])

        logger.info(f"  Nodes with real self-loops from OD data: {len(nodes_with_self_loops)}")

        # Identify nodes that need artificial self-loops
        all_nodes = set(range(num_nodes))
        nodes_needing_self_loops = all_nodes - nodes_with_self_loops

        logger.info(f"  Nodes needing artificial self-loops: {len(nodes_needing_self_loops)}")

        # Create artificial self-loops ONLY for nodes that don't have one
        if len(nodes_needing_self_loops) > 0:
            nodes_needing_array = np.array(sorted(nodes_needing_self_loops))
            artificial_self_loops = np.stack([nodes_needing_array, nodes_needing_array], axis=0)
            artificial_weights = np.full(len(nodes_needing_array), weight)

            # Concatenate with existing edges
            edge_index_with_loops = np.concatenate([edge_index, artificial_self_loops], axis=1)
            edge_attr_with_loops = np.concatenate([edge_attr, artificial_weights])
        else:
            # All nodes already have self-loops, no need to add
            edge_index_with_loops = edge_index
            edge_attr_with_loops = edge_attr

        logger.info(f"  Original edges: {edge_index.shape[1] if len(edge_index) > 0 else 0}")
        logger.info(f"  Final edges (with intelligent self-loops): {edge_index_with_loops.shape[1]}")
        logger.info(f"  Real self-loop flows: PRESERVED ({len(nodes_with_self_loops)} nodes)")
        logger.info(f"  Artificial self-loops: ADDED ({len(nodes_needing_self_loops)} nodes)")

        return edge_index_with_loops, edge_attr_with_loops

    @staticmethod
    def add_self_loops(edge_index: np.ndarray, edge_attr: np.ndarray,
                      num_nodes: int, weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        DEPRECATED: Use add_self_loops_intelligently() instead.

        Old method that blindly adds self-loops to ALL nodes,
        which can override real self-loop flows from OD data.

        Kept for backward compatibility. Will be removed in future versions.
        """
        logger.warning("add_self_loops() is DEPRECATED. Use add_self_loops_intelligently() instead.")
        return SpatialGraphBuilder.add_self_loops_intelligently(
            edge_index, edge_attr, num_nodes, weight
        )

    @staticmethod
    def compute_degrees(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        """
        Compute node degrees from edge_index

        Args:
            edge_index: Edge index array (2, num_edges)
            num_nodes: Total number of nodes

        Returns:
            degrees: Array of node degrees (num_nodes,)
        """
        if edge_index.shape[1] == 0:
            return np.zeros(num_nodes, dtype=int)

        # Count occurrences of each node in edge_index
        all_nodes = np.concatenate([edge_index[0], edge_index[1]])
        degrees = np.bincount(all_nodes, minlength=num_nodes)
        return degrees

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

    def build_flow_graph(self, od_df: pd.DataFrame, threshold: float = 0.0,
                        include_neighbors: bool = True,
                        topk_out: Optional[int] = None,
                        topk_in: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on OD flow patterns

        REFACTORING: Default threshold lowered to 0.0 to preserve ALL real connectivity.
        We now use GCN/GraphSAGE which can handle large sparse graphs efficiently via SpMM.
        No edge pruning for memory savings - we run the full sparse graph.

        CRITICAL FIX: Use raw num_total values instead of normalized values.
        Edge weights should represent actual connection strength (non-negative).

        NEW: If include_neighbors=True, include nodes connected to labeled grids even if they
        don't have labels themselves. This provides spatial context.

        Args:
            od_df: OD flow DataFrame
            threshold: Minimum raw flow to create edge (default 0.0 to include all positive flows)
            include_neighbors: If True, include neighbors of labeled grids (default: True)

        Returns:
            edge_index: Edge indices (2, num_edges)
            edge_weights: Edge weights based on flow volume (non-negative)
        """
        logger.info(f"Building flow graph with threshold={threshold} (using raw flow values, minimal pruning)")
        if include_neighbors:
            logger.info("Including neighbors of labeled grids for spatial context")

        # Use raw num_total instead of normalized values
        flow_column = 'num_total' if 'num_total' in od_df.columns else 'num_total_normalized'

        # Aggregate flows between grid pairs
        flow_agg = od_df.groupby(['o_grid_500', 'd_grid_500'])[flow_column].sum().reset_index()
        flow_agg = flow_agg[flow_agg[flow_column] > threshold]  # Only positive flows

        # Dynamic node expansion: track all nodes that should be in the graph
        expanded_grid_ids = set(self.grid_id_to_idx.keys())
        initial_node_count = len(expanded_grid_ids)

        # Convert grid IDs to indices
        edge_list = []
        edge_weights = []

        for _, row in flow_agg.iterrows():
            o_grid = row['o_grid_500']
            d_grid = row['d_grid_500']

            # NEW: Include edge if at least one endpoint is a labeled grid
            # This adds neighbors (unlabeled grids) to provide spatial context
            if include_neighbors:
                should_include = (o_grid in self.grid_id_to_idx) or (d_grid in self.grid_id_to_idx)
            else:
                should_include = (o_grid in self.grid_id_to_idx) and (d_grid in self.grid_id_to_idx)

            if should_include:
                # Add nodes to expanded set if not already present
                if o_grid not in expanded_grid_ids:
                    expanded_grid_ids.add(o_grid)
                if d_grid not in expanded_grid_ids:
                    expanded_grid_ids.add(d_grid)

                edge_list.append((o_grid, d_grid))
                edge_weights.append(row[flow_column])

        # Create new grid_id_to_idx mapping with expanded nodes
        # IMPORTANT: Preserve original indices for labeled grids
        new_grid_id_to_idx = {}
        new_idx_to_grid_id = {}
        next_idx = 0

        # First, add original labeled grids (preserve their indices if possible)
        for grid_id in self.grid_id_to_idx.keys():
            new_grid_id_to_idx[grid_id] = next_idx
            new_idx_to_grid_id[next_idx] = grid_id
            next_idx += 1

        # Then, add newly discovered neighbor grids
        newly_added = []
        for grid_id in expanded_grid_ids:
            if grid_id not in new_grid_id_to_idx:
                new_grid_id_to_idx[grid_id] = next_idx
                new_idx_to_grid_id[next_idx] = grid_id
                newly_added.append(grid_id)
                next_idx += 1

        # Update instance variables for future use
        self.grid_id_to_idx = new_grid_id_to_idx
        self.idx_to_grid_id = new_idx_to_grid_id
        self._refresh_node_coords()

        # Now convert edge list to indices
        final_edge_list = []
        final_edge_weights = []

        for (o_grid, d_grid), weight in zip(edge_list, edge_weights):
            o_idx = new_grid_id_to_idx[o_grid]
            d_idx = new_grid_id_to_idx[d_grid]
            final_edge_list.append([o_idx, d_idx])
            final_edge_weights.append(weight)

        edge_index = np.array(final_edge_list).T if final_edge_list else np.zeros((2, 0))
        edge_weights = np.array(final_edge_weights) if final_edge_weights else np.array([])

        # Optional node-wise sparsification: keep top-k outgoing/incoming edges per node.
        if topk_out is not None or topk_in is not None:
            edge_index, edge_weights = self._apply_nodewise_topk(
                edge_index=edge_index,
                edge_weights=edge_weights,
                topk_out=topk_out,
                topk_in=topk_in
            )
            # Keep list and array views consistent for downstream KNN fallback merge.
            final_edge_list = edge_index.T.tolist()
            final_edge_weights = edge_weights.tolist()

        logger.info(f"Created flow graph:")
        logger.info(f"  - Original labeled grids: {initial_node_count}")
        logger.info(f"  - Newly added neighbor grids: {len(newly_added)}")
        logger.info(f"  - Total nodes: {len(new_grid_id_to_idx)}")
        logger.info(f"  - Total edges: {edge_index.shape[1]}")

        # ====================================================================
        # GRAPH OPTIMIZATION: Intelligently supplement self-loops
        # ====================================================================
        import config

        num_nodes = len(new_grid_id_to_idx)

        # Phase 2.1: Intelligently add self-loops (PRESERVE REAL OD FLOWS!)
        if config.ADD_SELF_LOOPS:
            logger.info("Applying Phase 2.1: Intelligently adding self-loops (preserving real OD flows)")
            edge_index, edge_weights = self.add_self_loops_intelligently(
                edge_index, edge_weights, num_nodes, weight=config.SELF_LOOP_WEIGHT
            )
            final_edge_list = edge_index.T.tolist()
            final_edge_weights = edge_weights.tolist()

        # Phase 2.2: KNN fallback for isolated nodes
        if config.USE_KNN_FALLBACK:
            # Check for isolated nodes (after self-loop addition, if enabled)
            degrees = self.compute_degrees(edge_index, num_nodes)
            isolated_mask = (degrees == 0)
            isolated_indices = np.where(isolated_mask)[0]

            if len(isolated_indices) > 0:
                logger.info(f"Applying Phase 2.2: Found {len(isolated_indices)} isolated nodes, adding KNN fallback")

                # Get isolated grid IDs
                isolated_grid_ids = [new_idx_to_grid_id[idx] for idx in isolated_indices]

                # Build KNN edges for isolated nodes
                knn_edges = self._build_knn_for_isolated_nodes(
                    isolated_grid_ids, isolated_indices, new_grid_id_to_idx, config.KNN_FALLBACK_K
                )

                # Merge KNN edges
                for o_idx, d_idx, weight in knn_edges:
                    final_edge_list.append([o_idx, d_idx])
                    final_edge_weights.append(weight)

                # Rebuild edge_index and edge_weights with new edges
                if final_edge_list:
                    edge_index = np.array(final_edge_list).T
                    edge_weights = np.array(final_edge_weights)

                    logger.info(f"  Added {len(knn_edges)} KNN fallback edges")
                else:
                    logger.warning("  No KNN edges were added despite isolated nodes")
            else:
                logger.info("Phase 2.2: No isolated nodes found (KNN fallback not needed)")

        return edge_index, edge_weights

    @staticmethod
    def _topk_indices_by_group(groups: np.ndarray, weights: np.ndarray, k: Optional[int]) -> np.ndarray:
        """Return original edge indices for top-k edges within each group."""
        n = groups.shape[0]
        if n == 0:
            return np.array([], dtype=np.int64)

        if k is None or k <= 0:
            return np.arange(n, dtype=np.int64)

        # Sort by group asc, weight desc
        order = np.lexsort((-weights, groups))
        sorted_groups = groups[order]

        keep_mask_sorted = np.zeros(n, dtype=bool)
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_groups)) + 1]
        ends = np.r_[starts[1:], n]

        for start, end in zip(starts, ends):
            keep_mask_sorted[start:min(start + k, end)] = True

        return order[keep_mask_sorted]

    def _apply_nodewise_topk(
        self,
        edge_index: np.ndarray,
        edge_weights: np.ndarray,
        topk_out: Optional[int],
        topk_in: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply node-wise top-k filtering and keep union of outgoing/incoming selections.
        Real self-loops are always preserved.
        """
        if edge_index.shape[1] == 0:
            return edge_index, edge_weights

        src = edge_index[0]
        dst = edge_index[1]
        ranking_weights = edge_weights
        if ranking_weights.ndim > 1:
            ranking_weights = ranking_weights[:, 0]
        weights = np.asarray(ranking_weights, dtype=np.float64)

        keep_out = self._topk_indices_by_group(src, weights, topk_out)
        keep_in = self._topk_indices_by_group(dst, weights, topk_in)

        keep_idx = np.union1d(keep_out, keep_in)

        # Preserve self-loops from OD data regardless of top-k cut.
        self_loop_idx = np.where(src == dst)[0]
        if self_loop_idx.size > 0:
            keep_idx = np.union1d(keep_idx, self_loop_idx)

        before_edges = edge_index.shape[1]
        edge_index = edge_index[:, keep_idx]
        edge_weights = edge_weights[keep_idx]

        logger.info(
            f"Applied node-wise top-k filtering: out={topk_out}, in={topk_in} | "
            f"edges {before_edges} -> {edge_index.shape[1]}"
        )

        return edge_index, edge_weights

    def _build_knn_for_isolated_nodes(
        self,
        isolated_grid_ids: List[int],
        isolated_indices: np.ndarray,
        grid_id_to_idx: Dict[int, int],
        k: int = 8
    ) -> List[Tuple[int, int, float]]:
        """
        Build KNN edges for isolated nodes using spatial proximity

        Args:
            isolated_grid_ids: List of grid IDs that are isolated
            isolated_indices: List of node indices that are isolated
            grid_id_to_idx: Mapping from grid_id to node index
            k: Number of nearest neighbors to find

        Returns:
            knn_edges: List of (source_idx, target_idx, weight) tuples
        """
        logger.info(f"  Building KNN edges for {len(isolated_grid_ids)} isolated nodes (k={k})")

        # Extract coordinates for all grids
        all_coords = self.metadata_df[['lon', 'lat']].values
        all_grid_ids = self.metadata_df['grid_id'].values

        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(all_coords)

        knn_edges = []

        for grid_id, idx in zip(isolated_grid_ids, isolated_indices):
            # Get grid coordinates
            grid_row = self.metadata_df[self.metadata_df['grid_id'] == grid_id]
            if len(grid_row) == 0:
                continue

            coord = grid_row[['lon', 'lat']].values[0]

            # Find k+1 nearest neighbors (includes self)
            distances, indices = tree.query(coord, k=k+1)

            # Skip first result (self) and create edges
            for neighbor_idx, dist in zip(indices[1:], distances[1:]):
                neighbor_grid_id = all_grid_ids[neighbor_idx]

                # Only add edge if neighbor is in our graph
                if neighbor_grid_id in grid_id_to_idx:
                    neighbor_node_idx = grid_id_to_idx[neighbor_grid_id]

                    # Weight: inverse distance (closer = stronger)
                    # Apply KNN_FALLBACK_WEIGHT multiplier
                    import config
                    weight = (1.0 / (dist + 1e-6)) * config.KNN_FALLBACK_WEIGHT

                    knn_edges.append((idx, neighbor_node_idx, weight))

        logger.info(f"  Created {len(knn_edges)} KNN edges for isolated nodes")

        return knn_edges

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

    def build_daily_graphs(self, od_df: pd.DataFrame, num_days: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build sequence of daily graphs (one graph per day)
        NEW: For dual-year model with daily snapshots

        Args:
            od_df: OD flow DataFrame with date_dt column
            num_days: Number of days (default 7)

        Returns:
            List of (edge_index, edge_weights) for each day
        """
        logger.info(f"Building daily graphs for {num_days} days")

        graphs = []

        # Get date range
        min_date = od_df['date_dt'].min()

        for day_idx in range(num_days):
            # Get data for current day
            current_date = min_date + pd.Timedelta(days=day_idx)
            next_date = current_date + pd.Timedelta(days=1)

            day_df = od_df[
                (od_df['date_dt'] >= current_date) &
                (od_df['date_dt'] < next_date)
            ]

            if len(day_df) > 0:
                # Build graph for this day
                edge_index, edge_weights = self.graph_builder.build_hybrid_graph(day_df)
            else:
                # If no data for this day, use empty graph
                logger.warning(f"No data for day {day_idx}, using empty graph")
                edge_index = np.zeros((2, 0))
                edge_weights = np.array([])

            graphs.append((edge_index, edge_weights))

        logger.info(f"Created {len(graphs)} daily graphs")

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

"""
Dataset class for mobility pattern classification
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MobilityDataset(Dataset):
    """Dataset for mobility pattern classification"""

    def __init__(self,
                 grid_flows: Dict[int, np.ndarray],
                 labels: Dict[int, int],
                 grid_ids: List[int],
                 metadata_df=None):
        """
        Initialize dataset

        Args:
            grid_flows: Dictionary mapping grid_id to temporal flow array (time_steps, 2)
            labels: Dictionary mapping grid_id to label (0-8)
            grid_ids: List of grid IDs to include in dataset
            metadata_df: Grid metadata DataFrame (optional)
        """
        self.grid_flows = grid_flows
        self.labels = labels
        self.grid_ids = [gid for gid in grid_ids if gid in labels]
        self.metadata_df = metadata_df

        logger.info(f"Created dataset with {len(self.grid_ids)} samples")

    def __len__(self):
        return len(self.grid_ids)

    def __getitem__(self, idx):
        """
        Get item by index

        Returns:
            temporal_features: Temporal flow data (seq_len, 2)
            spatial_features: Temporal flow for spatial branch (seq_len, 2) - NOT flattened
            label: Class label (0-8)
            grid_id: Grid ID
        """
        grid_id = self.grid_ids[idx]

        # Get temporal features
        temporal_features = self.grid_flows[grid_id]  # (seq_len, 2)

        # CRITICAL FIX: Keep spatial features as 2D for DySAT temporal processing
        # Do NOT flatten - DySAT needs (time_steps, features) to use temporal attention
        spatial_features = temporal_features  # (seq_len, 2)

        # Get label
        label = self.labels[grid_id]

        return {
            'temporal': torch.FloatTensor(temporal_features),
            'spatial': torch.FloatTensor(spatial_features),
            'label': torch.LongTensor([label])[0],
            'grid_id': grid_id
        }


class GraphBatchCollator:
    """Custom collator for batching graph data"""

    def __init__(self, edge_index, edge_attr=None, grid_id_to_idx=None, all_spatial_features=None):
        """
        Initialize collator

        Args:
            edge_index: Graph edge indices (2, num_edges)
            edge_attr: Edge attributes (num_edges, edge_dim)
            grid_id_to_idx: Mapping from grid_id to node index
            all_spatial_features: All spatial features for the full graph (num_nodes, feature_dim)
        """
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_attr = torch.FloatTensor(edge_attr) if edge_attr is not None else None
        self.grid_id_to_idx = grid_id_to_idx
        self.all_spatial_features = all_spatial_features

    def __call__(self, batch):
        """
        Collate batch with k-hop subgraph extraction

        Args:
            batch: List of samples from dataset

        Returns:
            Batched data dictionary with subgraph information
        """
        temporal_batch = torch.stack([item['temporal'] for item in batch])
        spatial_batch = torch.stack([item['spatial'] for item in batch])
        label_batch = torch.stack([item['label'] for item in batch])
        grid_ids = [item['grid_id'] for item in batch]

        # Get node indices for this batch
        if self.grid_id_to_idx is not None:
            node_indices = torch.LongTensor([self.grid_id_to_idx[gid] for gid in grid_ids])
        else:
            node_indices = None

        # Extract k-hop subgraph for batch nodes (MEMORY OPTIMIZATION)
        if node_indices is not None and self.all_spatial_features is not None:
            # Extract 1-hop subgraph to preserve local graph structure while minimizing memory
            # Using 1-hop instead of 2-hop for better memory efficiency
            subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
                node_idx=node_indices,
                num_hops=1,  # Reduced from 2 to 1 for better memory efficiency
                edge_index=self.edge_index,
                relabel_nodes=True,
                num_nodes=self.all_spatial_features.size(0)
            )

            # CRITICAL: If subgraph is too large, use only batch nodes (no neighbors)
            # This prevents OOM on densely connected regions
            MAX_SUBGRAPH_NODES = 500  # Limit to 500 nodes max
            if len(subset) > MAX_SUBGRAPH_NODES:
                # Use only batch nodes, no neighbors
                subset = node_indices
                # Create edges only between batch nodes
                batch_set = set(node_indices.tolist())
                mask = torch.tensor([
                    (self.edge_index[0, i].item() in batch_set and
                     self.edge_index[1, i].item() in batch_set)
                    for i in range(self.edge_index.size(1))
                ])
                edge_index_sub = self.edge_index[:, mask]
                # Relabel nodes to 0, 1, 2, ...
                old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
                edge_index_sub = torch.tensor([
                    [old_to_new[edge_index_sub[0, i].item()], old_to_new[edge_index_sub[1, i].item()]]
                    for i in range(edge_index_sub.size(1))
                ]).t()
                edge_mask = mask
                mapping = torch.arange(len(node_indices))
                logger.warning(f"Subgraph too large ({len(subset)} nodes), using batch nodes only")
                subset = node_indices

            # Extract spatial features for subgraph nodes only
            # This reduces memory from (num_nodes, time_steps, features) to (num_subgraph_nodes, time_steps, features)
            spatial_features_sub = self.all_spatial_features[subset]

            # Get edge attributes for subgraph
            if self.edge_attr is not None:
                edge_attr_sub = self.edge_attr[edge_mask]
            else:
                edge_attr_sub = None

            # Batch node indices in subgraph (first len(batch) nodes after relabeling)
            batch_indices_in_subgraph = mapping[:len(node_indices)]

            # Log subgraph size for monitoring (always log for first few batches)
            num_edges = edge_index_sub.size(1) if edge_index_sub.numel() > 0 else 0
            logger.info(f"Subgraph extraction: {len(subset)} nodes (from {self.all_spatial_features.size(0)}), "
                       f"{num_edges} edges (from {self.edge_index.size(1)}), "
                       f"batch_size={len(node_indices)}")

            return {
                'temporal': temporal_batch,  # (batch_size, time_steps, features)
                'spatial': spatial_batch,    # (batch_size, time_steps, features) - NOT flattened
                'labels': label_batch,
                'grid_ids': grid_ids,
                'node_indices': batch_indices_in_subgraph,  # Indices within subgraph
                'edge_index': edge_index_sub,  # Subgraph edges
                'edge_attr': edge_attr_sub,    # Subgraph edge attributes
                'all_spatial_features': spatial_features_sub  # Subgraph features only (MEMORY OPTIMIZED)
            }
        else:
            # Fallback to original behavior (no subgraph extraction)
            return {
                'temporal': temporal_batch,  # (batch_size, time_steps, features)
                'spatial': spatial_batch,    # (batch_size, time_steps, features) - NOT flattened
                'labels': label_batch,
                'grid_ids': grid_ids,
                'node_indices': node_indices,
                'edge_index': self.edge_index,
                'edge_attr': self.edge_attr,
                'all_spatial_features': self.all_spatial_features  # (num_nodes, time_steps, features)
            }


class ImprovedDualYearDataset(Dataset):
    """Dataset for improved dual-year mobility pattern classification"""

    def __init__(self,
                 change_features: Dict[int, np.ndarray],
                 labels: Dict[int, int],
                 grid_ids: List[int],
                 metadata_df=None):
        """
        Initialize improved dual-year dataset

        Args:
            change_features: Dictionary mapping grid_id to change features (7, 4)
                            [2021_total_log, 2024_total_log, 2021_net_flow_log, 2024_net_flow_log]
            labels: Dictionary mapping grid_id to label (0-8)
            grid_ids: List of grid IDs to include in dataset
            metadata_df: Grid metadata DataFrame (optional)
        """
        self.change_features = change_features
        self.labels = labels
        self.grid_ids = [gid for gid in grid_ids if gid in labels]
        self.metadata_df = metadata_df

        logger.info(f"Created improved dual-year dataset with {len(self.grid_ids)} samples")

    def __len__(self):
        return len(self.grid_ids)

    def __getitem__(self, idx):
        """
        Get item by index

        Returns:
            x_2021: Features for 2021 (7, 2) - [total_log, net_flow_log]
            x_2024: Features for 2024 (7, 2) - [total_log, net_flow_log]
            label: Class label (0-8)
            grid_id: Grid ID
        """
        grid_id = self.grid_ids[idx]

        # Get change features (7, 4)
        features = self.change_features[grid_id]

        # Split into 2021 and 2024 features
        x_2021 = features[:, [0, 2]]  # (7, 2) - [total_log, net_flow_log]
        x_2024 = features[:, [1, 3]]  # (7, 2) - [total_log, net_flow_log]

        # Get label
        label = self.labels[grid_id]

        return {
            'x_2021': torch.FloatTensor(x_2021),
            'x_2024': torch.FloatTensor(x_2024),
            'label': torch.LongTensor([label])[0],
            'grid_id': grid_id
        }


class ImprovedGraphBatchCollator:
    """Custom collator for batching improved dual-year graph data"""

    def __init__(self,
                 graphs_2021: List[Tuple[np.ndarray, np.ndarray]],
                 graphs_2024: List[Tuple[np.ndarray, np.ndarray]],
                 grid_id_to_idx: Dict[int, int],
                 all_features_2021: np.ndarray,
                 all_features_2024: np.ndarray):
        """
        Initialize improved collator

        Args:
            graphs_2021: List of 7 (edge_index, edge_attr) tuples for 2021
            graphs_2024: List of 7 (edge_index, edge_attr) tuples for 2024
            grid_id_to_idx: Mapping from grid_id to node index
            all_features_2021: All features for 2021 (num_nodes, 7, 2)
            all_features_2024: All features for 2024 (num_nodes, 7, 2)
        """
        # Convert graphs to tensors
        self.graphs_2021 = [
            (torch.LongTensor(edge_index), torch.FloatTensor(edge_attr))
            for edge_index, edge_attr in graphs_2021
        ]
        self.graphs_2024 = [
            (torch.LongTensor(edge_index), torch.FloatTensor(edge_attr))
            for edge_index, edge_attr in graphs_2024
        ]

        self.grid_id_to_idx = grid_id_to_idx
        self.all_features_2021 = torch.FloatTensor(all_features_2021)
        self.all_features_2024 = torch.FloatTensor(all_features_2024)

    def __call__(self, batch):
        """
        Collate batch with dynamic graphs

        Args:
            batch: List of samples from dataset

        Returns:
            Batched data dictionary with dynamic graph information
        """
        x_2021_batch = torch.stack([item['x_2021'] for item in batch])
        x_2024_batch = torch.stack([item['x_2024'] for item in batch])
        label_batch = torch.stack([item['label'] for item in batch])
        grid_ids = [item['grid_id'] for item in batch]

        # Get node indices for this batch
        node_indices = torch.LongTensor([self.grid_id_to_idx[gid] for gid in grid_ids])

        # For improved model, we pass full graph features and let model handle extraction
        # This is simpler and allows spatial branch to see full graph context
        # NOTE: Graphs stay on CPU here, will be moved to device in train loop

        return {
            'x_2021': x_2021_batch,  # (batch_size, 7, 2)
            'x_2024': x_2024_batch,  # (batch_size, 7, 2)
            'labels': label_batch,
            'grid_ids': grid_ids,
            'node_indices': node_indices,
            'graphs_2021': self.graphs_2021,  # List of 7 graphs (on CPU)
            'graphs_2024': self.graphs_2024,  # List of 7 graphs (on CPU)
            'all_features_2021': self.all_features_2021,  # (num_nodes, 7, 2)
            'all_features_2024': self.all_features_2024   # (num_nodes, 7, 2)
        }

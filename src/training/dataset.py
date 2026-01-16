"""
Dataset class for mobility pattern classification
"""
import torch
from torch.utils.data import Dataset
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
            spatial_features: Flattened temporal flow for spatial branch (seq_len * 2,)
            label: Class label (0-8)
            grid_id: Grid ID
        """
        grid_id = self.grid_ids[idx]

        # Get temporal features
        temporal_features = self.grid_flows[grid_id]  # (seq_len, 2)

        # Create spatial features (flattened temporal features)
        spatial_features = temporal_features.flatten()  # (seq_len * 2,)

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
        Collate batch

        Args:
            batch: List of samples from dataset

        Returns:
            Batched data dictionary
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

        return {
            'temporal': temporal_batch,
            'spatial': spatial_batch,
            'labels': label_batch,
            'grid_ids': grid_ids,
            'node_indices': node_indices,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'all_spatial_features': self.all_spatial_features
        }

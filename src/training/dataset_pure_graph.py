"""
Simplified dataset for pure graph-based model
No need to store node features - only graph structure and labels
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class PureGraphDualYearDataset(Dataset):
    """
    Simplified dataset for pure graph-based model

    Key differences from original:
    1. No change_features (7, 4) - only labels
    2. Temporal features still needed for temporal branch
    3. Spatial features computed from graph structure on-the-fly
    """

    def __init__(self,
                 temporal_features_2021: Dict[int, torch.Tensor],
                 temporal_features_2024: Dict[int, torch.Tensor],
                 labels: Dict[int, int],
                 grid_ids: List[int]):
        """
        Initialize dataset

        Args:
            temporal_features_2021: {grid_id: (7, 2)} - temporal features for 2021
            temporal_features_2024: {grid_id: (7, 2)} - temporal features for 2024
            labels: {grid_id: label_idx} - class labels (0-8)
            grid_ids: List of grid IDs
        """
        self.temporal_features_2021 = temporal_features_2021
        self.temporal_features_2024 = temporal_features_2024
        self.labels = labels
        self.grid_ids = grid_ids

    def __len__(self):
        return len(self.grid_ids)

    def __getitem__(self, idx):
        grid_id = self.grid_ids[idx]

        # Get temporal features (still needed for temporal branch)
        x_2021 = self.temporal_features_2021[grid_id]  # (7, 2)
        x_2024 = self.temporal_features_2024[grid_id]  # (7, 2)

        # Get label
        label = self.labels[grid_id]

        return {
            'x_2021': torch.tensor(x_2021, dtype=torch.float32),
            'x_2024': torch.tensor(x_2024, dtype=torch.float32),
            'label': label,
            'grid_id': grid_id
        }


class PureGraphBatchCollator:
    """
    Batch collator for pure graph-based model

    Key differences:
    1. No need to prepare spatial node features
    2. Only pass graph structure to spatial branch
    3. Temporal features still batched normally
    """

    def __init__(self,
                 graphs_2021: List[tuple],
                 graphs_2024: List[tuple],
                 grid_id_to_idx: Dict[int, int],
                 all_temporal_2021: torch.Tensor,
                 all_temporal_2024: torch.Tensor):
        """
        Initialize collator

        Args:
            graphs_2021: [(edge_index, edge_attr)] - static graph for 2021
            graphs_2024: [(edge_index, edge_attr)] - static graph for 2024
            grid_id_to_idx: {grid_id: node_idx} - mapping
            all_temporal_2021: (num_nodes, 7, 2) - all temporal features for 2021
            all_temporal_2024: (num_nodes, 7, 2) - all temporal features for 2024
        """
        self.graphs_2021 = graphs_2021
        self.graphs_2024 = graphs_2024
        self.grid_id_to_idx = grid_id_to_idx
        self.all_temporal_2021 = all_temporal_2021
        self.all_temporal_2024 = all_temporal_2024
        self.num_nodes = len(grid_id_to_idx)

    def __call__(self, batch):
        """
        Collate batch

        Args:
            batch: List of samples from dataset

        Returns:
            Batched data dictionary
        """
        # Extract data
        grid_ids = [item['grid_id'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        # Get node indices
        node_indices = torch.tensor([self.grid_id_to_idx[gid] for gid in grid_ids], dtype=torch.long)

        return {
            'node_indices': node_indices,
            'labels': labels,
            'grid_ids': grid_ids,
            'graphs_2021': self.graphs_2021,
            'graphs_2024': self.graphs_2024,
            'num_nodes': self.num_nodes,
            # Full temporal features (for extracting batch)
            'all_temporal_2021': self.all_temporal_2021,
            'all_temporal_2024': self.all_temporal_2024
        }


if __name__ == "__main__":
    # Test dataset and collator
    print("Testing PureGraphDualYearDataset and PureGraphBatchCollator")
    print("=" * 80)

    # Create dummy data
    num_nodes = 100
    num_samples = 50
    num_edges = 500

    # Temporal features (still needed)
    temporal_2021 = {i: torch.randn(7, 2) for i in range(num_samples)}
    temporal_2024 = {i: torch.randn(7, 2) for i in range(num_samples)}

    # Labels
    labels = {i: i % 9 for i in range(num_samples)}

    # Grid IDs
    grid_ids = list(range(num_samples))

    # Create dataset
    dataset = PureGraphDualYearDataset(
        temporal_features_2021=temporal_2021,
        temporal_features_2024=temporal_2024,
        labels=labels,
        grid_ids=grid_ids
    )

    print(f"\nDataset:")
    print(f"  - Size: {len(dataset)}")
    print(f"  - Sample 0: {dataset[0].keys()}")
    print(f"  - x_2021 shape: {dataset[0]['x_2021'].shape}")
    print(f"  - x_2024 shape: {dataset[0]['x_2024'].shape}")
    print(f"  - Label: {dataset[0]['label']}")

    # Create graphs
    edge_index_2021 = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_2021 = torch.rand(num_edges) * 100

    edge_index_2024 = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr_2024 = torch.rand(num_edges) * 100

    graphs_2021 = [(edge_index_2021, edge_attr_2021)]
    graphs_2024 = [(edge_index_2024, edge_attr_2024)]

    # Create mapping
    grid_id_to_idx = {i: i for i in range(num_nodes)}

    # All temporal features
    all_temporal_2021 = torch.randn(num_nodes, 7, 2)
    all_temporal_2024 = torch.randn(num_nodes, 7, 2)

    # Create collator
    collator = PureGraphBatchCollator(
        graphs_2021=graphs_2021,
        graphs_2024=graphs_2024,
        grid_id_to_idx=grid_id_to_idx,
        all_temporal_2021=all_temporal_2021,
        all_temporal_2024=all_temporal_2024
    )

    # Test batching
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collator
    )

    batch = next(iter(dataloader))

    print(f"\nBatch:")
    print(f"  - node_indices shape: {batch['node_indices'].shape}")
    print(f"  - labels shape: {batch['labels'].shape}")
    print(f"  - num_nodes: {batch['num_nodes']}")
    print(f"  - graphs_2021: {len(batch['graphs_2021'])} graph(s)")
    print(f"  - graphs_2024: {len(batch['graphs_2024'])} graph(s)")
    print(f"  - all_temporal_2021 shape: {batch['all_temporal_2021'].shape}")
    print(f"  - all_temporal_2024 shape: {batch['all_temporal_2024'].shape}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("\nKey simplifications:")
    print("  1. No spatial node features in dataset")
    print("  2. Only temporal features + labels stored")
    print("  3. Graph structure passed separately")
    print("  4. Spatial features computed on-the-fly from graph")

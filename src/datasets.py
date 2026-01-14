import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import load_numpyz, setup_logger, class_weights_from_counts


logger = setup_logger(__name__)


class NodeChangeDataset(Dataset):
    """Per-node dataset containing 2021/2024 time series and labels.

    Graph sequences are used in the model separately and are not part of each sample.

    Args:
        processed_dir: Path to processed data directory
        indices: Sample indices for this dataset split
    """

    def __init__(self, processed_dir: str, indices: np.ndarray) -> None:
        processed_dir = os.path.abspath(processed_dir)
        # Load time series (remapped: 2018->2021, 2021->2024)
        X21 = load_numpyz(os.path.join(processed_dir, "time_series_2021.npz"))["X"]
        X24 = load_numpyz(os.path.join(processed_dir, "time_series_2024.npz"))["X"]
        assert X21.shape == X24.shape
        self.X21 = torch.from_numpy(X21)  # [N, 168, 2]
        self.X24 = torch.from_numpy(X24)

        # Load labels and mapping
        labels_df = pd.read_csv(os.path.join(processed_dir, "labels_filtered.csv"))
        # Map labels 1..9 -> 0..8
        y = labels_df["label"].values.astype(np.int64) - 1
        self.y = torch.from_numpy(y)

        # Map each label grid to graph idx
        with open(os.path.join(processed_dir, "graph_node_mapping.json"), "r") as f:
            graph_map = json.load(f)
        grid_to_idx_raw = graph_map["grid_to_idx"]
        grid_to_idx = {int(k): int(v) for k, v in grid_to_idx_raw.items()}
        grid_ids = labels_df["grid_id"].astype(int).tolist()
        gidx = [grid_to_idx.get(int(g), -1) for g in grid_ids]
        if any(v < 0 for v in gidx):
            missing = sum(1 for v in gidx if v < 0)
            raise RuntimeError(f"Found {missing} labeled grids not present in graph mapping. Ensure preprocess consistency.")
        self.graph_idx = torch.tensor(gidx, dtype=torch.long)

        # Indices subset for split
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return self.indices.numel()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        x21 = self.X21[idx].clone()
        x24 = self.X24[idx].clone()

        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "gidx": self.graph_idx[idx],
            "x21": x21,
            "x24": x24,
            "y": self.y[idx],
        }


def train_val_test_split(N: int, ratios=(0.7, 0.15, 0.15), seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_dataloaders(processed_dir: str, batch_size: int = 256, seed: int = 42,
                     ratios=(0.7, 0.15, 0.15)):
    """Create dataloaders."""
    X21 = load_numpyz(os.path.join(processed_dir, "time_series_2021.npz"))["X"]
    N = X21.shape[0]

    labels_df = pd.read_csv(os.path.join(processed_dir, "labels_filtered.csv"))
    y = labels_df["label"].values.astype(np.int64) - 1

    train_idx, val_idx, test_idx = train_val_test_split(N, ratios, seed)

    ds_train = NodeChangeDataset(processed_dir, train_idx)
    ds_val = NodeChangeDataset(processed_dir, val_idx)
    ds_test = NodeChangeDataset(processed_dir, test_idx)

    # Class weights for sampler
    counts = np.bincount(y[train_idx], minlength=9)
    class_weights = class_weights_from_counts(counts).numpy()
    sample_weights = class_weights[y[train_idx]]
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(train_idx), replacement=True)

    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return ds_train, ds_val, ds_test, dl_train, dl_val, dl_test

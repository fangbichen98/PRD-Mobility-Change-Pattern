"""
GINE-based Spatial Branch: Graph Isomorphism Network with Edge Features

Key improvements over GCN:
1. GINE (Graph Isomorphism Network with Edge features) - theoretically most powerful GNN
2. Deep edge feature processing via MLP (not just linear weighting)
3. Laplacian Positional Encoding - gives nodes spatial coordinates
4. Same memory footprint as GCN (no attention computation)

Architecture:
- Input: Laplacian PE features (10-16 dim) instead of all-1
- Edge processing: MLP transforms edge weights before aggregation
- Node update: MLP combines neighbor features
- Output: Node embeddings for 2021, 2024, and their difference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PureGraphDualYearGINE(nn.Module):
    """
    Pure graph-based spatial branch using GINE with Laplacian PE

    Advantages over GCN:
    1. Stronger expressiveness: GIN has maximum discriminative power among MPNNs
    2. Edge-aware: MLP processes edge features (OD flows) non-linearly
    3. Position-aware: Laplacian PE distinguishes city center vs suburbs
    4. Memory efficient: No attention computation, similar to GCN
    """

    def __init__(self,
                 input_size: int = 16,  # Laplacian PE dimension
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 output_size: int = 256,
                 edge_dim: int = 1):  # Edge feature dimension (flow weight)
        """
        Initialize GINE-based dual-year spatial branch

        Args:
            input_size: Input feature dimension (Laplacian PE)
            hidden_size: Hidden feature dimension
            num_layers: Number of GINE layers
            dropout: Dropout rate
            output_size: Output feature size
            edge_dim: Edge feature dimension (1 for flow weight)
        """
        super(PureGraphDualYearGINE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        self.edge_dim = edge_dim

        # Cache for storing computed graph embeddings
        self._cached_2021 = None
        self._cached_2024 = None
        self._cache_valid = False

        # GINE layers
        self.gine_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                # First layer: Laplacian PE -> hidden_size
                in_channels = input_size
            else:
                # Subsequent layers: hidden_size -> hidden_size
                in_channels = hidden_size

            # MLP for node feature transformation
            node_mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

            # GINE layer with edge features
            self.gine_layers.append(
                GINEConv(
                    nn=node_mlp,
                    edge_dim=edge_dim,  # Process edge weights via MLP
                    train_eps=True  # Learn epsilon parameter
                )
            )

        # Batch normalization for each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
        ])

        # Output projection: hidden_size -> output_size
        self.output_proj = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu

    def process_year(self, edge_index, edge_attr, node_features):
        """
        Process one year's graph with GINE

        Args:
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge weights (num_edges, 1) - OD flow values
            node_features: Node features (num_nodes, input_size) - Laplacian PE

        Returns:
            h_out: Node embeddings (num_nodes, output_size)
        """
        # Log-transform edge weights for numerical stability
        # GINE's MLP can handle this, but log transform helps with extreme values
        edge_weights = torch.log1p(edge_attr.float())

        # Ensure edge_index is int64
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        h = node_features

        # Apply GINE layers with batch normalization
        for i, (gine_layer, bn) in enumerate(zip(self.gine_layers, self.batch_norms)):
            h = gine_layer(h, edge_index, edge_attr=edge_weights)
            h = bn(h)
            h = self.act(h)
            h = self.dropout(h)

        # Project to output size
        h_out = self.output_proj(h)

        return h_out

    def clear_cache(self):
        """Clear cached embeddings"""
        self._cached_2021 = None
        self._cached_2024 = None
        self._cache_valid = False

    def forward(self,
                graphs_2021,
                graphs_2024,
                node_features_2021,
                node_features_2024,
                node_indices: Optional[torch.Tensor] = None):
        """
        Forward pass for both years with Laplacian PE

        Args:
            graphs_2021: List with single (edge_index, edge_attr) tuple for 2021
            graphs_2024: List with single (edge_index, edge_attr) tuple for 2024
            node_features_2021: Laplacian PE for 2021 (num_nodes, input_size)
            node_features_2024: Laplacian PE for 2024 (num_nodes, input_size)
            node_indices: Optional node indices for batch extraction

        Returns:
            Tuple of (h_2021, h_2024, h_diff) each of shape (batch_size, output_size)
        """
        # Extract graph data
        edge_index_2021, edge_attr_2021 = graphs_2021[0]
        edge_index_2024, edge_attr_2024 = graphs_2024[0]

        # Process 2021 graph
        if self.training or self._cached_2021 is None:
            h_2021_full = self.process_year(
                edge_index_2021,
                edge_attr_2021,
                node_features_2021
            )
            if not self.training:
                self._cached_2021 = h_2021_full
        else:
            h_2021_full = self._cached_2021

        # Process 2024 graph
        if self.training or self._cached_2024 is None:
            h_2024_full = self.process_year(
                edge_index_2024,
                edge_attr_2024,
                node_features_2024
            )
            if not self.training:
                self._cached_2024 = h_2024_full
        else:
            h_2024_full = self._cached_2024

        # Extract batch nodes if indices provided
        if node_indices is not None:
            h_2021 = h_2021_full[node_indices]
            h_2024 = h_2024_full[node_indices]
        else:
            h_2021 = h_2021_full
            h_2024 = h_2024_full

        # Compute difference
        h_diff = h_2024 - h_2021

        return h_2021, h_2024, h_diff


def compute_laplacian_pe(edge_index, edge_weight, num_nodes, k=16, device='cuda'):
    """
    Compute Laplacian Positional Encoding for graph nodes

    Args:
        edge_index: Edge connectivity (2, num_edges)
        edge_weight: Edge weights (num_edges,)
        num_nodes: Number of nodes
        k: Number of eigenvectors to use (PE dimension)
        device: Device to compute on

    Returns:
        pe: Laplacian PE (num_nodes, k)
    """
    try:
        from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
        from scipy.sparse.linalg import eigsh
        import numpy as np

        # Get normalized Laplacian
        edge_index_lap, edge_weight_lap = get_laplacian(
            edge_index,
            edge_weight,
            normalization='sym',
            num_nodes=num_nodes
        )

        # Convert to scipy sparse matrix
        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)

        # Compute k smallest eigenvectors (excluding the trivial one)
        # Use k+1 because the first eigenvector is constant (trivial)
        try:
            eigenvalues, eigenvectors = eigsh(L, k=min(k+1, num_nodes-2), which='SM')
        except:
            # Fallback: if eigsh fails, use random features
            logger.warning(f"Laplacian eigsh failed, using random PE")
            pe = torch.randn(num_nodes, k, device=device)
            return pe

        # Take k eigenvectors (skip the first trivial one)
        pe = torch.from_numpy(eigenvectors[:, 1:k+1]).float().to(device)

        # Handle case where we got fewer eigenvectors than requested
        if pe.shape[1] < k:
            # Pad with zeros
            padding = torch.zeros(num_nodes, k - pe.shape[1], device=device)
            pe = torch.cat([pe, padding], dim=1)

        logger.info(f"Computed Laplacian PE: {pe.shape}")
        return pe

    except Exception as e:
        logger.error(f"Error computing Laplacian PE: {e}")
        # Fallback to random features
        pe = torch.randn(num_nodes, k, device=device)
        return pe


# Test the module
if __name__ == "__main__":
    # Test with dummy data
    num_nodes = 1000
    num_edges = 5000
    batch_size = 32
    pe_dim = 16

    # Create dummy graph
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.rand(num_edges, 1) * 100  # Flow weights

    # Create dummy Laplacian PE
    node_features = torch.randn(num_nodes, pe_dim)

    # Create model
    model = PureGraphDualYearGINE(
        input_size=pe_dim,
        hidden_size=128,
        num_layers=3,
        dropout=0.2,
        output_size=256
    )

    # Test forward pass
    graphs_2021 = [(edge_index, edge_attr)]
    graphs_2024 = [(edge_index, edge_attr)]
    node_indices = torch.randint(0, num_nodes, (batch_size,))

    h_2021, h_2024, h_diff = model(
        graphs_2021,
        graphs_2024,
        node_features,
        node_features,
        node_indices
    )

    print(f"Output shapes:")
    print(f"  h_2021: {h_2021.shape}")
    print(f"  h_2024: {h_2024.shape}")
    print(f"  h_diff: {h_diff.shape}")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

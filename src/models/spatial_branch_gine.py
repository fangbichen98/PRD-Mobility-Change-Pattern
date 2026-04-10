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

    def _normalize_edge_attr(self, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Normalize edge features to shape (num_edges, edge_dim) for GINEConv.

        Some graph builders may output edge_attr as (E,), (1, E), or (E, 1).
        GINEConv requires (E, edge_dim).
        """
        num_edges = edge_index.size(1)

        if edge_attr is None:
            return torch.ones(num_edges, self.edge_dim, device=edge_index.device)

        edge_attr = edge_attr.float()

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        elif edge_attr.dim() == 2 and edge_attr.size(0) == 1 and edge_attr.size(1) == num_edges:
            # Convert (1, E) -> (E, 1)
            edge_attr = edge_attr.t().contiguous()
        elif edge_attr.dim() > 2:
            # Flatten unexpected high-rank shapes while preserving edge count
            edge_attr = edge_attr.view(num_edges, -1)

        # Ensure first dimension matches number of edges
        if edge_attr.size(0) != num_edges:
            edge_attr = edge_attr.view(num_edges, -1)

        # Match configured edge_dim
        if edge_attr.size(1) < self.edge_dim:
            pad = torch.zeros(num_edges, self.edge_dim - edge_attr.size(1), device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, pad], dim=1)
        elif edge_attr.size(1) > self.edge_dim:
            edge_attr = edge_attr[:, :self.edge_dim]

        return edge_attr

    def _run_gine_layers(self, edge_index, edge_attr, node_features):
        """Run GINE message-passing layers and return hidden representation before output_proj.

        Args:
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge weights (num_edges, edge_dim)
            node_features: Node features (num_nodes, input_size)

        Returns:
            h: Node hidden embeddings (num_nodes, hidden_size=128)
        """
        # Normalize edge features. Only flow channel (index 0) is log-transformed.
        edge_features = self._normalize_edge_attr(edge_attr, edge_index)
        edge_features = edge_features.clone()
        edge_features[:, 0] = torch.log1p(torch.clamp(edge_features[:, 0], min=0.0))

        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        h = node_features
        for gine_layer, bn in zip(self.gine_layers, self.batch_norms):
            h = gine_layer(h, edge_index, edge_attr=edge_features)
            h = bn(h)
            h = self.act(h)
            h = self.dropout(h)

        return h  # (num_nodes, hidden_size)

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
        return self.output_proj(self._run_gine_layers(edge_index, edge_attr, node_features))

    def clear_cache(self):
        """Clear cached embeddings.
        Note: cache now stores 128-dim hidden (from _run_gine_layers),
        not the 256-dim projected output as in earlier versions.
        """
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

        # Run GINE layers for 2021, cache 128-dim hidden (not 256-dim output).
        # Caching hidden saves half the memory vs caching output and is equivalent
        # since output_proj is deterministic at inference.
        if self.training or self._cached_2021 is None:
            h_2021_hidden = self._run_gine_layers(
                edge_index_2021, edge_attr_2021, node_features_2021
            )
            if not self.training:
                self._cached_2021 = h_2021_hidden
        else:
            h_2021_hidden = self._cached_2021

        # Run GINE layers for 2024
        if self.training or self._cached_2024 is None:
            h_2024_hidden = self._run_gine_layers(
                edge_index_2024, edge_attr_2024, node_features_2024
            )
            if not self.training:
                self._cached_2024 = h_2024_hidden
        else:
            h_2024_hidden = self._cached_2024

        # Extract batch nodes (still at 128-dim hidden level)
        if node_indices is not None:
            h_2021_batch = h_2021_hidden[node_indices]
            h_2024_batch = h_2024_hidden[node_indices]
        else:
            h_2021_batch = h_2021_hidden
            h_2024_batch = h_2024_hidden

        # Project to output_size. Compute diff at 128-dim before projection so
        # the linear map acts on the true hidden-space change signal, not on the
        # difference of two nearly-identical 256-dim projected outputs.
        h_2021 = self.output_proj(h_2021_batch)
        h_2024 = self.output_proj(h_2024_batch)
        h_diff = self.output_proj(h_2024_batch - h_2021_batch)

        return h_2021, h_2024, h_diff


def compute_laplacian_pe(edge_index, edge_weight, num_nodes, k=16, device='cuda'):
    """
    Compute Laplacian Positional Encoding for graph nodes.

    Uses shift-invert mode (sigma=1e-10) which is numerically stable for large
    sparse graphs, unlike the default which='SM' mode.

    Args:
        edge_index: Edge connectivity (2, num_edges)
        edge_weight: Edge weights (num_edges,)
        num_nodes: Number of nodes
        k: Number of eigenvectors to use (PE dimension); 0 = return empty tensor
        device: Device to compute on

    Returns:
        pe: Laplacian PE (num_nodes, k)
    """
    # k=0 means PE is disabled — return empty tensor directly
    if k == 0:
        return torch.zeros(num_nodes, 0, device=device)

    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        from torch_geometric.utils import get_laplacian

        # Extract scalar flow weight: if edge_weight is 2-D (e.g. (E, 4) for
        # flow_distance_direction mode), use only the first column (flow).
        ew = edge_weight.float()
        if ew.dim() > 1:
            ew = ew[:, 0]
        ew = ew.reshape(-1)  # ensure 1-D
        ew = torch.nan_to_num(ew, nan=0.0, posinf=1.0, neginf=0.0)
        ew = ew.clamp(min=0.0)

        # Build normalized Laplacian via PyG
        ei_lap, ew_lap = get_laplacian(
            edge_index, ew, normalization='sym', num_nodes=num_nodes
        )

        # Convert to scipy CSR sparse matrix (compatible with all PyG versions)
        row = ei_lap[0].cpu().numpy()
        col = ei_lap[1].cpu().numpy()
        val = ew_lap.cpu().numpy().astype(np.float64)
        L = sp.csr_matrix((val, (row, col)), shape=(num_nodes, num_nodes))

        k_req = min(k + 1, num_nodes - 2)

        # Attempt 1: shift-invert mode — stable and fast for small eigenvalues
        try:
            eigenvalues, eigenvectors = eigsh(
                L, k=k_req, which='LM', sigma=1e-10,
                tol=1e-4, maxiter=num_nodes * 5
            )
        except Exception as e1:
            logger.warning(f"Laplacian eigsh (shift-invert) failed: {e1}. Trying SM mode.")
            # Attempt 2: standard SM mode as fallback
            try:
                eigenvalues, eigenvectors = eigsh(
                    L, k=k_req, which='SM', tol=1e-4
                )
            except Exception as e2:
                logger.warning(f"Laplacian eigsh (SM) also failed: {e2}. Using random PE.")
                return torch.randn(num_nodes, k, device=device)

        # Sort by ascending eigenvalue and skip the trivial near-zero eigenvector
        sort_idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, sort_idx]
        pe_np = eigenvectors[:, 1:k + 1].copy()  # skip index 0 (trivial)

        pe = torch.from_numpy(pe_np).float().to(device)

        # Pad with zeros if fewer eigenvectors were returned than requested
        if pe.shape[1] < k:
            padding = torch.zeros(num_nodes, k - pe.shape[1], device=device)
            pe = torch.cat([pe, padding], dim=1)

        logger.info(f"Computed Laplacian PE: {pe.shape}")
        return pe

    except Exception as e:
        logger.error(f"Error computing Laplacian PE: {e}. Using random PE.")
        return torch.randn(num_nodes, k, device=device)


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

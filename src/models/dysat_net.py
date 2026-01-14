from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class DySATNet(nn.Module):
    """Simplified DySAT-like encoder with structural (GATv2) and temporal attention.

    - Uses a learnable node embedding table as input features.
    - For each snapshot, applies 2 layers of GATv2Conv with edge attributes (weight as 1-dim attr).
    - Applies temporal self-attention across the 24 hourly snapshots, then averages across time.

    Returns per-node embeddings for the last structural layer, and also exposes per-layer
    time-averaged features (to support differencing across periods).
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 32,
        hid_dim: int = 64,
        out_dim: int = 64,
        heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.emb = nn.Embedding(num_nodes, in_dim)

        self.gat1 = GATv2Conv(in_dim, hid_dim, heads=heads, edge_dim=1, dropout=dropout)
        self.act1 = nn.ELU()
        self.gat2 = GATv2Conv(hid_dim * heads, out_dim, heads=heads, edge_dim=1, dropout=dropout)
        self.act2 = nn.ELU()

        self.temporal_attn = nn.MultiheadAttention(embed_dim=out_dim * heads, num_heads=heads, dropout=dropout, batch_first=False)

    def forward(self, snapshots: List[Data]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward over a list of 24 snapshots.

        Args:
            snapshots: List[Data], each with edge_index [2,E], edge_attr [E,1]

        Returns:
            z: [N, F] final per-node embedding (time-averaged after temporal attention)
            s1: [N, F1] time-averaged features after first GAT layer
            s2: [N, F2] time-averaged features after second GAT layer (before temporal attention)
        """
        assert len(snapshots) > 0, "Snapshots list is empty"
        N = self.num_nodes

        x0 = self.emb.weight  # [N, in_dim]

        # Structural attention per snapshot
        H1: List[torch.Tensor] = []
        H2: List[torch.Tensor] = []
        device = x0.device
        for g in snapshots:
            edge_index = g.edge_index.to(device)
            edge_attr = g.edge_attr.to(device) if hasattr(g, 'edge_attr') and g.edge_attr is not None else None
            h1 = self.gat1(x0, edge_index, edge_attr=edge_attr)
            h1 = self.act1(h1)
            h2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
            h2 = self.act2(h2)
            H1.append(h1)
            H2.append(h2)

        # Stack over time: [T, N, F]
        H1t = torch.stack(H1, dim=0)
        H2t = torch.stack(H2, dim=0)

        # Temporal attention on H2 (final structural)
        # MultiheadAttention expects [T, B, F], treat nodes as batch B=N
        Q = K = V = H2t  # [T, N, F]
        attn_out, _ = self.temporal_attn(Q, K, V)  # [T, N, F]
        # Average across time to get per-node embedding
        z = attn_out.mean(dim=0)  # [N, F]

        # Also average S1 and S2 across time
        s1 = H1t.mean(dim=0)  # [N, F1]
        s2 = H2t.mean(dim=0)  # [N, F2]
        return z, s1, s2

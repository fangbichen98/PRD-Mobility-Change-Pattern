from typing import List, Tuple, Dict

import torch
from torch import nn
from torch_geometric.data import Data

from .lstm_net import LSTMNet
from .spp_net import SPPNet
from .dysat_net import DySATNet


class SpatioTemporalChangeNet(nn.Module):
    """Full model: LSTM + SPP temporal branches, DySAT spatial branch with cross-period differencing.

    Fusion strategy: concat [ΔS1, ΔS2, ΔT3, ΔT4] -> MLP classifier (9 classes).
    """

    def __init__(
        self,
        num_nodes: int,
        ts_in_dim: int = 2,
        ts_hidden: int = 64,
        spp_channels: int = 64,
        gnn_in: int = 32,
        gnn_hid: int = 64,
        gnn_out: int = 64,
        gnn_heads: int = 2,
        dropout: float = 0.2,
        num_classes: int = 9,
    ) -> None:
        super().__init__()
        # Temporal encoders
        self.enc_lstm = LSTMNet(in_dim=ts_in_dim, hidden_dim=ts_hidden, num_layers=2, dropout=dropout)
        self.enc_spp = SPPNet(in_dim=ts_in_dim, channels=spp_channels, hidden_dim=ts_hidden, dropout=dropout)

        # Spatial encoder (shared across periods)
        self.gnn = DySATNet(num_nodes=num_nodes, in_dim=gnn_in, hid_dim=gnn_hid, out_dim=gnn_out, heads=gnn_heads, dropout=dropout)

        s1_dim = gnn_hid * gnn_heads
        s2_dim = gnn_out * gnn_heads
        z_dim = gnn_out * gnn_heads
        t_dim = ts_hidden

        # Fuse spatial deltas (including z) and temporal deltas
        # Note: z is the temporally-attended structural embedding from DySAT.
        fusion_in = s1_dim + s2_dim + z_dim + t_dim + t_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    @torch.no_grad()
    def compute_spatial_deltas(self, graphs18: List[Data], graphs21: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute ΔS1 and ΔS2 for all nodes to avoid redundant GNN passes per batch.
        Compute with GNN in eval mode to avoid dropout-induced variance during training.
        """
        was_training = self.gnn.training
        self.gnn.train(False)
        try:
            z18, s1_18, s2_18 = self.gnn(graphs18)
            z21, s1_21, s2_21 = self.gnn(graphs21)
        finally:
            self.gnn.train(was_training)
        dS1 = s1_21 - s1_18  # [N, F1]
        dS2 = s2_21 - s2_18  # [N, F2]
        return dS1, dS2

    def forward(
        self,
        batch_x18: torch.Tensor,
        batch_x21: torch.Tensor,
        batch_node_idx: torch.Tensor,
        graphs18: List[Data],
        graphs21: List[Data],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Temporal branches (per-node sequences in the batch)
        t3_18 = self.enc_lstm(batch_x18)  # [B, H]
        t3_21 = self.enc_lstm(batch_x21)
        dT3 = t3_21 - t3_18

        t4_18 = self.enc_spp(batch_x18)   # [B, H]
        t4_21 = self.enc_spp(batch_x21)
        dT4 = t4_21 - t4_18

        # Spatial branch (global computation); then index to batch nodes
        z18, s1_18, s2_18 = self.gnn(graphs18)
        z21, s1_21, s2_21 = self.gnn(graphs21)
        dS1 = s1_21 - s1_18
        dS2 = s2_21 - s2_18
        dZ = z21 - z18

        # Index per batch nodes
        dS1_b = dS1[batch_node_idx]
        dS2_b = dS2[batch_node_idx]
        dZ_b = dZ[batch_node_idx]

        fused = torch.cat([dS1_b, dS2_b, dZ_b, dT3, dT4], dim=-1)
        logits = self.classifier(fused)

        aux = {
            "dS1": dS1_b,
            "dS2": dS2_b,
            "dZ": dZ_b,
            "dT3": dT3,
            "dT4": dT4,
        }
        return logits, aux

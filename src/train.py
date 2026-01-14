import os
from typing import Dict, List, Tuple
import time
import csv as _csv

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from utils import setup_logger, load_yaml, set_seed, device_from_config, ensure_dir
from datasets import make_dataloaders
from models.fusion_classifier import SpatioTemporalChangeNet


logger = setup_logger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        alpha: Weighting factor in range (0, 1) to balance positive/negative examples
        gamma: Focusing parameter for modulating loss. gamma=0 recovers CE loss
        reduction: 'mean' or 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, C] raw logits
            targets: [B] class indices in [0, C-1]
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def load_graphs(processed_dir: str) -> Tuple[List[Data], List[Data]]:
    # Remapped to 2021/2024
    g21 = torch.load(os.path.join(processed_dir, "graphs_2021.pt"))
    g24 = torch.load(os.path.join(processed_dir, "graphs_2024.pt"))
    return g21, g24


def train_one_epoch(
    model: SpatioTemporalChangeNet,
    dl: DataLoader,
    graphs21: List[Data],
    graphs24: List[Data],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    # Precompute ΔS1, ΔS2, ΔZ for all nodes once this epoch (on device)
    # Note: this keeps forward path consistent with classifier input dims.
    with torch.no_grad():
        was = model.gnn.training
        model.gnn.train(False)
        try:
            z21, s1_21, s2_21 = model.gnn(graphs21)
            z24, s1_24, s2_24 = model.gnn(graphs24)
        finally:
            model.gnn.train(was)
    dS1_all = (s1_24 - s1_21).to(device)
    dS2_all = (s2_24 - s2_21).to(device)
    dZ_all  = (z24  - z21 ).to(device)

    for batch in dl:
        idx = batch["idx"].to(device)
        gidx = batch["gidx"].to(device)
        x21 = batch["x21"].to(device)
        x24 = batch["x24"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            # Temporal encoders
            dT3 = model.enc_lstm(x24) - model.enc_lstm(x21)
            dT4 = model.enc_spp(x24) - model.enc_spp(x21)

            # Gather spatial deltas for batch nodes (including ΔZ)
            dS1 = dS1_all[gidx]
            dS2 = dS2_all[gidx]
            dZ  = dZ_all[gidx]

            fused = torch.cat([dS1, dS2, dZ, dT3, dT4], dim=-1)
            logits = model.classifier(fused)
            loss = criterion(logits, y)

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss) * y.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == y).sum().item())
        total += y.size(0)

    return {"loss": total_loss / total, "acc": total_correct / total}


@torch.no_grad()
def evaluate(
    model: SpatioTemporalChangeNet,
    dl: DataLoader,
    graphs21: List[Data],
    graphs24: List[Data],
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    # Precompute ΔS1, ΔS2, ΔZ to match classifier input
    with torch.no_grad():
        was = model.gnn.training
        model.gnn.train(False)
        try:
            z21, s1_21, s2_21 = model.gnn(graphs21)
            z24, s1_24, s2_24 = model.gnn(graphs24)
        finally:
            model.gnn.train(was)
    dS1_all = (s1_24 - s1_21).to(device)
    dS2_all = (s2_24 - s2_21).to(device)
    dZ_all  = (z24  - z21 ).to(device)

    # Unweighted CE for validation loss to avoid skew from class weights
    criterion_unweighted = nn.CrossEntropyLoss()

    for batch in dl:
        idx = batch["idx"].to(device)
        gidx = batch["gidx"].to(device)
        x21 = batch["x21"].to(device)
        x24 = batch["x24"].to(device)
        y = batch["y"].to(device)

        with torch.cuda.amp.autocast(enabled=amp):
            dT3 = model.enc_lstm(x24) - model.enc_lstm(x21)
            dT4 = model.enc_spp(x24) - model.enc_spp(x21)
            dS1 = dS1_all[gidx]
            dS2 = dS2_all[gidx]
            dZ  = dZ_all[gidx]
            fused = torch.cat([dS1, dS2, dZ, dT3, dT4], dim=-1)
            logits = model.classifier(fused)
            loss = criterion_unweighted(logits, y)

        total_loss += float(loss) * y.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == y).sum().item())
        total += y.size(0)

    return {"loss": total_loss / total, "acc": total_correct / total}


def _setup_file_logger(outputs_dir: str):
    """Attach a file handler to the module logger to write to outputs_dir/train.log."""
    import logging
    log_path = os.path.join(outputs_dir, "train.log")
    # Avoid duplicate handlers if called twice
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_path):
            return
    fh = logging.FileHandler(log_path, mode="a")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def run_train(config_path: str, processed_dir: str, outputs_dir: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg.get("seed", 42))
    device = device_from_config(cfg)
    ensure_dir(outputs_dir)
    _setup_file_logger(outputs_dir)

    # Data
    ds_train, ds_val, ds_test, dl_train, dl_val, dl_test = make_dataloaders(
        processed_dir=processed_dir,
        batch_size=cfg.get("batch_size", 256),
        seed=cfg.get("seed", 42),
        ratios=tuple(cfg.get("split", [0.7, 0.15, 0.15])),
    )

    # Graphs
    g21, g24 = load_graphs(processed_dir)
    # Move graphs to device is not needed; PyG ops will use tensors' devices

    # Graph GNN num_nodes comes from graph mapping
    import json as _json
    with open(os.path.join(processed_dir, "graph_node_mapping.json"), "r") as f:
        num_nodes = int(_json.load(f)["num_nodes"])

    # Model
    model = SpatioTemporalChangeNet(
        num_nodes=num_nodes,
        ts_in_dim=2,
        ts_hidden=cfg.get("ts_hidden", 64),
        spp_channels=cfg.get("spp_channels", 64),
        gnn_in=cfg.get("gnn_in", 32),
        gnn_hid=cfg.get("gnn_hid", 64),
        gnn_out=cfg.get("gnn_out", 64),
        gnn_heads=cfg.get("gnn_heads", 2),
        dropout=cfg.get("dropout", 0.2),
        num_classes=9,
    ).to(device)

    # Loss and optimizer
    loss_type = cfg.get("loss", "ce")  # 'ce' for CrossEntropy, 'focal' for Focal Loss

    if loss_type == "focal":
        # Use Focal Loss for handling class imbalance
        focal_alpha = cfg.get("focal_alpha", 0.25)
        focal_gamma = cfg.get("focal_gamma", 2.0)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        logger.info(f"Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    else:
        # Default: CrossEntropyLoss with class weights
        y_train = ds_train.y[ds_train.indices.numpy()].numpy()
        counts = np.bincount(y_train, minlength=9).astype(np.float32)
        weights = torch.tensor((1.0 / counts) * (len(counts) / (1e-6 + (1.0 / counts).sum())), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.get("label_smoothing", 0.0))
        logger.info(f"Using CrossEntropyLoss with class weights")

    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 5e-4))

    # Scheduler
    sched_type = cfg.get("scheduler", "plateau")
    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.get("epochs", 20))
    elif sched_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=cfg.get("lr_patience", 3))
    elif sched_type == "step":
        scheduler = StepLR(optimizer, step_size=cfg.get("lr_step", 5), gamma=cfg.get("lr_gamma", 0.5))
    else:
        scheduler = None

    # AMP & Early stopping
    amp = bool(cfg.get("amp", True)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    early_stop_enabled = bool(cfg.get("early_stop", True))
    patience = int(cfg.get("early_stop_patience", 5))
    no_improve = 0

    # Training loop
    best_val = 0.0
    best_path = os.path.join(outputs_dir, "best_model.pt")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    # CSV writer for per-epoch metrics
    metrics_csv = os.path.join(outputs_dir, "epoch_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])  # header

    for epoch in range(1, cfg.get("epochs", 20) + 1):
        t0 = time.time()
        tr = train_one_epoch(model, dl_train, g21, g24, optimizer, criterion, device, scaler, amp)
        va = evaluate(model, dl_val, g21, g24, criterion, device, amp)
        history["train_loss"].append(tr["loss"]) ; history["train_acc"].append(tr["acc"]) 
        history["val_loss"].append(va["loss"]) ; history["val_acc"].append(va["acc"]) 
        # Current LR (read from first param group)
        cur_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float('nan')
        secs = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d} | lr {cur_lr:.6g} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | val loss {va['loss']:.4f} acc {va['acc']:.4f} | {secs:.1f}s"
        )
        with open(metrics_csv, "a", newline="") as f:
            w = _csv.writer(f)
            w.writerow([epoch, cur_lr, tr['loss'], tr['acc'], va['loss'], va['acc'], secs])
        if va["acc"] > best_val:
            best_val = va["acc"]
            torch.save({"model": model.state_dict(), "config": cfg}, best_path)
            no_improve = 0
        else:
            no_improve += 1

        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(va["loss"])  # minimize validation loss
            else:
                scheduler.step()

        if early_stop_enabled and no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save training history
    torch.save(history, os.path.join(outputs_dir, "train_history.pt"))
    logger.info(f"Training finished. Best val acc={best_val:.4f}. Model saved to {best_path}")

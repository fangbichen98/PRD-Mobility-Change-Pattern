import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from utils import setup_logger, ensure_dir
import logging
from datasets import NodeChangeDataset
from models.fusion_classifier import SpatioTemporalChangeNet
from train import load_graphs


logger = setup_logger(__name__)


@torch.no_grad()
def infer_all(
    model: SpatioTemporalChangeNet,
    ds: NodeChangeDataset,
    graphs21,
    graphs24,
    device: torch.device,
    batch_size: int = 512,
):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Precompute ΔS and ΔZ on device to match classifier fusion
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

    logits_list = []
    y_list = []
    idx_list = []
    feats = {"dS1": [], "dS2": [], "dZ": [], "dT3": [], "dT4": []}

    for batch in loader:
        idx = batch["idx"].to(device)
        gidx = batch["gidx"].to(device)
        x21 = batch["x21"].to(device)
        x24 = batch["x24"].to(device)
        y = batch["y"].to(device)

        dT3 = model.enc_lstm(x24) - model.enc_lstm(x21)
        dT4 = model.enc_spp(x24) - model.enc_spp(x21)
        dS1 = dS1_all[gidx]
        dS2 = dS2_all[gidx]
        dZ  = dZ_all[gidx]
        fused = torch.cat([dS1, dS2, dZ, dT3, dT4], dim=-1)
        logits = model.classifier(fused)

        logits_list.append(logits.cpu())
        y_list.append(y.cpu())
        idx_list.append(idx.cpu())
        feats["dS1"].append(dS1.cpu())
        feats["dS2"].append(dS2.cpu())
        feats["dZ"].append(dZ.cpu())
        feats["dT3"].append(dT3.cpu())
        feats["dT4"].append(dT4.cpu())

    logits = torch.cat(logits_list, dim=0)
    y_true = torch.cat(y_list, dim=0)
    idx_all = torch.cat(idx_list, dim=0)
    for k in feats:
        feats[k] = torch.cat(feats[k], dim=0)
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    return idx_all.numpy(), y_true.numpy(), preds.numpy(), probs.numpy(), {k: v.numpy() for k, v in feats.items()}


def plot_training_curves(history_path: str, out_path: str) -> None:
    hist = torch.load(history_path)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["val_acc"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _setup_file_logger(outputs_dir: str):
    log_path = os.path.join(outputs_dir, "eval.log")
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_path):
            return
    fh = logging.FileHandler(log_path, mode="a")
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def run_evaluate(processed_dir: str, outputs_dir: str, best_model_path: str) -> None:
    ensure_dir(outputs_dir)
    _setup_file_logger(outputs_dir)
    # Dataset over all labeled nodes (time series subset)
    label_idx_path = os.path.join(processed_dir, "label_idx_to_grid.npy")
    if os.path.exists(label_idx_path):
        N_labels = np.load(label_idx_path).shape[0]
    else:
        # Fallback to time series file
        N_labels = np.load(os.path.join(processed_dir, "time_series_2021.npz"))['X'].shape[0]
    full_idx = np.arange(N_labels)
    ds = NodeChangeDataset(processed_dir, full_idx)

    # Load graphs
    g21, g24 = load_graphs(processed_dir)

    # Build model
    ckpt = torch.load(best_model_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_nodes for GNN is graph mapping size
    import json as _json
    with open(os.path.join(processed_dir, "graph_node_mapping.json"), "r") as f:
        num_nodes = int(_json.load(f)["num_nodes"])
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
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    idx, y_true, y_pred, probs, feats = infer_all(model, ds, g21, g24, device)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    logger.info(f"Test accuracy={acc:.4f} macro-F1={f1:.4f}")
    with open(os.path.join(outputs_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(outputs_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Save per-grid predictions
    idx_to_grid = np.load(os.path.join(processed_dir, "label_idx_to_grid.npy"))
    pred_df = pd.DataFrame({
        "grid_id": idx_to_grid[idx],
        "pred_label": y_pred + 1,  # back to 1..9
    })
    for k, v in feats.items():
        # Save as separate npz
        np.savez_compressed(os.path.join(outputs_dir, f"features_{k}.npz"), X=v, idx=idx)
    np.savez_compressed(os.path.join(outputs_dir, "probs.npz"), probs=probs, idx=idx)
    pred_df.to_csv(os.path.join(outputs_dir, "predictions.csv"), index=False)

    # Plot training curves if available
    hist_path = os.path.join(outputs_dir, "train_history.pt")
    if os.path.exists(hist_path):
        plot_training_curves(hist_path, os.path.join(outputs_dir, "training_curves.png"))

    # Geo visualization: scatter by lon/lat colored by predicted label
    labels_f = pd.read_csv(os.path.join(processed_dir, "labels_filtered.csv"))
    pred_geo = pred_df.merge(labels_f[["grid_id", "lon", "lat"]], on="grid_id", how="left")
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    sc = ax.scatter(pred_geo["lon"], pred_geo["lat"], c=pred_geo["pred_label"], s=6, cmap="tab10")
    plt.colorbar(sc, label="Pred Label")
    plt.title("Predicted Change Types (Scatter)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # Keep geographic proportion between lon/lat degrees: 1 deg lon ~= cos(lat) deg lat
    mean_lat = np.nanmean(pred_geo["lat"].values)
    aspect = 1.0 / max(1e-6, np.cos(np.deg2rad(mean_lat)))  # y/x data unit ratio
    ax.set_aspect(aspect, adjustable="box")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "predictions_scatter.png"), dpi=200)
    plt.close()

    # Optional: Predict for ALL graph-intersection nodes if full time series are available
    try:
        ts_g21_p = os.path.join(processed_dir, "time_series_graph_2021.npz")
        ts_g24_p = os.path.join(processed_dir, "time_series_graph_2024.npz")
        meta_p = os.path.join(processed_dir, "graph_nodes_lonlat.csv")
        idx2grid_p = os.path.join(processed_dir, "graph_idx_to_grid.npy")
        if os.path.exists(ts_g21_p) and os.path.exists(ts_g24_p) and os.path.exists(idx2grid_p):
            X21g = np.load(ts_g21_p)["X"]  # [Ng, 168, 2]
            X24g = np.load(ts_g24_p)["X"]
            Ng = X21g.shape[0]

            class _GraphAllDS(Dataset):
                def __init__(self, X21, X24):
                    self.X21 = X21
                    self.X24 = X24
                def __len__(self):
                    return self.X21.shape[0]
                def __getitem__(self, i: int):
                    return {
                        "idx": torch.tensor(i, dtype=torch.long),
                        "gidx": torch.tensor(i, dtype=torch.long),
                        "x21": torch.from_numpy(self.X21[i]),
                        "x24": torch.from_numpy(self.X24[i]),
                    }

            ds_all = _GraphAllDS(X21g, X24g)
            loader = DataLoader(ds_all, batch_size=1024, shuffle=False)

            model.eval()
            dS1_all, dS2_all = model.compute_spatial_deltas(g21, g24)
            dS1_all = dS1_all.to(device)
            dS2_all = dS2_all.to(device)

            logits_list = []
            idx_list = []
            with torch.no_grad():
                for batch in loader:
                    idxb = batch["idx"].to(device)
                    gidx = batch["gidx"].to(device)
                    x21 = batch["x21"].to(device)
                    x24 = batch["x24"].to(device)
                    # temporal deltas
                    dT3 = model.enc_lstm(x24) - model.enc_lstm(x21)
                    dT4 = model.enc_spp(x24) - model.enc_spp(x21)
                    # spatial deltas (index)
                    dS1b = dS1_all[gidx]
                    dS2b = dS2_all[gidx]
                    fused = torch.cat([dS1b, dS2b, dT3, dT4], dim=-1)
                    logits = model.classifier(fused)
                    logits_list.append(logits.cpu())
                    idx_list.append(idxb.cpu())

            logits_all = torch.cat(logits_list, dim=0)
            idx_all = torch.cat(idx_list, dim=0).numpy()
            probs_all = torch.softmax(logits_all, dim=-1).cpu().numpy()
            preds_all = logits_all.argmax(dim=-1).cpu().numpy()

            graph_idx_to_grid = np.load(idx2grid_p)
            pred_all_df = pd.DataFrame({
                "grid_id": graph_idx_to_grid[idx_all],
                "pred_label": preds_all + 1,
            })
            # Attach lon/lat if available before saving CSV
            if os.path.exists(meta_p):
                meta = pd.read_csv(meta_p)
                pred_all_df = pred_all_df.merge(meta[["grid_id", "lon", "lat"]], on="grid_id", how="left")
                # reorder columns
                pred_all_df = pred_all_df[["grid_id", "lon", "lat", "pred_label"]]
            # Save all-node predictions
            np.savez_compressed(os.path.join(outputs_dir, "probs_all.npz"), probs=probs_all, idx=idx_all)
            pred_all_df.to_csv(os.path.join(outputs_dir, "predictions_all.csv"), index=False)

            # Geo scatter for all nodes (if lon/lat available)
            if os.path.exists(meta_p):
                pred_geo_all = pred_all_df  # already has lon/lat
                plt.figure(figsize=(6, 6))
                ax = plt.gca()
                sc = ax.scatter(pred_geo_all["lon"], pred_geo_all["lat"], c=pred_geo_all["pred_label"], s=3, cmap="tab10")
                plt.colorbar(sc, label="Pred Label")
                plt.title("Predicted Change Types (All Graph Nodes)")
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                mean_lat = np.nanmean(pred_geo_all["lat"].values)
                aspect = 1.0 / max(1e-6, np.cos(np.deg2rad(mean_lat)))
                ax.set_aspect(aspect, adjustable="box")
                plt.tight_layout()
                plt.savefig(os.path.join(outputs_dir, "predictions_scatter_all.png"), dpi=200)
                plt.close()
            else:
                logger.warning("graph_nodes_lonlat.csv not found; skipped all-node scatter plot.")
        else:
            logger.info("Full-graph time series not found; skipped all-node prediction.")
    except Exception as e:
        logger.warning(f"All-node prediction failed: {e}")

    # Class counts bar chart
    counts = pred_df["pred_label"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Predicted Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "class_distribution.png"), dpi=150)
    plt.close()

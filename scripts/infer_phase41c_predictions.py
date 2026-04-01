"""
Inference script for phase41c: load best_model.pth, run on test split,
output per-sample CSV with columns:
  grid_id, true_label, pred_label, correct, confidence, prob_1..prob_9
"""
import os, sys, json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── phase41c paths ────────────────────────────────────────────────────────────
RUN_DIR    = "outputs/multiscale_temporal_20260328_231856_sampled_labels_spc250_seed202_reconstructed_sgh_phase41c_layers3_topk20_p30_e300"
MODEL_PATH = f"{RUN_DIR}/models/best_model.pth"
SPLIT_PATH = f"{RUN_DIR}/metrics/split_manifest.json"
LABEL_PATH = "data/sampled_labels_spc250_seed202_reconstructed.csv"
OUT_CSV    = "output_analysis/phase41c_test_predictions.csv"

# ── override config to match phase41c ────────────────────────────────────────
import config
config.SPATIAL_MODEL          = "GINE"
config.TEMPORAL_MODEL         = "TRANSFORMER"
config.SPATIAL_LAYERS         = 3
config.SPATIAL_HIDDEN_SIZE    = 128
config.LAPLACIAN_PE_DIM       = 16          # phase41c used 16
config.GINE_EDGE_FEATURE_MODE = "flow_distance_direction"
config.SPATIAL_NODE_FEATURE_MODE = "raw_temporal_mean"
config.GINE_USE_SPATIAL_COORDS   = True
config.TEMPORAL_FEATURE_MODE  = "inflow_outflow"
config.GRAPH_TOPK_OUT         = 20
config.GRAPH_TOPK_IN          = 20
config.GRAPH_TEMPORAL_MODE    = "static"
config.FUSION_HIDDEN_SIZE     = 256
config.LSTM_DROPOUT           = 0.4
config.TEMPORAL_INPUT_SIZE    = 2
config.BATCH_SIZE             = 12

from src.preprocessing.dual_year_processor import prepare_dual_year_experiment_data
from src.models.enhanced_dual_branch_model import EnhancedDualBranchModel
from src.training.dataset_pure_graph import PureGraphDualYearDataset, PureGraphBatchCollator

def main():
    os.makedirs("output_analysis", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    data = prepare_dual_year_experiment_data(
        label_path=LABEL_PATH,
        use_cache=True,
        spatial_model="GINE",
        edge_feature_mode="flow_distance_direction"
    )
    num_nodes = len(data["grid_id_to_idx"])

    # temporal features
    temporal_2021, temporal_2024 = {}, {}
    for gid, feat in data["change_features"].items():
        temporal_2021[gid] = feat[:, [0, 1]]
        temporal_2024[gid] = feat[:, [2, 3]]

    grid_ids = list(data["labels"].keys())
    dataset  = PureGraphDualYearDataset(
        temporal_features_2021=temporal_2021,
        temporal_features_2024=temporal_2024,
        labels=data["labels"],
        grid_ids=grid_ids,
    )

    # build full-graph temporal tensors for collator
    grid_id_to_idx = data["grid_id_to_idx"]
    all_t21 = torch.zeros(num_nodes, 168, 2)
    all_t24 = torch.zeros(num_nodes, 168, 2)
    for gid, feat21 in temporal_2021.items():
        idx = grid_id_to_idx[gid]
        all_t21[idx] = torch.from_numpy(feat21) if isinstance(feat21, np.ndarray) else feat21
    for gid, feat24 in temporal_2024.items():
        idx = grid_id_to_idx[gid]
        all_t24[idx] = torch.from_numpy(feat24) if isinstance(feat24, np.ndarray) else feat24

    # raw temporal for raw_temporal_mean node feature mode
    raw_t21 = torch.zeros(num_nodes, 168, 2)
    raw_t24 = torch.zeros(num_nodes, 168, 2)
    for gid, feat21 in temporal_2021.items():
        idx = grid_id_to_idx[gid]
        raw_t21[idx] = torch.from_numpy(feat21) if isinstance(feat21, np.ndarray) else feat21
    for gid, feat24 in temporal_2024.items():
        idx = grid_id_to_idx[gid]
        raw_t24[idx] = torch.from_numpy(feat24) if isinstance(feat24, np.ndarray) else feat24

    # ── load split ────────────────────────────────────────────────────────────
    with open(SPLIT_PATH) as f:
        manifest = json.load(f)
    splits      = manifest.get("splits", manifest)
    gid_to_idx  = {int(gid): i for i, gid in enumerate(grid_ids)}
    test_indices = [gid_to_idx[int(g)] for g in splits["test"]]
    test_subset  = Subset(dataset, test_indices)

    collator = PureGraphBatchCollator(
        graphs_2021=data["graphs_2021"],
        graphs_2024=data["graphs_2024"],
        grid_id_to_idx=grid_id_to_idx,
        all_temporal_2021=all_t21,
        all_temporal_2024=all_t24,
        all_raw_temporal_2021=raw_t21,
        all_raw_temporal_2024=raw_t24,
    )
    loader = DataLoader(test_subset, batch_size=32, shuffle=False,
                        collate_fn=collator)

    # ── build model ───────────────────────────────────────────────────────────
    print("Building model...")
    model = EnhancedDualBranchModel(
        temporal_input_size=2,
        hidden_size=256,
        num_classes=9,
        num_time_steps=168,
        dropout=0.4,
        spatial_model="GINE",
        temporal_model="TRANSFORMER",
        branch_ablation_mode="full",
        fusion_ablation_mode="gated",
    )

    # register spatial coords
    node_coords   = torch.zeros(num_nodes, 2)
    coord_lookup  = data["metadata_df"].set_index("grid_id")[["lon", "lat"]]
    for gid, idx in data["grid_id_to_idx"].items():
        if gid in coord_lookup.index:
            node_coords[idx, 0] = float(coord_lookup.at[gid, "lon"])
            node_coords[idx, 1] = float(coord_lookup.at[gid, "lat"])
    model.register_node_coords(node_coords)

    # load weights
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {MODEL_PATH}")

    # ── inference ─────────────────────────────────────────────────────────────
    all_gids, all_true, all_pred, all_probs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            node_indices = batch["node_indices"].to(device)
            labels       = batch["labels"].to(device)
            x21 = batch["all_temporal_2021"].to(device)
            x24 = batch["all_temporal_2024"].to(device)
            rx21 = batch["all_raw_temporal_2021"]
            rx24 = batch["all_raw_temporal_2024"]
            if rx21 is not None: rx21 = rx21.to(device)
            if rx24 is not None: rx24 = rx24.to(device)

            g21 = [(ei.to(device) if isinstance(ei, torch.Tensor) else torch.from_numpy(ei).to(device),
                    ea.to(device) if isinstance(ea, torch.Tensor) else torch.from_numpy(ea).to(device))
                   for ei, ea in batch["graphs_2021"]]
            g24 = [(ei.to(device) if isinstance(ei, torch.Tensor) else torch.from_numpy(ei).to(device),
                    ea.to(device) if isinstance(ea, torch.Tensor) else torch.from_numpy(ea).to(device))
                   for ei, ea in batch["graphs_2024"]]

            logits = model(x21, x24, g21, g24,
                           num_nodes=batch["num_nodes"],
                           node_indices=node_indices,
                           raw_x_2021=rx21, raw_x_2024=rx24)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_gids.extend(batch["grid_ids"])
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    # ── build CSV ─────────────────────────────────────────────────────────────
    label_df = pd.read_csv(LABEL_PATH)
    gid_to_label = dict(zip(label_df["grid_id"].astype(int),
                            label_df["label"].astype(int)))

    rows = []
    for i, gid in enumerate(all_gids):
        true_idx  = all_true[i]          # 0-based
        pred_idx  = all_pred[i]          # 0-based
        true_lbl  = true_idx + 1         # 1-based class label
        pred_lbl  = pred_idx + 1
        conf      = float(all_probs[i, pred_idx])
        correct   = int(true_idx == pred_idx)
        row = {
            "grid_id":    int(gid),
            "true_label": true_lbl,
            "pred_label": pred_lbl,
            "correct":    correct,
            "confidence": round(conf, 4),
        }
        for c in range(9):
            row[f"prob_class{c+1}"] = round(float(all_probs[i, c]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(df)} samples)")

    # ── summary ───────────────────────────────────────────────────────────────
    acc = df["correct"].mean() * 100
    print(f"Test accuracy: {acc:.2f}%")

    wrong = df[df["correct"] == 0]
    print(f"\nMisclassified: {len(wrong)} / {len(df)}")
    print("\nTop misclassification pairs (true → pred, count):")
    pairs = wrong.groupby(["true_label", "pred_label"]).size() \
                 .reset_index(name="count") \
                 .sort_values("count", ascending=False)
    print(pairs.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

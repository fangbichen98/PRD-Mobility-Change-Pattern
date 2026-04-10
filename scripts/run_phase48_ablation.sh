#!/bin/bash
# Phase48 ablation & comparison on entropy 0.05 per-450 labels
# Baseline already done: 55.31% / 0.5581

LABEL="data/labels_sgh_entropy_0.05_per_450.csv"
BASE_ARGS="--label-path $LABEL --temporal-model TRANSFORMER --gine-edge-feature-mode flow_distance_direction --spatial-node-feature-mode raw_temporal_mean --temporal-feature-mode inflow_outflow --graph-topk-out 20 --graph-topk-in 20 --graph-temporal-mode static --laplacian-pe-dim 8 --random-seed 202 --num-epochs 300 --early-stopping-patience 30 --batch-size 12"

run() {
    local tag=$1; shift
    echo "=== Starting: $tag ==="
    python train_multiscale_temporal.py $BASE_ARGS --run-tag $tag "$@" \
        2>&1 | tee logs/phase48_${tag}.log
    echo "=== Done: $tag ==="
}

# A1a: GINE 2 layers
run ablation_A1a_gine2l --spatial-model GINE --spatial-layers 2 --gine-use-spatial-coords

# A1b: GINE 1 layer
run ablation_A1b_gine1l --spatial-model GINE --spatial-layers 1 --gine-use-spatial-coords

# A3: No PE / No spatial coords
run ablation_A3_nope_nocoord --spatial-model GINE --spatial-layers 3 --no-gine-spatial-coords --laplacian-pe-dim 0

# A4: Spatial only
run ablation_A4_spatial_only --spatial-model GINE --spatial-layers 3 --gine-use-spatial-coords --branch-ablation-mode spatial_only

# B2: Temporal only
run ablation_B2_temporal_only --spatial-model GINE --spatial-layers 3 --gine-use-spatial-coords --branch-ablation-mode temporal_only

# C1: Concat fusion
run ablation_C1_concat --spatial-model GINE --spatial-layers 3 --gine-use-spatial-coords --fusion-ablation-mode concat

# A2: GCN
run comparison_A2_gcn --spatial-model GCN --spatial-layers 3

# A2: GAT
run comparison_A2_gat --spatial-model GAT --spatial-layers 3

# B1: LSTM
run comparison_B1_lstm --spatial-model GINE --spatial-layers 3 --gine-use-spatial-coords --temporal-model LSTM

# B1: GRU
run comparison_B1_gru --spatial-model GINE --spatial-layers 3 --gine-use-spatial-coords --temporal-model GRU

echo "=== All phase48 ablation experiments done ==="

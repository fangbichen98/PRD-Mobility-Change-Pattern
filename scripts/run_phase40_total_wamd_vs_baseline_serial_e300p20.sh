#!/bin/bash
# Phase40: SGH, Transformer+GINE+flow_distance_direction, topk20, frozen split
# 40a: baseline inflow_outflow + raw_temporal_mean (controlled reference)
# 40b: new total_wamd + flow_wamd (label-aligned features)
set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "Phase40a: baseline inflow_outflow + raw_temporal_mean"
echo "========================================"
python train_multiscale_temporal.py \
    --spatial-model GINE \
    --temporal-model TRANSFORMER \
    --gine-edge-feature-mode flow_distance_direction \
    --temporal-feature-mode inflow_outflow \
    --spatial-node-feature-mode raw_temporal_mean \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --split-manifest data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json \
    --run-tag "sgh_phase40a_transformer_gine_flowdistdir_inflow_outflow_rawmean_topk20_e300p20" \
    2>&1 | tee outputs/phase40a_sgh_baseline_inflow_outflow.log

echo ""
echo "========================================"
echo "Phase40b: new total_wamd + flow_wamd"
echo "========================================"
python train_multiscale_temporal.py \
    --spatial-model GINE \
    --temporal-model TRANSFORMER \
    --gine-edge-feature-mode flow_distance_direction \
    --temporal-feature-mode total_wamd \
    --spatial-node-feature-mode flow_wamd \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --split-manifest data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json \
    --run-tag "sgh_phase40b_transformer_gine_flowdistdir_total_wamd_flowwamd_topk20_e300p20" \
    2>&1 | tee outputs/phase40b_sgh_total_wamd_flow_wamd.log

echo ""
echo "========================================"
echo "Phase40 complete."
echo "========================================"

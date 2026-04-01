#!/bin/bash
# Phase40: SGH, Transformer+GINE+flow_distance_direction, topk20
# Label: label_i0.095_d3.0_spc500.csv (4500 samples, 500/class, balanced)
# 40a: baseline inflow_outflow + raw_temporal_mean
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
    --run-tag "sgh_phase40a_spc500_transformer_gine_flowdistdir_inflow_outflow_rawmean_topk20_e300p20" \
    2>&1 | tee outputs/phase40a_spc500_baseline_inflow_outflow.log

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
    --run-tag "sgh_phase40b_spc500_transformer_gine_flowdistdir_total_wamd_flowwamd_topk20_e300p20" \
    2>&1 | tee outputs/phase40b_spc500_total_wamd_flow_wamd.log

echo ""
echo "========================================"
echo "Phase40 complete."
echo "========================================"

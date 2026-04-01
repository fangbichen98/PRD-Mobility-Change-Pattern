#!/bin/bash
# Phase41: SGH, 基于 phase40a 基线，系统提升 Class 4/5/9 弱势类
#
# 基线(phase40a): test_acc=71.56%, F1=0.7167
#   弱势类: Class5 F1=0.48, Class4 F1=0.55, Class9 F1=0.60
#
# 实验设计：
#   41a: topk=50 + patience=30          → 更大图邻域 + 更长收敛窗口（纯 CLI 验证）
#   41b: focal_loss(γ=2)                → 聚焦难分样本，隔离 focal loss 贡献
#   41c: spatial_layers=3               → 3层GINE（3-hop邻域），隔离深度贡献
#   41d: focal(γ=2) + layers=3 + topk=50 + patience=30  → 全组合
#
# 所有实验固定: Transformer+GINE+flow_distance_direction, inflow_outflow, raw_temporal_mean
#              seed42 frozen split manifest, spc250_seed202_reconstructed labels

set -e

cd "$(dirname "$0")/.."

LABEL_PATH="data/sampled_labels_spc250_seed202_reconstructed.csv"
SPLIT_MANIFEST="data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json"
BASE_ARGS="
    --label-path ${LABEL_PATH}
    --spatial-model GINE
    --temporal-model TRANSFORMER
    --gine-edge-feature-mode flow_distance_direction
    --temporal-feature-mode inflow_outflow
    --spatial-node-feature-mode raw_temporal_mean
    --split-manifest ${SPLIT_MANIFEST}
"

echo "========================================"
echo "Phase41a: topk=50 + patience=30（扩大图邻域 + 更长耐心）"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --graph-topk-out 50 \
    --graph-topk-in 50 \
    --early-stopping-patience 30 \
    --run-tag "sgh_phase41a_topk50_p30_baseline_e300" \
    2>&1 | tee outputs/phase41a_sgh_topk50_p30.log

echo ""
echo "========================================"
echo "Phase41b: Focal Loss(γ=2)，其余与 40a 一致"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --early-stopping-patience 30 \
    --focal-loss-gamma 2.0 \
    --run-tag "sgh_phase41b_focal2_topk20_p30_e300" \
    2>&1 | tee outputs/phase41b_sgh_focal2_topk20_p30.log

echo ""
echo "========================================"
echo "Phase41c: spatial_layers=3（3层GINE，3-hop邻域）"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --early-stopping-patience 30 \
    --spatial-layers 3 \
    --run-tag "sgh_phase41c_layers3_topk20_p30_e300" \
    2>&1 | tee outputs/phase41c_sgh_layers3_topk20_p30.log

echo ""
echo "========================================"
echo "Phase41d: 全组合 focal(γ=2) + layers=3 + topk=50 + patience=30"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --graph-topk-out 50 \
    --graph-topk-in 50 \
    --early-stopping-patience 30 \
    --spatial-layers 3 \
    --focal-loss-gamma 2.0 \
    --run-tag "sgh_phase41d_focal2_layers3_topk50_p30_e300" \
    2>&1 | tee outputs/phase41d_sgh_focal2_layers3_topk50_p30.log

echo ""
echo "========================================"
echo "Phase41 complete."
echo "========================================"

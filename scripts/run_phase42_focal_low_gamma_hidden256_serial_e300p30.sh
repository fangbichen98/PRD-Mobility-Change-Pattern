#!/bin/bash
# Phase42: 基于 phase41c 最优配置（layers=3, topk=20, p=30），
#          探索两个提升方向 + 组合
#
# 基线(phase41c): test_acc=73.33%, F1=0.7340
#   layers=3, topk=20, CrossEntropy, hidden=128
#
# 实验设计：
#   42a: focal(γ=0.5) + layers=3                  → 小γ focal，温和聚焦难分样本
#   42b: layers=3 + hidden=256                     → 更宽空间分支，提升表达能力
#   42c: focal(γ=0.5) + layers=3 + hidden=256      → 两个方向组合

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
    --graph-topk-out 20
    --graph-topk-in 20
    --spatial-layers 3
    --early-stopping-patience 30
    --split-manifest ${SPLIT_MANIFEST}
"

echo "========================================"
echo "Phase42a: focal(γ=0.5) + layers=3（温和focal，隔离γ影响）"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --focal-loss-gamma 0.5 \
    --run-tag "sgh_phase42a_focal05_layers3_topk20_p30_e300" \
    2>&1 | tee outputs/phase42a_sgh_focal05_layers3_topk20_p30.log

echo ""
echo "========================================"
echo "Phase42b: layers=3 + hidden=256（加宽空间分支）"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --spatial-hidden-size 256 \
    --run-tag "sgh_phase42b_layers3_hidden256_topk20_p30_e300" \
    2>&1 | tee outputs/phase42b_sgh_layers3_hidden256_topk20_p30.log

echo ""
echo "========================================"
echo "Phase42c: focal(γ=0.5) + layers=3 + hidden=256（全组合）"
echo "========================================"
python train_multiscale_temporal.py \
    ${BASE_ARGS} \
    --spatial-hidden-size 256 \
    --focal-loss-gamma 0.5 \
    --run-tag "sgh_phase42c_focal05_layers3_hidden256_topk20_p30_e300" \
    2>&1 | tee outputs/phase42c_sgh_focal05_layers3_hidden256_topk20_p30.log

echo ""
echo "========================================"
echo "Phase42 complete."
echo "========================================"

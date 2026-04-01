#!/bin/bash
# Phase39: GBA region, Transformer + GINE, serial experiments
# 39a: flow_distance_direction, no spatial coords (phase31 best config replicated on GBA)
# 39b: flow_distance_direction + spatial coords (new feature)
# Both: topk50/50, static, seed42 random split, labels_gba_entropy_1200_per_class.csv
set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "Phase39a: GBA Transformer+GINE+flowdistdir+topk50"
echo "========================================"
python train_multiscale_temporal.py \
    --spatial-model GINE \
    --temporal-model TRANSFORMER \
    --gine-edge-feature-mode flow_distance_direction \
    --graph-topk-out 50 \
    --graph-topk-in 50 \
    --run-tag "gba_phase39a_transformer_gine_flowdistdir_topk50_e300p20" \
    2>&1 | tee outputs/phase39a_gba_transformer_gine_flowdistdir_topk50.log

echo ""
echo "========================================"
echo "Phase39b: GBA Transformer+GINE+flowdistdir+spatialcoord+topk50"
echo "========================================"
python train_multiscale_temporal.py \
    --spatial-model GINE \
    --temporal-model TRANSFORMER \
    --gine-edge-feature-mode flow_distance_direction \
    --gine-use-spatial-coords \
    --graph-topk-out 50 \
    --graph-topk-in 50 \
    --run-tag "gba_phase39b_transformer_gine_flowdistdir_spatialcoord_topk50_e300p20" \
    2>&1 | tee outputs/phase39b_gba_transformer_gine_flowdistdir_spatialcoord_topk50.log

echo ""
echo "========================================"
echo "Phase39 complete."
echo "========================================"

#!/usr/bin/env bash
# 从 B1-LSTM 开始跑剩余实验
set -e

LABEL="data/sampled_labels_spc250_seed202_reconstructed.csv"
SPLIT="data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json"
BASE_ARGS="--label-path ${LABEL} --split-manifest ${SPLIT} --num-epochs 300 --early-stopping-patience 30"

echo "============================================================"
echo "Phase41c 剩余实验 (B1-LSTM 起)"
echo "============================================================"

# B1-LSTM
echo "[B1-LSTM] Temporal model: LSTM"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --temporal-model LSTM \
  --spatial-model GINE \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpB1_lstm_gine3l_topk20"

# B1-GRU
echo "[B1-GRU] Temporal model: GRU"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --temporal-model GRU \
  --spatial-model GINE \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpB1_gru_gine3l_topk20"

# topk10
echo "[topk10] Graph sparsity: topk=10"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --graph-topk-out 10 --graph-topk-in 10 \
  --run-tag "phase41c_topk10_gine3l"

# topk30
echo "[topk30] Graph sparsity: topk=30"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --graph-topk-out 30 --graph-topk-in 30 \
  --run-tag "phase41c_topk30_gine3l"

# topk40
echo "[topk40] Graph sparsity: topk=40"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --graph-topk-out 40 --graph-topk-in 40 \
  --run-tag "phase41c_topk40_gine3l"

# topk50
echo "[topk50] Graph sparsity: topk=50"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --graph-topk-out 50 --graph-topk-in 50 \
  --run-tag "phase41c_topk50_gine3l"

echo "============================================================"
echo "剩余实验完成！运行: python scripts/aggregate_ablation_results.py"
echo "============================================================"

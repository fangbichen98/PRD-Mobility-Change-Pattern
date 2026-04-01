#!/usr/bin/env bash
# =============================================================================
# Phase41c 消融实验 & 对比实验 & topk系列
# Baseline: phase41c — TRANSFORMER + GINE(3层) + flow_distance_direction
#           + raw_temporal_mean + topk20 + frozen_split
#           acc=73.33 / F1=0.7340
#
# 实验清单:
#   ① 消融实验 (Ablation): A1a, A1b, A3, A4, B2, C1
#   ② 对比实验 (Comparison): A2-GCN, A2-SAGE, A2-GAT, B1-LSTM, B1-GRU
#   ③ topk系列: topk10, topk30, topk40, topk50 (topk20=phase41c baseline)
#
# 用法:
#   bash scripts/run_phase41c_ablation_comparison.sh
#   或逐条复制粘贴到服务器执行
# =============================================================================

set -e

LABEL="data/sampled_labels_spc250_seed202_reconstructed.csv"
SPLIT="data/splits/sampled_labels_spc250_seed202_reconstructed_seed42_split.json"
BASE_ARGS="--label-path ${LABEL} --split-manifest ${SPLIT} --num-epochs 300 --early-stopping-patience 30"

echo "============================================================"
echo "Phase41c 消融 & 对比实验"
echo "Baseline: acc=73.33 / F1=0.7340 / Kappa=?"
echo "============================================================"

# ===========================================================================
# ① 消融实验 (Ablation)
# ===========================================================================

# A1a: GINE 3层 → 2层 (验证第3层贡献)
echo "[A1a] GINE layers=2 (ablation: depth)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 2 \
  --run-tag "phase41c_ablA1a_gine2l_topk20"

# A1b: GINE 3层 → 1层 (极简空间分支)
echo "[A1b] GINE layers=1 (ablation: minimal spatial)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 1 \
  --run-tag "phase41c_ablA1b_gine1l_topk20"

# A3: 移除 Laplacian PE 和空间坐标 (验证结构位置编码贡献)
echo "[A3] No Laplacian PE, no spatial coords (ablation: PE)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --laplacian-pe-dim 0 \
  --no-gine-spatial-coords \
  --run-tag "phase41c_ablA3_gine3l_noPE_noCoords_topk20"

# A4: 去掉时序分支 (spatial only)
echo "[A4] Spatial only (ablation: remove temporal branch)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --branch-ablation-mode spatial_only \
  --run-tag "phase41c_ablA4_spatialOnly_gine3l_topk20"

# B2: 去掉空间分支 (temporal only)
echo "[B2] Temporal only (ablation: remove spatial branch)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --temporal-model TRANSFORMER \
  --branch-ablation-mode temporal_only \
  --run-tag "phase41c_ablB2_temporalOnly_transformer_topk20"

# C1: Attention Fusion → Concat (验证门控融合贡献)
echo "[C1] Concat fusion (ablation: fusion method)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --fusion-ablation-mode concat \
  --run-tag "phase41c_ablC1_concatFusion_gine3l_topk20"

# ===========================================================================
# ② 对比实验 (Comparison)
# ===========================================================================

# A2-GCN: GINE → GCN (证明GINE对边特征的表达优势)
echo "[A2-GCN] Spatial model: GCN (comparison)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GCN \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpA2_gcn3l_topk20"

# A2-SAGE: GINE → GraphSAGE
echo "[A2-SAGE] Spatial model: SAGE (comparison)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model SAGE \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpA2_sage3l_topk20"

# A2-GAT: GINE → GAT
echo "[A2-GAT] Spatial model: GAT (comparison)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GAT \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpA2_gat3l_topk20"

# B1-LSTM: Transformer → LSTM (时序模型对比)
echo "[B1-LSTM] Temporal model: LSTM (comparison)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --temporal-model LSTM \
  --spatial-model GINE \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpB1_lstm_gine3l_topk20"

# B1-GRU: Transformer → GRU
echo "[B1-GRU] Temporal model: GRU (comparison)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --temporal-model GRU \
  --spatial-model GINE \
  --spatial-layers 3 \
  --run-tag "phase41c_cmpB1_gru_gine3l_topk20"

# ===========================================================================
# ③ topk 系列 (topk20 = phase41c baseline, 已有结果)
# ===========================================================================

# topk10
echo "[topk10] Graph sparsity: topk=10"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --graph-topk-out 10 --graph-topk-in 10 \
  --run-tag "phase41c_topk10_gine3l"

# topk20 基线实验
echo "[topk20] Graph sparsity: topk=20 (baseline)"
python train_multiscale_temporal.py ${BASE_ARGS} \
  --spatial-model GINE \
  --spatial-layers 3 \
  --graph-topk-out 20 --graph-topk-in 20 \
  --run-tag "phase41c_topk20_gine3l"

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
echo "所有实验已完成！运行汇总脚本: python scripts/aggregate_ablation_results.py"
echo "============================================================"

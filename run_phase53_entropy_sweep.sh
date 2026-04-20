#!/bin/bash
# Phase53 protocol on 4 entropy-filtered label files
# Config: TRANSFORMER + GINE + flow_distance_direction + hourly-only subscale
# e300p30 / seed202 / topk20

set -e

mkdir -p logs

COMMON="python train_multiscale_temporal.py \
  --temporal-model TRANSFORMER \
  --spatial-model GINE \
  --gine-edge-feature-mode flow_distance_direction \
  --graph-topk-out 20 \
  --graph-topk-in 20 \
  --random-seed 202 \
  --num-epochs 300 \
  --early-stopping-patience 30 \
  --temporal-subscales hourly"

# ── 1/4  entropy=0.03  spc≈700 ────────────────────────────────────────────────
echo "====== [1/4] phase53_entropy003_sample700 ======"
$COMMON \
  --label-path data/labels_sgh_entropy_0.03_sample700.csv \
  --run-tag    phase53_entropy003_sample700 \
  2>&1 | tee logs/phase53_entropy003_sample700.log
echo "====== [1/4] DONE ======"

# ── 2/4  entropy=0.05  spc≈450 ────────────────────────────────────────────────
echo "====== [2/4] phase53_entropy005_sample450 ======"
$COMMON \
  --label-path data/labels_sgh_entropy_0.05_sample450.csv \
  --run-tag    phase53_entropy005_sample450 \
  2>&1 | tee logs/phase53_entropy005_sample450.log
echo "====== [2/4] DONE ======"

# ── 3/4  entropy=0.07  spc≈350 ────────────────────────────────────────────────
echo "====== [3/4] phase53_entropy007_sample350 ======"
$COMMON \
  --label-path data/labels_sgh_entropy_0.07_sample350.csv \
  --run-tag    phase53_entropy007_sample350 \
  2>&1 | tee logs/phase53_entropy007_sample350.log
echo "====== [3/4] DONE ======"

# ── 4/4  entropy=0.09  spc≈300 ────────────────────────────────────────────────
echo "====== [4/4] phase53_entropy009_sample300 ======"
$COMMON \
  --label-path data/labels_sgh_entropy_0.09_sample300.csv \
  --run-tag    phase53_entropy009_sample300 \
  2>&1 | tee logs/phase53_entropy009_sample300.log
echo "====== [4/4] DONE ======"

echo ""
echo "All 4 phase53 entropy experiments completed."
echo "Logs: logs/phase53_entropy00[3579]_sample*.log"

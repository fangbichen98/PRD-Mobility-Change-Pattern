#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase23_gcn_temporal_logs"
mkdir -p "$LOG_DIR"

RUN_TAG="phase23_rawflow_gcn_topk_static_transformer_full_tuned_spc250_seed202_e300p30"
LOG_FILE="$LOG_DIR/${RUN_TAG}.log"

has_active_transformer_full() {
  pgrep -af "train_multiscale_temporal.py.*phase23_rawflow_gcn_topk_static_transformer_full_spc250_seed202_e300p30|run_phase23_transformer_full_e300p30.sh" >/dev/null 2>&1
}

echo "[PHASE23-TF_FULL-TUNED] queue started at $(date '+%F %T')" | tee -a "$LOG_FILE"
while has_active_transformer_full; do
  echo "[PHASE23-TF_FULL-TUNED] waiting current transformer_full run to finish... $(date '+%F %T')" | tee -a "$LOG_FILE"
  sleep 60
done

echo "[PHASE23-TF_FULL-TUNED] start training run_tag=${RUN_TAG} at $(date '+%F %T')" | tee -a "$LOG_FILE"
PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
  /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
  train_multiscale_temporal.py \
  --label-path data/label_sgh.csv \
  --samples-per-class 250 \
  --spatial-model GCN \
  --temporal-model TRANSFORMER_FULL \
  --spatial-node-feature-mode raw_temporal_mean \
  --graph-topk-out 20 \
  --graph-topk-in 20 \
  --graph-temporal-mode static \
  --random-seed 202 \
  --learning-rate 5e-5 \
  --num-epochs 300 \
  --early-stopping-patience 30 \
  --run-tag "$RUN_TAG" \
  >> "$LOG_FILE" 2>&1

echo "[PHASE23-TF_FULL-TUNED] finished at $(date '+%F %T')" | tee -a "$LOG_FILE"

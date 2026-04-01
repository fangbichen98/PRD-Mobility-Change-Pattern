#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase23_gcn_temporal_logs"
mkdir -p "$LOG_DIR"

run_one() {
  local temporal_model="$1"
  local run_tag="$2"
  local log_file="$3"

  echo "[PHASE23] start temporal=${temporal_model} run_tag=${run_tag}"
  PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
    train_multiscale_temporal.py \
    --label-path data/label_sgh.csv \
    --samples-per-class 250 \
    --spatial-model GCN \
    --temporal-model "${temporal_model}" \
    --spatial-node-feature-mode raw_temporal_mean \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --graph-temporal-mode static \
    --random-seed 202 \
    --num-epochs 300 \
    --early-stopping-patience 30 \
    --run-tag "${run_tag}" \
    > "${log_file}" 2>&1
  echo "[PHASE23] done temporal=${temporal_model} run_tag=${run_tag}"
}

run_one "TCN" "phase23_rawflow_gcn_topk_static_tcn_spc250_seed202_e300p30" \
  "$LOG_DIR/phase23_rawflow_gcn_topk_static_tcn_spc250_seed202_e300p30.log"

run_one "TRANSFORMER" "phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30" \
  "$LOG_DIR/phase23_rawflow_gcn_topk_static_transformer_spc250_seed202_e300p30.log"

run_one "BIGRU" "phase23_rawflow_gcn_topk_static_bigru_spc250_seed202_e300p30" \
  "$LOG_DIR/phase23_rawflow_gcn_topk_static_bigru_spc250_seed202_e300p30.log"

echo "[PHASE23] all runs completed"

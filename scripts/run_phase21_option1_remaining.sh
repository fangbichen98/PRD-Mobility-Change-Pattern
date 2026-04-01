#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/workspace/Graph_Deep_Learning/20251001-PRD_18-21-24-mobility_change_pattern/analysis/PRD-Mobility-Change-Pattern"
PY_RUNNER="/root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"
cd "$ROOT"
mkdir -p outputs/phase21_longrun_supplement_logs

run_one () {
  local log_file="$1"
  shift
  PYTHONPATH="$ROOT" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python "$PY_RUNNER" train_multiscale_temporal.py "$@" > "$log_file" 2>&1
}

# Graph model supplements under unified long-run protocol
run_one outputs/phase21_longrun_supplement_logs/phase21_rawflow_wgcn_topk_static_spc250_seed202_e300p30.log \
  --label-path data/label_sgh.csv --samples-per-class 250 --spatial-model WGCN \
  --spatial-node-feature-mode raw_temporal_mean --graph-topk-out 20 --graph-topk-in 20 \
  --graph-temporal-mode static --random-seed 202 --num-epochs 300 --early-stopping-patience 30 \
  --run-tag phase21_rawflow_wgcn_topk_static_spc250_seed202_e300p30

run_one outputs/phase21_longrun_supplement_logs/phase21_rawflow_gat_topk_static_spc250_seed202_e300p30.log \
  --label-path data/label_sgh.csv --samples-per-class 250 --spatial-model GAT \
  --spatial-node-feature-mode raw_temporal_mean --graph-topk-out 20 --graph-topk-in 20 \
  --graph-temporal-mode static --random-seed 202 --num-epochs 300 --early-stopping-patience 30 \
  --run-tag phase21_rawflow_gat_topk_static_spc250_seed202_e300p30

run_one outputs/phase21_longrun_supplement_logs/phase21_rawflow_gine_topk_static_spc250_seed202_e300p30.log \
  --label-path data/label_sgh.csv --samples-per-class 250 --spatial-model GINE \
  --spatial-node-feature-mode raw_temporal_mean --graph-topk-out 20 --graph-topk-in 20 \
  --graph-temporal-mode static --random-seed 202 --num-epochs 300 --early-stopping-patience 30 \
  --run-tag phase21_rawflow_gine_topk_static_spc250_seed202_e300p30

# Temporal replacement baseline (GRU)
run_one outputs/phase21_longrun_supplement_logs/phase21_rawflow_gcn_topk_static_gru_spc250_seed202_e300p30.log \
  --label-path data/label_sgh.csv --samples-per-class 250 --spatial-model GCN --temporal-model GRU \
  --spatial-node-feature-mode raw_temporal_mean --graph-topk-out 20 --graph-topk-in 20 \
  --graph-temporal-mode static --random-seed 202 --num-epochs 300 --early-stopping-patience 30 \
  --run-tag phase21_rawflow_gcn_topk_static_gru_spc250_seed202_e300p30

echo "phase21 option1 remaining long runs done"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase24_250x9_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase24_results.csv"
RESULTS_MD="$LOG_DIR/phase24_results_table.md"

echo "model,acc,f1,run_dir,status" > "$RESULTS_CSV"

run_one() {
  local model_label="$1"
  local temporal_model="$2"
  local spatial_model="$3"
  local node_mode="$4"
  local graph_temporal_mode="$5"
  local run_tag="$6"

  local log_file="$LOG_DIR/${run_tag}.log"
  local run_dir_before
  run_dir_before="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  echo "[PHASE24] start model=${model_label} temporal=${temporal_model} spatial=${spatial_model} node_mode=${node_mode} graph_temporal_mode=${graph_temporal_mode}"

  set +e
  PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
    train_multiscale_temporal.py \
    --label-path data/label_sgh.csv \
    --samples-per-class 250 \
    --spatial-model "$spatial_model" \
    --temporal-model "$temporal_model" \
    --spatial-node-feature-mode "$node_mode" \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --graph-temporal-mode "$graph_temporal_mode" \
    --random-seed 202 \
    --num-epochs 300 \
    --early-stopping-patience 30 \
    --run-tag "$run_tag" \
    > "$log_file" 2>&1
  local exit_code=$?
  set -e

  local run_dir
  run_dir="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  if [[ $exit_code -ne 0 || -z "$run_dir" || "$run_dir" == "$run_dir_before" || ! -f "$run_dir/metrics/test_results.json" ]]; then
    echo "[PHASE24] failed model=${model_label} exit=${exit_code} run_dir=${run_dir:-N/A}"
    echo "${model_label},NA,NA,${run_dir:-N/A},FAILED" >> "$RESULTS_CSV"
    return
  fi

  local acc
  local f1
  acc="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json','r',encoding='utf-8') as f:
    d=json.load(f)
print(f"{float(d['test_accuracy'])/100.0:.4f}")
PY
)"
  f1="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json','r',encoding='utf-8') as f:
    d=json.load(f)
print(f"{float(d['test_f1']):.4f}")
PY
)"

  echo "[PHASE24] done model=${model_label} acc=${acc} f1=${f1} run_dir=${run_dir}"
  echo "${model_label},${acc},${f1},${run_dir},OK" >> "$RESULTS_CSV"
}

# Run lighter/faster settings first, put heavy settings last.
# Temporal model comparison with GCN branch
run_one "LSTM+GCN" "LSTM" "GCN" "raw_temporal_mean" "static" "phase24_lstm_gcn_spc250_seed202_e300p30"
run_one "GRU+GCN" "GRU" "GCN" "raw_temporal_mean" "static" "phase24_gru_gcn_spc250_seed202_e300p30"
run_one "轻量Transformer+GCN" "TRANSFORMER" "GCN" "raw_temporal_mean" "static" "phase24_transformer_gcn_spc250_seed202_e300p30"
run_one "TCN+GCN" "TCN" "GCN" "raw_temporal_mean" "static" "phase24_tcn_gcn_spc250_seed202_e300p30"

# Spatial model comparison with LSTM branch
run_one "LSTM+GraphSAGE" "LSTM" "SAGE" "ones" "static" "phase24_lstm_sage_spc250_seed202_e300p30"
run_one "LSTM+WGCN" "LSTM" "WGCN" "ones" "static" "phase24_lstm_wgcn_spc250_seed202_e300p30"
run_one "LSTM+GAT" "LSTM" "GAT" "ones" "static" "phase24_lstm_gat_spc250_seed202_e300p30"
run_one "LSTM+GINE" "LSTM" "GINE" "ones" "static" "phase24_lstm_gine_spc250_seed202_e300p30"
run_one "LSTM+EvolveGCN" "LSTM" "EVOLVEGCN" "ones" "daily" "phase24_lstm_evolvegcn_spc250_seed202_e300p30"

python - <<PY
import csv
from pathlib import Path
csv_path = Path('${RESULTS_CSV}')
md_path = Path('${RESULTS_MD}')
rows = []
with csv_path.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

ok_rows = [r for r in rows if r['status'] == 'OK']
best_acc = None
best_f1 = None
if ok_rows:
    best_acc = max(float(r['acc']) for r in ok_rows)
    best_f1 = max(float(r['f1']) for r in ok_rows)

lines = []
lines.append('# Phase24 250x9 对比实验结果')
lines.append('')
lines.append('| 双分支模型 | Acc | F1 | 运行目录 | 状态 |')
lines.append('|---|---:|---:|---|---|')
for r in rows:
    acc = r['acc']
    f1 = r['f1']
    if r['status'] == 'OK':
        acc_val = float(acc)
        f1_val = float(f1)
        if best_acc is not None and abs(acc_val - best_acc) < 1e-12:
            acc = f'**{acc}**'
        if best_f1 is not None and abs(f1_val - best_f1) < 1e-12:
            f1 = f'**{f1}**'
    lines.append(f"| {r['model']} | {acc} | {f1} | {r['run_dir']} | {r['status']} |")

md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote markdown table: {md_path}')
PY

echo "[PHASE24] all runs completed."
echo "[PHASE24] CSV: ${RESULTS_CSV}"
echo "[PHASE24] Table: ${RESULTS_MD}"

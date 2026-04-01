#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase28_temporal3_graphsage_gcn_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase28_results.csv"
RESULTS_MD="$LOG_DIR/phase28_results_table.md"

echo "model,acc,f1,run_dir,status" > "$RESULTS_CSV"

run_one() {
  local model_label="$1"
  local temporal_model="$2"
  local spatial_model="$3"
  local node_mode="$4"
  local run_tag="$5"

  local log_file="$LOG_DIR/${run_tag}.log"
  local run_dir_before
  run_dir_before="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  echo "[PHASE28] start model=${model_label} temporal=${temporal_model} spatial=${spatial_model} node_mode=${node_mode}" | tee -a "$log_file"

  set +e
  PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
    train_multiscale_temporal.py \
    --label-path data/label_sgh.csv \
    --samples-per-class 250 \
    --spatial-model "$spatial_model" \
    --temporal-model "$temporal_model" \
    --temporal-layers 3 \
    --spatial-node-feature-mode "$node_mode" \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --graph-temporal-mode static \
    --random-seed 202 \
    --num-epochs 300 \
    --early-stopping-patience 30 \
    --run-tag "$run_tag" \
    >> "$log_file" 2>&1
  local exit_code=$?
  set -e

  local run_dir
  run_dir="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  if [[ $exit_code -ne 0 || -z "$run_dir" || "$run_dir" == "$run_dir_before" || ! -f "$run_dir/metrics/test_results.json" ]]; then
    echo "[PHASE28] failed model=${model_label} exit=${exit_code} run_dir=${run_dir:-N/A}" | tee -a "$log_file"
    echo "${model_label},NA,NA,${run_dir:-N/A},FAILED" >> "$RESULTS_CSV"
    return
  fi

  local acc
  local f1
  acc="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"{float(data['test_accuracy'])/100.0:.4f}")
PY
)"
  f1="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"{float(data['test_f1']):.4f}")
PY
)"

  echo "[PHASE28] done model=${model_label} acc=${acc} f1=${f1} run_dir=${run_dir}" | tee -a "$log_file"
  echo "${model_label},${acc},${f1},${run_dir},OK" >> "$RESULTS_CSV"
}

run_one "LSTM(3层)+GraphSAGE" "LSTM" "SAGE" "ones" "phase28_lstm3_sage_spc250_seed202_e300p30"
run_one "LSTM(3层)+GCN" "LSTM" "GCN" "raw_temporal_mean" "phase28_lstm3_gcn_spc250_seed202_e300p30"

python - <<PY
import csv
from pathlib import Path

csv_path = Path('${RESULTS_CSV}')
md_path = Path('${RESULTS_MD}')
rows = []
with csv_path.open('r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

ok_rows = [row for row in rows if row['status'] == 'OK']
best_acc = max((float(row['acc']) for row in ok_rows), default=None)
best_f1 = max((float(row['f1']) for row in ok_rows), default=None)

lines = []
lines.append('# Phase28 3层时序对比结果')
lines.append('')
lines.append('| 模型 | Acc | F1 | 运行目录 | 状态 |')
lines.append('|---|---:|---:|---|---|')
for row in rows:
    acc = row['acc']
    f1 = row['f1']
    if row['status'] == 'OK':
        if best_acc is not None and abs(float(acc) - best_acc) < 1e-12:
            acc = f'**{acc}**'
        if best_f1 is not None and abs(float(f1) - best_f1) < 1e-12:
            f1 = f'**{f1}**'
    lines.append(f"| {row['model']} | {acc} | {f1} | {row['run_dir']} | {row['status']} |")

md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote markdown table: {md_path}')
PY

echo "[PHASE28] all runs completed."
echo "[PHASE28] CSV: ${RESULTS_CSV}"
echo "[PHASE28] Table: ${RESULTS_MD}"
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="outputs/phase31_transformer_gine_edge_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase31_results.csv"
RESULTS_MD="$LOG_DIR/phase31_results_table.md"

echo "variant,temporal_model,edge_feature_mode,acc,f1,run_dir,status" > "$RESULTS_CSV"

run_one() {
  local variant="$1"
  local edge_mode="$2"
  local run_tag="$3"

  local log_file="$LOG_DIR/${run_tag}.log"
  local run_dir_before
  run_dir_before="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  echo "[PHASE31] start variant=${variant} edge_mode=${edge_mode}"

  set +e
  PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
    train_multiscale_temporal.py \
    --label-path data/label_sgh.csv \
    --samples-per-class 250 \
    --random-seed 202 \
    --num-epochs 300 \
    --early-stopping-patience 30 \
    --spatial-model GINE \
    --temporal-model TRANSFORMER \
    --spatial-node-feature-mode raw_temporal_mean \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --graph-temporal-mode static \
    --gine-edge-feature-mode "$edge_mode" \
    --run-tag "$run_tag" \
    > "$log_file" 2>&1
  local exit_code=$?
  set -e

  local run_dir
  run_dir="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  if [[ $exit_code -ne 0 || -z "$run_dir" || "$run_dir" == "$run_dir_before" || ! -f "$run_dir/metrics/test_results.json" ]]; then
    echo "[PHASE31] failed variant=${variant} exit=${exit_code} run_dir=${run_dir:-N/A}"
    echo "${variant},TRANSFORMER,${edge_mode},NA,NA,${run_dir:-N/A},FAILED" >> "$RESULTS_CSV"
    return
  fi

  local acc
  local f1
  acc="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(f"{float(d['test_accuracy'])/100.0:.4f}")
PY
)"
  f1="$(python - <<PY
import json
with open('${run_dir}/metrics/test_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(f"{float(d['test_f1']):.4f}")
PY
)"

  echo "[PHASE31] done variant=${variant} acc=${acc} f1=${f1} run_dir=${run_dir}"
  echo "${variant},TRANSFORMER,${edge_mode},${acc},${f1},${run_dir},OK" >> "$RESULTS_CSV"
}

run_one "transformer_gine_flow_only" "flow_only" "phase31_transformer_gine_flow_only_spc250_seed202_e300p30"
run_one "transformer_gine_flow_distance_direction" "flow_distance_direction" "phase31_transformer_gine_flowdistdir_spc250_seed202_e300p30"

python - <<PY
import csv
from pathlib import Path

csv_path = Path('${RESULTS_CSV}')
md_path = Path('${RESULTS_MD}')
with csv_path.open('r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

ok_rows = [r for r in rows if r['status'] == 'OK']
best_acc = max((float(r['acc']) for r in ok_rows), default=None)
best_f1 = max((float(r['f1']) for r in ok_rows), default=None)

lines = []
lines.append('# Phase31 Transformer+GINE Edge Follow-Up')
lines.append('')
lines.append('| Variant | Temporal | Edge Feature | Acc | F1 | Run Dir | Status |')
lines.append('|---|---|---|---:|---:|---|---|')
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
    lines.append(f"| {r['variant']} | {r['temporal_model']} | {r['edge_feature_mode']} | {acc} | {f1} | {r['run_dir']} | {r['status']} |")

md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote markdown table: {md_path}')
PY

echo "[PHASE31] all runs completed."
echo "[PHASE31] CSV: ${RESULTS_CSV}"
echo "[PHASE31] Table: ${RESULTS_MD}"
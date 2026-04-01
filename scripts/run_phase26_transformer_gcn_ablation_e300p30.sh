#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${SEED:-202}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-250}"
LABEL_PATH="${LABEL_PATH:-data/label_sgh.csv}"

LOG_DIR="outputs/phase26_transformer_gcn_ablation_seed${SEED}_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase26_results.csv"
RESULTS_MD="$LOG_DIR/phase26_results_table.md"

echo "model,acc,f1,run_dir,status" > "$RESULTS_CSV"

run_one() {
  local model_label="$1"
  local branch_mode="$2"
  local fusion_mode="$3"
  local run_tag="$4"

  local log_file="$LOG_DIR/${run_tag}.log"
  local run_dir_before
  run_dir_before="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  echo "[PHASE26] start model=${model_label} seed=${SEED} branch=${branch_mode} fusion=${fusion_mode}" | tee -a "$log_file"

  set +e
  PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
    train_multiscale_temporal.py \
    --label-path "$LABEL_PATH" \
    --samples-per-class "$SAMPLES_PER_CLASS" \
    --spatial-model GCN \
    --temporal-model TRANSFORMER \
    --spatial-node-feature-mode raw_temporal_mean \
    --graph-topk-out 20 \
    --graph-topk-in 20 \
    --graph-temporal-mode static \
    --branch-ablation-mode "$branch_mode" \
    --fusion-ablation-mode "$fusion_mode" \
    --random-seed "$SEED" \
    --num-epochs 300 \
    --early-stopping-patience 30 \
    --run-tag "$run_tag" \
    >> "$log_file" 2>&1
  local exit_code=$?
  set -e

  local run_dir
  run_dir="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  if [[ $exit_code -ne 0 || -z "$run_dir" || "$run_dir" == "$run_dir_before" || ! -f "$run_dir/metrics/test_results.json" ]]; then
    echo "[PHASE26] failed model=${model_label} exit=${exit_code} run_dir=${run_dir:-N/A}" | tee -a "$log_file"
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

  echo "[PHASE26] done model=${model_label} acc=${acc} f1=${f1} run_dir=${run_dir}" | tee -a "$log_file"
  echo "${model_label},${acc},${f1},${run_dir},OK" >> "$RESULTS_CSV"
}

run_one "Transformer" "temporal_only" "gated" "phase26_transformer_only_label_sgh_spc${SAMPLES_PER_CLASS}_seed${SEED}_e300p30"
run_one "GCN" "spatial_only" "gated" "phase26_gcn_only_label_sgh_spc${SAMPLES_PER_CLASS}_seed${SEED}_e300p30"
run_one "Transformer+GCN (concat)" "full" "concat" "phase26_transformer_gcn_concat_label_sgh_spc${SAMPLES_PER_CLASS}_seed${SEED}_e300p30"
run_one "Transformer+GCN (gated fusion)" "full" "gated" "phase26_transformer_gcn_gated_label_sgh_spc${SAMPLES_PER_CLASS}_seed${SEED}_e300p30"

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
lines.append('# Phase26 Transformer-GCN 消融结果')
lines.append('')
lines.append(f'- label_path: ${LABEL_PATH}')
lines.append(f'- seed: ${SEED}')
lines.append(f'- samples_per_class: ${SAMPLES_PER_CLASS}')
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

echo "[PHASE26] all runs completed."
echo "[PHASE26] CSV: ${RESULTS_CSV}"
echo "[PHASE26] Table: ${RESULTS_MD}"
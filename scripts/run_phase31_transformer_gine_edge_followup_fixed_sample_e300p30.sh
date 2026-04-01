#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

FIXED_LABEL="data/sampled_labels_spc250_seed202_reconstructed.csv"
LOG_DIR="outputs/phase31_fixedsample_transformer_gine_edge_logs"
mkdir -p "$LOG_DIR"

RESULTS_CSV="$LOG_DIR/phase31_fixedsample_results.csv"
RESULTS_MD="$LOG_DIR/phase31_fixedsample_results_table.md"

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "variant,temporal_model,label_path,edge_feature_mode,acc,f1,run_dir,status" > "$RESULTS_CSV"
fi

upsert_result() {
  local variant="$1"
  local temporal_model="$2"
  local label_path="$3"
  local edge_mode="$4"
  local acc="$5"
  local f1="$6"
  local run_dir="$7"
  local status="$8"

  python - <<PY
import csv
from pathlib import Path

csv_path = Path('${RESULTS_CSV}')
fieldnames = ['variant', 'temporal_model', 'label_path', 'edge_feature_mode', 'acc', 'f1', 'run_dir', 'status']
new_row = {
    'variant': ${variant@Q},
    'temporal_model': ${temporal_model@Q},
    'label_path': ${label_path@Q},
    'edge_feature_mode': ${edge_mode@Q},
    'acc': ${acc@Q},
    'f1': ${f1@Q},
    'run_dir': ${run_dir@Q},
    'status': ${status@Q},
}

rows = []
if csv_path.exists():
    with csv_path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

rows = [row for row in rows if row['variant'] != new_row['variant']]
rows.append(new_row)

with csv_path.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

run_one() {
  local variant="$1"
  local edge_mode="$2"
  local run_tag="$3"

  local log_file="$LOG_DIR/${run_tag}.log"
  local run_dir_before
  run_dir_before="$(ls -dt outputs/multiscale_temporal_*_${run_tag} 2>/dev/null | head -n 1 || true)"

  echo "[PHASE31-FIXED] start variant=${variant} edge_mode=${edge_mode} label=${FIXED_LABEL}"

  set +e
  PYTHONPATH="$PWD" /opt/conda/bin/conda run -p /opt/conda --no-capture-output python \
    /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py \
    train_multiscale_temporal.py \
    --label-path "$FIXED_LABEL" \
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
    echo "[PHASE31-FIXED] failed variant=${variant} exit=${exit_code} run_dir=${run_dir:-N/A}"
    upsert_result "$variant" "TRANSFORMER" "$FIXED_LABEL" "$edge_mode" "NA" "NA" "${run_dir:-N/A}" "FAILED"
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

  echo "[PHASE31-FIXED] done variant=${variant} acc=${acc} f1=${f1} run_dir=${run_dir}"
  upsert_result "$variant" "TRANSFORMER" "$FIXED_LABEL" "$edge_mode" "$acc" "$f1" "$run_dir" "OK"
}

if [[ $# -eq 0 ]]; then
  run_one "transformer_gine_flow_only" "flow_only" "phase31_fixedsample_transformer_gine_flow_only_e300p30"
  run_one "transformer_gine_flow_distance_direction" "flow_distance_direction" "phase31_fixedsample_transformer_gine_flowdistdir_e300p30"
else
  case "$1" in
    flow_only)
      run_one "transformer_gine_flow_only" "flow_only" "phase31_fixedsample_transformer_gine_flow_only_e300p30"
      ;;
    flow_distance_direction)
      run_one "transformer_gine_flow_distance_direction" "flow_distance_direction" "phase31_fixedsample_transformer_gine_flowdistdir_e300p30"
      ;;
    *)
      echo "Usage: $0 [flow_only|flow_distance_direction]" >&2
      exit 1
      ;;
  esac
fi

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
lines.append('# Phase31 Transformer+GINE Edge Follow-Up (Fixed Sample)')
lines.append('')
lines.append('| Variant | Temporal | Label Path | Edge Feature | Acc | F1 | Run Dir | Status |')
lines.append('|---|---|---|---|---:|---:|---|---|')
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
    lines.append(f"| {r['variant']} | {r['temporal_model']} | {r['label_path']} | {r['edge_feature_mode']} | {acc} | {f1} | {r['run_dir']} | {r['status']} |")

md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f'Wrote markdown table: {md_path}')
PY

echo "[PHASE31-FIXED] all requested runs completed."
echo "[PHASE31-FIXED] CSV: ${RESULTS_CSV}"
echo "[PHASE31-FIXED] Table: ${RESULTS_MD}"
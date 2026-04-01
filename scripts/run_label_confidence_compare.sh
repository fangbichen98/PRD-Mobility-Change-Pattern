#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

SEED="${SEED:-42}"
TOPK="${TOPK:-20}"
SPATIAL_MODEL="${SPATIAL_MODEL:-SAGE}"
NODE_MODE="${NODE_MODE:-temporal_mean}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-300}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"

BASELINE_LABEL="${BASELINE_LABEL:-data/labels_sgh_entropy_0.03_random.csv}"

# Prefer workspace-root absolute path to avoid cwd-relative path mistakes.
DEFAULT_HIGHCONF_LABEL="${ROOT_DIR}/../../vis/labels/labels_sgh_entropy_0.03_highconf.csv"
ALT_HIGHCONF_LABEL="${ROOT_DIR}/../vis/labels/labels_sgh_entropy_0.03_highconf.csv"
if [[ -z "${HIGHCONF_LABEL:-}" ]]; then
  if [[ -f "$DEFAULT_HIGHCONF_LABEL" ]]; then
    HIGHCONF_LABEL="$DEFAULT_HIGHCONF_LABEL"
  elif [[ -f "$ALT_HIGHCONF_LABEL" ]]; then
    HIGHCONF_LABEL="$ALT_HIGHCONF_LABEL"
  else
    HIGHCONF_LABEL="$DEFAULT_HIGHCONF_LABEL"
  fi
fi

LOG_DIR="outputs/label_confidence_compare_logs"
mkdir -p "$LOG_DIR"

run_one() {
  local label_path="$1"
  local name="$2"
  local run_tag="phase9_labelcmp_${name}_spc${SAMPLES_PER_CLASS}_${SPATIAL_MODEL,,}_topk${TOPK}_seed${SEED}"
  local run_log="${LOG_DIR}/${run_tag}.log"

  local existing_result
  existing_result="$(find outputs -type f -path "*${run_tag}/metrics/test_results.json" | sort | tail -n 1)"
  if [[ -n "$existing_result" ]]; then
    echo "[SKIP] ${name}: existing result ${existing_result}" | tee -a "$run_log"
    return 0
  fi

  echo "[RUN] ${name} label=${label_path}" | tee -a "$run_log"
  PYTHONPATH="$ROOT_DIR" $PY_PREFIX \
    train_multiscale_temporal.py \
    --label-path "$label_path" \
    --samples-per-class "$SAMPLES_PER_CLASS" \
    --spatial-model "$SPATIAL_MODEL" \
    --spatial-node-feature-mode "$NODE_MODE" \
    --graph-topk-out "$TOPK" \
    --graph-topk-in "$TOPK" \
    --random-seed "$SEED" \
    --num-epochs "$NUM_EPOCHS" \
    --early-stopping-patience "$PATIENCE" \
    --run-tag "$run_tag" >> "$run_log" 2>&1
  local exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo "[FAIL] ${name} exit=${exit_code}" | tee -a "$run_log"
    return $exit_code
  fi

  echo "[DONE] ${name}" | tee -a "$run_log"
  return 0
}

run_one "$BASELINE_LABEL" "baseline_random"
run_one "$HIGHCONF_LABEL" "highconf"

python - <<'PY'
import glob
import json
from pathlib import Path

rows = []
for p in sorted(glob.glob('outputs/multiscale_temporal_*_phase9_labelcmp_*_spc*_sage_topk20_seed42/metrics/test_results.json')):
    d = json.loads(Path(p).read_text())
    name = 'unknown'
    if 'baseline_random' in p:
        name = 'baseline_random'
    elif 'highconf' in p:
        name = 'highconf'
    rows.append({
        'name': name,
        'acc': d.get('test_accuracy'),
        'f1': d.get('test_f1'),
        'edges2021': d.get('data_info', {}).get('graph_2021_edges'),
        'edges2024': d.get('data_info', {}).get('graph_2024_edges'),
        'path': p,
    })

rows.sort(key=lambda x: x['f1'] if isinstance(x['f1'], (float, int)) else -1, reverse=True)
out = []
out.append('# Label Confidence Comparison (SAGE + temporal_mean + topk20)')
out.append('')
out.append('| Label Set | Test Acc (%) | Test Macro-F1 | Edges 2021 | Edges 2024 | Result Path |')
out.append('|---|---:|---:|---:|---:|---|')
for r in rows:
    acc = f"{r['acc']:.4f}" if isinstance(r['acc'], (float, int)) else 'N/A'
    f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], (float, int)) else 'N/A'
    out.append(f"| {r['name']} | {acc} | {f1} | {r['edges2021']} | {r['edges2024']} | {r['path']} |")

Path('outputs/label_confidence_compare_results.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('Wrote outputs/label_confidence_compare_results.md')
PY

echo "[SUMMARY] outputs/label_confidence_compare_results.md"

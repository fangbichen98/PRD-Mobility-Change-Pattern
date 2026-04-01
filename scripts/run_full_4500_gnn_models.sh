#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

LABEL_PATH="${LABEL_PATH:-data/labels_sgh_entropy_0.03_random.csv}"
TOPK="${TOPK:-20}"
NODE_MODE="${NODE_MODE:-temporal_mean}"
SEED="${SEED:-42}"
LOG_DIR="outputs/full4500_model_logs"
mkdir -p "$LOG_DIR"

# Keep model names aligned with user request: GIN is accepted and mapped to GINE in train entry.
MODELS=(GCN SAGE GIN GAT)

failed_runs=0
summary_file="outputs/full4500_model_results_table.md"

for model in "${MODELS[@]}"; do
  run_tag="phase8_full4500_${model,,}_topk${TOPK}_seed${SEED}"
  run_log="${LOG_DIR}/${run_tag}.log"

  existing_result="$(find outputs -type f -path "*${run_tag}/metrics/test_results.json" | sort | tail -n 1)"
  if [[ -n "$existing_result" ]]; then
    echo "[SKIP] model=${model} result exists: ${existing_result}" | tee -a "$run_log"
    continue
  fi

  echo "[RUN] model=${model} tag=${run_tag}" | tee -a "$run_log"

  PYTHONPATH="$ROOT_DIR" $PY_PREFIX \
    train_multiscale_temporal.py \
    --label-path "$LABEL_PATH" \
    --spatial-model "$model" \
    --spatial-node-feature-mode "$NODE_MODE" \
    --graph-topk-out "$TOPK" \
    --graph-topk-in "$TOPK" \
    --random-seed "$SEED" \
    --run-tag "$run_tag" >> "$run_log" 2>&1
  exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    failed_runs=$((failed_runs + 1))
    echo "[FAIL] model=${model} exit=${exit_code}" | tee -a "$run_log"
    continue
  fi

  echo "[DONE] model=${model}" | tee -a "$run_log"
done

python - <<'PY'
import glob
import json
import re
from pathlib import Path

rows = []
for p in sorted(glob.glob('outputs/multiscale_temporal_*_phase8_full4500_*_topk20_seed42/metrics/test_results.json')):
    m = re.search(r'phase8_full4500_([a-z0-9]+)_topk', p)
    model = (m.group(1).upper() if m else 'UNKNOWN')
    d = json.loads(Path(p).read_text())
    rows.append({
        'model': model,
        'acc': d.get('test_accuracy'),
        'f1': d.get('test_f1'),
        'e2021': d.get('data_info', {}).get('graph_2021_edges'),
        'e2024': d.get('data_info', {}).get('graph_2024_edges'),
        'path': p
    })

rows.sort(key=lambda x: x['f1'] if x['f1'] is not None else -1, reverse=True)

out = []
out.append('# Full 4500 Samples Model Comparison (GCN/SAGE/GIN/GAT)')
out.append('')
out.append('| Model | Test Acc (%) | Test Macro-F1 | Graph Edges 2021 | Graph Edges 2024 | Result Path |')
out.append('|---|---:|---:|---:|---:|---|')
for r in rows:
    acc = f"{r['acc']:.4f}" if isinstance(r['acc'], (int, float)) else 'N/A'
    f1 = f"{r['f1']:.4f}" if isinstance(r['f1'], (int, float)) else 'N/A'
    out.append(f"| {r['model']} | {acc} | {f1} | {r['e2021']} | {r['e2024']} | {r['path']} |")

Path('outputs/full4500_model_results_table.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('Wrote outputs/full4500_model_results_table.md')
PY

echo "[SUMMARY] Failed runs: ${failed_runs}"
echo "[SUMMARY] Table: outputs/full4500_model_results_table.md"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi
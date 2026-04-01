#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

LABEL_PATH="data/labels_sgh_entropy_0.03_random.csv"
TOPK_VALUES=(5 10 20 30 50)
SEED="${SEED:-42}"
LOG_DIR="outputs/topk_full_logs"
mkdir -p "$LOG_DIR"

failed_runs=0
completed_runs=0
skipped_runs=0

for k in "${TOPK_VALUES[@]}"; do
  run_tag="phase6_full_random_d4_topk_${k}_temporal_mean_allsamples"
  run_log="${LOG_DIR}/${run_tag}.log"

  existing_result="$(find outputs -type f -path "*${run_tag}/metrics/test_results.json" | sort | tail -n 1)"
  if [[ -n "$existing_result" ]]; then
    echo "[SKIP] topk=${k} result already exists: ${existing_result}" | tee -a "$run_log"
    skipped_runs=$((skipped_runs + 1))
    continue
  fi

  echo "[RUN] topk=${k} tag=${run_tag}" | tee -a "$run_log"
  echo "[INFO] Reusing two-level cache if available; first unseen graph variant may still need one-time build." | tee -a "$run_log"

  PYTHONPATH="$ROOT_DIR" $PY_PREFIX \
    train_multiscale_temporal.py \
    --label-path "$LABEL_PATH" \
    --spatial-model SAGE \
    --spatial-node-feature-mode temporal_mean \
    --graph-topk-out "$k" \
    --graph-topk-in "$k" \
    --random-seed "$SEED" \
    --run-tag "$run_tag" >> "$run_log" 2>&1
  exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo "[FAIL] topk=${k} exit=${exit_code}" | tee -a "$run_log"
    failed_runs=$((failed_runs + 1))
    continue
  fi

  echo "[DONE] topk=${k} tag=${run_tag}" | tee -a "$run_log"
  completed_runs=$((completed_runs + 1))
done

echo "[SUMMARY] completed=${completed_runs} skipped=${skipped_runs} failed=${failed_runs}"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi
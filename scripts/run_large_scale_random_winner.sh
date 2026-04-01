#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

LABEL_PATH="data/labels_sgh_entropy_0.03_random.csv"

RUNS=(
  "winner|phase4_full_random_d4_topk_temporal_mean"
)

# Optional control run: INCLUDE_CONTROL_D2=1 scripts/run_large_scale_random_winner.sh
if [[ "${INCLUDE_CONTROL_D2:-0}" == "1" ]]; then
  RUNS+=("control|phase4_full_random_d2_topk_ones")
fi

failed_runs=0
total_runs=0

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r run_type run_tag <<< "$run_spec"
  total_runs=$((total_runs + 1))

  echo "[RUN] type=${run_type} tag=${run_tag}"

  cmd=(
    train_multiscale_temporal.py
    --label-path "$LABEL_PATH"
    --spatial-model SAGE
    --graph-topk-out 20
    --graph-topk-in 20
    --run-tag "$run_tag"
  )

  if [[ "$run_type" == "winner" ]]; then
    cmd+=(--spatial-node-feature-mode temporal_mean)
  else
    cmd+=(--spatial-node-feature-mode ones)
  fi

  PYTHONPATH="$ROOT_DIR" $PY_PREFIX "${cmd[@]}"
  exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    failed_runs=$((failed_runs + 1))
    echo "[WARN] Run failed (exit=${exit_code}): type=${run_type} tag=${run_tag}"
    continue
  fi

  echo "[OK] Run completed: type=${run_type} tag=${run_tag}"
done

echo "[DONE] Large-scale random winner runs completed. total=${total_runs} failed=${failed_runs}"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi

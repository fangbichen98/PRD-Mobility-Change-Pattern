#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

RUNS=(
  "data/label_sgh.csv|GCN|phase2_full_label_sgh_gcn"
  "data/labels_sgh_entropy_0.03_random.csv|SAGE|phase2_full_entropy_random_sage"
)

if [[ "${INCLUDE_CONTROL_GCN_ENTROPY:-0}" == "1" ]]; then
  RUNS+=("data/labels_sgh_entropy_0.03_random.csv|GCN|phase2_full_entropy_random_gcn_control")
fi

failed_runs=0
total_runs=0

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r label_path spatial_model run_tag <<< "$run_spec"
  total_runs=$((total_runs + 1))

  echo "[RUN] label=${label_path} model=${spatial_model} tag=${run_tag}"
  PYTHONPATH="$ROOT_DIR" $PY_PREFIX train_multiscale_temporal.py \
    --label-path "$label_path" \
    --spatial-model "$spatial_model" \
    --run-tag "$run_tag"

  exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    failed_runs=$((failed_runs + 1))
    echo "[WARN] Run failed (exit=${exit_code}): label=${label_path} model=${spatial_model}"
    continue
  fi

  echo "[OK] Run completed: label=${label_path} model=${spatial_model}"
done

echo "[DONE] Large-scale candidate runs completed. total=${total_runs} failed=${failed_runs}"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi
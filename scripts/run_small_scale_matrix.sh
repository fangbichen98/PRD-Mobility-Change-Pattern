#!/usr/bin/env bash
set -uo pipefail

# Small-scale matrix runner for Phase 1 screening.
# It keeps raw inflow/outflow features unchanged and only sweeps label sets + spatial models.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

LABELS=(
  "data/label_sgh.csv"
  "data/labels_sgh_entropy_0.03_random.csv"
)

# Default screening uses memory-feasible models on full graphs.
# To include GINE manually: INCLUDE_GINE=1 scripts/run_small_scale_matrix.sh
SPATIAL_MODELS=("GCN" "SAGE")
if [[ "${INCLUDE_GINE:-0}" == "1" ]]; then
  SPATIAL_MODELS+=("GINE")
fi

SAMPLES_PER_CLASS="50"
NUM_EPOCHS="10"
EARLY_STOP="3"
BATCH_SIZE="8"
GRAD_ACC="1"

failed_runs=0
total_runs=0

for label_path in "${LABELS[@]}"; do
  label_tag="$(basename "$label_path" .csv)"

  for spatial_model in "${SPATIAL_MODELS[@]}"; do
    run_tag="phase1_small_${label_tag}_${spatial_model,,}"
    total_runs=$((total_runs + 1))

    echo "[RUN] label=${label_path} model=${spatial_model} tag=${run_tag}"
    PYTHONPATH="$ROOT_DIR" $PY_PREFIX train_multiscale_temporal.py \
      --label-path "$label_path" \
      --samples-per-class "$SAMPLES_PER_CLASS" \
      --spatial-model "$spatial_model" \
      --num-epochs "$NUM_EPOCHS" \
      --early-stopping-patience "$EARLY_STOP" \
      --batch-size "$BATCH_SIZE" \
      --gradient-accumulation "$GRAD_ACC" \
      --run-tag "$run_tag"

    exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
      failed_runs=$((failed_runs + 1))
      echo "[WARN] Run failed (exit=${exit_code}): label=${label_path} model=${spatial_model}"
      continue
    fi

    echo "[OK] Run completed: label=${label_path} model=${spatial_model}"
  done

done

echo "[DONE] Small-scale matrix completed. total=${total_runs} failed=${failed_runs}"
echo "[DONE] Check outputs/multiscale_temporal_*_phase1_small_*"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi

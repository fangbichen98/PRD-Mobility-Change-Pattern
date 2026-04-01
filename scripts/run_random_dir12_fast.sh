#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

LABEL_PATH="data/labels_sgh_entropy_0.03_random.csv"
SPATIAL_MODEL="SAGE"
SAMPLES_PER_CLASS="50"
NUM_EPOCHS="10"
EARLY_STOP="3"
BATCH_SIZE="8"
GRAD_ACC="1"

# Quick validation seeds for gate check.
SEEDS=(42 202)

failed_runs=0
total_runs=0

for seed in "${SEEDS[@]}"; do
  for mode in full topk; do
    total_runs=$((total_runs + 1))

    run_tag="phase2_dir12_random_${mode}_seed${seed}"
    echo "[RUN] mode=${mode} seed=${seed} tag=${run_tag}"

    cmd=(
      train_multiscale_temporal.py
      --label-path "$LABEL_PATH"
      --samples-per-class "$SAMPLES_PER_CLASS"
      --spatial-model "$SPATIAL_MODEL"
      --num-epochs "$NUM_EPOCHS"
      --early-stopping-patience "$EARLY_STOP"
      --batch-size "$BATCH_SIZE"
      --gradient-accumulation "$GRAD_ACC"
      --random-seed "$seed"
      --run-tag "$run_tag"
    )

    if [[ "$mode" == "topk" ]]; then
      cmd+=(--graph-topk-out 20 --graph-topk-in 20)
    fi

    PYTHONPATH="$ROOT_DIR" $PY_PREFIX "${cmd[@]}"
    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
      failed_runs=$((failed_runs + 1))
      echo "[WARN] Run failed (exit=${exit_code}) mode=${mode} seed=${seed}"
      continue
    fi

    echo "[OK] Run completed mode=${mode} seed=${seed}"
  done
done

echo "[DONE] Direction1/2 random fast runs completed. total=${total_runs} failed=${failed_runs}"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi

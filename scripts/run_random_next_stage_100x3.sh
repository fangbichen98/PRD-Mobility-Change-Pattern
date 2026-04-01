#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_PREFIX="/opt/conda/bin/conda run -p /opt/conda --no-capture-output python /root/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

LABEL_PATH="data/labels_sgh_entropy_0.03_random.csv"
SPATIAL_MODEL="SAGE"
SAMPLES_PER_CLASS="100"
NUM_EPOCHS="20"
EARLY_STOP="5"
BATCH_SIZE="8"
GRAD_ACC="1"

# Three seeds for next-stage robustness gate.
SEEDS=(42 202 3407)

failed_runs=0
total_runs=0

for candidate in d2_topk_ones d4_topk_temporal_mean; do
  for seed in "${SEEDS[@]}"; do
    total_runs=$((total_runs + 1))

    run_tag="phase3_random_100x3_${candidate}_seed${seed}"
    echo "[RUN] candidate=${candidate} seed=${seed} tag=${run_tag}"

    cmd=(
      train_multiscale_temporal.py
      --label-path "$LABEL_PATH"
      --samples-per-class "$SAMPLES_PER_CLASS"
      --spatial-model "$SPATIAL_MODEL"
      --graph-topk-out 20
      --graph-topk-in 20
      --num-epochs "$NUM_EPOCHS"
      --early-stopping-patience "$EARLY_STOP"
      --batch-size "$BATCH_SIZE"
      --gradient-accumulation "$GRAD_ACC"
      --random-seed "$seed"
      --run-tag "$run_tag"
    )

    if [[ "$candidate" == "d4_topk_temporal_mean" ]]; then
      cmd+=(--spatial-node-feature-mode temporal_mean)
    fi

    PYTHONPATH="$ROOT_DIR" $PY_PREFIX "${cmd[@]}"
    exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
      failed_runs=$((failed_runs + 1))
      echo "[WARN] Run failed (exit=${exit_code}) candidate=${candidate} seed=${seed}"
      continue
    fi

    echo "[OK] Run completed candidate=${candidate} seed=${seed}"
  done
done

echo "[DONE] Next-stage random 100x3 completed. total=${total_runs} failed=${failed_runs}"

if [[ $failed_runs -gt 0 ]]; then
  exit 1
fi

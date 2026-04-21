#!/bin/bash
# Label sensitivity experiments
# Runs training for each label file sequentially

set -e

SCRIPT="train_multiscale_temporal.py"
TAG="label_exp"

LABELS=(
    "data/labels_sgh_entropy_0.03_per700.csv"
    "data/labels_sgh_entropy_0.05_per450.csv"
    "data/labels_sgh_entropy_0.07_per350.csv"
    "data/labels_sgh_entropy_0.09_per300.csv"
    "data/labels_sgh_entropy_0.10_per250.csv"
    "data/labels_sgh_entropy_0.12_per250.csv"
)

for LABEL in "${LABELS[@]}"; do
    echo "========================================"
    echo "Running: $LABEL"
    echo "========================================"
    python $SCRIPT --label-path "$LABEL" --run-tag "$TAG"
    echo "Done: $LABEL"
    echo ""
done

echo "All experiments completed."

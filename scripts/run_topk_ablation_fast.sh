#!/bin/bash
mkdir -p outputs/topk_ablation_logs

TOPK_VALUES=(5 10 20 30 50)
SEED=42

for k in "${TOPK_VALUES[@]}"; do
    TAG="ablation_topk_${k}_seed_${SEED}"
    echo "Running $TAG..."
    python train_multiscale_temporal.py \
        --label-path data/labels_sgh_entropy_0.03_random.csv \
        --samples-per-class 50 \
        --spatial-model SAGE \
        --spatial-node-feature-mode temporal_mean \
        --graph-topk-out $k \
        --graph-topk-in $k \
        --random-seed $SEED \
        --num-epochs 30 \
        --early-stopping-patience 10 \
        --batch-size 8 \
        --run-tag $TAG > outputs/topk_ablation_logs/${TAG}.log 2>&1
done
echo "Top-K ablation completed."

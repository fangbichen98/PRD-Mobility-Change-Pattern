#!/bin/bash
mkdir -p outputs/topk_ablation_logs

TOPK_VALUES=(5 10 30 50)
SEEDS=(42 202)

for k in "${TOPK_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TAG="ablation_topk_${k}_seed_${seed}"
        echo "Running $TAG..."
        python train_multiscale_temporal.py \
            --label-path data/labels_sgh_entropy_0.03_random.csv \
            --samples-per-class 50 \
            --spatial-model SAGE \
            --spatial-node-feature-mode temporal_mean \
            --graph-topk-out $k \
            --graph-topk-in $k \
            --random-seed $seed \
            --num-epochs 50 \
            --early-stopping-patience 10 \
            --batch-size 8 \
            --run-tag $TAG > outputs/topk_ablation_logs/${TAG}.log 2>&1
    done
done
echo "Top-K ablation completed."

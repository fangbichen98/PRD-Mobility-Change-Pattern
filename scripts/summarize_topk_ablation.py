import os
import json
import glob
from collections import defaultdict
import numpy as np

log_dir = "outputs"
# We want to find folders like multiscale_temporal_*_ablation_topk_*_seed_*
pattern = os.path.join(log_dir, "multiscale_temporal_*_ablation_topk_*")
dirs = glob.glob(pattern)

results = defaultdict(list)

for d in dirs:
    metrics_file = os.path.join(d, "metrics", "test_results.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            # Extarct k and seed from dir name
            basename = os.path.basename(d)
            parts = basename.split('_')
            # form is multiscale_temporal_YYYYMMDD_HHMMSS_labels_sgh_entropy_0.03_random_ablation_topk_<k>_seed_<seed>
            # Let's just find 'ablation_topk'
            try:
                idx = parts.index('topk')
                k = parts[idx+1]
                idx_seed = parts.index('seed')
                seed = parts[idx_seed+1]
                acc = data.get('test_accuracy', 0)
                f1 = data.get('test_f1', 0)
                results[int(k)].append({'seed': int(seed), 'acc': acc, 'f1': f1})
            except Exception as e:
                pass
                
print("Top-K Ablation Results (so far):")
print("-" * 50)
print(f"{'Top-K':<10} | {'Runs':<10} | {'Avg Acc (%)':<15} | {'Avg F1':<10}")
print("-" * 50)

for k in sorted(results.keys()):
    runs = results[k]
    avg_acc = np.mean([r['acc'] for r in runs])
    avg_f1 = np.mean([r['f1'] for r in runs])
    print(f"{k:<10} | {len(runs):<10} | {avg_acc:<15.2f} | {avg_f1:<10.4f}")
    

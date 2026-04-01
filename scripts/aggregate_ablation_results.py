"""
汇总 phase41c 消融/对比实验结果，生成 CSV 表格。
用法:
    python scripts/aggregate_ablation_results.py
输出:
    outputs/phase41c_ablation_comparison_results.csv
"""

import os
import json
import glob
import numpy as np
import csv

# ============================================================
# 实验映射表：run_tag → 实验标签
# ============================================================
EXPERIMENT_MAP = [
    # Baseline
    {
        "exp_id": "Baseline",
        "category": "Baseline",
        "description": "phase41c: TRANSFORMER+GINE(3L)+topk20",
        "run_tag": "phase41c_layers3_topk20_p30",
    },
    # ① Ablation
    {
        "exp_id": "A1a",
        "category": "Ablation",
        "description": "GINE layers=2 (depth ablation)",
        "run_tag": "phase41c_ablA1a_gine2l_topk20",
    },
    {
        "exp_id": "A1b",
        "category": "Ablation",
        "description": "GINE layers=1 (minimal spatial)",
        "run_tag": "phase41c_ablA1b_gine1l_topk20",
    },
    {
        "exp_id": "A3",
        "category": "Ablation",
        "description": "No Laplacian PE, no spatial coords",
        "run_tag": "phase41c_ablA3_gine3l_noPE_noCoords_topk20",
    },
    {
        "exp_id": "A4",
        "category": "Ablation",
        "description": "Spatial only (no temporal branch)",
        "run_tag": "phase41c_ablA4_spatialOnly_gine3l_topk20",
    },
    {
        "exp_id": "B2",
        "category": "Ablation",
        "description": "Temporal only (no spatial branch)",
        "run_tag": "phase41c_ablB2_temporalOnly_transformer_topk20",
    },
    {
        "exp_id": "C1",
        "category": "Ablation",
        "description": "Concat fusion (replace gated)",
        "run_tag": "phase41c_ablC1_concatFusion_gine3l_topk20",
    },
    # ② Comparison
    {
        "exp_id": "A2-GCN",
        "category": "Comparison",
        "description": "Spatial: GCN (vs GINE)",
        "run_tag": "phase41c_cmpA2_gcn3l_topk20",
    },
    {
        "exp_id": "A2-SAGE",
        "category": "Comparison",
        "description": "Spatial: GraphSAGE (vs GINE)",
        "run_tag": "phase41c_cmpA2_sage3l_topk20",
    },
    {
        "exp_id": "A2-GAT",
        "category": "Comparison",
        "description": "Spatial: GAT (vs GINE)",
        "run_tag": "phase41c_cmpA2_gat3l_topk20",
    },
    {
        "exp_id": "B1-LSTM",
        "category": "Comparison",
        "description": "Temporal: LSTM (vs Transformer)",
        "run_tag": "phase41c_cmpB1_lstm_gine3l_topk20",
    },
    {
        "exp_id": "B1-GRU",
        "category": "Comparison",
        "description": "Temporal: GRU (vs Transformer)",
        "run_tag": "phase41c_cmpB1_gru_gine3l_topk20",
    },
    # ③ topk series
    {
        "exp_id": "topk10",
        "category": "TopK",
        "description": "Graph topk=10",
        "run_tag": "phase41c_topk10_gine3l",
    },
    {
        "exp_id": "topk20",
        "category": "TopK",
        "description": "Graph topk=20 (= Baseline)",
        "run_tag": "phase41c_layers3_topk20_p30",
    },
    {
        "exp_id": "topk30",
        "category": "TopK",
        "description": "Graph topk=30",
        "run_tag": "phase41c_topk30_gine3l",
    },
    {
        "exp_id": "topk40",
        "category": "TopK",
        "description": "Graph topk=40",
        "run_tag": "phase41c_topk40_gine3l",
    },
    {
        "exp_id": "topk50",
        "category": "TopK",
        "description": "Graph topk=50",
        "run_tag": "phase41c_topk50_gine3l",
    },
]

OUTPUTS_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUTS_DIR, "phase41c_ablation_comparison_results.csv")
CONFUSION_MATRIX_DIR = os.path.join(OUTPUTS_DIR, "phase41c_ablation_confusion_matrices")


def find_run_dir(run_tag):
    """Find the output directory matching a run tag."""
    pattern = os.path.join(OUTPUTS_DIR, f"*{run_tag}*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    # Return the most recent one
    return matches[-1]


def load_test_results(run_dir):
    """Load test_results.json from a run directory."""
    path = os.path.join(run_dir, "metrics", "test_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_confusion_matrix(run_dir):
    """Load confusion_matrix.npy from a run directory."""
    path = os.path.join(run_dir, "metrics", "confusion_matrix.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def compute_kappa_from_confusion_matrix(cm):
    """Compute Cohen's kappa from a confusion matrix (numpy array)."""
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.diag(cm).sum() / n  # observed accuracy
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (n * n)  # expected accuracy
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)

    rows = []
    for exp in EXPERIMENT_MAP:
        run_dir = find_run_dir(exp["run_tag"])
        if run_dir is None:
            print(f"[MISSING] {exp['exp_id']}: no dir matching '{exp['run_tag']}'")
            rows.append({
                "exp_id": exp["exp_id"],
                "category": exp["category"],
                "description": exp["description"],
                "run_dir": "NOT FOUND",
                "overall_accuracy": "",
                "macro_f1": "",
                "kappa": "",
            })
            continue

        results = load_test_results(run_dir)
        if results is None:
            print(f"[NO RESULTS] {exp['exp_id']}: dir={run_dir}")
            rows.append({
                "exp_id": exp["exp_id"],
                "category": exp["category"],
                "description": exp["description"],
                "run_dir": run_dir,
                "overall_accuracy": "",
                "macro_f1": "",
                "kappa": "",
            })
            continue

        acc = results.get("test_accuracy", "")
        f1 = results.get("test_f1", "")
        kappa = results.get("test_kappa", "")

        # If kappa not in json (old runs), compute from confusion matrix
        if kappa == "":
            cm = load_confusion_matrix(run_dir)
            if cm is not None:
                kappa = compute_kappa_from_confusion_matrix(cm)

        print(f"[OK] {exp['exp_id']}: acc={acc:.2f}% F1={f1:.4f} kappa={kappa}")

        # Copy confusion matrix to central dir
        cm = load_confusion_matrix(run_dir)
        if cm is not None:
            dst = os.path.join(CONFUSION_MATRIX_DIR, f"{exp['exp_id']}_confusion_matrix.npy")
            np.save(dst, cm)
            print(f"       → CM saved to {dst}")

        rows.append({
            "exp_id": exp["exp_id"],
            "category": exp["category"],
            "description": exp["description"],
            "run_dir": os.path.basename(run_dir),
            "overall_accuracy": f"{acc:.2f}" if acc != "" else "",
            "macro_f1": f"{f1:.4f}" if f1 != "" else "",
            "kappa": f"{kappa:.4f}" if isinstance(kappa, float) else kappa,
        })

    # Write CSV
    fieldnames = ["exp_id", "category", "description", "overall_accuracy", "macro_f1", "kappa", "run_dir"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ 汇总表已保存: {OUTPUT_CSV}")
    print(f"✓ 混淆矩阵已复制至: {CONFUSION_MATRIX_DIR}/")

    # Pretty print table
    print("\n" + "=" * 80)
    print(f"{'Exp ID':<12} {'Category':<12} {'Acc %':<10} {'Macro F1':<12} {'Kappa':<10}")
    print("-" * 80)
    for row in rows:
        print(f"{row['exp_id']:<12} {row['category']:<12} {row['overall_accuracy']:<10} "
              f"{row['macro_f1']:<12} {row['kappa']:<10}")
    print("=" * 80)


if __name__ == "__main__":
    main()

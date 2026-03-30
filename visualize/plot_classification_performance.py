import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from viz_config import CLASS_NAMES_MULTILINE, EXPERIMENT_DIR

# --- Publication style (scientific-figure-making skill) ---
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "teal":           "#42949E",
}

PUBLICATION_RCPARAMS = {
    "font.family":        ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size":          24,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.linewidth":     3,
    "legend.frameon":     False,
    "svg.fonttype":       "none",
}

METRIC_COLORS = {
    "precision": PALETTE["blue_main"],
    "recall":    PALETTE["teal"],
    "f1_score":  PALETTE["green_3"],
}

EXPORT_DPI = 600


def parse_classification_report(report_path: Path):
    pattern = re.compile(
        r'^\s*Class\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*$'
    )
    metrics = []
    with report_path.open('r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                class_id, precision, recall, f1, support = m.groups()
                metrics.append({
                    'class_id':  int(class_id),
                    'precision': float(precision),
                    'recall':    float(recall),
                    'f1_score':  float(f1),
                    'support':   int(support),
                })
    if len(metrics) != 9:
        raise ValueError(f'Expected 9 class rows, found {len(metrics)} in {report_path}')
    return sorted(metrics, key=lambda x: x['class_id'])


def plot_classification_performance(report_path: Path, output_dir: Path):
    metrics = parse_classification_report(report_path)

    class_ids = [item['class_id'] for item in metrics]
    precision = np.array([item['precision'] for item in metrics])
    recall    = np.array([item['recall']    for item in metrics])
    f1_score  = np.array([item['f1_score']  for item in metrics])
    support   = np.array([item['support']   for item in metrics])

    x     = np.arange(len(class_ids))
    width = 0.24

    plt.rcParams.update(PUBLICATION_RCPARAMS)
    fig, ax = plt.subplots(figsize=(28, 8), dpi=EXPORT_DPI)

    bars_p = ax.bar(x - width, precision, width, label='Precision',
                    color=METRIC_COLORS['precision'],
                    edgecolor='black', linewidth=1.5)
    bars_r = ax.bar(x,          recall,   width, label='Recall',
                    color=METRIC_COLORS['recall'],
                    edgecolor='black', linewidth=1.5)
    bars_f = ax.bar(x + width,  f1_score, width, label='F1-score',
                    color=METRIC_COLORS['f1_score'],
                    edgecolor='black', linewidth=1.5)

    # Dynamic y-axis: tighten to relevant range
    all_vals = np.concatenate([precision, recall, f1_score])
    y_min = max(0.0, all_vals.min() - 0.12)
    ax.set_ylim(y_min, 1.12)

    ax.set_ylabel('Score', fontsize=26, fontweight='bold')
    ax.set_xlabel('Mobility Pattern Category', fontsize=26, fontweight='bold')
    ax.set_title('Per-Class Classification Performance (Phase 41c)',
                 fontsize=28, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{CLASS_NAMES_MULTILINE[cid]}\n(n={cnt})"
         for cid, cnt in zip(class_ids, support)],
        fontsize=14,
    )
    ax.tick_params(axis='y', labelsize=18)
    ax.yaxis.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=18, title='Metric',
              title_fontsize=19)

    # Value annotations above each bar
    for offset, values in [(-width, precision), (0, recall), (width, f1_score)]:
        for xpos, val in zip(x + offset, values):
            ax.text(xpos, val + 0.018, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout(pad=2)

    output_png = output_dir / 'classification_performance_by_category.png'
    output_pdf = output_dir / 'classification_performance_by_category.pdf'
    fig.savefig(output_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {output_png}')
    print(f'Saved: {output_pdf}')


def main():
    output_dir = Path(__file__).resolve().parent / 'output' / 'phase41c'
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = EXPERIMENT_DIR / 'metrics'
    report_path = metrics_dir / 'classification_report.txt'
    plot_classification_performance(report_path, output_dir)


if __name__ == '__main__':
    main()

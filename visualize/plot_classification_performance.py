import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from viz_config import CLASS_NAMES_MULTILINE, EXPORT_DPI, EXPERIMENT_DIR, FONT_FAMILY, PERFORMANCE_METRIC_COLORS


def parse_classification_report(report_path: Path):
    pattern = re.compile(
        r'^\s*Class\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*$'
    )

    metrics = []
    with report_path.open('r', encoding='utf-8') as file:
        for line in file:
            match = pattern.match(line)
            if match:
                class_id, precision, recall, f1_score, support = match.groups()
                metrics.append({
                    'class_id': int(class_id),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1_score),
                    'support': int(support),
                })

    if len(metrics) != 9:
        raise ValueError(f'Expected 9 class rows, found {len(metrics)} in {report_path}')

    return sorted(metrics, key=lambda item: item['class_id'])


def plot_classification_performance(report_path: Path, output_dir: Path):
    metrics = parse_classification_report(report_path)

    class_ids = [item['class_id'] for item in metrics]
    precision = np.array([item['precision'] for item in metrics])
    recall = np.array([item['recall'] for item in metrics])
    f1_score = np.array([item['f1_score'] for item in metrics])
    support = np.array([item['support'] for item in metrics])

    x = np.arange(len(class_ids))
    width = 0.24

    plt.rcParams['font.family'] = FONT_FAMILY
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)

    ax.bar(x - width, precision, width, label='Precision', color=PERFORMANCE_METRIC_COLORS['precision'], edgecolor='white', linewidth=0.8)
    ax.bar(x, recall, width, label='Recall', color=PERFORMANCE_METRIC_COLORS['recall'], edgecolor='white', linewidth=0.8)
    ax.bar(x + width, f1_score, width, label='F1-score', color=PERFORMANCE_METRIC_COLORS['f1_score'], edgecolor='white', linewidth=0.8)

    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', fontsize=17, fontweight='bold')
    ax.set_xlabel('Mobility Pattern Category', fontsize=17, fontweight='bold')
    ax.set_title('Classification Performance for Each Mobility Pattern Category',
                 fontsize=21, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{CLASS_NAMES_MULTILINE[class_id]}\n(n={count})" for class_id, count in zip(class_ids, support)],
        fontsize=11
    )
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, axis='y', alpha=0.28, linestyle='--')
    ax.legend(loc='upper right', fontsize=13, title='Metric', title_fontsize=14, frameon=True)

    for offset, values in [(-width, precision), (0, recall), (width, f1_score)]:
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.015, f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.22)

    output_png = output_dir / 'classification_performance_by_category.png'
    output_pdf = output_dir / 'classification_performance_by_category.pdf'
    fig.savefig(output_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {output_png}')
    print(f'Saved: {output_pdf}')


def main():
    experiment_dir = EXPERIMENT_DIR
    metrics_dir = experiment_dir / 'metrics'
    report_path = metrics_dir / 'classification_report.txt'
    plot_classification_performance(report_path, metrics_dir)


if __name__ == '__main__':
    main()
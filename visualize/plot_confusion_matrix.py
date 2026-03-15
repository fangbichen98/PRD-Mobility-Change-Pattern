from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from viz_config import CONFUSION_MATRIX_LABELS, EXPORT_DPI, EXPERIMENT_DIR, FONT_FAMILY


def plot_confusion_matrix(metrics_dir: Path):
    cm = np.load(metrics_dir / 'confusion_matrix.npy')
    labels = CONFUSION_MATRIX_LABELS
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.rcParams['font.family'] = FONT_FAMILY

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(9):
        for j in range(9):
            value = cm[i, j]
            color = 'white' if value > cm.max() * 0.5 else 'black'
            ax.text(j, i, f'{value}', ha='center', va='center', fontsize=10, color=color)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=11)

    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Row-normalized %)', fontsize=18, fontweight='bold', pad=12)
    ax2.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(9))
    ax2.set_yticks(range(9))
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_yticklabels(labels, fontsize=12)

    for i in range(9):
        for j in range(9):
            value = cm_norm[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax2.text(j, i, f'{value * 100:.1f}%', ha='center', va='center', fontsize=9, color=color)

    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=11)

    fig.suptitle(
        'Confusion matrix of the proposed HMP-CD framework for 9-class mobility pattern classification',
        fontsize=18,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()

    output_png = metrics_dir / 'confusion_matrix_hmp_cd_9class.png'
    output_pdf = metrics_dir / 'confusion_matrix_hmp_cd_9class.pdf'
    fig.savefig(output_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {output_png}')
    print(f'Saved: {output_pdf}')


def main():
    experiment_dir = EXPERIMENT_DIR
    metrics_dir = experiment_dir / 'metrics'
    plot_confusion_matrix(metrics_dir)


if __name__ == '__main__':
    main()

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from viz_config import CONFUSION_MATRIX_LABELS, EXPERIMENT_DIR

# --- Publication style (scientific-figure-making skill) ---
PUBLICATION_RCPARAMS = {
    "font.family":        ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size":          16,
    "axes.spines.right":  False,
    "axes.spines.top":    False,
    "axes.linewidth":     2.5,
    "legend.frameon":     False,
    "svg.fonttype":       "none",
}

EXPORT_DPI = 300


def plot_confusion_matrix(metrics_dir: Path, output_dir: Path = None):
    cm = np.load(metrics_dir / 'confusion_matrix.npy')
    labels = CONFUSION_MATRIX_LABELS
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    plt.rcParams.update(PUBLICATION_RCPARAMS)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=EXPORT_DPI)

    # --- Left: raw counts ---
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', fontsize=15, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=15, fontweight='bold')
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(9):
        for j in range(9):
            val = cm[i, j]
            color = 'white' if val > cm.max() * 0.5 else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=12)

    # --- Right: row-normalised % ---
    ax2 = axes[1]
    im2 = ax2.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Row-normalised %)', fontsize=18, fontweight='bold', pad=12)
    ax2.set_xlabel('Predicted Class', fontsize=15, fontweight='bold')
    ax2.set_ylabel('True Class', fontsize=15, fontweight='bold')
    ax2.set_xticks(range(9))
    ax2.set_yticks(range(9))
    ax2.set_xticklabels(labels, fontsize=13)
    ax2.set_yticklabels(labels, fontsize=13)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    for i in range(9):
        for j in range(9):
            val = cm_norm[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f'{val * 100:.1f}%', ha='center', va='center',
                     fontsize=10, color=color, fontweight='bold')

    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=12)

    fig.suptitle(
        'Confusion matrix — HMP-CD framework, 9-class mobility pattern (Phase 41c)',
        fontsize=17, fontweight='bold', y=1.02,
    )
    fig.tight_layout(pad=2)

    out = output_dir if output_dir is not None else metrics_dir
    output_png = out / 'confusion_matrix_hmp_cd_9class.png'
    output_pdf = out / 'confusion_matrix_hmp_cd_9class.pdf'
    fig.savefig(output_png, dpi=EXPORT_DPI, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved: {output_png}')
    print(f'Saved: {output_pdf}')


def main():
    output_dir = Path(__file__).resolve().parent / 'output' / 'phase41c'
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = EXPERIMENT_DIR / 'metrics'
    plot_confusion_matrix(metrics_dir, output_dir)


if __name__ == '__main__':
    main()

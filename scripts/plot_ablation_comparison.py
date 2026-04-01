"""
Publication-ready figure for phase41c ablation & comparison experiments.
Two panels:
  Left:  Ablation study (A1a, A1b, A3, A4, B2)
  Right: Comparison study (A2-GCN, A2-SAGE, A2-GAT, B1-LSTM, B1-GRU, C1)
Metrics shown: Overall Accuracy, Macro F1, Kappa
Output: output_analysis/phase41c_ablation_comparison_figure.png/pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "red_strong":     "#B64342",
    "neutral":        "#CFCECE",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "highlight":      "#FFD700",
}

# ── style ─────────────────────────────────────────────────────────────────────
def apply_style(font_size=14, lw=2.0):
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          font_size,
        "axes.linewidth":     lw,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.frameon":     False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
    })

# ── data ──────────────────────────────────────────────────────────────────────
BASELINE = {"label": "Baseline\n(phase41c)", "acc": 73.33, "f1": 0.7340, "kappa": 0.7005}

ABLATION = [
    {"label": "A1a\nGINE 2L",      "acc": 70.67, "f1": 0.7084, "kappa": 0.6704},
    {"label": "A1b\nGINE 1L",      "acc": 71.56, "f1": 0.7161, "kappa": 0.6804},
    {"label": "A3\nNo PE/Coord",   "acc": 70.67, "f1": 0.7084, "kappa": 0.6704},
    {"label": "A4\nSpatial Only",  "acc": 46.22, "f1": 0.4239, "kappa": 0.3931},
    {"label": "B2\nTemporal Only", "acc": 69.11, "f1": 0.6941, "kappa": 0.6528},
]

COMPARISON = [
    {"label": "A2-GCN\nGCN",         "acc": 68.00, "f1": 0.6505, "kappa": 0.6410},
    {"label": "A2-SAGE\nGraphSAGE",  "acc": 68.44, "f1": 0.6815, "kappa": 0.6459},
    {"label": "A2-GAT\nGAT",         "acc": 70.67, "f1": 0.6995, "kappa": 0.6708},
    {"label": "B1-LSTM\nLSTM",       "acc": 64.67, "f1": 0.6382, "kappa": 0.6032},
    {"label": "B1-GRU\nGRU",         "acc": 66.67, "f1": 0.6639, "kappa": 0.6254},
    {"label": "C1\nConcat Fusion",   "acc": 68.22, "f1": 0.6767, "kappa": 0.6434},
]

METRICS = [
    ("acc",   "Overall Accuracy (%)", 1.0),
    ("f1",    "Macro F1",             0.01),
    ("kappa", "Kappa",                0.01),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def _bar_group(ax, rows, baseline_val, metric_key, ylabel, scale,
               bar_color, baseline_color, title):
    labels = [r["label"] for r in rows]
    vals   = [r[metric_key] for r in rows]
    x      = np.arange(len(labels))
    width  = 0.55

    bars = ax.bar(x, vals, width, color=bar_color, zorder=3, linewidth=0)

    # baseline dashed line
    ax.axhline(baseline_val, color=baseline_color, linewidth=1.8,
               linestyle="--", zorder=4, label=f"Baseline ({baseline_val:.2f})")

    # value annotations
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + scale * 0.3,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

    # y range: leave headroom above baseline
    ymin = min(vals) - scale * 3
    ymax = max(baseline_val, max(vals)) + scale * 6
    ax.set_ylim(ymin, ymax)


def make_figure():
    apply_style(font_size=11, lw=1.8)

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        n_metrics, 2,
        figsize=(14, 4.2 * n_metrics),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35}
    )

    abl_color = PALETTE["red_strong"]
    cmp_color = PALETTE["teal"]
    base_color = PALETTE["blue_main"]

    for row_i, (mkey, mlabel, scale) in enumerate(METRICS):
        bval = BASELINE[mkey]

        # left: ablation
        _bar_group(axes[row_i, 0], ABLATION, bval, mkey, mlabel, scale,
                   abl_color, base_color, "Ablation Study")

        # right: comparison
        _bar_group(axes[row_i, 1], COMPARISON, bval, mkey, mlabel, scale,
                   cmp_color, base_color, "Comparison Study")

    # super title
    fig.suptitle(
        "Phase41c Ablation & Comparison Experiments\n"
        "Baseline: TRANSFORMER + GINE(3L) + flow_distance_direction + topk20  "
        f"(Acc={BASELINE['acc']:.2f}%, F1={BASELINE['f1']:.4f}, κ={BASELINE['kappa']:.4f})",
        fontsize=12, y=1.01
    )

    out_dir = "output_analysis"
    os.makedirs(out_dir, exist_ok=True)
    out_base = os.path.join(out_dir, "phase41c_ablation_comparison_figure")

    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_base}.png / .pdf")


if __name__ == "__main__":
    make_figure()

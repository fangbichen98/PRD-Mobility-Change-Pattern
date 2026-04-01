"""
Two-panel line chart for phase41c ablation & comparison experiments.
Left:  Ablation  (A1a, A1b, A3, A4, B2)
Right: Comparison (A2-GCN, A2-SAGE, A2-GAT, B1-LSTM, B1-GRU, C1)
Three lines per panel: Overall Accuracy (%), Macro F1 (×100), Kappa (×100)
Baseline shown as horizontal dashed line per metric.
Output: output_analysis/phase41c_ablation_comparison_lines.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── palette ───────────────────────────────────────────────────────────────────
C_ACC   = "#0F4D92"   # blue  – accuracy
C_F1    = "#42949E"   # teal  – macro F1
C_KAPPA = "#9A4D8E"   # violet – kappa
C_BASE  = "#B64342"   # red   – baseline dashed

# ── data ──────────────────────────────────────────────────────────────────────
BASELINE = dict(acc=73.33, f1=73.40, kappa=70.05)   # F1/Kappa scaled ×100

ABLATION = [
    dict(label="GINE 2L",      acc=70.67, f1=70.84, kappa=67.04),
    dict(label="GINE 1L",      acc=71.56, f1=71.61, kappa=68.04),
    dict(label="No PE/Coord",   acc=70.67, f1=70.84, kappa=67.04),
    dict(label="Spatial Only",  acc=46.22, f1=42.39, kappa=39.31),
    dict(label="Temporal Only", acc=69.11, f1=69.41, kappa=65.28),
]

COMPARISON = [
    dict(label="GAT",        acc=70.67, f1=69.95, kappa=67.08),
    dict(label="MPNN",        acc=50.00, f1=46.66, kappa=43.70),
    dict(label="LSTM",      acc=64.67, f1=63.82, kappa=60.32),
    dict(label="GRU",        acc=66.67, f1=66.39, kappa=62.54),
    dict(label="Concat Fusion",  acc=68.22, f1=67.67, kappa=64.34),
]

# ── style ─────────────────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.linewidth":     1.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.frameon":     False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
    })

# ── plot helper ───────────────────────────────────────────────────────────────
def plot_panel(ax, rows, baseline, title):
    x      = np.arange(len(rows))
    labels = [r["label"] for r in rows]

    acc_vals   = [r["acc"]   for r in rows]
    f1_vals    = [r["f1"]    for r in rows]
    kappa_vals = [r["kappa"] for r in rows]

    lw, ms = 2.0, 7

    ax.plot(x, acc_vals,   color=C_ACC,   marker="o", linewidth=lw,
            markersize=ms, label="Accuracy (%)", zorder=3)
    ax.plot(x, f1_vals,    color=C_F1,    marker="s", linewidth=lw,
            markersize=ms, label="Macro F1 (×100)", zorder=3)
    ax.plot(x, kappa_vals, color=C_KAPPA, marker="^", linewidth=lw,
            markersize=ms, label="Kappa (×100)", zorder=3)

    # baseline dashed lines
    ax.axhline(baseline["acc"],   color=C_ACC,   linewidth=1.2,
               linestyle="--", alpha=0.55, zorder=2)
    ax.axhline(baseline["f1"],    color=C_F1,    linewidth=1.2,
               linestyle="--", alpha=0.55, zorder=2)
    ax.axhline(baseline["kappa"], color=C_KAPPA, linewidth=1.2,
               linestyle="--", alpha=0.55, zorder=2)

    # value annotations — stagger to avoid overlap
    for xi, (a, f, k) in enumerate(zip(acc_vals, f1_vals, kappa_vals)):
        ax.text(xi - 0.18, a + 0.9,  f"{a:.1f}",  ha="center", va="bottom",
                fontsize=7.5, color=C_ACC)
        ax.text(xi,        f + 0.9,  f"{f:.1f}",  ha="center", va="bottom",
                fontsize=7.5, color=C_F1)
        ax.text(xi + 0.18, k + 0.9,  f"{k:.1f}",  ha="center", va="bottom",
                fontsize=7.5, color=C_KAPPA)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5, zorder=0)

    ymin = min(min(acc_vals), min(f1_vals), min(kappa_vals)) - 5
    ymax = max(baseline["acc"], max(acc_vals), max(f1_vals), max(kappa_vals)) + 8
    ax.set_ylim(ymin, ymax)

    # shade baseline region label
    ax.text(len(rows) - 0.05, baseline["acc"] + 0.5,
            f"Baseline {baseline['acc']:.1f}", ha="right", va="bottom",
            fontsize=8, color=C_ACC, alpha=0.7)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    apply_style()
    fig, (ax_abl, ax_cmp) = plt.subplots(
        1, 2, figsize=(15, 5.5),
        gridspec_kw={"wspace": 0.38}
    )

    plot_panel(ax_abl, ABLATION,   BASELINE, "Ablation Study")
    plot_panel(ax_cmp, COMPARISON, BASELINE, "Comparison Study")

    fig.suptitle(
        "Phase41c Ablation & Comparison — Accuracy / Macro F1 / Kappa\n"
        "Dashed lines = Baseline (phase41c: Acc 73.33%, F1 0.7340, κ 0.7005)",
        fontsize=11, y=1.02
    )

    out_dir = "output_analysis"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "phase41c_ablation_comparison_lines")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {base}.png / .pdf")


if __name__ == "__main__":
    main()

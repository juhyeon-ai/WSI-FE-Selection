"""Render the README result charts from camera-ready Table 1.

Outputs light and dark variants (transparent background) so the README can
switch them with a ``<picture>`` element, plus a light PNG for slides/posts.

    python scripts/plot_benchmark.py

Requires matplotlib. Text is exported as paths so the SVGs render identically
without the source font installed.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"

# Theme tokens. The accent marks the proposed metrics; baselines wear a neutral
# so the chart reads as "highlight one, gray the rest".
THEMES = {
    "light": {
        "surface": "#ffffff",
        "accent": "#008f9c",
        "neutral": "#9fb0bf",
        "connector": "#d0d7de",
        "ink": "#1f2328",
        "ink2": "#57606a",
        "muted": "#6e7781",
        "grid": "#d8dee4",
    },
    "dark": {
        "surface": "#0d1117",
        "accent": "#1f9ea6",
        "neutral": "#5b6875",
        "connector": "#30363d",
        "ink": "#e6edf3",
        "ink2": "#c9d1d9",
        "muted": "#8b949e",
        "grid": "#30363d",
    },
}

UNSUPERVISED = [
    ("SAM-Cluster", "sam_cluster", True),
    ("NESum", "nesum", False),
    ("Self-Cluster", "self_cluster", False),
    ("Effective Dimension", "effective_dimension", False),
]
SUPERVISED = [
    ("SAM-LP", "sam_lp", True),
    ("Linear Probing", "linear_probing", False),
    ("LogME", "logme", False),
]
DATASETS = [
    "Camelyon16", "BRACS", "UBC-OCEAN", "TCGA-GLIOMA",
    "TCGA-NSCLC", "TCGA-RCC", "Histai-skin-b1", "PANDA",
]


def load_table():
    with (ROOT / "docs/results/table1.csv").open(newline="") as stream:
        rows = {row["dataset"]: row for row in csv.DictReader(stream)}
    return rows


def style(theme):
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "path",
        "figure.dpi": 100,
        "text.color": theme["ink"],
        "axes.labelcolor": theme["ink2"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["ink"],
    })


def clean_axis(ax, theme, xlim, xticks):
    ax.set_facecolor("none")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, pad=8, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def rounded_barh(ax, fig, y, value, color, thickness_px=22, radius_px=4):
    """Bar with a square baseline end and a rounded data end (4px radius)."""
    px_per_unit = (ax.transData.transform((1, 0)) - ax.transData.transform((0, 0)))[0]
    radius = radius_px / px_per_unit
    lw_pt = thickness_px * 72 / fig.dpi
    if value <= 2 * radius:
        ax.barh(y, value, height=thickness_px / px_per_unit, color=color, zorder=3)
        return
    # Round caps extend lw/2 past each end. The left cap falls outside the
    # axes (x < 0) and is clipped; the right cap is pulled back to end at value.
    cap = (lw_pt / 72 * fig.dpi / 2) / px_per_unit
    ax.plot([0, value - cap], [y, y], lw=lw_pt, solid_capstyle="round",
            color=color, zorder=3, clip_on=True)


def plot_average(rows, theme, name):
    style(theme)
    average = rows["Average"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.75), gridspec_kw={"wspace": 0.62})
    panels = [("Unsupervised  ·  no slide labels", UNSUPERVISED),
              ("Supervised  ·  with slide labels", SUPERVISED)]
    fig.subplots_adjust(left=0.165, right=0.985, top=0.80, bottom=0.25)
    for ax, (title, metrics) in zip(axes, panels):
        clean_axis(ax, theme, (0, 0.76), [0, 0.2, 0.4, 0.6])
        ax.set_yticks(range(len(metrics)), [m[0] for m in metrics])
        ax.set_ylim(len(metrics) - 0.45, -0.55)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=14, color=theme["ink"])
        ax.set_xlabel("Mean Spearman ρ  ·  higher is better", fontsize=10, labelpad=10)
        fig.canvas.draw()
        for i, (label, key, ours) in enumerate(metrics):
            value = float(average[key])
            rounded_barh(ax, fig, i, value, theme["accent"] if ours else theme["neutral"])
            ax.text(value + 0.016, i, f"{value:.4f}", va="center", fontsize=11,
                    fontweight="bold" if ours else "normal", color=theme["ink"], zorder=4)
        for tick, (_, _, ours) in zip(ax.get_yticklabels(), metrics):
            tick.set_fontweight("bold" if ours else "normal")
    fig.text(0.165, 0.04, "Camera-ready Table 1  ·  average over 8 public WSI datasets  ·  "
             "teal = proposed metric", fontsize=9, color=theme["muted"])
    fig.savefig(OUT / f"benchmark_{name}.svg", metadata={"Date": None}, transparent=True)
    if name == "light":
        fig.savefig(OUT / "benchmark.png", dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def plot_per_dataset(rows, theme, name):
    style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), gridspec_kw={"wspace": 0.5})
    fig.subplots_adjust(left=0.135, right=0.985, top=0.80, bottom=0.19)
    panels = [
        ("Unsupervised", "Self-Cluster", "self_cluster", "SAM-Cluster", "sam_cluster"),
        ("Supervised", "Linear Probing", "linear_probing", "SAM-LP", "sam_lp"),
    ]
    for ax, (title, base_label, base_key, ours_label, ours_key) in zip(axes, panels):
        clean_axis(ax, theme, (0, 1.08), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks(range(len(DATASETS)), DATASETS)
        ax.set_ylim(len(DATASETS) - 0.4, -0.6)
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=30,
                     color=theme["ink"])
        ax.set_xlabel("Spearman ρ with ground-truth MIL ranking", fontsize=10, labelpad=10)
        for i, dataset in enumerate(DATASETS):
            base = float(rows[dataset][base_key])
            ours = float(rows[dataset][ours_key])
            ax.plot([base, ours], [i, i], color=theme["connector"], lw=2.2,
                    solid_capstyle="round", zorder=1)
            ax.scatter([base], [i], s=95, color=theme["neutral"], zorder=2,
                       edgecolor=theme["surface"], linewidth=1.6)
            ax.scatter([ours], [i], s=115, color=theme["accent"], zorder=3,
                       edgecolor=theme["surface"], linewidth=1.6)
            if ours >= base:
                ax.text(ours + 0.028, i, f"{ours:.3f}", va="center", ha="left",
                        fontsize=10, fontweight="bold", color=theme["ink"], zorder=4)
            else:
                ax.text(ours - 0.028, i, f"{ours:.3f}", va="center", ha="right",
                        fontsize=10, fontweight="bold", color=theme["ink"], zorder=4)
        handles = [
            Line2D([], [], marker="o", ls="", ms=8.5, color=theme["neutral"],
                   markeredgecolor=theme["surface"], label=base_label),
            Line2D([], [], marker="o", ls="", ms=9.5, color=theme["accent"],
                   markeredgecolor=theme["surface"], label=f"{ours_label} (ours)"),
        ]
        # Legend sits above the plot on the right, one line below the title,
        # so it never covers a data row or the title.
        legend = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                           ncol=2, frameon=False, fontsize=10, handletextpad=0.4,
                           columnspacing=1.4, borderaxespad=0.3)
        for text in legend.get_texts():
            text.set_color(theme["ink2"])
    fig.text(0.135, 0.035, "Camera-ready Table 1  ·  each row is one dataset  ·  "
             "gray = baseline, teal = proposed", fontsize=9, color=theme["muted"])
    fig.savefig(OUT / f"per_dataset_{name}.svg", metadata={"Date": None}, transparent=True)
    if name == "light":
        fig.savefig(OUT / "per_dataset.png", dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    rows = load_table()
    for name, theme in THEMES.items():
        plot_average(rows, theme, name)
        plot_per_dataset(rows, theme, name)


if __name__ == "__main__":
    main()

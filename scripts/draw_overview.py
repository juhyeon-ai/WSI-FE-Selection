"""Draw the README overview figure: the suitability test versus exhaustive search.

    python scripts/draw_overview.py

Writes assets/overview_light.svg, assets/overview_dark.svg (transparent, for the
README <picture> element) and assets/overview.png (light, on white, for slides).
Requires matplotlib; text is exported as paths so no font is needed to view it.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
W, H = 1600, 740  # canvas in pixels; 1 data unit == 1 px at dpi 100

THEMES = {
    "light": dict(
        surface="#ffffff", ink="#1f2328", ink2="#57606a", muted="#6e7781",
        box="#f6f8fa", edge="#d0d7de", chip="#ffffff", chip_edge="#c7ced6",
        accent="#008f9c", accent_fill="#eaf6f7", accent_edge="#8fcfd5",
        warn="#b4534b", warn_fill="#fbeeec", warn_edge="#e9b8b3",
        slide="#eef1f5", slide_edge="#c7ced6", tissue="#cfa6dc",
        bar="#b9c4cf", dot="#aab6c2",
        regions=["#8fb8e8", "#f0b27a", "#9ad19a", "#e8a0c6"],
    ),
    "dark": dict(
        surface="#0d1117", ink="#e6edf3", ink2="#c9d1d9", muted="#8b949e",
        box="#161b22", edge="#30363d", chip="#21262d", chip_edge="#3d444d",
        accent="#1f9ea6", accent_fill="#0f2a30", accent_edge="#1f5f66",
        warn="#d97b73", warn_fill="#331b1b", warn_edge="#6b3532",
        slide="#21262d", slide_edge="#3d444d", tissue="#8f6aa8",
        bar="#5b6875", dot="#4b5866",
        regions=["#4f7fb8", "#c48a4c", "#5f9f5f", "#b06a92"],
    ),
}

FES = ["CONCH v1.5", "UNI v2", "Virchow2", "Phikon-v2", "Hibou-B", "MUSK"]

# Type scale in canvas pixels. The README shows the figure at roughly 1000 px,
# so 17 px here reads as about 10.5 px on the page.
T_TITLE, T_SUB, T_ROW, T_BOX, T_BODY, T_CAP, T_CHIP, T_BAR = 32, 19, 17, 21, 17, 16, 18, 15


def pt(px):
    """Font size in points for a pixel height on a 100-dpi canvas."""
    return px * 72 / 100


class Canvas:
    def __init__(self, theme):
        self.t = theme
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)  # y grows downward, like a screen
        self.ax.axis("off")

    # -- primitives -------------------------------------------------------
    def box(self, x, y, w, h, fill, edge, r=14, lw=1.4, z=1):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fill, ec=edge, lw=lw, zorder=z))

    def text(self, x, y, s, size, color, weight="normal", ha="left", va="center", z=5):
        return self.ax.text(x, y, s, fontsize=pt(size), color=color, fontweight=weight,
                            ha=ha, va=va, zorder=z)

    def arrow(self, x1, y1, x2, y2, color, lw=2.2):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                          lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=4))

    def pill(self, x_right, yc, s, fg, bg, edge, size=T_ROW, pad=15, h=34):
        t = self.text(x_right - pad, yc, s, size, fg, weight="bold", ha="right", z=6)
        bb = t.get_window_extent(renderer=self.fig.canvas.get_renderer())
        w = bb.width + 2 * pad
        self.box(x_right - w, yc - h / 2, w, h, bg, edge, r=h / 2, lw=1.2, z=5)

    def chip(self, x, y, w, h, s, fg, bg, edge, size=T_CHIP, weight="normal"):
        self.box(x, y, w, h, bg, edge, r=9, lw=1.2, z=3)
        self.text(x + w / 2, y + h / 2, s, size, fg, weight=weight, ha="center", z=6)

    # -- icons ------------------------------------------------------------
    def slide(self, x, y, w=21, h=16):
        t = self.t
        self.box(x, y, w, h, t["slide"], t["slide_edge"], r=3, lw=0.9, z=3)
        self.ax.add_patch(Ellipse((x + w * 0.52, y + h * 0.5), w * 0.58, h * 0.6, angle=-20,
                                  fc=t["tissue"], ec="none", zorder=4))

    def slide_grid(self, x, y, cols, rows, cw=24, ch=20):
        for r in range(rows):
            for c in range(cols):
                self.slide(x + c * cw, y + r * ch)

    def sam_thumb(self, x, y, w=78, h=114):
        t = self.t
        self.box(x, y, w, h, t["slide"], t["slide_edge"], r=6, lw=1.0, z=3)
        cx, cy = x + w / 2, y + h / 2
        self.ax.add_patch(Ellipse((cx, cy), w * 0.74, h * 0.78, fc=t["tissue"], ec="none",
                                  alpha=0.45, zorder=4))
        blobs = [((cx - 11, cy - 20), 32, 30, 15), ((cx + 13, cy - 8), 28, 34, -20),
                 ((cx - 13, cy + 16), 30, 26, 5), ((cx + 11, cy + 24), 24, 22, 25)]
        for (c, bw, bh, ang), col in zip(blobs, t["regions"]):
            self.ax.add_patch(Ellipse(c, bw, bh, angle=ang, fc=col, ec=t["surface"], lw=1.2,
                                      alpha=0.95, zorder=5))

    def dot_grid(self, x, y, cols, rows, step=17, r=5.5):
        for i in range(rows):
            for j in range(cols):
                self.ax.add_patch(Circle((x + j * step, y + i * step), r, fc=self.t["dot"],
                                         ec="none", zorder=4))

    def bars(self, x, y, names, values, top_accent=False, bar_h=14, gap=26, w=96, label_w=84):
        t = self.t
        for i, (n, v) in enumerate(zip(names, values)):
            yy = y + i * gap
            self.text(x + label_w - 8, yy + bar_h / 2, n, T_BAR, t["ink2"], ha="right")
            col = t["accent"] if (top_accent and i == 0) else t["bar"]
            self.ax.add_patch(FancyBboxPatch((x + label_w, yy), w * v, bar_h,
                                             boxstyle="round,pad=0,rounding_size=3",
                                             fc=col, ec="none", zorder=4))

    def cluster_icon(self, x, y):
        """Two well-separated groups of points."""
        t = self.t
        pts_a = [(0, 0), (12, -6), (8, 10), (-10, 8), (-4, -12)]
        pts_b = [(46, 4), (58, -4), (54, 12), (40, 14), (62, 8)]
        for dx, dy in pts_a:
            self.ax.add_patch(Circle((x + 14 + dx, y + 16 + dy), 4.8, fc=t["accent"],
                                     ec=t["surface"], lw=1, zorder=5))
        for dx, dy in pts_b:
            self.ax.add_patch(Circle((x + 14 + dx, y + 16 + dy), 4.8, fc=t["bar"],
                                     ec=t["surface"], lw=1, zorder=5))

    def prototype_icon(self, x, y):
        """Region prototypes converging on a label."""
        t = self.t
        for i, col in enumerate(t["regions"][:3]):
            cy = y + 4 + i * 13
            self.ax.add_patch(Circle((x + 8, cy), 5, fc=col, ec=t["surface"], lw=1, zorder=5))
            self.ax.plot([x + 15, x + 40], [cy, y + 17], lw=1.4, color=t["bar"], zorder=4,
                         solid_capstyle="round")
        self.ax.add_patch(FancyBboxPatch((x + 44, y + 6), 36, 22, boxstyle="round,pad=0,rounding_size=6",
                                         fc=t["accent"], ec="none", zorder=5))
        self.text(x + 62, y + 17, "Y", 13, t["surface"], weight="bold", ha="center", z=6)

    def check(self, x, y, r=32):
        t = self.t
        self.ax.add_patch(Circle((x, y), r, fc=t["accent"], ec="none", zorder=5))
        self.ax.plot([x - 15, x - 4, x + 17], [y + 1, y + 12, y - 12], lw=6.5, color=t["surface"],
                     solid_capstyle="round", solid_joinstyle="round", zorder=6)


def draw(theme_name):
    t = THEMES[theme_name]
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "svg.fonttype": "path"})
    c = Canvas(t)

    # Title ----------------------------------------------------------------
    c.text(40, 50, "Which feature extractor should I use for this cohort?", T_TITLE, t["ink"], weight="bold")
    c.text(40, 93, "The usual answer trains hundreds of MIL models.  "
                   "The suitability test ranks the candidates from a 30-slide sample instead.",
           T_SUB, t["ink2"])

    # Shared input (left) and output (right) -------------------------------
    ROW_A, ROW_B, ROW_H = 180, 470, 230
    IN_X, IN_W = 40, 270
    OUT_X, OUT_W = 1300, 260
    span = ROW_B + ROW_H - ROW_A

    c.box(IN_X, ROW_A, IN_W, span, t["box"], t["edge"], r=16)
    c.text(IN_X + 22, ROW_A + 38, "Candidate feature", T_BOX, t["ink"], weight="bold")
    c.text(IN_X + 22, ROW_A + 66, "extractors", T_BOX, t["ink"], weight="bold")
    c.text(IN_X + 22, ROW_A + 96, "pathology foundation models", T_CAP, t["muted"])
    for i, name in enumerate(FES):
        c.chip(IN_X + 22, ROW_A + 124 + i * 48, IN_W - 44, 38, name, t["ink"], t["chip"], t["chip_edge"])
    c.text(IN_X + IN_W / 2, ROW_A + 124 + len(FES) * 48 + 12, "… 10 candidates in the paper", T_CAP,
           t["muted"], ha="center")

    c.box(OUT_X, ROW_A, OUT_W, span, t["accent_fill"], t["accent_edge"], r=16)
    c.text(OUT_X + OUT_W / 2, ROW_A + 42, "Best FE for", T_BOX, t["ink"], weight="bold", ha="center")
    c.text(OUT_X + OUT_W / 2, ROW_A + 70, "this cohort", T_BOX, t["ink"], weight="bold", ha="center")
    c.check(OUT_X + OUT_W / 2, ROW_A + 158)
    c.chip(OUT_X + 45, ROW_A + 220, OUT_W - 90, 42, "Virchow2", t["ink"], t["chip"], t["chip_edge"],
           size=19, weight="bold")
    c.text(OUT_X + OUT_W / 2, ROW_A + 298, "Train one MIL model", T_BODY, t["ink2"], ha="center")
    c.text(OUT_X + OUT_W / 2, ROW_A + 322, "with the winner", T_BODY, t["ink2"], ha="center")
    c.text(OUT_X + OUT_W / 2, ROW_A + 420, "Both routes agree:", T_CAP, t["muted"], ha="center")
    c.text(OUT_X + OUT_W / 2, ROW_A + 444, "mean Spearman ρ = 0.63", T_CAP, t["muted"], ha="center")
    c.text(OUT_X + OUT_W / 2, ROW_A + 468, "between the rankings", T_CAP, t["muted"], ha="center")

    # Column geometry shared by both rows -----------------------------------
    cols = [(360, 236), (646, 322), (1018, 232)]
    CAP_Y = ROW_H - 26  # caption baseline inside a row box

    def row(y, title, title_color, pill, pill_colors, fill, edge, arrow_color):
        c.text(cols[0][0], y - 28, title, T_ROW, title_color, weight="bold")
        c.pill(cols[2][0] + cols[2][1], y - 28, pill, *pill_colors)
        for x, w in cols:
            c.box(x, y, w, ROW_H, fill, edge, r=14)
        c.arrow(IN_X + IN_W, y + ROW_H / 2, cols[0][0], y + ROW_H / 2, arrow_color)
        for (x, w), (nx, _) in zip(cols, cols[1:]):
            c.arrow(x + w, y + ROW_H / 2, nx, y + ROW_H / 2, arrow_color)
        c.arrow(cols[2][0] + cols[2][1], y + ROW_H / 2, OUT_X, y + ROW_H / 2, arrow_color)

    # Row A: exhaustive search ---------------------------------------------
    y = ROW_A
    row(y, "EXHAUSTIVE SEARCH  ·  the usual way", t["muted"],
        "450 MIL trainings per dataset", (t["warn"], t["warn_fill"], t["warn_edge"]),
        t["box"], t["edge"], t["muted"])
    x, w = cols[0]
    c.text(x + 20, y + 36, "Whole cohort", T_BOX, t["ink"], weight="bold")
    c.slide_grid(x + 20, y + 66, 8, 5)
    c.text(x + 20, y + CAP_Y, "all N slides", T_CAP, t["muted"])
    x, w = cols[1]
    c.text(x + 20, y + 36, "Train every MIL model", T_BOX, t["ink"], weight="bold")
    c.dot_grid(x + 28, y + 80, 9, 5)
    c.text(x + 196, y + 82, "9 aggregators", T_BODY, t["ink2"])
    c.text(x + 196, y + 108, "× 5 seeds", T_BODY, t["ink2"])
    c.text(x + 196, y + 134, "× 10 FEs", T_BODY, t["ink2"])
    c.text(x + 20, y + CAP_Y, "on the whole cohort", T_CAP, t["muted"])
    x, w = cols[2]
    c.text(x + 20, y + 36, "Compare accuracy", T_BOX, t["ink"], weight="bold")
    c.bars(x + 20, y + 62, ["Virchow2", "UNI v2", "CONCH", "Phikon", "Hibou"],
           [0.98, 0.93, 0.90, 0.80, 0.86])
    c.text(x + 20, y + CAP_Y, "after all runs finish", T_CAP, t["muted"])

    # Row B: suitability test -----------------------------------------------
    y = ROW_B
    row(y, "SUITABILITY TEST  ·  this work", t["accent"],
        "no MIL training  ·  seconds per FE", (t["accent"], t["accent_fill"], t["accent_edge"]),
        t["accent_fill"], t["accent_edge"], t["accent"])
    x, w = cols[0]
    c.text(x + 20, y + 36, "30 random slides", T_BOX, t["ink"], weight="bold")
    c.slide_grid(x + 20, y + 64, 5, 6, cw=24, ch=19)
    c.sam_thumb(x + 138, y + 60)
    c.text(x + 20, y + CAP_Y, "+ SAM regions, once per slide", T_CAP, t["muted"])
    x, w = cols[1]
    c.text(x + 20, y + 36, "Suitability score", T_BOX, t["ink"], weight="bold")
    c.cluster_icon(x + 24, y + 66)
    c.text(x + 116, y + 72, "SAM-Cluster  ·  no labels", T_BODY, t["ink"], weight="bold")
    c.text(x + 116, y + 96, "are regions separable?", T_CAP, t["ink2"])
    c.prototype_icon(x + 26, y + 124)
    c.text(x + 116, y + 132, "SAM-LP  ·  with labels", T_BODY, t["ink"], weight="bold")
    c.text(x + 116, y + 156, "do regions predict labels?", T_CAP, t["ink2"])
    c.text(x + 20, y + CAP_Y, "per FE, on the sample only", T_CAP, t["muted"])
    x, w = cols[2]
    c.text(x + 20, y + 36, "Rank candidates", T_BOX, t["ink"], weight="bold")
    c.bars(x + 20, y + 62, ["Virchow2", "UNI v2", "CONCH", "Hibou", "Phikon"],
           [0.98, 0.9, 0.84, 0.7, 0.6], top_accent=True)
    c.text(x + 20, y + CAP_Y, "take the top, or a shortlist", T_CAP, t["muted"])

    return c.fig


def main():
    OUT.mkdir(exist_ok=True)
    for name in THEMES:
        fig = draw(name)
        fig.savefig(OUT / f"overview_{name}.svg", metadata={"Date": None}, transparent=True)
        if name == "light":
            fig.savefig(OUT / "overview.png", dpi=200, facecolor=THEMES[name]["surface"])
        plt.close(fig)


if __name__ == "__main__":
    main()

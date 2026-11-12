"""Shared matplotlib style: recessive axes, thin marks, fixed categorical order."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#d9d8d4"
GRID = "#ebeae6"
# Fixed categorical order (validated reference palette); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
HIGHLIGHT = SERIES[0]


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def hbar(ax, labels, values, color=HIGHLIGHT, fmt="{:,.0f}", log=False):
    """Horizontal bar chart, sorted as given, values direct-labeled at bar ends."""
    if log:
        ax.set_xscale("log")
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.72, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        x = v * 1.08 if log else v + vmax * 0.01
        ax.text(x, yi, fmt.format(v), va="center", fontsize=7, color=TEXT2)
    if log:
        ax.set_xlim(min(values) * 0.5, vmax * 4)
    else:
        ax.set_xlim(0, vmax * 1.15)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(compact))


def compact(x, _=None):
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def highlight_grid(emb, labels, order, title, path, ncols=5, s=1.0, max_bg=40000, seed=0):
    """Small multiples: one panel per category, that category in the highlight hue,
    everything else muted. Avoids >3 hues on a scatter."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    bg = rng.choice(len(emb), min(max_bg, len(emb)), replace=False)
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, order):
        ax.scatter(emb[bg, 0], emb[bg, 1], s=s, c=MUTED, linewidths=0, rasterized=True)
        m = np.where(labels == c)[0]
        if len(m) > max_bg // 4:
            m = rng.choice(m, max_bg // 4, replace=False)
        ax.scatter(emb[m, 0], emb[m, 1], s=s, c=HIGHLIGHT, linewidths=0, rasterized=True)
        ax.set_title(f"{c} (n={int((labels == c).sum()):,})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

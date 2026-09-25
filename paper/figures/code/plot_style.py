from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


HOUSE_COLORS = {
    "blue": "#0077BB",
    "red": "#CC3311",
    "teal": "#009988",
    "orange": "#EE7733",
    "purple": "#AA4499",
    "gold": "#EECC66",
    "black": "#1B1B1E",
    "gray": "#666666",
    "light_gray": "#D9DDE3",
}

HOUSE_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_house_diverging",
    ["#2166AC", "#F7F7F7", "#B2182B"],
)


def setup_paper_house_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def style_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)
    if grid:
        ax.grid(True, which="major", alpha=0.15, linewidth=0.6)
    else:
        ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )


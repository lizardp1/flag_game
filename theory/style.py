"""Typography and colors shared by the final Flag Game theory figure."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

BLUE_DARK = '#1764B5'
ORANGE_DARK = '#B65116'
GRAY = '#747C85'
LIGHT_GRAY = '#D9DEE5'
INK = '#25272B'
WHITE = '#FFFFFF'
FIELD_CMAP = LinearSegmentedColormap.from_list(
    'truth_positive_field', [ORANGE_DARK, '#F4F5F7', BLUE_DARK])


def setup_style():
    """Use Arial when installed, with portable sans-serif fallbacks."""
    for path in (Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                 Path('/Library/Fonts/Arial.ttf')):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans', 'text.usetex': False,
        'font.size': 8.7, 'axes.labelsize': 9.1, 'axes.titlesize': 9.7,
        'xtick.labelsize': 8., 'ytick.labelsize': 8., 'legend.fontsize': 7.3,
        'axes.linewidth': 1., 'xtick.major.width': .9, 'ytick.major.width': .9,
        'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 400,
    })


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(INK)
    ax.spines['bottom'].set_color(INK)
    ax.tick_params(axis='both', colors=INK, direction='out', top=False, right=False)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=LIGHT_GRAY, alpha=.65, linewidth=.7)


def panel_title(ax, text):
    ax.set_title(text, loc='left', fontweight='bold', color=INK, pad=4)


def population_axis(ax):
    ax.set_xscale('log', base=2)
    ax.set_xlim(2, 128)
    ax.set_ylim(0, 1)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128],
                 labels=['2', '4', '8', '16', '32', '64', '128'])
    ax.set_yticks([0, .2, .4, .6, .8, 1],
                 labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_xlabel('Population size N', labelpad=3)
    style_axis(ax)

#!/usr/bin/env python3
"""Make paper-facing memory-conflict probe figures."""

from __future__ import annotations

import csv
import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/memetic-drift-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memetic-drift-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import HOUSE_COLORS, setup_paper_house_style, style_axis


ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT / "results" / "flag_game" / "memory_conflict_probe_pilot" / "results.csv"
SOURCE_CSVS = [SOURCE_CSV]
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"

ALIGNMENT_STEM = "flag_game_memory_conflict_probe_alignment"
CHOICE_STEM = "flag_game_memory_conflict_probe_choice_axis"
MAIN_PANEL_STEM = "flag_game_memory_conflict_probe_main_panel"
SUMMARY_STEM = "flag_game_memory_conflict_probe"

MECH_BLUE = "#2D8CFF"
MECH_ORANGE = "#F17C2E"
MECH_GREEN = "#00A36F"
MECH_PURPLE = "#7C6FB6"
MECH_GRAY = "#747C85"
MECH_LIGHT_GRAY = "#D9DEE5"
MECH_INK = "#25272B"
MECH_WHITE = "#FFFFFF"
PAPER_SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]

MODELS = ["gpt-4o", "gpt-5.4"]
M_VALUES = [1, 3]
X_COUNTS = list(range(9))
FACETS = [
    ("weak_private_evidence", "compatible"),
    ("weak_private_evidence", "incompatible"),
    ("strong_private_evidence", "incompatible"),
]
FACET_CLI_KEYS = {
    "weak-compatible": ("weak_private_evidence", "compatible"),
    "weak-incompatible": ("weak_private_evidence", "incompatible"),
    "strong-incompatible": ("strong_private_evidence", "incompatible"),
}
FACET_TITLES = {
    ("weak_private_evidence", "compatible"): (
        "Weak private evidence\nsocial evidence agrees"
    ),
    ("weak_private_evidence", "incompatible"): (
        "Weak private evidence\nsocial evidence contradicts"
    ),
    ("strong_private_evidence", "incompatible"): (
        "Strong private evidence\nsocial evidence contradicts"
    ),
}
FACET_SUBTITLES: dict[tuple[str, str], str] = {}
OVERLAY_MODEL_LABELS = False
M_SECTION_TITLES = {
    1: "A. m=1: GPT-5.4 chooses private-compatible alternatives",
    3: "B. m=3: reasoned memory reduces direct social copying",
}

MODEL_COLORS = {
    "gpt-4o": HOUSE_COLORS["blue"],
    "gpt-5.4": HOUSE_COLORS["red"],
    "claude-haiku-4-5-20251001": HOUSE_COLORS["purple"],
    "claude-sonnet-4-6": HOUSE_COLORS["teal"],
    "claude-opus-4-1": HOUSE_COLORS["gold"],
    "claude-opus-4-6": HOUSE_COLORS["gold"],
    "claude-opus-4-7": HOUSE_COLORS["gold"],
}
MODEL_MARKERS = {
    "gpt-4o": "o",
    "gpt-5.4": "s",
    "claude-haiku-4-5-20251001": "o",
    "claude-sonnet-4-6": "s",
    "claude-opus-4-1": "^",
    "claude-opus-4-6": "^",
    "claude-opus-4-7": "^",
}
MODEL_DISPLAY_LABELS = {
    "gpt-4o": "gpt-4o",
    "gpt-5.4": "gpt-5.4",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-1": "Claude Opus 4.1",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-7": "Claude Opus 4.7",
}

ALIGNMENT_LEGEND_ORDER = [
    "private_evidence",
    "social_evidence",
    "other_private_compatible",
    "other_incompatible",
]
ALIGNMENT_STACK_ORDER = [
    "private_evidence",
    "other_private_compatible",
    "other_incompatible",
    "social_evidence",
]
ALIGNMENT_GROUPS = {
    "private_evidence": ("private_evidence",),
    "social_evidence": ("social_evidence",),
    "other_private_compatible": ("other_private_and_social", "other_private_only"),
    "other_incompatible": ("other_social_only", "unsupported_other"),
}
ALIGNMENT_LABELS = {
    "private_evidence": "Private target country",
    "social_evidence": "Social evidence country",
    "other_private_compatible": "Other compatible country",
    "other_incompatible": "Incompatible",
}
ALIGNMENT_COLORS = {
    "private_evidence": MECH_BLUE,
    "social_evidence": MECH_ORANGE,
    "other_private_compatible": MECH_GREEN,
    "other_incompatible": MECH_LIGHT_GRAY,
}

MAIN_PANEL_STACK_ORDER = [
    "private_evidence",
    "other_private_compatible",
    "social_evidence",
]
MAIN_PANEL_COLORS = {
    "private_evidence": "#2D8CFF",
    "social_evidence": "#FF8A3D",
    "other_private_compatible": "#00B87A",
}
MAIN_PANEL_LABELS = {
    "private_evidence": "Private target country",
    "social_evidence": "Social evidence country",
    "other_private_compatible": "Other compatible country",
}


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def add_paper_sans_font() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def setup_mechanism_like_style() -> None:
    add_paper_sans_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": PAPER_SANS_FONTS,
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 8.7,
            "axes.labelsize": 9.1,
            "axes.titlesize": 9.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
        }
    )


def style_alignment_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MECH_INK)
    ax.spines["bottom"].set_color(MECH_INK)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=MECH_INK, direction="out", top=False, right=False)
    ax.xaxis.label.set_color(MECH_INK)
    ax.yaxis.label.set_color(MECH_INK)
    ax.title.set_color(MECH_INK)
    ax.set_axisbelow(True)
    ax.grid(axis="both", color=MECH_LIGHT_GRAY, alpha=0.65, linewidth=0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        action="append",
        dest="source_csvs",
        default=None,
        help="Probe results CSV. May be passed multiple times to combine model families.",
    )
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--stem-prefix", default=SUMMARY_STEM)
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument(
        "--m",
        action="append",
        dest="m_values",
        type=int,
        choices=sorted(M_VALUES),
        default=None,
        help="Memory bandwidth value to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--facet",
        action="append",
        dest="facets",
        choices=sorted(FACET_CLI_KEYS),
        default=None,
        help="Facet to include. May be passed multiple times.",
    )
    parser.add_argument(
        "--facet-title",
        action="append",
        dest="facet_titles",
        default=None,
        help="Override a facet title as key=text, e.g. weak-incompatible=Title.",
    )
    parser.add_argument(
        "--facet-subtitle",
        action="append",
        dest="facet_subtitles",
        default=None,
        help="Add a facet subtitle as key=text, e.g. weak-incompatible=Subtitle.",
    )
    parser.add_argument("--alignment-only", action="store_true", help="Only write the alignment figure files.")
    parser.add_argument("--section-title-m1", default=M_SECTION_TITLES[1])
    parser.add_argument("--section-title-m3", default=M_SECTION_TITLES[3])
    parser.add_argument("--hide-section-titles", action="store_true")
    parser.add_argument(
        "--overlay-model-labels",
        action="store_true",
        help="Draw model names inside each alignment panel and use a shared probability y-axis.",
    )
    return parser.parse_args()


def parse_facet_text_overrides(items: list[str] | None, *, field: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{field} overrides must use key=text syntax: {item}")
        key, text = item.split("=", 1)
        key = key.strip()
        if key not in FACET_CLI_KEYS:
            allowed = ", ".join(sorted(FACET_CLI_KEYS))
            raise ValueError(f"Unknown {field} facet key {key!r}; expected one of: {allowed}")
        out[FACET_CLI_KEYS[key]] = text.strip().replace("\\n", "\n")
    return out


def infer_models(rows: list[dict[str, Any]]) -> list[str]:
    available = {str(row["model"]) for row in rows}
    preferred = [
        "gpt-4o",
        "gpt-5.4",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-1",
    ]
    ordered = [model for model in preferred if model in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def model_display_label(model: str) -> str:
    return MODEL_DISPLAY_LABELS.get(model, model)


def model_color(model: str) -> str:
    fallback = [HOUSE_COLORS["blue"], HOUSE_COLORS["red"], HOUSE_COLORS["teal"], HOUSE_COLORS["purple"]]
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    return fallback[MODELS.index(model) % len(fallback)]


def model_marker(model: str) -> str:
    if model in MODEL_MARKERS:
        return MODEL_MARKERS[model]
    return ["o", "s", "^", "D"][MODELS.index(model) % 4]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing probe results: {path}")
    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"No rows found in {path}")

    out: list[dict[str, Any]] = []
    for row in rows:
        if not is_true(row.get("valid")):
            if "not in allowed labels" not in str(row.get("error", "")):
                continue
            row["alignment_type"] = "unsupported_other"
            row["alignment_type_label"] = "Other: incompatible"
            row["response_type"] = "unsupported_other"
            row["response_type_label"] = "Unsupported other"
        row["m"] = int(row["m"])
        row["false_memory_count"] = int(row["false_memory_count"])
        row["true_memory_count"] = int(row.get("true_memory_count", 8 - row["false_memory_count"]))
        row["chose_lure"] = is_true(row.get("chose_lure"))
        row["chose_truth"] = is_true(row.get("chose_truth"))
        row["choice_score"] = 1.0 if row["chose_lure"] else 0.0
        out.append(row)
    return out


def read_all_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_rows(path))
    if not rows:
        raise ValueError("No valid rows found in source CSVs")
    return rows


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def memory_tick_labels() -> list[str]:
    return [f"{8 - k}:{k}" for k in X_COUNTS]


def latex_bold_lines(label: str) -> str:
    return "\n".join(r"\textbf{" + line + "}" for line in label.split("\n"))


def add_facet_title(ax: plt.Axes, facet: tuple[str, str]) -> None:
    subtitle = FACET_SUBTITLES.get(facet, "")
    title = FACET_TITLES[facet]
    if not subtitle:
        title_obj = ax.set_title(title, loc="left", fontweight="bold", pad=5)
        title_obj.set_multialignment("left")
        return

    ax.set_title("")
    ax.text(
        0.0,
        1.145,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=MECH_INK,
        linespacing=1.05,
        clip_on=False,
    )
    ax.text(
        0.0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=MECH_GRAY,
        clip_on=False,
    )


def add_overlay_model_label(ax: plt.Axes, model: str) -> None:
    ax.text(
        0.016,
        0.925,
        model_display_label(model),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=MECH_INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        clip_on=True,
    )


def rows_for(
    rows: list[dict[str, Any]],
    *,
    model: str | None = None,
    m: int | None = None,
    private_strength: str | None = None,
    relation: str | None = None,
    count: int | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if model is not None:
        out = [row for row in out if row["model"] == model]
    if m is not None:
        out = [row for row in out if row["m"] == m]
    if private_strength is not None:
        out = [row for row in out if row["private_evidence_strength"] == private_strength]
    if relation is not None:
        out = [row for row in out if row["lure_relation"] == relation]
    if count is not None:
        out = [row for row in out if row["false_memory_count"] == count]
    return out


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def alignment_series(rows: list[dict[str, Any]], category: str) -> list[float]:
    raw_categories = set(ALIGNMENT_GROUPS[category])
    series: list[float] = []
    for count in X_COUNTS:
        cell = [row for row in rows if row["false_memory_count"] == count]
        if not cell:
            series.append(0.0)
        else:
            series.append(sum(row["alignment_type"] in raw_categories for row in cell) / len(cell))
    return series


def add_m_section_guides(fig: plt.Figure, axes: np.ndarray) -> None:
    n_models = len(MODELS)
    blocks = [
        (m_value, block_index * n_models, block_index * n_models + n_models - 1)
        for block_index, m_value in enumerate(M_VALUES)
    ]
    for block_index, (m_value, first_row, last_row) in enumerate(blocks):
        section_title = M_SECTION_TITLES.get(m_value, f"m={m_value}")
        if not section_title:
            continue
        top = axes[first_row, 0].get_position().y1
        if block_index > 0:
            prev_bottom = axes[first_row - 1, 0].get_position().y0
            next_top = axes[first_row, 0].get_position().y1
            y_separator = (prev_bottom + next_top) / 2.0
            fig.add_artist(
                Line2D(
                    [0.095, 0.985],
                    [y_separator, y_separator],
                    transform=fig.transFigure,
                    color=MECH_LIGHT_GRAY,
                    linewidth=1.1,
                )
            )
            title_y = y_separator - (0.003 if len(MODELS) > 2 else 0.010)
            va = "top"
        else:
            title_y = top + (0.026 if len(MODELS) > 2 else 0.035)
            va = "bottom"
        fig.text(
            0.215,
            title_y,
            section_title,
            ha="left",
            va=va,
            fontsize=10.0,
            fontweight="bold",
            color=MECH_INK,
        )


def draw_alignment_figure(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    setup_mechanism_like_style()
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.8,
        }
    )

    row_specs = [(m_value, model) for m_value in M_VALUES for model in MODELS]
    n_model_rows = len(MODELS)
    gap_ratio = 0.70 if n_model_rows > 2 else 0.36
    grid_rows: list[int] = []
    height_ratios: list[float] = []
    grid_row = 0
    for block_index, _m_value in enumerate(M_VALUES):
        for _model_index in range(n_model_rows):
            grid_rows.append(grid_row)
            height_ratios.append(1.0)
            grid_row += 1
        if block_index < len(M_VALUES) - 1:
            height_ratios.append(gap_ratio)
            grid_row += 1
    fig_height = max(4.80, 0.96 * len(row_specs) + 1.40) if OVERLAY_MODEL_LABELS else max(4.95, 1.18 * len(row_specs) + 1.45)
    if len(M_VALUES) > 1:
        fig_height = max(7.25, 1.20 * len(row_specs) + 1.90)
    fig_width_by_facets = {1: 3.45, 2: 5.10, 3: 7.15}
    fig_width = fig_width_by_facets.get(len(FACETS), max(7.15, 2.2 * len(FACETS) + 0.5))
    fig = plt.figure(figsize=(fig_width, fig_height))
    has_facet_subtitles = any(FACET_SUBTITLES.get(facet) for facet in FACETS)
    grid_top = 0.845 if has_facet_subtitles else (0.900 if n_model_rows > 2 else 0.820)
    if len(M_VALUES) == 1:
        grid_top = 0.805 if has_facet_subtitles else 0.875
    grid_bottom = 0.175 if n_model_rows > 2 else 0.220
    if len(M_VALUES) == 1:
        grid_bottom = 0.205
    grid = fig.add_gridspec(
        len(height_ratios),
        len(FACETS),
        height_ratios=height_ratios,
        left=0.145 if OVERLAY_MODEL_LABELS else 0.215,
        right=0.985,
        top=grid_top,
        bottom=grid_bottom,
        wspace=0.22,
        hspace=0.28 if OVERLAY_MODEL_LABELS else (0.48 if n_model_rows > 2 else 0.50),
    )
    axes = np.empty((len(row_specs), len(FACETS)), dtype=object)
    shared_ax = None
    for row_index, grid_row in enumerate(grid_rows):
        for col_index in range(len(FACETS)):
            if shared_ax is None:
                ax = fig.add_subplot(grid[grid_row, col_index])
                shared_ax = ax
            else:
                ax = fig.add_subplot(grid[grid_row, col_index], sharex=shared_ax, sharey=shared_ax)
            axes[row_index, col_index] = ax

    for row_index, (m_value, model) in enumerate(row_specs):
        for col_index, facet in enumerate(FACETS):
            private_strength, relation = facet
            ax = axes[row_index, col_index]
            cell = rows_for(
                rows,
                model=model,
                m=m_value,
                private_strength=private_strength,
                relation=relation,
            )
            y_series = [alignment_series(cell, category) for category in ALIGNMENT_STACK_ORDER]
            ax.stackplot(
                X_COUNTS,
                *y_series,
                colors=[ALIGNMENT_COLORS[category] for category in ALIGNMENT_STACK_ORDER],
                labels=[ALIGNMENT_LABELS[category] for category in ALIGNMENT_STACK_ORDER],
                linewidth=0.35,
                edgecolor="white",
            alpha=1.0,
            )
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 1.02)
            ax.set_xticks(X_COUNTS)
            ax.set_xticklabels(memory_tick_labels())
            ax.set_yticks([0.0, 0.5, 1.0])
            if col_index == 0:
                if OVERLAY_MODEL_LABELS:
                    ax.set_ylabel("")
                    ax.set_yticklabels(["0", ".5", "1"])
                else:
                    ax.set_ylabel(model_display_label(model), labelpad=8)
                    ax.set_yticklabels(["Target country", "", "Social evidence"])
            else:
                ax.tick_params(axis="y", labelleft=False)
            if OVERLAY_MODEL_LABELS:
                add_overlay_model_label(ax, model)
            if row_index < len(row_specs) - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                ax.tick_params(axis="x", bottom=True, labelbottom=True)
            if row_index in {block_index * len(MODELS) for block_index in range(len(M_VALUES))}:
                add_facet_title(ax, facet)
            style_alignment_axis(ax)

    for row_index in range(len(row_specs)):
        axes[row_index, 0].tick_params(axis="y", labelleft=True)
        if OVERLAY_MODEL_LABELS:
            axes[row_index, 0].set_yticklabels(["0", ".5", "1"])
        else:
            axes[row_index, 0].set_yticklabels(["Target country", "", "Social evidence"])

    legend_handles = [
        Patch(color=ALIGNMENT_COLORS[category], label=ALIGNMENT_LABELS[category])
        for category in ALIGNMENT_LEGEND_ORDER
    ]
    fig.legend(
        legend_handles,
        [ALIGNMENT_LABELS[category] for category in ALIGNMENT_LEGEND_ORDER],
        frameon=False,
        ncol=2 if OVERLAY_MODEL_LABELS else 4,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.058 if OVERLAY_MODEL_LABELS else (0.100 if len(MODELS) > 2 else 0.095)),
        columnspacing=1.25,
        handlelength=1.4,
    )
    fig.text(
        0.50,
        0.155 if OVERLAY_MODEL_LABELS else (0.148 if len(MODELS) > 2 else 0.160),
        "Memory entries (target country : social evidence)",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MECH_INK,
    )
    if OVERLAY_MODEL_LABELS:
        fig.text(
            0.067,
            (grid_bottom + grid_top) / 2.0,
            "Agent response probability",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.8,
            color=MECH_INK,
        )
    add_m_section_guides(fig, axes)
    return save_figure(fig, ALIGNMENT_STEM)


def draw_main_panel(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    setup_paper_house_style()
    arial_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if arial_path.exists():
        font_manager.fontManager.addfont(str(arial_path))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig_height = max(3.95, 1.35 * len(MODELS) + 1.25)
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(2.55, fig_height), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for row_index, model in enumerate(MODELS):
        ax = axes[row_index]
        cell = rows_for(
            rows,
            model=model,
            m=1,
            private_strength="weak_private_evidence",
            relation="compatible",
        )
        y_series = [alignment_series(cell, category) for category in MAIN_PANEL_STACK_ORDER]
        ax.stackplot(
            X_COUNTS,
            *y_series,
            colors=[MAIN_PANEL_COLORS[category] for category in MAIN_PANEL_STACK_ORDER],
            linewidth=0.65,
            edgecolor="white",
            alpha=1.0,
        )
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 4, 8])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"])
        ax.text(
            0.015,
            0.89,
            model_display_label(model),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=HOUSE_COLORS["black"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        )
        ax.set_ylabel("")
        if row_index == len(MODELS) - 1:
            ax.set_xticklabels(["8:0", "4:4", "0:8"])
        else:
            ax.set_xticklabels([])
        style_axis(ax, grid=True)
        ax.grid(axis="y", alpha=0.13, linewidth=0.55)
        ax.grid(axis="x", alpha=0.06, linewidth=0.45)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    legend_handles = [
        Patch(color=MAIN_PANEL_COLORS[category], label=MAIN_PANEL_LABELS[category])
        for category in MAIN_PANEL_STACK_ORDER
    ]
    fig.legend(
        legend_handles,
        ["Private target", "Other compatible", "Social evidence"],
        frameon=False,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.19, 0.075),
        handlelength=1.15,
        handleheight=0.8,
        labelspacing=0.22,
        borderaxespad=0,
    )
    fig.text(
        0.61,
        0.205,
        "Memory entries (target : social)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        0.105,
        0.64,
        "Agent response probability",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.2,
        color=HOUSE_COLORS["black"],
    )
    fig.subplots_adjust(left=0.25, right=0.985, top=0.98, bottom=0.29, hspace=0.22)
    return save_figure(fig, MAIN_PANEL_STEM)


def draw_choice_figure(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    setup_paper_house_style()
    plt.rcParams.update(
        {
            "font.size": 7.4,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.2,
        }
    )

    fig, axes = plt.subplots(
        len(M_VALUES),
        len(FACETS),
        figsize=(7.15, 3.65),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    binary_rows = [row for row in rows if row["chose_truth"] or row["chose_lure"]]

    for row_index, m_value in enumerate(M_VALUES):
        for col_index, facet in enumerate(FACETS):
            private_strength, relation = facet
            ax = axes[row_index, col_index]
            for model in MODELS:
                means: list[float] = []
                sems: list[float] = []
                color = model_color(model)
                for count in X_COUNTS:
                    cell = rows_for(
                        binary_rows,
                        model=model,
                        m=m_value,
                        private_strength=private_strength,
                        relation=relation,
                        count=count,
                    )
                    mean, sem = mean_and_sem([float(row["choice_score"]) for row in cell])
                    means.append(mean)
                    sems.append(sem)
                ax.errorbar(
                    X_COUNTS,
                    means,
                    yerr=sems,
                    color=color,
                    marker=model_marker(model),
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    linewidth=1.25,
                    capsize=2.0,
                    label=model_display_label(model),
                )
            ax.axhline(0.5, color=HOUSE_COLORS["gray"], linewidth=0.65, linestyle=":")
            ax.set_xlim(-0.2, 8.2)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(X_COUNTS)
            ax.set_xticklabels(memory_tick_labels())
            ax.set_yticks([0.0, 0.5, 1.0])
            if col_index == 0:
                ax.set_ylabel(f"m={m_value}\nAgent choice")
                ax.set_yticklabels(["Target country", "", "Social evidence"])
            if row_index == 0:
                ax.set_title(FACET_TITLES[facet], loc="left", fontweight="bold")
            style_axis(ax, grid=True)
            ax.grid(axis="y", alpha=0.14, linewidth=0.55)
            ax.grid(axis="x", alpha=0.08, linewidth=0.45)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.060),
        columnspacing=1.4,
    )
    fig.text(
        0.50,
        0.165,
        "Memory entries (target country : social evidence)",
        ha="center",
        va="center",
        fontsize=7.4,
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        0.50,
        0.020,
        (
            "Binary view: includes only trials where the agent chose either the target country "
            "or the social-evidence country. Points show means; error bars show SEM."
        ),
        ha="center",
        va="center",
        fontsize=6.4,
        color=HOUSE_COLORS["gray"],
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.25, wspace=0.25, hspace=0.50)
    return save_figure(fig, CHOICE_STEM)


def summarize_alignment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in MODELS:
        for m_value in M_VALUES:
            for private_strength, relation in FACETS:
                for count in X_COUNTS:
                    cell = rows_for(
                        rows,
                        model=model,
                        m=m_value,
                        private_strength=private_strength,
                        relation=relation,
                        count=count,
                    )
                    if not cell:
                        continue
                    row: dict[str, Any] = {
                        "model": model,
                        "m": m_value,
                        "private_evidence_strength": private_strength,
                        "social_evidence_relation": relation,
                        "target_memory_entries": 8 - count,
                        "social_memory_entries": count,
                        "n_trials": len(cell),
                    }
                    for category in ALIGNMENT_LEGEND_ORDER:
                        raw_categories = set(ALIGNMENT_GROUPS[category])
                        n = sum(item["alignment_type"] in raw_categories for item in cell)
                        row[f"{category}_count"] = n
                        row[f"{category}_rate"] = n / len(cell)
                    out.append(row)
    return out


def aggregate_rate(
    rows: list[dict[str, Any]],
    *,
    model: str,
    m: int,
    private_strength: str | None = None,
    relation: str | None = None,
    category: str,
) -> float:
    cell = rows_for(rows, model=model, m=m, private_strength=private_strength, relation=relation)
    if not cell:
        return float("nan")
    raw_categories = set(ALIGNMENT_GROUPS.get(category, (category,)))
    return sum(row["alignment_type"] in raw_categories for row in cell) / len(cell)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_findings_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Memory Conflict Probe Findings",
        "",
        f"- Source: `{payload['source_csv']}`.",
        f"- Valid trials: {payload['n_valid_trials']}.",
        "",
        "## Key Rates",
        "",
    ]
    for model in MODELS:
        label = model_display_label(model)
        rates = payload["key_rates"][model]
        lines.extend(
            [
                f"### {label}",
                "",
                "- Other compatible country under weak private evidence/social evidence agrees: "
                f"{rates['other_compatible_weak_agrees_m1']:.1%} at m=1; "
                f"{rates['other_compatible_weak_agrees_m3']:.1%} at m=3.",
                "- Direct social-evidence country choice overall: "
                f"{rates['social_overall_m1']:.1%} at m=1; "
                f"{rates['social_overall_m3']:.1%} at m=3.",
                "- Incompatible other under strong private evidence/social evidence contradicts: "
                f"{rates['incompatible_strong_contradicts_m1']:.1%} at m=1; "
                f"{rates['incompatible_strong_contradicts_m3']:.1%} at m=3.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_summary_payload(rows: list[dict[str, Any]], outputs: dict[str, str]) -> dict[str, Any]:
    key_rates: dict[str, dict[str, float]] = {}
    for model in MODELS:
        key_rates[model] = {
            "other_compatible_weak_agrees_m1": aggregate_rate(
                rows,
                model=model,
                m=1,
                private_strength="weak_private_evidence",
                relation="compatible",
                category="other_private_compatible",
            ),
            "other_compatible_weak_agrees_m3": aggregate_rate(
                rows,
                model=model,
                m=3,
                private_strength="weak_private_evidence",
                relation="compatible",
                category="other_private_compatible",
            ),
            "social_overall_m1": aggregate_rate(rows, model=model, m=1, category="social_evidence"),
            "social_overall_m3": aggregate_rate(rows, model=model, m=3, category="social_evidence"),
            "incompatible_strong_contradicts_m1": aggregate_rate(
                rows,
                model=model,
                m=1,
                private_strength="strong_private_evidence",
                relation="incompatible",
                category="other_incompatible",
            ),
            "incompatible_strong_contradicts_m3": aggregate_rate(
                rows,
                model=model,
                m=3,
                private_strength="strong_private_evidence",
                relation="incompatible",
                category="other_incompatible",
            ),
        }
    return {
        "source_csv": [str(path) for path in SOURCE_CSVS],
        "n_valid_trials": len(rows),
        "models": MODELS,
        "outputs": outputs,
        "key_rates": key_rates,
        "interpretation_notes": [
            "The paper-facing alignment figure merges third-country answers into private-compatible versus incompatible outcomes.",
            "The m=1 versus m=3 contrast is label-only social memory versus social memory with reasons.",
            "The incompatible band should be framed as unsupported choices, not as evidence integration.",
        ],
    }


def main() -> None:
    global SOURCE_CSV, SOURCE_CSVS, FIGURE_DIR, DATA_DIR, ALIGNMENT_STEM, CHOICE_STEM, MAIN_PANEL_STEM
    global SUMMARY_STEM, MODELS, M_VALUES, FACETS, FACET_TITLES, FACET_SUBTITLES
    global OVERLAY_MODEL_LABELS, M_SECTION_TITLES

    args = parse_args()
    SOURCE_CSVS = args.source_csvs or [SOURCE_CSV]
    SOURCE_CSV = SOURCE_CSVS[0]
    FIGURE_DIR = args.figure_dir
    DATA_DIR = args.data_dir
    SUMMARY_STEM = args.stem_prefix
    ALIGNMENT_STEM = f"{args.stem_prefix}_alignment"
    CHOICE_STEM = f"{args.stem_prefix}_choice_axis"
    MAIN_PANEL_STEM = f"{args.stem_prefix}_main_panel"
    M_VALUES = args.m_values or M_VALUES
    FACETS = [FACET_CLI_KEYS[key] for key in args.facets] if args.facets else FACETS
    FACET_TITLES = {
        **FACET_TITLES,
        **parse_facet_text_overrides(args.facet_titles, field="facet-title"),
    }
    FACET_SUBTITLES = parse_facet_text_overrides(args.facet_subtitles, field="facet-subtitle")
    OVERLAY_MODEL_LABELS = args.overlay_model_labels
    M_SECTION_TITLES = {
        1: "" if args.hide_section_titles else args.section_title_m1,
        3: "" if args.hide_section_titles else args.section_title_m3,
    }

    rows = read_all_rows(SOURCE_CSVS)
    MODELS = args.models or infer_models(rows)
    alignment_png, alignment_pdf = draw_alignment_figure(rows)
    if args.alignment_only:
        print(f"Wrote {alignment_png}")
        print(f"Wrote {alignment_pdf}")
        print(f"Wrote {FIGURE_DIR / f'{ALIGNMENT_STEM}.svg'}")
        return

    main_panel_png, main_panel_pdf = draw_main_panel(rows)
    choice_png, choice_pdf = draw_choice_figure(rows)

    summary_csv = DATA_DIR / f"{SUMMARY_STEM}_alignment_summary.csv"
    summary_json = DATA_DIR / f"{SUMMARY_STEM}_summary.json"
    findings_md = DATA_DIR / f"{SUMMARY_STEM}_findings.md"

    write_csv(summary_csv, summarize_alignment(rows))
    payload = build_summary_payload(
        rows,
        {
            "alignment_png": str(alignment_png),
            "alignment_pdf": str(alignment_pdf),
            "main_panel_png": str(main_panel_png),
            "main_panel_pdf": str(main_panel_pdf),
            "choice_png": str(choice_png),
            "choice_pdf": str(choice_pdf),
            "alignment_summary_csv": str(summary_csv),
        },
    )
    write_json(summary_json, payload)
    write_findings_md(findings_md, payload)

    print(f"Wrote {alignment_png}")
    print(f"Wrote {alignment_pdf}")
    print(f"Wrote {main_panel_png}")
    print(f"Wrote {main_panel_pdf}")
    print(f"Wrote {choice_png}")
    print(f"Wrote {choice_pdf}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {findings_md}")


if __name__ == "__main__":
    main()

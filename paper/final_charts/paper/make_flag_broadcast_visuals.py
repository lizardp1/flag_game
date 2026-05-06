#!/usr/bin/env python3
"""Make paper-facing Broadcast Flag Game figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/memetic-drift-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager

from plot_style import HOUSE_COLORS, setup_paper_house_style, style_axis


ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUMMARY_CSV = (
    ROOT / "poster" / "exports" / "support_visuals" / "flag_broadcast_alpha_mix_N8_m3_summary.csv"
)
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"

ALPHA_MAIN_STEM = "flag_game_broadcast_m3_alpha_main"
DIVERSITY_MAIN_STEM = "flag_game_broadcast_m3_diversity_main"
DIVERSITY_METRIC_STEM = f"{DIVERSITY_MAIN_STEM}_with_empirical_metrics"
APPENDIX_STEM = "flag_game_broadcast_m3_appendix_phase_surface"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
COUNTS = list(range(9))
SUCCESS_THRESHOLD = 0.85

CIQ_CMAP = LinearSegmentedColormap.from_list(
    "broadcast_ciq",
    ["#F7FBFF", "#9ECAE1", HOUSE_COLORS["blue"]],
)
VOTE_CMAP = LinearSegmentedColormap.from_list(
    "broadcast_vote_accuracy",
    ["#F7FCF5", "#8ED1C2", HOUSE_COLORS["teal"]],
)
LIFT_CMAP = LinearSegmentedColormap.from_list(
    "broadcast_lift",
    ["#F7FCF5", "#A1D99B", HOUSE_COLORS["teal"]],
)

EMPIRICAL_BLUE_DARK = "#1764B5"
EMPIRICAL_BLUE_LIGHT = "#D7ECFF"
EMPIRICAL_GREEN = "#00A36F"
EMPIRICAL_GRAY = "#747C85"
EMPIRICAL_LIGHT_GRAY = "#D9DEE5"
EMPIRICAL_INK = HOUSE_COLORS["black"]
PAPER_SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]


def add_paper_sans_font() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return


def apply_paper_sans_rc() -> None:
    add_paper_sans_font()
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": PAPER_SANS_FONTS,
            "mathtext.fontset": "dejavusans",
            "font.size": 8.3,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No broadcast summary rows found in {path}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def lookup(rows: list[dict[str, str]], alpha: float, count: int, key: str) -> float:
    for row in rows:
        if float(row["alpha"]) == alpha and int(row["gpt54_count"]) == count:
            return float(row[key])
    raise KeyError(f"No row for alpha={alpha}, gpt54_count={count}, key={key}")


def alpha_one_composition(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    return [
        {
            "gpt54_count": float(count),
            "iiq": lookup(rows, 1.0, count, "initial_accuracy_mean"),
            "ciq": lookup(rows, 1.0, count, "final_accuracy_mean"),
            "vote_accuracy": lookup(rows, 1.0, count, "final_vote_accuracy_rate"),
            "social_lift": lookup(rows, 1.0, count, "social_lift_mean"),
            "success85_rate": lookup(rows, 1.0, count, "success85_rate"),
            "n_trials": lookup(rows, 1.0, count, "n_trials"),
        }
        for count in COUNTS
    ]


def alpha_marginal(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for alpha in ALPHAS:
        group = [row for row in rows if float(row["alpha"]) == alpha]
        out.append(
            {
                "alpha": alpha,
                "ciq": float(np.mean([float(row["final_accuracy_mean"]) for row in group])),
                "vote_accuracy": float(np.mean([float(row["final_vote_accuracy_rate"]) for row in group])),
                "social_lift": float(np.mean([float(row["social_lift_mean"]) for row in group])),
                "success85_rate": float(np.mean([float(row["success85_rate"]) for row in group])),
                "n_cells": float(len(group)),
                "n_trials": float(sum(float(row["n_trials"]) for row in group)),
            }
        )
    return out


def metric_matrix(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    data = np.full((len(COUNTS), len(ALPHAS)), np.nan)
    for y, count in enumerate(COUNTS):
        for x, alpha in enumerate(ALPHAS):
            data[y, x] = lookup(rows, alpha, count, metric)
    return data


def draw_alpha_figure(
    rows: list[dict[str, str]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = ALPHA_MAIN_STEM,
) -> tuple[Path, Path, dict[str, Any]]:
    setup_paper_house_style()
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
        }
    )

    marginal = alpha_marginal(rows)

    blue = HOUSE_COLORS["blue"]
    x_alpha = np.array([row["alpha"] for row in marginal], dtype=float)
    y_alpha = np.array([row["ciq"] for row in marginal], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.65), constrained_layout=False)
    ax.plot(
        x_alpha,
        y_alpha,
        color=blue,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=blue,
        markeredgewidth=1.5,
        linewidth=1.7,
        label="CIQ" if include_vote_accuracy else None,
        zorder=3,
    )
    if include_vote_accuracy:
        vote_alpha = np.array([row["vote_accuracy"] for row in marginal], dtype=float)
        ax.plot(
            x_alpha,
            vote_alpha,
            color=HOUSE_COLORS["teal"],
            marker="s",
            markerfacecolor="white",
            markeredgecolor=HOUSE_COLORS["teal"],
            markeredgewidth=1.2,
            linewidth=1.35,
            linestyle="--",
            label="Final vote accuracy",
            zorder=3,
        )

    if not include_vote_accuracy:
        for x, y in zip(x_alpha, y_alpha):
            ax.text(
                float(x),
                float(y) + 0.018,
                f"{float(y):.2f}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                color=HOUSE_COLORS["black"],
            )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.50, 1.02 if include_vote_accuracy else 0.94)
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel(r"Prompted social evidence uptake $\alpha$")
    ax.set_ylabel("Mean CIQ / accuracy rate" if include_vote_accuracy else "Mean CIQ")
    ax.set_title(r"\textbf{Broadcast: social evidence uptake}", loc="left", fontweight="bold")
    style_axis(ax, grid=True)
    if include_vote_accuracy:
        ax.legend(frameon=False, loc="lower right", fontsize=6.8, handlelength=2.0)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.20)

    png_path, pdf_path = save_figure(fig, stem)
    summary = {
        "source_summary_csv": str(SOURCE_SUMMARY_CSV),
        "outputs": [str(png_path), str(pdf_path)],
        "description": "m=3 prompted social evidence uptake marginal averaged over all GPT-5.4 counts",
        "final_vote_accuracy_definition": (
            "For broadcast runs, final vote accuracy is 1 when the unique top-voted final "
            "country is the truth country; ties receive 0."
        ),
        "data": marginal,
    }
    return png_path, pdf_path, summary


def high_ciq_region(composition: list[dict[str, float]], *, tolerance: float = 0.025) -> tuple[float, float]:
    counts = np.array([row["gpt54_count"] for row in composition], dtype=float)
    ciq = np.array([row["ciq"] for row in composition], dtype=float)
    if counts.size == 0:
        return 0.5, 7.5
    threshold = float(np.nanmax(ciq) - tolerance)
    high_indices = np.flatnonzero(ciq >= threshold)
    if high_indices.size == 0:
        return float(counts[0] - 0.5), float(counts[-1] + 0.5)

    runs: list[np.ndarray] = np.split(high_indices, np.where(np.diff(high_indices) != 1)[0] + 1)
    best_run = max(runs, key=len)
    left = float(counts[int(best_run[0])] - 0.5)
    right = float(counts[int(best_run[-1])] + 0.5)
    return left, right


def draw_diversity_metric_figure(
    rows: list[dict[str, str]],
    *,
    stem: str = DIVERSITY_METRIC_STEM,
) -> tuple[Path, Path, dict[str, Any]]:
    setup_paper_house_style()
    apply_paper_sans_rc()
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
        }
    )

    composition = alpha_one_composition(rows)
    x = np.array([row["gpt54_count"] for row in composition], dtype=float)
    iiq = np.array([row["iiq"] for row in composition], dtype=float)
    ciq = np.array([row["ciq"] for row in composition], dtype=float)
    vote_accuracy = np.array([row["vote_accuracy"] for row in composition], dtype=float)
    high_left, high_right = high_ciq_region(composition)

    fig, ax = plt.subplots(1, 1, figsize=(3.75, 3.2), constrained_layout=False)
    ax.fill_between(
        x,
        iiq,
        ciq,
        color=EMPIRICAL_BLUE_LIGHT,
        alpha=0.68,
        linewidth=0,
        zorder=1,
    )
    for edge in (high_left, high_right):
        ax.axvline(
            edge,
            color=EMPIRICAL_BLUE_DARK,
            linewidth=1.0,
            linestyle=(0, (1.1, 1.8)),
            alpha=0.64,
            ymin=0.03,
            ymax=0.965,
            zorder=2,
        )
    ax.plot(
        x,
        iiq,
        color=EMPIRICAL_GRAY,
        marker="o",
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linewidth=1.45,
        linestyle=(0, (2.0, 1.4)),
        zorder=3,
    )
    ax.plot(
        x,
        vote_accuracy,
        color=EMPIRICAL_GREEN,
        marker="^",
        markersize=4.7,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linewidth=1.55,
        zorder=4,
    )
    ax.plot(
        x,
        ciq,
        color=EMPIRICAL_BLUE_DARK,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.55,
        linewidth=2.05,
        zorder=5,
    )
    ax.text(
        (high_left + high_right) / 2,
        0.79,
        "highest collective accuracy",
        ha="center",
        va="top",
        fontsize=6.8,
        color=EMPIRICAL_BLUE_DARK,
    )
    ax.text(
        4.0,
        0.225,
        "mixed teams",
        ha="center",
        va="center",
        fontsize=7.0,
        color=EMPIRICAL_GRAY,
    )

    ax.set_xlim(-0.25, 8.25)
    ax.set_ylim(0.2, 0.8)
    ax.set_xticks(COUNTS)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("GPT-5.4 agents in team (of 8)", labelpad=3)
    ax.set_ylabel("Accuracy")
    ax.set_title("Broadcast: team diversity", loc="left", fontweight="normal", color=EMPIRICAL_INK, pad=4)
    ax.grid(axis="y", color=EMPIRICAL_LIGHT_GRAY, alpha=0.38, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(EMPIRICAL_INK)
    ax.spines["bottom"].set_color(EMPIRICAL_INK)
    ax.tick_params(colors=EMPIRICAL_INK)
    ax.yaxis.label.set_color(EMPIRICAL_INK)
    ax.xaxis.label.set_color(EMPIRICAL_INK)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=EMPIRICAL_BLUE_DARK,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.55,
                linewidth=2.05,
                label="Collective",
            ),
            Line2D(
                [0],
                [0],
                color=EMPIRICAL_GREEN,
                marker="^",
                markerfacecolor="white",
                markeredgewidth=1.25,
                linewidth=1.55,
                label="Majority vote",
            ),
            Line2D(
                [0],
                [0],
                color=EMPIRICAL_GRAY,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.25,
                linewidth=1.45,
                linestyle=(0, (2.0, 1.4)),
                label="Initial",
            ),
            Patch(facecolor=EMPIRICAL_BLUE_LIGHT, edgecolor="none", label="Social uplift"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        labelcolor=EMPIRICAL_INK,
        handlelength=1.45,
        handletextpad=0.42,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.145, right=0.985, top=0.84, bottom=0.42)

    png_path, pdf_path = save_figure(fig, stem)
    summary = {
        "source_summary_csv": str(SOURCE_SUMMARY_CSV),
        "outputs": [str(png_path), str(pdf_path)],
        "description": (
            "prompted social evidence uptake alpha=1, m=3 broadcast team diversity "
            "with initial accuracy, collective accuracy, majority-vote accuracy, and social uplift"
        ),
        "definitions": {
            "initial": "initial private observer accuracy",
            "collective": "mean final group accuracy after broadcast social exchange",
            "majority_vote": (
                "1 when the unique top-voted final country is the truth country; "
                "ties receive 0"
            ),
            "social_uplift": "collective - initial",
            "high_ciq_region": (
                "Dotted edges mark the contiguous composition region within 0.025 "
                "of the maximum collective accuracy."
            ),
        },
        "high_ciq_region_edges": {"left": high_left, "right": high_right},
        "data": composition,
    }
    return png_path, pdf_path, summary


def draw_diversity_figure(
    rows: list[dict[str, str]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = DIVERSITY_MAIN_STEM,
) -> tuple[Path, Path, dict[str, Any]]:
    setup_paper_house_style()
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
        }
    )

    composition = alpha_one_composition(rows)

    blue = HOUSE_COLORS["blue"]
    x_comp = np.array([row["gpt54_count"] for row in composition], dtype=float)
    y_comp = np.array([row["ciq"] for row in composition], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.65), constrained_layout=False)
    ax.axvspan(0.5, 7.5, color=blue, alpha=0.07, zorder=0)
    ax.plot(
        x_comp,
        y_comp,
        color=blue,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=blue,
        markeredgewidth=1.5,
        linewidth=1.7,
        label="CIQ" if include_vote_accuracy else None,
        zorder=3,
    )
    if include_vote_accuracy:
        vote_comp = np.array([row["vote_accuracy"] for row in composition], dtype=float)
        ax.plot(
            x_comp,
            vote_comp,
            color=HOUSE_COLORS["teal"],
            marker="s",
            markerfacecolor="white",
            markeredgecolor=HOUSE_COLORS["teal"],
            markeredgewidth=1.2,
            linewidth=1.35,
            linestyle="--",
            label="Final vote accuracy",
            zorder=3,
        )
    ax.text(
        4.0,
        0.535,
        "mixed teams",
        ha="center",
        va="center",
        fontsize=7.4,
        color=HOUSE_COLORS["gray"],
    )
    if not include_vote_accuracy:
        for x, y in zip(x_comp, y_comp):
            x_text = float(x)
            ha = "center"
            if np.isclose(x_text, 0.0):
                x_text += 0.12
                ha = "left"
            ax.text(
                x_text,
                float(y) + 0.018,
                f"{float(y):.2f}",
                ha=ha,
                va="bottom",
                fontsize=6.7,
                color=HOUSE_COLORS["black"],
            )

    ax.set_xlim(-0.25, 8.25)
    ax.set_ylim(0.50, 1.02 if include_vote_accuracy else 0.94)
    ax.set_xticks(COUNTS)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("GPT-5.4 agents in team (of 8)")
    ax.set_ylabel("Mean CIQ / accuracy rate" if include_vote_accuracy else "Mean CIQ")
    ax.set_title(r"\textbf{Broadcast: team diversity}", loc="left", fontweight="bold")
    style_axis(ax, grid=True)
    if include_vote_accuracy:
        ax.legend(frameon=False, loc="lower right", fontsize=6.8, handlelength=2.0)

    fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.20)

    png_path, pdf_path = save_figure(fig, stem)
    summary = {
        "source_summary_csv": str(SOURCE_SUMMARY_CSV),
        "outputs": [str(png_path), str(pdf_path)],
        "description": "prompted social evidence uptake alpha=1, m=3 mean CIQ by GPT-5.4 count",
        "final_vote_accuracy_definition": (
            "For broadcast runs, final vote accuracy is 1 when the unique top-voted final "
            "country is the truth country; ties receive 0."
        ),
        "data": composition,
    }
    return png_path, pdf_path, summary


def draw_heatmap(ax: plt.Axes, data: np.ndarray, *, title: str, cmap: Any, norm: Normalize) -> None:
    image = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(rf"\textbf{{{title}}}", loc="left", fontweight="bold", pad=4)
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_yticks(range(len(COUNTS)))
    ax.set_yticklabels([str(count) for count in COUNTS])
    ax.tick_params(length=0)
    ax.set_xlabel(r"Prompted social evidence uptake $\alpha$")
    threshold = norm.vmin + 0.62 * (norm.vmax - norm.vmin)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = float(data[row, col])
            color = "white" if value >= threshold else HOUSE_COLORS["black"]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.8, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def draw_appendix_figure(
    rows: list[dict[str, str]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = APPENDIX_STEM,
) -> tuple[Path, Path, dict[str, Any]]:
    setup_paper_house_style()
    plt.rcParams.update(
        {
            "font.size": 7.7,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
        }
    )

    ciq = metric_matrix(rows, "final_accuracy_mean")
    vote_accuracy = metric_matrix(rows, "final_vote_accuracy_rate")
    lift = metric_matrix(rows, "social_lift_mean")
    ciq_norm = Normalize(vmin=0.45, vmax=0.92)
    vote_norm = Normalize(vmin=0.0, vmax=1.0)
    lift_norm = Normalize(vmin=0.0, vmax=0.55)

    ncols = 3 if include_vote_accuracy else 2
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(7.15, 2.85 if include_vote_accuracy else 3.05),
        constrained_layout=False,
    )
    ax_ciq = axes[0]
    ax_lift = axes[2] if include_vote_accuracy else axes[1]
    ciq_image = draw_heatmap(
        ax_ciq,
        ciq,
        title="A. Mean CIQ" if include_vote_accuracy else "A. Mean CIQ across social evidence uptake and composition",
        cmap=CIQ_CMAP,
        norm=ciq_norm,
    )
    if include_vote_accuracy:
        ax_vote = axes[1]
        vote_image = draw_heatmap(
            ax_vote,
            vote_accuracy,
            title="B. Final vote accuracy",
            cmap=VOTE_CMAP,
            norm=vote_norm,
        )
    lift_image = draw_heatmap(
        ax_lift,
        lift,
        title="C. Social uplift" if include_vote_accuracy else "B. Social uplift across social evidence uptake and composition",
        cmap=LIFT_CMAP,
        norm=lift_norm,
    )
    ax_ciq.set_ylabel("GPT-5.4 agents in team (of 8)")
    if include_vote_accuracy:
        ax_vote.set_ylabel("")
    ax_lift.set_ylabel("")

    cbar_ciq = fig.colorbar(ciq_image, ax=ax_ciq, orientation="horizontal", fraction=0.055, pad=0.16)
    if include_vote_accuracy:
        cbar_vote = fig.colorbar(vote_image, ax=ax_vote, orientation="horizontal", fraction=0.055, pad=0.16)
    cbar_lift = fig.colorbar(lift_image, ax=ax_lift, orientation="horizontal", fraction=0.055, pad=0.16)
    cbar_ciq.set_label("Mean CIQ", fontsize=7.2)
    if include_vote_accuracy:
        cbar_vote.set_label("Accuracy rate", fontsize=7.2)
    cbar_lift.set_label("Social uplift", fontsize=7.2)
    cbar_ciq.ax.tick_params(labelsize=6.8, length=2)
    if include_vote_accuracy:
        cbar_vote.ax.tick_params(labelsize=6.8, length=2)
    cbar_lift.ax.tick_params(labelsize=6.8, length=2)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.86 if include_vote_accuracy else 0.88,
        bottom=0.25 if include_vote_accuracy else 0.24,
        wspace=0.20,
    )
    fig.text(
        0.50,
        0.055,
        "Broadcast Flag Game, N=8, m=3. Each heatmap cell is a mean across available seeds.",
        ha="center",
        va="center",
        fontsize=7.2,
        color=HOUSE_COLORS["gray"],
    )

    png_path, pdf_path = save_figure(fig, stem)
    summary = {
        "source_summary_csv": str(SOURCE_SUMMARY_CSV),
        "outputs": [str(png_path), str(pdf_path)],
        "description": "Appendix heatmap surface for current m=3 broadcast prompted social evidence uptake/composition sweep.",
        "final_vote_accuracy_definition": (
            "For broadcast runs, final vote accuracy is 1 when the unique top-voted final "
            "country is the truth country; ties receive 0."
        ),
        "alphas": ALPHAS,
        "gpt54_counts": COUNTS,
        "matrices": {
            "final_accuracy_mean": ciq.tolist(),
            "final_vote_accuracy_rate": vote_accuracy.tolist(),
            "social_lift_mean": lift.tolist(),
        },
        "fixed_color_scales": {
            "final_accuracy_mean": {"vmin": ciq_norm.vmin, "vmax": ciq_norm.vmax},
            "final_vote_accuracy_rate": {"vmin": vote_norm.vmin, "vmax": vote_norm.vmax},
            "social_lift_mean": {"vmin": lift_norm.vmin, "vmax": lift_norm.vmax},
        },
    }
    return png_path, pdf_path, summary


def main() -> None:
    rows = read_summary_rows(SOURCE_SUMMARY_CSV)
    alpha_png, alpha_pdf, alpha_summary = draw_alpha_figure(rows)
    alpha_v2_png, alpha_v2_pdf, alpha_v2_summary = draw_alpha_figure(
        rows,
        include_vote_accuracy=True,
        stem=f"{ALPHA_MAIN_STEM}_v2",
    )
    diversity_png, diversity_pdf, diversity_summary = draw_diversity_figure(rows)
    diversity_v2_png, diversity_v2_pdf, diversity_v2_summary = draw_diversity_figure(
        rows,
        include_vote_accuracy=True,
        stem=f"{DIVERSITY_MAIN_STEM}_v2",
    )
    diversity_metric_png, diversity_metric_pdf, diversity_metric_summary = draw_diversity_metric_figure(
        rows,
        stem=DIVERSITY_METRIC_STEM,
    )
    appendix_png, appendix_pdf, appendix_summary = draw_appendix_figure(rows)
    appendix_v2_png, appendix_v2_pdf, appendix_v2_summary = draw_appendix_figure(
        rows,
        include_vote_accuracy=True,
        stem=f"{APPENDIX_STEM}_v2",
    )

    write_json(DATA_DIR / f"{ALPHA_MAIN_STEM}_summary.json", alpha_summary)
    write_json(DATA_DIR / f"{ALPHA_MAIN_STEM}_v2_summary.json", alpha_v2_summary)
    write_json(DATA_DIR / f"{DIVERSITY_MAIN_STEM}_summary.json", diversity_summary)
    write_json(DATA_DIR / f"{DIVERSITY_MAIN_STEM}_v2_summary.json", diversity_v2_summary)
    write_json(DATA_DIR / f"{DIVERSITY_METRIC_STEM}_summary.json", diversity_metric_summary)
    write_json(DATA_DIR / f"{APPENDIX_STEM}_summary.json", appendix_summary)
    write_json(DATA_DIR / f"{APPENDIX_STEM}_v2_summary.json", appendix_v2_summary)

    print(f"saved {alpha_png}")
    print(f"saved {alpha_pdf}")
    print(f"saved {DATA_DIR / f'{ALPHA_MAIN_STEM}_summary.json'}")
    print(f"saved {alpha_v2_png}")
    print(f"saved {alpha_v2_pdf}")
    print(f"saved {DATA_DIR / f'{ALPHA_MAIN_STEM}_v2_summary.json'}")
    print(f"saved {diversity_png}")
    print(f"saved {diversity_pdf}")
    print(f"saved {DATA_DIR / f'{DIVERSITY_MAIN_STEM}_summary.json'}")
    print(f"saved {diversity_v2_png}")
    print(f"saved {diversity_v2_pdf}")
    print(f"saved {DATA_DIR / f'{DIVERSITY_MAIN_STEM}_v2_summary.json'}")
    print(f"saved {diversity_metric_png}")
    print(f"saved {diversity_metric_pdf}")
    print(f"saved {DATA_DIR / f'{DIVERSITY_METRIC_STEM}_summary.json'}")
    print(f"saved {appendix_png}")
    print(f"saved {appendix_pdf}")
    print(f"saved {DATA_DIR / f'{APPENDIX_STEM}_summary.json'}")
    print(f"saved {appendix_v2_png}")
    print(f"saved {appendix_v2_pdf}")
    print(f"saved {DATA_DIR / f'{APPENDIX_STEM}_v2_summary.json'}")


if __name__ == "__main__":
    main()

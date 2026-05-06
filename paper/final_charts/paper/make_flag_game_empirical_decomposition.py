#!/usr/bin/env python3
"""Make the empirical Flag Game decomposition figure."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

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


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"

PAIRWISE_JSON = DATA_DIR / "flag_game_pairwise_n_scaling_main_v3_summary.json"
PAIRWISE_ALPHA_CSV = DATA_DIR / "flag_game_pairwise_alpha_main_candidate_summary.csv"
BROADCAST_CSV = ROOT / "poster" / "exports" / "support_visuals" / "flag_broadcast_alpha_mix_N8_m3_summary.csv"
PROTOCOL_CSV = DATA_DIR / "flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv"

STEM = "flag_game_empirical_decomposition"

BLUE = "#2D8CFF"
BLUE_DARK = "#1764B5"
BLUE_LIGHT = "#D7ECFF"
GREEN = "#00A36F"
GREEN_LIGHT = "#C9F2E5"
ORANGE = "#FF8A3D"
RED = "#D34A3A"
PURPLE = "#7C6FB6"
GRAY = "#747C85"
LIGHT_GRAY = "#D9DEE5"
INK = "#25272B"

OUTCOME_COLORS = {
    "correct_consensus": GREEN,
    "wrong_consensus": ORANGE,
    "polarization": PURPLE,
}

PROTOCOL_COLORS = {
    "pairwise": ORANGE,
    "broadcast": PURPLE,
    "org": GREEN,
}


def add_arial() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def setup_style() -> None:
    add_arial()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 9.4,
            "axes.labelsize": 9.8,
            "axes.titlesize": 10.1,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.6,
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def safe_float(value: str | float | int | None, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    return float(value)


def style_metric_axis(ax: plt.Axes, *, xlabel: str, ymax: float = 1.03) -> None:
    ax.set_ylim(0.0, ymax)
    if ymax <= 0.75:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75])
    else:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel, labelpad=3)
    ax.grid(axis="y", color=LIGHT_GRAY, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold", color=INK, pad=4)


def draw_metric_lines(
    ax: plt.Axes,
    x: np.ndarray,
    iiq: np.ndarray,
    terminal: np.ndarray,
    ciq: np.ndarray,
    *,
    xlabel: str,
    categorical: bool = False,
    ymax: float = 1.03,
) -> None:
    ax.fill_between(x, iiq, ciq, color=BLUE_LIGHT, alpha=0.75, linewidth=0, zorder=0)
    ax.plot(
        x,
        iiq,
        color=GRAY,
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
        terminal,
        color=GREEN,
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
        color=BLUE_DARK,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.55,
        linewidth=2.05,
        zorder=5,
    )
    style_metric_axis(ax, xlabel=xlabel, ymax=ymax)
    if categorical:
        ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)


def draw_endpoint_lines(ax: plt.Axes, rows: list[dict[str, float]]) -> None:
    x = np.array([row["x"] for row in rows], dtype=float)
    specs = [
        ("correct_consensus", "Correct consensus", "o"),
        ("wrong_consensus", "Wrong consensus", "s"),
        ("polarization", "Polarization", "^"),
    ]
    for key, label, marker in specs:
        y = np.array([row[key] for row in rows], dtype=float)
        ax.plot(
            x,
            y,
            color=OUTCOME_COLORS[key],
            marker=marker,
            markersize=4.9,
            markerfacecolor="white",
            markeredgewidth=1.35,
            linewidth=1.85,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows])
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Fraction of runs")
    ax.set_xlabel("Population size N", labelpad=3)
    ax.grid(axis="y", color=LIGHT_GRAY, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=1.65, borderaxespad=0.15)


def draw_social_lift_panel(
    ax: plt.Axes,
    rows: list[dict[str, float]],
    *,
    color: str,
    marker: str,
) -> None:
    x = np.array([row["x"] for row in rows], dtype=float)
    y = np.array([row["social_lift"] for row in rows], dtype=float)
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.45,
        linewidth=1.95,
        zorder=4,
    )
    ax.axhline(0.0, color=LIGHT_GRAY, linewidth=0.9, zorder=0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_ylim(-0.02, 0.43)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylabel("Social uplift")
    ax.set_xlabel("Prompted social evidence uptake", labelpad=3)
    ax.grid(axis="y", color=LIGHT_GRAY, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)


def pairwise_rows() -> list[dict[str, float]]:
    payload = load_json(PAIRWISE_JSON)
    rows = [
        row
        for row in payload["summary_rows"]
        if str(row["condition"]) == "all_gpt_4o"
    ]
    return [
        {
            "cut": "population",
            "x": float(row["N"]),
            "label": str(row["N"]),
            "iiq": float(row["initial_mean"]),
            "ciq": float(row["final_mean"]),
            "terminal": float(row["final_vote_accuracy_rate"]),
            "social_lift": float(row["final_mean"]) - float(row["initial_mean"]),
            "correct_consensus": float(row["correct_consensus_rate"]),
            "wrong_consensus": float(row["wrong_consensus_rate"]),
            "polarization": float(row["polarization_rate"]),
        }
        for row in sorted(rows, key=lambda item: int(item["N"]))
    ]


def broadcast_alpha_rows() -> list[dict[str, float]]:
    rows = read_csv(BROADCAST_CSV)
    alphas = sorted({safe_float(row["alpha"]) for row in rows})
    out: list[dict[str, float]] = []
    for alpha in alphas:
        group = [row for row in rows if safe_float(row["alpha"]) == alpha]
        iiq = mean(safe_float(row["initial_accuracy_mean"]) for row in group)
        ciq = mean(safe_float(row["final_accuracy_mean"]) for row in group)
        out.append(
            {
                "cut": "alpha",
                "protocol": "broadcast",
                "x": alpha,
                "label": f"{alpha:g}",
                "social_lift": ciq - iiq,
                "ciq": ciq,
            }
        )
    return out


def pairwise_alpha_rows() -> list[dict[str, float]]:
    rows = [
        row
        for row in read_csv(PAIRWISE_ALPHA_CSV)
        if row["source"] == "poster_alpha_sweep" and row["model"] == "gpt4o"
    ]
    out: list[dict[str, float]] = []
    for row in sorted(rows, key=lambda item: safe_float(item["alpha"])):
        iiq = safe_float(row["initial_mean"])
        ciq = safe_float(row["final_mean"])
        out.append(
            {
                "cut": "alpha",
                "protocol": "pairwise",
                "x": safe_float(row["alpha"]),
                "label": f"{safe_float(row['alpha']):g}",
                "social_lift": ciq - iiq,
                "ciq": ciq,
            }
        )
    return out


def broadcast_diversity_rows() -> list[dict[str, float]]:
    rows = [row for row in read_csv(BROADCAST_CSV) if safe_float(row["alpha"]) == 1.0]
    counts = sorted({int(safe_float(row["gpt54_count"])) for row in rows})
    out: list[dict[str, float]] = []
    for count in counts:
        row = next(row for row in rows if int(safe_float(row["gpt54_count"])) == count)
        out.append(
            {
                "cut": "diversity",
                "x": float(count),
                "label": str(count),
                "iiq": safe_float(row["initial_accuracy_mean"]),
                "terminal": safe_float(row["final_accuracy_mean"]),
                "ciq": safe_float(row["final_vote_accuracy_rate"]),
            }
        )
    for row in out:
        row["social_lift"] = row["ciq"] - row["iiq"]
    return out


def protocol_rows() -> list[dict[str, float]]:
    rows = read_csv(PROTOCOL_CSV)
    specs = [
        ("Broadcast", "broadcast", "balanced_4_gpt_5_4_4_gpt_4o"),
        ("Pairwise", "pairwise", "balanced_4_gpt_5_4_4_gpt_4o"),
        ("4o mgr", "org", "manager_gpt_4o_observers_4_gpt_5_4_4_gpt_4o"),
        ("5.4 mgr", "org", "manager_gpt_5_4_observers_4_gpt_5_4_4_gpt_4o"),
    ]
    out: list[dict[str, float]] = []
    for idx, (label, protocol, condition) in enumerate(specs):
        row = next(
            row
            for row in rows
            if row["protocol"] == protocol and row["condition_name"] == condition
        )
        out.append(
            {
                "cut": "protocol",
                "x": float(idx),
                "label": label,
                "iiq": safe_float(row["initial_accuracy_mean"]),
                "terminal": safe_float(row["final_accuracy_mean"]),
                "ciq": safe_float(row["final_vote_accuracy_rate"]),
            }
        )
    for row in out:
        row["social_lift"] = row["ciq"] - row["iiq"]
    return out


def protocol_comparison_rows() -> list[dict[str, object]]:
    rows = list(reversed(read_csv(PROTOCOL_CSV)))
    out: list[dict[str, object]] = []
    for row in rows:
        protocol = str(row["protocol"])
        condition_name = str(row["condition_name"])
        if protocol == "broadcast":
            if condition_name == "all_gpt_4o":
                label = "all 4o"
            elif condition_name == "all_gpt_5_4":
                label = "all 5.4"
            else:
                label = "mixed"
        elif protocol == "pairwise":
            if condition_name == "all_gpt_4o":
                label = "all 4o"
            elif condition_name == "all_gpt_5_4":
                label = "all 5.4"
            else:
                label = "mixed"
        else:
            manager = "4o manager" if "manager_gpt_4o" in condition_name else "5.4 manager"
            if "observers_gpt_4o" in condition_name or condition_name == "all_gpt_4o":
                observers = "all 4o"
            elif "observers_gpt_5_4" in condition_name or condition_name == "all_gpt_5_4":
                observers = "all 5.4"
            else:
                observers = "mixed"
            label = f"{manager}, {observers}"
        out.append(
            {
                "label": label,
                "protocol": protocol,
                "iiq": safe_float(row["initial_accuracy_mean"]),
                "ciq": safe_float(row["final_accuracy_mean"]),
                "ciq_sem": safe_float(row["final_accuracy_sem"], 0.0),
                "color": PROTOCOL_COLORS[protocol],
            }
        )
    return out


def draw_protocol_comparison_panel(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    y = np.arange(len(rows), dtype=float)
    iiq = np.array([float(row["iiq"]) for row in rows], dtype=float)
    ciq = np.array([float(row["ciq"]) for row in rows], dtype=float)
    ciq_sem = np.array([float(row["ciq_sem"]) for row in rows], dtype=float)
    colors = [str(row["color"]) for row in rows]

    ax.barh(
        y,
        ciq,
        xerr=ciq_sem,
        color=colors,
        alpha=0.76,
        edgecolor="none",
        height=0.54,
        capsize=2.0,
        error_kw={"elinewidth": 0.75, "ecolor": INK, "capthick": 0.8},
        zorder=2,
    )
    ax.vlines(iiq, y - 0.36, y + 0.36, color=INK, linewidth=1.25, zorder=4)
    ax.set_xlim(0.0, 1.03)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Accuracy", labelpad=2)
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["label"]) for row in rows], fontsize=7.5)
    ax.grid(axis="x", color=LIGHT_GRAY, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="y", length=0, pad=3, colors=INK)
    ax.tick_params(axis="x", colors=INK)
    ax.set_title(
        "E. Protocol CIQ",
        loc="left",
        fontweight="bold",
        color=INK,
        pad=3,
        fontsize=10.1,
    )
    legend_handles = [
        Patch(facecolor=ORANGE, edgecolor="none", alpha=0.76, label="Pairwise"),
        Patch(facecolor=PURPLE, edgecolor="none", alpha=0.76, label="Broadcast"),
        Patch(facecolor=GREEN, edgecolor="none", alpha=0.76, label="Manager"),
        Line2D(
            [0],
            [0],
            color=INK,
            marker="|",
            linestyle="None",
            markersize=10.0,
            markeredgewidth=1.5,
            label="IIQ",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=False,
        fontsize=7.5,
        handlelength=1.3,
        handletextpad=0.42,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    legend._legend_box.align = "left"


def draw_figure() -> tuple[Path, Path, Path, Path]:
    setup_style()
    panel_data = {
        "population": pairwise_rows(),
        "pairwise_alpha": pairwise_alpha_rows(),
        "broadcast_alpha": broadcast_alpha_rows(),
    }
    protocol_comparison = protocol_comparison_rows()

    fig = plt.figure(figsize=(10.9, 5.72), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.02],
        height_ratios=[1.0, 1.0],
        hspace=0.58,
        wspace=0.52,
    )
    axes = {
        "population": fig.add_subplot(grid[0, 0]),
        "endpoint": fig.add_subplot(grid[0, 1]),
        "social_pairwise": fig.add_subplot(grid[1, 0]),
        "social_broadcast": fig.add_subplot(grid[1, 1]),
        "protocol_comparison": fig.add_subplot(grid[:, 2]),
    }

    population = panel_data["population"]
    x = np.array([row["x"] for row in population], dtype=float)
    draw_metric_lines(
        axes["population"],
        x,
        np.array([row["iiq"] for row in population], dtype=float),
        np.array([row["terminal"] for row in population], dtype=float),
        np.array([row["ciq"] for row in population], dtype=float),
        xlabel="Population size N",
    )
    axes["population"].set_xscale("log", base=2)
    axes["population"].set_xticks(x)
    axes["population"].set_xticklabels([row["label"] for row in population])
    panel_title(axes["population"], "A. Population-size sweep (pairwise)")

    draw_endpoint_lines(axes["endpoint"], population)
    panel_title(axes["endpoint"], "B. Endpoint outcomes by N (pairwise)")

    draw_social_lift_panel(
        axes["social_pairwise"],
        panel_data["pairwise_alpha"],
        color=BLUE_DARK,
        marker="o",
    )
    panel_title(axes["social_pairwise"], "C. Social-awareness effect (pairwise)")

    draw_social_lift_panel(
        axes["social_broadcast"],
        panel_data["broadcast_alpha"],
        color=BLUE_DARK,
        marker="o",
    )
    panel_title(axes["social_broadcast"], "D. Social-awareness effect (broadcast)")

    draw_protocol_comparison_panel(axes["protocol_comparison"], protocol_comparison)

    metric_handles = [
        Line2D([0], [0], color=BLUE_DARK, marker="o", markerfacecolor="white", markeredgewidth=1.4, linewidth=2.0, label="CIQ"),
        Line2D([0], [0], color=GREEN, marker="^", markerfacecolor="white", markeredgewidth=1.2, linewidth=1.6, label="Terminal accuracy"),
        Line2D([0], [0], color=GRAY, marker="o", markerfacecolor="white", markeredgewidth=1.2, linewidth=1.4, linestyle=(0, (2.0, 1.4)), label="IIQ"),
        Patch(facecolor=BLUE_LIGHT, edgecolor="none", label="Social uplift"),
    ]
    fig.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.987),
        ncol=4,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.1,
    )
    fig.subplots_adjust(left=0.086, right=0.992, top=0.900, bottom=0.110)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{STEM}.png"
    pdf_path = FIGURE_DIR / f"{STEM}.pdf"
    svg_path = FIGURE_DIR / f"{STEM}.svg"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary_path = DATA_DIR / f"{STEM}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "outputs": [str(png_path), str(pdf_path), str(svg_path)],
                "metric_definition": {
                    "IIQ": "initial private observer accuracy",
                    "CIQ": "mean final group accuracy after social exchange; manager rows use manager correctness",
                    "terminal_accuracy": "unique top-voted final country accuracy for pairwise/broadcast",
                    "social_uplift": "CIQ - IIQ",
                },
                "panel_data": panel_data,
                "protocol_side_by_side_rows": protocol_comparison,
            },
            indent=2,
        )
    )
    return png_path, pdf_path, svg_path, summary_path


def main() -> None:
    for path in draw_figure():
        print(path)


if __name__ == "__main__":
    main()

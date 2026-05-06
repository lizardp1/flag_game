#!/usr/bin/env python3
"""Make compact slot-ready Flag Game CIQ panels for the schematic poster layout."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp/memetic-drift-mplconfig")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache_dir = Path("/tmp/memetic-drift-cache")
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"

PAIRWISE_CSV = DATA_DIR / "flag_game_pairwise_n_scaling_main_v3_summary.csv"
PAIRWISE_V3_CSV = PAIRWISE_CSV
ALPHA_JSON = DATA_DIR / "flag_game_broadcast_m3_alpha_main_summary.json"
DIVERSITY_JSON = DATA_DIR / "flag_game_broadcast_m3_diversity_main_summary.json"
PROTOCOL_CSV = DATA_DIR / "flag_gam_protocol_side_by_side_N8_final_ranked_performance.csv"
SLOT_BROADCAST_COMPACT_JSON = DATA_DIR / "flag_game_slot_broadcast_compact_summary.json"
BROADCAST_ALPHA_RESULT_ROOT = ROOT / "results" / "flag_game_broadcast" / "alpha_mix_count_sweep_triangle28_N8_seeds5"

BLUE = "#2F80C0"
BLUE_LIGHT = "#CFE5F5"
TEAL = "#068B78"
TEAL_LIGHT = "#9ED8CD"
PURPLE = "#7C6FB6"
ORANGE = "#E07A3F"
GRAY = "#6E7780"
LIGHT_GRID = "#D9DEE5"
INK = "#2E3035"

SLOT_STEMS = {
    "pairwise_n_scaling": "flag_game_slot_pairwise_n_scaling_mean_ciq",
    "pairwise_n_scaling_v3": "flag_game_slot_pairwise_n_scaling_mean_ciq_v3",
    "broadcast_alpha": "flag_game_slot_broadcast_m3_alpha_mean_ciq",
    "broadcast_diversity": "flag_game_slot_broadcast_m3_diversity_mean_ciq",
    "protocol_mixed": "flag_game_slot_protocol_side_by_side_N8_mixed_ciq",
}


def add_arial() -> None:
    for path in (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def setup_style() -> None:
    add_arial()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "xtick.major.size": 3.3,
            "ytick.major.size": 3.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_data(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text())
    return list(payload["data"])


def read_slot_broadcast_compact() -> dict[str, list[dict[str, float]]]:
    if not SLOT_BROADCAST_COMPACT_JSON.exists():
        return {}
    payload = json.loads(SLOT_BROADCAST_COMPACT_JSON.read_text())
    return {
        "alpha_common_seed_summary": list(payload.get("alpha_common_seed_summary", [])),
        "diversity_alpha_1_ciq_se": list(payload.get("diversity_alpha_1_ciq_se", [])),
    }


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def alpha_from_dir(path: Path) -> float:
    return float(path.name.removeprefix("alpha_").replace("_", "."))


def gpt54_count_from_condition(condition: str) -> int:
    if condition == "all_gpt_4o":
        return 0
    if condition == "all_gpt_5_4":
        return 8
    match = re.search(r"plus_(\d+)_gpt_5_4", condition)
    if not match:
        raise ValueError(f"Cannot infer GPT-5.4 count from condition: {condition}")
    return int(match.group(1))


def standard_error(values: Iterable[float]) -> float:
    clean = np.array([float(value) for value in values], dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= 1:
        return 0.0
    return float(np.std(clean, ddof=1) / np.sqrt(clean.size))


def load_broadcast_alpha_common_seed_summary() -> tuple[list[dict[str, float]], list[int]]:
    alpha_condition_rows: dict[float, list[list[dict[str, str]]]] = {}
    alpha_seed_sets: dict[float, list[set[int]]] = {}
    for alpha_dir in sorted(BROADCAST_ALPHA_RESULT_ROOT.glob("alpha_*"), key=alpha_from_dir):
        condition_rows: list[list[dict[str, str]]] = []
        condition_seed_sets: list[set[int]] = []
        for summary_path in sorted(alpha_dir.glob("*/batch_summary.csv")):
            rows = read_csv(summary_path)
            if not rows:
                continue
            condition_rows.append(rows)
            condition_seed_sets.append({int(float(row["seed"])) for row in rows})
        if condition_rows:
            alpha = alpha_from_dir(alpha_dir)
            alpha_condition_rows[alpha] = condition_rows
            alpha_seed_sets[alpha] = condition_seed_sets

    if not alpha_condition_rows:
        compact_rows = read_slot_broadcast_compact().get("alpha_common_seed_summary", [])
        if compact_rows:
            return [
                {
                    "alpha": float(row["alpha"]),
                    "ciq": float(row["ciq"]),
                    "ciq_se": float(row["ciq_se"]),
                    "n_seeds": float(row.get("n_seeds", 0.0)),
                }
                for row in compact_rows
            ], []
        rows = read_json_data(ALPHA_JSON)
        return [
            {
                "alpha": float(row["alpha"]),
                "ciq": float(row["ciq"]),
                "ciq_se": float(row.get("ciq_se", 0.0)),
                "n_seeds": float(row.get("n_trials", 0.0)),
            }
            for row in rows
        ], []

    complete_seed_sets = [set.intersection(*sets) for sets in alpha_seed_sets.values()]
    common_seeds = sorted(set.intersection(*complete_seed_sets))
    if not common_seeds:
        raise ValueError("No seed IDs are complete across all broadcast alpha conditions.")

    summary: list[dict[str, float]] = []
    for alpha in sorted(alpha_condition_rows):
        per_seed_values: list[float] = []
        for seed in common_seeds:
            values = [
                as_float(row, "final_accuracy")
                for rows in alpha_condition_rows[alpha]
                for row in rows
                if int(float(row["seed"])) == seed
            ]
            if len(values) != len(alpha_condition_rows[alpha]):
                raise ValueError(
                    f"Seed {seed} is incomplete for alpha={alpha}: "
                    f"{len(values)} rows across {len(alpha_condition_rows[alpha])} conditions"
                )
            per_seed_values.append(float(np.mean(values)))
        summary.append(
            {
                "alpha": alpha,
                "ciq": float(np.mean(per_seed_values)),
                "ciq_se": standard_error(per_seed_values),
                "n_seeds": float(len(common_seeds)),
            }
        )
    return summary, common_seeds


def load_broadcast_diversity_se() -> dict[int, float]:
    alpha_dir = BROADCAST_ALPHA_RESULT_ROOT / "alpha_1_0"
    if not alpha_dir.exists():
        return {
            int(float(row["gpt54_count"])): float(row["ciq_se"])
            for row in read_slot_broadcast_compact().get("diversity_alpha_1_ciq_se", [])
        }
    sem_by_count: dict[int, float] = {}
    for summary_path in sorted(alpha_dir.glob("*/batch_summary.csv")):
        rows = read_csv(summary_path)
        if not rows:
            continue
        count = gpt54_count_from_condition(summary_path.parent.name)
        sem_by_count[count] = standard_error(as_float(row, "final_accuracy") for row in rows)
    return sem_by_count


def high_ciq_window(data: list[dict[str, float]], *, tolerance: float = 0.025) -> tuple[float, float]:
    rows = sorted(data, key=lambda row: row["gpt54_count"])
    counts = np.array([float(row["gpt54_count"]) for row in rows], dtype=float)
    ciq = np.array([float(row["ciq"]) for row in rows], dtype=float)
    if counts.size == 0:
        return 0.5, 7.5
    threshold = float(np.nanmax(ciq) - tolerance)
    high_indices = np.flatnonzero(ciq >= threshold)
    if high_indices.size == 0:
        return float(counts[0] - 0.5), float(counts[-1] + 0.5)
    runs: list[np.ndarray] = np.split(high_indices, np.where(np.diff(high_indices) != 1)[0] + 1)
    best_run = max(runs, key=len)
    return float(counts[int(best_run[0])] - 0.5), float(counts[int(best_run[-1])] + 0.5)


def style_axis(ax: plt.Axes, *, y_min: float = 0.45, y_max: float = 0.95) -> None:
    ax.set_ylim(y_min, y_max)
    tick_start = np.ceil(y_min * 10) / 10
    tick_stop = np.floor(y_max * 10) / 10
    ax.set_yticks(np.arange(tick_start, tick_stop + 0.001, 0.1))
    ax.set_ylabel("Collective mean accuracy", labelpad=3)
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def save(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURE_DIR / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return paths


def compact_figure() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(3.15, 2.28), constrained_layout=False)


def plot_pairwise_n_scaling(
    *,
    source_csv: Path = PAIRWISE_CSV,
    stem: str = SLOT_STEMS["pairwise_n_scaling"],
) -> list[Path]:
    rows = [
        row
        for row in read_csv(source_csv)
        if row["condition"] == "all_gpt_4o"
    ]
    rows = sorted(rows, key=lambda row: int(row["N"]))
    x = np.array([int(row["N"]) for row in rows], dtype=float)
    y = np.array([as_float(row, "final_mean") for row in rows], dtype=float)
    lo = np.array([as_float(row, "final_ci_low") for row in rows], dtype=float)
    hi = np.array([as_float(row, "final_ci_high") for row in rows], dtype=float)

    fig, ax = compact_figure()
    ax.fill_between(x, lo, hi, color=BLUE_LIGHT, alpha=0.72, linewidth=0)
    ax.plot(x, y, color=BLUE, linewidth=2.1, marker="o", markersize=5.2, markerfacecolor="white", markeredgewidth=1.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Population size")
    style_axis(ax, y_min=0.35, y_max=0.75)
    ax.margins(x=0.04)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.965, bottom=0.24)
    return save(fig, stem)


def plot_broadcast_alpha() -> list[Path]:
    data, _common_seeds = load_broadcast_alpha_common_seed_summary()
    x = np.array([float(row["alpha"]) for row in data], dtype=float)
    y = np.array([float(row["ciq"]) for row in data], dtype=float)
    se = np.array([float(row["ciq_se"]) for row in data], dtype=float)

    fig, ax = compact_figure()
    ax.fill_between(x, y - se, y + se, color=BLUE_LIGHT, alpha=0.68, linewidth=0, zorder=1)
    ax.plot(x, y, color=BLUE, linewidth=2.1, marker="o", markersize=5.2, markerfacecolor="white", markeredgewidth=1.7, zorder=3)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("Social evidence uptake")
    style_axis(ax, y_min=0.45, y_max=0.94)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.965, bottom=0.24)
    return save(fig, SLOT_STEMS["broadcast_alpha"])


def plot_broadcast_diversity() -> list[Path]:
    data = sorted(read_json_data(DIVERSITY_JSON), key=lambda row: row["gpt54_count"])
    x = np.array([float(row["gpt54_count"]) for row in data], dtype=float)
    y = np.array([float(row["ciq"]) for row in data], dtype=float)
    se_by_count = load_broadcast_diversity_se()
    se = np.array([se_by_count.get(int(row["gpt54_count"]), 0.0) for row in data], dtype=float)
    high_left, high_right = high_ciq_window(data)

    fig, ax = compact_figure()
    ax.fill_between(x, y - se, y + se, color=BLUE_LIGHT, alpha=0.68, linewidth=0, zorder=1)
    for edge in (high_left, high_right):
        ax.axvline(edge, color=BLUE, linewidth=0.85, linestyle=(0, (2.0, 2.0)), alpha=0.58, zorder=2)
    ax.plot(x, y, color=BLUE, linewidth=2.1, marker="o", markersize=5.0, markerfacecolor="white", markeredgewidth=1.65, zorder=3)
    ax.set_xlim(-0.25, 8.25)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("GPT-5.4 agents in team")
    style_axis(ax, y_min=0.40, y_max=0.80)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.965, bottom=0.24)
    return save(fig, SLOT_STEMS["broadcast_diversity"])


def protocol_lookup(rows: Iterable[dict[str, str]], protocol: str, condition: str) -> dict[str, str]:
    for row in rows:
        if row["protocol"] == protocol and row["condition_name"] == condition:
            return row
    raise KeyError(f"Missing protocol={protocol}, condition={condition}")


def plot_protocol_mixed_slice() -> list[Path]:
    rows = read_csv(PROTOCOL_CSV)
    specs = [
        ("Broadcast", "broadcast", "balanced_4_gpt_5_4_4_gpt_4o", PURPLE),
        ("Pairwise", "pairwise", "balanced_4_gpt_5_4_4_gpt_4o", ORANGE),
        ("4o mgr", "org", "manager_gpt_4o_observers_4_gpt_5_4_4_gpt_4o", TEAL_LIGHT),
        ("5.4 mgr", "org", "manager_gpt_5_4_observers_4_gpt_5_4_4_gpt_4o", TEAL),
    ]
    labels: list[str] = []
    values: list[float] = []
    sems: list[float] = []
    colors: list[str] = []
    for label, protocol, condition, color in specs:
        row = protocol_lookup(rows, protocol, condition)
        labels.append(label)
        values.append(as_float(row, "final_accuracy_mean"))
        sems.append(as_float(row, "final_accuracy_sem"))
        colors.append(color)

    fig, ax = compact_figure()
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=sems, color=colors, width=0.68, edgecolor="none", capsize=2.6, error_kw={"elinewidth": 0.9, "ecolor": INK})
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Mixed observers")
    style_axis(ax, y_min=0.35, y_max=0.65)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.965, bottom=0.24)
    return save(fig, SLOT_STEMS["protocol_mixed"])


def make_contact_sheet(paths: list[Path]) -> Path | None:
    try:
        from PIL import Image, ImageOps
    except Exception:
        return None

    pngs = [path for path in paths if path.suffix == ".png"]
    if len(pngs) != 4:
        return None
    images = [Image.open(path).convert("RGB") for path in pngs]
    thumb_w = max(image.width for image in images)
    thumb_h = max(image.height for image in images)
    canvas = Image.new("RGB", (thumb_w * 2 + 36, thumb_h * 2 + 36), "white")
    for idx, image in enumerate(images):
        padded = ImageOps.pad(image, (thumb_w, thumb_h), color="white", centering=(0.5, 0.5))
        x = (idx % 2) * (thumb_w + 36)
        y = (idx // 2) * (thumb_h + 36)
        canvas.paste(padded, (x, y))
    out = FIGURE_DIR / "flag_game_slot_figures_contact_sheet.png"
    canvas.save(out)
    return out


def main() -> None:
    setup_style()
    outputs: list[Path] = []
    outputs.extend(plot_pairwise_n_scaling())
    outputs.extend(plot_pairwise_n_scaling(source_csv=PAIRWISE_V3_CSV, stem=SLOT_STEMS["pairwise_n_scaling_v3"]))
    outputs.extend(plot_broadcast_alpha())
    outputs.extend(plot_broadcast_diversity())
    outputs.extend(plot_protocol_mixed_slice())
    contact = make_contact_sheet(outputs)
    for path in outputs:
        print(path)
    if contact:
        print(contact)


if __name__ == "__main__":
    main()

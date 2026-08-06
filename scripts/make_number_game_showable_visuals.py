#!/usr/bin/env python3
"""Make showable number-game visuals that mirror the flag-game chart grammar."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import ast
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/number-game-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/number-game-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnd.number_game.domain import (
    DEFAULT_CLUES,
    candidate_numbers,
    clue_candidates,
    clue_information_bits,
    clue_information_phase,
)


PAIRWISE_DIR = Path("outputs/number_game_prompt100/local_qwen3_1_7b_pairwise_m_compare_5seeds")
RATIO_DIR = Path("outputs/number_game_prompt100/local_qwen3_1_7b_conflict_ratio8_1seed")
OUT_DIR = Path("outputs/number_game_prompt100/showable_visuals")
TRAJECTORY_SEED_COUNTS: dict[int, int] = {}

MECH_BLUE = "#1F77D0"
MECH_ORANGE = "#F17C2E"
MECH_GREEN = "#00A36F"
MECH_LIGHT_GRAY = "#D9DEE5"
MECH_INK = "#25272B"
MECH_PURPLE = "#7C6FB6"
MECH_RED = "#D84A4A"
MECH_TEAL = "#00A6A6"
MECH_DARK_GRAY = "#747C85"
PHASE_COLORS = {"weak": "#B7791F", "medium": "#4C78A8", "strong": "#7C6FB6", "unknown": MECH_DARK_GRAY}

STACK_ORDER = [
    ("private_target_rate", "Private target / correct", MECH_BLUE),
    ("other_compatible_rate", "Other clue-compatible", MECH_GREEN),
    ("incompatible_rate", "Other clue-incompatible", MECH_LIGHT_GRAY),
    ("social_evidence_rate", "Contradictory social memory", MECH_ORANGE),
]
LEGEND_ORDER = [
    ("private_target_rate", "Private target / correct", MECH_BLUE),
    ("social_evidence_rate", "Contradictory social memory", MECH_ORANGE),
    ("other_compatible_rate", "Other clue-compatible", MECH_GREEN),
    ("incompatible_rate", "Other clue-incompatible", MECH_LIGHT_GRAY),
]


def main() -> None:
    global PAIRWISE_DIR, RATIO_DIR, OUT_DIR
    args = parse_args()
    PAIRWISE_DIR = args.pairwise_dir
    RATIO_DIR = args.conflict_dir
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    write_clue_information_outputs()
    write_memory_ratio_figure()
    write_memory_ratio_phase_figure()
    write_actual_trajectory_grid(m=1)
    write_actual_trajectory_grid(m=3)
    write_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build showable figures from number-game outputs.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=PAIRWISE_DIR,
        help="Directory produced by `compare-pairwise-m`; must contain m1/ and m3/ seed folders.",
    )
    parser.add_argument(
        "--conflict-dir",
        type=Path,
        default=RATIO_DIR,
        help="Directory produced by `conflict-battery`; must contain conflict_summary.csv and conflict_trials.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where figures and derived CSVs should be written.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.2,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.2,
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.9,
            "legend.fontsize": 9.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MECH_INK,
            "xtick.color": MECH_INK,
            "ytick.color": MECH_INK,
            "text.color": MECH_INK,
            "axes.labelcolor": MECH_INK,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def ratio_candidate_numbers() -> list[int]:
    config_path = RATIO_DIR / "config_resolved.yaml"
    if not config_path.exists():
        return candidate_numbers(1, 30)
    text = config_path.read_text()
    min_number = _config_value(text, "min_number", 1)
    max_number = _config_value(text, "max_number", 30)
    return candidate_numbers(int(min_number), int(max_number))


def _config_value(text: str, key: str, default: Any) -> Any:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return default
    try:
        return ast.literal_eval(match.group(1).strip())
    except (SyntaxError, ValueError):
        return match.group(1).strip()


def clue_information_rows(numbers: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clue in DEFAULT_CLUES:
        candidates = clue_candidates(numbers, clue.text)
        bits = clue_information_bits(numbers, clue.text)
        rows.append(
            {
                "clue_name": clue.name,
                "clue_text": clue.text,
                "candidate_range_count": len(numbers),
                "candidate_count": len(candidates),
                "prior_probability": len(candidates) / float(len(numbers)) if numbers else None,
                "information_bits": bits,
                "information_phase": clue_information_phase(bits),
            }
        )
    return rows


def write_clue_information_outputs() -> None:
    rows = clue_information_rows(ratio_candidate_numbers())
    write_csv(OUT_DIR / "clue_information.csv", rows)
    sorted_rows = sorted(rows, key=lambda row: float(row["information_bits"] or 0.0))
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    y = np.arange(len(sorted_rows))
    colors = [PHASE_COLORS[str(row["information_phase"])] for row in sorted_rows]
    bits = [float(row["information_bits"] or 0.0) for row in sorted_rows]
    ax.barh(y, bits, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["clue_text"]).replace("the number ", "") for row in sorted_rows])
    ax.set_xlabel("Private-clue self-information I(clue), bits")
    fig.suptitle(
        "Private clue information values",
        x=0.23,
        y=0.98,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.text(
        0.23,
        0.94,
        "I(clue) = -log2 P(clue) = log2(total candidates / clue-consistent candidates), uniform analysis baseline",
        ha="left",
        fontsize=9.4,
        color="#4A5565",
    )
    for index, row in enumerate(sorted_rows):
        candidate_count = int(row["candidate_count"])
        candidate_range_count = int(row["candidate_range_count"])
        ax.text(
            bits[index] + 0.05,
            index,
            f"I={bits[index]:.2f} bits | {candidate_count}/{candidate_range_count} candidates | {row['information_phase']}",
            va="center",
            fontsize=8.8,
            color=MECH_INK,
        )
    max_bits = max(bits) if bits else 1.0
    ax.set_xlim(0, max_bits + 2.0)
    ax.grid(axis="x", color="#EEF1F5", linewidth=0.7)
    ax.set_axisbelow(True)
    legend_handles = [
        Patch(color=PHASE_COLORS[phase], label=label)
        for phase, label in [
            ("weak", "weak <= 1 bit"),
            ("medium", "medium 1-2 bits"),
            ("strong", "strong > 2 bits"),
        ]
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.23, right=0.985, top=0.86, bottom=0.12)
    save_figure(fig, OUT_DIR / "00_clue_information_values")


def add_clue_info_to_trial(row: dict[str, str], numbers: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    bits = clue_information_bits(numbers, row["private_clue"])
    candidates = clue_candidates(numbers, row["private_clue"])
    out["private_clue_info_bits"] = bits
    out["private_clue_info_phase"] = clue_information_phase(bits)
    out["private_clue_candidate_count"] = len(candidates)
    out["private_clue_prior_probability"] = len(candidates) / float(len(numbers)) if numbers else None
    return out


def write_memory_ratio_figure() -> None:
    rows = read_csv(RATIO_DIR / "conflict_summary.csv")
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.25), sharex=True, sharey=True)
    x_labels: list[str] = []
    fig.suptitle(
        "Private clue vs contradictory social memory",
        x=0.13,
        y=0.975,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.938,
        "Blue = correct/private target. Orange = contradictory social-memory answer.",
        ha="left",
        va="center",
        fontsize=9.6,
        color="#4A5565",
    )

    for row_index, m_value in enumerate((1, 3)):
        ax = axes[row_index]
        cell = sorted(
            [row for row in rows if int(row["m"]) == m_value],
            key=lambda row: (-int(row["target_memory_count"]), int(row["social_memory_count"])),
        )
        x = np.arange(len(cell))
        x_labels = [row["memory_ratio_label"] for row in cell]
        y_series = [[float(row[metric]) for row in cell] for metric, _label, _color in STACK_ORDER]
        ax.stackplot(
            x,
            *y_series,
            colors=[color for _metric, _label, color in STACK_ORDER],
            linewidth=0.35,
            edgecolor="white",
        )
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, max(len(cell) - 1, 1))
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"])
        ax.grid(axis="y", color="#EEF1F5", linewidth=0.7)
        ax.set_axisbelow(True)
        label = "Qwen3-1.7B m=1" if m_value == 1 else "Qwen3-1.7B m=3"
        ax.text(
            0.01,
            0.88,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
        )
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.1)

    axes[-1].set_xticks(np.arange(len(x_labels)))
    axes[-1].set_xticklabels(x_labels)
    fig.text(
        0.51,
        0.125,
        "Memory entries (private target number : contradictory social evidence)",
        ha="center",
        va="center",
        fontsize=10.7,
    )
    fig.text(
        0.035,
        0.54,
        "Agent response probability",
        ha="center",
        va="center",
        rotation=90,
        fontsize=10.7,
    )
    legend_handles = [Patch(color=color, label=label) for _metric, label, color in LEGEND_ORDER]
    fig.legend(
        legend_handles,
        [label for _metric, label, _color in LEGEND_ORDER],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.015),
        columnspacing=1.1,
        handlelength=1.25,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.885, bottom=0.29, hspace=0.26)
    save_figure(fig, OUT_DIR / "03_private_vs_social_memory_ratio_flag_style")


def write_memory_ratio_phase_figure() -> None:
    numbers = ratio_candidate_numbers()
    rows = [add_clue_info_to_trial(row, numbers) for row in read_csv(RATIO_DIR / "conflict_trials.csv")]
    phase_rows = aggregate_conflict_trials_by_phase(rows)
    write_csv(OUT_DIR / "conflict_phase_summary_from_trials.csv", phase_rows)

    phases = ["weak", "medium", "strong"]
    phase_titles = {
        "weak": "weak clue\n<= 1 bit",
        "medium": "medium clue\n1-2 bits",
        "strong": "strong clue\n> 2 bits",
    }
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.95), sharex=True, sharey=True)
    x_labels: list[str] = []
    for row_index, m_value in enumerate((1, 3)):
        for col_index, phase in enumerate(phases):
            ax = axes[row_index, col_index]
            cell = sorted(
                [
                    row
                    for row in phase_rows
                    if int(row["m"]) == m_value and str(row["private_clue_info_phase"]) == phase
                ],
                key=lambda row: (-int(row["target_memory_count"]), int(row["social_memory_count"])),
            )
            if not cell:
                ax.axis("off")
                continue
            x = np.arange(len(cell))
            x_labels = [str(row["memory_ratio_label"]) for row in cell]
            y_series = [[float(row[metric]) for row in cell] for metric, _label, _color in STACK_ORDER]
            ax.stackplot(
                x,
                *y_series,
                colors=[color for _metric, _label, color in STACK_ORDER],
                linewidth=0.35,
                edgecolor="white",
            )
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, max(len(cell) - 1, 1))
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_yticklabels(["0", ".5", "1"])
            ax.grid(axis="y", color="#EEF1F5", linewidth=0.7)
            ax.set_axisbelow(True)
            if row_index == 0:
                ax.set_title(phase_titles[phase], fontsize=10.5, fontweight="bold")
            ax.text(
                0.02,
                0.88,
                f"m={m_value}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=9.0,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.3},
            )
            n_trials = sum(int(row["n"]) for row in cell)
            bit_min = min(float(row["private_clue_info_bits_min"]) for row in cell)
            bit_max = max(float(row["private_clue_info_bits_max"]) for row in cell)
            ax.text(
                0.02,
                0.06,
                f"trials={n_trials}; I={bit_min:.2f}-{bit_max:.2f} bits",
                transform=ax.transAxes,
                fontsize=7.8,
                color="#4A5565",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
            )
            for spine in ("left", "bottom"):
                ax.spines[spine].set_linewidth(1.0)

    for ax in axes[-1]:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
    fig.suptitle(
        "Conflict probe split by private-clue information",
        x=0.09,
        y=0.985,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.947,
        "Blue = correct/private target; orange = contradictory social-memory answer.",
        ha="left",
        fontsize=9.6,
        color="#4A5565",
    )
    fig.text(
        0.51,
        0.105,
        "Memory entries (private target number : contradictory social evidence)",
        ha="center",
        fontsize=10.3,
    )
    fig.text(
        0.025,
        0.52,
        "Agent response probability",
        va="center",
        rotation=90,
        fontsize=10.3,
    )
    legend_handles = [Patch(color=color, label=label) for _metric, label, color in LEGEND_ORDER]
    fig.legend(
        legend_handles,
        [label for _metric, label, _color in LEGEND_ORDER],
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.0),
        columnspacing=0.9,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.855, bottom=0.22, hspace=0.25, wspace=0.18)
    save_figure(fig, OUT_DIR / "04_conflict_probe_by_clue_information_phase")


def aggregate_conflict_trials_by_phase(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["m"]),
                str(row["private_clue_info_phase"]),
                int(row["target_memory_count"]),
                int(row["social_memory_count"]),
                str(row["memory_relation"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (m, phase, target_count, social_count, memory_relation), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], {"weak": 0, "medium": 1, "strong": 2}.get(item[0][1], 3), -item[0][2], item[0][3]),
    ):
        valid = [row for row in group if str(row["valid"]) == "True"]
        out.append(
            {
                "m": m,
                "private_clue_info_phase": phase,
                "private_clue_info_bits_mean": mean(float(row["private_clue_info_bits"]) for row in group),
                "private_clue_info_bits_min": min(float(row["private_clue_info_bits"]) for row in group),
                "private_clue_info_bits_max": max(float(row["private_clue_info_bits"]) for row in group),
                "target_memory_count": target_count,
                "social_memory_count": social_count,
                "memory_total": target_count + social_count,
                "memory_ratio_label": f"{target_count}:{social_count}",
                "memory_relation": memory_relation,
                "n": len(group),
                "valid_rate": len(valid) / max(len(group), 1),
                "private_target_rate": response_rate(valid, "private_target"),
                "social_evidence_rate": response_rate(valid, "social_evidence"),
                "other_compatible_rate": response_rate(valid, "other_compatible"),
                "incompatible_rate": response_rate(valid, "incompatible"),
            }
        )
    return out


def response_rate(rows: list[dict[str, Any]], category: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("response_category") == category) / float(len(rows))


def mean(values: Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / float(len(collected))


def write_actual_trajectory_grid(*, m: int) -> None:
    seed_dirs = sorted((PAIRWISE_DIR / f"m{m}").glob("seed_*"))
    seed_dirs = [path for path in seed_dirs if (path / "summary.json").exists() and (path / "probes.csv").exists()]
    if not seed_dirs:
        return
    TRAJECTORY_SEED_COUNTS[m] = len(seed_dirs)
    fig_width = max(5.2, 2.55 * len(seed_dirs))
    fig, axes_raw = plt.subplots(1, len(seed_dirs), figsize=(fig_width, 3.55), sharey=True)
    axes = np.atleast_1d(axes_raw)
    for run_dir, ax in zip(seed_dirs, axes):
        seed = int(run_dir.name.removeprefix("seed_"))
        summary = json.loads((run_dir / "summary.json").read_text())
        rows = read_csv(run_dir / "probes.csv")
        valid = [row for row in rows if row["valid"] == "True" and row["number"]]
        times = sorted({int(row["t"]) for row in valid})
        counts_by_t = {
            t: Counter(int(row["number"]) for row in valid if int(row["t"]) == t)
            for t in times
        }
        truth = int(summary["truth_number"])
        numbers = [number for number, _count in Counter(int(row["number"]) for row in valid).most_common(6)]
        if truth not in numbers:
            numbers = [truth] + numbers[:5]
        other_palette = [MECH_ORANGE, MECH_GREEN, MECH_PURPLE, MECH_RED, MECH_TEAL, MECH_DARK_GRAY]
        other_numbers = [number for number in numbers if number != truth]
        colors = {number: other_palette[index % len(other_palette)] for index, number in enumerate(other_numbers)}
        colors[truth] = MECH_BLUE
        for number in numbers:
            shares = []
            for t in times:
                total = sum(counts_by_t[t].values()) or 1
                shares.append(counts_by_t[t].get(number, 0) / total)
            ax.plot(
                times,
                shares,
                color=colors[number],
                linewidth=3.0 if number == truth else 1.7,
                alpha=1.0 if number == truth else 0.72,
                solid_capstyle="round",
            )
            if shares and shares[-1] >= 0.20:
                label = f"truth {number}" if number == truth else f"{number}"
                ax.text(
                    times[-1] + 0.4,
                    shares[-1],
                    label,
                    color=colors[number],
                    fontsize=8.7,
                    fontweight="bold" if number == truth else "normal",
                    va="center",
                )
        ax.set_title(
            f"seed {seed}",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=3,
        )
        ax.text(
            0.02,
            0.89,
            f"truth = {truth} (blue)",
            transform=ax.transAxes,
            fontsize=8.8,
            color=MECH_BLUE,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 1.1},
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(min(times), max(times) + 8)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"])
        ax.grid(axis="y", color="#EEF1F5", linewidth=0.7)
        ax.tick_params(axis="x", labelsize=8.8)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.0)
    title = "Actual pairwise social trajectories, m=1" if m == 1 else "Actual pairwise social trajectories, m=3"
    fig.suptitle(title, x=0.075, ha="left", fontsize=13.2, fontweight="bold")
    fig.text(
        0.075,
        0.88,
        "Blue is always the true hidden number; end labels mark numbers with final share >= 0.20.",
        ha="left",
        fontsize=9.8,
        color="#4A5565",
    )
    fig.text(0.51, 0.055, "Interaction round", ha="center", fontsize=10.8)
    fig.text(0.018, 0.52, "Share of agents choosing number", va="center", rotation=90, fontsize=10.8)
    fig.subplots_adjust(left=0.067, right=0.99, top=0.76, bottom=0.21, wspace=0.30)
    prefix = "01" if m == 1 else "02"
    save_figure(fig, OUT_DIR / f"{prefix}_actual_trajectories_pairwise_m{m}_{len(seed_dirs)}seeds")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def write_index() -> None:
    m1_seeds = TRAJECTORY_SEED_COUNTS.get(1, 0)
    m3_seeds = TRAJECTORY_SEED_COUNTS.get(3, 0)
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Number Game Pre-RunPod Visuals v2",
                "",
                "These are the showable figures that match the flag-game chart grammar more closely. In trajectory plots, blue is reserved for the true hidden number.",
                "",
                f"- `01_actual_trajectories_pairwise_m1_{m1_seeds}seeds.svg`: actual social-interaction trajectories for every pairwise m=1 seed.",
                f"- `02_actual_trajectories_pairwise_m3_{m3_seeds}seeds.svg`: actual social-interaction trajectories for every pairwise m=3 seed.",
                "- `03_private_vs_social_memory_ratio_flag_style.svg`: flag-style stacked response composition over private-target versus social-evidence memory ratios.",
                "- `04_conflict_probe_by_clue_information_phase.svg`: the same conflict probe split into weak/medium/strong private-clue information phases.",
                "- `00_clue_information_values.svg`: information value for every clue under the configured candidate range.",
                "- `clue_information.csv`: candidate counts, analysis-baseline probabilities, information bits, and phase for every configured clue.",
                "- `conflict_phase_summary_from_trials.csv`: phase-split response composition derived from the corrected local Qwen conflict trials.",
                "",
                "Note: the ratio figures use the corrected local Qwen3-1.7B reason-schema ratio-conflict run with `memory_total=4`. The RunPod version should use `memory_total=8` and keep the phase split.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()

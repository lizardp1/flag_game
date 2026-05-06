#!/usr/bin/env python3
"""Make paper-facing pairwise Flag Game N-scaling figures."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from itertools import combinations
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
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from make_flag_game_empirical_decomposition import (
    BLUE_DARK as EMPIRICAL_BLUE_DARK,
    BLUE_LIGHT as EMPIRICAL_BLUE_LIGHT,
    GRAY as EMPIRICAL_GRAY,
    GREEN as EMPIRICAL_GREEN,
    INK as EMPIRICAL_INK,
    LIGHT_GRAY as EMPIRICAL_LIGHT_GRAY,
    ORANGE as EMPIRICAL_ORANGE,
    PURPLE as EMPIRICAL_PURPLE,
    draw_metric_lines as draw_empirical_metric_lines,
    pairwise_rows as empirical_pairwise_rows,
    panel_title as empirical_panel_title,
)
from plot_style import HOUSE_COLORS, setup_paper_house_style, style_axis

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = (
    ROOT
    / "results"
    / "flag_game"
    / "proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5"
    / "04_model_mix_alpha_neutral"
)
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"


N_VALUES = [4, 8, 16, 32, 64, 128]
PLOTTED_CONDITIONS = ["all_gpt_4o", "all_gpt_5_4"]
CONDITION_STYLES = {
    "all_gpt_4o": {
        "label": "GPT-4o",
        "short": "4o",
        "color": HOUSE_COLORS["blue"],
        "marker": "o",
    },
    "all_gpt_5_4": {
        "label": "GPT-5.4",
        "short": "5.4",
        "color": HOUSE_COLORS["red"],
        "marker": "s",
    },
}
CONSENSUS_THRESHOLD = 0.85
POLARIZATION_THRESHOLD = 0.25
OUTCOME_ORDER = ["correct_consensus", "wrong_consensus", "polarization", "fragmentation"]
OUTCOME_LABELS = {
    "correct_consensus": "Correct consensus",
    "wrong_consensus": "Wrong consensus",
    "polarization": "Polarization",
    "fragmentation": "Fragmentation",
}
OUTCOME_DEFINITIONS = {
    "correct_consensus": f"Correct consensus: top country share >= {CONSENSUS_THRESHOLD:.2f} and true",
    "wrong_consensus": f"Wrong consensus: top country share >= {CONSENSUS_THRESHOLD:.2f} and false",
    "polarization": (
        f"Polarization: no consensus; at least two countries each >= {POLARIZATION_THRESHOLD:.2f}"
    ),
    "fragmentation": (
        f"Fragmentation: no consensus; fewer than two countries >= {POLARIZATION_THRESHOLD:.2f}"
    ),
}
OUTCOME_COLORS = {
    "correct_consensus": HOUSE_COLORS["teal"],
    "wrong_consensus": HOUSE_COLORS["orange"],
    "polarization": HOUSE_COLORS["purple"],
    "fragmentation": "#B8BFC9",
}

APPENDIX_CASE = {"seed": "seed_0003", "truth": "France", "rival": "Peru"}
APPENDIX_OUT_STEM = "flag_game_pairwise_n_scaling_appendix_france_peru_example"
MAIN_TEXT_MECHANISM_N_VALUES = N_VALUES[:-1]
MAIN_TEXT_MECHANISM_OUT_STEM = "flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N64"
EMPIRICAL_AB_MECHANISM_N_VALUES = [4, 16, 64]
EMPIRICAL_AB_MECHANISM_BASE_STEM = (
    "flag_game_pairwise_n_scaling_france_peru_mechanism_N4_N16_N64_with_empirical_ab"
)
EMPIRICAL_AB_MECHANISM_SPLIT_OUT_STEM = f"{EMPIRICAL_AB_MECHANISM_BASE_STEM}_endpoint_split"
EMPIRICAL_AB_MECHANISM_FAILURE_OUT_STEM = f"{EMPIRICAL_AB_MECHANISM_BASE_STEM}_failure_decomposition"
MODEL_COMPARISON_OUT_STEM = "flag_game_pairwise_n_scaling_model_comparison_appendix"
COUNTRY_COLORS = {
    "Guinea": HOUSE_COLORS["teal"],
    "Mali": HOUSE_COLORS["red"],
    "France": HOUSE_COLORS["blue"],
    "Peru": HOUSE_COLORS["orange"],
    "Missing": "#94A3B8",
    "Invalid": "#6B7280",
}
FALLBACK_COUNTRY_COLORS = [
    HOUSE_COLORS["purple"],
    HOUSE_COLORS["gold"],
    HOUSE_COLORS["gray"],
    HOUSE_COLORS["blue"],
]
PAPER_SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]
MAIN_V3_STEM = "flag_game_pairwise_n_scaling_main_v3"
MAIN_V3_EXCLUDED_SEEDS = [12, 16, 17, 19, 25, 29]
META_COLUMNS = {
    "t",
    "truth_country",
    "truth_mass",
    "mean_accuracy",
    "valid_probe_count",
    "invalid_probe_count",
    "support_size",
    "entropy",
    "U",
    "top1_share",
    "top2_share",
    "outcome",
    "consensus_country",
    "consensus_correct",
}
INTERACTION_ROUND_DEFINITION = (
    "Interaction rounds are measured as t/N, where t is the number of directed "
    "pairwise speaker-listener updates and N is the population size."
)

def panel_title_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"fontweight": "normal"}
    kwargs.update(overrides)
    return kwargs


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
    setup_paper_house_style()
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


def setup_mechanism_sans_style() -> None:
    setup_paper_house_style()
    add_paper_sans_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": PAPER_SANS_FONTS,
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
            "font.size": 11.0,
            "axes.labelsize": 11.3,
            "axes.titlesize": 11.0,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.4,
            "legend.fontsize": 10.6,
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str, *, alias_stems: tuple[str, ...] = ()) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    for alias_stem in alias_stems:
        fig.savefig(FIGURE_DIR / f"{alias_stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{alias_stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURE_DIR / f"{alias_stem}.svg", bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def read_last_csv_row(path: Path) -> dict[str, str]:
    last: dict[str, str] | None = None
    with path.open(newline="") as handle:
        for last in csv.DictReader(handle):
            pass
    if last is None:
        raise ValueError(f"No rows in {path}")
    return last


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float("nan")
    return float(value)


def endpoint_outcome(final_probe: dict[str, str], summary: dict[str, Any]) -> str:
    country_shares = sorted(
        (
            (str(country), as_float(final_probe, str(country)))
            for country in summary["countries"]
            if final_probe.get(str(country), "") != ""
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not country_shares:
        return "fragmentation"

    consensus_country, top1_share = country_shares[0]
    truth_country = str(summary["truth_country"])
    if top1_share >= CONSENSUS_THRESHOLD:
        return "correct_consensus" if consensus_country == truth_country else "wrong_consensus"

    substantial_count = sum(1 for _, share in country_shares if share >= POLARIZATION_THRESHOLD)
    return "polarization" if substantial_count >= 2 else "fragmentation"


def unique_top_country(country_shares: list[tuple[str, float]]) -> str | None:
    if not country_shares:
        return None
    top_share = max(share for _, share in country_shares)
    winners = [country for country, share in country_shares if share == top_share]
    if len(winners) != 1 or top_share <= 0.0:
        return None
    return winners[0]


def final_vote_from_probe(final_probe: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    country_shares = [
        (str(country), as_float(final_probe, str(country)))
        for country in summary["countries"]
        if final_probe.get(str(country), "") != ""
    ]
    top_country = unique_top_country(country_shares)
    truth_country = str(summary["truth_country"])
    correct = top_country == truth_country
    return {
        "country": top_country,
        "correct": correct,
        "accuracy": 1.0 if correct else 0.0,
    }


def final_vote_from_counts(counts: Counter[str], truth_country: str) -> dict[str, Any]:
    top_country = unique_top_country([(country, float(count)) for country, count in counts.items()])
    correct = top_country == truth_country
    return {
        "country": top_country,
        "correct": correct,
        "accuracy": 1.0 if correct else 0.0,
    }


def load_seed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(SOURCE_ROOT.glob("N*/**/seed_*")):
        summary_path = seed_dir / "summary.json"
        per_round_path = seed_dir / "per_round.csv"
        if not summary_path.exists() or not per_round_path.exists():
            continue

        n = int(seed_dir.parts[-3].removeprefix("N"))
        condition = seed_dir.parts[-2]
        if n not in N_VALUES or condition not in PLOTTED_CONDITIONS:
            continue

        summary = json.loads(summary_path.read_text())
        final_probe = read_last_csv_row(per_round_path)
        final_vote = final_vote_from_probe(final_probe, summary)
        rows.append(
            {
                "N": n,
                "condition": condition,
                "seed": seed_dir.name,
                "initial_accuracy": float(summary["initial_accuracy"]),
                "final_accuracy": float(summary["final_accuracy"]),
                "final_vote_country": final_vote["country"],
                "final_vote_correct": final_vote["correct"],
                "final_vote_accuracy": final_vote["accuracy"],
                "gain": float(summary["collaboration_gain_over_initial_accuracy"]),
                "final_outcome": endpoint_outcome(final_probe, summary),
                "stored_final_outcome": str(summary["final_outcome"]),
                "top1_share": as_float(final_probe, "top1_share"),
                "top2_share": as_float(final_probe, "top2_share"),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No seed outputs found under {SOURCE_ROOT}")
    return rows


def bootstrap_interval(values: list[float], *, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) <= 1:
        value = values[0] if values else float("nan")
        return value, value
    samples = np.asarray(values, dtype=float)
    draws = rng.choice(samples, size=(8000, len(samples)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260428)
    summary_rows: list[dict[str, Any]] = []
    for condition in PLOTTED_CONDITIONS:
        for n in N_VALUES:
            part = [row for row in seed_rows if row["condition"] == condition and row["N"] == n]
            if not part:
                continue
            final_values = [float(row["final_accuracy"]) for row in part]
            vote_values = [float(row["final_vote_accuracy"]) for row in part]
            lo, hi = bootstrap_interval(final_values, rng=rng)
            vote_lo, vote_hi = bootstrap_interval(vote_values, rng=rng)
            outcomes = Counter(str(row["final_outcome"]) for row in part)
            row: dict[str, Any] = {
                "condition": condition,
                "N": n,
                "seeds": len(part),
                "initial_mean": float(np.mean([float(item["initial_accuracy"]) for item in part])),
                "final_mean": float(np.mean(final_values)),
                "final_ci_low": lo,
                "final_ci_high": hi,
                "final_vote_accuracy_rate": float(np.mean(vote_values)),
                "final_vote_accuracy_ci_low": vote_lo,
                "final_vote_accuracy_ci_high": vote_hi,
                "gain_mean": float(np.mean([float(item["gain"]) for item in part])),
                "top1_mean": float(np.mean([float(item["top1_share"]) for item in part])),
                "top2_mean": float(np.mean([float(item["top2_share"]) for item in part])),
            }
            for outcome in OUTCOME_ORDER:
                row[f"{outcome}_count"] = int(outcomes.get(outcome, 0))
                row[f"{outcome}_rate"] = float(outcomes.get(outcome, 0) / len(part))
            summary_rows.append(row)
    return summary_rows


def lookup(summary_rows: list[dict[str, Any]], condition: str, n: int, key: str, default: float = np.nan) -> float:
    for row in summary_rows:
        if row["condition"] == condition and row["N"] == n:
            return float(row.get(key, default))
    return default


def lookup_row(summary_rows: list[dict[str, Any]], condition: str, n: int) -> dict[str, Any]:
    for row in summary_rows:
        if row["condition"] == condition and row["N"] == n:
            return row
    raise KeyError(f"Missing summary row for condition={condition}, N={n}")


def write_main_summary(
    seed_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    stem: str = "flag_game_pairwise_n_scaling_main",
    extra_payload: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{stem}_summary.csv"
    json_path = DATA_DIR / f"{stem}_summary.json"
    fieldnames = [
        "condition",
        "N",
        "seeds",
        "initial_mean",
        "final_mean",
        "final_ci_low",
        "final_ci_high",
        "final_vote_accuracy_rate",
        "final_vote_accuracy_ci_low",
        "final_vote_accuracy_ci_high",
        "gain_mean",
        "correct_consensus_rate",
        "wrong_consensus_rate",
        "polarization_rate",
        "fragmentation_rate",
        "correct_consensus_count",
        "wrong_consensus_count",
        "polarization_count",
        "fragmentation_count",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    payload = {
        "source_root": str(SOURCE_ROOT),
        "seed_rows": len(seed_rows),
        "n_values": N_VALUES,
        "plotted_conditions": PLOTTED_CONDITIONS,
        "outcome_thresholds": {
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "polarization_threshold": POLARIZATION_THRESHOLD,
        },
        "outcome_definitions": OUTCOME_DEFINITIONS,
        "final_vote_accuracy_definition": (
            "For pairwise runs, final_vote_accuracy is 1 when the unique top-voted final "
            "country is the truth country; ties receive 0."
        ),
        "summary_rows": summary_rows,
    }
    if extra_payload:
        payload.update(extra_payload)
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


def seed_number(seed_name: str) -> int:
    return int(seed_name.removeprefix("seed_"))


def filter_main_v3_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded = set(MAIN_V3_EXCLUDED_SEEDS)
    return [row for row in seed_rows if seed_number(str(row["seed"])) not in excluded]


def format_seed_count_note(summary_rows: list[dict[str, Any]], *, condition: str | None = None) -> str:
    plotted_conditions = [condition] if condition else PLOTTED_CONDITIONS
    count_by_n: dict[int, tuple[int | None, ...]] = {}
    for n in N_VALUES:
        count_by_n[n] = tuple(
            int(row["seeds"]) if row else None
            for row in (
                next((r for r in summary_rows if r["condition"] == plotted_condition and r["N"] == n), None)
                for plotted_condition in plotted_conditions
            )
        )

    groups: list[tuple[int, int, tuple[int | None, ...]]] = []
    for n in N_VALUES:
        counts = count_by_n[n]
        if groups and groups[-1][2] == counts:
            groups[-1] = (groups[-1][0], n, counts)
        else:
            groups.append((n, n, counts))

    parts: list[str] = []
    for first_n, last_n, counts in groups:
        n_label = f"N={first_n}" if first_n == last_n else f"N={first_n}-{last_n}"
        nonmissing_counts = [count for count in counts if count is not None]
        if condition and nonmissing_counts:
            parts.append(f"{n_label} n={nonmissing_counts[0]}")
            continue
        if nonmissing_counts and len(set(nonmissing_counts)) == 1:
            parts.append(f"{n_label} n={nonmissing_counts[0]}/model")
            continue
        model_counts = []
        for plotted_condition, count in zip(plotted_conditions, counts, strict=True):
            if count is None:
                continue
            model_counts.append(f"{CONDITION_STYLES[plotted_condition]['label']} n={count}")
        parts.append(f"{n_label} " + ", ".join(model_counts))

    prefix = f"{CONDITION_STYLES[condition]['label']} seeds" if condition else "Seed counts"
    return prefix + ": " + "; ".join(parts) + "."


def draw_main_figure(
    summary_rows: list[dict[str, Any]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = "flag_game_pairwise_n_scaling_main",
    seed_count_note: str | None = None,
) -> tuple[Path, Path]:
    apply_paper_sans_rc()
    condition = "all_gpt_4o"
    x = np.array(N_VALUES, dtype=float)
    fig_height = 2.42 if seed_count_note else 2.18
    fig, axes = plt.subplots(1, 2, figsize=(6.55, fig_height), constrained_layout=False, sharex=True)

    def setup_scaling_axis(ax: plt.Axes, *, title: str, ylabel: str) -> None:
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES])
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title, loc="left", **panel_title_kwargs(color=HOUSE_COLORS["black"]))
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Population size $N$")
        style_axis(ax, grid=True)
        ax.title.set_color(HOUSE_COLORS["black"])
        ax.xaxis.label.set_color(HOUSE_COLORS["black"])
        ax.yaxis.label.set_color(HOUSE_COLORS["black"])
        ax.tick_params(axis="both", colors=HOUSE_COLORS["black"])

    def draw_ciq_series(
        ax: plt.Axes,
        values: list[float],
        lows: list[float],
        highs: list[float],
        *,
        color: str,
        title: str,
        ylabel: str,
    ) -> None:
        y = np.asarray(values, dtype=float)
        lo = np.asarray(lows, dtype=float)
        hi = np.asarray(highs, dtype=float)
        ax.plot(
            x,
            y,
            "-o",
            color=color,
            linewidth=1.55,
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.05,
            label="Collective",
        )
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt="none",
            ecolor=color,
            alpha=0.45,
            capsize=2.0,
            capthick=0.8,
            elinewidth=0.75,
        )
        setup_scaling_axis(ax, title=title, ylabel=ylabel)

    ciq_values = [lookup(summary_rows, condition, n, "final_mean") for n in N_VALUES]
    ciq_lows = [lookup(summary_rows, condition, n, "final_ci_low") for n in N_VALUES]
    ciq_highs = [lookup(summary_rows, condition, n, "final_ci_high") for n in N_VALUES]
    draw_ciq_series(
        axes[0],
        ciq_values,
        ciq_lows,
        ciq_highs,
        color=HOUSE_COLORS["blue"],
        title="a) Collective mean accuracy",
        ylabel="Accuracy rate" if include_vote_accuracy else "Collective mean accuracy",
    )
    if include_vote_accuracy:
        vote_values = np.array([lookup(summary_rows, condition, n, "final_vote_accuracy_rate") for n in N_VALUES])
        vote_lows = np.array([lookup(summary_rows, condition, n, "final_vote_accuracy_ci_low") for n in N_VALUES])
        vote_highs = np.array([lookup(summary_rows, condition, n, "final_vote_accuracy_ci_high") for n in N_VALUES])
        axes[0].plot(
            x,
            vote_values,
            "--s",
            color=HOUSE_COLORS["teal"],
            linewidth=1.35,
            markersize=3.6,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="Final vote accuracy",
        )
        axes[0].errorbar(
            x,
            vote_values,
            yerr=np.vstack([vote_values - vote_lows, vote_highs - vote_values]),
            fmt="none",
            ecolor=HOUSE_COLORS["teal"],
            alpha=0.35,
            capsize=2.0,
            capthick=0.75,
            elinewidth=0.7,
        )
        axes[0].legend(frameon=False, loc="lower right", fontsize=6.8, handlelength=2.0)

    outcome_specs = [
        ("correct_consensus", "Correct consensus", OUTCOME_COLORS["correct_consensus"], "o"),
        ("wrong_consensus", "Wrong consensus", OUTCOME_COLORS["wrong_consensus"], "s"),
        ("polarization", "Polarization", OUTCOME_COLORS["polarization"], "^"),
    ]
    for outcome, label, color, marker in outcome_specs:
        y = [lookup(summary_rows, condition, n, f"{outcome}_rate") for n in N_VALUES]
        axes[1].plot(
            x,
            y,
            f"-{marker}",
            color=color,
            linewidth=1.45,
            markersize=4.0,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.05,
            label=label,
        )
    setup_scaling_axis(
        axes[1],
        title="b) Endpoint outcomes",
        ylabel="Fraction of runs",
    )
    axes[1].legend(
        loc="upper right",
        frameon=False,
        fontsize=6.9,
        handlelength=1.7,
        borderaxespad=0.2,
    )

    if seed_count_note:
        fig.text(
            0.5,
            0.055,
            seed_count_note,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=HOUSE_COLORS["gray"],
        )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.84 if seed_count_note else 0.86,
        bottom=0.29 if seed_count_note else 0.235,
        wspace=0.30,
    )
    return save_figure(fig, stem)


def draw_model_comparison_figure(
    summary_rows: list[dict[str, Any]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = MODEL_COMPARISON_OUT_STEM,
) -> tuple[Path, Path]:
    setup_paper_house_style()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.7,
        }
    )
    fig = plt.figure(figsize=(7.15, 3.45), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.07])
    ax_ciq = fig.add_subplot(grid[0, 0])
    ax_mix = fig.add_subplot(grid[0, 1])

    x = np.arange(len(N_VALUES), dtype=float)
    for condition in PLOTTED_CONDITIONS:
        style = CONDITION_STYLES[condition]
        y = np.array([lookup(summary_rows, condition, n, "final_mean") for n in N_VALUES])
        lo = np.array([lookup(summary_rows, condition, n, "final_ci_low") for n in N_VALUES])
        hi = np.array([lookup(summary_rows, condition, n, "final_ci_high") for n in N_VALUES])
        yerr = np.vstack([y - lo, hi - y])
        ax_ciq.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"-{style['marker']}",
            color=str(style["color"]),
            linewidth=1.45,
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=1.1,
            capsize=2.0,
            capthick=0.8,
            elinewidth=0.8,
            label=str(style["label"]),
        )
        if include_vote_accuracy:
            vote_y = np.array([lookup(summary_rows, condition, n, "final_vote_accuracy_rate") for n in N_VALUES])
            vote_lo = np.array([lookup(summary_rows, condition, n, "final_vote_accuracy_ci_low") for n in N_VALUES])
            vote_hi = np.array([lookup(summary_rows, condition, n, "final_vote_accuracy_ci_high") for n in N_VALUES])
            ax_ciq.errorbar(
                x,
                vote_y,
                yerr=np.vstack([vote_y - vote_lo, vote_hi - vote_y]),
                fmt=f"--{style['marker']}",
                color=str(style["color"]),
                linewidth=1.15,
                markersize=3.4,
                markerfacecolor=str(style["color"]),
                markeredgewidth=0.8,
                capsize=1.8,
                capthick=0.7,
                elinewidth=0.7,
                alpha=0.78,
            )
    ax_ciq.set_title(
        "a) Collective mean accuracy by population size",
        loc="left",
        **panel_title_kwargs(color=HOUSE_COLORS["black"]),
    )
    ax_ciq.set_xlabel("Population size N")
    ax_ciq.set_ylabel("Accuracy rate" if include_vote_accuracy else "Collective mean accuracy")
    ax_ciq.set_xticks(x)
    ax_ciq.set_xticklabels([str(n) for n in N_VALUES])
    ax_ciq.set_ylim(0.0, 1.02)
    ax_ciq.set_xlim(-0.25, x[-1] + 0.92)
    style_axis(ax_ciq, grid=True)
    if include_vote_accuracy:
        ax_ciq.legend(
            handles=[
                Line2D([0], [0], color=HOUSE_COLORS["gray"], linestyle="-", linewidth=1.35, label="Collective"),
                Line2D(
                    [0],
                    [0],
                    color=HOUSE_COLORS["gray"],
                    linestyle="--",
                    linewidth=1.15,
                    label="Final vote accuracy",
                ),
            ],
            frameon=False,
            loc="lower left",
            fontsize=6.8,
            handlelength=2.0,
        )
    for condition in PLOTTED_CONDITIONS:
        style = CONDITION_STYLES[condition]
        ax_ciq.text(
            x[-1] + 0.24,
            lookup(summary_rows, condition, N_VALUES[-1], "final_mean"),
            str(style["label"]),
            ha="left",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=str(style["color"]),
            clip_on=False,
        )

    width = 0.34
    offsets = [-width / 1.85, width / 1.85]
    for offset, condition in zip(offsets, PLOTTED_CONDITIONS, strict=True):
        bottoms = np.zeros(len(N_VALUES))
        for outcome in OUTCOME_ORDER:
            heights = np.array([lookup(summary_rows, condition, n, f"{outcome}_rate", 0.0) for n in N_VALUES])
            ax_mix.bar(
                x + offset,
                heights,
                width=width,
                bottom=bottoms,
                color=OUTCOME_COLORS[outcome],
                edgecolor="white",
                linewidth=0.55,
            )
            bottoms += heights
        for xpos in x + offset:
            ax_mix.text(
                xpos,
                1.018,
                str(CONDITION_STYLES[condition]["short"]),
                ha="center",
                va="bottom",
                fontsize=7.6,
                color=HOUSE_COLORS["black"],
                clip_on=False,
            )
    ax_mix.set_title(
        "b) Endpoint outcomes",
        loc="left",
        **panel_title_kwargs(color=HOUSE_COLORS["black"]),
    )
    ax_mix.set_xlabel("Population size N")
    ax_mix.set_ylabel("Fraction of runs")
    ax_mix.set_xticks(x)
    ax_mix.set_xticklabels([str(n) for n in N_VALUES])
    ax_mix.set_ylim(0.0, 1.08)
    style_axis(ax_mix, grid=True)

    outcome_handles = [
        Patch(facecolor=OUTCOME_COLORS[outcome], edgecolor="white", label=OUTCOME_LABELS[outcome])
        for outcome in OUTCOME_ORDER
    ]
    fig.legend(
        handles=outcome_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=4,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.0,
        borderaxespad=0.0,
    )
    definition_text = (
        r"Correct: top country share $\geq 0.85$ and true; "
        r"Wrong: top country share $\geq 0.85$ and false;" "\n"
        r"Polarization: no consensus and two countries $\geq 0.25$; "
        r"Fragmentation: no consensus and fewer than two countries $\geq 0.25$."
    )
    fig.text(
        0.5,
        0.068,
        definition_text,
        ha="center",
        va="bottom",
        fontsize=6.1,
        color=HOUSE_COLORS["gray"],
        linespacing=1.18,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.31, wspace=0.32)
    return save_figure(fig, stem)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def seed_dir_for(n: int) -> Path:
    return SOURCE_ROOT / f"N{n}" / "all_gpt_4o" / APPENDIX_CASE["seed"]


def load_final_probe_rows(seed_dir: Path) -> dict[int, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (seed_dir / "probes.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No probe rows found in {seed_dir}")
    final_t = max(int(row["t"]) for row in rows)
    return {int(row["agent_id"]): row for row in rows if int(row["t"]) == final_t}


def final_country(row: dict[str, Any] | None) -> str:
    if not row:
        return "Missing"
    if not row.get("valid", False):
        return "Invalid"
    return str(row.get("country") or "Invalid")


def assignment_center(assignment: dict[str, Any]) -> tuple[float, float]:
    return (
        float(assignment["left"]) + float(assignment["width"]) / 2.0,
        float(assignment["top"]) + float(assignment["height"]) / 2.0,
    )


def assignment_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax0 = float(a["left"])
    ay0 = float(a["top"])
    ax1 = ax0 + float(a["width"])
    ay1 = ay0 + float(a["height"])
    bx0 = float(b["left"])
    by0 = float(b["top"])
    bx1 = bx0 + float(b["width"])
    by1 = by0 + float(b["height"])
    overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = overlap_w * overlap_h
    if intersection == 0.0:
        return 0.0
    area_a = float(a["width"]) * float(a["height"])
    area_b = float(b["width"]) * float(b["height"])
    return intersection / (area_a + area_b - intersection)


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def spatial_metrics(assignments: list[dict[str, Any]], final_by_agent: dict[int, dict[str, Any]]) -> dict[str, Any]:
    centers = {int(a["agent_id"]): assignment_center(a) for a in assignments}
    assignment_by_agent = {int(a["agent_id"]): a for a in assignments}
    countries = {int(a["agent_id"]): final_country(final_by_agent.get(int(a["agent_id"]))) for a in assignments}

    if len(centers) <= 1:
        return {
            "nearest_same_country_fraction": None,
            "mean_same_country_center_distance": None,
            "mean_different_country_center_distance": None,
            "mean_same_country_iou": None,
            "mean_different_country_iou": None,
            "same_country_pair_count": 0,
            "different_country_pair_count": 0,
        }

    nearest_same: list[float] = []
    for agent_id, center in centers.items():
        distances = []
        for other_id, other_center in centers.items():
            if other_id == agent_id:
                continue
            distance = float(np.hypot(center[0] - other_center[0], center[1] - other_center[1]))
            distances.append((distance, other_id))
        nearest_id = min(distances, key=lambda item: item[0])[1]
        nearest_same.append(1.0 if countries[agent_id] == countries[nearest_id] else 0.0)

    same_distances: list[float] = []
    different_distances: list[float] = []
    same_ious: list[float] = []
    different_ious: list[float] = []
    for left_id, right_id in combinations(centers, 2):
        distance = float(np.hypot(centers[left_id][0] - centers[right_id][0], centers[left_id][1] - centers[right_id][1]))
        iou = assignment_iou(assignment_by_agent[left_id], assignment_by_agent[right_id])
        if countries[left_id] == countries[right_id]:
            same_distances.append(distance)
            same_ious.append(iou)
        else:
            different_distances.append(distance)
            different_ious.append(iou)

    return {
        "nearest_same_country_fraction": mean_or_none(nearest_same),
        "mean_same_country_center_distance": mean_or_none(same_distances),
        "mean_different_country_center_distance": mean_or_none(different_distances),
        "mean_same_country_iou": mean_or_none(same_ious),
        "mean_different_country_iou": mean_or_none(different_ious),
        "same_country_pair_count": len(same_distances),
        "different_country_pair_count": len(different_distances),
    }


def load_map_panel(n: int) -> dict[str, Any]:
    seed_dir = seed_dir_for(n)
    manifest = load_json(seed_dir / "trial_manifest.json")
    summary = load_json(seed_dir / "summary.json")
    final_by_agent = load_final_probe_rows(seed_dir)
    assignments = list(manifest["assignments"])
    final_countries = {
        int(assignment["agent_id"]): final_country(final_by_agent.get(int(assignment["agent_id"])))
        for assignment in assignments
    }
    return {
        "N": n,
        "seed_dir": seed_dir,
        "manifest": manifest,
        "summary": summary,
        "assignments": assignments,
        "final_countries": final_countries,
        "counts": Counter(final_countries.values()),
        "metrics": spatial_metrics(assignments, final_by_agent),
    }


def load_trace(n: int) -> tuple[dict[str, Any], list[dict[str, float]]]:
    seed_dir = seed_dir_for(n)
    summary = load_json(seed_dir / "summary.json")
    rows: list[dict[str, float]] = []
    with (seed_dir / "per_round.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            truth_share = float(row[APPENDIX_CASE["truth"]])
            rival_share = float(row[APPENDIX_CASE["rival"]])
            rows.append(
                {
                    "t": float(row["t"]),
                    "interaction_round": float(row["t"]) / n,
                    "truth": truth_share,
                    "rival": rival_share,
                    "other": max(0.0, 1.0 - truth_share - rival_share),
                    "support": float(row["support_size"]),
                    "top1": float(row["top1_share"]),
                    "top2": float(row["top2_share"]),
                }
            )
    if not rows:
        raise ValueError(f"No per-round rows found in {seed_dir}")
    return summary, rows


def color_for_country(country: str, used: dict[str, str]) -> str:
    if country not in {APPENDIX_CASE["truth"], APPENDIX_CASE["rival"]}:
        return HOUSE_COLORS["gray"]
    if country in COUNTRY_COLORS:
        return COUNTRY_COLORS[country]
    if country not in used:
        used[country] = FALLBACK_COUNTRY_COLORS[len(used) % len(FALLBACK_COUNTRY_COLORS)]
    return used[country]


def top_count_text(counts: Counter[str], max_items: int = 2) -> str:
    display_counts: Counter[str] = Counter()
    focal = {APPENDIX_CASE["truth"], APPENDIX_CASE["rival"]}
    for country, count in counts.items():
        display_counts[country if country in focal else "Other"] += count
    parts = [f"{country} {count}" for country, count in display_counts.most_common(max_items)]
    if len(display_counts) > max_items:
        parts.append("...")
    return ", ".join(parts)


def draw_crop_map(ax: plt.Axes, panel: dict[str, Any], extra_colors: dict[str, str]) -> None:
    manifest = panel["manifest"]
    canvas = manifest["canvas"]
    width = float(canvas["width"])
    height = float(canvas["height"])
    truth_path = panel["seed_dir"] / "artifacts" / "truth_flag.png"
    if truth_path.exists():
        image = plt.imread(truth_path)
        ax.imshow(image, extent=[0.0, width, height, 0.0], alpha=0.28, zorder=0)

    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(color="#FFFFFF", alpha=0.35, linewidth=0.6)

    for assignment in panel["assignments"]:
        agent_id = int(assignment["agent_id"])
        country = panel["final_countries"][agent_id]
        color = color_for_country(country, extra_colors)
        left = float(assignment["left"])
        top = float(assignment["top"])
        crop_width = float(assignment["width"])
        crop_height = float(assignment["height"])
        ax.add_patch(
            Rectangle(
                (left, top),
                crop_width,
                crop_height,
                facecolor=color,
                edgecolor=color,
                linewidth=1.1,
                alpha=0.28,
                zorder=2,
            )
        )
        center_x, center_y = assignment_center(assignment)
        marker_size = 30 if panel["N"] <= 16 else 20 if panel["N"] <= 64 else 10
        ax.scatter(
            [center_x],
            [center_y],
            s=marker_size,
            facecolors="white",
            edgecolors=color,
            linewidth=0.9,
            zorder=4,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(HOUSE_COLORS["light_gray"])
        spine.set_linewidth(0.8)

    outcome = str(panel["summary"]["final_outcome"]).replace("_", " ")
    ax.set_title(
        f"N = {panel['N']}: {outcome}\n{top_count_text(panel['counts'])}",
        **panel_title_kwargs(fontsize=9.8),
    )


def draw_trace(ax: plt.Axes, n: int) -> dict[str, Any]:
    summary, rows = load_trace(n)
    x = [row["interaction_round"] for row in rows]
    ax.plot(x, [row["truth"] for row in rows], color=COUNTRY_COLORS[APPENDIX_CASE["truth"]], linewidth=2.25)
    ax.plot(x, [row["rival"] for row in rows], color=COUNTRY_COLORS[APPENDIX_CASE["rival"]], linewidth=2.05)
    ax.plot(x, [row["other"] for row in rows], color=HOUSE_COLORS["gray"], linewidth=1.55, linestyle="--")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("")
    ax.set_yticks([0.0, 0.5, 1.0])
    style_axis(ax, grid=True)
    return {
        "N": n,
        "final_outcome": summary["final_outcome"],
        "time_to_correct_consensus": summary.get("time_to_correct_consensus"),
        "final_truth_share": rows[-1]["truth"],
        "final_rival_share": rows[-1]["rival"],
        "final_other_share": rows[-1]["other"],
        "final_interaction_round": rows[-1]["interaction_round"],
    }


def write_mechanism_summary(
    map_panels: list[dict[str, Any]],
    trace_summaries: list[dict[str, Any]],
    *,
    stem: str,
    n_values: list[int],
    description: str,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{stem}_summary.json"
    payload = {
        "source_root": str(SOURCE_ROOT),
        "case": {
            "truth": APPENDIX_CASE["truth"],
            "rival": APPENDIX_CASE["rival"],
            "description": description,
        },
        "n_values": n_values,
        "interaction_round_definition": INTERACTION_ROUND_DEFINITION,
        "map_panels": [
            {
                "N": panel["N"],
                "final_outcome": panel["summary"]["final_outcome"],
                "final_accuracy": panel["summary"]["final_accuracy"],
                "final_vote_accuracy": panel["summary"].get(
                    "final_vote_accuracy",
                    final_vote_from_counts(panel["counts"], str(panel["summary"]["truth_country"]))["accuracy"],
                ),
                "final_vote_country": panel["summary"].get(
                    "final_vote_country",
                    final_vote_from_counts(panel["counts"], str(panel["summary"]["truth_country"]))["country"],
                ),
                "final_counts": dict(panel["counts"].most_common()),
                **panel["metrics"],
            }
            for panel in map_panels
        ],
        "trace_panels": trace_summaries,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def draw_mechanism_figure(
    *,
    n_values: list[int],
    stem: str,
    figsize: tuple[float, float],
    description: str,
) -> tuple[Path, Path, Path]:
    setup_mechanism_sans_style()
    map_panels = [load_map_panel(n) for n in n_values]
    extra_colors: dict[str, str] = {}
    fig, axes = plt.subplots(
        2,
        len(n_values),
        figsize=figsize,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [0.92, 0.86]},
        sharey="row",
    )

    for col_index, (_n, panel) in enumerate(zip(n_values, map_panels, strict=True)):
        draw_crop_map(axes[0, col_index], panel, extra_colors)

    trace_summaries: list[dict[str, Any]] = []
    for col_index, n in enumerate(n_values):
        trace_summaries.append(draw_trace(axes[1, col_index], n))
        axes[1, col_index].set_box_aspect(0.72)
        if col_index == 0:
            axes[1, col_index].set_ylabel("Share of agents", fontsize=11.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COUNTRY_COLORS[APPENDIX_CASE["truth"]],
            marker="s",
            linewidth=2.0,
            markersize=5.2,
            markerfacecolor=COUNTRY_COLORS[APPENDIX_CASE["truth"]],
            markeredgecolor=COUNTRY_COLORS[APPENDIX_CASE["truth"]],
            markeredgewidth=1.0,
            label=f"{APPENDIX_CASE['truth']} (truth)",
        ),
        Line2D(
            [0],
            [0],
            color=COUNTRY_COLORS[APPENDIX_CASE["rival"]],
            marker="s",
            linewidth=1.9,
            markersize=5.2,
            markerfacecolor=COUNTRY_COLORS[APPENDIX_CASE["rival"]],
            markeredgecolor=COUNTRY_COLORS[APPENDIX_CASE["rival"]],
            markeredgewidth=1.0,
            label=f"{APPENDIX_CASE['rival']} (rival)",
        ),
        Line2D([0], [0], color=HOUSE_COLORS["gray"], linewidth=1.45, linestyle="--", label="Other countries"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.9,
    )
    fig.subplots_adjust(left=0.058, right=0.997, top=0.765, bottom=0.17, wspace=0.22, hspace=0.08)
    top_row = axes[0, 0].get_position()
    bottom_row = axes[1, 0].get_position()
    fig.text(
        0.055,
        min(0.855, top_row.y1 + 0.078),
        "a) Final predictions over initial crops",
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        0.055,
        bottom_row.y1 + 0.018,
        "b) Country shares over interaction rounds",
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        0.52,
        0.128,
        "Interaction rounds",
        ha="center",
        va="center",
        fontsize=11.3,
        color=HOUSE_COLORS["black"],
    )
    png_path, pdf_path = save_figure(
        fig,
        stem,
    )
    summary_path = write_mechanism_summary(
        map_panels,
        trace_summaries,
        stem=stem,
        n_values=n_values,
        description=description,
    )
    return png_path, pdf_path, summary_path


def style_empirical_endpoint_axis(
    ax: plt.Axes,
    population: list[dict[str, float]],
    *,
    ylabel: str,
    xlabel: str = "Population size N",
    ymax: float = 1.03,
) -> None:
    x = np.array([row["x"] for row in population], dtype=float)
    ax.set_xscale("log", base=2)
    ax.set_xlim(x[0] * 0.86, x[-1] * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in population])
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=3)
    ax.grid(axis="y", color=EMPIRICAL_LIGHT_GRAY, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(EMPIRICAL_INK)
    ax.spines["bottom"].set_color(EMPIRICAL_INK)
    ax.tick_params(colors=EMPIRICAL_INK)
    ax.yaxis.label.set_color(EMPIRICAL_INK)
    ax.xaxis.label.set_color(EMPIRICAL_INK)


def draw_correct_consensus_panel(
    ax: plt.Axes,
    population: list[dict[str, float]],
    *,
    title: str | None,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    x = np.array([row["x"] for row in population], dtype=float)
    correct = np.array([row["correct_consensus"] for row in population], dtype=float)
    ax.plot(
        x,
        correct,
        color=EMPIRICAL_GREEN,
        marker="o",
        markersize=4.7,
        markerfacecolor="white",
        markeredgewidth=1.35,
        linewidth=1.85,
        label="Correct consensus",
        zorder=3,
    )
    style_empirical_endpoint_axis(
        ax,
        population,
        ylabel="Fraction of runs" if show_ylabel else "",
        xlabel="Population size N" if show_xlabel else "",
        ymax=1.03,
    )
    ax.set_yticks([0.0, 0.5, 1.0])
    if not show_xlabel:
        ax.tick_params(axis="x", labelbottom=False)
    if title:
        empirical_panel_title(ax, title)
    ax.legend(frameon=False, loc="lower left", fontsize=6.5, handlelength=1.65, borderaxespad=0.15)


def draw_failure_decomposition_panel(
    ax: plt.Axes,
    population: list[dict[str, float]],
    *,
    title: str | None,
    show_total_line: bool,
    ylabel: str | None = None,
    ymax: float = 0.75,
) -> None:
    x = np.array([row["x"] for row in population], dtype=float)
    wrong = np.array([row["wrong_consensus"] for row in population], dtype=float)
    polarization = np.array([row["polarization"] for row in population], dtype=float)
    ax.plot(
        x,
        wrong,
        color=EMPIRICAL_ORANGE,
        marker="s",
        markersize=4.7,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linewidth=1.75,
        label="Wrong consensus",
        zorder=4,
    )
    ax.plot(
        x,
        polarization,
        color=EMPIRICAL_PURPLE,
        marker="^",
        markersize=4.9,
        markerfacecolor="white",
        markeredgewidth=1.25,
        linewidth=1.75,
        label="Polarization",
        zorder=4,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlim(x[0] / 1.28, x[-1] * 1.28)
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in population])
    ax.set_ylim(0.0, ymax)
    ax.set_ylabel(ylabel if ylabel is not None else ("Failure fraction" if show_total_line else "Fraction of runs"))
    ax.set_xlabel("Population size N", labelpad=3)
    ax.grid(axis="y", color=EMPIRICAL_LIGHT_GRAY, alpha=0.65, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(EMPIRICAL_INK)
    ax.spines["bottom"].set_color(EMPIRICAL_INK)
    ax.tick_params(colors=EMPIRICAL_INK)
    ax.yaxis.label.set_color(EMPIRICAL_INK)
    ax.xaxis.label.set_color(EMPIRICAL_INK)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75] if ymax <= 0.75 else [0.0, 0.5, 1.0])
    if title:
        empirical_panel_title(ax, title)
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=6.4,
        handlelength=1.4,
        handletextpad=0.42,
        borderaxespad=0.12,
    )


def draw_empirical_ab_panels(
    ax_population: plt.Axes,
    endpoint_axes: dict[str, plt.Axes],
    *,
    endpoint_mode: str,
) -> list[dict[str, float]]:
    population = empirical_pairwise_rows()
    x = np.array([row["x"] for row in population], dtype=float)
    draw_empirical_metric_lines(
        ax_population,
        x,
        np.array([row["iiq"] for row in population], dtype=float),
        np.array([row["terminal"] for row in population], dtype=float),
        np.array([row["ciq"] for row in population], dtype=float),
        xlabel="Population size N",
        ymax=0.75,
    )
    ax_population.set_xscale("log", base=2)
    ax_population.set_xticks(x)
    ax_population.set_xticklabels([row["label"] for row in population])
    ax_population.set_xlabel("")
    ax_population.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=EMPIRICAL_BLUE_DARK,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.25,
                linewidth=1.85,
                label="Collective",
            ),
            Line2D(
                [0],
                [0],
                color=EMPIRICAL_GREEN,
                marker="^",
                markerfacecolor="white",
                markeredgewidth=1.05,
                linewidth=1.45,
                label="Majority vote",
            ),
            Line2D(
                [0],
                [0],
                color=EMPIRICAL_GRAY,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.05,
                linewidth=1.3,
                linestyle=(0, (2.0, 1.4)),
                label="Initial",
            ),
            Patch(facecolor=EMPIRICAL_BLUE_LIGHT, edgecolor="none", label="Social uplift"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=6.4,
        handlelength=1.45,
        handletextpad=0.42,
        borderaxespad=0.1,
    )

    if endpoint_mode == "endpoint_split":
        draw_correct_consensus_panel(
            endpoint_axes["correct"],
            population,
            title=None,
            show_xlabel=False,
            show_ylabel=False,
        )
        draw_failure_decomposition_panel(
            endpoint_axes["failures"],
            population,
            title=None,
            show_total_line=False,
            ylabel="",
        )
    elif endpoint_mode == "failure_decomposition":
        draw_failure_decomposition_panel(
            endpoint_axes["failures"],
            population,
            title=None,
            show_total_line=True,
        )
    else:
        raise ValueError(f"Unknown endpoint_mode: {endpoint_mode}")
    return population


def draw_empirical_ab_mechanism_figure(
    *,
    n_values: list[int],
    stem: str,
    figsize: tuple[float, float],
    description: str,
    endpoint_mode: str,
) -> tuple[Path, Path, Path]:
    setup_mechanism_sans_style()
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.labelsize": 9.4,
            "axes.titlesize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.2,
        }
    )
    map_panels = [load_map_panel(n) for n in n_values]
    extra_colors: dict[str, str] = {}
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.08, 3.18], wspace=0.17)
    left_grid = outer[0, 0].subgridspec(2, 1, hspace=0.52)
    right_grid = outer[0, 1].subgridspec(
        2,
        len(n_values),
        height_ratios=[0.92, 0.86],
        hspace=0.18,
        wspace=0.22,
    )

    ax_population = fig.add_subplot(left_grid[0, 0])
    if endpoint_mode == "endpoint_split":
        endpoint_grid = left_grid[1, 0].subgridspec(2, 1, height_ratios=[0.78, 0.94], hspace=0.42)
        endpoint_axes = {
            "correct": fig.add_subplot(endpoint_grid[0, 0]),
            "failures": fig.add_subplot(endpoint_grid[1, 0]),
        }
    elif endpoint_mode == "failure_decomposition":
        endpoint_axes = {
            "failures": fig.add_subplot(left_grid[1, 0]),
        }
    else:
        raise ValueError(f"Unknown endpoint_mode: {endpoint_mode}")
    map_axes = [fig.add_subplot(right_grid[0, col]) for col in range(len(n_values))]
    trace_axes: list[plt.Axes] = []
    for col in range(len(n_values)):
        trace_axes.append(
            fig.add_subplot(
                right_grid[1, col],
                sharey=trace_axes[0] if trace_axes else None,
            )
        )

    population = draw_empirical_ab_panels(ax_population, endpoint_axes, endpoint_mode=endpoint_mode)
    for col_index, panel in enumerate(map_panels):
        draw_crop_map(map_axes[col_index], panel, extra_colors)

    trace_summaries: list[dict[str, Any]] = []
    for col_index, n in enumerate(n_values):
        trace_summaries.append(draw_trace(trace_axes[col_index], n))
        trace_axes[col_index].set_box_aspect(0.72)
        if col_index == 0:
            trace_axes[col_index].set_ylabel("Share of agents", fontsize=9.4, labelpad=1.0)
        else:
            trace_axes[col_index].tick_params(labelleft=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COUNTRY_COLORS[APPENDIX_CASE["truth"]],
            marker="s",
            linewidth=2.25,
            markersize=6.3,
            markerfacecolor=COUNTRY_COLORS[APPENDIX_CASE["truth"]],
            markeredgecolor=COUNTRY_COLORS[APPENDIX_CASE["truth"]],
            markeredgewidth=1.0,
            label=f"{APPENDIX_CASE['truth']} (truth)",
        ),
        Line2D(
            [0],
            [0],
            color=COUNTRY_COLORS[APPENDIX_CASE["rival"]],
            marker="s",
            linewidth=2.05,
            markersize=6.3,
            markerfacecolor=COUNTRY_COLORS[APPENDIX_CASE["rival"]],
            markeredgecolor=COUNTRY_COLORS[APPENDIX_CASE["rival"]],
            markeredgewidth=1.0,
            label=f"{APPENDIX_CASE['rival']} (rival)",
        ),
        Line2D([0], [0], color=HOUSE_COLORS["gray"], linewidth=1.55, linestyle="--", label="Other countries"),
    ]
    fig.subplots_adjust(left=0.055, right=0.997, top=0.795, bottom=0.165)
    left_panel = ax_population.get_position()
    map_left = map_axes[0].get_position()
    map_right = map_axes[-1].get_position()
    trace_left = trace_axes[0].get_position()
    trace_right = trace_axes[-1].get_position()
    right_center = (map_left.x0 + map_right.x1) / 2.0
    divider_x = left_panel.x1 + 0.018
    top_title_y = min(0.855, map_left.y1 + 0.078)
    bottom_title_y = trace_left.y1 + 0.018
    fig.add_artist(
        Line2D(
            [divider_x, divider_x],
            [0.105, 0.915],
            transform=fig.transFigure,
            color=HOUSE_COLORS["light_gray"],
            linewidth=0.9,
            alpha=0.9,
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(right_center, 0.091),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.45,
        fontsize=7.5,
    )
    fig.text(
        left_panel.x0,
        top_title_y,
        "a) Population-size sweep",
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        left_panel.x0,
        bottom_title_y,
        (
            "b) Endpoint outcomes by N"
            if endpoint_mode == "endpoint_split"
            else "b) Failure decomposition by N"
        ),
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        map_left.x0,
        top_title_y,
        "c) Final predictions over initial crops",
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        trace_left.x0,
        bottom_title_y,
        "d) Country shares over interaction rounds",
        ha="left",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        color=HOUSE_COLORS["black"],
    )
    fig.text(
        (trace_left.x0 + trace_right.x1) / 2.0,
        0.105,
        "Interaction rounds",
        ha="center",
        va="center",
        fontsize=9.4,
        color=HOUSE_COLORS["black"],
    )

    png_path, pdf_path = save_figure(fig, stem)
    summary_path = write_mechanism_summary(
        map_panels,
        trace_summaries,
        stem=stem,
        n_values=n_values,
        description=description,
    )
    payload = json.loads(summary_path.read_text())
    payload["empirical_decomposition_panels"] = {
        "source_stem": "flag_game_empirical_decomposition",
        "included_panels": [
            "a) Population-size sweep",
            (
                "b) Endpoint outcomes by N with failure decomposition"
                if endpoint_mode == "endpoint_split"
                else "b) Failure decomposition by N"
            ),
        ],
        "endpoint_mode": endpoint_mode,
        "panel_scope": {
            "A_B": "aggregate over runs in the GPT-4o pairwise population-size sweep",
            "C_D": f"single representative France/Peru seed family ({APPENDIX_CASE['seed']})",
        },
        "pairwise_rows": population,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    return png_path, pdf_path, summary_path


def draw_appendix_figure() -> tuple[Path, Path, Path]:
    return draw_mechanism_figure(
        n_values=N_VALUES,
        stem=APPENDIX_OUT_STEM,
        figsize=(10.7, 4.05),
        description="representative homogeneous GPT-4o France/Peru run family",
    )


def draw_main_text_mechanism_figure() -> tuple[Path, Path, Path]:
    return draw_mechanism_figure(
        n_values=MAIN_TEXT_MECHANISM_N_VALUES,
        stem=MAIN_TEXT_MECHANISM_OUT_STEM,
        figsize=(8.9, 5.2),
        description="main-text-width France/Peru mechanism variant without N=128",
    )


def draw_empirical_ab_main_text_mechanism_figure(
    *,
    endpoint_mode: str,
) -> tuple[Path, Path, Path]:
    if endpoint_mode == "endpoint_split":
        stem = EMPIRICAL_AB_MECHANISM_SPLIT_OUT_STEM
        figsize = (10.95, 5.42)
        description_suffix = (
            "with a split B panel showing correct consensus above a wrong-consensus/"
            "polarization failure decomposition"
        )
    elif endpoint_mode == "failure_decomposition":
        stem = EMPIRICAL_AB_MECHANISM_FAILURE_OUT_STEM
        figsize = (10.95, 5.15)
        description_suffix = (
            "with a B panel showing only the wrong-consensus/polarization failure decomposition"
        )
    else:
        raise ValueError(f"Unknown endpoint_mode: {endpoint_mode}")
    return draw_empirical_ab_mechanism_figure(
        n_values=EMPIRICAL_AB_MECHANISM_N_VALUES,
        stem=stem,
        figsize=figsize,
        description=(
            "main-text France/Peru mechanism variant with empirical decomposition "
            f"panels A/B and mechanism panels for N=4,16,64; {description_suffix}"
        ),
        endpoint_mode=endpoint_mode,
    )


def main() -> None:
    if os.environ.get("FINAL_CHARTS_DERIVED_ONLY") == "1":
        empirical_ab_failure_png, empirical_ab_failure_pdf, empirical_ab_failure_json = (
            draw_empirical_ab_main_text_mechanism_figure(endpoint_mode="failure_decomposition")
        )
        for path in [
            empirical_ab_failure_png,
            empirical_ab_failure_pdf,
            empirical_ab_failure_json,
        ]:
            print(f"Wrote {path}")
        return

    seed_rows = load_seed_rows()
    summary_rows = summarize_seed_rows(seed_rows)
    main_csv, main_json = write_main_summary(seed_rows, summary_rows)
    main_png, main_pdf = draw_main_figure(summary_rows)
    main_v2_png, main_v2_pdf = draw_main_figure(
        summary_rows,
        include_vote_accuracy=True,
        stem="flag_game_pairwise_n_scaling_main_v2",
    )
    main_v3_seed_rows = filter_main_v3_seed_rows(seed_rows)
    main_v3_summary_rows = summarize_seed_rows(main_v3_seed_rows)
    main_v3_all_seed_count_note = format_seed_count_note(main_v3_summary_rows)
    main_v3_seed_count_note = format_seed_count_note(main_v3_summary_rows, condition="all_gpt_4o")
    main_v3_csv, main_v3_json = write_main_summary(
        main_v3_seed_rows,
        main_v3_summary_rows,
        stem=MAIN_V3_STEM,
        extra_payload={
            "excluded_seeds": MAIN_V3_EXCLUDED_SEEDS,
            "seed_count_note": main_v3_seed_count_note,
            "all_condition_seed_count_note": main_v3_all_seed_count_note,
            "version_note": (
                "Filtered v3 sensitivity chart regenerated from all complete seed folders; "
                "raw run folders are untouched."
            ),
        },
    )
    main_v3_png, main_v3_pdf = draw_main_figure(
        main_v3_summary_rows,
        stem=MAIN_V3_STEM,
    )
    model_comparison_png, model_comparison_pdf = draw_model_comparison_figure(summary_rows)
    model_comparison_v2_png, model_comparison_v2_pdf = draw_model_comparison_figure(
        summary_rows,
        include_vote_accuracy=True,
        stem=f"{MODEL_COMPARISON_OUT_STEM}_v2",
    )
    appendix_png, appendix_pdf, appendix_json = draw_appendix_figure()
    mechanism_png, mechanism_pdf, mechanism_json = draw_main_text_mechanism_figure()
    empirical_ab_split_png, empirical_ab_split_pdf, empirical_ab_split_json = (
        draw_empirical_ab_main_text_mechanism_figure(endpoint_mode="endpoint_split")
    )
    empirical_ab_failure_png, empirical_ab_failure_pdf, empirical_ab_failure_json = (
        draw_empirical_ab_main_text_mechanism_figure(endpoint_mode="failure_decomposition")
    )
    for path in [
        main_png,
        main_pdf,
        main_v2_png,
        main_v2_pdf,
        main_v3_png,
        main_v3_pdf,
        model_comparison_png,
        model_comparison_pdf,
        model_comparison_v2_png,
        model_comparison_v2_pdf,
        main_csv,
        main_json,
        main_v3_csv,
        main_v3_json,
        appendix_png,
        appendix_pdf,
        appendix_json,
        mechanism_png,
        mechanism_pdf,
        mechanism_json,
        empirical_ab_split_png,
        empirical_ab_split_pdf,
        empirical_ab_split_json,
        empirical_ab_failure_png,
        empirical_ab_failure_pdf,
        empirical_ab_failure_json,
    ]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Make paper-facing pairwise Flag Game alpha-sweep figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from plot_style import HOUSE_COLORS, setup_paper_house_style, style_axis


ROOT = Path(__file__).resolve().parent.parent
LIVE_ROOT = (
    ROOT
    / "results"
    / "flag_game"
    / "proposal_core_remaining_seeds7_w6_h4_scale25_noexamples_stop5"
)
POSTER_ALPHA_SOURCE_ROOT = LIVE_ROOT / "03_phase_diagram"
CORRECTED_ALPHA_SOURCE_ROOT = LIVE_ROOT / "03_phase_diagram_corrected"
POSTER_ALPHA_REFERENCE_FIGURE = (
    ROOT / "poster" / "exports" / "support_visuals" / "flag_game_scaling_by_alpha.png"
)
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"
OUT_STEM = "flag_game_pairwise_alpha_main_candidate"

MODELS = ["gpt4o"]
MODEL_LABELS = {
    "gpt4o": "GPT-4o",
}
CIQ_BLUE = "#1764B5"
VOTE_GREEN = "#00A36F"
MODEL_STYLES = {
    "gpt4o": {"color": CIQ_BLUE, "marker": "o"},
}
PAPER_SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]
OVERLAPS = [0.0, 0.3, 0.6, 0.9]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
METRICS = {
    "initial_accuracy": "initial_mean",
    "final_accuracy": "final_mean",
    "collaboration_gain_over_initial_accuracy": "gain_mean",
}
SOURCE_NOTE = (
    "This redraws the same poster alpha sweep used for flag_game_scaling_by_alpha. "
    "The old sampler's nominal rho labels collapsed, so this should be read as "
    "a prompted social evidence uptake trend at N=16 rather than rho evidence."
)
CORRECTED_NOTE = (
    "The corrected phase sweep is stored alongside the primary data for sensitivity. "
    "It preserves the qualitative GPT-4o improvement away from alpha=0, but the exact "
    "prompted social evidence uptake optimum is less sharp."
)


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


def alpha_label(alpha: float) -> str:
    return f"{alpha:.2f}".rstrip("0").rstrip(".")


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def read_seed_summaries(root: Path, *, model: str, alpha: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for overlap in OVERLAPS:
        condition_dir = root / model / f"overlap_{overlap}" / f"a_{alpha}"
        for summary_path in sorted(condition_dir.glob("seed_*/summary.json")):
            summary = json.loads(summary_path.read_text())
            final_vote = final_vote_from_summary(summary_path, summary)
            rows.append(
                {
                    "model": model,
                    "alpha": alpha,
                    "overlap": overlap,
                    "seed": summary_path.parent.name,
                    "final_vote_country": final_vote["country"],
                    "final_vote_correct": final_vote["correct"],
                    "final_vote_accuracy": final_vote["accuracy"],
                    **summary,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No seed summaries found for {model=} {alpha=} under {root}")
    return rows


def read_last_csv_row(path: Path) -> dict[str, str]:
    last: dict[str, str] | None = None
    with path.open(newline="") as handle:
        for last in csv.DictReader(handle):
            pass
    if last is None:
        raise ValueError(f"No rows in {path}")
    return last


def unique_top_country(country_shares: list[tuple[str, float]]) -> str | None:
    if not country_shares:
        return None
    top_share = max(share for _, share in country_shares)
    winners = [country for country, share in country_shares if share == top_share]
    if len(winners) != 1 or top_share <= 0.0:
        return None
    return winners[0]


def final_vote_from_summary(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    if "final_vote_accuracy" in summary:
        country = summary.get("final_vote_country")
        accuracy = float(summary["final_vote_accuracy"])
        return {
            "country": country if isinstance(country, str) else None,
            "correct": bool(summary.get("final_vote_correct", accuracy > 0.0)),
            "accuracy": accuracy,
        }

    final_probe = read_last_csv_row(summary_path.parent / "per_round.csv")
    country_shares = [
        (str(country), float(final_probe[str(country)]))
        for country in summary["countries"]
        if final_probe.get(str(country), "") != ""
    ]
    top_country = unique_top_country(country_shares)
    correct = top_country == str(summary["truth_country"])
    return {
        "country": top_country,
        "correct": correct,
        "accuracy": 1.0 if correct else 0.0,
    }


def bootstrap_interval(values: list[float], *, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) <= 1:
        value = values[0] if values else float("nan")
        return value, value
    samples = np.asarray(values, dtype=float)
    draws = rng.choice(samples, size=(8000, len(samples)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_source(root: Path, *, source_name: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260429)
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for alpha in ALPHAS:
            seed_rows = read_seed_summaries(root, model=model, alpha=alpha)
            row: dict[str, Any] = {
                "source": source_name,
                "model": model,
                "model_label": MODEL_LABELS[model],
                "alpha": alpha,
                "n_trials": len(seed_rows),
                "n_cells": len({(seed_row["overlap"], seed_row["alpha"]) for seed_row in seed_rows}),
                "overlaps": sorted({float(seed_row["overlap"]) for seed_row in seed_rows}),
            }
            for source_key, out_key in METRICS.items():
                values = [float(seed_row[source_key]) for seed_row in seed_rows]
                lo, hi = bootstrap_interval(values, rng=rng)
                row[out_key] = float(np.mean(values))
                row[f"{out_key}_ci_low"] = lo
                row[f"{out_key}_ci_high"] = hi
            vote_values = [float(seed_row["final_vote_accuracy"]) for seed_row in seed_rows]
            vote_lo, vote_hi = bootstrap_interval(vote_values, rng=rng)
            row["final_vote_accuracy_rate"] = float(np.mean(vote_values))
            row["final_vote_accuracy_rate_ci_low"] = vote_lo
            row["final_vote_accuracy_rate_ci_high"] = vote_hi
            row["success85_rate"] = float(
                np.mean([float(seed_row["final_accuracy"]) >= 0.85 for seed_row in seed_rows])
            )
            summary_rows.append(row)
    return summary_rows


def row_for(rows: list[dict[str, Any]], *, model: str, alpha: float) -> dict[str, Any]:
    for row in rows:
        if row["model"] == model and float(row["alpha"]) == alpha:
            return row
    raise KeyError(f"No row for {model=} {alpha=}")


def errorbar_metric(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    *,
    label_suffix: str,
    linestyle: str = "-",
    alpha: float = 1.0,
    color: str | None = None,
    marker: str | None = None,
) -> None:
    for model in MODELS:
        style = MODEL_STYLES[model]
        line_color = color or style["color"]
        y_values = np.array([row_for(rows, model=model, alpha=alpha)[metric] for alpha in ALPHAS])
        lo_values = np.array(
            [row_for(rows, model=model, alpha=alpha)[f"{metric}_ci_low"] for alpha in ALPHAS]
        )
        hi_values = np.array(
            [row_for(rows, model=model, alpha=alpha)[f"{metric}_ci_high"] for alpha in ALPHAS]
        )
        ax.errorbar(
            ALPHAS,
            y_values,
            yerr=np.vstack([y_values - lo_values, hi_values - y_values]),
            color=line_color,
            marker=marker or style["marker"],
            markerfacecolor="white",
            markeredgecolor=line_color,
            markeredgewidth=1.35,
            linewidth=1.65,
            linestyle=linestyle,
            elinewidth=0.9,
            capsize=2.4,
            label=label_suffix,
            alpha=alpha,
            zorder=3,
        )


def draw_alpha_figure(
    rows: list[dict[str, Any]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = OUT_STEM,
) -> tuple[Path, Path, dict[str, Any]]:
    setup_paper_house_style()
    apply_paper_sans_rc()
    plt.rcParams.update(
        {
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.65), constrained_layout=False)

    errorbar_metric(ax, rows, "final_mean", label_suffix="CIQ")
    if include_vote_accuracy:
        errorbar_metric(
            ax,
            rows,
            "final_vote_accuracy_rate",
            label_suffix="Final vote accuracy",
            linestyle="--",
            alpha=0.78,
            color=VOTE_GREEN,
            marker="^",
        )
    ax.set_title(
        "Pairwise GPT-4o: social evidence uptake",
        loc="left",
        fontweight="normal",
        color=HOUSE_COLORS["black"],
        pad=4,
    )
    ax.set_ylabel("Mean CIQ / accuracy rate" if include_vote_accuracy else "Mean CIQ")
    ax.set_ylim(0.35, 1.02 if include_vote_accuracy else 0.88)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks(ALPHAS)
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel("Prompted social evidence uptake alpha")
    style_axis(ax, grid=True)
    ax.grid(True, which="major", alpha=0.12, linewidth=0.5)
    ax.tick_params(colors=HOUSE_COLORS["black"])
    ax.xaxis.label.set_color(HOUSE_COLORS["black"])
    ax.yaxis.label.set_color(HOUSE_COLORS["black"])
    if include_vote_accuracy:
        ax.legend(
            frameon=False,
            loc="lower right",
            fontsize=7.8,
            handlelength=2.0,
            labelcolor=HOUSE_COLORS["black"],
        )

    fig.subplots_adjust(left=0.16, right=0.985, top=0.86, bottom=0.20)

    png_path, pdf_path = save_figure(fig, stem)
    summary = {
        "source_root": str(POSTER_ALPHA_SOURCE_ROOT),
        "poster_reference_figure": str(POSTER_ALPHA_REFERENCE_FIGURE),
        "outputs": [str(png_path), str(pdf_path)],
        "source_note": SOURCE_NOTE,
        "final_vote_accuracy_definition": (
            "For pairwise runs, final_vote_accuracy is 1 when the unique top-voted final "
            "country is the truth country; ties receive 0."
        ),
        "data": rows,
    }
    return png_path, pdf_path, summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "model",
        "model_label",
        "alpha",
        "n_trials",
        "n_cells",
        "initial_mean",
        "initial_mean_ci_low",
        "initial_mean_ci_high",
        "final_mean",
        "final_mean_ci_low",
        "final_mean_ci_high",
        "final_vote_accuracy_rate",
        "final_vote_accuracy_rate_ci_low",
        "final_vote_accuracy_rate_ci_high",
        "gain_mean",
        "gain_mean_ci_low",
        "gain_mean_ci_high",
        "success85_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_fields = {
        "alpha",
        "n_trials",
        "n_cells",
        "initial_mean",
        "initial_mean_ci_low",
        "initial_mean_ci_high",
        "final_mean",
        "final_mean_ci_low",
        "final_mean_ci_high",
        "final_vote_accuracy_rate",
        "final_vote_accuracy_rate_ci_low",
        "final_vote_accuracy_rate_ci_high",
        "gain_mean",
        "gain_mean_ci_low",
        "gain_mean_ci_high",
        "success85_rate",
    }
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for field in numeric_fields:
                if str(parsed.get(field, "")).strip() != "":
                    parsed[field] = float(parsed[field])
            rows.append(parsed)
    return rows


def main() -> None:
    csv_path = DATA_DIR / f"{OUT_STEM}_summary.csv"
    if os.environ.get("FINAL_CHARTS_DERIVED_ONLY") == "1":
        all_rows = read_summary_csv(csv_path)
        poster_rows = [row for row in all_rows if row.get("source") == "poster_alpha_sweep"]
        corrected_rows = [row for row in all_rows if row.get("source") == "corrected_phase_sweep"]
    else:
        poster_rows = summarize_source(POSTER_ALPHA_SOURCE_ROOT, source_name="poster_alpha_sweep")
        corrected_rows = summarize_source(CORRECTED_ALPHA_SOURCE_ROOT, source_name="corrected_phase_sweep")
    png_path, pdf_path, figure_summary = draw_alpha_figure(poster_rows)
    v2_png_path, v2_pdf_path, v2_figure_summary = draw_alpha_figure(
        poster_rows,
        include_vote_accuracy=True,
        stem=f"{OUT_STEM}_v2",
    )

    summary_path = DATA_DIR / f"{OUT_STEM}_summary.json"
    v2_summary_path = DATA_DIR / f"{OUT_STEM}_v2_summary.json"
    write_json(
        summary_path,
        {
            **figure_summary,
            "corrected_source_root": str(CORRECTED_ALPHA_SOURCE_ROOT),
            "corrected_note": CORRECTED_NOTE,
            "corrected_sensitivity_data": corrected_rows,
        },
    )
    write_json(
        v2_summary_path,
        {
            **v2_figure_summary,
            "corrected_source_root": str(CORRECTED_ALPHA_SOURCE_ROOT),
            "corrected_note": CORRECTED_NOTE,
            "corrected_sensitivity_data": corrected_rows,
        },
    )
    write_csv(csv_path, poster_rows + corrected_rows)

    print(f"saved {png_path}")
    print(f"saved {pdf_path}")
    print(f"saved {summary_path}")
    print(f"saved {v2_png_path}")
    print(f"saved {v2_pdf_path}")
    print(f"saved {v2_summary_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()

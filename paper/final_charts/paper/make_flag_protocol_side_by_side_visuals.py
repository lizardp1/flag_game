#!/usr/bin/env python3
"""Make paper-facing N=8 protocol side-by-side Flag Game figure."""

from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D

from plot_style import HOUSE_COLORS, setup_paper_house_style, style_axis


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = (
    ROOT
    / "results"
    / "flag_game_side_by_side"
    / "protocol_side_by_side_N8_m3_full_10seed"
)
SOURCE_RANKED_CSV = SOURCE_DIR / "side_by_side_ranked_performance.csv"
SOURCE_SEED_CSV = SOURCE_DIR / "side_by_side_seed_metrics.csv"
SOURCE_FULL_DIR = SOURCE_DIR / "full"
FIGURE_DIR = ROOT / "paper" / "exports" / "figures"
DATA_DIR = ROOT / "paper" / "exports" / "data"
OUT_STEM = "flag_game_protocol_side_by_side_N8"

PROTOCOL_COLORS = {
    "pairwise": HOUSE_COLORS["orange"],
    "broadcast": HOUSE_COLORS["purple"],
    "org": HOUSE_COLORS["teal"],
}
PROTOCOL_LABELS = {
    "pairwise": "Pairwise",
    "broadcast": "Broadcast",
    "org": "Manager",
}
FINAL_CONDITION_SPECS = [
    ("broadcast", "all_gpt_4o", "All 4o", "4o agents"),
    ("broadcast", "all_gpt_5_4", "All 5.4", "5.4 agents"),
    ("broadcast", "balanced_4_gpt_5_4_4_gpt_4o", "Mixed", "4 5.4 + 4 4o agents"),
    ("pairwise", "all_gpt_4o", "All 4o", "4o agents"),
    ("pairwise", "all_gpt_5_4", "All 5.4", "5.4 agents"),
    ("pairwise", "balanced_4_gpt_5_4_4_gpt_4o", "Mixed", "4 5.4 + 4 4o agents"),
    ("org", "all_gpt_4o", "4o mgr + 4o obs", "4o agents"),
    ("org", "all_gpt_5_4", "5.4 mgr + 5.4 obs", "5.4 agents"),
    ("org", "manager_gpt_4o_observers_gpt_5_4", "4o mgr + 5.4 obs", "5.4 agents"),
    ("org", "manager_gpt_5_4_observers_gpt_4o", "5.4 mgr + 4o obs", "4o agents"),
    (
        "org",
        "manager_gpt_4o_observers_4_gpt_5_4_4_gpt_4o",
        "4o mgr + mixed obs",
        "4 5.4 + 4 4o agents",
    ),
    (
        "org",
        "manager_gpt_5_4_observers_4_gpt_5_4_4_gpt_4o",
        "5.4 mgr + mixed obs",
        "4 5.4 + 4 4o agents",
    ),
]
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


def read_ranked_rows(path: Path, *, seed_path: Path = SOURCE_SEED_CSV) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ranked-performance CSV: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No ranked-performance rows found in {path}")

    numeric_fields = (
        "n_trials",
        "initial_accuracy_mean",
        "initial_accuracy_sem",
        "final_accuracy_mean",
        "final_accuracy_sem",
        "social_lift_mean",
        "social_lift_sem",
    )
    for row in rows:
        for field in numeric_fields:
            value = row.get(field, "")
            row[field] = int(value) if field == "n_trials" else float(value)
        row["color"] = PROTOCOL_COLORS[str(row["protocol"])]
        row["protocol_label"] = PROTOCOL_LABELS[str(row["protocol"])]
    has_final_vote_stats = all(
        str(row.get("final_vote_accuracy_rate", "")).strip() != ""
        and str(row.get("final_vote_accuracy_sem", "")).strip() != ""
        for row in rows
    )
    if has_final_vote_stats:
        for row in rows:
            row["final_vote_accuracy_rate"] = float(row["final_vote_accuracy_rate"])
            row["final_vote_accuracy_sem"] = float(row["final_vote_accuracy_sem"])
    else:
        accuracy_stats = final_vote_accuracy_stats(read_seed_rows(seed_path))
        for row in rows:
            stats = accuracy_stats.get((str(row["protocol"]), str(row["condition_name"])), {})
            row["final_vote_accuracy_rate"] = float(stats.get("mean", row["final_accuracy_mean"]))
            row["final_vote_accuracy_sem"] = float(stats.get("sem", 0.0))
    return sorted(rows, key=lambda row: (row["final_accuracy_mean"], row["social_lift_mean"]))


def read_seed_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed-metrics CSV: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["final_vote_accuracy"] = final_vote_accuracy_for_seed(row)
    return rows


def sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def read_ranked_rows_from_seed_folders(full_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol, condition_name, display_label, observer_group in FINAL_CONDITION_SPECS:
        condition_dir = full_dir / protocol / condition_name
        seed_dirs = sorted(path for path in condition_dir.glob("seed_*") if path.is_dir())
        summaries: list[dict[str, Any]] = []
        for seed_dir in seed_dirs:
            summary_path = seed_dir / "summary.json"
            if not summary_path.exists():
                continue
            summaries.append(json.loads(summary_path.read_text()))
        if not summaries:
            raise FileNotFoundError(f"No summary.json files found under {condition_dir}")

        initial_values = [float(row["initial_accuracy"]) for row in summaries]
        final_values = [float(row["final_accuracy"]) for row in summaries]
        lift_values = [float(row["collaboration_gain_over_initial_accuracy"]) for row in summaries]
        rows.append(
            {
                "display_label": display_label,
                "protocol": protocol,
                "protocol_label": PROTOCOL_LABELS[protocol],
                "condition_name": condition_name,
                "observer_group": observer_group,
                "color": PROTOCOL_COLORS[protocol],
                "n_trials": len(summaries),
                "initial_accuracy_mean": float(np.mean(initial_values)),
                "initial_accuracy_sem": sem(initial_values),
                "final_accuracy_mean": float(np.mean(final_values)),
                "final_accuracy_sem": sem(final_values),
                "social_lift_mean": float(np.mean(lift_values)),
                "social_lift_sem": sem(lift_values),
                "final_vote_accuracy_rate": float(np.mean(final_values)),
                "final_vote_accuracy_sem": sem(final_values),
            }
        )
    return sorted(rows, key=lambda row: (row["final_accuracy_mean"], row["social_lift_mean"]))


def final_vote_accuracy_for_seed(row: dict[str, Any]) -> float:
    protocol = str(row["protocol"])
    if protocol == "org":
        return float(row["final_accuracy"])

    trial_dir = Path(str(row["trial_dir"]))
    summary_path = trial_dir / "summary.json"
    per_round_path = trial_dir / "per_round.csv"
    if not summary_path.exists() or not per_round_path.exists():
        return float("nan")

    summary = json.loads(summary_path.read_text())
    if "final_vote_accuracy" in summary:
        return float(summary["final_vote_accuracy"])

    final_row = read_last_csv_row(per_round_path)
    country_shares = []
    for country in summary["countries"]:
        country_text = str(country)
        value = final_row.get(f"final_share_{country_text}", final_row.get(country_text))
        if value not in (None, ""):
            country_shares.append((country_text, float(value)))
    top_country = unique_top_country(country_shares)
    return 1.0 if top_country == str(summary["truth_country"]) else 0.0


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


def final_vote_accuracy_stats(seed_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in seed_rows:
        value = float(row["final_vote_accuracy"])
        if np.isnan(value):
            continue
        grouped.setdefault((str(row["protocol"]), str(row["condition_name"])), []).append(value)
    stats: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        stats[key] = {"mean": float(arr.mean()), "sem": sem}
    return stats


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "display_label",
        "protocol",
        "protocol_label",
        "condition_name",
        "observer_group",
        "n_trials",
        "initial_accuracy_mean",
        "initial_accuracy_sem",
        "final_accuracy_mean",
        "final_accuracy_sem",
        "final_vote_accuracy_rate",
        "final_vote_accuracy_sem",
        "social_lift_mean",
        "social_lift_sem",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def draw_figure(
    rows: list[dict[str, Any]],
    *,
    include_vote_accuracy: bool = False,
    stem: str = OUT_STEM,
) -> tuple[Path, Path]:
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

    y = np.arange(len(rows))
    colors = [row["color"] for row in rows]
    labels = [str(row["display_label"]) for row in rows]
    n_trials_values = sorted({int(row["n_trials"]) for row in rows})
    if len(n_trials_values) == 1:
        trial_count_text = f"{n_trials_values[0]} same-seed trials per row"
    else:
        trial_count_text = f"{n_trials_values[0]}-{n_trials_values[-1]} same-seed trials per row"

    final_mean = np.array([row["final_accuracy_mean"] for row in rows], dtype=float)
    final_sem = np.array([row["final_accuracy_sem"] for row in rows], dtype=float)
    vote_accuracy = np.array([row["final_vote_accuracy_rate"] for row in rows], dtype=float)
    vote_accuracy_sem = np.array([row["final_vote_accuracy_sem"] for row in rows], dtype=float)
    initial_mean = np.array([row["initial_accuracy_mean"] for row in rows], dtype=float)
    lift_mean = np.array([row["social_lift_mean"] for row in rows], dtype=float)
    lift_sem = np.array([row["social_lift_sem"] for row in rows], dtype=float)

    fig, (ax_ciq, ax_lift) = plt.subplots(
        1,
        2,
        figsize=(7.15, 4.45),
        sharey=True,
        gridspec_kw={"width_ratios": [1.38, 0.95], "wspace": 0.10},
    )

    ax_ciq.barh(
        y,
        final_mean,
        xerr=final_sem,
        color=colors,
        alpha=0.78,
        edgecolor="none",
        error_kw={"ecolor": HOUSE_COLORS["black"], "elinewidth": 1.0, "capthick": 1.0},
        capsize=2.4,
        zorder=2,
    )
    ax_ciq.vlines(
        initial_mean,
        y - 0.35,
        y + 0.35,
        color=HOUSE_COLORS["black"],
        linewidth=1.25,
        zorder=4,
    )
    if include_vote_accuracy:
        ax_ciq.errorbar(
            vote_accuracy,
            y,
            xerr=vote_accuracy_sem,
            fmt="D",
            color=HOUSE_COLORS["black"],
            markerfacecolor="white",
            markeredgewidth=1.05,
            markersize=4.0,
            capsize=2.2,
            elinewidth=0.85,
            zorder=5,
        )
    for idx, value in enumerate(final_mean):
        label_x = min(value + final_sem[idx] + 0.018, 1.075)
        ax_ciq.text(
            label_x,
            idx,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=HOUSE_COLORS["black"],
        )
    ax_ciq.set_xlim(0.0, 1.11)
    ax_ciq.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax_ciq.set_xlabel("Collective accuracy (CIQ)")
    ax_ciq.set_yticks(y)
    ax_ciq.set_yticklabels(labels)
    ax_ciq.set_title(
        "A. Final collective accuracy",
        loc="left",
        fontweight="normal",
        color=HOUSE_COLORS["black"],
        pad=4,
    )
    style_axis(ax_ciq, grid=True)
    ax_ciq.grid(axis="x", alpha=0.16, linewidth=0.55)
    ax_ciq.tick_params(colors=HOUSE_COLORS["black"])
    ax_ciq.xaxis.label.set_color(HOUSE_COLORS["black"])

    ax_lift.barh(
        y,
        lift_mean,
        xerr=lift_sem,
        color=colors,
        alpha=0.54,
        edgecolor="none",
        error_kw={"ecolor": HOUSE_COLORS["black"], "elinewidth": 1.0, "capthick": 1.0},
        capsize=2.4,
        zorder=2,
    )
    ax_lift.axvline(0.0, color=HOUSE_COLORS["gray"], linewidth=0.9)
    for idx, value in enumerate(lift_mean):
        label_x = min(value + lift_sem[idx] + 0.014, 0.655)
        ax_lift.text(
            label_x,
            idx,
            f"{value:+.2f}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=HOUSE_COLORS["black"],
        )
    ax_lift.set_xlim(-0.02, 0.68)
    ax_lift.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax_lift.set_xlabel("Social uplift")
    ax_lift.set_title(
        "B. Gain over private baseline",
        loc="left",
        fontweight="normal",
        color=HOUSE_COLORS["black"],
        pad=4,
    )
    style_axis(ax_lift, grid=True)
    ax_lift.grid(axis="x", alpha=0.16, linewidth=0.55)
    ax_lift.tick_params(colors=HOUSE_COLORS["black"])
    ax_lift.xaxis.label.set_color(HOUSE_COLORS["black"])

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", color=PROTOCOL_COLORS["pairwise"], markersize=5.0),
        Line2D([0], [0], marker="s", linestyle="", color=PROTOCOL_COLORS["broadcast"], markersize=5.0),
        Line2D([0], [0], marker="s", linestyle="", color=PROTOCOL_COLORS["org"], markersize=5.0),
        Line2D([0], [0], color=HOUSE_COLORS["black"], linewidth=1.25),
    ]
    legend_labels = ["Pairwise", "Broadcast", "Manager", "IIQ"]
    if include_vote_accuracy:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                color=HOUSE_COLORS["black"],
                markerfacecolor="white",
                markersize=4.5,
            )
        )
        legend_labels.append("Final vote acc.")
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=5 if include_vote_accuracy else 4,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.070),
        columnspacing=1.05 if include_vote_accuracy else 1.35,
        handlelength=1.8,
        fontsize=7.8,
        labelcolor=HOUSE_COLORS["black"],
    )
    fig.text(
        0.50,
        0.030,
        (
            f"N=8, m=3; {trial_count_text}. Bars show CIQ means with SEM; ticks mark IIQ; diamonds mark final vote accuracy."
            if include_vote_accuracy
            else f"N=8, m=3; {trial_count_text}. Bars show means with SEM; dark ticks in Panel A mark mean private accuracy."
        ),
        ha="center",
        va="center",
        fontsize=6.9,
        color=HOUSE_COLORS["gray"],
    )
    fig.subplots_adjust(left=0.215, right=0.985, top=0.925, bottom=0.215)

    return save_figure(fig, stem)


def draw_final_accuracy_figure(
    rows: list[dict[str, Any]],
    *,
    stem: str,
    x_max: float = 0.8,
) -> tuple[Path, Path]:
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

    y = np.arange(len(rows))
    colors = [row["color"] for row in rows]
    labels = [str(row["display_label"]) for row in rows]
    final_mean = np.array([row["final_accuracy_mean"] for row in rows], dtype=float)
    final_sem = np.array([row["final_accuracy_sem"] for row in rows], dtype=float)
    initial_mean = np.array([row["initial_accuracy_mean"] for row in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(4.95, 4.45))
    ax.barh(
        y,
        final_mean,
        xerr=final_sem,
        color=colors,
        alpha=0.78,
        edgecolor="none",
        error_kw={"ecolor": HOUSE_COLORS["black"], "elinewidth": 1.0, "capthick": 1.0},
        capsize=2.4,
        zorder=2,
    )
    ax.vlines(
        initial_mean,
        y - 0.35,
        y + 0.35,
        color=HOUSE_COLORS["black"],
        linewidth=1.25,
        zorder=4,
    )
    for idx, value in enumerate(final_mean):
        label_x = min(value + final_sem[idx] + 0.018, x_max - 0.014)
        ax.text(
            label_x,
            idx,
            f"{value:.2f}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=HOUSE_COLORS["black"],
        )

    ax.set_xlim(0.0, x_max)
    ax.set_xticks(np.arange(0.0, x_max + 0.001, 0.2))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(
        "Collective mean accuracy",
        loc="left",
        fontweight="normal",
        color=HOUSE_COLORS["black"],
        pad=4,
    )
    style_axis(ax, grid=True)
    ax.grid(axis="x", alpha=0.16, linewidth=0.55)
    ax.tick_params(colors=HOUSE_COLORS["black"])
    ax.xaxis.label.set_color(HOUSE_COLORS["black"])

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="", color=PROTOCOL_COLORS["pairwise"], markersize=5.0),
        Line2D([0], [0], marker="s", linestyle="", color=PROTOCOL_COLORS["broadcast"], markersize=5.0),
        Line2D([0], [0], marker="s", linestyle="", color=PROTOCOL_COLORS["org"], markersize=5.0),
        Line2D([0], [0], color=HOUSE_COLORS["black"], linewidth=1.25),
    ]
    legend_labels = ["Pairwise", "Broadcast", "Manager", "Initial mean accuracy"]
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.040),
        columnspacing=1.12,
        handlelength=1.8,
        fontsize=7.6,
        labelcolor=HOUSE_COLORS["black"],
    )
    fig.subplots_adjust(left=0.330, right=0.985, top=0.925, bottom=0.165)
    return save_figure(fig, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make paper-facing N=8 protocol side-by-side Flag Game figure.")
    parser.add_argument("--source-ranked-csv", type=Path, default=None)
    parser.add_argument("--source-seed-csv", type=Path, default=SOURCE_SEED_CSV)
    parser.add_argument("--source-full-dir", type=Path, default=SOURCE_FULL_DIR)
    parser.add_argument("--out-stem", default=OUT_STEM)
    parser.add_argument("--final-only", action="store_true", help="Write only a single collective-accuracy panel.")
    parser.add_argument("--x-max", type=float, default=0.8, help="Maximum x-axis value for --final-only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_ranked_csv is None:
        rows = read_ranked_rows_from_seed_folders(args.source_full_dir)
    else:
        rows = read_ranked_rows(args.source_ranked_csv, seed_path=args.source_seed_csv)
    if args.final_only:
        png_path, pdf_path = draw_final_accuracy_figure(rows, stem=args.out_stem, x_max=args.x_max)
        output_paths = [str(png_path), str(pdf_path)]
    else:
        png_path, pdf_path = draw_figure(rows, stem=args.out_stem)
        v2_png_path, v2_pdf_path = draw_figure(
            rows,
            include_vote_accuracy=True,
            stem=f"{args.out_stem}_v2",
        )
        output_paths = [str(png_path), str(pdf_path), str(v2_png_path), str(v2_pdf_path)]

    export_csv = DATA_DIR / f"{args.out_stem}_ranked_performance.csv"
    export_json = DATA_DIR / f"{args.out_stem}_summary.json"
    write_csv(export_csv, list(reversed(rows)))
    write_json(
        export_json,
        {
            "source_ranked_csv": str(args.source_ranked_csv) if args.source_ranked_csv is not None else None,
            "source_seed_metrics_csv": str(args.source_seed_csv) if args.source_ranked_csv is not None else None,
            "source_full_dir": str(args.source_full_dir) if args.source_ranked_csv is None else None,
            "outputs": [*output_paths, str(export_csv)],
            "notes": [
                "Rows in the figure are sorted from lowest to highest mean CIQ so the best-performing condition appears at the top.",
                "CIQ is final_accuracy; IIQ is initial_accuracy; social uplift is final_accuracy - initial_accuracy.",
                "Final vote accuracy is computed from the unique top-voted final country for pairwise and broadcast runs; manager rows use the manager's final correctness as-is.",
                "Pairwise, broadcast, and manager runs are compared on the matched N=8, m=3, full flag set using the original broadcast data.",
            ],
            "final_only": bool(args.final_only),
            "rows_by_descending_ciq": list(reversed(rows)),
        },
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    if not args.final_only:
        print(f"Wrote {v2_png_path}")
        print(f"Wrote {v2_pdf_path}")
    print(f"Wrote {export_csv}")
    print(f"Wrote {export_json}")


if __name__ == "__main__":
    main()

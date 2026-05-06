from __future__ import annotations

from pathlib import Path
import os
import tempfile

_plot_cache_dir = Path(tempfile.gettempdir()) / "nnd_matplotlib_cache"
_plot_cache_dir.mkdir(parents=True, exist_ok=True)
(_plot_cache_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache_dir / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_plot_cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


_PER_ROUND_METADATA_COLUMNS = {
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
    "top_vote_country",
    "top_vote_correct",
}


def plot_country_share_trajectories(per_round_df: pd.DataFrame, out_dir: Path) -> None:
    if per_round_df.empty:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    countries = [
        column
        for column in per_round_df.columns
        if column not in _PER_ROUND_METADATA_COLUMNS
    ]
    truth_country = str(per_round_df.iloc[0]["truth_country"])
    max_share_by_country = {
        country: float(per_round_df[country].max())
        for country in countries
    }
    ranked_countries = sorted(
        countries,
        key=lambda country: (-max_share_by_country[country], country),
    )
    tracked = ranked_countries[: min(6, len(ranked_countries))]
    if truth_country not in tracked:
        tracked.append(truth_country)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for country in tracked:
        linewidth = 2.5 if country == truth_country else 1.8
        alpha = 1.0 if country == truth_country else 0.9
        ax.plot(
            per_round_df["t"],
            per_round_df[country],
            marker="o",
            linewidth=linewidth,
            alpha=alpha,
            label=country,
        )
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Probe share")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Country Share Trajectories (truth: {truth_country})")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "country_share_trajectories.png", dpi=150)
    plt.close(fig)


def plot_run_overview(per_round_df: pd.DataFrame, out_dir: Path) -> None:
    if per_round_df.empty:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    countries = [
        column
        for column in per_round_df.columns
        if column not in _PER_ROUND_METADATA_COLUMNS
    ]
    truth_country = str(per_round_df.iloc[0]["truth_country"])
    initial_row = per_round_df.iloc[0]
    final_row = per_round_df.iloc[-1]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(per_round_df["t"], per_round_df["truth_mass"], marker="o")
    axes[0].set_ylabel("Truth Mass")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title(f"Truth Country: {truth_country}")

    axes[1].plot(per_round_df["t"], per_round_df["top1_share"], marker="o", label="Top-1 share")
    axes[1].plot(per_round_df["t"], per_round_df["top2_share"], marker="o", label="Top-2 share")
    axes[1].set_ylabel("Support")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="best")

    axes[2].plot(per_round_df["t"], per_round_df["entropy"], marker="o", label="Entropy")
    axes[2].plot(per_round_df["t"], per_round_df["U"], marker="o", label="U")
    axes[2].set_xlabel("Interaction step")
    axes[2].set_ylabel("Diversity")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(plots_dir / "run_overview.png", dpi=150)
    plt.close(fig)

    x = np.arange(len(countries))
    initial_values = np.array([float(initial_row[country]) for country in countries], dtype=float)
    final_values = np.array([float(final_row[country]) for country in countries], dtype=float)
    truth_index = countries.index(truth_country) if truth_country in countries else None

    fig, ax = plt.subplots(figsize=(max(10, len(countries) * 0.6), 5.5))
    width = 0.38
    initial_bars = ax.bar(x - width / 2, initial_values, width, label="Initial probe")
    final_bars = ax.bar(x + width / 2, final_values, width, label="Final probe")
    if truth_index is not None:
        initial_bars[truth_index].set_edgecolor("black")
        initial_bars[truth_index].set_linewidth(2.0)
        final_bars[truth_index].set_edgecolor("black")
        final_bars[truth_index].set_linewidth(2.0)
    ax.set_ylabel("Probe Share")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha="right")
    ax.set_title(f"Initial vs Final Country Distribution (truth: {truth_country})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "initial_vs_final_distribution.png", dpi=150)
    plt.close(fig)

    plot_country_share_trajectories(per_round_df, out_dir)


def plot_sweep_summary(sweep_df: pd.DataFrame, out_dir: Path) -> None:
    if sweep_df.empty:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    m_values = sorted(int(value) for value in sweep_df["interaction_m"].unique())
    has_tile_columns = {"tile_width", "tile_height"}.issubset(set(sweep_df.columns))

    def _series_label(row: pd.Series) -> str:
        if has_tile_columns:
            return f"m={int(row['interaction_m'])}, {int(row['tile_width'])}x{int(row['tile_height'])}"
        return f"m={int(row['interaction_m'])}"

    if has_tile_columns:
        group_cols = ["interaction_m", "tile_width", "tile_height"]
    else:
        group_cols = ["interaction_m"]

    for _, subset in sweep_df.groupby(group_cols, sort=True):
        subset = subset.sort_values("N")
        label = _series_label(subset.iloc[0])
        axes[0].plot(
            subset["N"],
            subset["correct_consensus_rate"],
            marker="o",
            linewidth=2.0,
            label=label,
        )
        axes[1].plot(
            subset["N"],
            subset["final_accuracy_mean"],
            marker="o",
            linewidth=2.0,
            label=label,
        )
        if "time_to_correct_consensus_mean" in subset.columns:
            valid = subset.dropna(subset=["time_to_correct_consensus_mean"])
            if not valid.empty:
                axes[2].plot(
                    valid["N"],
                    valid["time_to_correct_consensus_mean"],
                    marker="o",
                    linewidth=2.0,
                    label=label,
                )

    axes[0].set_ylabel("Correct Consensus Rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="best")

    axes[1].set_ylabel("Final Accuracy Mean")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="best")

    axes[2].set_xlabel("Population Size N")
    axes[2].set_ylabel("Mean Time To Correct")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(plots_dir / "sweep_summary.png", dpi=150)
    plt.close(fig)

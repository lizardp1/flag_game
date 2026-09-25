from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any

_plot_cache_dir = Path(tempfile.gettempdir()) / "nnd_matplotlib_cache"
_plot_cache_dir.mkdir(parents=True, exist_ok=True)
(_plot_cache_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache_dir / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_plot_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


_MODEL_COLORS = {
    "gpt-5.4": "#d95f02",
    "gpt-4o": "#1b9e77",
}

_LINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
_LINE_STYLES = ["-", "--", "-.", ":"]


def _share_metadata(
    per_round_df: pd.DataFrame,
    *,
    truth_country: str,
) -> tuple[list[str], dict[str, float]]:
    share_columns = [
        column
        for column in per_round_df.columns
        if column.startswith("final_share_")
    ]
    countries = [column.removeprefix("final_share_") for column in share_columns]
    max_share_by_country = {
        country: float(per_round_df[f"final_share_{country}"].max())
        for country in countries
    }
    active_countries = [
        country
        for country in countries
        if max_share_by_country[country] > 0.0
    ]
    ranked_countries = sorted(
        active_countries,
        key=lambda country: (-max_share_by_country[country], country),
    )
    if truth_country in countries and truth_country not in ranked_countries:
        ranked_countries.append(truth_country)
    return ranked_countries, max_share_by_country


def _tracked_countries(
    per_round_df: pd.DataFrame,
    *,
    truth_country: str,
    max_countries: int,
) -> tuple[list[str], list[str]]:
    ranked_countries, _ = _share_metadata(per_round_df, truth_country=truth_country)
    if len(ranked_countries) <= max_countries:
        return ranked_countries, []

    if truth_country in ranked_countries[:max_countries]:
        tracked = ranked_countries[:max_countries]
    else:
        tracked = ranked_countries[: max_countries - 1] + [truth_country]

    seen: set[str] = set()
    deduped_tracked: list[str] = []
    for country in tracked:
        if country in seen:
            continue
        seen.add(country)
        deduped_tracked.append(country)

    omitted = [country for country in ranked_countries if country not in seen]
    return deduped_tracked, omitted


def _with_t0_round(
    per_round_df: pd.DataFrame,
    *,
    t0_decision_rows: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    if per_round_df.empty or not t0_decision_rows:
        return per_round_df

    share_columns = [
        column
        for column in per_round_df.columns
        if column.startswith("final_share_")
    ]
    if not share_columns:
        return per_round_df

    countries = [column.removeprefix("final_share_") for column in share_columns]
    valid_t0_rows = [
        row
        for row in t0_decision_rows
        if bool(row.get("valid", False)) and isinstance(row.get("country"), str)
    ]
    if not valid_t0_rows:
        return per_round_df

    denominator = float(len(valid_t0_rows))
    t0_row = {
        column: np.nan
        for column in per_round_df.columns
    }
    t0_row["round"] = 0
    for country in countries:
        count = sum(1 for row in valid_t0_rows if row["country"] == country)
        t0_row[f"final_share_{country}"] = float(count / denominator)
    t0_row["final_valid_count"] = int(len(valid_t0_rows))
    t0_row["final_invalid_count"] = int(len(t0_decision_rows) - len(valid_t0_rows))
    t0_row["final_support_size"] = int(
        sum(1 for country in countries if float(t0_row[f"final_share_{country}"]) > 0.0)
    )
    t0_row["final_consensus_country"] = None
    t0_row["final_consensus_correct"] = False

    augmented = pd.concat([pd.DataFrame([t0_row]), per_round_df], ignore_index=True)
    return augmented


def _decision_rows_with_t0(
    decision_rows: list[dict[str, Any]],
    *,
    t0_decision_rows: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    if not decision_rows:
        return pd.DataFrame()

    frame = pd.DataFrame(decision_rows)
    if not t0_decision_rows:
        return frame

    t0_rows = [
        {
            "round": 0,
            "agent_id": int(row["agent_id"]),
            "model": row["model"],
            "valid": bool(row.get("valid", False)),
            "country": row.get("country"),
        }
        for row in t0_decision_rows
    ]
    return pd.concat([pd.DataFrame(t0_rows), frame], ignore_index=True, sort=False)


def plot_country_share_trajectories(
    per_round_df: pd.DataFrame,
    *,
    truth_country: str,
    out_dir: Path,
    t0_decision_rows: list[dict[str, Any]] | None = None,
) -> None:
    if per_round_df.empty:
        return
    plot_df = _with_t0_round(per_round_df, t0_decision_rows=t0_decision_rows)

    ranked_countries, _ = _share_metadata(plot_df, truth_country=truth_country)
    if not ranked_countries:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tracked, omitted = _tracked_countries(
        plot_df,
        truth_country=truth_country,
        max_countries=8,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    color_values = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(tracked), 1)))
    for idx, country in enumerate(tracked):
        linewidth = 2.5 if country == truth_country else 1.8
        alpha = 1.0 if country == truth_country else 0.9
        marker = _LINE_MARKERS[idx % len(_LINE_MARKERS)]
        linestyle = _LINE_STYLES[(idx // len(_LINE_MARKERS)) % len(_LINE_STYLES)]
        ax.plot(
            plot_df["round"],
            plot_df[f"final_share_{country}"],
            color=color_values[idx],
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=country,
        )

    if omitted:
        tracked_columns = [f"final_share_{country}" for country in tracked]
        other_share = 1.0 - plot_df[tracked_columns].sum(axis=1)
        ax.plot(
            plot_df["round"],
            other_share.clip(lower=0.0),
            color="#666666",
            marker="x",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=f"Other ({len(omitted)})",
        )

    consensus_rows = plot_df[
        plot_df["final_consensus_country"].notna() & plot_df["final_consensus_correct"].fillna(False).astype(bool)
    ]
    if not consensus_rows.empty:
        consensus_round = int(consensus_rows.iloc[0]["round"])
        ax.axvline(consensus_round, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Round")
    ax.set_ylabel("Final decision share")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Country Share Trajectories (truth: {truth_country})")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "country_share_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_country_share_stacked(
    per_round_df: pd.DataFrame,
    *,
    truth_country: str,
    out_dir: Path,
    t0_decision_rows: list[dict[str, Any]] | None = None,
) -> None:
    if per_round_df.empty:
        return
    plot_df = _with_t0_round(per_round_df, t0_decision_rows=t0_decision_rows)

    ranked_countries, _ = _share_metadata(plot_df, truth_country=truth_country)
    if not ranked_countries:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    tracked, omitted = _tracked_countries(
        plot_df,
        truth_country=truth_country,
        max_countries=12,
    )
    countries_to_plot = list(tracked)
    share_arrays = [
        plot_df[f"final_share_{country}"].to_numpy(dtype=float)
        for country in tracked
    ]
    if omitted:
        tracked_sum = np.sum(np.vstack(share_arrays), axis=0) if share_arrays else np.zeros(len(plot_df))
        share_arrays.append(np.clip(1.0 - tracked_sum, a_min=0.0, a_max=1.0))
        countries_to_plot.append(f"Other ({len(omitted)})")

    rounds = plot_df["round"].to_numpy(dtype=int)
    color_values = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(countries_to_plot), 1)))

    fig_width = max(9.0, 0.55 * len(rounds) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    bottom = np.zeros(len(rounds), dtype=float)
    for idx, (country, values) in enumerate(zip(countries_to_plot, share_arrays, strict=False)):
        ax.bar(
            rounds,
            values,
            bottom=bottom,
            width=0.78,
            color=color_values[idx],
            edgecolor="white",
            linewidth=0.7,
            label=country,
        )
        bottom += values

    consensus_rows = plot_df[
        plot_df["final_consensus_country"].notna() & plot_df["final_consensus_correct"].fillna(False).astype(bool)
    ]
    if not consensus_rows.empty:
        consensus_round = int(consensus_rows.iloc[0]["round"])
        ax.axvline(consensus_round, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Round")
    ax.set_ylabel("Final decision share")
    ax.set_xticks(rounds)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        f"Round-Level Final Decision Composition (truth: {truth_country})\n"
        "Each bar stacks the countries stored after that round"
    )
    ncol = 2 if len(countries_to_plot) > 8 else 1
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, ncol=ncol)
    fig.tight_layout()
    fig.savefig(plots_dir / "country_share_stacked.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_decision_memory_trajectories(
    decision_rows: list[dict[str, Any]],
    *,
    truth_country: str,
    agent_models: list[str],
    memory_capacity: int,
    out_dir: Path,
    t0_decision_rows: list[dict[str, Any]] | None = None,
) -> None:
    if not decision_rows:
        return
    frame = _decision_rows_with_t0(decision_rows, t0_decision_rows=t0_decision_rows)
    valid = frame[frame["valid"] & frame["country"].notna()].copy()
    if valid.empty:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rounds = sorted(int(value) for value in valid["round"].unique())
    agents = sorted(int(value) for value in valid["agent_id"].unique())
    countries = sorted(str(value) for value in valid["country"].unique())
    if truth_country in countries:
        countries = [truth_country] + [country for country in countries if country != truth_country]

    country_to_index = {country: idx for idx, country in enumerate(countries)}
    matrix = np.full((len(agents), len(rounds)), np.nan)
    for row in valid.itertuples(index=False):
        agent_idx = agents.index(int(row.agent_id))
        round_idx = rounds.index(int(row.round))
        matrix[agent_idx, round_idx] = float(country_to_index[str(row.country)])

    color_values = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(countries), 1)))
    cmap = ListedColormap(color_values[: len(countries)])
    cmap.set_bad(color="#f4f4f4")

    fig_height = max(4.5, 0.45 * len(agents) + 2.0)
    fig_width = max(8.0, 0.55 * len(rounds) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=max(len(countries) - 1, 0))

    ax.set_xlabel("Round")
    ax.set_ylabel("Agent")
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_xticklabels(rounds)
    ax.set_yticks(np.arange(len(agents)))
    ax.set_yticklabels([f"agent {agent_id}" for agent_id in agents])
    for tick, agent_id in zip(ax.get_yticklabels(), agents, strict=False):
        tick.set_color(_model_color(agent_models[agent_id]))

    consensus_round = _first_consensus_round(valid)
    if consensus_round is not None and consensus_round in rounds:
        ax.axvline(rounds.index(consensus_round), color="black", linestyle="--", linewidth=1.2, alpha=0.9)

    model_handles = [
        Patch(color=_MODEL_COLORS["gpt-5.4"], label="agent label color: gpt-5.4"),
        Patch(color=_MODEL_COLORS["gpt-4o"], label="agent label color: gpt-4o"),
    ]
    country_handles = [
        Patch(color=cmap(index), label=country)
        for country, index in country_to_index.items()
    ]
    handles = model_handles + country_handles
    ncol = 2 if len(handles) > 8 else 1
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, ncol=ncol)
    ax.set_title(
        f"Stored Decision Trajectories (truth: {truth_country}, H={memory_capacity})\n"
        "Each cell is the final country stored by that agent after that round"
    )
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, ticks=[])
    fig.tight_layout()
    fig.savefig(plots_dir / "decision_memory_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_influence_heatmap(
    decision_rows: list[dict[str, Any]],
    *,
    agent_models: list[str],
    out_dir: Path,
) -> None:
    if not decision_rows:
        return
    frame = pd.DataFrame(decision_rows)
    if frame.empty:
        return

    agent_ids = list(range(len(agent_models)))
    matrix = np.zeros((len(agent_ids), len(agent_ids)), dtype=float)
    for row in frame.itertuples(index=False):
        if not bool(row.valid):
            continue
        source = int(row.agent_id)
        for influencer in getattr(row, "influential_agent_ids", []) or []:
            matrix[source, int(influencer)] += 1.0

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_size = max(6.0, 0.5 * len(agent_ids) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xlabel("Cited Influential Agent")
    ax.set_ylabel("Decision-Making Agent")
    ax.set_xticks(np.arange(len(agent_ids)))
    ax.set_xticklabels([f"a{agent_id}" for agent_id in agent_ids], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(agent_ids)))
    ax.set_yticklabels([f"a{agent_id}" for agent_id in agent_ids])
    for tick, agent_id in zip(ax.get_xticklabels(), agent_ids, strict=False):
        tick.set_color(_model_color(agent_models[agent_id]))
    for tick, agent_id in zip(ax.get_yticklabels(), agent_ids, strict=False):
        tick.set_color(_model_color(agent_models[agent_id]))
    ax.set_title("Influence Citation Heatmap\n(axis label color = agent model)")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Times cited influential")
    fig.tight_layout()
    fig.savefig(plots_dir / "influence_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_influence_received_ranking(
    decision_rows: list[dict[str, Any]],
    *,
    agent_models: list[str],
    out_dir: Path,
) -> None:
    if not decision_rows:
        return
    frame = pd.DataFrame(decision_rows)
    if frame.empty:
        return

    citation_counts = {agent_id: 0 for agent_id in range(len(agent_models))}
    for row in frame.itertuples(index=False):
        if not bool(row.valid):
            continue
        for influencer in getattr(row, "influential_agent_ids", []) or []:
            citation_counts[int(influencer)] += 1

    ordered_agents = sorted(
        citation_counts,
        key=lambda agent_id: (-citation_counts[agent_id], agent_id),
    )
    counts = [citation_counts[agent_id] for agent_id in ordered_agents]
    colors = [_model_color(agent_models[agent_id]) for agent_id in ordered_agents]

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_width = max(7.0, 0.45 * len(ordered_agents) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(np.arange(len(ordered_agents)), counts, color=colors)
    ax.set_xticks(np.arange(len(ordered_agents)))
    ax.set_xticklabels([f"a{agent_id}" for agent_id in ordered_agents], rotation=45, ha="right")
    ax.set_ylabel("Times Cited Influential")
    ax.set_title("Most Frequently Cited Influential Agents")
    handles = [
        Patch(color=_MODEL_COLORS["gpt-5.4"], label="gpt-5.4"),
        Patch(color=_MODEL_COLORS["gpt-4o"], label="gpt-4o"),
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "influence_received_ranking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_influence_family_progression(
    decision_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
) -> None:
    if not decision_rows:
        return
    frame = pd.DataFrame(decision_rows)
    if frame.empty:
        return

    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        if not bool(row.valid):
            continue
        round_value = int(row.round)
        influential_models = list(getattr(row, "influential_models", []) or [])
        rows.append(
            {
                "round": round_value,
                "gpt-5.4": sum(model == "gpt-5.4" for model in influential_models),
                "gpt-4o": sum(model == "gpt-4o" for model in influential_models),
            }
        )

    if not rows:
        return

    counts_df = pd.DataFrame(rows).groupby("round", sort=True)[["gpt-5.4", "gpt-4o"]].sum().reset_index()
    rounds = counts_df["round"].to_numpy(dtype=int)
    gpt54_counts = counts_df["gpt-5.4"].to_numpy(dtype=float)
    gpt4o_counts = counts_df["gpt-4o"].to_numpy(dtype=float)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_width = max(8.0, 0.65 * len(rounds) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(rounds, gpt4o_counts, color=_MODEL_COLORS["gpt-4o"], label="Influence citations to gpt-4o")
    ax.bar(
        rounds,
        gpt54_counts,
        bottom=gpt4o_counts,
        color=_MODEL_COLORS["gpt-5.4"],
        label="Influence citations to gpt-5.4",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Influence citations received")
    ax.set_title("Progression of Most Influential Model Family")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plots_dir / "influence_family_progression.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _first_consensus_round(valid_decision_frame: pd.DataFrame) -> int | None:
    for round_value, subset in valid_decision_frame.groupby("round", sort=True):
        countries = {str(value) for value in subset["country"]}
        if len(countries) == 1 and len(subset) == len(valid_decision_frame["agent_id"].unique()):
            return int(round_value)
    return None


def _model_color(model: str) -> str:
    return _MODEL_COLORS.get(model, "#6a6a6a")

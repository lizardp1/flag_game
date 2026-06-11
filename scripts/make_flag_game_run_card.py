#!/usr/bin/env python3
"""Create a one-page visual summary card for a flag-game run."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_plot_cache_dir = Path(tempfile.gettempdir()) / "nnd_matplotlib_cache"
_plot_cache_dir.mkdir(parents=True, exist_ok=True)
(_plot_cache_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache_dir / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_plot_cache_dir))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image


METADATA_COLUMNS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a paper-style visual card from a completed flag-game run."
    )
    parser.add_argument("--run", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <run>/plots/run_card.png.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-countries", type=int, default=8)
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run.resolve()
    out_path = args.out or (run_dir / "plots" / "run_card.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_json(run_dir / "trial_manifest.json")
    summary = read_json(run_dir / "summary.json")
    per_round = read_csv(run_dir / "per_round.csv")
    probes = read_jsonl(run_dir / "probes.jsonl")

    truth_country = str(
        summary.get("truth_country")
        or manifest.get("truth_country")
        or (per_round.iloc[0]["truth_country"] if not per_round.empty and "truth_country" in per_round else "unknown")
    )

    fig = plt.figure(figsize=(18, 8.6), facecolor="#fbf8f1")
    outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.02, 1.08],
        left=0.025,
        right=0.985,
        top=0.94,
        bottom=0.08,
        wspace=0.08,
    )
    left = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[4.6, 1.25], hspace=0.16)
    right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[3.1, 1.1], hspace=0.26)

    draw_flag_panel(
        fig=fig,
        spec=left[0],
        run_dir=run_dir,
        manifest=manifest,
        probes=probes,
        truth_country=truth_country,
    )
    draw_crop_thumbnails(fig=fig, spec=left[1], run_dir=run_dir, probes=probes)
    draw_trajectory_panel(
        fig=fig,
        spec=right[0],
        per_round=per_round,
        truth_country=truth_country,
        max_countries=args.max_countries,
    )
    draw_summary_panel(fig=fig, spec=right[1], summary=summary, truth_country=truth_country)

    title = args.title or f"Flag Game Run Card: {truth_country}"
    fig.suptitle(title, x=0.5, y=0.985, fontsize=18, fontweight="bold", color="#2f2a24")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    write_index(run_dir=run_dir, card_path=out_path)
    print(f"Wrote run card to {out_path}")


def draw_flag_panel(
    *,
    fig: plt.Figure,
    spec: Any,
    run_dir: Path,
    manifest: dict[str, Any],
    probes: list[dict[str, Any]],
    truth_country: str,
) -> None:
    ax = fig.add_subplot(spec)
    style_card_axis(ax)
    truth_path = run_dir / "artifacts" / "truth_flag.png"
    if truth_path.exists():
        image = np.asarray(Image.open(truth_path).convert("RGB"))
        ax.imshow(image)
        width = image.shape[1]
        height = image.shape[0]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
    else:
        width = 1
        height = 1
        ax.text(0.5, 0.5, "truth_flag.png missing", ha="center", va="center")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    t0_by_agent = {
        int(row.get("agent_id", -1)): row
        for row in probes
        if int(row.get("t", -1)) == 0
    }
    for assignment in manifest.get("pixel_assignments") or manifest.get("assignments") or []:
        agent_id = int(assignment.get("agent_id", -1))
        left = float(assignment.get("left", 0))
        top = float(assignment.get("top", 0))
        box_width = float(assignment.get("width", 0))
        box_height = float(assignment.get("height", 0))
        if box_width <= 0 or box_height <= 0:
            continue
        color = "#bdf25a"
        ax.add_patch(
            Rectangle(
                (left, top),
                box_width,
                box_height,
                fill=False,
                edgecolor=color,
                linewidth=3.2,
                alpha=0.92,
                joinstyle="round",
            )
        )
        predicted = t0_by_agent.get(agent_id, {}).get("country") or "?"
        label = f"A{agent_id}: {predicted}"
        ax.text(
            left + 6,
            top + 18,
            label,
            fontsize=8.5,
            color="#1f2937",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#f7ffd5",
                "edgecolor": color,
                "linewidth": 1.2,
                "alpha": 0.88,
            },
        )
    ax.set_title(f"Truth: {truth_country}", fontsize=16, color="#2f2a24", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_crop_thumbnails(
    *,
    fig: plt.Figure,
    spec: Any,
    run_dir: Path,
    probes: list[dict[str, Any]],
) -> None:
    crop_paths = sorted((run_dir / "artifacts").glob("agent_*_crop.png"))
    if not crop_paths:
        ax = fig.add_subplot(spec)
        style_card_axis(ax)
        ax.text(0.5, 0.5, "No agent crop images saved", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    grid = GridSpecFromSubplotSpec(1, len(crop_paths), subplot_spec=spec, wspace=0.12)
    t0_by_agent = {
        int(row.get("agent_id", -1)): row
        for row in probes
        if int(row.get("t", -1)) == 0
    }
    for idx, path in enumerate(crop_paths):
        ax = fig.add_subplot(grid[0, idx])
        style_card_axis(ax)
        ax.imshow(Image.open(path).convert("RGB"))
        agent_id = int(path.stem.split("_")[1])
        predicted = t0_by_agent.get(agent_id, {}).get("country") or "?"
        correctness = "OK" if bool(t0_by_agent.get(agent_id, {}).get("correct", False)) else ""
        title = f"A{agent_id}: {predicted}"
        if correctness:
            title = f"{title} ({correctness})"
        ax.set_title(title, fontsize=9, color="#374151")
        ax.set_xticks([])
        ax.set_yticks([])


def draw_trajectory_panel(
    *,
    fig: plt.Figure,
    spec: Any,
    per_round: pd.DataFrame,
    truth_country: str,
    max_countries: int,
) -> None:
    ax = fig.add_subplot(spec)
    style_card_axis(ax)
    if per_round.empty:
        ax.text(0.5, 0.5, "per_round.csv missing", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    countries = [col for col in per_round.columns if col not in METADATA_COLUMNS]
    ranked = sorted(
        countries,
        key=lambda country: (-float(pd.to_numeric(per_round[country], errors="coerce").fillna(0).max()), country),
    )
    tracked = ranked[: max(1, max_countries)]
    if truth_country in countries and truth_country not in tracked:
        tracked = tracked[:-1] + [truth_country] if tracked else [truth_country]

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(tracked), 1)))
    for idx, country in enumerate(tracked):
        values = pd.to_numeric(per_round[country], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        is_truth = country == truth_country
        ax.plot(
            per_round["t"],
            values,
            marker="o",
            markersize=4.2 if is_truth else 3.2,
            linewidth=3.0 if is_truth else 1.8,
            alpha=1.0 if is_truth else 0.78,
            color="#d137d6" if is_truth else colors[idx],
            label=f"{country} (truth)" if is_truth else country,
        )
    ax.set_title("Country Share Trajectories", fontsize=16, color="#2f2a24", pad=12)
    ax.set_xlabel("Interaction step")
    ax.set_ylabel("Country share")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, color="#decfbd", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=9)


def draw_summary_panel(
    *,
    fig: plt.Figure,
    spec: Any,
    summary: dict[str, Any],
    truth_country: str,
) -> None:
    ax = fig.add_subplot(spec)
    style_card_axis(ax)
    outcome = str(summary.get("final_outcome", "unknown"))
    consensus = summary.get("final_consensus_country") or summary.get("final_vote_country") or "none"
    final_accuracy = as_float(summary.get("final_accuracy"))
    initial_accuracy = as_float(summary.get("initial_accuracy"))
    support = summary.get("final_support_size", "n/a")
    consensus_correct = bool(summary.get("final_consensus_correct", False))
    icon = "OK" if consensus_correct else "X"
    color = "#2f9e44" if consensus_correct else "#d9776f"
    ax.text(0.045, 0.66, icon, fontsize=28, fontweight="bold", color=color, va="center")
    ax.text(
        0.12,
        0.76,
        outcome.replace("_", " ").title(),
        fontsize=15,
        color=color,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.12,
        0.56,
        f"Consensus/top vote: {consensus}    Truth: {truth_country}",
        fontsize=11,
        color="#4b5563",
        transform=ax.transAxes,
    )
    ax.text(
        0.12,
        0.34,
        f"Initial accuracy: {format_percent(initial_accuracy)}    Final accuracy: {format_percent(final_accuracy)}    Countries in play: {support}",
        fontsize=10,
        color="#6b7280",
        transform=ax.transAxes,
    )
    ax.text(
        0.12,
        0.15,
        f"N={summary.get('N', 'n/a')}  T={summary.get('executed_T', summary.get('T', 'n/a'))}  H={summary.get('H', 'n/a')}  m={summary.get('interaction_m', 'n/a')}  model={summary.get('model', 'n/a')}",
        fontsize=9.5,
        color="#6b7280",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def style_card_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fffdf8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="#e2d8c8",
            facecolor="none",
            clip_on=False,
            zorder=10,
        )
    )


def write_index(*, run_dir: Path, card_path: Path) -> None:
    rel_card = card_path.relative_to(run_dir) if card_path.is_relative_to(run_dir) else card_path
    links = [
        ("Run card", rel_card),
        ("Country share trajectories", Path("plots/country_share_trajectories.png")),
        ("Run overview", Path("plots/run_overview.png")),
        ("Initial vs final distribution", Path("plots/initial_vs_final_distribution.png")),
        ("Truth flag", Path("artifacts/truth_flag.png")),
    ]
    body = ["<!doctype html>", "<meta charset='utf-8'>", "<title>Flag Game Visual Report</title>"]
    body.append("<style>body{font-family:system-ui;margin:32px;background:#fbf8f1;color:#2f2a24} img{max-width:100%;border:1px solid #e2d8c8;border-radius:10px;background:white} li{margin:8px 0}</style>")
    body.append("<h1>Flag Game Visual Report</h1>")
    body.append(f"<p><img src='{rel_card.as_posix()}' alt='run card'></p>")
    body.append("<ul>")
    for label, path in links:
        if (run_dir / path).exists():
            body.append(f"<li><a href='{path.as_posix()}'>{label}</a></li>")
    body.append("</ul>")
    (run_dir / "visual_report.html").write_text("\n".join(body))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


if __name__ == "__main__":
    main()

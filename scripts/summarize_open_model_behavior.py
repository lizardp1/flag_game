#!/usr/bin/env python3
"""Summarize early open-model flag-game behavior across run types.

Phase 1 is deliberately behavior-first: before interpreting hidden states, this
script collects the visible capabilities and social behaviors that make the
activation study worth doing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate visual capability, flag-game batch, and memory-conflict "
            "probe outputs into one early-phase model behavior report."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs"),
        help="Root to discover runs from when explicit run dirs are omitted.",
    )
    parser.add_argument("--visual-run", type=Path, action="append", default=[])
    parser.add_argument("--flag-run", type=Path, action="append", default=[])
    parser.add_argument("--memory-conflict-run", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="Only use explicitly supplied run directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    visual_runs = unique_paths(args.visual_run)
    flag_runs = unique_paths(args.flag_run)
    memory_runs = unique_paths(args.memory_conflict_run)
    if not args.no_discover:
        visual_runs.extend(path for path in discover_visual_runs(args.root) if path not in visual_runs)
        flag_runs.extend(path for path in discover_flag_runs(args.root) if path not in flag_runs)
        memory_runs.extend(path for path in discover_memory_conflict_runs(args.root) if path not in memory_runs)

    visual_summary, visual_color_summary = summarize_visual_runs(visual_runs)
    flag_summary, flag_country_dist = summarize_flag_runs(flag_runs)
    memory_thresholds, memory_count_summary, memory_choice_dist = summarize_memory_runs(memory_runs)

    write_frame(visual_summary, out_dir / "visual_capability_summary.csv")
    write_frame(visual_color_summary, out_dir / "visual_color_summary.csv")
    write_frame(flag_summary, out_dir / "flag_game_behavior_summary.csv")
    write_frame(flag_country_dist, out_dir / "flag_game_initial_country_distribution.csv")
    write_frame(memory_thresholds, out_dir / "memory_conflict_thresholds.csv")
    write_frame(memory_count_summary, out_dir / "memory_conflict_by_count.csv")
    write_frame(memory_choice_dist, out_dir / "memory_conflict_choice_distribution.csv")
    write_report(
        out_dir / "model_behavior_report.md",
        visual_summary=visual_summary,
        visual_color_summary=visual_color_summary,
        flag_summary=flag_summary,
        flag_country_dist=flag_country_dist,
        memory_thresholds=memory_thresholds,
        memory_choice_dist=memory_choice_dist,
        runs={
            "visual_runs": [str(path) for path in visual_runs],
            "flag_runs": [str(path) for path in flag_runs],
            "memory_conflict_runs": [str(path) for path in memory_runs],
        },
    )
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "visual_runs": [str(path) for path in visual_runs],
                "flag_runs": [str(path) for path in flag_runs],
                "memory_conflict_runs": [str(path) for path in memory_runs],
                "outputs": [
                    "visual_capability_summary.csv",
                    "visual_color_summary.csv",
                    "flag_game_behavior_summary.csv",
                    "flag_game_initial_country_distribution.csv",
                    "memory_conflict_thresholds.csv",
                    "memory_conflict_by_count.csv",
                    "memory_conflict_choice_distribution.csv",
                    "model_behavior_report.md",
                ],
            },
            handle,
            indent=2,
        )
    print(f"Wrote open-model behavior summary to {out_dir}")


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    resolved_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            resolved_paths.append(resolved)
    return resolved_paths


def discover_visual_runs(root: Path) -> list[Path]:
    return unique_paths([path.parent for path in root.glob("**/color_group_summary.csv")])


def discover_flag_runs(root: Path) -> list[Path]:
    return unique_paths([path.parent for path in root.glob("**/batch_summary.csv")])


def discover_memory_conflict_runs(root: Path) -> list[Path]:
    return unique_paths([path.parent for path in root.glob("**/threshold_summary.csv")])


def summarize_visual_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    color_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        group_path = run_dir / "color_group_summary.csv"
        if group_path.exists():
            group_df = pd.read_csv(group_path)
            for _, row in group_df.iterrows():
                group_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "model_id": first_present(row, ["model_id", "model"]),
                        "expected_color_group": row.get("expected_color_group"),
                        "trial_count": row.get("trial_count"),
                        "valid_json_rate": row.get("valid_json_rate"),
                        "color_correct_rate": row.get("color_correct_rate"),
                        "expected_colors": row.get("expected_colors"),
                        "predicted_colors": row.get("predicted_colors"),
                    }
                )
        color_path = run_dir / "color_summary.csv"
        if color_path.exists():
            color_df = pd.read_csv(color_path)
            for _, row in color_df.iterrows():
                color_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "model_id": first_present(row, ["model_id", "model"]),
                        "expected_color": row.get("expected_color"),
                        "trial_count": row.get("trial_count"),
                        "valid_json_rate": row.get("valid_json_rate"),
                        "color_correct_rate": row.get("color_correct_rate"),
                        "predicted_colors": row.get("predicted_colors"),
                    }
                )
        results_path = run_dir / "results.csv"
        if results_path.exists():
            results_df = pd.read_csv(results_path)
            if "task_type" in results_df and "model_id" in results_df:
                for keys, group in results_df.groupby(["model_id", "task_type"], dropna=False):
                    model_id, task_type = keys
                    group_rows.append(
                        {
                            "run_dir": str(run_dir),
                            "model_id": model_id,
                            "expected_color_group": f"overall_{task_type}",
                            "trial_count": int(len(group)),
                            "valid_json_rate": mean_if_present(group, "valid_json"),
                            "color_correct_rate": mean_if_present(
                                group,
                                "color_correct",
                                fallback_cols=["orientation_correct", "stripe_count_correct"],
                            ),
                            "expected_colors": None,
                            "predicted_colors": None,
                        }
                    )
    return pd.DataFrame(group_rows), pd.DataFrame(color_rows)


def summarize_flag_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    country_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        batch_path = run_dir / "batch_summary.csv"
        if not batch_path.exists():
            continue
        batch_df = pd.read_csv(batch_path)
        model_name = infer_model_name(run_dir, batch_df)
        final_vote = batch_df["final_vote_country"].fillna("") if "final_vote_country" in batch_df else pd.Series([])
        initial_vote = batch_df["initial_vote_country"].fillna("") if "initial_vote_country" in batch_df else pd.Series([])
        summary_rows.append(
            {
                "run_dir": str(run_dir),
                "model_name": model_name,
                "seed_count": int(len(batch_df)),
                "initial_accuracy_mean": mean_if_present(batch_df, "initial_accuracy"),
                "final_accuracy_mean": mean_if_present(batch_df, "final_accuracy"),
                "oracle_accuracy_mean": mean_if_present(batch_df, "oracle_accuracy"),
                "correct_consensus_rate": mean_bool_if_present(batch_df, "final_consensus_correct"),
                "wrong_consensus_rate": mean_condition(batch_df, "final_outcome", "wrong_consensus"),
                "initial_vote_france_rate": float((initial_vote == "France").mean()) if len(initial_vote) else None,
                "final_vote_france_rate": float((final_vote == "France").mean()) if len(final_vote) else None,
                "unique_initial_vote_countries": int(initial_vote[initial_vote != ""].nunique()) if len(initial_vote) else None,
                "unique_final_vote_countries": int(final_vote[final_vote != ""].nunique()) if len(final_vote) else None,
            }
        )
        country_rows.extend(load_initial_probe_country_distribution(run_dir, model_name))
    return pd.DataFrame(summary_rows), pd.DataFrame(country_rows)


def summarize_memory_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_frames: list[pd.DataFrame] = []
    count_frames: list[pd.DataFrame] = []
    choice_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        threshold_path = run_dir / "threshold_summary.csv"
        if threshold_path.exists():
            threshold_df = pd.read_csv(threshold_path)
            threshold_df.insert(0, "run_dir", str(run_dir))
            threshold_frames.append(threshold_df)
        count_path = run_dir / "summary_by_count.csv"
        if count_path.exists():
            count_df = pd.read_csv(count_path)
            count_df.insert(0, "run_dir", str(run_dir))
            count_frames.append(count_df)
        results_path = run_dir / "results.csv"
        if results_path.exists():
            results_df = pd.read_csv(results_path)
            if "response_type" in results_df:
                group_cols = [
                    col
                    for col in [
                        "model",
                        "m",
                        "crop_condition",
                        "lure_relation",
                        "false_memory_count",
                        "response_type",
                    ]
                    if col in results_df
                ]
                counts = results_df.groupby(group_cols, dropna=False).size().reset_index(name="count")
                totals = counts.groupby(group_cols[:-1])["count"].transform("sum") if len(group_cols) > 1 else len(results_df)
                counts["fraction"] = counts["count"] / totals
                for _, row in counts.iterrows():
                    output = row.to_dict()
                    output["run_dir"] = str(run_dir)
                    choice_rows.append(output)
    thresholds = concat_or_empty(threshold_frames)
    count_summary = concat_or_empty(count_frames)
    choice_dist = pd.DataFrame(choice_rows)
    if not choice_dist.empty:
        front = ["run_dir"]
        choice_dist = choice_dist[front + [col for col in choice_dist.columns if col not in front]]
    return thresholds, count_summary, choice_dist


def load_initial_probe_country_distribution(run_dir: Path, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    seed_dirs = sorted(path for path in run_dir.glob("seed_*") if path.is_dir())
    for seed_dir in seed_dirs:
        for probe in read_jsonl(seed_dir / "probes.jsonl"):
            if int(probe.get("t", -1)) != 0:
                continue
            raw_rows.append(
                {
                    "run_dir": str(run_dir),
                    "model_name": model_name,
                    "seed_dir": seed_dir.name,
                    "predicted_country": probe.get("country"),
                    "valid": bool(probe.get("valid", False)),
                    "correct": bool(probe.get("correct", False)),
                }
            )
    if not raw_rows:
        return rows
    df = pd.DataFrame(raw_rows)
    valid = df[df["valid"] == True]  # noqa: E712
    if valid.empty:
        return rows
    counts = valid.groupby("predicted_country", dropna=False).size().reset_index(name="count")
    total = int(counts["count"].sum())
    for _, row in counts.sort_values("count", ascending=False).iterrows():
        rows.append(
            {
                "run_dir": str(run_dir),
                "model_name": model_name,
                "predicted_country": row["predicted_country"],
                "count": int(row["count"]),
                "fraction": float(row["count"] / total) if total else None,
            }
        )
    return rows


def write_report(
    out_path: Path,
    *,
    visual_summary: pd.DataFrame,
    visual_color_summary: pd.DataFrame,
    flag_summary: pd.DataFrame,
    flag_country_dist: pd.DataFrame,
    memory_thresholds: pd.DataFrame,
    memory_choice_dist: pd.DataFrame,
    runs: dict[str, list[str]],
) -> None:
    lines = [
        "# Open-Model Behavior Report",
        "",
        "This report is Phase 1: visible model capability and social behavior before any internal-state claims.",
        "",
        "## Inputs",
        "",
    ]
    for key, values in runs.items():
        lines.append(f"- `{key}`: {len(values)} run(s)")
    lines.extend(["", "## Visual Capability", ""])
    lines.extend(markdown_table(visual_summary, max_rows=12))
    if not visual_color_summary.empty:
        lines.extend(["", "### Per-Color Checks", ""])
        lines.extend(markdown_table(visual_color_summary, max_rows=16))
    lines.extend(["", "## Flag-Game Social Behavior", ""])
    lines.extend(markdown_table(flag_summary, max_rows=12))
    if not flag_country_dist.empty:
        lines.extend(["", "### Initial Country Distribution", ""])
        lines.extend(markdown_table(flag_country_dist, max_rows=16))
    lines.extend(["", "## Memory-Conflict Psychology Probe", ""])
    lines.extend(markdown_table(memory_thresholds, max_rows=16))
    if not memory_choice_dist.empty:
        lines.extend(["", "### Response-Type Distribution", ""])
        lines.extend(markdown_table(memory_choice_dist, max_rows=16))
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- Visual color/stripe failures mean later social failures may be perceptual, not social.",
            "- A single-country collapse in the flag game means the country list or prior is dominating.",
            "- Memory-conflict thresholds show whether social evidence can logically override private evidence under controlled conditions.",
            "- Phase 3 activation claims should be limited to cells where Phase 1/2 behavior is meaningful.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def markdown_table(df: pd.DataFrame, *, max_rows: int) -> list[str]:
    if df.empty:
        return ["No rows found."]
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        shown[col] = shown[col].map(format_cell)
    columns = [str(col) for col in shown.columns]
    lines = [
        "| " + " | ".join(escape_markdown_cell(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append(
            "| "
            + " | ".join(escape_markdown_cell(row[col]) for col in shown.columns)
            + " |"
        )
    return lines


def format_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def infer_model_name(run_dir: Path, batch_df: pd.DataFrame) -> str:
    if "model" in batch_df and batch_df["model"].notna().any():
        return str(batch_df["model"].dropna().iloc[0])
    if "model_name" in batch_df and batch_df["model_name"].notna().any():
        return str(batch_df["model_name"].dropna().iloc[0])
    return run_dir.name


def first_present(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def mean_if_present(df: pd.DataFrame, col: str, *, fallback_cols: list[str] | None = None) -> float | None:
    if col in df:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None
    for fallback_col in fallback_cols or []:
        if fallback_col in df:
            values = pd.to_numeric(df[fallback_col], errors="coerce").dropna()
            return float(values.mean()) if len(values) else None
    return None


def mean_bool_if_present(df: pd.DataFrame, col: str) -> float | None:
    if col not in df:
        return None
    values = df[col].dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def mean_condition(df: pd.DataFrame, col: str, value: Any) -> float | None:
    if col not in df or df.empty:
        return None
    return float((df[col] == value).mean())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


if __name__ == "__main__":
    main()

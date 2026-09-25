from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from nnd.flag_game.config import FlagGameConfig
from nnd.flag_game.runner import run_flag_game_batch


def build_agent_model_assignment(
    *,
    n_agents: int,
    base_model: str,
    boosted_model: str,
    boosted_agent_ids: Sequence[int] | None = None,
) -> list[str]:
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    active_ids = [0] if boosted_agent_ids is None else [int(agent_id) for agent_id in boosted_agent_ids]
    if not active_ids:
        raise ValueError("boosted_agent_ids must contain at least one agent id")
    if len(set(active_ids)) != len(active_ids):
        raise ValueError("boosted_agent_ids must be unique")
    if any(agent_id < 0 or agent_id >= n_agents for agent_id in active_ids):
        raise ValueError(f"boosted_agent_ids must be between 0 and {n_agents - 1}")

    agent_models = [base_model for _ in range(n_agents)]
    for agent_id in active_ids:
        agent_models[agent_id] = boosted_model
    return agent_models


def run_flag_game_model_mix_comparison(
    base_config: FlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
    baseline_model: str,
    boosted_model: str,
    boosted_agent_ids: Sequence[int] | None = None,
    include_pure_controls: bool = True,
    include_mixed_condition: bool = True,
    condition_workers: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mixed_agent_models = build_agent_model_assignment(
        n_agents=base_config.N,
        base_model=baseline_model,
        boosted_model=boosted_model,
        boosted_agent_ids=boosted_agent_ids,
    )
    active_boosted_ids = [idx for idx, model in enumerate(mixed_agent_models) if model == boosted_model]

    conditions: list[dict[str, Any]] = []
    if include_pure_controls:
        conditions.extend(
            [
                {
                    "name": f"all_{_slug(baseline_model)}",
                    "label": f"All {baseline_model}",
                    "config": base_config.model_copy(update={"model": baseline_model, "agent_models": None}),
                    "agent_models": [baseline_model for _ in range(base_config.N)],
                    "boosted_agent_ids": [],
                },
                {
                    "name": f"all_{_slug(boosted_model)}",
                    "label": f"All {boosted_model}",
                    "config": base_config.model_copy(update={"model": boosted_model, "agent_models": None}),
                    "agent_models": [boosted_model for _ in range(base_config.N)],
                    "boosted_agent_ids": list(range(base_config.N)),
                },
            ]
        )
    if include_mixed_condition:
        conditions.append(
            {
                "name": f"mix_{_slug(baseline_model)}_plus_{len(active_boosted_ids)}_{_slug(boosted_model)}",
                "label": f"{baseline_model} + {len(active_boosted_ids)}x {boosted_model}",
                "config": base_config.model_copy(update={"model": baseline_model, "agent_models": mixed_agent_models}),
                "agent_models": mixed_agent_models,
                "boosted_agent_ids": active_boosted_ids,
            }
        )
    if not conditions:
        raise ValueError("At least one model-mix condition must be enabled")

    spec = {
        "seeds": seeds,
        "baseline_model": baseline_model,
        "boosted_model": boosted_model,
        "boosted_agent_ids": active_boosted_ids,
        "N": base_config.N,
        "T": base_config.T,
        "H": base_config.H,
        "interaction_m": base_config.interaction_m,
        "social_susceptibility": base_config.social_susceptibility,
        "prompt_social_susceptibility": base_config.prompt_social_susceptibility,
        "prompt_style": base_config.prompt_style,
        "country_pool": base_config.country_pool,
        "tile_width": base_config.tile_width,
        "tile_height": base_config.tile_height,
        "observation_overlap": base_config.observation_overlap,
        "observation_overlap_mode": base_config.observation_overlap_mode,
        "render_scale": base_config.render_scale,
        "speaker_weights": base_config.speaker_weights,
        "include_pure_controls": include_pure_controls,
        "include_mixed_condition": include_mixed_condition,
    }
    _append_model_mix_spec(out_dir / "model_mix_spec.json", spec)

    existing_condition_rows = _load_existing_rows(out_dir / "model_mix_condition_results.csv")
    existing_boosted_rows = _load_existing_rows(out_dir / "model_mix_boosted_agent_diagnostics.csv")
    skipped_conditions: list[str] = []
    pending_conditions: list[dict[str, Any]] = []

    for condition in conditions:
        if _condition_complete(existing_condition_rows, condition["name"], seeds):
            skipped_conditions.append(condition["name"])
            continue

        existing_condition_rows = _drop_condition_seed_rows(existing_condition_rows, condition["name"], seeds)
        existing_boosted_rows = _drop_condition_seed_rows(existing_boosted_rows, condition["name"], seeds)
        pending_conditions.append(condition)

    def _run_condition(condition: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        condition_dir = out_dir / condition["name"]
        results = run_flag_game_batch(condition["config"], out_dir=condition_dir, seeds=seeds)
        return condition, results

    if condition_workers <= 1 or len(pending_conditions) <= 1:
        condition_outputs = [_run_condition(condition) for condition in pending_conditions]
    else:
        with ThreadPoolExecutor(max_workers=condition_workers) as executor:
            condition_outputs = list(executor.map(_run_condition, pending_conditions))

    for condition, results in condition_outputs:
        condition_dir = out_dir / condition["name"]
        summaries = [result["summary"] for result in results]
        summary_df = pd.DataFrame(summaries)

        correct_mask = summary_df["final_outcome"] == "correct_consensus"
        wrong_mask = summary_df["final_outcome"] == "wrong_consensus"
        polarization_mask = summary_df["final_outcome"] == "polarization"
        fragmentation_mask = summary_df["final_outcome"] == "fragmentation"

        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            boosted_diagnostics = _boosted_agent_diagnostics(result, condition["boosted_agent_ids"])
            row.update(
                {
                    "seed": seed,
                    "condition_name": condition["name"],
                    "condition_label": condition["label"],
                    "condition_dir": str(condition_dir),
                    "boosted_model": boosted_model,
                    "baseline_model": baseline_model,
                    "agent_model_signature": _agent_model_signature(condition["agent_models"]),
                    "boosted_agent_ids_json": json.dumps(condition["boosted_agent_ids"]),
                    "boosted_agent_count": len(condition["boosted_agent_ids"]),
                    "boosted_agent_compatible_country_count_mean": _mean_or_none(
                        [diag["compatible_country_count"] for diag in boosted_diagnostics]
                    ),
                    "boosted_agent_informativeness_bits_mean": _mean_or_none(
                        [diag["informativeness_bits"] for diag in boosted_diagnostics]
                    ),
                    "boosted_agent_informativeness_score_mean": _mean_or_none(
                        [diag["informativeness_score"] for diag in boosted_diagnostics]
                    ),
                }
            )
            existing_condition_rows.append(row)

            for diagnostic in boosted_diagnostics:
                existing_boosted_rows.append(
                    {
                        "seed": seed,
                        "condition_name": condition["name"],
                        "condition_label": condition["label"],
                        "agent_id": diagnostic["agent_id"],
                        "model": diagnostic["model"],
                        "truth_country": diagnostic["truth_country"],
                        "compatible_country_count": diagnostic["compatible_country_count"],
                        "compatible_countries_json": json.dumps(diagnostic["compatible_countries"]),
                        "ambiguity_fraction": diagnostic["ambiguity_fraction"],
                        "informativeness_bits": diagnostic["informativeness_bits"],
                        "informativeness_score": diagnostic["informativeness_score"],
                        "informativeness_label": diagnostic["informativeness_label"],
                        "is_unique": diagnostic["is_unique"],
                    }
                )

    condition_df = pd.DataFrame(existing_condition_rows)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["condition_name", "seed"]).reset_index(drop=True)

    boosted_df = pd.DataFrame(existing_boosted_rows)
    if not boosted_df.empty:
        boosted_df = boosted_df.sort_values(["condition_name", "seed", "agent_id"]).reset_index(drop=True)

    aggregate_rows = _aggregate_condition_rows(condition_df, boosted_df)
    summary_df = pd.DataFrame(aggregate_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("condition_name").reset_index(drop=True)

    pair_specs = _pair_specs_from_conditions(condition_df["condition_name"].tolist()) if not condition_df.empty else []
    paired_rows: list[dict[str, Any]] = []
    paired_summary_rows: list[dict[str, Any]] = []
    for left_condition, right_condition in pair_specs:
        rows = _paired_condition_rows(condition_df, left_condition=left_condition, right_condition=right_condition)
        paired_rows.extend(rows)
        if rows:
            paired_summary_rows.append(_summarize_paired_rows(rows))

    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        paired_df = paired_df.sort_values(["comparison_name", "seed"]).reset_index(drop=True)

    paired_summary_df = pd.DataFrame(paired_summary_rows)
    if not paired_summary_df.empty:
        paired_summary_df = paired_summary_df.sort_values("comparison_name").reset_index(drop=True)

    condition_df.to_csv(out_dir / "model_mix_condition_results.csv", index=False)
    summary_df.to_csv(out_dir / "model_mix_summary.csv", index=False)
    boosted_df.to_csv(out_dir / "model_mix_boosted_agent_diagnostics.csv", index=False)
    paired_df.to_csv(out_dir / "model_mix_paired_deltas.csv", index=False)
    paired_summary_df.to_csv(out_dir / "model_mix_paired_summary.csv", index=False)
    _write_report(
        out_dir / "model_mix_report.md",
        summary_rows=summary_df.to_dict(orient="records"),
        paired_rows=paired_summary_df.to_dict(orient="records"),
        boosted_rows=boosted_df.to_dict(orient="records"),
    )

    return {
        "condition_results": condition_df.to_dict(orient="records"),
        "summary": summary_df.to_dict(orient="records"),
        "boosted_agent_diagnostics": boosted_df.to_dict(orient="records"),
        "paired_deltas": paired_df.to_dict(orient="records"),
        "paired_summary": paired_summary_df.to_dict(orient="records"),
        "skipped_conditions": skipped_conditions,
    }


def _aggregate_condition_rows(condition_df: pd.DataFrame, boosted_df: pd.DataFrame) -> list[dict[str, Any]]:
    if condition_df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for condition_name, group in condition_df.groupby("condition_name", sort=True):
        boosted_group = (
            boosted_df[boosted_df["condition_name"] == condition_name]
            if not boosted_df.empty
            else pd.DataFrame()
        )
        first_row = group.iloc[0]
        correct_mask = group["final_outcome"] == "correct_consensus"
        wrong_mask = group["final_outcome"] == "wrong_consensus"
        polarization_mask = group["final_outcome"] == "polarization"
        fragmentation_mask = group["final_outcome"] == "fragmentation"

        rows.append(
            {
                "condition_name": condition_name,
                "condition_label": first_row["condition_label"],
                "baseline_model": first_row["baseline_model"],
                "boosted_model": first_row["boosted_model"],
                "agent_model_signature": first_row["agent_model_signature"],
                "boosted_agent_ids_json": first_row["boosted_agent_ids_json"],
                "boosted_agent_count": int(first_row["boosted_agent_count"]),
                "n_trials": len(group),
                "correct_consensus_rate": float(correct_mask.mean()) if len(group) else 0.0,
                "wrong_consensus_rate": float(wrong_mask.mean()) if len(group) else 0.0,
                "polarization_rate": float(polarization_mask.mean()) if len(group) else 0.0,
                "fragmentation_rate": float(fragmentation_mask.mean()) if len(group) else 0.0,
                "initial_accuracy_mean": float(group["initial_accuracy"].mean()) if not group.empty else None,
                "final_accuracy_mean": float(group["final_accuracy"].mean()) if not group.empty else None,
                "collaboration_gain_mean": (
                    float(group["collaboration_gain_over_initial_accuracy"].mean()) if not group.empty else None
                ),
                "time_to_correct_consensus_mean": (
                    float(group["time_to_correct_consensus"].dropna().mean())
                    if not group.empty and not group["time_to_correct_consensus"].dropna().empty
                    else None
                ),
                "boosted_agent_compatible_country_count_mean": _mean_or_none(
                    boosted_group["compatible_country_count"].tolist() if not boosted_group.empty else []
                ),
                "boosted_agent_informativeness_bits_mean": _mean_or_none(
                    boosted_group["informativeness_bits"].tolist() if not boosted_group.empty else []
                ),
                "boosted_agent_informativeness_score_mean": _mean_or_none(
                    boosted_group["informativeness_score"].tolist() if not boosted_group.empty else []
                ),
            }
        )
    return rows


def _pair_specs_from_conditions(condition_names: Sequence[str]) -> list[tuple[str, str]]:
    unique_names = sorted(set(str(name) for name in condition_names))
    control_names = sorted(name for name in unique_names if name.startswith("all_"))
    mix_names = sorted(name for name in unique_names if name.startswith("mix_"))

    specs: list[tuple[str, str]] = []
    if len(control_names) == 2:
        specs.append((control_names[1], control_names[0]))
    for mix_name in mix_names:
        for control_name in control_names:
            specs.append((mix_name, control_name))
    return specs


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _condition_complete(rows: Sequence[dict[str, Any]], condition_name: str, seeds: Sequence[int]) -> bool:
    present = {
        int(row["seed"])
        for row in rows
        if row.get("condition_name") == condition_name and row.get("seed") is not None
    }
    return all(int(seed) in present for seed in seeds)


def _drop_condition_seed_rows(
    rows: Sequence[dict[str, Any]],
    condition_name: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    seed_set = {int(seed) for seed in seeds}
    kept: list[dict[str, Any]] = []
    for row in rows:
        row_condition = row.get("condition_name")
        row_seed = row.get("seed")
        if row_condition == condition_name and row_seed is not None and int(row_seed) in seed_set:
            continue
        kept.append(row)
    return kept


def _append_model_mix_spec(path: Path, spec: dict[str, Any]) -> None:
    if not path.exists():
        path.write_text(json.dumps(spec, indent=2) + "\n")
        return

    existing = json.loads(path.read_text())
    if isinstance(existing, dict) and "runs" in existing:
        runs = list(existing["runs"])
    else:
        runs = [existing]
    if spec not in runs:
        runs.append(spec)
    path.write_text(json.dumps({"runs": runs}, indent=2) + "\n")


def _boosted_agent_diagnostics(result: dict[str, Any], boosted_agent_ids: Sequence[int]) -> list[dict[str, Any]]:
    active = set(int(agent_id) for agent_id in boosted_agent_ids)
    diagnostics = result.get("crop_diagnostics", [])
    return [row for row in diagnostics if int(row["agent_id"]) in active]


def _paired_condition_rows(
    condition_df: pd.DataFrame,
    *,
    left_condition: str,
    right_condition: str,
) -> list[dict[str, Any]]:
    left = condition_df[condition_df["condition_name"] == left_condition].copy()
    right = condition_df[condition_df["condition_name"] == right_condition].copy()
    merged = left.merge(
        right,
        on="seed",
        suffixes=("_left", "_right"),
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "comparison_name": f"{left_condition}_minus_{right_condition}",
                "left_condition": left_condition,
                "right_condition": right_condition,
                "seed": int(row["seed"]),
                "initial_accuracy_delta": float(row["initial_accuracy_left"] - row["initial_accuracy_right"]),
                "final_accuracy_delta": float(row["final_accuracy_left"] - row["final_accuracy_right"]),
                "collaboration_gain_delta": float(
                    row["collaboration_gain_over_initial_accuracy_left"]
                    - row["collaboration_gain_over_initial_accuracy_right"]
                ),
                "final_consensus_correct_delta": float(
                    float(bool(row["final_consensus_correct_left"]))
                    - float(bool(row["final_consensus_correct_right"]))
                ),
                "time_to_correct_consensus_delta": _subtract_or_none(
                    row["time_to_correct_consensus_left"],
                    row["time_to_correct_consensus_right"],
                ),
            }
        )
    return rows


def _summarize_paired_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    return {
        "comparison_name": rows[0]["comparison_name"],
        "left_condition": rows[0]["left_condition"],
        "right_condition": rows[0]["right_condition"],
        "n_paired_seeds": len(rows),
        "mean_initial_accuracy_delta": float(frame["initial_accuracy_delta"].mean()),
        "mean_final_accuracy_delta": float(frame["final_accuracy_delta"].mean()),
        "mean_collaboration_gain_delta": float(frame["collaboration_gain_delta"].mean()),
        "mean_final_consensus_correct_delta": float(frame["final_consensus_correct_delta"].mean()),
        "mean_time_to_correct_consensus_delta": (
            float(frame["time_to_correct_consensus_delta"].dropna().mean())
            if not frame["time_to_correct_consensus_delta"].dropna().empty
            else None
        ),
        "left_wins_final_accuracy_rate": float((frame["final_accuracy_delta"] > 0).mean()),
        "left_wins_consensus_rate": float((frame["final_consensus_correct_delta"] > 0).mean()),
    }


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    boosted_rows: list[dict[str, Any]],
) -> None:
    lines = ["# Flag Game Model-Mix Report", ""]
    lines.append("## Condition Summary")
    lines.append("")
    for row in summary_rows:
        lines.append(
            "- "
            f"{row['condition_label']}: final_accuracy_mean={_fmt(row['final_accuracy_mean'])}, "
            f"correct_consensus_rate={_fmt(row['correct_consensus_rate'])}, "
            f"collaboration_gain_mean={_fmt(row['collaboration_gain_mean'])}, "
            f"boosted_agent_compatible_country_count_mean={_fmt(row['boosted_agent_compatible_country_count_mean'])}"
        )
    lines.append("")
    lines.append("## Paired Deltas")
    lines.append("")
    for row in paired_rows:
        lines.append(
            "- "
            f"{row['comparison_name']}: mean_final_accuracy_delta={_fmt(row['mean_final_accuracy_delta'])}, "
            f"mean_collaboration_gain_delta={_fmt(row['mean_collaboration_gain_delta'])}, "
            f"mean_final_consensus_correct_delta={_fmt(row['mean_final_consensus_correct_delta'])}"
        )
    if boosted_rows:
        lines.append("")
        lines.append("## Boosted Agent Crop Diagnostics")
        lines.append("")
        for row in boosted_rows:
            lines.append(
                "- "
                f"seed={row['seed']}, agent={row['agent_id']}, compatible_country_count={row['compatible_country_count']}, "
                f"informativeness_label={row['informativeness_label']}, "
                f"informativeness_bits={_fmt(row['informativeness_bits'])}"
            )
    path.write_text("\n".join(lines) + "\n")


def _agent_model_signature(agent_models: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for model in agent_models:
        counts[str(model)] = counts.get(str(model), 0) + 1
    if len(counts) == 1:
        only_model = next(iter(counts))
        return f"all:{only_model}"
    return " + ".join(f"{model} x{count}" for model, count in counts.items())


def _mean_or_none(values: Sequence[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _subtract_or_none(left: Any, right: Any) -> float | None:
    if pd.isna(left) or pd.isna(right):
        return None
    return float(left - right)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"

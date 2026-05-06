from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

import pandas as pd

from nnd.backends.parsing import ParseError
from nnd.flag_game.analysis import summarize_initial_probe_rows
from nnd.flag_game.catalog import StripeFlag, get_country_pool
from nnd.flag_game.crops import (
    crop_image,
    mean_pairwise_overlap,
    sample_random_crops,
    scale_crop_box,
)
from nnd.flag_game.diagnostics import build_crop_compatibility_cache, describe_crop_informativeness
from nnd.flag_game.diagnostics import describe_crop_informativeness_fast
from nnd.flag_game.render import render_flag, save_png
from nnd.flag_game.runner import choose_default_backend, oracle_summary_from_crop_diagnostics
from nnd.flag_game_broadcast.analysis import summarize_broadcast_rounds
from nnd.flag_game_broadcast.backend import build_backend
from nnd.flag_game_broadcast.config import BroadcastFlagGameConfig, save_resolved_config
from nnd.flag_game_broadcast.parsing import BroadcastStatement, FinalDecision
from nnd.flag_game_broadcast.viz import (
    plot_country_share_stacked,
    plot_country_share_trajectories,
    plot_decision_memory_trajectories,
    plot_influence_family_progression,
    plot_influence_heatmap,
    plot_influence_received_ranking,
)


SKIP_PLOTS = os.environ.get("NND_SKIP_PLOTS") == "1"


@dataclass(frozen=True)
class BroadcastRecord:
    round: int
    agent_id: int
    model: str
    self_reported_model: str | None
    self_report_matches_assigned: bool
    m: int
    valid: bool
    country: str | None
    reason: str | None
    normalized_broadcast: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True)
class InitialDecisionRecord:
    round: int
    t: int
    agent_id: int
    model: str
    self_reported_model: str | None
    self_report_matches_assigned: bool
    m: int
    valid: bool
    country: str | None
    reason: str | None
    normalized_decision: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    round: int
    agent_id: int
    model: str
    valid: bool
    initial_country: str | None
    initial_correct: bool | None
    country: str | None
    reason: str | None
    influential_agent_ids: list[int]
    influential_models: list[str]
    changed_mind: bool | None
    correct: bool | None
    final_support_prestige_count: int | None
    final_support_comparison_count: int | None
    influential_prestige_count: int | None
    influential_comparison_count: int | None
    influential_prestige_fraction: float | None
    influential_comparison_fraction: float | None
    aligns_with_prestige_only_country: bool | None
    aligns_with_comparison_only_country: bool | None
    switched_toward_prestige_majority: bool | None
    switched_toward_comparison_majority: bool | None
    error: str | None = None


def _broadcast_row_from_initial_decision(
    initial_row: dict[str, Any],
    *,
    round_idx: int,
) -> dict[str, Any]:
    return asdict(
        BroadcastRecord(
            round=round_idx,
            agent_id=int(initial_row["agent_id"]),
            model=str(initial_row["model"]),
            self_reported_model=initial_row.get("self_reported_model"),
            self_report_matches_assigned=bool(initial_row.get("self_report_matches_assigned", False)),
            m=int(initial_row["m"]),
            valid=bool(initial_row.get("valid", False)),
            country=initial_row.get("country"),
            reason=initial_row.get("reason"),
            normalized_broadcast=initial_row.get("normalized_decision"),
            correct=initial_row.get("correct"),
            error=initial_row.get("error"),
        )
    )


def _resolve_agent_models(config: BroadcastFlagGameConfig) -> list[str]:
    if config.agent_models is not None:
        return list(config.agent_models)
    return [config.model for _ in range(config.N)]


def _ordered_model_counts(agent_models: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for model in agent_models:
        counts[model] = counts.get(model, 0) + 1
    return counts


def _agent_model_signature(agent_models: list[str]) -> str:
    counts = _ordered_model_counts(agent_models)
    if len(counts) == 1:
        return f"all:{agent_models[0]}"
    return " + ".join(f"{model} x{count}" for model, count in counts.items())


def _model_debug_dir(debug_root: Path, model: str) -> Path:
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._")
    return debug_root / (suffix or "model")


def _aggregate_backend_usage(agent_backends: list[Any]) -> dict[str, Any]:
    unique_backends: list[Any] = []
    seen_ids: set[int] = set()
    for backend in agent_backends:
        backend_id = id(backend)
        if backend_id in seen_ids:
            continue
        seen_ids.add(backend_id)
        unique_backends.append(backend)

    by_model: list[dict[str, Any]] = []
    total_api_call_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    estimated_cost_values: list[float] = []
    pricing_fully_known = True

    for backend in unique_backends:
        if not hasattr(backend, "usage_summary"):
            pricing_fully_known = False
            continue
        usage_summary = backend.usage_summary()
        by_model.append(usage_summary)
        total_api_call_count += int(usage_summary.get("api_call_count", 0) or 0)
        total_prompt_tokens += int(usage_summary.get("prompt_tokens", 0) or 0)
        total_completion_tokens += int(usage_summary.get("completion_tokens", 0) or 0)
        total_tokens += int(usage_summary.get("total_tokens", 0) or 0)
        if usage_summary.get("estimated_cost_usd") is not None:
            estimated_cost_values.append(float(usage_summary["estimated_cost_usd"]))
        if not bool(usage_summary.get("pricing_known", False)):
            pricing_fully_known = False

    return {
        "by_model": by_model,
        "api_call_count": total_api_call_count,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": sum(estimated_cost_values) if estimated_cost_values else None,
        "pricing_fully_known": pricing_fully_known,
    }


def _build_agent_backends(
    config: BroadcastFlagGameConfig,
    *,
    out_dir: Path,
    seed: int,
    agent_models: list[str],
    country_lookup: dict[str, Any],
) -> list[Any]:
    backend_cache: dict[str, Any] = {}
    for model in agent_models:
        if model in backend_cache:
            continue
        backend_cache[model] = build_backend(
            backend_name=config.backend,
            model=model,
            assigned_model_identity=model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            debug_dir=_model_debug_dir(out_dir / "debug", model),
            image_detail=config.image_detail,
            seed=seed,
            social_susceptibility=config.social_susceptibility,
            prompt_social_susceptibility=config.prompt_social_susceptibility,
            country_lookup=country_lookup,
        )
    return [backend_cache[model] for model in agent_models]


def _broadcast_line_from_row(row: dict[str, Any]) -> str:
    model = row.get("self_reported_model") or row.get("model")
    if bool(row.get("valid", False)) and row.get("country"):
        if row.get("reason"):
            return (
                f"agent {row['agent_id']} | model {model} | country {row['country']} | "
                f"reason {row['reason']}"
            )
        return f"agent {row['agent_id']} | model {model} | country {row['country']}"
    return f"agent {row['agent_id']} | model {model} | invalid broadcast"


def _majority_country(rows: list[dict[str, Any]], model_label: str) -> str | None:
    counts = Counter(
        str(row["country"])
        for row in rows
        if bool(row.get("valid", False))
        and row.get("country")
        and (row.get("self_reported_model") or row.get("model")) == model_label
    )
    if not counts:
        return None
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _exclusive_country_sets(
    rows: list[dict[str, Any]],
    prestige_label: str,
    comparison_label: str,
) -> tuple[set[str], set[str]]:
    prestige_countries = {
        str(row["country"])
        for row in rows
        if bool(row.get("valid", False))
        and row.get("country")
        and (row.get("self_reported_model") or row.get("model")) == prestige_label
    }
    comparison_countries = {
        str(row["country"])
        for row in rows
        if bool(row.get("valid", False))
        and row.get("country")
        and (row.get("self_reported_model") or row.get("model")) == comparison_label
    }
    return prestige_countries - comparison_countries, comparison_countries - prestige_countries


def _has_stable_final_consensus(
    per_round_df: pd.DataFrame,
    *,
    window: int,
    expected_valid_count: int,
) -> tuple[bool, str | None]:
    if window <= 0 or per_round_df.empty or len(per_round_df) < window:
        return False, None
    recent = per_round_df.tail(window)
    countries = list(recent["final_consensus_country"])
    if any(not isinstance(country, str) or not country for country in countries):
        return False, None
    if len(set(countries)) != 1:
        return False, None
    if not all(abs(float(value) - 1.0) < 1e-9 for value in recent["final_top1_share"]):
        return False, None
    if not all(int(value) == expected_valid_count for value in recent["final_valid_count"]):
        return False, None
    return True, countries[0]


def run_broadcast_flag_game_experiment(
    config: BroadcastFlagGameConfig,
    *,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, out_dir)

    rng = random.Random(seed)
    pool = get_country_pool(config.country_pool)
    country_lookup = {flag.country: flag for flag in pool}
    countries = [flag.country for flag in pool]
    if config.fixed_truth_country is not None:
        truth_flag = country_lookup[config.fixed_truth_country]
    else:
        truth_flag = rng.choice(pool)

    render_width = config.canvas_width * config.render_scale
    render_height = config.canvas_height * config.render_scale
    full_image = render_flag(truth_flag, width=render_width, height=render_height)

    assignments = sample_random_crops(
        canvas_width=config.canvas_width,
        canvas_height=config.canvas_height,
        tile_width=config.tile_width,
        tile_height=config.tile_height,
        n_agents=config.N,
        rng=rng,
        target_overlap=config.observation_overlap,
        search_trials=config.overlap_search_trials,
    )
    actual_overlap = mean_pairwise_overlap(assignments)
    scaled_assignments = [scale_crop_box(box, config.render_scale) for box in assignments]
    crop_images = [crop_image(full_image, box) for box in scaled_assignments]

    if config.output.save_crop_images:
        save_png(out_dir / "artifacts" / "truth_flag.png", full_image)
        for agent_id, image in enumerate(crop_images):
            save_png(out_dir / "artifacts" / f"agent_{agent_id:02d}_crop.png", image)

    agent_models = _resolve_agent_models(config)
    compute_crop_diagnostics = True
    compatibility_cache: dict[str, set[bytes]] = {}
    use_fast_crop_diagnostics = all(
        isinstance(flag, StripeFlag) and flag.triangle_color is None for flag in pool
    )
    if compute_crop_diagnostics and not use_fast_crop_diagnostics:
        compatibility_cache = build_crop_compatibility_cache(
            pool,
            canvas_width=config.canvas_width,
            canvas_height=config.canvas_height,
            tile_width=config.tile_width,
            tile_height=config.tile_height,
            render_scale=config.render_scale,
        )

    crop_diagnostics: list[dict[str, Any]] = []
    for agent_id, crop in enumerate(crop_images):
        if use_fast_crop_diagnostics:
            diagnostic = describe_crop_informativeness_fast(
                crop,
                country_order=countries,
                flags=pool,
            )
        else:
            diagnostic = describe_crop_informativeness(
                crop,
                country_order=countries,
                compatibility_cache=compatibility_cache,
            )
        crop_diagnostics.append(
            {
                "agent_id": agent_id,
                "model": agent_models[agent_id],
                "truth_country": truth_flag.country,
                "truth_compatible": truth_flag.country in diagnostic["compatible_countries"],
                **diagnostic,
            }
        )

    agent_backends = _build_agent_backends(
        config,
        out_dir=out_dir,
        seed=seed,
        agent_models=agent_models,
        country_lookup=country_lookup,
    )
    prepared_crops = [agent_backends[agent_id].prepare_crop(image) for agent_id, image in enumerate(crop_images)]
    oracle_summary = oracle_summary_from_crop_diagnostics(
        crop_diagnostics,
        countries=countries,
        truth_country=truth_flag.country,
    )

    broadcast_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    memory_snapshot_rows: list[dict[str, Any]] = []
    memories = [deque(maxlen=config.H) for _ in range(config.N)]
    executor = ThreadPoolExecutor(max_workers=config.agent_workers) if config.agent_workers > 1 else None
    stopped_early = False
    early_stop_country: str | None = None
    executed_rounds = 0
    t0_decision_rows: list[dict[str, Any]] = []

    t0_memory_snapshots = [list(memory) for memory in memories]

    def _t0_decision_one(agent_id: int) -> dict[str, Any]:
        backend = agent_backends[agent_id]
        try:
            statement = backend.broadcast_statement(
                countries=countries,
                prepared_crop=prepared_crops[agent_id],
                memory_lines=t0_memory_snapshots[agent_id],
                m=config.interaction_m,
            )
            return asdict(
                InitialDecisionRecord(
                    round=0,
                    t=0,
                    agent_id=agent_id,
                    model=agent_models[agent_id],
                    self_reported_model=statement.model_identity,
                    self_report_matches_assigned=statement.model_identity == agent_models[agent_id],
                    m=config.interaction_m,
                    valid=True,
                    country=statement.country,
                    reason=statement.reason,
                    normalized_decision=statement.normalized_broadcast(),
                    correct=statement.country == truth_flag.country,
                )
            )
        except ParseError as exc:
            return asdict(
                InitialDecisionRecord(
                    round=0,
                    t=0,
                    agent_id=agent_id,
                    model=agent_models[agent_id],
                    self_reported_model=None,
                    self_report_matches_assigned=False,
                    m=config.interaction_m,
                    valid=False,
                    country=None,
                    reason=None,
                    normalized_decision=None,
                    correct=None,
                    error=str(exc),
                )
            )

    try:
        if executor is None:
            t0_decision_rows = [_t0_decision_one(agent_id) for agent_id in range(config.N)]
        else:
            t0_decision_rows = list(executor.map(_t0_decision_one, range(config.N)))

        for round_idx in range(1, config.rounds + 1):
            executed_rounds = round_idx
            memory_snapshots = [list(memory) for memory in memories]

            def _broadcast_one(agent_id: int) -> dict[str, Any]:
                backend = agent_backends[agent_id]
                try:
                    statement = backend.broadcast_statement(
                        countries=countries,
                        prepared_crop=prepared_crops[agent_id],
                        memory_lines=memory_snapshots[agent_id],
                        m=config.interaction_m,
                    )
                    return asdict(
                        BroadcastRecord(
                            round=round_idx,
                            agent_id=agent_id,
                            model=agent_models[agent_id],
                            self_reported_model=statement.model_identity,
                            self_report_matches_assigned=statement.model_identity == agent_models[agent_id],
                            m=config.interaction_m,
                            valid=True,
                            country=statement.country,
                            reason=statement.reason,
                            normalized_broadcast=statement.normalized_broadcast(),
                            correct=statement.country == truth_flag.country,
                        )
                    )
                except ParseError as exc:
                    return asdict(
                        BroadcastRecord(
                            round=round_idx,
                            agent_id=agent_id,
                            model=agent_models[agent_id],
                            self_reported_model=None,
                            self_report_matches_assigned=False,
                            m=config.interaction_m,
                            valid=False,
                            country=None,
                            reason=None,
                            normalized_broadcast=None,
                            correct=None,
                            error=str(exc),
                        )
                    )

            if round_idx == 1:
                round_broadcast_rows = [
                    _broadcast_row_from_initial_decision(row, round_idx=round_idx)
                    for row in t0_decision_rows
                ]
            elif executor is None:
                round_broadcast_rows = [_broadcast_one(agent_id) for agent_id in range(config.N)]
            else:
                round_broadcast_rows = list(executor.map(_broadcast_one, range(config.N)))
            broadcast_rows.extend(round_broadcast_rows)

            round_broadcast_map = {int(row["agent_id"]): row for row in round_broadcast_rows}
            sorted_round_broadcast_rows = sorted(round_broadcast_rows, key=lambda item: int(item["agent_id"]))
            prestige_only_countries, comparison_only_countries = _exclusive_country_sets(
                round_broadcast_rows,
                config.prestige_model_label,
                config.comparison_model_label,
            )
            prestige_majority_country = _majority_country(round_broadcast_rows, config.prestige_model_label)
            comparison_majority_country = _majority_country(round_broadcast_rows, config.comparison_model_label)

            def _decision_one(agent_id: int) -> dict[str, Any]:
                backend = agent_backends[agent_id]
                initial_row = round_broadcast_map[agent_id]
                initial_country = str(initial_row["country"]) if initial_row.get("country") else None
                visible_broadcast_rows = [
                    row for row in sorted_round_broadcast_rows if int(row["agent_id"]) != agent_id
                ]
                visible_broadcast_lines = [
                    _broadcast_line_from_row(row)
                    for row in visible_broadcast_rows
                ]
                visible_agent_ids = {
                    int(row["agent_id"]) for row in visible_broadcast_rows if bool(row.get("valid", False))
                }
                try:
                    decision = backend.final_decision(
                        countries=countries,
                        prepared_crop=prepared_crops[agent_id],
                        memory_lines=memory_snapshots[agent_id],
                        round_broadcast_lines=visible_broadcast_lines,
                        m=config.interaction_m,
                        max_influential_agents=config.max_influential_agents,
                        valid_agent_ids=visible_agent_ids,
                    )
                    influential_ids = list(decision.influential_agent_ids)
                    influential_models = [
                        str(round_broadcast_map[idx].get("self_reported_model") or round_broadcast_map[idx]["model"])
                        for idx in influential_ids
                        if idx in round_broadcast_map
                    ]
                    influential_prestige_count = sum(
                        model == config.prestige_model_label for model in influential_models
                    )
                    influential_comparison_count = sum(
                        model == config.comparison_model_label for model in influential_models
                    )
                    total_influential = len(influential_models)
                    visible_support_rows = [
                        row
                        for row in visible_broadcast_rows
                        if bool(row.get("valid", False)) and row.get("country") == decision.country
                    ]
                    final_support_prestige_count = sum(
                        1
                        for row in visible_support_rows
                        if (row.get("self_reported_model") or row.get("model")) == config.prestige_model_label
                    )
                    final_support_comparison_count = sum(
                        1
                        for row in visible_support_rows
                        if (row.get("self_reported_model") or row.get("model")) == config.comparison_model_label
                    )
                    changed_mind = decision.country != initial_country if initial_country is not None else None
                    switched_toward_prestige_majority = bool(
                        changed_mind
                        and prestige_majority_country is not None
                        and prestige_majority_country != comparison_majority_country
                        and decision.country == prestige_majority_country
                    )
                    switched_toward_comparison_majority = bool(
                        changed_mind
                        and comparison_majority_country is not None
                        and prestige_majority_country != comparison_majority_country
                        and decision.country == comparison_majority_country
                    )
                    return asdict(
                        DecisionRecord(
                            round=round_idx,
                            agent_id=agent_id,
                            model=agent_models[agent_id],
                            valid=True,
                            initial_country=initial_country,
                            initial_correct=(initial_country == truth_flag.country) if initial_country else None,
                            country=decision.country,
                            reason=decision.reason,
                            influential_agent_ids=influential_ids,
                            influential_models=influential_models,
                            changed_mind=changed_mind,
                            correct=decision.country == truth_flag.country,
                            final_support_prestige_count=final_support_prestige_count,
                            final_support_comparison_count=final_support_comparison_count,
                            influential_prestige_count=influential_prestige_count,
                            influential_comparison_count=influential_comparison_count,
                            influential_prestige_fraction=(
                                influential_prestige_count / float(total_influential)
                                if total_influential > 0
                                else None
                            ),
                            influential_comparison_fraction=(
                                influential_comparison_count / float(total_influential)
                                if total_influential > 0
                                else None
                            ),
                            aligns_with_prestige_only_country=decision.country in prestige_only_countries,
                            aligns_with_comparison_only_country=decision.country in comparison_only_countries,
                            switched_toward_prestige_majority=switched_toward_prestige_majority,
                            switched_toward_comparison_majority=switched_toward_comparison_majority,
                        )
                    )
                except ParseError as exc:
                    return asdict(
                        DecisionRecord(
                            round=round_idx,
                            agent_id=agent_id,
                            model=agent_models[agent_id],
                            valid=False,
                            initial_country=initial_country,
                            initial_correct=(initial_country == truth_flag.country) if initial_country else None,
                            country=None,
                            reason=None,
                            influential_agent_ids=[],
                            influential_models=[],
                            changed_mind=None,
                            correct=None,
                            final_support_prestige_count=None,
                            final_support_comparison_count=None,
                            influential_prestige_count=None,
                            influential_comparison_count=None,
                            influential_prestige_fraction=None,
                            influential_comparison_fraction=None,
                            aligns_with_prestige_only_country=None,
                            aligns_with_comparison_only_country=None,
                            switched_toward_prestige_majority=None,
                            switched_toward_comparison_majority=None,
                            error=str(exc),
                        )
                    )

            if executor is None:
                round_decision_rows = [_decision_one(agent_id) for agent_id in range(config.N)]
            else:
                round_decision_rows = list(executor.map(_decision_one, range(config.N)))
            decision_rows.extend(round_decision_rows)

            if config.H > 0:
                for row in round_decision_rows:
                    if not bool(row.get("valid", False)) or not row.get("country"):
                        continue
                    memory_entry = str(row["country"])
                    if row.get("reason"):
                        memory_entry = f"{memory_entry} | {row['reason']}"
                    memories[int(row["agent_id"])].append(memory_entry)

            for agent_id in range(config.N):
                memory_lines = list(memories[agent_id])
                memory_snapshot_rows.append(
                    {
                        "round": round_idx,
                        "agent_id": agent_id,
                        "model": agent_models[agent_id],
                        "memory_length": len(memory_lines),
                        "memory_lines": memory_lines,
                    }
                )

            if config.early_stop_round_window > 0:
                partial_df, _ = summarize_broadcast_rounds(
                    broadcast_rows,
                    decision_rows,
                    countries=countries,
                    truth_country=truth_flag.country,
                    consensus_threshold=config.consensus_threshold,
                    polarization_threshold=config.polarization_threshold,
                )
                should_stop, stop_country = _has_stable_final_consensus(
                    partial_df,
                    window=config.early_stop_round_window,
                    expected_valid_count=config.N,
                )
                if should_stop:
                    stopped_early = True
                    early_stop_country = stop_country
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    per_round_df, summary = summarize_broadcast_rounds(
        broadcast_rows,
        decision_rows,
        countries=countries,
        truth_country=truth_flag.country,
        consensus_threshold=config.consensus_threshold,
        polarization_threshold=config.polarization_threshold,
    )
    t0_probe_like_rows = [
        {
            "t": 0,
            "agent_id": row["agent_id"],
            "model": row["model"],
            "valid": row["valid"],
            "country": row["country"],
            "correct": row["correct"],
        }
        for row in t0_decision_rows
    ]
    t0_decision_df, t0_summary = summarize_initial_probe_rows(
        t0_probe_like_rows,
        crop_diagnostics=crop_diagnostics,
        truth_country=truth_flag.country,
    )
    usage_summary = _aggregate_backend_usage(agent_backends)

    summary.update(
        {
            "truth_country": truth_flag.country,
            "country_pool": config.country_pool,
            "fixed_truth_country": config.fixed_truth_country,
            "countries": countries,
            "backend": config.backend,
            "model": config.model,
            "heterogeneous_models": len(set(agent_models)) > 1,
            "agent_model_signature": _agent_model_signature(agent_models),
            "n_unique_models": len(set(agent_models)),
            "prestige_model_label": config.prestige_model_label,
            "comparison_model_label": config.comparison_model_label,
            "N": config.N,
            "rounds": config.rounds,
            "H": config.H,
            "interaction_m": config.interaction_m,
            "social_susceptibility": config.social_susceptibility,
            "prompt_social_susceptibility": config.prompt_social_susceptibility,
            "max_influential_agents": config.max_influential_agents,
            "render_scale": config.render_scale,
            "image_detail": config.image_detail,
            "observation_overlap_target": config.observation_overlap,
            "observation_overlap_realized": actual_overlap,
            "executed_rounds": executed_rounds,
            "stopped_early": stopped_early,
            "early_stop_country": early_stop_country,
        }
    )
    summary.update(oracle_summary)
    summary.update(t0_summary)
    summary["oracle_gap_over_final_accuracy"] = float(summary["oracle_accuracy"] - summary["final_accuracy"])
    summary.update(
        {
            "api_call_count": usage_summary["api_call_count"],
            "prompt_tokens": usage_summary["prompt_tokens"],
            "completion_tokens": usage_summary["completion_tokens"],
            "total_tokens": usage_summary["total_tokens"],
            "estimated_cost_usd": usage_summary["estimated_cost_usd"],
            "pricing_fully_known": usage_summary["pricing_fully_known"],
        }
    )

    _write_jsonl(out_dir / "t0_decisions.jsonl", t0_decision_rows)
    _write_jsonl(out_dir / "broadcasts.jsonl", broadcast_rows)
    _write_jsonl(out_dir / "decisions.jsonl", decision_rows)
    _write_jsonl(out_dir / "memory_snapshots.jsonl", memory_snapshot_rows)
    per_round_df.to_csv(out_dir / "per_round.csv", index=False)
    t0_decision_df.to_csv(out_dir / "t0_decision_diagnostics.csv", index=False)
    t0_decision_df.to_csv(out_dir / "t0_broadcast_diagnostics.csv", index=False)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "api_usage_summary.json", "w") as handle:
        json.dump(usage_summary, handle, indent=2)
    with open(out_dir / "trial_manifest.json", "w") as handle:
        json.dump(
            {
                "truth_country": truth_flag.country,
                "countries": countries,
                "country_pool": config.country_pool,
                "fixed_truth_country": config.fixed_truth_country,
                "backend": config.backend,
                "default_model": config.model,
                "agent_model_signature": _agent_model_signature(agent_models),
                "agent_models": [{"agent_id": idx, "model": model} for idx, model in enumerate(agent_models)],
                "prestige_model_label": config.prestige_model_label,
                "comparison_model_label": config.comparison_model_label,
                "crop_diagnostics": crop_diagnostics,
                "canvas": {"width": config.canvas_width, "height": config.canvas_height},
                "render": {"scale": config.render_scale, "width": render_width, "height": render_height},
                "image_detail": config.image_detail,
                "social_susceptibility": config.social_susceptibility,
                "prompt_social_susceptibility": config.prompt_social_susceptibility,
                "max_influential_agents": config.max_influential_agents,
                "observation_overlap_target": config.observation_overlap,
                "observation_overlap_realized": actual_overlap,
                "oracle_summary": oracle_summary,
                "api_usage_summary": usage_summary,
                "tile": {"width": config.tile_width, "height": config.tile_height},
                "executed_rounds": executed_rounds,
                "stopped_early": stopped_early,
                "early_stop_country": early_stop_country,
                "memory_capacity": config.H,
                "assignments": [{"agent_id": idx, **box.to_dict()} for idx, box in enumerate(assignments)],
                "pixel_assignments": [{"agent_id": idx, **box.to_dict()} for idx, box in enumerate(scaled_assignments)],
            },
            handle,
            indent=2,
        )

    if config.output.make_plots and not SKIP_PLOTS:
        plot_country_share_trajectories(
            per_round_df,
            truth_country=truth_flag.country,
            out_dir=out_dir,
            t0_decision_rows=t0_decision_rows,
        )
        plot_country_share_stacked(
            per_round_df,
            truth_country=truth_flag.country,
            out_dir=out_dir,
            t0_decision_rows=t0_decision_rows,
        )
        plot_decision_memory_trajectories(
            decision_rows,
            truth_country=truth_flag.country,
            agent_models=agent_models,
            memory_capacity=config.H,
            out_dir=out_dir,
            t0_decision_rows=t0_decision_rows,
        )
        plot_influence_heatmap(
            decision_rows,
            agent_models=agent_models,
            out_dir=out_dir,
        )
        plot_influence_received_ranking(
            decision_rows,
            agent_models=agent_models,
            out_dir=out_dir,
        )
        plot_influence_family_progression(
            decision_rows,
            out_dir=out_dir,
        )

    return {
        "summary": summary,
        "per_round": per_round_df.to_dict(orient="records"),
        "t0_decisions": t0_decision_rows,
        "broadcasts": broadcast_rows,
        "decisions": decision_rows,
        "memory_snapshots": memory_snapshot_rows,
        "assignments": [box.to_dict() for box in assignments],
        "crop_diagnostics": crop_diagnostics,
        "t0_decision_diagnostics": t0_decision_df.to_dict(orient="records"),
        "t0_broadcast_diagnostics": t0_decision_df.to_dict(orient="records"),
    }


def run_broadcast_flag_game_batch(
    base_config: BroadcastFlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _run_seed(seed: int) -> tuple[int, dict[str, Any]]:
        trial_out = out_dir / f"seed_{seed:04d}"
        return seed, run_broadcast_flag_game_experiment(base_config, out_dir=trial_out, seed=seed)

    if base_config.seed_workers == 1:
        pairs = [_run_seed(seed) for seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=base_config.seed_workers) as executor:
            pairs = list(executor.map(_run_seed, seeds))
    pairs.sort(key=lambda item: item[0])
    results = [result for _, result in pairs]
    summary_rows = _load_seed_summary_rows(out_dir)
    if summary_rows:
        pd.DataFrame(summary_rows).sort_values("seed").to_csv(out_dir / "batch_summary.csv", index=False)
    else:
        pd.DataFrame([result["summary"] for result in results]).to_csv(out_dir / "batch_summary.csv", index=False)
    return results


def build_agent_model_assignment_by_count(
    *,
    n_agents: int,
    comparison_model: str,
    prestige_model: str,
    prestige_agent_count: int,
) -> list[str]:
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    if prestige_agent_count < 0 or prestige_agent_count > n_agents:
        raise ValueError(f"prestige_agent_count must be in [0, {n_agents}]")
    return [
        prestige_model if agent_id < prestige_agent_count else comparison_model
        for agent_id in range(n_agents)
    ]


def run_broadcast_flag_game_mix_sweep(
    base_config: BroadcastFlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
    prestige_counts: list[int],
    include_pure_controls: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_counts = sorted(set(int(value) for value in prestige_counts))
    condition_counts = list(requested_counts)
    if include_pure_controls:
        condition_counts = sorted(set(condition_counts + [0, base_config.N]))

    condition_rows: list[dict[str, Any]] = _load_existing_rows(out_dir / "mix_condition_results.csv")
    skipped_conditions: list[str] = []

    for prestige_count in condition_counts:
        agent_models = build_agent_model_assignment_by_count(
            n_agents=base_config.N,
            comparison_model=comparison_model,
            prestige_model=prestige_model,
            prestige_agent_count=prestige_count,
        )
        if prestige_count == 0:
            condition_name = f"all_{_slug(comparison_model)}"
        elif prestige_count == base_config.N:
            condition_name = f"all_{_slug(prestige_model)}"
        else:
            condition_name = f"mix_{_slug(comparison_model)}_plus_{prestige_count}_{_slug(prestige_model)}"
        condition_dir = out_dir / condition_name
        condition_config = base_config.model_copy(
            update={
                "model": comparison_model,
                "agent_models": agent_models,
                "comparison_model_label": comparison_model,
                "prestige_model_label": prestige_model,
            }
        )
        if _condition_complete(condition_rows, condition_name, seeds):
            skipped_conditions.append(condition_name)
            continue

        condition_rows = _drop_condition_seed_rows(condition_rows, condition_name, seeds)
        results = run_broadcast_flag_game_batch(condition_config, out_dir=condition_dir, seeds=seeds)
        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            row.update(
                {
                    "seed": seed,
                    "condition_name": condition_name,
                    "condition_dir": str(condition_dir),
                    "prestige_agent_count": prestige_count,
                    "prestige_agent_fraction": prestige_count / float(base_config.N),
                    "comparison_model": comparison_model,
                    "prestige_model": prestige_model,
                    "agent_model_signature": _agent_model_signature(agent_models),
                }
            )
            condition_rows.append(row)

    condition_df = pd.DataFrame(condition_rows)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["condition_name", "seed"]).reset_index(drop=True)
    summary_df = pd.DataFrame(_aggregate_mix_condition_rows(condition_df))
    if not summary_df.empty:
        summary_df = summary_df.sort_values("prestige_agent_count").reset_index(drop=True)
    condition_df.to_csv(out_dir / "mix_condition_results.csv", index=False)
    summary_df.to_csv(out_dir / "mix_summary.csv", index=False)
    _write_mix_report(
        out_dir / "mix_report.md",
        summary_rows=summary_df.to_dict(orient="records"),
        comparison_model=comparison_model,
        prestige_model=prestige_model,
    )
    return {
        "condition_results": condition_df.to_dict(orient="records"),
        "summary": summary_df.to_dict(orient="records"),
        "skipped_conditions": skipped_conditions,
    }


def _load_seed_summary_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(out_dir.glob("seed_*")):
        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            continue
        match = re.search(r"seed_(\d+)", seed_dir.name)
        seed = int(match.group(1)) if match else None
        row = json.loads(summary_path.read_text())
        if seed is not None:
            row["seed"] = seed
        rows.append(row)
    return rows


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
        if row.get("condition_name") == condition_name
        and row.get("seed") is not None
        and not pd.isna(row.get("seed"))
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
        if row_condition == condition_name and row_seed is not None and not pd.isna(row_seed) and int(row_seed) in seed_set:
            continue
        kept.append(row)
    return kept


def _aggregate_mix_condition_rows(condition_df: pd.DataFrame) -> list[dict[str, Any]]:
    if condition_df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, group in condition_df.groupby("condition_name", sort=False):
        first_row = group.iloc[0]
        rows.append(
            {
                "condition_name": first_row["condition_name"],
                "condition_dir": first_row["condition_dir"],
                "prestige_agent_count": int(first_row["prestige_agent_count"]),
                "prestige_agent_fraction": float(first_row["prestige_agent_fraction"]),
                "comparison_model": first_row["comparison_model"],
                "prestige_model": first_row["prestige_model"],
                "agent_model_signature": first_row["agent_model_signature"],
                "n_trials": len(group),
                "initial_accuracy_mean": float(group["initial_accuracy"].mean()) if not group.empty else None,
                "final_accuracy_mean": float(group["final_accuracy"].mean()) if not group.empty else None,
                "final_vote_accuracy_rate": float(group["final_vote_accuracy"].mean())
                if not group.empty and "final_vote_accuracy" in group
                else None,
                "correct_consensus_rate": float(group["final_consensus_correct"].mean()) if not group.empty else 0.0,
                "collaboration_gain_mean": (
                    float(group["collaboration_gain_over_initial_accuracy"].mean()) if not group.empty else None
                ),
                "mean_changed_mind_fraction": (
                    float(group["mean_changed_mind_fraction"].mean()) if not group.empty else None
                ),
                "switch_delta_toward_prestige_mean": (
                    float(group["switch_delta_toward_prestige"].mean()) if not group.empty else None
                ),
                "decision_alignment_prestige_only_rate_mean": (
                    float(group["decision_alignment_prestige_only_rate"].mean()) if not group.empty else None
                ),
                "decision_alignment_comparison_only_rate_mean": (
                    float(group["decision_alignment_comparison_only_rate"].mean()) if not group.empty else None
                ),
                "estimated_cost_usd_mean": (
                    float(group["estimated_cost_usd"].dropna().mean())
                    if not group.empty and not group["estimated_cost_usd"].dropna().empty
                    else None
                ),
            }
        )
    return rows


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _write_mix_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    comparison_model: str,
    prestige_model: str,
) -> None:
    lines = [
        "# Broadcast Flag-Game Mix Sweep",
        "",
        f"- comparison model: `{comparison_model}`",
        f"- prestige model: `{prestige_model}`",
        "",
        "| condition | prestige count | final acc mean | correct consensus rate | switch delta toward prestige | prestige-only alignment | comparison-only alignment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['condition_name']} | "
            f"{row['prestige_agent_count']} | "
            f"{_fmt(row.get('final_accuracy_mean'))} | "
            f"{_fmt(row.get('correct_consensus_rate'))} | "
            f"{_fmt(row.get('switch_delta_toward_prestige_mean'))} | "
            f"{_fmt(row.get('decision_alignment_prestige_only_rate_mean'))} | "
            f"{_fmt(row.get('decision_alignment_comparison_only_rate_mean'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

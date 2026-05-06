from __future__ import annotations

from collections import deque
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
from nnd.flag_game.runner import oracle_summary_from_crop_diagnostics
from nnd.flag_game_org.analysis import summarize_org_rounds
from nnd.flag_game_org.backend import build_backend
from nnd.flag_game_org.config import OrgFlagGameConfig, save_resolved_config
from nnd.flag_game_org.viz import (
    plot_country_share_stacked,
    plot_country_share_trajectories,
    plot_decision_memory_trajectories,
)


SKIP_PLOTS = os.environ.get("NND_SKIP_PLOTS") == "1"


@dataclass(frozen=True)
class ObservationRecord:
    round: int
    agent_id: int
    model: str
    m: int
    valid: bool
    country: str | None
    reason: str | None
    normalized_statement: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True)
class InitialObservationRecord:
    round: int
    t: int
    agent_id: int
    model: str
    m: int
    valid: bool
    country: str | None
    reason: str | None
    normalized_statement: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    round: int
    agent_id: int
    model: str
    valid: bool
    country: str | None
    reason: str | None
    correct: bool | None
    error: str | None = None


def _observation_row_from_initial_observation(
    initial_row: dict[str, Any],
    *,
    round_idx: int,
) -> dict[str, Any]:
    return asdict(
        ObservationRecord(
            round=round_idx,
            agent_id=int(initial_row["agent_id"]),
            model=str(initial_row["model"]),
            m=int(initial_row["m"]),
            valid=bool(initial_row.get("valid", False)),
            country=initial_row.get("country"),
            reason=initial_row.get("reason"),
            normalized_statement=initial_row.get("normalized_statement"),
            correct=initial_row.get("correct"),
            error=initial_row.get("error"),
        )
    )


def _observer_agent_ids(config: OrgFlagGameConfig) -> list[int]:
    return [agent_id for agent_id in range(_total_agent_count(config)) if agent_id != config.aggregator_agent_id]


def _total_agent_count(config: OrgFlagGameConfig) -> int:
    return config.N + 1


def _resolve_agent_models(config: OrgFlagGameConfig) -> list[str]:
    if config.agent_models is not None:
        return list(config.agent_models)
    return [config.model for _ in range(_total_agent_count(config))]


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
    config: OrgFlagGameConfig,
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
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            debug_dir=_model_debug_dir(out_dir / "debug", model),
            image_detail=config.image_detail,
            seed=seed,
            country_lookup=country_lookup,
        )
    return [backend_cache[model] for model in agent_models]


def _observation_line_from_row(row: dict[str, Any]) -> str:
    if bool(row.get("valid", False)) and row.get("country"):
        return json.dumps(
            {
                "country": row["country"],
                "reason": row.get("reason") or "",
            },
            ensure_ascii=True,
        )
    return json.dumps({"valid": False}, ensure_ascii=True)


def _has_stable_manager_decision(
    decision_rows: list[dict[str, Any]],
    *,
    window: int,
) -> tuple[bool, str | None]:
    if window <= 0 or len(decision_rows) < window:
        return False, None

    recent = decision_rows[-window:]
    countries: list[str] = []
    for row in recent:
        country = row.get("country")
        if not bool(row.get("valid", False)) or not isinstance(country, str) or not country:
            return False, None
        countries.append(country)
    if len(set(countries)) != 1:
        return False, None
    return True, countries[0]


def run_org_flag_game_experiment(
    config: OrgFlagGameConfig,
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

    aggregator_id = config.aggregator_agent_id
    observer_ids = _observer_agent_ids(config)
    total_agent_count = _total_agent_count(config)
    render_width = config.canvas_width * config.render_scale
    render_height = config.canvas_height * config.render_scale
    full_image = render_flag(truth_flag, width=render_width, height=render_height)

    assignments = sample_random_crops(
        canvas_width=config.canvas_width,
        canvas_height=config.canvas_height,
        tile_width=config.tile_width,
        tile_height=config.tile_height,
        n_agents=len(observer_ids),
        rng=rng,
        target_overlap=config.observation_overlap,
        search_trials=config.overlap_search_trials,
    )
    actual_overlap = mean_pairwise_overlap(assignments)
    assignment_by_agent = dict(zip(observer_ids, assignments, strict=True))
    scaled_assignment_by_agent = {
        agent_id: scale_crop_box(box, config.render_scale)
        for agent_id, box in assignment_by_agent.items()
    }
    crop_images = {
        agent_id: crop_image(full_image, scaled_box)
        for agent_id, scaled_box in scaled_assignment_by_agent.items()
    }

    if config.output.save_crop_images:
        save_png(out_dir / "artifacts" / "truth_flag.png", full_image)
        for agent_id, image in crop_images.items():
            save_png(out_dir / "artifacts" / f"agent_{agent_id:02d}_crop.png", image)

    agent_models = _resolve_agent_models(config)
    compatibility_cache: dict[str, set[bytes]] = {}
    use_fast_crop_diagnostics = all(
        isinstance(flag, StripeFlag) and flag.triangle_color is None for flag in pool
    )
    if not use_fast_crop_diagnostics:
        compatibility_cache = build_crop_compatibility_cache(
            pool,
            canvas_width=config.canvas_width,
            canvas_height=config.canvas_height,
            tile_width=config.tile_width,
            tile_height=config.tile_height,
            render_scale=config.render_scale,
        )

    crop_diagnostics: list[dict[str, Any]] = []
    for agent_id in observer_ids:
        crop = crop_images[agent_id]
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
                "role": "observer",
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
    prepared_crops = {
        agent_id: agent_backends[agent_id].prepare_crop(crop_images[agent_id])
        for agent_id in observer_ids
    }
    oracle_summary = oracle_summary_from_crop_diagnostics(
        crop_diagnostics,
        countries=countries,
        truth_country=truth_flag.country,
    )

    observation_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    memory_snapshot_rows: list[dict[str, Any]] = []
    shared_memory: deque[str] = deque(maxlen=config.H)
    executor = ThreadPoolExecutor(max_workers=config.agent_workers) if config.agent_workers > 1 else None
    stopped_early = False
    early_stop_country: str | None = None
    executed_rounds = 0
    t0_observation_rows: list[dict[str, Any]] = []

    def _observation_one(agent_id: int, *, round_idx: int, memory_snapshot: list[str]) -> dict[str, Any]:
        backend = agent_backends[agent_id]
        try:
            statement = backend.observer_statement(
                countries=countries,
                prepared_crop=prepared_crops[agent_id],
                memory_lines=memory_snapshot,
                m=config.interaction_m,
            )
            return asdict(
                ObservationRecord(
                    round=round_idx,
                    agent_id=agent_id,
                    model=agent_models[agent_id],
                    m=config.interaction_m,
                    valid=True,
                    country=statement.country,
                    reason=statement.reason,
                    normalized_statement=statement.normalized_statement(),
                    correct=statement.country == truth_flag.country,
                )
            )
        except ParseError as exc:
            return asdict(
                ObservationRecord(
                    round=round_idx,
                    agent_id=agent_id,
                    model=agent_models[agent_id],
                    m=config.interaction_m,
                    valid=False,
                    country=None,
                    reason=None,
                    normalized_statement=None,
                    correct=None,
                    error=str(exc),
                )
            )

    def _initial_observation_row(row: dict[str, Any]) -> dict[str, Any]:
        return asdict(
            InitialObservationRecord(
                round=0,
                t=0,
                agent_id=int(row["agent_id"]),
                model=str(row["model"]),
                m=int(row["m"]),
                valid=bool(row.get("valid", False)),
                country=row.get("country"),
                reason=row.get("reason"),
                normalized_statement=row.get("normalized_statement"),
                correct=row.get("correct"),
                error=row.get("error"),
            )
        )

    try:
        t0_memory_snapshot: list[str] = []
        if executor is None:
            t0_observation_rows = [
                _initial_observation_row(
                    _observation_one(agent_id, round_idx=0, memory_snapshot=t0_memory_snapshot)
                )
                for agent_id in observer_ids
            ]
        else:
            t0_observation_rows = [
                _initial_observation_row(row)
                for row in executor.map(
                    lambda agent_id: _observation_one(
                        agent_id,
                        round_idx=0,
                        memory_snapshot=t0_memory_snapshot,
                    ),
                    observer_ids,
                )
            ]

        for round_idx in range(1, config.rounds + 1):
            executed_rounds = round_idx
            memory_snapshot = list(shared_memory)

            if round_idx == 1:
                round_observation_rows = [
                    _observation_row_from_initial_observation(row, round_idx=round_idx)
                    for row in t0_observation_rows
                ]
            elif executor is None:
                round_observation_rows = [
                    _observation_one(agent_id, round_idx=round_idx, memory_snapshot=memory_snapshot)
                    for agent_id in observer_ids
                ]
            else:
                round_observation_rows = list(
                    executor.map(
                        lambda agent_id: _observation_one(
                            agent_id,
                            round_idx=round_idx,
                            memory_snapshot=memory_snapshot,
                        ),
                        observer_ids,
                    )
                )
            observation_rows.extend(round_observation_rows)

            sorted_observation_rows = sorted(round_observation_rows, key=lambda item: int(item["agent_id"]))

            observer_statement_lines = [
                _observation_line_from_row(row)
                for row in sorted_observation_rows
            ]

            try:
                decision = agent_backends[aggregator_id].organization_decision(
                    countries=countries,
                    memory_lines=memory_snapshot,
                    observer_statement_lines=observer_statement_lines,
                    m=config.interaction_m,
                )
                round_decision_rows = [
                    asdict(
                        DecisionRecord(
                            round=round_idx,
                            agent_id=aggregator_id,
                            model=agent_models[aggregator_id],
                            valid=True,
                            country=decision.country,
                            reason=decision.reason,
                            correct=decision.country == truth_flag.country,
                        )
                    )
                ]
            except ParseError as exc:
                round_decision_rows = [
                    asdict(
                        DecisionRecord(
                            round=round_idx,
                            agent_id=aggregator_id,
                            model=agent_models[aggregator_id],
                            valid=False,
                            country=None,
                            reason=None,
                            correct=None,
                            error=str(exc),
                        )
                    )
                ]
            decision_rows.extend(round_decision_rows)

            if config.H > 0:
                for row in round_decision_rows:
                    if not bool(row.get("valid", False)) or not row.get("country"):
                        continue
                    memory_entry = str(row["country"])
                    if row.get("reason"):
                        memory_entry = f"{memory_entry} | {row['reason']}"
                    shared_memory.append(memory_entry)

            for agent_id in range(total_agent_count):
                memory_lines = list(shared_memory)
                memory_snapshot_rows.append(
                    {
                        "round": round_idx,
                        "agent_id": agent_id,
                        "role": "aggregator" if agent_id == aggregator_id else "observer",
                        "model": agent_models[agent_id],
                        "memory_length": len(memory_lines),
                        "memory_lines": memory_lines,
                    }
                )

            if config.early_stop_round_window > 0:
                should_stop, stop_country = _has_stable_manager_decision(
                    decision_rows,
                    window=config.early_stop_round_window,
                )
                if should_stop:
                    stopped_early = True
                    early_stop_country = stop_country
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    per_round_df, summary = summarize_org_rounds(
        observation_rows,
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
        for row in t0_observation_rows
    ]
    t0_observation_df, t0_summary = summarize_initial_probe_rows(
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
            "N": config.N,
            "total_agent_count": total_agent_count,
            "observer_count": len(observer_ids),
            "aggregator_agent_id": aggregator_id,
            "observer_agent_ids": observer_ids,
            "rounds": config.rounds,
            "round_index_start": 0,
            "round0_is_initial_observer_iq": True,
            "round0_has_manager_decision": False,
            "manager_round_index_start": 1,
            "shared_manager_memory": True,
            "H": config.H,
            "interaction_m": config.interaction_m,
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

    _write_jsonl(out_dir / "t0_observations.jsonl", t0_observation_rows)
    _write_jsonl(out_dir / "observations.jsonl", observation_rows)
    _write_jsonl(out_dir / "decisions.jsonl", decision_rows)
    _write_jsonl(out_dir / "memory_snapshots.jsonl", memory_snapshot_rows)
    per_round_df.to_csv(out_dir / "per_round.csv", index=False)
    t0_observation_df.to_csv(out_dir / "t0_observation_diagnostics.csv", index=False)
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
                "agent_models": [
                    {
                        "agent_id": idx,
                        "model": model,
                        "role": "aggregator" if idx == aggregator_id else "observer",
                    }
                    for idx, model in enumerate(agent_models)
                ],
                "aggregator_agent_id": aggregator_id,
                "observer_agent_ids": observer_ids,
                "total_agent_count": total_agent_count,
                "round_index_start": 0,
                "round0_is_initial_observer_iq": True,
                "round0_has_manager_decision": False,
                "manager_round_index_start": 1,
                "shared_manager_memory": True,
                "crop_diagnostics": crop_diagnostics,
                "canvas": {"width": config.canvas_width, "height": config.canvas_height},
                "render": {"scale": config.render_scale, "width": render_width, "height": render_height},
                "image_detail": config.image_detail,
                "observation_overlap_target": config.observation_overlap,
                "observation_overlap_realized": actual_overlap,
                "oracle_summary": oracle_summary,
                "api_usage_summary": usage_summary,
                "tile": {"width": config.tile_width, "height": config.tile_height},
                "executed_rounds": executed_rounds,
                "stopped_early": stopped_early,
                "early_stop_country": early_stop_country,
                "memory_capacity": config.H,
                "assignments": [
                    {"agent_id": agent_id, **box.to_dict()}
                    for agent_id, box in assignment_by_agent.items()
                ],
                "pixel_assignments": [
                    {"agent_id": agent_id, **box.to_dict()}
                    for agent_id, box in scaled_assignment_by_agent.items()
                ],
            },
            handle,
            indent=2,
        )

    if config.output.make_plots and not SKIP_PLOTS:
        plot_country_share_trajectories(
            per_round_df,
            truth_country=truth_flag.country,
            out_dir=out_dir,
            t0_decision_rows=t0_observation_rows,
        )
        plot_country_share_stacked(
            per_round_df,
            truth_country=truth_flag.country,
            out_dir=out_dir,
            t0_decision_rows=t0_observation_rows,
        )
        plot_decision_memory_trajectories(
            decision_rows,
            truth_country=truth_flag.country,
            agent_models=agent_models,
            memory_capacity=config.H,
            out_dir=out_dir,
            t0_decision_rows=t0_observation_rows,
        )

    return {
        "summary": summary,
        "per_round": per_round_df.to_dict(orient="records"),
        "t0_observations": t0_observation_rows,
        "observations": observation_rows,
        "decisions": decision_rows,
        "memory_snapshots": memory_snapshot_rows,
        "assignments": [
            {"agent_id": agent_id, **box.to_dict()}
            for agent_id, box in assignment_by_agent.items()
        ],
        "crop_diagnostics": crop_diagnostics,
        "t0_observation_diagnostics": t0_observation_df.to_dict(orient="records"),
    }


def run_org_flag_game_batch(
    base_config: OrgFlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _run_seed(seed: int) -> tuple[int, dict[str, Any]]:
        trial_out = out_dir / f"seed_{seed:04d}"
        return seed, run_org_flag_game_experiment(base_config, out_dir=trial_out, seed=seed)

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


def build_role_model_assignment(
    *,
    n_observers: int,
    aggregator_agent_id: int,
    manager_model: str,
    observer_model: str,
) -> list[str]:
    total_agents = n_observers + 1
    if n_observers < 1:
        raise ValueError("n_observers must be >= 1")
    if aggregator_agent_id < 0 or aggregator_agent_id >= total_agents:
        raise ValueError("aggregator_agent_id must be in [0, n_observers]")
    agent_models = [observer_model for _ in range(total_agents)]
    agent_models[aggregator_agent_id] = manager_model
    return agent_models


def run_org_flag_game_mix_sweep(
    base_config: OrgFlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
    prestige_counts: list[int],
    include_pure_controls: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_agents = _total_agent_count(base_config)
    requested_counts = sorted(set(int(value) for value in prestige_counts))
    condition_counts = list(requested_counts)
    if include_pure_controls:
        condition_counts = sorted(set(condition_counts + [0, total_agents]))

    condition_rows: list[dict[str, Any]] = _load_existing_rows(out_dir / "mix_condition_results.csv")
    skipped_conditions: list[str] = []

    for prestige_count in condition_counts:
        agent_models = build_agent_model_assignment_by_count(
            n_agents=total_agents,
            comparison_model=comparison_model,
            prestige_model=prestige_model,
            prestige_agent_count=prestige_count,
        )
        if prestige_count == 0:
            condition_name = f"all_{_slug(comparison_model)}"
        elif prestige_count == total_agents:
            condition_name = f"all_{_slug(prestige_model)}"
        else:
            condition_name = f"mix_{_slug(comparison_model)}_plus_{prestige_count}_{_slug(prestige_model)}"
        condition_dir = out_dir / condition_name
        condition_config = base_config.model_copy(
            update={
                "model": comparison_model,
                "agent_models": agent_models,
            }
        )
        if _condition_complete(condition_rows, condition_name, seeds):
            skipped_conditions.append(condition_name)
            continue

        condition_rows = _drop_condition_seed_rows(condition_rows, condition_name, seeds)
        results = run_org_flag_game_batch(condition_config, out_dir=condition_dir, seeds=seeds)
        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            row.update(
                {
                    "seed": seed,
                    "condition_name": condition_name,
                    "condition_dir": str(condition_dir),
                    "prestige_agent_count": prestige_count,
                    "prestige_agent_fraction": prestige_count / float(total_agents),
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


def run_org_flag_game_role_mix_comparison(
    base_config: OrgFlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _role_mix_conditions(
        comparison_model=comparison_model,
        prestige_model=prestige_model,
    )
    condition_rows: list[dict[str, Any]] = _load_existing_rows(out_dir / "role_mix_condition_results.csv")
    skipped_conditions: list[str] = []

    for condition in conditions:
        condition_name = str(condition["condition_name"])
        manager_model = str(condition["manager_model"])
        observer_model = str(condition["observer_model"])
        agent_models = build_role_model_assignment(
            n_observers=base_config.N,
            aggregator_agent_id=base_config.aggregator_agent_id,
            manager_model=manager_model,
            observer_model=observer_model,
        )
        condition_dir = out_dir / condition_name
        condition_config = base_config.model_copy(
            update={
                "model": manager_model,
                "agent_models": agent_models,
            }
        )
        if _condition_complete(condition_rows, condition_name, seeds):
            skipped_conditions.append(condition_name)
            continue

        condition_rows = _drop_condition_seed_rows(condition_rows, condition_name, seeds)
        results = run_org_flag_game_batch(condition_config, out_dir=condition_dir, seeds=seeds)
        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            row.update(
                {
                    "seed": seed,
                    "condition_name": condition_name,
                    "condition_label": condition["condition_label"],
                    "condition_short_label": condition["condition_short_label"],
                    "condition_order": condition["condition_order"],
                    "condition_kind": condition["condition_kind"],
                    "condition_dir": str(condition_dir),
                    "manager_model": manager_model,
                    "observer_model": observer_model,
                    "comparison_model": comparison_model,
                    "prestige_model": prestige_model,
                    "manager_is_prestige": manager_model == prestige_model,
                    "observer_is_prestige": observer_model == prestige_model,
                    "agent_model_signature": _agent_model_signature(agent_models),
                }
            )
            condition_rows.append(row)

    condition_df = pd.DataFrame(condition_rows)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["condition_order", "seed"]).reset_index(drop=True)
    summary_df = pd.DataFrame(_aggregate_role_mix_condition_rows(condition_df))
    if not summary_df.empty:
        summary_df = summary_df.sort_values("condition_order").reset_index(drop=True)
    condition_df.to_csv(out_dir / "role_mix_condition_results.csv", index=False)
    summary_df.to_csv(out_dir / "role_mix_summary.csv", index=False)
    _write_role_mix_report(
        out_dir / "role_mix_report.md",
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
                "estimated_cost_usd_mean": (
                    float(group["estimated_cost_usd"].dropna().mean())
                    if not group.empty and not group["estimated_cost_usd"].dropna().empty
                    else None
                ),
            }
        )
    return rows


def _role_mix_conditions(*, comparison_model: str, prestige_model: str) -> list[dict[str, Any]]:
    comparison_slug = _slug(comparison_model)
    prestige_slug = _slug(prestige_model)
    comparison_short = _short_model_label(comparison_model)
    prestige_short = _short_model_label(prestige_model)
    return [
        {
            "condition_name": f"all_{comparison_slug}",
            "condition_label": f"all {comparison_model}",
            "condition_short_label": f"all\n{comparison_short}",
            "condition_order": 0,
            "condition_kind": "all_comparison",
            "manager_model": comparison_model,
            "observer_model": comparison_model,
        },
        {
            "condition_name": f"all_{prestige_slug}",
            "condition_label": f"all {prestige_model}",
            "condition_short_label": f"all\n{prestige_short}",
            "condition_order": 1,
            "condition_kind": "all_prestige",
            "manager_model": prestige_model,
            "observer_model": prestige_model,
        },
        {
            "condition_name": f"manager_{prestige_slug}_observers_{comparison_slug}",
            "condition_label": f"{prestige_model} manager, {comparison_model} observers",
            "condition_short_label": f"{prestige_short} mgr\n{comparison_short} obs",
            "condition_order": 2,
            "condition_kind": "prestige_manager_comparison_observers",
            "manager_model": prestige_model,
            "observer_model": comparison_model,
        },
        {
            "condition_name": f"manager_{comparison_slug}_observers_{prestige_slug}",
            "condition_label": f"{comparison_model} manager, {prestige_model} observers",
            "condition_short_label": f"{comparison_short} mgr\n{prestige_short} obs",
            "condition_order": 3,
            "condition_kind": "comparison_manager_prestige_observers",
            "manager_model": comparison_model,
            "observer_model": prestige_model,
        },
    ]


def _aggregate_role_mix_condition_rows(condition_df: pd.DataFrame) -> list[dict[str, Any]]:
    if condition_df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, group in condition_df.groupby("condition_name", sort=False):
        first_row = group.iloc[0]
        final_correct = group["final_consensus_correct"].map(_coerce_bool)
        rows.append(
            {
                "condition_name": first_row["condition_name"],
                "condition_label": first_row["condition_label"],
                "condition_short_label": first_row["condition_short_label"],
                "condition_order": int(first_row["condition_order"]),
                "condition_kind": first_row["condition_kind"],
                "condition_dir": first_row["condition_dir"],
                "manager_model": first_row["manager_model"],
                "observer_model": first_row["observer_model"],
                "comparison_model": first_row["comparison_model"],
                "prestige_model": first_row["prestige_model"],
                "manager_is_prestige": bool(first_row["manager_is_prestige"]),
                "observer_is_prestige": bool(first_row["observer_is_prestige"]),
                "agent_model_signature": first_row["agent_model_signature"],
                "n_trials": len(group),
                "seed_min": int(group["seed"].min()) if "seed" in group else None,
                "seed_max": int(group["seed"].max()) if "seed" in group else None,
                "initial_accuracy_mean": _series_mean(group, "initial_accuracy"),
                "initial_accuracy_std": _series_std(group, "initial_accuracy"),
                "final_accuracy_mean": _series_mean(group, "final_accuracy"),
                "final_accuracy_std": _series_std(group, "final_accuracy"),
                "final_vote_accuracy_rate": _series_mean(group, "final_vote_accuracy"),
                "manager_correct_rate": float(final_correct.mean()) if not final_correct.empty else 0.0,
                "collaboration_gain_mean": _series_mean(group, "collaboration_gain_over_initial_accuracy"),
                "collaboration_gain_std": _series_std(group, "collaboration_gain_over_initial_accuracy"),
                "oracle_accuracy_mean": _series_mean(group, "oracle_accuracy"),
                "oracle_gap_mean": _series_mean(group, "oracle_gap_over_final_accuracy"),
                "initial_crop_compatibility_rate_mean": _series_mean(group, "initial_crop_compatibility_rate"),
                "executed_rounds_mean": _series_mean(group, "executed_rounds"),
                "estimated_cost_usd_mean": _series_mean(group, "estimated_cost_usd"),
                "estimated_cost_usd_total": _series_sum(group, "estimated_cost_usd"),
            }
        )
    return rows


def _series_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _series_std(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else None
    return float(values.std(ddof=1))


def _series_sum(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no", "", "nan", "none"}:
        return False
    return bool(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _short_model_label(model: str) -> str:
    if model.startswith("gpt-"):
        return model.removeprefix("gpt-")
    return model


def _write_mix_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    comparison_model: str,
    prestige_model: str,
) -> None:
    lines = [
        "# Org Flag-Game Mix Sweep",
        "",
        f"- comparison model: `{comparison_model}`",
        f"- prestige model: `{prestige_model}`",
        "",
        "| condition | prestige count | final acc mean | correct consensus rate | collaboration gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['condition_name']} | "
            f"{row['prestige_agent_count']} | "
            f"{_fmt(row.get('final_accuracy_mean'))} | "
            f"{_fmt(row.get('correct_consensus_rate'))} | "
            f"{_fmt(row.get('collaboration_gain_mean'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_role_mix_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    comparison_model: str,
    prestige_model: str,
) -> None:
    lines = [
        "# Org Flag-Game Role Mix Comparison",
        "",
        f"- comparison model: `{comparison_model}`",
        f"- prestige model: `{prestige_model}`",
        "- manager receives observer text only; observers receive private crops",
        "",
        "| condition | manager | observers | n | IIQ | CIQ manager correct | CIQ - IIQ | oracle gap |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['condition_label']} | "
            f"{row['manager_model']} | "
            f"{row['observer_model']} | "
            f"{row['n_trials']} | "
            f"{_fmt(row.get('initial_accuracy_mean'))} | "
            f"{_fmt(row.get('manager_correct_rate'))} | "
            f"{_fmt(row.get('collaboration_gain_mean'))} | "
            f"{_fmt(row.get('oracle_gap_mean'))} |"
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

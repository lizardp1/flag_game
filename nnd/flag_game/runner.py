from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import pandas as pd

from nnd.backends.parsing import ParseError
from nnd.flag_game.analysis import summarize_initial_probe_rows, summarize_probe_rows
from nnd.flag_game.backend import build_backend
from nnd.flag_game.catalog import COLOR_MAP, StripeFlag, get_country_pool
from nnd.flag_game.config import FlagGameConfig, save_resolved_config
from nnd.flag_game.crops import (
    CropBox,
    all_crop_boxes,
    crop_image,
    mean_pairwise_overlap,
    sample_random_crops,
    scale_crop_box,
)
from nnd.flag_game.diagnostics import build_crop_compatibility_cache, describe_crop_informativeness
from nnd.flag_game.diagnostics import describe_crop_informativeness_fast
from nnd.flag_game.parsing import InteractionMessage
from nnd.flag_game.render import render_flag, save_png
from nnd.flag_game.viz import plot_country_share_trajectories, plot_run_overview, plot_sweep_summary


SKIP_PLOTS = os.environ.get("NND_SKIP_PLOTS") == "1"

# In the 6x4 real-triangle regime, these hand-picked crops are more faithful to
# the intended "triangle witness" intervention than a random tie-break among
# all renderer-unique crops. Each box shows the triangle plus a stripe boundary.
_HARDCODED_TRIANGLE_BEST_6X4_BOXES: dict[str, tuple[int, int]] = {
    "Bahamas": (7, 7),
    "Czech Republic": (6, 8),
    "Palestine": (7, 7),
    "Sudan": (4, 6),
}


@dataclass(frozen=True)
class InteractionRecord:
    t: int
    speaker_id: int
    listener_id: int
    speaker_model: str
    listener_model: str
    m: int
    valid: bool
    country: str | None
    clue: str | None
    reason: str | None
    normalized_message: str | None
    error: str | None = None


@dataclass(frozen=True)
class ProbeRecord:
    t: int
    agent_id: int
    model: str
    valid: bool
    country: str | None
    clue: str | None
    reason: str | None
    correct: bool | None
    error: str | None = None


def choose_default_backend() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "scripted"


def generate_pair_schedule(
    n_agents: int,
    steps: int,
    seed: int,
    speaker_weights: list[float] | None = None,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    schedule: list[tuple[int, int]] = []
    speaker_ids = list(range(n_agents))
    for _ in range(steps):
        if speaker_weights is None:
            speaker = rng.randrange(n_agents)
        else:
            speaker = rng.choices(speaker_ids, weights=speaker_weights, k=1)[0]
        listener = rng.randrange(n_agents - 1)
        if listener >= speaker:
            listener += 1
        schedule.append((speaker, listener))
    return schedule


def oracle_summary_from_crop_diagnostics(
    crop_diagnostics: list[dict[str, Any]],
    *,
    countries: list[str],
    truth_country: str,
) -> dict[str, Any]:
    if not crop_diagnostics:
        return {
            "oracle_country": None,
            "oracle_correct": False,
            "oracle_accuracy": 0.0,
            "oracle_candidate_count": 0,
            "oracle_basis": "none",
            "oracle_truth_in_candidates": False,
        }

    compatibility_sets = [
        {str(country) for country in row.get("compatible_countries", [])}
        for row in crop_diagnostics
    ]
    shared_candidates = set(countries)
    for compatible in compatibility_sets:
        shared_candidates &= compatible

    support_counts = Counter()
    weighted_support: dict[str, float] = {country: 0.0 for country in countries}
    for row in crop_diagnostics:
        compatible = [str(country) for country in row.get("compatible_countries", [])]
        if not compatible:
            continue
        weight = 1.0 / float(len(compatible))
        for country in compatible:
            support_counts[country] += 1
            weighted_support[country] += weight

    if shared_candidates:
        candidate_pool = set(shared_candidates)
        basis = "intersection"
    elif support_counts:
        max_support = max(support_counts.values())
        candidate_pool = {country for country in countries if support_counts.get(country, 0) == max_support}
        basis = "max_support"
    else:
        candidate_pool = set(countries)
        basis = "fallback"

    oracle_country = sorted(
        candidate_pool,
        key=lambda country: (-weighted_support.get(country, 0.0), country),
    )[0]
    return {
        "oracle_country": oracle_country,
        "oracle_correct": oracle_country == truth_country,
        "oracle_accuracy": 1.0 if oracle_country == truth_country else 0.0,
        "oracle_candidate_count": len(candidate_pool),
        "oracle_basis": basis,
        "oracle_truth_in_candidates": truth_country in candidate_pool,
    }


def _has_stable_full_consensus(
    per_round_df: pd.DataFrame,
    window: int,
    expected_probe_count: int,
) -> tuple[bool, str | None]:
    if window <= 0 or per_round_df.empty or len(per_round_df) < window:
        return False, None
    recent = per_round_df.tail(window)
    countries = list(recent["consensus_country"])
    if any(not isinstance(country, str) or not country for country in countries):
        return False, None
    if len(set(countries)) != 1:
        return False, None
    if not all(abs(float(value) - 1.0) < 1e-9 for value in recent["top1_share"]):
        return False, None
    if not all(int(value) == expected_probe_count for value in recent["valid_probe_count"]):
        return False, None
    return True, countries[0]


def _resolve_agent_models(config: FlagGameConfig) -> list[str]:
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
    config: FlagGameConfig,
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
            social_susceptibility=config.social_susceptibility,
            prompt_social_susceptibility=config.prompt_social_susceptibility,
            prompt_style=config.prompt_style,
            country_lookup=country_lookup,
        )
    return [backend_cache[model] for model in agent_models]


def _select_engineered_crop_box(
    *,
    config: FlagGameConfig,
    truth_flag: Any,
    full_image: Any,
    countries: list[str],
    compatibility_cache: dict[str, set[bytes]],
    rng: random.Random,
) -> tuple[CropBox, dict[str, Any]]:
    if (
        config.engineered_crop_preference == "best"
        and config.tile_width == 6
        and config.tile_height == 4
    ):
        hardcoded_coords = _HARDCODED_TRIANGLE_BEST_6X4_BOXES.get(getattr(truth_flag, "country", ""))
        if hardcoded_coords is not None:
            hardcoded_box = next(
                (
                    box
                    for box in all_crop_boxes(
                        canvas_width=config.canvas_width,
                        canvas_height=config.canvas_height,
                        tile_width=config.tile_width,
                        tile_height=config.tile_height,
                    )
                    if (box.top, box.left) == hardcoded_coords
                ),
                None,
            )
            if hardcoded_box is None:
                raise ValueError(
                    "Hardcoded triangle crop box was not found for "
                    f"{truth_flag.country} at tile {config.tile_width}x{config.tile_height}"
                )
            scaled_box = scale_crop_box(hardcoded_box, config.render_scale)
            crop = crop_image(full_image, scaled_box)
            diagnostic = describe_crop_informativeness(
                crop,
                country_order=countries,
                compatibility_cache=compatibility_cache,
            )
            return hardcoded_box, diagnostic

    candidates: list[tuple[CropBox, dict[str, Any], bool]] = []
    triangle_rgb = (
        COLOR_MAP[truth_flag.triangle_color]
        if getattr(truth_flag, "triangle_color", None) is not None
        else None
    )
    for box in all_crop_boxes(
        canvas_width=config.canvas_width,
        canvas_height=config.canvas_height,
        tile_width=config.tile_width,
        tile_height=config.tile_height,
    ):
        scaled_box = scale_crop_box(box, config.render_scale)
        crop = crop_image(full_image, scaled_box)
        diagnostic = describe_crop_informativeness(
            crop,
            country_order=countries,
            compatibility_cache=compatibility_cache,
        )
        triangle_visible = False
        if triangle_rgb is not None:
            triangle_visible = bool((crop == triangle_rgb).all(axis=2).any())
        candidates.append((box, diagnostic, triangle_visible))

    if not candidates:
        raise ValueError("No candidate crops found for engineered crop selection")

    # In the real-triangle conditions, the engineered witness should actually
    # contain triangle pixels rather than a stripe-only crop that happens to be
    # renderer-unique. If no such crops exist, fall back to the full set.
    candidate_pool = [item for item in candidates if item[2]] if triangle_rgb is not None else candidates
    if not candidate_pool:
        candidate_pool = candidates

    if config.engineered_crop_preference == "best":
        ranked = sorted(
            candidate_pool,
            key=lambda item: (
                int(item[1]["compatible_country_count"]),
                -float(item[1]["informativeness_bits"] or 0.0),
                int(item[0].top),
                int(item[0].left),
            ),
        )
        best_count = int(ranked[0][1]["compatible_country_count"])
        tied = [item for item in ranked if int(item[1]["compatible_country_count"]) == best_count]
    elif config.engineered_crop_preference == "worst":
        ranked = sorted(
            candidate_pool,
            key=lambda item: (
                -int(item[1]["compatible_country_count"]),
                float(item[1]["informativeness_bits"] or 0.0),
                int(item[0].top),
                int(item[0].left),
            ),
        )
        worst_count = int(ranked[0][1]["compatible_country_count"])
        tied = [item for item in ranked if int(item[1]["compatible_country_count"]) == worst_count]
    else:
        raise ValueError(f"Unsupported engineered_crop_preference: {config.engineered_crop_preference}")

    chosen_box, chosen_diagnostic, _ = tied[rng.randrange(len(tied))]
    return chosen_box, chosen_diagnostic


def run_flag_game_experiment(
    config: FlagGameConfig,
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
    compute_crop_diagnostics = not (
        config.T == 0 and config.H == 0 and config.engineered_crop_agent_id is None
    )
    compatibility_cache: dict[str, set[bytes]] = {}
    use_fast_crop_diagnostics = compute_crop_diagnostics and all(
        isinstance(flag, StripeFlag) and flag.triangle_color is None for flag in pool
    )
    assignments = sample_random_crops(
        canvas_width=config.canvas_width,
        canvas_height=config.canvas_height,
        tile_width=config.tile_width,
        tile_height=config.tile_height,
        n_agents=config.N,
        rng=rng,
        target_overlap=config.observation_overlap,
        search_trials=config.overlap_search_trials,
        overlap_mode=config.observation_overlap_mode,
    )
    actual_overlap = mean_pairwise_overlap(assignments)
    assignment_position_counts = Counter((box.top, box.left, box.height, box.width) for box in assignments)
    if compute_crop_diagnostics and not use_fast_crop_diagnostics:
        compatibility_cache = build_crop_compatibility_cache(
            pool,
            canvas_width=config.canvas_width,
            canvas_height=config.canvas_height,
            tile_width=config.tile_width,
            tile_height=config.tile_height,
            render_scale=config.render_scale,
        )
    engineered_crop_info: dict[str, Any] | None = None
    if config.engineered_crop_agent_id is not None:
        engineered_box, engineered_diagnostic = _select_engineered_crop_box(
            config=config,
            truth_flag=truth_flag,
            full_image=full_image,
            countries=countries,
            compatibility_cache=compatibility_cache,
            rng=rng,
        )
        assignments[config.engineered_crop_agent_id] = engineered_box
        engineered_crop_info = {
            "agent_id": config.engineered_crop_agent_id,
            "preference": config.engineered_crop_preference,
            "assignment": engineered_box.to_dict(),
            "diagnostic": engineered_diagnostic,
        }

    scaled_assignments = [scale_crop_box(box, config.render_scale) for box in assignments]
    crop_images = [crop_image(full_image, box) for box in scaled_assignments]

    if config.output.save_crop_images:
        save_png(out_dir / "artifacts" / "truth_flag.png", full_image)
        for agent_id, image in enumerate(crop_images):
            save_png(out_dir / "artifacts" / f"agent_{agent_id:02d}_crop.png", image)

    memories = [deque(maxlen=config.H) for _ in range(config.N)]
    agent_models = _resolve_agent_models(config)
    crop_diagnostics = []
    if compute_crop_diagnostics:
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

    interaction_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    probe_executor = ThreadPoolExecutor(max_workers=config.probe_workers) if config.probe_workers > 1 else None
    stopped_early = False
    early_stop_country: str | None = None
    executed_steps = 0

    def run_probe(t: int) -> tuple[pd.DataFrame | None, bool, str | None]:
        memory_snapshots = [list(memories[agent_id]) for agent_id in range(config.N)]

        def _probe_one(agent_id: int) -> dict[str, Any]:
            backend = agent_backends[agent_id]
            try:
                probe_m = config.interaction_m if t == 0 and config.interaction_m == 3 else 1
                message = backend.probe(
                    countries=countries,
                    prepared_crop=prepared_crops[agent_id],
                    memory_lines=memory_snapshots[agent_id],
                    m=probe_m,
                )
                if isinstance(message, str):
                    message = InteractionMessage(country=message)
                return asdict(
                    ProbeRecord(
                        t=t,
                        agent_id=agent_id,
                        model=agent_models[agent_id],
                        valid=True,
                        country=message.country,
                        clue=message.clue,
                        reason=message.reason,
                        correct=(message.country == truth_flag.country),
                    )
                )
            except ParseError as exc:
                return asdict(
                    ProbeRecord(
                        t=t,
                        agent_id=agent_id,
                        model=agent_models[agent_id],
                        valid=False,
                        country=None,
                        clue=None,
                        reason=None,
                        correct=None,
                        error=str(exc),
                    )
                )

        if probe_executor is None:
            rows = [_probe_one(agent_id) for agent_id in range(config.N)]
        else:
            rows = list(probe_executor.map(_probe_one, range(config.N)))
        probe_rows.extend(rows)
        partial_df: pd.DataFrame | None = None
        if config.output.make_plots and not SKIP_PLOTS or config.early_stop_probe_window > 0:
            partial_df, _ = summarize_probe_rows(
                probe_rows,
                countries=countries,
                truth_country=truth_flag.country,
                consensus_threshold=config.consensus_threshold,
                polarization_threshold=config.polarization_threshold,
            )
        if config.output.make_plots and not SKIP_PLOTS and partial_df is not None:
            plot_country_share_trajectories(partial_df, out_dir)
        should_stop, stop_country = _has_stable_full_consensus(
            partial_df if partial_df is not None else pd.DataFrame(),
            config.early_stop_probe_window,
            config.N,
        )
        return partial_df, should_stop, stop_country

    try:
        _, should_stop, stop_country = run_probe(t=0)
        if should_stop:
            stopped_early = True
            early_stop_country = stop_country
        if stopped_early:
            schedule: list[tuple[int, int]] = []
        else:
            schedule = generate_pair_schedule(
                config.N,
                config.T,
                seed + 13,
                speaker_weights=config.speaker_weights,
            )
        for t, (speaker_id, listener_id) in enumerate(schedule, start=1):
            executed_steps = t
            speaker_backend = agent_backends[speaker_id]
            try:
                message = speaker_backend.interaction(
                    countries=countries,
                    prepared_crop=prepared_crops[speaker_id],
                    memory_lines=list(memories[speaker_id]),
                    m=config.interaction_m,
                )
            except ParseError as exc:
                interaction_rows.append(
                    asdict(
                        InteractionRecord(
                            t=t,
                            speaker_id=speaker_id,
                            listener_id=listener_id,
                            speaker_model=agent_models[speaker_id],
                            listener_model=agent_models[listener_id],
                            m=config.interaction_m,
                            valid=False,
                            country=None,
                            clue=None,
                            reason=None,
                            normalized_message=None,
                            error=str(exc),
                        )
                    )
                )
                if t % config.probe_every == 0 or t == config.T:
                    _, should_stop, stop_country = run_probe(t=t)
                    if should_stop:
                        stopped_early = True
                        early_stop_country = stop_country
                        break
                continue

            normalized = message.normalized_memory_entry()
            interaction_rows.append(
                asdict(
                    InteractionRecord(
                        t=t,
                        speaker_id=speaker_id,
                        listener_id=listener_id,
                        speaker_model=agent_models[speaker_id],
                        listener_model=agent_models[listener_id],
                        m=config.interaction_m,
                        valid=True,
                        country=message.country,
                        clue=message.clue,
                        reason=message.reason,
                        normalized_message=normalized,
                    )
                )
            )
            if not config.disable_memory_updates and config.H > 0:
                memories[listener_id].append(normalized)
            if t % config.probe_every == 0 or t == config.T:
                _, should_stop, stop_country = run_probe(t=t)
                if should_stop:
                    stopped_early = True
                    early_stop_country = stop_country
                    break
    finally:
        if probe_executor is not None:
            probe_executor.shutdown(wait=True)

    per_round_df, summary = summarize_probe_rows(
        probe_rows,
        countries=countries,
        truth_country=truth_flag.country,
        consensus_threshold=config.consensus_threshold,
        polarization_threshold=config.polarization_threshold,
    )
    t0_probe_df, t0_summary = summarize_initial_probe_rows(
        probe_rows,
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
            "T": config.T,
            "H": config.H,
            "interaction_m": config.interaction_m,
            "social_susceptibility": config.social_susceptibility,
            "prompt_social_susceptibility": config.prompt_social_susceptibility,
            "prompt_style": config.prompt_style,
            "render_scale": config.render_scale,
            "image_detail": config.image_detail,
            "observation_overlap_target": config.observation_overlap,
            "observation_overlap_mode": config.observation_overlap_mode,
            "observation_overlap_realized": actual_overlap,
            "distinct_crop_location_count": len(assignment_position_counts),
            "max_agents_per_crop_location": max(assignment_position_counts.values())
            if assignment_position_counts
            else 0,
            "duplicate_crop_location_count": sum(
                count - 1 for count in assignment_position_counts.values() if count > 1
            ),
            "speaker_weights": config.speaker_weights,
            "engineered_crop_agent_id": config.engineered_crop_agent_id,
            "engineered_crop_preference": config.engineered_crop_preference,
            "memory_updates_enabled": not config.disable_memory_updates,
            "executed_T": executed_steps,
            "stopped_early": stopped_early,
            "early_stop_country": early_stop_country,
            "invalid_probe_count": int(sum(1 for row in probe_rows if not bool(row.get("valid", False)))),
            "invalid_interaction_count": int(sum(1 for row in interaction_rows if not bool(row.get("valid", False)))),
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

    _write_jsonl(out_dir / "interactions.jsonl", interaction_rows)
    _write_jsonl(out_dir / "probes.jsonl", probe_rows)
    per_round_df.to_csv(out_dir / "per_round.csv", index=False)
    t0_probe_df.to_csv(out_dir / "t0_probe_diagnostics.csv", index=False)
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
                    {"agent_id": idx, "model": model}
                    for idx, model in enumerate(agent_models)
                ],
                "crop_diagnostics": crop_diagnostics,
                "canvas": {"width": config.canvas_width, "height": config.canvas_height},
                "render": {"scale": config.render_scale, "width": render_width, "height": render_height},
                "image_detail": config.image_detail,
                "social_susceptibility": config.social_susceptibility,
                "prompt_social_susceptibility": config.prompt_social_susceptibility,
                "prompt_style": config.prompt_style,
                "observation_overlap_target": config.observation_overlap,
                "observation_overlap_mode": config.observation_overlap_mode,
                "observation_overlap_realized": actual_overlap,
                "distinct_crop_location_count": len(assignment_position_counts),
                "max_agents_per_crop_location": max(assignment_position_counts.values())
                if assignment_position_counts
                else 0,
                "duplicate_crop_location_count": sum(
                    count - 1 for count in assignment_position_counts.values() if count > 1
                ),
                "speaker_weights": config.speaker_weights,
                "engineered_crop_agent_id": config.engineered_crop_agent_id,
                "engineered_crop_preference": config.engineered_crop_preference,
                "engineered_crop_info": engineered_crop_info,
                "oracle_summary": oracle_summary,
                "api_usage_summary": usage_summary,
                "tile": {"width": config.tile_width, "height": config.tile_height},
                "executed_T": executed_steps,
                "stopped_early": stopped_early,
                "early_stop_country": early_stop_country,
                "pair_schedule": [
                    {"t": idx + 1, "speaker_id": speaker_id, "listener_id": listener_id}
                    for idx, (speaker_id, listener_id) in enumerate(schedule)
                ],
                "assignments": [
                    {"agent_id": idx, **box.to_dict()}
                    for idx, box in enumerate(assignments)
                ],
                "pixel_assignments": [
                    {"agent_id": idx, **box.to_dict()}
                    for idx, box in enumerate(scaled_assignments)
                ],
            },
            handle,
            indent=2,
        )

    if config.output.make_plots and not SKIP_PLOTS:
        plot_run_overview(per_round_df, out_dir)

    return {
        "summary": summary,
        "per_round": per_round_df.to_dict(orient="records"),
        "probes": probe_rows,
        "interactions": interaction_rows,
        "assignments": [box.to_dict() for box in assignments],
        "crop_diagnostics": crop_diagnostics,
        "t0_probe_diagnostics": t0_probe_df.to_dict(orient="records"),
    }


def run_flag_game_batch(
    base_config: FlagGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    def _run_seed(seed: int) -> tuple[int, dict[str, Any]]:
        trial_out = out_dir / f"seed_{seed:04d}"
        return seed, run_flag_game_experiment(base_config, out_dir=trial_out, seed=seed)

    if base_config.seed_workers == 1:
        pairs = [_run_seed(seed) for seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=base_config.seed_workers) as executor:
            pairs = list(executor.map(_run_seed, seeds))
    pairs.sort(key=lambda item: item[0])
    results = [result for _, result in pairs]
    summary_rows = [result["summary"] for result in results]
    pd.DataFrame(summary_rows).to_csv(out_dir / "batch_summary.csv", index=False)
    return results


def run_flag_game_sweep(
    base_config: FlagGameConfig,
    *,
    out_dir: Path,
    n_values: list[int],
    m_values: list[int],
    seeds: list[int],
    tile_sizes: list[tuple[int, int]] | None = None,
    scale_t_with_n: bool = True,
    rounds: int | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    default_probe_every = max(base_config.N // 2, 1)
    base_probe_was_default = base_config.probe_every == default_probe_every
    inferred_rounds = max(base_config.T // max(base_config.N, 1), 1)
    rounds_per_condition = rounds if rounds is not None else inferred_rounds
    active_tile_sizes = tile_sizes or [(base_config.tile_width, base_config.tile_height)]
    valid_conditions: list[dict[str, Any]] = []

    for tile_width, tile_height in sorted(set((int(w), int(h)) for w, h in active_tile_sizes)):
        tile_spec = f"{tile_width}x{tile_height}"
        for n in sorted(set(int(value) for value in n_values)):
            for interaction_m in sorted(set(int(value) for value in m_values)):
                condition_t = base_config.T
                if scale_t_with_n:
                    condition_t = max(rounds_per_condition * n, 1)

                condition_probe_every = base_config.probe_every
                if base_probe_was_default:
                    condition_probe_every = max(n // 2, 1)

                try:
                    condition_config = base_config.model_copy(
                        update={
                            "N": n,
                            "T": condition_t,
                            "interaction_m": interaction_m,
                            "probe_every": condition_probe_every,
                            "tile_width": tile_width,
                            "tile_height": tile_height,
                            "seed_workers": base_config.seed_workers,
                            "condition_workers": base_config.condition_workers,
                            "probe_workers": base_config.probe_workers,
                            "output": base_config.output.model_copy(update={"make_plots": make_plots}),
                        }
                    )
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "N": n,
                            "interaction_m": interaction_m,
                            "tile_width": tile_width,
                            "tile_height": tile_height,
                            "tile_spec": tile_spec,
                            "reason": str(exc),
                        }
                    )
                    continue
                valid_conditions.append(
                    {
                        "config": condition_config,
                        "condition_dir": out_dir / f"tile_{tile_spec}" / f"N{n}_m{interaction_m}",
                        "n": n,
                        "interaction_m": interaction_m,
                        "condition_t": condition_t,
                        "condition_probe_every": condition_probe_every,
                        "tile_width": tile_width,
                        "tile_height": tile_height,
                        "tile_spec": tile_spec,
                    }
                )

    def _run_condition(condition: dict[str, Any]) -> dict[str, Any]:
        results = run_flag_game_batch(condition["config"], out_dir=condition["condition_dir"], seeds=seeds)
        return {
            "condition": condition,
            "results": results,
        }

    if base_config.condition_workers == 1:
        condition_outputs = [_run_condition(condition) for condition in valid_conditions]
    else:
        with ThreadPoolExecutor(max_workers=base_config.condition_workers) as executor:
            condition_outputs = list(executor.map(_run_condition, valid_conditions))

    for output in condition_outputs:
        condition = output["condition"]
        results = output["results"]
        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            row.update(
                {
                    "seed": seed,
                    "N": condition["n"],
                    "interaction_m": condition["interaction_m"],
                    "T": condition["condition_t"],
                    "probe_every": condition["condition_probe_every"],
                    "tile_width": condition["tile_width"],
                    "tile_height": condition["tile_height"],
                    "tile_spec": condition["tile_spec"],
                    "condition_dir": str(condition["condition_dir"]),
                }
            )
            condition_rows.append(row)

        summary_df = pd.DataFrame([result["summary"] for result in results])
        correct_mask = summary_df["final_outcome"] == "correct_consensus"
        wrong_mask = summary_df["final_outcome"] == "wrong_consensus"
        polarization_mask = summary_df["final_outcome"] == "polarization"
        fragmentation_mask = summary_df["final_outcome"] == "fragmentation"

        aggregate_rows.append(
            {
                "N": condition["n"],
                "interaction_m": condition["interaction_m"],
                "T": condition["condition_t"],
                "probe_every": condition["condition_probe_every"],
                "tile_width": condition["tile_width"],
                "tile_height": condition["tile_height"],
                "tile_spec": condition["tile_spec"],
                "n_trials": len(results),
                "correct_consensus_rate": float(correct_mask.mean()) if len(results) else 0.0,
                "wrong_consensus_rate": float(wrong_mask.mean()) if len(results) else 0.0,
                "polarization_rate": float(polarization_mask.mean()) if len(results) else 0.0,
                "fragmentation_rate": float(fragmentation_mask.mean()) if len(results) else 0.0,
                "initial_accuracy_mean": float(summary_df["initial_accuracy"].mean()) if not summary_df.empty else None,
                "final_accuracy_mean": float(summary_df["final_accuracy"].mean()) if not summary_df.empty else None,
                "final_vote_accuracy_rate": float(summary_df["final_vote_accuracy"].mean())
                if not summary_df.empty and "final_vote_accuracy" in summary_df
                else None,
                "collaboration_gain_mean": float(summary_df["collaboration_gain_over_initial_accuracy"].mean()) if not summary_df.empty else None,
                "time_to_correct_consensus_mean": (
                    float(summary_df["time_to_correct_consensus"].dropna().mean())
                    if not summary_df.empty and not summary_df["time_to_correct_consensus"].dropna().empty
                    else None
                ),
            }
        )

    condition_df = pd.DataFrame(condition_rows)
    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(["tile_width", "tile_height", "interaction_m", "N"]).reset_index(drop=True)
    condition_df.to_csv(out_dir / "sweep_condition_results.csv", index=False)
    aggregate_df.to_csv(out_dir / "sweep_summary.csv", index=False)
    skipped_df = pd.DataFrame(skipped_rows)
    if not skipped_df.empty:
        skipped_df.to_csv(out_dir / "sweep_skipped_conditions.csv", index=False)

    if make_plots and not SKIP_PLOTS:
        plot_sweep_summary(aggregate_df, out_dir)

    return {
        "condition_results": condition_df.to_dict(orient="records"),
        "summary": aggregate_df.to_dict(orient="records"),
        "skipped_conditions": skipped_df.to_dict(orient="records") if not skipped_df.empty else [],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

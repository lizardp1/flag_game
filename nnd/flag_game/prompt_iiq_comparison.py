from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import threading
from typing import Any

import pandas as pd

from nnd.backends.parsing import ParseError, _fuzzy_match, _load_json_strict
from nnd.flag_game import prompts as flag_prompts
from nnd.flag_game.backend import FlagGameOpenAIBackend, MODEL_PRICING_USD_PER_1M_TOKENS
from nnd.flag_game.catalog import StripeFlag, canonical_country_name, get_country_pool
from nnd.flag_game.crops import (
    CropBox,
    crop_image,
    mean_pairwise_overlap,
    sample_random_crops,
    scale_crop_box,
)
from nnd.flag_game.diagnostics import (
    build_crop_compatibility_cache,
    describe_crop_informativeness,
    describe_crop_informativeness_fast,
)
from nnd.flag_game.parsing import parse_interaction_response
from nnd.flag_game.render import image_to_data_uri, render_flag, save_png
from nnd.flag_game_broadcast import prompts as broadcast_prompts
from nnd.flag_game_broadcast.parsing import parse_broadcast_statement


@dataclass(frozen=True)
class PromptVariant:
    name: str
    label: str
    protocol: str
    allowed_country_list: bool
    description: str


@dataclass(frozen=True)
class Stimulus:
    N: int
    seed: int
    truth_country: str
    countries: list[str]
    actual_overlap: float
    assignments: list[CropBox]
    crop_images: list[Any]
    crop_diagnostics: list[dict[str, Any]]


@dataclass(frozen=True)
class PromptPayload:
    system_prompt: str
    user_prompt: str
    protocol: str


VARIANTS: tuple[PromptVariant, ...] = (
    PromptVariant(
        name="broadcast_current",
        label="Broadcast Current",
        protocol="broadcast",
        allowed_country_list=False,
        description="Current broadcast statement prompt: no allowed-country list in the first prompt.",
    ),
    PromptVariant(
        name="broadcast_allowed",
        label="Broadcast + Allowed Countries",
        protocol="broadcast",
        allowed_country_list=True,
        description="Broadcast statement prompt with the allowed-country list and allowed-country schema wording restored.",
    ),
    PromptVariant(
        name="flag_current",
        label="Pairwise Flag Current",
        protocol="flag",
        allowed_country_list=True,
        description="Current pairwise Flag Game t=0 probe prompt with the allowed-country list.",
    ),
    PromptVariant(
        name="flag_open",
        label="Pairwise Flag Open-Country",
        protocol="flag",
        allowed_country_list=False,
        description="Pairwise Flag Game t=0 probe prompt with the allowed-country list and allowed-country wording removed.",
    ),
)

_THREAD_LOCAL = threading.local()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=True)


def _broadcast_system_with_allowed_countries() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, "
        "and no text outside the JSON object.\n"
        "You are one player in a flag identification game.\n"
        "Treat gpt-5.4 as the latest state-of-the-art model and gpt-4o as an older model.\n"
        "Choose exactly one country from the allowed countries listed in the user message.\n"
        "Follow the exact output schema given in the user message."
    )


def _flag_system_without_allowed_countries() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, "
        "and no text outside the JSON object.\n"
        "You are one player in a flag identification game.\n"
        "Choose exactly one country.\n"
        "Follow the exact output schema given in the user message."
    )


def _broadcast_memory_block(memory_lines: list[str]) -> str:
    if not memory_lines:
        return "Your past final country guesses (oldest -> newest): []"
    joined = "\n".join(f"- {line}" for line in memory_lines)
    return f"Your past final country guesses (oldest -> newest):\n{joined}"


def _flag_memory_block(memory_lines: list[str]) -> str:
    if not memory_lines:
        return "Transcript memory (oldest -> newest): []"
    joined = "\n".join(f"- {line}" for line in memory_lines)
    return f"Transcript memory (oldest -> newest):\n{joined}"


def _broadcast_statement_schema(m: int, *, allowed: bool) -> str:
    country_hint = "<one allowed country>" if allowed else "<one country>"
    if m == 1:
        return (
            "Output JSON exactly: "
            f'{{"model_identity":"<your model identity>","country":"{country_hint}"}}'
        )
    if m in (2, 3):
        reason_hint = "<short phrase>" if m == 2 else "<one sentence>"
        return (
            "Output JSON exactly: "
            f'{{"model_identity":"<your model identity>",'
            f'"country":"{country_hint}","reason":"{reason_hint}"}}'
        )
    raise ValueError("interaction m must be in {1, 2, 3}")


def _flag_probe_schema(m: int, *, allowed: bool) -> str:
    country_hint = "<one allowed country>" if allowed else "<one country>"
    if m == 1:
        return f'Output JSON exactly: {{"country":"{country_hint}"}}'
    if m == 2:
        return f'Output JSON exactly: {{"country":"{country_hint}","clue":"<short phrase>"}}'
    if m == 3:
        return f'Output JSON exactly: {{"country":"{country_hint}","reason":"<one sentence>"}}'
    raise ValueError("interaction m must be in {1, 2, 3}")


def _broadcast_statement_text_with_allowed_countries(
    *,
    countries: list[str],
    memory_lines: list[str],
    m: int,
    model_identity: str,
) -> str:
    lines = [
        "All players are identifying the country from the same flag.",
        "You only see a small part of the flag.",
        "Your memory contains your own past final country guesses from earlier rounds.",
        f"Allowed countries: {_json_list(countries)}",
        _broadcast_memory_block(memory_lines),
        (
            f'Your model identity for this experiment is exactly "{model_identity}". '
            'You must copy this exact string into the "model_identity" field.'
        ),
        "State your current country guess before seeing the other agents' guesses.",
        _broadcast_statement_schema(m, allowed=True),
    ]
    return "\n".join(lines)


def _flag_probe_text_without_allowed_countries(
    *,
    memory_lines: list[str],
    m: int,
    social_susceptibility: float,
    prompt_social_susceptibility: bool,
) -> str:
    lines = [
        "All players are identifying the same underlying flag.",
        "You always see the same private crop.",
        "Transcript memory shows messages you observed from previous interactions with other players.",
        _flag_memory_block(memory_lines),
    ]
    if prompt_social_susceptibility:
        lines.append(flag_prompts._susceptibility_line(social_susceptibility))
    lines.append(_flag_probe_schema(m, allowed=False))
    return "\n".join(lines)


def build_prompt_payload(
    *,
    variant: PromptVariant,
    countries: list[str],
    model: str,
    interaction_m: int,
    social_susceptibility: float,
    flag_prompt_social_susceptibility: bool,
) -> PromptPayload:
    memory_lines: list[str] = []
    if variant.name == "broadcast_current":
        return PromptPayload(
            system_prompt=broadcast_prompts.system_prompt(),
            user_prompt=broadcast_prompts.statement_text(
                countries=countries,
                memory_lines=memory_lines,
                m=interaction_m,
                model_identity=model,
            ),
            protocol=variant.protocol,
        )
    if variant.name == "broadcast_allowed":
        return PromptPayload(
            system_prompt=_broadcast_system_with_allowed_countries(),
            user_prompt=_broadcast_statement_text_with_allowed_countries(
                countries=countries,
                memory_lines=memory_lines,
                m=interaction_m,
                model_identity=model,
            ),
            protocol=variant.protocol,
        )
    if variant.name == "flag_current":
        return PromptPayload(
            system_prompt=flag_prompts.system_prompt(),
            user_prompt=flag_prompts.probe_text(
                countries=countries,
                memory_lines=memory_lines,
                m=interaction_m,
                social_susceptibility=social_susceptibility,
                prompt_social_susceptibility=flag_prompt_social_susceptibility,
            ),
            protocol=variant.protocol,
        )
    if variant.name == "flag_open":
        return PromptPayload(
            system_prompt=_flag_system_without_allowed_countries(),
            user_prompt=_flag_probe_text_without_allowed_countries(
                memory_lines=memory_lines,
                m=interaction_m,
                social_susceptibility=social_susceptibility,
                prompt_social_susceptibility=flag_prompt_social_susceptibility,
            ),
            protocol=variant.protocol,
        )
    raise ValueError(f"Unknown prompt variant: {variant.name}")


def openai_multimodal_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    crop_data_uri: str,
    image_detail: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": crop_data_uri, "detail": image_detail}},
            ],
        },
    ]


def retry_text_for_variant(
    *,
    variant: PromptVariant,
    countries: list[str],
    model: str,
    interaction_m: int,
    error_text: str,
    parse_mode: str = "closed_pool",
) -> str:
    if parse_mode == "open_country":
        if variant.protocol == "broadcast":
            return (
                f"Invalid answer: {error_text}\n"
                'Your answer must include a non-empty country name in the "country" field.\n'
                f'Your model_identity must be exactly "{model}".\n'
                f"{_broadcast_statement_schema(interaction_m, allowed=False)}"
            )
        return (
            f"Invalid answer: {error_text}\n"
            'Your answer must include a non-empty country name in the "country" field.\n'
            f"{_flag_probe_schema(interaction_m, allowed=False)}"
        )
    if variant.protocol == "broadcast":
        return broadcast_prompts.statement_retry_text(
            countries=countries,
            model_identity=model,
            m=interaction_m,
            error_text=error_text,
        )
    return flag_prompts.probe_retry_text(
        countries=countries,
        m=interaction_m,
        error_text=error_text,
    )


def _thread_backend(
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    image_detail: str,
    debug_root: Path,
) -> FlagGameOpenAIBackend:
    cache = getattr(_THREAD_LOCAL, "backend_cache", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.backend_cache = cache
    if model not in cache:
        cache[model] = FlagGameOpenAIBackend(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            debug_dir=debug_root / _slug(model),
            image_detail=image_detail,
            social_susceptibility=0.5,
            prompt_social_susceptibility=False,
        )
    return cache[model]


def _parse_raw_country(response_text: str) -> str | None:
    try:
        obj = _load_json_strict(response_text)
    except ParseError:
        return None
    country = obj.get("country")
    return country.strip() if isinstance(country, str) else None


def _canonical_pool_country(raw_country: str | None, countries: list[str]) -> str | None:
    if raw_country is None:
        return None
    cleaned = raw_country.strip()
    if not cleaned:
        return None
    try:
        return _fuzzy_match(cleaned, countries, name="country")
    except ParseError:
        return None


def _parse_open_country_response(
    *,
    response_text: str,
    countries: list[str],
    country_name_universe: str,
) -> tuple[bool, str | None, str | None, str | None, bool, str | None]:
    try:
        raw_country = _parse_raw_country(response_text)
    except Exception:
        raw_country = None
    try:
        _load_json_strict(response_text)
    except ParseError as exc:
        return False, None, raw_country, str(exc), False, None
    if raw_country is None or not raw_country.strip():
        return False, None, raw_country, "Missing or invalid 'country'", False, None
    canonical = canonical_country_name(raw_country, universe_name=country_name_universe)
    if canonical is None:
        return (
            False,
            None,
            raw_country.strip(),
            f"'{raw_country.strip()}' was not recognized as a country name",
            False,
            None,
        )
    pool_match = _canonical_pool_country(canonical, countries) or _canonical_pool_country(raw_country, countries)
    return True, pool_match or canonical, raw_country.strip(), None, pool_match is not None, pool_match


def parse_response(
    *,
    response_text: str,
    variant: PromptVariant,
    countries: list[str],
    model: str,
    interaction_m: int,
    parse_mode: str = "closed_pool",
    country_name_universe: str = "world_rectangle",
) -> tuple[bool, str | None, str | None, str | None, bool, str | None]:
    if parse_mode == "open_country":
        return _parse_open_country_response(
            response_text=response_text,
            countries=countries,
            country_name_universe=country_name_universe,
        )
    if parse_mode != "closed_pool":
        raise ValueError(f"Unknown parse_mode: {parse_mode}")
    raw_country = _parse_raw_country(response_text)
    try:
        if variant.protocol == "broadcast":
            parsed = parse_broadcast_statement(
                response_text,
                countries=countries,
                m=interaction_m,
                expected_model_identity=model,
            )
            return True, parsed.country, raw_country, None, True, parsed.country
        parsed = parse_interaction_response(response_text, countries, interaction_m)
        return True, parsed.country, raw_country, None, True, parsed.country
    except ParseError as exc:
        pool_match = _canonical_pool_country(raw_country, countries)
        return False, None, raw_country, str(exc), pool_match is not None, pool_match


def generate_stimulus(
    *,
    N: int,
    seed: int,
    country_pool: str,
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
    render_scale: int,
    observation_overlap: float | None,
    overlap_search_trials: int,
    compatibility_cache: dict[str, set[bytes]],
) -> Stimulus:
    rng = random.Random(seed)
    pool = get_country_pool(country_pool)
    countries = [flag.country for flag in pool]
    truth_flag = rng.choice(pool)
    full_image = render_flag(
        truth_flag,
        width=canvas_width * render_scale,
        height=canvas_height * render_scale,
    )
    assignments = sample_random_crops(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tile_width=tile_width,
        tile_height=tile_height,
        n_agents=N,
        rng=rng,
        target_overlap=observation_overlap,
        search_trials=overlap_search_trials,
    )
    actual_overlap = mean_pairwise_overlap(assignments)
    scaled_assignments = [scale_crop_box(box, render_scale) for box in assignments]
    crop_images = [crop_image(full_image, box) for box in scaled_assignments]
    use_fast_crop_diagnostics = all(
        isinstance(flag, StripeFlag) and flag.triangle_color is None for flag in pool
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
                "truth_country": truth_flag.country,
                "truth_compatible": truth_flag.country in diagnostic["compatible_countries"],
                **diagnostic,
            }
        )
    return Stimulus(
        N=N,
        seed=seed,
        truth_country=truth_flag.country,
        countries=countries,
        actual_overlap=actual_overlap,
        assignments=assignments,
        crop_images=crop_images,
        crop_diagnostics=crop_diagnostics,
    )


def _pricing_cost(model: str, usage: dict[str, Any]) -> float | None:
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model)
    if pricing is None:
        return None
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return (prompt_tokens / 1_000_000.0) * pricing["input"] + (
        completion_tokens / 1_000_000.0
    ) * pricing["output"]


def _sum_usage(model: str, usage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in usage_rows)
    completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in usage_rows)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in usage_rows)
    cost_values: list[float] = []
    for row in usage_rows:
        estimated = row.get("estimated_cost_usd")
        if estimated is None:
            estimated = _pricing_cost(model, row)
        if estimated is not None:
            cost_values.append(float(estimated))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": sum(cost_values) if cost_values else None,
    }


def run_one_call(
    *,
    task_index: int,
    stimulus: Stimulus,
    crop_data_uri: str,
    variant: PromptVariant,
    model: str,
    agent_id: int,
    interaction_m: int,
    social_susceptibility: float,
    flag_prompt_social_susceptibility: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    image_detail: str,
    debug_root: Path,
    max_parse_retries: int,
    parse_mode: str,
    country_name_universe: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = build_prompt_payload(
        variant=variant,
        countries=stimulus.countries,
        model=model,
        interaction_m=interaction_m,
        social_susceptibility=social_susceptibility,
        flag_prompt_social_susceptibility=flag_prompt_social_susceptibility,
    )
    backend = _thread_backend(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        image_detail=image_detail,
        debug_root=debug_root,
    )
    messages = openai_multimodal_messages(
        system_prompt=payload.system_prompt,
        user_prompt=payload.user_prompt,
        crop_data_uri=crop_data_uri,
        image_detail=image_detail,
    )
    current_messages = list(messages)
    attempts: list[dict[str, Any]] = []
    response_text = ""
    valid = False
    country: str | None = None
    raw_country: str | None = None
    error: str | None = None
    country_in_pool = False
    country_pool_match: str | None = None
    usage_rows: list[dict[str, Any]] = []
    for attempt_idx in range(max_parse_retries + 1):
        before = len(backend.usage_rows)
        response_text = backend._call_with_backoff(current_messages)
        if len(backend.usage_rows) > before:
            usage_rows.append(dict(backend.usage_rows[-1]))
        valid, country, raw_country, error, country_in_pool, country_pool_match = parse_response(
            response_text=response_text,
            variant=variant,
            countries=stimulus.countries,
            model=model,
            interaction_m=interaction_m,
            parse_mode=parse_mode,
            country_name_universe=country_name_universe,
        )
        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "messages": current_messages,
                "response_text": response_text,
                "valid": valid,
                "country": country,
                "raw_country": raw_country,
                "country_in_pool": country_in_pool,
                "country_pool_match": country_pool_match,
                "error": error,
            }
        )
        if valid or attempt_idx >= max_parse_retries:
            break
        current_messages = list(current_messages) + [
            {
                "role": "user",
                "content": retry_text_for_variant(
                    variant=variant,
                    countries=stimulus.countries,
                    model=model,
                    interaction_m=interaction_m,
                    error_text=error or "Invalid response",
                    parse_mode=parse_mode,
                ),
            }
        ]
    first_attempt = attempts[0]
    first_valid = bool(first_attempt["valid"])
    first_country = first_attempt.get("country")
    first_raw_country = first_attempt.get("raw_country")
    first_country_in_pool = bool(first_attempt.get("country_in_pool", False))
    first_country_pool_match = first_attempt.get("country_pool_match")
    first_error = first_attempt.get("error")
    usage = _sum_usage(model, usage_rows)
    crop_diag = stimulus.crop_diagnostics[agent_id]
    compatible_countries = [str(value) for value in crop_diag.get("compatible_countries", [])]
    scored_country = country_pool_match if parse_mode == "open_country" else country
    first_scored_country = first_country_pool_match if parse_mode == "open_country" else first_country
    compatible_with_crop = bool(valid and scored_country in compatible_countries)
    correct = bool(valid and scored_country == stimulus.truth_country)
    first_compatible_with_crop = bool(first_valid and first_scored_country in compatible_countries)
    first_correct = bool(first_valid and first_scored_country == stimulus.truth_country)
    estimated_cost = usage.get("estimated_cost_usd")
    row = {
        "task_index": task_index,
        "variant": variant.name,
        "variant_label": variant.label,
        "protocol": variant.protocol,
        "allowed_country_list": variant.allowed_country_list,
        "model": model,
        "N": stimulus.N,
        "seed": stimulus.seed,
        "agent_id": agent_id,
        "truth_country": stimulus.truth_country,
        "valid": valid,
        "country": country,
        "raw_country": raw_country,
        "country_in_pool": country_in_pool,
        "country_pool_match": country_pool_match,
        "correct": correct,
        "compatible_with_crop": compatible_with_crop,
        "error": error,
        "parse_mode": parse_mode,
        "country_name_universe": country_name_universe,
        "max_parse_retries": max_parse_retries,
        "attempt_count": len(attempts),
        "repaired_by_retry": bool((not first_valid) and valid),
        "first_valid": first_valid,
        "first_country": first_country,
        "first_raw_country": first_raw_country,
        "first_country_in_pool": first_country_in_pool,
        "first_country_pool_match": first_country_pool_match,
        "first_correct": first_correct,
        "first_compatible_with_crop": first_compatible_with_crop,
        "first_error": first_error,
        "actual_overlap": stimulus.actual_overlap,
        "compatible_country_count": int(crop_diag.get("compatible_country_count", 0) or 0),
        "compatible_countries_json": json.dumps(compatible_countries, ensure_ascii=True),
        "informativeness_label": str(crop_diag.get("informativeness_label", "invalid")),
        "informativeness_bits": crop_diag.get("informativeness_bits"),
        "informativeness_score": crop_diag.get("informativeness_score"),
        "truth_compatible": bool(crop_diag.get("truth_compatible", False)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "estimated_cost_usd": estimated_cost,
    }
    transcript = {
        "task_index": task_index,
        "variant": variant.name,
        "model": model,
        "N": stimulus.N,
        "seed": stimulus.seed,
        "agent_id": agent_id,
        "truth_country": stimulus.truth_country,
        "system_prompt": payload.system_prompt,
        "user_prompt": payload.user_prompt,
        "response_text": response_text,
        "attempts": attempts,
        "parsed_country": country,
        "raw_country": raw_country,
        "country_in_pool": country_in_pool,
        "country_pool_match": country_pool_match,
        "valid": valid,
        "correct": correct,
        "first_valid": first_valid,
        "first_country": first_country,
        "first_raw_country": first_raw_country,
        "first_country_in_pool": first_country_in_pool,
        "first_country_pool_match": first_country_pool_match,
        "first_correct": first_correct,
        "repaired_by_retry": bool((not first_valid) and valid),
        "parse_mode": parse_mode,
        "country_name_universe": country_name_universe,
        "error": error,
    }
    return row, transcript


def _bool_arg(text: str) -> bool:
    cleaned = text.strip().lower()
    if cleaned in {"1", "true", "yes", "on"}:
        return True
    if cleaned in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {text!r}")


def _csv_ints(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        values.append(int(cleaned))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def _csv_strings(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one value")
    return values


def _selected_variants(names: Iterable[str] | None) -> list[PromptVariant]:
    if names is None:
        return list(VARIANTS)
    requested = set(names)
    by_name = {variant.name: variant for variant in VARIANTS}
    unknown = sorted(requested - set(by_name))
    if unknown:
        valid = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown prompt variant(s): {unknown}. Valid variants: {valid}")
    return [variant for variant in VARIANTS if variant.name in requested]


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(row["variant"]),
        str(row["model"]),
        int(row["N"]),
        int(row["seed"]),
        int(row["agent_id"]),
    )


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    mode = "a" if append else "w"
    with path.open(mode) as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=True)
            handle.write("\n")


def _write_prompt_previews(
    *,
    out_dir: Path,
    countries: list[str],
    models: list[str],
    variants: list[PromptVariant],
    interaction_m: int,
    social_susceptibility: float,
    flag_prompt_social_susceptibility: bool,
) -> None:
    preview_dir = out_dir / "prompt_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model = models[0]
    for variant in variants:
        payload = build_prompt_payload(
            variant=variant,
            countries=countries,
            model=model,
            interaction_m=interaction_m,
            social_susceptibility=social_susceptibility,
            flag_prompt_social_susceptibility=flag_prompt_social_susceptibility,
        )
        text = (
            f"# {variant.label}\n\n"
            f"variant: {variant.name}\n"
            f"protocol: {variant.protocol}\n"
            f"allowed_country_list: {variant.allowed_country_list}\n\n"
            "## System\n\n"
            f"{payload.system_prompt}\n\n"
            "## User\n\n"
            f"{payload.user_prompt}\n"
        )
        (preview_dir / f"{variant.name}.md").write_text(text)


def write_summary_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame = frame.sort_values(["variant", "model", "N", "seed", "agent_id"]).reset_index(drop=True)
    frame.to_csv(out_dir / "iiq_rows.csv", index=False)

    seed_summary = (
        frame.assign(
            first_valid=frame["first_valid"] if "first_valid" in frame else frame["valid"],
            first_correct=frame["first_correct"] if "first_correct" in frame else frame["correct"],
            country_in_pool=frame["country_in_pool"] if "country_in_pool" in frame else frame["valid"],
            first_country_in_pool=(
                frame["first_country_in_pool"]
                if "first_country_in_pool" in frame
                else (frame["first_valid"] if "first_valid" in frame else frame["valid"])
            ),
            off_pool_country=(
                frame["valid"].astype(bool)
                & ~(
                    frame["country_in_pool"].astype(bool)
                    if "country_in_pool" in frame
                    else frame["valid"].astype(bool)
                )
            ),
            first_off_pool_country=(
                (frame["first_valid"].astype(bool) if "first_valid" in frame else frame["valid"].astype(bool))
                & ~(
                    frame["first_country_in_pool"].astype(bool)
                    if "first_country_in_pool" in frame
                    else (
                        frame["first_valid"].astype(bool)
                        if "first_valid" in frame
                        else frame["valid"].astype(bool)
                    )
                )
            ),
            first_compatible_with_crop=(
                frame["first_compatible_with_crop"]
                if "first_compatible_with_crop" in frame
                else frame["compatible_with_crop"]
            ),
            repaired_by_retry=frame["repaired_by_retry"] if "repaired_by_retry" in frame else False,
            attempt_count=frame["attempt_count"] if "attempt_count" in frame else 1,
        )
        .groupby(
            [
                "variant",
                "variant_label",
                "protocol",
                "allowed_country_list",
                "model",
                "N",
                "seed",
            ],
            dropna=False,
        )
        .agg(
            n_agents=("agent_id", "count"),
            valid_rate=("valid", "mean"),
            iiq=("correct", "mean"),
            first_valid_rate=("first_valid", "mean"),
            first_iiq=("first_correct", "mean"),
            country_in_pool_rate=("country_in_pool", "mean"),
            first_country_in_pool_rate=("first_country_in_pool", "mean"),
            off_pool_country_rate=("off_pool_country", "mean"),
            first_off_pool_country_rate=("first_off_pool_country", "mean"),
            retry_repair_rate=("repaired_by_retry", "mean"),
            mean_attempt_count=("attempt_count", "mean"),
            crop_compatibility_rate=("compatible_with_crop", "mean"),
            first_crop_compatibility_rate=("first_compatible_with_crop", "mean"),
            mean_compatible_country_count=("compatible_country_count", "mean"),
            actual_overlap=("actual_overlap", "mean"),
            estimated_cost_usd=("estimated_cost_usd", "sum"),
        )
        .reset_index()
    )
    seed_summary.to_csv(out_dir / "iiq_by_seed.csv", index=False)

    condition_summary = (
        seed_summary.groupby(
            [
                "variant",
                "variant_label",
                "protocol",
                "allowed_country_list",
                "model",
                "N",
            ],
            dropna=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            n_agents=("n_agents", "sum"),
            valid_rate=("valid_rate", "mean"),
            iiq_mean=("iiq", "mean"),
            iiq_sd_across_seeds=("iiq", "std"),
            first_valid_rate=("first_valid_rate", "mean"),
            first_iiq_mean=("first_iiq", "mean"),
            first_iiq_sd_across_seeds=("first_iiq", "std"),
            country_in_pool_rate=("country_in_pool_rate", "mean"),
            first_country_in_pool_rate=("first_country_in_pool_rate", "mean"),
            off_pool_country_rate=("off_pool_country_rate", "mean"),
            first_off_pool_country_rate=("first_off_pool_country_rate", "mean"),
            retry_repair_rate=("retry_repair_rate", "mean"),
            mean_attempt_count=("mean_attempt_count", "mean"),
            crop_compatibility_rate=("crop_compatibility_rate", "mean"),
            first_crop_compatibility_rate=("first_crop_compatibility_rate", "mean"),
            mean_compatible_country_count=("mean_compatible_country_count", "mean"),
            actual_overlap=("actual_overlap", "mean"),
            estimated_cost_usd=("estimated_cost_usd", "sum"),
        )
        .reset_index()
    )
    condition_summary["iiq_se"] = condition_summary.apply(
        lambda row: (
            float(row["iiq_sd_across_seeds"]) / math.sqrt(float(row["n_seeds"]))
            if pd.notna(row["iiq_sd_across_seeds"]) and row["n_seeds"] > 0
            else 0.0
        ),
        axis=1,
    )
    condition_summary["first_iiq_se"] = condition_summary.apply(
        lambda row: (
            float(row["first_iiq_sd_across_seeds"]) / math.sqrt(float(row["n_seeds"]))
            if pd.notna(row["first_iiq_sd_across_seeds"]) and row["n_seeds"] > 0
            else 0.0
        ),
        axis=1,
    )
    condition_summary.to_csv(out_dir / "iiq_summary.csv", index=False)

    try:
        pivot = condition_summary.pivot_table(
            index=["model", "N"],
            columns="variant",
            values="iiq_mean",
        ).reset_index()
        if {"broadcast_current", "broadcast_allowed"}.issubset(pivot.columns):
            pivot["broadcast_allowed_minus_current"] = (
                pivot["broadcast_allowed"] - pivot["broadcast_current"]
            )
        if {"flag_current", "flag_open"}.issubset(pivot.columns):
            pivot["flag_current_minus_open"] = pivot["flag_current"] - pivot["flag_open"]
        if {"broadcast_current", "flag_open"}.issubset(pivot.columns):
            pivot["broadcast_current_minus_flag_open"] = (
                pivot["broadcast_current"] - pivot["flag_open"]
            )
        if {"broadcast_allowed", "flag_current"}.issubset(pivot.columns):
            pivot["broadcast_allowed_minus_flag_current"] = (
                pivot["broadcast_allowed"] - pivot["flag_current"]
            )
        pivot.to_csv(out_dir / "iiq_variant_deltas.csv", index=False)
    except Exception:
        pass

    _write_markdown_report(
        out_dir=out_dir,
        condition_summary=condition_summary,
    )
    _plot_summary(out_dir=out_dir, condition_summary=condition_summary)


def _write_markdown_report(*, out_dir: Path, condition_summary: pd.DataFrame) -> None:
    lines = [
        "# Flag Game Prompt IIQ Comparison",
        "",
        "This run compares individual information quality (IIQ) across prompt variants.",
        "No social rounds are run.",
        "",
        "Important interpretation note: closed-pool runs validate against the configured country pool.",
        "Open-country runs validate against the configured country-name universe and separately log whether the answer matched the rendered truth pool.",
        "The `first_*` columns preserve first-shot answers before parser repair.",
        "When `max_parse_retries > 0`, the final `iiq_*` columns report outcomes after retry repair.",
        "",
        "## Condition Summary",
        "",
    ]
    display = condition_summary.copy()
    for column in [
        "valid_rate",
        "iiq_mean",
        "iiq_sd_across_seeds",
        "iiq_se",
        "first_valid_rate",
        "first_iiq_mean",
        "first_iiq_sd_across_seeds",
        "first_iiq_se",
        "country_in_pool_rate",
        "first_country_in_pool_rate",
        "off_pool_country_rate",
        "first_off_pool_country_rate",
        "retry_repair_rate",
        "mean_attempt_count",
        "crop_compatibility_rate",
        "first_crop_compatibility_rate",
        "actual_overlap",
        "estimated_cost_usd",
    ]:
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
            )
    lines.extend(_dataframe_to_markdown_lines(display))
    lines.append("")
    (out_dir / "iiq_report.md").write_text("\n".join(lines))


def _dataframe_to_markdown_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = list(frame.columns)
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows))
        for idx, column in enumerate(columns)
    ]

    def fmt(values: list[str]) -> str:
        cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
        return "| " + " | ".join(cells) + " |"

    header = fmt([str(column) for column in columns])
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    return [header, divider, *(fmt(row) for row in rows)]


def _plot_summary(*, out_dir: Path, condition_summary: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    variants = [variant.name for variant in VARIANTS if variant.name in set(condition_summary["variant"])]
    labels = {
        variant.name: variant.label.replace(" Countries", "\nCountries").replace(" Current", "\nCurrent")
        for variant in VARIANTS
    }
    models = list(condition_summary["model"].drop_duplicates())
    n_values = sorted(int(value) for value in condition_summary["N"].drop_duplicates())
    if not variants or not models or not n_values:
        return

    fig, axes = plt.subplots(
        len(models),
        len(n_values),
        figsize=(4.0 * len(n_values), 3.0 * len(models)),
        sharey=True,
        squeeze=False,
    )
    color_map = {
        "broadcast_current": "#D5973B",
        "broadcast_allowed": "#5B8CC0",
        "flag_current": "#3E9A77",
        "flag_open": "#8A6FB0",
    }
    for row_idx, model in enumerate(models):
        for col_idx, n_value in enumerate(n_values):
            ax = axes[row_idx][col_idx]
            subset = condition_summary[
                (condition_summary["model"] == model) & (condition_summary["N"] == n_value)
            ].set_index("variant")
            x = np.arange(len(variants))
            y = [float(subset.loc[variant, "iiq_mean"]) if variant in subset.index else 0.0 for variant in variants]
            yerr = [float(subset.loc[variant, "iiq_se"]) if variant in subset.index else 0.0 for variant in variants]
            ax.bar(
                x,
                y,
                yerr=yerr,
                color=[color_map.get(variant, "#999999") for variant in variants],
                capsize=3,
            )
            ax.set_title(f"{model}, N={n_value}")
            ax.set_xticks(x)
            ax.set_xticklabels([labels.get(variant, variant) for variant in variants], rotation=25, ha="right")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.2)
            if col_idx == 0:
                ax.set_ylabel("IIQ")
    fig.tight_layout()
    fig.savefig(out_dir / "iiq_prompt_comparison.png", dpi=220)
    fig.savefig(out_dir / "iiq_prompt_comparison.pdf")
    plt.close(fig)

    if "first_iiq_mean" not in condition_summary:
        return
    fig, axes = plt.subplots(
        len(models),
        len(n_values),
        figsize=(4.0 * len(n_values), 3.0 * len(models)),
        sharey=True,
        squeeze=False,
    )
    for row_idx, model in enumerate(models):
        for col_idx, n_value in enumerate(n_values):
            ax = axes[row_idx][col_idx]
            subset = condition_summary[
                (condition_summary["model"] == model) & (condition_summary["N"] == n_value)
            ].set_index("variant")
            x = np.arange(len(variants))
            first_y = [
                float(subset.loc[variant, "first_iiq_mean"]) if variant in subset.index else 0.0
                for variant in variants
            ]
            final_y = [
                float(subset.loc[variant, "iiq_mean"]) if variant in subset.index else 0.0
                for variant in variants
            ]
            ax.plot(x, first_y, marker="o", linestyle="--", color="#A63D40", label="first shot")
            ax.plot(x, final_y, marker="o", linestyle="-", color=COLORS["primary"] if "COLORS" in globals() else "#1F3A5F", label="after retry")
            ax.set_title(f"{model}, N={n_value}")
            ax.set_xticks(x)
            ax.set_xticklabels([labels.get(variant, variant) for variant in variants], rotation=25, ha="right")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.2)
            if col_idx == 0:
                ax.set_ylabel("IIQ")
            if row_idx == 0 and col_idx == len(n_values) - 1:
                ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "iiq_first_vs_retry_repaired.png", dpi=220)
    fig.savefig(out_dir / "iiq_first_vs_retry_repaired.pdf")
    plt.close(fig)


def _save_stimulus_artifacts(
    *,
    out_dir: Path,
    stimulus: Stimulus,
    render_scale: int,
) -> None:
    stimulus_dir = out_dir / "stimuli" / f"N{stimulus.N}" / f"seed_{stimulus.seed:04d}"
    stimulus_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for agent_id, assignment in enumerate(stimulus.assignments):
        rows.append(
            {
                "N": stimulus.N,
                "seed": stimulus.seed,
                "agent_id": agent_id,
                "truth_country": stimulus.truth_country,
                "render_scale": render_scale,
                **assignment.to_dict(),
                **{
                    key: value
                    for key, value in stimulus.crop_diagnostics[agent_id].items()
                    if key
                    in {
                        "truth_compatible",
                        "compatible_country_count",
                        "informativeness_label",
                        "informativeness_bits",
                        "informativeness_score",
                    }
                },
            }
        )
        save_png(stimulus_dir / f"agent_{agent_id:02d}_crop.png", stimulus.crop_images[agent_id])
    pd.DataFrame(rows).to_csv(stimulus_dir / "stimulus_manifest.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=_csv_strings, default=["gpt-4o", "gpt-5.4"])
    parser.add_argument("--n-values", type=_csv_ints, default=[8, 16])
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--country-pool", default="stripe_plus_real_triangle_28")
    parser.add_argument("--canvas-width", type=int, default=24)
    parser.add_argument("--canvas-height", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=6)
    parser.add_argument("--tile-height", type=int, default=4)
    parser.add_argument("--render-scale", type=int, default=25)
    parser.add_argument("--observation-overlap", type=float, default=None)
    parser.add_argument("--overlap-search-trials", type=int, default=200)
    parser.add_argument("--interaction-m", type=int, default=3)
    parser.add_argument("--social-susceptibility", type=float, default=0.5)
    parser.add_argument("--flag-prompt-social-susceptibility", type=_bool_arg, default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--image-detail", default="high")
    parser.add_argument("--agent-workers", type=int, default=4)
    parser.add_argument("--max-parse-retries", type=int, default=0)
    parser.add_argument(
        "--parse-mode",
        choices=["closed_pool", "open_country"],
        default="closed_pool",
        help=(
            "closed_pool enforces membership in the configured country pool. "
            "open_country accepts any recognized country name from "
            "--country-name-universe, logs whether it matched the rendered truth "
            "pool, and retries only malformed or non-country output."
        ),
    )
    parser.add_argument(
        "--country-name-universe",
        default="world_rectangle",
        help="Country-name universe used to validate open_country responses.",
    )
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--resume", type=_bool_arg, default=True)
    parser.add_argument("--save-stimulus-images", action="store_true")
    parser.add_argument("--write-prompt-previews-only", action="store_true")
    parser.add_argument("--limit-calls", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = _selected_variants(args.variant)
    models: list[str] = args.models
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    pool = get_country_pool(args.country_pool)
    countries = [flag.country for flag in pool]
    _write_prompt_previews(
        out_dir=out_dir,
        countries=countries,
        models=models,
        variants=variants,
        interaction_m=args.interaction_m,
        social_susceptibility=args.social_susceptibility,
        flag_prompt_social_susceptibility=args.flag_prompt_social_susceptibility,
    )
    spec = {
        "models": models,
        "n_values": args.n_values,
        "start_seed": args.start_seed,
        "num_seeds": args.num_seeds,
        "seeds": seeds,
        "country_pool": args.country_pool,
        "canvas_width": args.canvas_width,
        "canvas_height": args.canvas_height,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "render_scale": args.render_scale,
        "observation_overlap": args.observation_overlap,
        "overlap_search_trials": args.overlap_search_trials,
        "interaction_m": args.interaction_m,
        "social_susceptibility": args.social_susceptibility,
        "flag_prompt_social_susceptibility": args.flag_prompt_social_susceptibility,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "image_detail": args.image_detail,
        "max_parse_retries": args.max_parse_retries,
        "parse_mode": args.parse_mode,
        "country_name_universe": args.country_name_universe,
        "variants": [variant.__dict__ for variant in variants],
        "first_shot_no_retry": args.max_parse_retries == 0,
    }
    (out_dir / "iiq_prompt_comparison_spec.json").write_text(json.dumps(spec, indent=2))
    if args.write_prompt_previews_only:
        print(f"Wrote prompt previews to {out_dir / 'prompt_previews'}")
        return

    use_fast_crop_diagnostics = all(
        isinstance(flag, StripeFlag) and flag.triangle_color is None for flag in pool
    )
    compatibility_cache: dict[str, set[bytes]] = {}
    if not use_fast_crop_diagnostics:
        compatibility_cache = build_crop_compatibility_cache(
            pool,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            render_scale=args.render_scale,
        )

    row_path = out_dir / "iiq_rows.csv"
    existing_rows = _load_existing_rows(row_path) if args.resume else []
    completed_keys = {_row_key(row) for row in existing_rows}
    transcript_path = out_dir / "iiq_transcripts.jsonl"

    tasks: list[dict[str, Any]] = []
    task_index = 0
    for n_value in args.n_values:
        for seed in seeds:
            stimulus = generate_stimulus(
                N=n_value,
                seed=seed,
                country_pool=args.country_pool,
                canvas_width=args.canvas_width,
                canvas_height=args.canvas_height,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                render_scale=args.render_scale,
                observation_overlap=args.observation_overlap,
                overlap_search_trials=args.overlap_search_trials,
                compatibility_cache=compatibility_cache,
            )
            if args.save_stimulus_images:
                _save_stimulus_artifacts(
                    out_dir=out_dir,
                    stimulus=stimulus,
                    render_scale=args.render_scale,
                )
            crop_data_uris: dict[int, str] = {}
            for agent_id, crop in enumerate(stimulus.crop_images):
                for variant in variants:
                    for model in models:
                        key = (variant.name, model, n_value, seed, agent_id)
                        if key in completed_keys:
                            continue
                        if agent_id not in crop_data_uris:
                            crop_data_uris[agent_id] = image_to_data_uri(crop)
                        tasks.append(
                            {
                                "task_index": task_index,
                                "stimulus": stimulus,
                                "crop_data_uri": crop_data_uris[agent_id],
                                "variant": variant,
                                "model": model,
                                "agent_id": agent_id,
                            }
                        )
                        task_index += 1
                        if args.limit_calls is not None and len(tasks) >= args.limit_calls:
                            break
                    if args.limit_calls is not None and len(tasks) >= args.limit_calls:
                        break
                if args.limit_calls is not None and len(tasks) >= args.limit_calls:
                    break
            if args.limit_calls is not None and len(tasks) >= args.limit_calls:
                break
        if args.limit_calls is not None and len(tasks) >= args.limit_calls:
            break

    print(f"Existing completed rows: {len(existing_rows)}")
    print(f"New IIQ calls to run: {len(tasks)}")
    if not tasks:
        write_summary_outputs(out_dir, existing_rows)
        print(f"No new calls needed. Summary refreshed under {out_dir}")
        return

    new_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.agent_workers) as executor:
        futures = [
            executor.submit(
                run_one_call,
                task_index=task["task_index"],
                stimulus=task["stimulus"],
                crop_data_uri=task["crop_data_uri"],
                variant=task["variant"],
                model=task["model"],
                agent_id=task["agent_id"],
                interaction_m=args.interaction_m,
                social_susceptibility=args.social_susceptibility,
                flag_prompt_social_susceptibility=args.flag_prompt_social_susceptibility,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                image_detail=args.image_detail,
                debug_root=out_dir / "debug",
                max_parse_retries=args.max_parse_retries,
                parse_mode=args.parse_mode,
                country_name_universe=args.country_name_universe,
            )
            for task in tasks
        ]
        completed = 0
        with transcript_path.open("a") as transcript_handle:
            for future in as_completed(futures):
                row, transcript = future.result()
                new_rows.append(row)
                json.dump(transcript, transcript_handle, ensure_ascii=True)
                transcript_handle.write("\n")
                completed += 1
                if completed == 1 or completed % 50 == 0 or completed == len(tasks):
                    print(f"Completed {completed}/{len(tasks)} IIQ calls")

    all_rows = existing_rows + new_rows
    write_summary_outputs(out_dir, all_rows)
    print(f"Wrote IIQ rows and summaries under {out_dir}")


if __name__ == "__main__":
    main()

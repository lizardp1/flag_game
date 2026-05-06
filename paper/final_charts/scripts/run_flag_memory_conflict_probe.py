#!/usr/bin/env python3
"""Probe private visual evidence against synthetic transcript-memory majorities.

This is a single-agent probe for the Flag Game. It removes group interaction and
directly tests whether a model follows private visual evidence or synthetic
social evidence. The main comparison is GPT-4o versus GPT-5.4, and m=1 versus m=3:

- m=1: memory entries are country labels only; output is {"country": ...}
- m=3: memory entries include country + reason; output is {"country": ..., "reason": ...}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from nnd.backends.parsing import ParseError
from nnd.flag_game.backend import build_backend
from nnd.flag_game.catalog import COLOR_MAP, FlagSpec, StripeFlag, get_country_pool
from nnd.flag_game.crops import CropBox, all_crop_boxes, crop_image, scale_crop_box
from nnd.flag_game.diagnostics import build_crop_compatibility_cache, describe_crop_informativeness
from nnd.flag_game.parsing import InteractionMessage
from nnd.flag_game.render import render_flag, save_png
from nnd.flag_game.runner import choose_default_backend


DEFAULT_OUT = ROOT / "results" / "flag_game" / "memory_conflict_probe"
DEFAULT_MODELS = ["gpt-4o", "gpt-5.4"]
DEFAULT_TRUTHS = ["Czech Republic", "Peru", "Guinea", "Bahamas"]
DEFAULT_M_VALUES = [1, 3]
DEFAULT_CROP_CONDITIONS = ["diagnostic_true", "ambiguous_true"]
DEFAULT_LURE_RELATIONS = ["compatible", "incompatible"]
DEFAULT_COUNTS = list(range(9))

MODEL_COLORS = {
    "gpt-4o": "#2563eb",
    "gpt-5.4": "#d97706",
}
M_LINESTYLES = {
    1: "-",
    3: "--",
}
RESPONSE_TYPE_ORDER = [
    "private_evidence",
    "private_compatible_other",
    "unsupported_other",
    "social_evidence",
]
RESPONSE_TYPE_LABELS = {
    "private_evidence": "Private target country",
    "private_compatible_other": "Other crop-compatible country",
    "unsupported_other": "Unsupported other",
    "social_evidence": "Social evidence",
}
RESPONSE_TYPE_COLORS = {
    "private_evidence": "#2563eb",
    "private_compatible_other": "#60a5fa",
    "unsupported_other": "#9ca3af",
    "social_evidence": "#d97706",
}
ALIGNMENT_TYPE_ORDER = [
    "private_evidence",
    "social_evidence",
    "other_private_and_social",
    "other_private_only",
    "other_social_only",
    "unsupported_other",
]
ALIGNMENT_STACK_ORDER = [
    "private_evidence",
    "other_private_and_social",
    "other_private_only",
    "other_social_only",
    "unsupported_other",
    "social_evidence",
]
ALIGNMENT_TYPE_LABELS = {
    "private_evidence": "Private target country",
    "social_evidence": "Social evidence",
    "other_private_and_social": "Other: private + social compatible",
    "other_private_only": "Other: private compatible",
    "other_social_only": "Other: social compatible",
    "unsupported_other": "Other: incompatible",
}
ALIGNMENT_TYPE_COLORS = {
    "private_evidence": "#2563eb",
    "social_evidence": "#d97706",
    "other_private_and_social": "#14b8a6",
    "other_private_only": "#60a5fa",
    "other_social_only": "#f59e0b",
    "unsupported_other": "#9ca3af",
}
EPS = 1e-12


def _model_short_label(model: str) -> str:
    if model == "gpt-4o":
        return "gpt-4o"
    if model == "gpt-5.4":
        return "gpt-5.4"
    return str(model)


@dataclass(frozen=True)
class Stimulus:
    truth_country: str
    crop_condition: str
    crop_box: CropBox
    crop_image: Any
    diagnostic: dict[str, Any]


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    model: str
    m: int
    truth_country: str
    lure_country: str
    lure_relation: str
    crop_condition: str
    false_memory_count: int
    true_memory_count: int
    rep: int
    seed: int
    allowed_countries: tuple[str, ...]
    memory_lines: tuple[str, ...]
    crop_box: CropBox
    crop_diagnostic: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-agent flag probe where private visual evidence conflicts "
            "with synthetic social-evidence majorities."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--backend", default=None, choices=["openai", "anthropic", "scripted"])
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--m", action="append", dest="m_values", type=int, choices=[1, 3], default=None)
    parser.add_argument("--country-pool", default="stripe_plus_real_triangle_28")
    parser.add_argument("--truth-country", action="append", dest="truth_countries", default=None)
    parser.add_argument(
        "--crop-condition",
        action="append",
        dest="crop_conditions",
        choices=DEFAULT_CROP_CONDITIONS,
        default=None,
    )
    parser.add_argument(
        "--lure-relation",
        action="append",
        dest="lure_relations",
        choices=DEFAULT_LURE_RELATIONS,
        default=None,
    )
    parser.add_argument(
        "--memory-counts",
        default=",".join(str(value) for value in DEFAULT_COUNTS),
        help="Comma-separated social-evidence counts. Default: 0,1,2,3,4,5,6,7,8.",
    )
    parser.add_argument("--h", type=int, default=8, help="Synthetic memory size.")
    parser.add_argument("--replicates", type=int, default=5, help="Order/list shuffle replicates per cell.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canvas-width", type=int, default=24)
    parser.add_argument("--canvas-height", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=6)
    parser.add_argument("--tile-height", type=int, default=4)
    parser.add_argument("--render-scale", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--image-detail", default="high", choices=["auto", "low", "high", "original"])
    parser.add_argument(
        "--prompt-social-susceptibility",
        action="store_true",
        help="Include the explicit social-susceptibility prompt line. Off by default for a clean probe.",
    )
    parser.add_argument("--social-susceptibility", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Write the plan and exit before API calls.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild CSV summaries and plots.")
    parser.add_argument("--force", action="store_true", help="Rerun trials already present in results.jsonl.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-save-crops", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of planned trials, useful for smoke tests.",
    )
    return parser.parse_args()


def _parse_memory_counts(raw: str, h: int) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value > h:
            raise ValueError(f"false-memory count must be in [0, H={h}], got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one false-memory count is required")
    return sorted(set(values))


def _safe_model_dir(model: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._")
    return cleaned or "model"


def _country_names(pool: list[FlagSpec]) -> list[str]:
    return [flag.country for flag in pool]


def _country_lookup(pool: list[FlagSpec]) -> dict[str, FlagSpec]:
    return {flag.country: flag for flag in pool}


def _unique_color_count(image: Any) -> int:
    return int(np.unique(image.reshape(-1, image.shape[-1])[:, :3], axis=0).shape[0])


def _triangle_visible(flag: FlagSpec, image: Any) -> bool:
    if not isinstance(flag, StripeFlag) or flag.triangle_color is None:
        return False
    triangle_rgb = COLOR_MAP[flag.triangle_color]
    return bool((image[:, :, :3] == triangle_rgb).all(axis=2).any())


def _select_stimulus(
    *,
    truth_flag: FlagSpec,
    crop_condition: str,
    countries: list[str],
    compatibility_cache: dict[str, set[bytes]],
    canvas_width: int,
    canvas_height: int,
    tile_width: int,
    tile_height: int,
    render_scale: int,
) -> Stimulus:
    full_image = render_flag(
        truth_flag,
        width=canvas_width * render_scale,
        height=canvas_height * render_scale,
    )
    candidates: list[tuple[CropBox, Any, dict[str, Any], int, bool]] = []
    for box in all_crop_boxes(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        tile_width=tile_width,
        tile_height=tile_height,
    ):
        scaled_box = scale_crop_box(box, render_scale)
        image = crop_image(full_image, scaled_box)
        diagnostic = describe_crop_informativeness(
            image,
            country_order=countries,
            compatibility_cache=compatibility_cache,
        )
        if truth_flag.country not in diagnostic["compatible_countries"]:
            continue
        candidates.append(
            (
                box,
                image,
                diagnostic,
                _unique_color_count(image),
                _triangle_visible(truth_flag, image),
            )
        )
    if not candidates:
        raise RuntimeError(f"No truth-compatible crops found for {truth_flag.country}")

    if crop_condition == "diagnostic_true":
        ranked = sorted(
            candidates,
            key=lambda item: (
                int(item[2]["compatible_country_count"]),
                -int(item[4]),
                -float(item[2]["informativeness_bits"] or 0.0),
                -int(item[3]),
                int(item[0].top),
                int(item[0].left),
            ),
        )
    elif crop_condition == "ambiguous_true":
        ranked = sorted(
            candidates,
            key=lambda item: (
                -int(item[2]["compatible_country_count"]),
                int(item[3]),
                float(item[2]["informativeness_bits"] or 0.0),
                int(item[0].top),
                int(item[0].left),
            ),
        )
    else:
        raise ValueError(f"Unsupported crop_condition: {crop_condition}")

    box, image, diagnostic, _color_count, _has_triangle = ranked[0]
    return Stimulus(
        truth_country=truth_flag.country,
        crop_condition=crop_condition,
        crop_box=box,
        crop_image=image,
        diagnostic=diagnostic,
    )


def _flag_phrase(flag: FlagSpec) -> str:
    if isinstance(flag, StripeFlag):
        color_phrase = "-".join(color.replace("_", " ") for color in flag.colors)
        if flag.triangle_color:
            return (
                f"a {flag.triangle_color.replace('_', ' ')} hoist triangle "
                f"with {flag.orientation} {color_phrase} bands"
            )
        return f"{flag.orientation} {color_phrase} stripes"
    return "its visible national flag pattern"


def _memory_line(
    *,
    country: str,
    flag: FlagSpec,
    m: int,
    index: int,
    truth_aligned: bool,
) -> str:
    if m == 1:
        return country

    phrase = _flag_phrase(flag)
    if truth_aligned:
        templates = [
            "The visible crop is most consistent with {phrase}.",
            "I would choose {country}; the crop supports {phrase}.",
            "{country} fits the visible colors and geometry best.",
            "The crop evidence points toward {country}.",
        ]
    else:
        templates = [
            "This looks like {country} because it matches {phrase}.",
            "I would choose {country}; the visible colors fit {phrase}.",
            "{country} seems like the best match from the crop.",
            "The crop evidence points toward {country}.",
        ]
    reason = templates[index % len(templates)].format(country=country, phrase=phrase)
    return f"{country} | {reason}"


def _resolve_lure(
    *,
    truth_country: str,
    relation: str,
    compatible_countries: Iterable[str],
    countries: list[str],
    rng: random.Random,
) -> str | None:
    compatible = [country for country in compatible_countries if country != truth_country]
    incompatible = [
        country
        for country in countries
        if country != truth_country and country not in set(compatible_countries)
    ]
    if relation == "compatible":
        if not compatible:
            return None
        return rng.choice(sorted(compatible))
    if relation == "incompatible":
        if not incompatible:
            return None
        return rng.choice(sorted(incompatible))
    raise ValueError(f"Unsupported lure relation: {relation}")


def _build_memory_lines(
    *,
    truth_country: str,
    lure_country: str,
    lookup: dict[str, FlagSpec],
    m: int,
    false_count: int,
    h: int,
    rng: random.Random,
) -> tuple[str, ...]:
    true_count = h - false_count
    rows: list[str] = []
    for index in range(false_count):
        rows.append(
            _memory_line(
                country=lure_country,
                flag=lookup[lure_country],
                m=m,
                index=index,
                truth_aligned=False,
            )
        )
    for index in range(true_count):
        rows.append(
            _memory_line(
                country=truth_country,
                flag=lookup[truth_country],
                m=m,
                index=index,
                truth_aligned=True,
            )
        )
    rng.shuffle(rows)
    return tuple(rows)


def _trial_id(
    *,
    model: str,
    m: int,
    truth_country: str,
    crop_condition: str,
    lure_relation: str,
    lure_country: str,
    false_memory_count: int,
    rep: int,
) -> str:
    parts = [
        _safe_model_dir(model),
        f"m{m}",
        re.sub(r"[^A-Za-z0-9]+", "", truth_country).lower(),
        crop_condition,
        lure_relation,
        re.sub(r"[^A-Za-z0-9]+", "", lure_country).lower(),
        f"k{false_memory_count}",
        f"r{rep}",
    ]
    return "__".join(parts)


def build_trial_plan(args: argparse.Namespace) -> tuple[list[TrialSpec], dict[tuple[str, str], Stimulus]]:
    models = args.models or list(DEFAULT_MODELS)
    m_values = args.m_values or list(DEFAULT_M_VALUES)
    truth_countries = args.truth_countries or list(DEFAULT_TRUTHS)
    crop_conditions = args.crop_conditions or list(DEFAULT_CROP_CONDITIONS)
    lure_relations = args.lure_relations or list(DEFAULT_LURE_RELATIONS)
    memory_counts = _parse_memory_counts(args.memory_counts, args.h)

    pool = get_country_pool(args.country_pool)
    countries = _country_names(pool)
    lookup = _country_lookup(pool)
    missing_truths = [country for country in truth_countries if country not in lookup]
    if missing_truths:
        raise ValueError(f"Truth countries not in pool {args.country_pool}: {missing_truths}")

    compatibility_cache = build_crop_compatibility_cache(
        pool,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        render_scale=args.render_scale,
    )

    stimuli: dict[tuple[str, str], Stimulus] = {}
    for truth_country in truth_countries:
        truth_flag = lookup[truth_country]
        for crop_condition in crop_conditions:
            stimuli[(truth_country, crop_condition)] = _select_stimulus(
                truth_flag=truth_flag,
                crop_condition=crop_condition,
                countries=countries,
                compatibility_cache=compatibility_cache,
                canvas_width=args.canvas_width,
                canvas_height=args.canvas_height,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                render_scale=args.render_scale,
            )

    specs: list[TrialSpec] = []
    for model in models:
        for m in m_values:
            for truth_country in truth_countries:
                for crop_condition in crop_conditions:
                    stimulus = stimuli[(truth_country, crop_condition)]
                    compatible_countries = tuple(stimulus.diagnostic["compatible_countries"])
                    for lure_relation in lure_relations:
                        lure_rng = random.Random(
                            f"{args.seed}:{truth_country}:{crop_condition}:{lure_relation}"
                        )
                        lure_country = _resolve_lure(
                            truth_country=truth_country,
                            relation=lure_relation,
                            compatible_countries=compatible_countries,
                            countries=countries,
                            rng=lure_rng,
                        )
                        if lure_country is None:
                            continue
                        for false_count in memory_counts:
                            for rep in range(args.replicates):
                                trial_seed = (
                                    args.seed * 1_000_003
                                    + len(specs) * 97
                                    + false_count * 31
                                    + rep
                                )
                                rng = random.Random(trial_seed)
                                allowed = list(countries)
                                rng.shuffle(allowed)
                                memory_lines = _build_memory_lines(
                                    truth_country=truth_country,
                                    lure_country=lure_country,
                                    lookup=lookup,
                                    m=m,
                                    false_count=false_count,
                                    h=args.h,
                                    rng=rng,
                                )
                                specs.append(
                                    TrialSpec(
                                        trial_id=_trial_id(
                                            model=model,
                                            m=m,
                                            truth_country=truth_country,
                                            crop_condition=crop_condition,
                                            lure_relation=lure_relation,
                                            lure_country=lure_country,
                                            false_memory_count=false_count,
                                            rep=rep,
                                        ),
                                        model=model,
                                        m=m,
                                        truth_country=truth_country,
                                        lure_country=lure_country,
                                        lure_relation=lure_relation,
                                        crop_condition=crop_condition,
                                        false_memory_count=false_count,
                                        true_memory_count=args.h - false_count,
                                        rep=rep,
                                        seed=trial_seed,
                                        allowed_countries=tuple(allowed),
                                        memory_lines=memory_lines,
                                        crop_box=stimulus.crop_box,
                                        crop_diagnostic=stimulus.diagnostic,
                                    )
                                )
    if args.limit is not None:
        specs = specs[: args.limit]
    return specs, stimuli


def _existing_trial_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            trial_id = row.get("trial_id")
            if isinstance(trial_id, str):
                ids.add(trial_id)
    return ids


def write_plan(
    *,
    out_dir: Path,
    specs: list[TrialSpec],
    stimuli: dict[tuple[str, str], Stimulus],
    save_crops: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = []
    for spec in specs:
        plan_rows.append(
            {
                "trial_id": spec.trial_id,
                "model": spec.model,
                "m": spec.m,
                "truth_country": spec.truth_country,
                "lure_country": spec.lure_country,
                "lure_relation": spec.lure_relation,
                "crop_condition": spec.crop_condition,
                "false_memory_count": spec.false_memory_count,
                "true_memory_count": spec.true_memory_count,
                "rep": spec.rep,
                "seed": spec.seed,
                **{f"crop_{key}": value for key, value in spec.crop_box.to_dict().items()},
                "crop_compatible_country_count": spec.crop_diagnostic["compatible_country_count"],
                "crop_compatible_countries": json.dumps(spec.crop_diagnostic["compatible_countries"]),
                "crop_informativeness_bits": spec.crop_diagnostic["informativeness_bits"],
            }
        )
    pd.DataFrame(plan_rows).to_csv(out_dir / "trial_plan.csv", index=False)

    stimulus_rows = []
    for (truth_country, crop_condition), stimulus in sorted(stimuli.items()):
        stimulus_rows.append(
            {
                "truth_country": truth_country,
                "crop_condition": crop_condition,
                **stimulus.crop_box.to_dict(),
                "compatible_country_count": stimulus.diagnostic["compatible_country_count"],
                "compatible_countries": json.dumps(stimulus.diagnostic["compatible_countries"]),
                "informativeness_bits": stimulus.diagnostic["informativeness_bits"],
                "informativeness_label": stimulus.diagnostic["informativeness_label"],
            }
        )
        if save_crops:
            crop_path = (
                out_dir
                / "artifacts"
                / re.sub(r"[^A-Za-z0-9]+", "_", truth_country).strip("_").lower()
                / f"{crop_condition}.png"
            )
            save_png(crop_path, stimulus.crop_image)
    pd.DataFrame(stimulus_rows).to_csv(out_dir / "stimuli.csv", index=False)


def _reason_flags(reason: str | None, chose_lure: bool, lure_relation: str) -> dict[str, bool]:
    text = (reason or "").lower()
    cites_memory = bool(
        re.search(r"\b(memory|transcript|message|messages|player|players|others|consensus|prior|previous)\b", text)
    )
    cites_crop = bool(
        re.search(r"\b(crop|see|visible|flag|stripe|stripes|triangle|color|colors|band|bands|geometry)\b", text)
    )
    contradiction_awareness = bool(
        re.search(r"\b(conflict|contradict|contradicts|despite|although|however|but|inconsistent)\b", text)
    )
    rationale_laundering = bool(chose_lure and lure_relation == "incompatible" and cites_crop)
    return {
        "cites_memory": cites_memory,
        "cites_crop": cites_crop,
        "contradiction_awareness": contradiction_awareness,
        "rationale_laundering": rationale_laundering,
    }


def _row_from_result(
    *,
    spec: TrialSpec,
    message: InteractionMessage | None,
    valid: bool,
    error: str | None,
) -> dict[str, Any]:
    choice = message.country if message is not None else None
    reason = message.reason if message is not None else None
    chose_lure = choice == spec.lure_country
    chose_truth = choice == spec.truth_country
    false_majority = spec.false_memory_count > spec.true_memory_count
    true_majority = spec.true_memory_count > spec.false_memory_count
    memory_tie = spec.true_memory_count == spec.false_memory_count
    if false_majority:
        followed_memory_majority = chose_lure
    elif true_majority:
        followed_memory_majority = chose_truth
    else:
        followed_memory_majority = False
    flags = _reason_flags(reason, chose_lure=chose_lure, lure_relation=spec.lure_relation)

    return {
        "trial_id": spec.trial_id,
        "model": spec.model,
        "m": spec.m,
        "truth_country": spec.truth_country,
        "lure_country": spec.lure_country,
        "lure_relation": spec.lure_relation,
        "crop_condition": spec.crop_condition,
        "false_memory_count": spec.false_memory_count,
        "true_memory_count": spec.true_memory_count,
        "false_majority": false_majority,
        "true_majority": true_majority,
        "memory_tie": memory_tie,
        "rep": spec.rep,
        "seed": spec.seed,
        "valid": valid,
        "choice_country": choice,
        "reason": reason,
        "correct": chose_truth if valid else None,
        "chose_lure": chose_lure if valid else None,
        "chose_truth": chose_truth if valid else None,
        "chose_other": bool(valid and not chose_lure and not chose_truth),
        "followed_memory_majority": followed_memory_majority if valid and not memory_tie else None,
        "error": error,
        **{f"crop_{key}": value for key, value in spec.crop_box.to_dict().items()},
        "crop_compatible_country_count": spec.crop_diagnostic["compatible_country_count"],
        "crop_compatible_countries": json.dumps(spec.crop_diagnostic["compatible_countries"]),
        "crop_informativeness_bits": spec.crop_diagnostic["informativeness_bits"],
        "crop_informativeness_label": spec.crop_diagnostic["informativeness_label"],
        "memory_lines": json.dumps(spec.memory_lines, ensure_ascii=True),
        "allowed_countries": json.dumps(spec.allowed_countries, ensure_ascii=True),
        **flags,
    }


def run_trials(
    *,
    args: argparse.Namespace,
    specs: list[TrialSpec],
    stimuli: dict[tuple[str, str], Stimulus],
) -> None:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    completed = set() if args.force else _existing_trial_ids(results_path)
    backend_name = args.backend or choose_default_backend()
    pool = get_country_pool(args.country_pool)
    lookup = _country_lookup(pool)

    backend_cache: dict[str, Any] = {}
    prepared_crop_cache: dict[tuple[str, str, str], Any] = {}

    def get_backend(model: str) -> Any:
        if model not in backend_cache:
            backend_cache[model] = build_backend(
                backend_name=backend_name,
                model=model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                debug_dir=out_dir / "debug" / _safe_model_dir(model),
                image_detail=args.image_detail,
                seed=args.seed,
                social_susceptibility=args.social_susceptibility,
                prompt_social_susceptibility=args.prompt_social_susceptibility,
                prompt_style="closed_country_list",
                country_lookup=lookup,
            )
        return backend_cache[model]

    def prepared_crop_for(spec: TrialSpec) -> Any:
        key = (spec.model, spec.truth_country, spec.crop_condition)
        if key not in prepared_crop_cache:
            backend = get_backend(spec.model)
            stimulus = stimuli[(spec.truth_country, spec.crop_condition)]
            prepared_crop_cache[key] = backend.prepare_crop(stimulus.crop_image)
        return prepared_crop_cache[key]

    pending = [spec for spec in specs if spec.trial_id not in completed]
    print(
        f"Trial plan: {len(specs)} total, {len(completed)} already complete, "
        f"{len(pending)} pending. Backend={backend_name}."
    )
    if not pending:
        return

    with results_path.open("a") as handle:
        for index, spec in enumerate(pending, start=1):
            backend = get_backend(spec.model)
            try:
                message = backend.probe(
                    countries=list(spec.allowed_countries),
                    prepared_crop=prepared_crop_for(spec),
                    memory_lines=list(spec.memory_lines),
                    m=spec.m,
                )
                if isinstance(message, str):
                    message = InteractionMessage(country=message)
                row = _row_from_result(spec=spec, message=message, valid=True, error=None)
            except ParseError as exc:
                row = _row_from_result(spec=spec, message=None, valid=False, error=str(exc))
            except Exception as exc:
                row = _row_from_result(spec=spec, message=None, valid=False, error=repr(exc))
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"Completed {index}/{len(pending)} pending trials")

    usage = {
        model: backend.usage_summary()
        for model, backend in sorted(backend_cache.items())
        if hasattr(backend, "usage_summary")
    }
    (out_dir / "usage_summary.json").write_text(json.dumps(usage, indent=2))


def _mean_bool(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.astype(bool).mean())


def _private_evidence_strength(crop_condition: str) -> str:
    if crop_condition == "diagnostic_true":
        return "strong_private_evidence"
    if crop_condition == "ambiguous_true":
        return "weak_private_evidence"
    return crop_condition


def _social_evidence_type(lure_relation: str) -> str:
    if lure_relation == "incompatible":
        return "contradictory_social_evidence"
    if lure_relation == "compatible":
        return "private_compatible_social_evidence"
    return lure_relation


def _is_choice_crop_compatible(row: pd.Series) -> bool:
    if not isinstance(row.get("choice_country"), str) or not row.get("choice_country"):
        return False
    try:
        compatible = json.loads(str(row.get("crop_compatible_countries", "[]")))
    except json.JSONDecodeError:
        return False
    return row["choice_country"] in compatible


_FLAG_LOOKUP_CACHE: dict[str, FlagSpec] | None = None


def _flag_lookup_for_alignment() -> dict[str, FlagSpec]:
    global _FLAG_LOOKUP_CACHE
    if _FLAG_LOOKUP_CACHE is None:
        _FLAG_LOOKUP_CACHE = _country_lookup(get_country_pool("stripe_plus_real_triangle_28"))
    return _FLAG_LOOKUP_CACHE


def _is_social_feature_match(choice_country: Any, lure_country: Any) -> bool:
    if not isinstance(choice_country, str) or not isinstance(lure_country, str):
        return False
    if choice_country == lure_country:
        return True
    lookup = _flag_lookup_for_alignment()
    choice_flag = lookup.get(choice_country)
    lure_flag = lookup.get(lure_country)
    if not isinstance(choice_flag, StripeFlag) or not isinstance(lure_flag, StripeFlag):
        return False
    shared_colors = len(set(choice_flag.colors) & set(lure_flag.colors))
    same_orientation = choice_flag.orientation == lure_flag.orientation
    same_triangle_color = (
        choice_flag.triangle_color is not None
        and choice_flag.triangle_color == lure_flag.triangle_color
    )
    same_triangle_side = (
        choice_flag.triangle_side is not None
        and choice_flag.triangle_side == lure_flag.triangle_side
    )
    return bool(
        same_triangle_color
        or (same_orientation and shared_colors >= 2)
        or (same_orientation and same_triangle_side and shared_colors >= 1)
    )


def _is_choice_social_feature_compatible(row: pd.Series) -> bool:
    return _is_social_feature_match(row.get("choice_country"), row.get("lure_country"))


def add_response_type_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["choice_crop_compatible"] = df.apply(_is_choice_crop_compatible, axis=1)
    df["choice_social_feature_compatible"] = df.apply(_is_choice_social_feature_compatible, axis=1)
    response_types: list[str | None] = []
    alignment_types: list[str | None] = []
    for _, row in df.iterrows():
        if not bool(row.get("valid", False)):
            response_types.append(None)
            alignment_types.append(None)
        elif bool(row.get("chose_truth", False)):
            response_types.append("private_evidence")
            alignment_types.append("private_evidence")
        elif bool(row.get("chose_lure", False)):
            response_types.append("social_evidence")
            alignment_types.append("social_evidence")
        elif bool(row.get("choice_crop_compatible", False)):
            response_types.append("private_compatible_other")
            if bool(row.get("choice_social_feature_compatible", False)):
                alignment_types.append("other_private_and_social")
            else:
                alignment_types.append("other_private_only")
        elif bool(row.get("choice_social_feature_compatible", False)):
            response_types.append("unsupported_other")
            alignment_types.append("other_social_only")
        else:
            response_types.append("unsupported_other")
            alignment_types.append("unsupported_other")
    df["response_type"] = response_types
    df["response_type_label"] = df["response_type"].map(RESPONSE_TYPE_LABELS)
    df["alignment_type"] = alignment_types
    df["alignment_type_label"] = df["alignment_type"].map(ALIGNMENT_TYPE_LABELS)
    return df


def write_summaries_and_plots(out_dir: Path, *, make_plots: bool = True) -> None:
    results_path = out_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"No results found at {results_path}")
    df = pd.read_json(results_path, lines=True)
    if df.empty:
        raise RuntimeError(f"No rows found in {results_path}")
    df["private_evidence_strength"] = df["crop_condition"].map(_private_evidence_strength)
    df["social_evidence_type"] = df["lure_relation"].map(_social_evidence_type)
    df = add_response_type_columns(df)
    df.to_csv(out_dir / "results.csv", index=False)

    group_cols = ["model", "m", "crop_condition", "lure_relation", "false_memory_count"]
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        model, m, crop_condition, lure_relation, false_memory_count = key
        valid = group[group["valid"].astype(bool)]
        rows.append(
            {
                "model": model,
                "m": int(m),
                "crop_condition": crop_condition,
                "private_evidence_strength": _private_evidence_strength(str(crop_condition)),
                "lure_relation": lure_relation,
                "social_evidence_type": _social_evidence_type(str(lure_relation)),
                "false_memory_count": int(false_memory_count),
                "n_trials": int(len(group)),
                "valid_rate": float(len(valid) / max(len(group), 1)),
                "chose_lure_rate": _mean_bool(valid["chose_lure"]) if not valid.empty else float("nan"),
                "correct_rate": _mean_bool(valid["correct"]) if not valid.empty else float("nan"),
                "chose_other_rate": _mean_bool(valid["chose_other"]) if not valid.empty else float("nan"),
                "followed_memory_majority_rate": _mean_bool(valid["followed_memory_majority"])
                if "followed_memory_majority" in valid and not valid.empty
                else float("nan"),
                "cites_memory_rate": _mean_bool(valid["cites_memory"]) if "cites_memory" in valid else float("nan"),
                "cites_crop_rate": _mean_bool(valid["cites_crop"]) if "cites_crop" in valid else float("nan"),
                "contradiction_awareness_rate": _mean_bool(valid["contradiction_awareness"])
                if "contradiction_awareness" in valid
                else float("nan"),
                "rationale_laundering_rate": _mean_bool(valid["rationale_laundering"])
                if "rationale_laundering" in valid
                else float("nan"),
            }
        )
    summary = pd.DataFrame(rows).sort_values(group_cols)
    summary.to_csv(out_dir / "summary_by_count.csv", index=False)

    threshold_rows: list[dict[str, Any]] = []
    for key, group in summary.groupby(["model", "m", "crop_condition", "lure_relation"], dropna=False):
        model, m, crop_condition, lure_relation = key
        clean = group.dropna(subset=["chose_lure_rate"]).sort_values("false_memory_count")
        threshold = None
        if not clean.empty:
            above = clean[clean["chose_lure_rate"] >= 0.5]
            if not above.empty:
                threshold = int(above.iloc[0]["false_memory_count"])
        threshold_rows.append(
            {
                "model": model,
                "m": int(m),
                "crop_condition": crop_condition,
                "private_evidence_strength": _private_evidence_strength(str(crop_condition)),
                "lure_relation": lure_relation,
                "social_evidence_type": _social_evidence_type(str(lure_relation)),
                "first_false_memory_count_with_lure_rate_ge_0_5": threshold,
                "max_chose_lure_rate": float(clean["chose_lure_rate"].max()) if not clean.empty else float("nan"),
            }
        )
    pd.DataFrame(threshold_rows).to_csv(out_dir / "threshold_summary.csv", index=False)

    audit_cols = [
        "trial_id",
        "model",
        "m",
        "private_evidence_strength",
        "lure_relation",
        "false_memory_count",
        "truth_country",
        "lure_country",
        "choice_country",
        "response_type_label",
        "alignment_type_label",
        "choice_crop_compatible",
        "choice_social_feature_compatible",
        "crop_compatible_country_count",
        "crop_compatible_countries",
        "reason",
        "memory_lines",
    ]
    audit_cols = [col for col in audit_cols if col in df.columns]
    df[df["chose_other"].astype(bool)].to_csv(
        out_dir / "off_axis_choice_audit.csv",
        index=False,
        columns=audit_cols,
    )

    write_plot_notes(out_dir, df=df)

    if make_plots:
        plot_choice_axis(df, out_dir / "private_vs_social_evidence_choice_axis.png")
        plot_response_decomposition(df, out_dir / "agent_response_decomposition.png")
        plot_evidence_alignment_decomposition(df, out_dir / "agent_response_evidence_alignment.png")
        for m_value in sorted(int(value) for value in df["m"].dropna().unique()):
            plot_choice_axis(
                df[df["m"] == m_value],
                out_dir / f"private_vs_social_evidence_choice_axis_m{m_value}.png",
                title=f"Private Evidence vs Social Evidence, m={m_value}",
                row_by_m=False,
            )
            plot_response_decomposition(
                df[df["m"] == m_value],
                out_dir / f"agent_response_decomposition_m{m_value}.png",
                title=f"Agent Response Decomposition, m={m_value}",
                row_by_m=False,
            )
            plot_evidence_alignment_decomposition(
                df[df["m"] == m_value],
                out_dir / f"agent_response_evidence_alignment_m{m_value}.png",
                title=f"Agent Response Evidence Alignment, m={m_value}",
                row_by_m=False,
            )


def write_plot_notes(out_dir: Path, *, df: pd.DataFrame) -> None:
    third_country_rate = float(df["chose_other"].dropna().astype(bool).mean()) if "chose_other" in df else float("nan")
    notes = [
        "# Memory Conflict Probe Plot Notes",
        "",
        "## Choice Axis Definition",
        "",
        "- `0 = target country`: the agent chose the hidden target/original country that generated the private crop.",
        "- `1 = social evidence`: the agent chose the synthetic lure country supported by transcript memory.",
        "- In strong private evidence, the target country is also the only country compatible with the crop.",
        "- In weak private evidence, the target country is only one of several countries compatible with the crop. So `0` means target recovery, not unique visual certainty or the only private-image-supported answer.",
        "- The choice-axis plots use only trials where the agent chose either the target country or the social-evidence country. Other crop-compatible answers are not placed on this binary axis.",
        "",
        "## Memory Composition Axis",
        "",
        "- X-axis tick labels show `target memories : social-evidence memories`.",
        "- For example, `6:2` means six synthetic memory entries support the target country and two support the social-evidence country.",
        "- Moving right means less target-country memory support and more social-evidence memory support.",
        "",
        "## Private Evidence Strength",
        "",
        "- `Strong private evidence`: the private crop can be produced by exactly one allowed country. In this run, that country is the truth/private-evidence country.",
        "- `Weak private evidence`: the private crop can be produced by many allowed countries, including the truth/private-evidence country.",
        "- Strength is computed mechanically by rendering every allowed flag, enumerating all same-sized crop positions, and checking which countries can produce the exact private-crop pixels.",
        "",
        "## m Definition",
        "",
        "- `m=1`: label-only memory entries and label-only model outputs.",
        "- `m=3`: memory entries include a country plus a one-sentence reason, and outputs also include a reason.",
        "",
        "## Response Decomposition",
        "",
        "- `Private target country`: the agent chose the hidden target/original country that generated the private crop.",
        "- `Social evidence`: the agent chose the synthetic social-evidence lure.",
        "- `Other crop-compatible country`: the agent chose a different country that can produce the same private-crop pixels. This is still consistent with the private image, but it is not the hidden target/original country.",
        "- `Unsupported other`: the agent chose a third country that is neither the social lure nor compatible with the private crop.",
        "- If `--lure-relation compatible` is included, the social-evidence country also fits the private crop. This is possible for weak private evidence, but not for strong private evidence where only the target country fits.",
        "",
        "## Evidence Alignment Decomposition",
        "",
        "- `Private target country`: the agent chose the hidden target/original country that generated the private crop.",
        "- `Social evidence`: the agent chose the synthetic social-evidence lure.",
        "- `Other: private + social compatible`: a third-country answer compatible with the private image and also compatible with the social-evidence country's flag features.",
        "- `Other: private compatible`: a third-country answer compatible with the private image, but not compatible with the social-evidence country's flag features.",
        "- `Other: social compatible`: a third-country answer not compatible with the private image, but compatible with the social-evidence country's flag features.",
        "- `Other: incompatible`: a third-country answer compatible with neither the private image nor the social-evidence country's flag features.",
        "- Social compatibility is a coarse heuristic: same stripe orientation with at least two shared colors, or matching triangle features. Exact row-level cases are written to `off_axis_choice_audit.csv`.",
        "",
        "## Memory Order",
        "",
        "- Each trial has 8 synthetic memory entries: `false_memory_count` entries for the social-evidence country and `8 - false_memory_count` entries for the target country.",
        "- The memory-entry order is shuffled independently for each replicate. The allowed-country list is also shuffled independently for each trial.",
        "",
        "## Current Run Note",
        "",
        f"- Overall third-country answer rate: `{third_country_rate:.3f}`. These answers are still in `results.csv`, but excluded from binary private-vs-social choice-axis means.",
        "",
        "Recommended visual for interpretation: `agent_response_decomposition.png`, plus the `m1` and `m3` variants.",
        "",
        "`private_vs_social_evidence_choice_axis.png` is a narrower binary view. It excludes third-country answers and asks only: among runs where the agent chose either the target country or the social-evidence country, how often did it choose the social-evidence country?",
    ]
    (out_dir / "plot_notes.md").write_text("\n".join(notes))


def _humanize(value: str) -> str:
    return value.replace("_true", "").replace("_", " ").strip().title()


def _social_evidence_label(lure_relation: str) -> str:
    if _social_evidence_type(lure_relation) == "contradictory_social_evidence":
        return "Social evidence contradicts the private image"
    return "Social evidence is compatible with the private image"


def _facet_title(crop_condition: str, lure_relation: str, *, include_relation: bool) -> str:
    private = (
        "Strong Private Evidence\n(only target country fits)"
        if _private_evidence_strength(crop_condition) == "strong_private_evidence"
        else "Weak Private Evidence\n(many countries fit)"
    )
    if not include_relation:
        return private
    return f"{private}\n{_social_evidence_label(lure_relation)}"


def _facet_sort_key(facet: Any) -> tuple[int, int, str, str]:
    crop_condition = str(facet[0])
    lure_relation = str(facet[1])
    strength_rank = {
        "weak_private_evidence": 0,
        "strong_private_evidence": 1,
    }.get(_private_evidence_strength(crop_condition), 9)
    relation_rank = {
        "compatible": 0,
        "incompatible": 1,
    }.get(lure_relation, 9)
    return (strength_rank, relation_rank, crop_condition, lure_relation)


def _draw_grouped_bar_x_axis(
    axis: Any,
    *,
    x_counts: list[int],
    show_xlabel: bool,
) -> None:
    axis.set_xticks(x_counts)
    axis.set_xticklabels(_memory_composition_tick_labels(x_counts))
    axis.tick_params(axis="x", length=3)
    if show_xlabel:
        axis.set_xlabel("Memory composition (target : social evidence)")


def _memory_composition_tick_labels(x_counts: list[int]) -> list[str]:
    total = max(x_counts) if x_counts else 8
    return [f"{total - int(k)}:{int(k)}" for k in x_counts]


def _set_memory_composition_x_axis(axis: Any, x_counts: list[int], *, show_xlabel: bool) -> None:
    axis.set_xticks(x_counts)
    axis.set_xticklabels(_memory_composition_tick_labels(x_counts))
    if show_xlabel:
        axis.set_xlabel("Memory composition (target : social evidence)")


def _add_m_section_guides(fig: Any, axes: Any, row_specs: list[tuple[int, str]]) -> None:
    if not row_specs:
        return
    m_values = [m_value for m_value, _ in row_specs]
    if len(set(m_values)) <= 1:
        return

    blocks: list[tuple[int, int, int]] = []
    start = 0
    current_m = m_values[0]
    for index, m_value in enumerate(m_values[1:], start=1):
        if m_value != current_m:
            blocks.append((current_m, start, index - 1))
            start = index
            current_m = m_value
    blocks.append((current_m, start, len(m_values) - 1))

    for block_index, (m_value, first_row, last_row) in enumerate(blocks):
        top = axes[first_row][0].get_position().y1
        bottom = axes[last_row][0].get_position().y0
        fig.text(
            0.035,
            (top + bottom) / 2.0,
            f"m={m_value}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=13,
            fontweight="bold",
            color="#111827",
        )
        if block_index > 0:
            prev_bottom = axes[first_row - 1][0].get_position().y0
            next_top = axes[first_row][0].get_position().y1
            y_separator = (prev_bottom + next_top) / 2.0
            fig.add_artist(
                plt.Line2D(
                    [0.075, 0.98],
                    [y_separator, y_separator],
                    transform=fig.transFigure,
                    color="#111827",
                    linewidth=1.4,
                    alpha=0.28,
                )
            )


def _plot_subtitle(frame: pd.DataFrame, base: str) -> str:
    if "lure_relation" not in frame or frame["lure_relation"].nunique(dropna=True) != 1:
        return base
    relation = str(frame["lure_relation"].dropna().iloc[0])
    return f"{base} {_social_evidence_label(relation)}."


def plot_memory_curves(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "When Does Social Evidence Override Private Evidence?",
    row_by_m: bool = True,
) -> None:
    summary = summary.copy()
    if summary.empty:
        return
    facets = (
        summary[["crop_condition", "lure_relation"]]
        .drop_duplicates()
        .to_records(index=False)
    )
    facets = sorted(facets, key=_facet_sort_key)
    if len(facets) == 0:
        return
    include_relation = summary["lure_relation"].nunique(dropna=True) > 1

    m_values = sorted(int(value) for value in summary["m"].dropna().unique())
    if not row_by_m:
        m_values = [m_values[0]]

    ncols = len(facets)
    nrows = len(m_values)
    fig_width = max(5.2 * ncols, 7.0)
    fig_height = max(3.5 * nrows + 1.5, 4.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    for axis in axes.flat:
        axis.set_visible(False)

    for row_index, m_value in enumerate(m_values):
        for col_index, facet in enumerate(facets):
            axis = axes[row_index][col_index]
            crop_condition = str(facet[0])
            lure_relation = str(facet[1])
            axis.set_visible(True)
            subset = summary[
                (summary["crop_condition"] == crop_condition)
                & (summary["lure_relation"] == lure_relation)
                & (summary["m"].astype(int) == m_value)
            ]
            for model, group in subset.groupby("model", dropna=False):
                group = group.sort_values("false_memory_count")
                axis.plot(
                    group["false_memory_count"],
                    group["chose_lure_rate"],
                    marker="o",
                    markersize=5.5,
                    linewidth=2.3,
                    color=MODEL_COLORS.get(str(model), "#6b7280"),
                    label=str(model),
                )
            axis.axhline(0.5, color="#374151", linewidth=1.1, alpha=0.35, linestyle=":")
            axis.set_xlim(-0.2, 8.2)
            axis.set_ylim(-0.04, 1.04)
            _set_memory_composition_x_axis(
                axis,
                list(range(0, 9)),
                show_xlabel=row_index == nrows - 1,
            )
            if col_index == 0:
                label = "Chooses social-evidence lure"
                axis.set_ylabel(f"m={m_value}\n{label}" if row_by_m else label)
            axis.set_title(
                _facet_title(crop_condition, lure_relation, include_relation=include_relation),
                fontsize=12,
                pad=10,
            )
            axis.grid(axis="y", alpha=0.22)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="lower center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.02),
        )
    title_y = 0.985
    axes_top = 0.86 if nrows > 1 else 0.74
    fig.suptitle(title, y=title_y, fontsize=16)
    fig.subplots_adjust(top=axes_top, bottom=0.14, left=0.08, right=0.98, wspace=0.22, hspace=0.42)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _fit_logistic_curve(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    clean = pd.DataFrame({"x": x.astype(float), "y": y.astype(float)}).dropna()
    if clean.empty or clean["y"].nunique() < 2:
        return None
    x_values = clean["x"].to_numpy(dtype=float)
    y_values = clean["y"].to_numpy(dtype=float)
    mean = float(x_values.mean())
    scale = float(x_values.std(ddof=0)) or 1.0
    z_values = (x_values - mean) / scale
    beta = np.zeros(2, dtype=float)
    for _ in range(80):
        eta = beta[0] + beta[1] * z_values
        probs = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        weights = np.clip(probs * (1.0 - probs), 1e-6, None)
        gradient = np.array(
            [
                float(np.sum(y_values - probs)),
                float(np.sum((y_values - probs) * z_values)),
            ]
        )
        hessian = np.array(
            [
                [float(np.sum(weights)), float(np.sum(weights * z_values))],
                [float(np.sum(weights * z_values)), float(np.sum(weights * z_values * z_values))],
            ]
        )
        try:
            step = np.linalg.solve(hessian + np.eye(2) * 1e-6, gradient)
        except np.linalg.LinAlgError:
            return None
        beta += step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    intercept = float(beta[0] - beta[1] * mean / scale)
    slope = float(beta[1] / scale)
    return intercept, slope


def plot_choice_axis(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Private Evidence vs Social Evidence",
    row_by_m: bool = True,
) -> None:
    df = df[df["valid"].astype(bool)].copy()
    if df.empty:
        return
    # Binary choice axis: hidden target/original country is 0, social-evidence
    # country is 1. Other-country choices do not lie on this two-choice axis.
    df = df[df["chose_truth"].astype(bool) | df["chose_lure"].astype(bool)].copy()
    if df.empty:
        return
    df["choice_score"] = df["chose_lure"].astype(bool).astype(float)

    facets = (
        df[["crop_condition", "lure_relation"]]
        .drop_duplicates()
        .to_records(index=False)
    )
    facets = sorted(facets, key=_facet_sort_key)
    include_relation = df["lure_relation"].nunique(dropna=True) > 1
    m_values = sorted(int(value) for value in df["m"].dropna().unique())
    if not row_by_m:
        m_values = [m_values[0]]
    models = [model for model in ["gpt-4o", "gpt-5.4"] if model in set(df["model"])]
    models += [model for model in sorted(set(df["model"])) if model not in models]

    ncols = len(facets)
    nrows = len(m_values)
    fig_width = max(5.2 * ncols, 7.0)
    fig_height = max(3.5 * nrows + 1.5, 4.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    for axis in axes.flat:
        axis.set_visible(False)

    for row_index, m_value in enumerate(m_values):
        for col_index, facet in enumerate(facets):
            axis = axes[row_index][col_index]
            crop_condition = str(facet[0])
            lure_relation = str(facet[1])
            axis.set_visible(True)
            subset = df[
                (df["crop_condition"] == crop_condition)
                & (df["lure_relation"] == lure_relation)
                & (df["m"].astype(int) == m_value)
            ]
            if subset.empty:
                axis.set_axis_off()
                continue
            for model in models:
                group = subset[subset["model"] == model]
                if group.empty:
                    continue
                color = MODEL_COLORS.get(str(model), "#6b7280")
                grouped = (
                    group.groupby("false_memory_count", dropna=False)["choice_score"]
                    .agg(["mean", "count", "std"])
                    .reset_index()
                    .sort_values("false_memory_count")
                )
                sem = grouped["std"].fillna(0.0) / grouped["count"].clip(lower=1).pow(0.5)
                axis.errorbar(
                    grouped["false_memory_count"],
                    grouped["mean"],
                    yerr=sem,
                    marker="o",
                    markersize=5.5,
                    linewidth=2.0,
                    capsize=3,
                    color=color,
                    zorder=3,
                    label=_model_short_label(str(model)),
                )
            axis.axhline(0.5, color="#374151", linewidth=1.1, alpha=0.35, linestyle=":")
            axis.set_xlim(-0.2, 8.2)
            axis.set_ylim(-0.04, 1.04)
            _set_memory_composition_x_axis(
                axis,
                list(range(0, 9)),
                show_xlabel=row_index == nrows - 1,
            )
            axis.set_yticks([0.0, 0.5, 1.0])
            if col_index == 0:
                axis.set_yticklabels(["0 target country", "", "1 social evidence"])
            else:
                axis.set_yticklabels([])
            if col_index == 0:
                axis.set_ylabel(f"m={m_value}\nAgent response" if row_by_m else "Agent response")
            if row_index == 0:
                axis.set_title(
                    _facet_title(crop_condition, lure_relation, include_relation=include_relation),
                    fontsize=12,
                    pad=10,
                )
            axis.grid(axis="y", alpha=0.22)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique[label] = handle
        fig.legend(
            unique.values(),
            unique.keys(),
            frameon=False,
            loc="lower center",
            ncol=min(4, len(unique)),
            bbox_to_anchor=(0.5, 0.02),
        )
    axes_top = 0.84 if nrows > 1 else 0.78
    fig.suptitle(title, y=0.985, fontsize=16)
    fig.subplots_adjust(top=axes_top, bottom=0.16, left=0.08, right=0.98, wspace=0.22, hspace=0.42)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_stacked_area_decomposition(
    *,
    df: pd.DataFrame,
    out_path: Path,
    value_col: str,
    category_order: list[str],
    category_labels: dict[str, str],
    category_colors: dict[str, str],
    title: str,
    subtitle: str,
    row_by_m: bool,
    stack_order: list[str] | None = None,
) -> None:
    df = df[df["valid"].astype(bool)].copy()
    if df.empty:
        return
    stack_order = stack_order or category_order

    facets = (
        df[["crop_condition", "lure_relation"]]
        .drop_duplicates()
        .to_records(index=False)
    )
    facets = sorted(facets, key=_facet_sort_key)
    include_relation = df["lure_relation"].nunique(dropna=True) > 1
    models = [model for model in ["gpt-4o", "gpt-5.4"] if model in set(df["model"])]
    models += [model for model in sorted(set(df["model"])) if model not in models]
    m_values = sorted(int(value) for value in df["m"].dropna().unique())
    if not row_by_m:
        m_values = [m_values[0]]

    row_specs = [(m_value, model) for m_value in m_values for model in models]
    if not row_specs:
        return

    ncols = len(facets)
    nrows = len(row_specs)
    fig_width = max(6.0 * ncols, 8.0)
    fig_height = max(2.7 * nrows + 2.0, 5.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    for axis in axes.flat:
        axis.set_visible(False)

    x_counts = sorted(int(value) for value in df["false_memory_count"].dropna().unique())

    for row_index, (m_value, model) in enumerate(row_specs):
        for col_index, facet in enumerate(facets):
            axis = axes[row_index][col_index]
            axis.set_visible(True)
            crop_condition = str(facet[0])
            lure_relation = str(facet[1])
            subset = df[
                (df["m"].astype(int) == m_value)
                & (df["model"] == model)
                & (df["crop_condition"] == crop_condition)
                & (df["lure_relation"] == lure_relation)
            ]
            if subset.empty:
                axis.set_axis_off()
                continue

            series: list[list[float]] = []
            for category in stack_order:
                heights: list[float] = []
                for k in x_counts:
                    cell = subset[subset["false_memory_count"].astype(int) == k]
                    if cell.empty:
                        heights.append(0.0)
                    else:
                        heights.append(float((cell[value_col] == category).mean()))
                series.append(heights)

            axis.stackplot(
                x_counts,
                *series,
                colors=[category_colors[category] for category in stack_order],
                labels=[category_labels[category] for category in stack_order],
                alpha=0.96,
                linewidth=0.45,
                edgecolor="white",
            )
            axis.set_xlim(min(x_counts), max(x_counts))
            axis.set_ylim(0.0, 1.02)
            _set_memory_composition_x_axis(
                axis,
                x_counts,
                show_xlabel=row_index == nrows - 1,
            )
            if row_index != nrows - 1:
                axis.set_xticklabels([])
            if col_index == 0:
                axis.set_ylabel(f"{_model_short_label(model)}\nShare")
            if row_index == 0:
                axis.set_title(
                    _facet_title(crop_condition, lure_relation, include_relation=include_relation),
                    fontsize=12,
                    pad=10,
                )
            axis.grid(axis="y", alpha=0.22)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique[label] = handle
        ordered_labels = [
            category_labels[category]
            for category in category_order
            if category_labels[category] in unique
        ]
        fig.legend(
            [unique[label] for label in ordered_labels],
            ordered_labels,
            frameon=False,
            loc="lower center",
            ncol=min(3, len(ordered_labels)),
            bbox_to_anchor=(0.5, 0.02),
        )
    axes_top = 0.88 if nrows > 2 else 0.78
    fig.suptitle(title, y=0.985, fontsize=16)
    if subtitle:
        fig.text(
            0.5,
            0.94 if nrows > 2 else 0.89,
            _plot_subtitle(df, subtitle),
            ha="center",
            va="center",
            fontsize=10.5,
            color="#4b5563",
        )
    fig.subplots_adjust(top=axes_top, bottom=0.17, left=0.11, right=0.98, wspace=0.18, hspace=0.42)
    if row_by_m:
        _add_m_section_guides(fig, axes, row_specs)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_response_decomposition(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Agent Response Decomposition",
    row_by_m: bool = True,
) -> None:
    if "response_type" not in df:
        df = add_response_type_columns(df)
    _plot_stacked_area_decomposition(
        df=df,
        out_path=out_path,
        value_col="response_type",
        category_order=RESPONSE_TYPE_ORDER,
        category_labels=RESPONSE_TYPE_LABELS,
        category_colors=RESPONSE_TYPE_COLORS,
        title=title,
        subtitle="",
        row_by_m=row_by_m,
    )


def plot_evidence_alignment_decomposition(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "Agent Response Evidence Alignment",
    row_by_m: bool = True,
) -> None:
    if "alignment_type" not in df:
        df = add_response_type_columns(df)
    _plot_stacked_area_decomposition(
        df=df,
        out_path=out_path,
        value_col="alignment_type",
        category_order=ALIGNMENT_TYPE_ORDER,
        category_labels=ALIGNMENT_TYPE_LABELS,
        category_colors=ALIGNMENT_TYPE_COLORS,
        title=title,
        subtitle="",
        row_by_m=row_by_m,
        stack_order=ALIGNMENT_STACK_ORDER,
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        write_summaries_and_plots(args.out, make_plots=not args.no_plots)
        print(f"Wrote summaries under {args.out}")
        return

    specs, stimuli = build_trial_plan(args)
    write_plan(
        out_dir=args.out,
        specs=specs,
        stimuli=stimuli,
        save_crops=not args.no_save_crops,
    )

    plan_meta = {
        "n_trials": len(specs),
        "models": sorted(set(spec.model for spec in specs)),
        "m_values": sorted(set(spec.m for spec in specs)),
        "truth_countries": sorted(set(spec.truth_country for spec in specs)),
        "crop_conditions": sorted(set(spec.crop_condition for spec in specs)),
        "lure_relations": sorted(set(spec.lure_relation for spec in specs)),
        "false_memory_counts": sorted(set(spec.false_memory_count for spec in specs)),
        "replicates": args.replicates,
        "h": args.h,
        "backend": args.backend or choose_default_backend(),
    }
    (args.out / "plan_meta.json").write_text(json.dumps(plan_meta, indent=2))
    print(json.dumps(plan_meta, indent=2))

    if args.dry_run:
        print(f"Dry run only. Plan written to {args.out / 'trial_plan.csv'}")
        return

    run_trials(args=args, specs=specs, stimuli=stimuli)
    write_summaries_and_plots(args.out, make_plots=not args.no_plots)
    print(f"Wrote results under {args.out}")


if __name__ == "__main__":
    main()

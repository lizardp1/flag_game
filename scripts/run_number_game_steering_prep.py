from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnd.number_game import prompts
from nnd.number_game.backend import build_backend
from nnd.number_game.config import apply_overrides, load_number_game_config, save_resolved_config
from nnd.number_game.domain import DEFAULT_CLUES, candidate_numbers, clue_matches_number, matching_clues
from nnd.number_game.parsing import ParseError, parse_number_message


@dataclass(frozen=True)
class SteeringCase:
    case_id: str
    private_clue: str
    private_target: int
    private_reason: str
    social_number: int
    social_reason: str
    relation: str = "contradictory"
    source: str = "handwritten"
    split: str = "train"


CASES: tuple[SteeringCase, ...] = (
    SteeringCase(
        case_id="digit_sum7_vs_even12",
        private_clue="the digits sum to 7",
        private_target=7,
        private_reason="The digits sum to 7.",
        social_number=12,
        social_reason="The number is even.",
    ),
    SteeringCase(
        case_id="prime7_vs_even12",
        private_clue="the number is prime",
        private_target=7,
        private_reason="The number is prime.",
        social_number=12,
        social_reason="The number is even.",
    ),
    SteeringCase(
        case_id="square49_vs_under20_12",
        private_clue="the number is a perfect square",
        private_target=49,
        private_reason="The number is a perfect square.",
        social_number=12,
        social_reason="The number is under 20.",
    ),
    SteeringCase(
        case_id="multiple5_25_vs_prime7",
        private_clue="the number is a multiple of 5",
        private_target=25,
        private_reason="The number is a multiple of 5.",
        social_number=7,
        social_reason="The number is prime.",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steering-vector prep for the number game: contrast hidden states and steer full-number answers."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--memory-strength", action="append", type=int, default=[])
    parser.add_argument("--m", action="append", type=int, default=[])
    parser.add_argument("--layer", action="append", type=int, default=[])
    parser.add_argument("--alpha", action="append", type=float, default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-source", choices=["handwritten", "auto", "both"], default="handwritten")
    parser.add_argument("--max-cases", default=None, type=int)
    parser.add_argument("--targets-per-clue", default=4, type=int)
    parser.add_argument("--socials-per-target", default=2, type=int)
    parser.add_argument("--case-seed", default=0, type=int)
    parser.add_argument("--test-frac", default=0.25, type=float)
    parser.add_argument("--fit-split", choices=["all", "train"], default="all")
    parser.add_argument(
        "--direction-method",
        choices=["memory_contrast", "logprob_quantile", "svd_subspace"],
        default="memory_contrast",
        help=(
            "memory_contrast uses social-memory minus private/target-memory activations; "
            "logprob_quantile uses high minus low social-private logprob-margin prompts; "
            "svd_subspace uses the top contrast-diff components aligned to the mean direction."
        ),
    )
    parser.add_argument("--direction-quantile", default=0.25, type=float)
    parser.add_argument("--subspace-rank", default=3, type=int)
    parser.add_argument("--skip-alpha-sweep", action="store_true")
    parser.add_argument("--max-alpha-trials", default=None, type=int)
    parser.add_argument("--run-generation-steering", action="store_true")
    parser.add_argument("--max-generation-trials", default=96, type=int)
    parser.add_argument("--qualitative-examples-per-alpha", default=4, type=int)
    parser.add_argument("--data-scaling-size", action="append", type=int, default=[])
    parser.add_argument("--stability-subsets", default=12, type=int)
    parser.add_argument("--stability-seed", action="append", type=int, default=[])
    parser.add_argument("--ood-social-dir", action="append", type=Path, default=[])
    parser.add_argument("--max-ood-prompts", default=128, type=int)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    out_dir = args.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    config = load_number_game_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    config = config.model_copy(update={"backend": "transformers", "temperature": 0.0})
    save_resolved_config(config, out_dir)

    backend = build_backend(
        backend_name=config.backend,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        debug_dir=out_dir / "debug",
        seed=0,
        social_susceptibility=config.social_susceptibility,
        prompt_social_susceptibility=config.prompt_social_susceptibility,
        prompt_number_range=config.prompt_number_range,
        capture_hidden_states=False,
        hidden_state_layers=config.hidden_state_layers,
        use_response_format=config.use_response_format,
        api_base_url=config.api_base_url,
        api_key=config.api_key,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        enable_thinking=config.enable_thinking,
    )
    if not hasattr(backend, "model_obj") or not hasattr(backend, "tokenizer"):
        raise RuntimeError("Steering prep requires backend=transformers.")

    model_layer_count = len(get_transformer_layers(backend.model_obj))
    layers = args.layer or default_layers(model_layer_count)
    memory_strengths = args.memory_strength or [1, 4, 8]
    m_values = args.m or [1, 3]
    alphas = args.alpha or [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    scaling_sizes = args.data_scaling_size or [8, 16, 32, 64, 128]
    stability_seeds = args.stability_seed or [0, 1, 2]
    numbers = candidate_numbers(config.min_number, config.max_number)
    active_cases = build_active_cases(
        numbers=numbers,
        case_source=args.case_source,
        case_ids=set(args.case_id),
        targets_per_clue=args.targets_per_clue,
        socials_per_target=args.socials_per_target,
        max_cases=args.max_cases,
        seed=args.case_seed,
        test_frac=args.test_frac,
    )
    if not active_cases:
        raise ValueError("No steering cases selected.")

    trials, vectors = collect_trials(
        backend=backend,
        cases=active_cases,
        layers=layers,
        memory_strengths=memory_strengths,
        m_values=m_values,
        prompt_number_range=config.prompt_number_range,
    )
    directions, direction_summary = compute_direction_summary(
        trials,
        vectors,
        layers,
        fit_split=args.fit_split,
        direction_method=args.direction_method,
        direction_quantile=args.direction_quantile,
        subspace_rank=args.subspace_rank,
    )
    projection_rows = projection_distribution_rows(trials, layers)
    scaling_rows = data_scaling_curve(
        trials=trials,
        vectors=vectors,
        layers=layers,
        fit_split=args.fit_split,
        sizes=scaling_sizes,
        seed=args.case_seed,
        direction_method=args.direction_method,
        direction_quantile=args.direction_quantile,
        subspace_rank=args.subspace_rank,
    )
    stability_rows = vector_stability_rows(
        trials=trials,
        vectors=vectors,
        layers=layers,
        fit_split=args.fit_split,
        sizes=scaling_sizes,
        subset_count=args.stability_subsets,
        seeds=stability_seeds,
        direction_method=args.direction_method,
        direction_quantile=args.direction_quantile,
        subspace_rank=args.subspace_rank,
    )
    steering_rows: list[dict[str, Any]] = []
    if not args.skip_alpha_sweep:
        steering_rows = run_alpha_sweep(
            backend=backend,
            trials=trials,
            directions=directions,
            direction_summary=direction_summary,
            layers=layers,
            alphas=alphas,
            max_alpha_trials=args.max_alpha_trials,
        )
    sign_rows = calibrate_steering_signs(steering_rows)
    generation_rows: list[dict[str, Any]] = []
    if args.run_generation_steering:
        generation_rows = run_generation_steering(
            backend=backend,
            trials=trials,
            directions=directions,
            direction_summary=direction_summary,
            sign_rows=sign_rows,
            layers=layers,
            alphas=alphas,
            max_trials=args.max_generation_trials,
            dataset_name="synthetic_contrast",
        )
    generation_summary_rows = generation_side_effect_summary(generation_rows)
    behavioral_effect_rows = behavioral_steering_effect_summary(generation_summary_rows)

    ood_trials: list[dict[str, Any]] = []
    ood_vectors: dict[str, dict[int, np.ndarray]] = {}
    ood_projection_rows: list[dict[str, Any]] = []
    ood_steering_rows: list[dict[str, Any]] = []
    ood_generation_rows: list[dict[str, Any]] = []
    ood_generation_summary_rows: list[dict[str, Any]] = []
    if args.ood_social_dir:
        ood_trials, ood_vectors = collect_ood_social_trials(
            backend=backend,
            social_dirs=args.ood_social_dir,
            layers=layers,
            numbers=numbers,
            prompt_number_range=config.prompt_number_range,
            max_prompts=args.max_ood_prompts,
        )
        add_projections_to_trials(trials=ood_trials, vectors=ood_vectors, directions=directions, layers=layers)
        ood_projection_rows = projection_distribution_rows(ood_trials, layers)
        if not args.skip_alpha_sweep:
            ood_steering_rows = run_alpha_sweep(
                backend=backend,
                trials=ood_trials,
                directions=directions,
                direction_summary=direction_summary,
                layers=layers,
                alphas=alphas,
                max_alpha_trials=args.max_alpha_trials,
                eval_variants={"ood_actual_social"},
            )
            apply_existing_sign_calibration(ood_steering_rows, sign_rows)
        if args.run_generation_steering:
            ood_generation_rows = run_generation_steering(
                backend=backend,
                trials=ood_trials,
                directions=directions,
                direction_summary=direction_summary,
                sign_rows=sign_rows,
                layers=layers,
                alphas=alphas,
                max_trials=args.max_generation_trials,
                dataset_name="ood_actual_social",
            )
            ood_generation_summary_rows = generation_side_effect_summary(ood_generation_rows)
    ood_behavioral_effect_rows = behavioral_steering_effect_summary(ood_generation_summary_rows)

    write_csv(out_dir / "steering_prep_trials.csv", trials)
    write_csv(out_dir / "steering_direction_summary.csv", direction_summary)
    write_csv(out_dir / "projection_distributions.csv", projection_rows)
    write_csv(out_dir / "data_scaling_curve.csv", scaling_rows)
    write_csv(out_dir / "vector_stability.csv", stability_rows)
    write_csv(out_dir / "steering_alpha_sweep.csv", steering_rows)
    write_csv(out_dir / "steering_sign_summary.csv", sign_rows)
    write_csv(out_dir / "generation_steering_outputs.csv", generation_rows)
    write_jsonl(out_dir / "generation_steering_outputs.jsonl", generation_rows)
    write_csv(out_dir / "generation_side_effect_summary.csv", generation_summary_rows)
    write_csv(out_dir / "behavioral_steering_effect_summary.csv", behavioral_effect_rows)
    write_qualitative_examples(
        out_dir / "qualitative_alpha_examples.md",
        generation_rows,
        examples_per_alpha=args.qualitative_examples_per_alpha,
    )
    write_csv(out_dir / "ood_social_trials.csv", ood_trials)
    write_csv(out_dir / "ood_social_projection_distributions.csv", ood_projection_rows)
    write_csv(out_dir / "ood_social_alpha_sweep.csv", ood_steering_rows)
    write_csv(out_dir / "ood_social_generation_outputs.csv", ood_generation_rows)
    write_jsonl(out_dir / "ood_social_generation_outputs.jsonl", ood_generation_rows)
    write_csv(out_dir / "ood_social_side_effect_summary.csv", ood_generation_summary_rows)
    write_csv(out_dir / "ood_social_behavioral_steering_effect_summary.csv", ood_behavioral_effect_rows)
    write_qualitative_examples(
        out_dir / "ood_social_qualitative_alpha_examples.md",
        ood_generation_rows,
        examples_per_alpha=args.qualitative_examples_per_alpha,
    )
    save_vectors(out_dir / "steering_vectors_social_minus_private.npz", directions)
    save_calibrated_vectors(out_dir / "steering_vectors_empirical_social.npz", directions, sign_rows)
    write_plots(
        out_dir,
        direction_summary,
        steering_rows,
        projection_rows,
        scaling_rows,
        stability_rows,
        generation_summary_rows,
        behavioral_effect_rows,
        ood_steering_rows,
        ood_generation_summary_rows,
        ood_behavioral_effect_rows,
    )
    write_csv(out_dir / "steering_cases.csv", [case.__dict__ for case in active_cases])
    write_index(
        out_dir,
        config.model,
        active_cases,
        layers,
        memory_strengths,
        m_values,
        alphas,
        args.fit_split,
        args.direction_method,
        args.direction_quantile,
        args.subspace_rank,
    )

    print(f"Steering prep complete. Output saved to {out_dir}")
    best = best_layer(direction_summary)
    if best:
        print(
            "Best local layer: {layer} "
            "(projection/logprob corr={projection_logprob_margin_pearson:.3f}, "
            "social gap={social_projection_gap:.3f})".format(**best)
        )


def default_layers(model_layer_count: int) -> list[int]:
    if model_layer_count <= 0:
        return [-1]
    stride = max(model_layer_count // 7, 1)
    layers = list(range(0, model_layer_count + 1, stride))
    if model_layer_count not in layers:
        layers.append(model_layer_count)
    return sorted(set(layers))


def build_active_cases(
    *,
    numbers: list[int],
    case_source: str,
    case_ids: set[str],
    targets_per_clue: int,
    socials_per_target: int,
    max_cases: int | None,
    seed: int,
    test_frac: float,
) -> list[SteeringCase]:
    cases: list[SteeringCase] = []
    if case_source in ("handwritten", "both"):
        cases.extend(CASES)
    if case_source in ("auto", "both"):
        cases.extend(
            auto_cases(
                numbers=numbers,
                targets_per_clue=targets_per_clue,
                socials_per_target=socials_per_target,
                max_cases=max_cases,
                seed=seed,
                test_frac=test_frac,
            )
        )
    if case_ids:
        cases = [case for case in cases if case.case_id in case_ids]
    if max_cases is not None and case_source != "auto":
        cases = cases[:max_cases]
    return cases


def auto_cases(
    *,
    numbers: list[int],
    targets_per_clue: int,
    socials_per_target: int,
    max_cases: int | None,
    seed: int,
    test_frac: float,
) -> list[SteeringCase]:
    rng = random.Random(seed)
    cases: list[SteeringCase] = []
    for private_clue in DEFAULT_CLUES:
        private_targets = [number for number in numbers if private_clue.predicate(number)]
        social_pool = [number for number in numbers if not private_clue.predicate(number)]
        rng.shuffle(private_targets)
        for private_target in private_targets[: max(targets_per_clue, 0)]:
            socials = [number for number in social_pool if number != private_target]
            rng.shuffle(socials)
            for social_number in socials[: max(socials_per_target, 0)]:
                social_clue = social_reason_clue(social_number=social_number, private_target=private_target)
                if social_clue is None:
                    continue
                cases.append(
                    SteeringCase(
                        case_id=f"auto_{private_clue.name}_{private_target}_vs_{social_number}_{social_clue.name}",
                        private_clue=private_clue.text,
                        private_target=private_target,
                        private_reason=sentence_case(private_clue.text),
                        social_number=social_number,
                        social_reason=sentence_case(social_clue.text),
                        source="auto",
                    )
                )
    rng.shuffle(cases)
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        return []
    test_count = max(1, int(round(len(cases) * min(max(test_frac, 0.0), 1.0)))) if test_frac > 0 else 0
    test_ids = {case.case_id for case in cases[:test_count]}
    return [replace(case, split="test" if case.case_id in test_ids else "train") for case in cases]


def social_reason_clue(*, social_number: int, private_target: int) -> Any | None:
    clues = matching_clues(social_number)
    if not clues:
        return None
    preferred = [clue for clue in clues if not clue.predicate(private_target)]
    return (preferred or clues)[0]


def sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] + "."


def memory_lines(case: SteeringCase, *, variant: str, m: int, strength: int) -> list[str]:
    if variant == "empty_memory" or strength <= 0:
        return []
    if variant == "target_memory":
        number = case.private_target
        reason = case.private_reason
    elif variant == "social_memory":
        number = case.social_number
        reason = case.social_reason
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    if m == 1:
        return [str(number) for _ in range(strength)]
    return [f"{number} | {reason}" for _ in range(strength)]


def collect_trials(
    *,
    backend: Any,
    cases: list[SteeringCase],
    layers: list[int],
    memory_strengths: list[int],
    m_values: list[int],
    prompt_number_range: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, np.ndarray]]]:
    trials: list[dict[str, Any]] = []
    vectors: dict[str, dict[int, np.ndarray]] = {}
    variants = ["empty_memory", "target_memory", "social_memory"]
    for case in cases:
        for m in m_values:
            for strength in memory_strengths:
                for variant in variants:
                    if variant == "empty_memory" and strength != memory_strengths[0]:
                        continue
                    trial_id = f"{case.case_id}_m{m}_mem{strength}_{variant}"
                    lines = memory_lines(case, variant=variant, m=m, strength=strength)
                    prompt_text = prompts.interaction_text(
                        numbers=list(range(1, 101)),
                        private_clue=case.private_clue,
                        memory_lines=lines,
                        m=m,
                        prompt_social_susceptibility=False,
                        prompt_number_range=prompt_number_range,
                    )
                    hidden = prompt_hidden_vectors(backend, prompt_text, layers)
                    vectors[trial_id] = hidden
                    private_ids = number_token_ids(backend, case.private_target)
                    social_ids = number_token_ids(backend, case.social_number)
                    private_logit, social_logit = number_logits_after_prefix(
                        backend,
                        prompt_text,
                        private_ids[0],
                        social_ids[0],
                        layer=None,
                        steer_vector=None,
                    )
                    private_logprob, private_logprob_tokens = number_sequence_logprob_after_prefix(
                        backend,
                        prompt_text,
                        case.private_target,
                        layer=None,
                        steer_vector=None,
                    )
                    social_logprob, social_logprob_tokens = number_sequence_logprob_after_prefix(
                        backend,
                        prompt_text,
                        case.social_number,
                        layer=None,
                        steer_vector=None,
                    )
                    trials.append(
                        {
                            "trial_id": trial_id,
                            "pair_id": f"{case.case_id}_m{m}_mem{strength}",
                            "case_id": case.case_id,
                            "m": m,
                            "memory_strength": strength,
                            "variant": variant,
                            "private_clue": case.private_clue,
                            "private_target": case.private_target,
                            "social_number": case.social_number,
                            "relation": case.relation,
                            "case_source": case.source,
                            "split": case.split,
                            "memory_lines": json.dumps(lines),
                            "private_number_token_count": len(private_ids),
                            "social_number_token_count": len(social_ids),
                            "private_number_first_token_id": private_ids[0],
                            "social_number_first_token_id": social_ids[0],
                            "private_number_logit": private_logit,
                            "social_number_logit": social_logit,
                            "social_minus_private_logit_margin": social_logit - private_logit,
                            "private_number_sequence_logprob": private_logprob,
                            "social_number_sequence_logprob": social_logprob,
                            "private_number_sequence_token_count": private_logprob_tokens,
                            "social_number_sequence_token_count": social_logprob_tokens,
                            "social_minus_private_number_logprob_margin": social_logprob - private_logprob,
                            "prompt": prompt_text,
                        }
                    )
    return trials, vectors


def collect_ood_social_trials(
    *,
    backend: Any,
    social_dirs: list[Path],
    layers: list[int],
    numbers: list[int],
    prompt_number_range: bool,
    max_prompts: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, np.ndarray]]]:
    specs = load_ood_social_prompt_specs(
        social_dirs=social_dirs,
        numbers=numbers,
        prompt_number_range=prompt_number_range,
        max_prompts=max_prompts,
    )
    trials: list[dict[str, Any]] = []
    vectors: dict[str, dict[int, np.ndarray]] = {}
    for spec in specs:
        hidden = prompt_hidden_vectors(backend, spec["prompt"], layers)
        vectors[spec["trial_id"]] = hidden
        private_ids = number_token_ids(backend, int(spec["private_target"]))
        social_ids = number_token_ids(backend, int(spec["social_number"]))
        private_logit, social_logit = number_logits_after_prefix(
            backend,
            spec["prompt"],
            private_ids[0],
            social_ids[0],
            layer=None,
            steer_vector=None,
        )
        private_logprob, private_logprob_tokens = number_sequence_logprob_after_prefix(
            backend,
            spec["prompt"],
            int(spec["private_target"]),
            layer=None,
            steer_vector=None,
        )
        social_logprob, social_logprob_tokens = number_sequence_logprob_after_prefix(
            backend,
            spec["prompt"],
            int(spec["social_number"]),
            layer=None,
            steer_vector=None,
        )
        trials.append(
            {
                **spec,
                "private_number_token_count": len(private_ids),
                "social_number_token_count": len(social_ids),
                "private_number_first_token_id": private_ids[0],
                "social_number_first_token_id": social_ids[0],
                "private_number_logit": private_logit,
                "social_number_logit": social_logit,
                "social_minus_private_logit_margin": social_logit - private_logit,
                "private_number_sequence_logprob": private_logprob,
                "social_number_sequence_logprob": social_logprob,
                "private_number_sequence_token_count": private_logprob_tokens,
                "social_number_sequence_token_count": social_logprob_tokens,
                "social_minus_private_number_logprob_margin": social_logprob - private_logprob,
            }
        )
    return trials, vectors


def load_ood_social_prompt_specs(
    *,
    social_dirs: list[Path],
    numbers: list[int],
    prompt_number_range: bool,
    max_prompts: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    messages_paths: list[Path] = []
    for social_dir in social_dirs:
        if social_dir.is_file() and social_dir.name == "messages.csv":
            messages_paths.append(social_dir)
        elif social_dir.is_dir():
            messages_paths.extend(sorted(social_dir.rglob("messages.csv")))
    for messages_path in messages_paths:
        truth_number = read_truth_number(messages_path.parent)
        if truth_number is None:
            continue
        with open(messages_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, row in enumerate(reader):
                if len(specs) >= max_prompts:
                    return specs
                if str(row.get("valid", "")).lower() not in ("true", "1"):
                    continue
                private_clue = str(row.get("speaker_private_clue") or "").strip()
                if not private_clue:
                    continue
                memory_lines = parse_list_cell(row.get("speaker_memory_before"))
                if not memory_lines:
                    continue
                social_number = majority_non_target_memory_number(memory_lines, truth_number)
                if social_number is None:
                    continue
                m = int(row.get("m") or infer_m_from_memory(memory_lines))
                try:
                    prompt_text = prompts.interaction_text(
                        numbers=numbers,
                        private_clue=private_clue,
                        memory_lines=memory_lines,
                        m=m,
                        prompt_social_susceptibility=False,
                        prompt_number_range=prompt_number_range,
                    )
                except ValueError:
                    continue
                run_slug = safe_slug(str(messages_path.parent))
                trial_id = f"ood_{run_slug}_row{row_index:04d}"
                specs.append(
                    {
                        "trial_id": trial_id,
                        "pair_id": trial_id,
                        "case_id": trial_id,
                        "m": m,
                        "memory_strength": len(memory_lines),
                        "variant": "ood_actual_social",
                        "private_clue": private_clue,
                        "private_target": truth_number,
                        "social_number": social_number,
                        "relation": "actual_social_memory",
                        "case_source": "ood_actual_social",
                        "split": "ood_actual_social",
                        "memory_lines": json.dumps(memory_lines),
                        "source_messages_csv": str(messages_path),
                        "source_row_index": row_index,
                        "source_t": row.get("t"),
                        "source_agent_id": row.get("agent_id"),
                        "source_listener_id": row.get("listener_id"),
                        "source_model": row.get("model"),
                        "prompt": prompt_text,
                    }
                )
    return specs


def read_truth_number(run_dir: Path) -> int | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return maybe_int(data.get("truth_number"))


def parse_list_cell(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def memory_number(line: str) -> int | None:
    match = re.search(r"[+-]?\d+", line)
    return int(match.group(0)) if match else None


def majority_non_target_memory_number(memory_lines: list[str], target: int) -> int | None:
    counts: dict[int, int] = {}
    for line in memory_lines:
        number = memory_number(line)
        if number is None or number == target:
            continue
        counts[number] = counts.get(number, 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda number: counts[number])


def infer_m_from_memory(memory_lines: list[str]) -> int:
    return 3 if any(" | " in line for line in memory_lines) else 1


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")[-80:] or "run"


def prompt_hidden_vectors(backend: Any, prompt_text: str, layers: list[int]) -> dict[int, np.ndarray]:
    torch = backend.torch
    text = backend._format_prompt(prompts.openai_messages(prompt_text))
    inputs = backend.tokenizer(text, return_tensors="pt")
    inputs = move_inputs_to_model(backend, inputs)
    with torch.no_grad():
        output = backend.model_obj(**inputs, output_hidden_states=True)
    hidden_states = output.hidden_states
    out: dict[int, np.ndarray] = {}
    for layer in layers:
        index = resolve_hidden_index(layer, len(hidden_states))
        if index is None:
            continue
        out[layer] = hidden_states[index][0, -1].detach().float().cpu().numpy()
    return out


def compute_direction_summary(
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    fit_split: str,
    direction_method: str,
    direction_quantile: float,
    subspace_rank: int,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    directions: dict[int, np.ndarray] = {}
    summary: list[dict[str, Any]] = []

    for layer in layers:
        fit = fit_direction(
            trials=trials,
            vectors=vectors,
            layer=layer,
            fit_split=fit_split,
            direction_method=direction_method,
            direction_quantile=direction_quantile,
            subspace_rank=subspace_rank,
        )
        if fit is None:
            continue
        direction = fit["direction"]
        pair_norms = fit["pair_norms"]
        fit_count = int(fit["fit_count"])
        if float(np.linalg.norm(direction)) == 0.0:
            continue
        directions[layer] = direction
        unit = normalize(direction)

        for row in trials:
            vec = vectors[row["trial_id"]].get(layer)
            if vec is None:
                continue
            projection = float(np.dot(vec, unit))
            row[f"projection_layer_{layer}"] = projection

        for eval_split in eval_splits(trials):
            eval_rows = [
                row
                for row in trials
                if eval_split == "all" or row.get("split") == eval_split
            ]
            projection_rows = [
                (
                    float(row[f"projection_layer_{layer}"]),
                    float(row["social_minus_private_number_logprob_margin"]),
                    str(row["variant"]),
                )
                for row in eval_rows
                if f"projection_layer_{layer}" in row
            ]
            social_proj = [p for p, _, variant in projection_rows if variant == "social_memory"]
            target_proj = [p for p, _, variant in projection_rows if variant == "target_memory"]
            summary.append(
                {
                    "direction": f"social_minus_private_{direction_method}",
                    "direction_method": direction_method,
                    "direction_quantile": direction_quantile,
                    "subspace_rank": subspace_rank,
                    "layer": layer,
                    "fit_split": fit_split,
                    "eval_split": eval_split,
                    "n_fit_pairs": fit_count,
                    "n_fit_rows": fit.get("fit_row_count", fit_count),
                    "n_eval_rows": len(projection_rows),
                    "mean_pair_norm": mean(pair_norms),
                    "direction_norm": float(np.linalg.norm(direction)),
                    "social_projection_mean": mean(social_proj),
                    "target_projection_mean": mean(target_proj),
                    "social_projection_gap": mean(social_proj) - mean(target_proj),
                    "projection_logprob_margin_pearson": pearson(
                        [p for p, _, _ in projection_rows],
                        [m for _, m, _ in projection_rows],
                    ),
                    "projection_logit_margin_pearson": pearson(
                        [p for p, _, _ in projection_rows],
                        [m for _, m, _ in projection_rows],
                    ),
                }
            )
    return directions, summary


def fit_direction(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layer: int,
    fit_split: str,
    direction_method: str,
    direction_quantile: float,
    subspace_rank: int,
    pair_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    if direction_method == "memory_contrast":
        return fit_memory_contrast_direction(
            trials=trials,
            vectors=vectors,
            layer=layer,
            fit_split=fit_split,
            pair_ids=pair_ids,
        )
    if direction_method == "logprob_quantile":
        return fit_logprob_quantile_direction(
            trials=trials,
            vectors=vectors,
            layer=layer,
            fit_split=fit_split,
            direction_quantile=direction_quantile,
            pair_ids=pair_ids,
        )
    if direction_method == "svd_subspace":
        return fit_svd_subspace_direction(
            trials=trials,
            vectors=vectors,
            layer=layer,
            fit_split=fit_split,
            subspace_rank=subspace_rank,
            pair_ids=pair_ids,
        )
    raise ValueError(f"Unknown direction_method={direction_method!r}")


def fit_memory_contrast_direction(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layer: int,
    fit_split: str,
    pair_ids: list[str] | None,
) -> dict[str, Any] | None:
    by_pair_variant = {(row["pair_id"], row["variant"]): row for row in trials}
    active_pair_ids = pair_ids or fit_pair_ids(trials, fit_split)
    diffs: list[np.ndarray] = []
    pair_norms: list[float] = []
    for pair_id in active_pair_ids:
        target = by_pair_variant.get((pair_id, "target_memory"))
        social = by_pair_variant.get((pair_id, "social_memory"))
        if target is None or social is None:
            continue
        target_vec = vectors[target["trial_id"]].get(layer)
        social_vec = vectors[social["trial_id"]].get(layer)
        if target_vec is None or social_vec is None:
            continue
        diff = social_vec - target_vec
        diffs.append(diff)
        pair_norms.append(float(np.linalg.norm(diff)))
    if not diffs:
        return None
    return {
        "direction": np.mean(np.stack(diffs, axis=0), axis=0),
        "pair_norms": pair_norms,
        "fit_count": len(diffs),
        "fit_row_count": len(diffs) * 2,
    }


def fit_logprob_quantile_direction(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layer: int,
    fit_split: str,
    direction_quantile: float,
    pair_ids: list[str] | None,
) -> dict[str, Any] | None:
    allowed_pair_ids = set(pair_ids) if pair_ids is not None else None
    rows = [
        row
        for row in trials
        if row.get("variant") in ("target_memory", "social_memory")
        and (fit_split == "all" or row.get("split") == fit_split)
        and (allowed_pair_ids is None or row.get("pair_id") in allowed_pair_ids)
        and row.get("social_minus_private_number_logprob_margin") not in (None, "")
        and vectors.get(row["trial_id"], {}).get(layer) is not None
    ]
    if len(rows) < 4:
        return None
    rows = sorted(rows, key=lambda row: float(row["social_minus_private_number_logprob_margin"]))
    q = min(max(direction_quantile, 0.05), 0.5)
    bucket_size = max(1, int(round(len(rows) * q)))
    low_rows = rows[:bucket_size]
    high_rows = rows[-bucket_size:]
    low_vectors = [vectors[row["trial_id"]][layer] for row in low_rows]
    high_vectors = [vectors[row["trial_id"]][layer] for row in high_rows]
    low_mean = np.mean(np.stack(low_vectors, axis=0), axis=0)
    high_mean = np.mean(np.stack(high_vectors, axis=0), axis=0)
    pair_norms = [
        float(np.linalg.norm(high_vectors[index] - low_vectors[index]))
        for index in range(min(len(high_vectors), len(low_vectors)))
    ]
    return {
        "direction": high_mean - low_mean,
        "pair_norms": pair_norms,
        "fit_count": min(len(high_vectors), len(low_vectors)),
        "fit_row_count": len(high_vectors) + len(low_vectors),
    }


def fit_svd_subspace_direction(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layer: int,
    fit_split: str,
    subspace_rank: int,
    pair_ids: list[str] | None,
) -> dict[str, Any] | None:
    memory_fit = fit_memory_contrast_direction(
        trials=trials,
        vectors=vectors,
        layer=layer,
        fit_split=fit_split,
        pair_ids=pair_ids,
    )
    if memory_fit is None:
        return None
    by_pair_variant = {(row["pair_id"], row["variant"]): row for row in trials}
    active_pair_ids = pair_ids or fit_pair_ids(trials, fit_split)
    diffs: list[np.ndarray] = []
    for pair_id in active_pair_ids:
        target = by_pair_variant.get((pair_id, "target_memory"))
        social = by_pair_variant.get((pair_id, "social_memory"))
        if target is None or social is None:
            continue
        target_vec = vectors[target["trial_id"]].get(layer)
        social_vec = vectors[social["trial_id"]].get(layer)
        if target_vec is not None and social_vec is not None:
            diffs.append(social_vec - target_vec)
    if len(diffs) < 2:
        return memory_fit
    matrix = np.stack(diffs, axis=0)
    mean_direction = np.asarray(memory_fit["direction"])
    try:
        _u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return memory_fit
    rank = max(1, min(subspace_rank, vt.shape[0]))
    direction = np.zeros_like(mean_direction)
    for index in range(rank):
        component = vt[index]
        if float(np.dot(component, mean_direction)) < 0.0:
            component = -component
        direction = direction + float(singular_values[index]) * component
    return {
        "direction": direction,
        "pair_norms": list(memory_fit["pair_norms"]),
        "fit_count": int(memory_fit["fit_count"]),
        "fit_row_count": int(memory_fit["fit_row_count"]),
    }


def fit_pair_ids(trials: list[dict[str, Any]], fit_split: str) -> list[str]:
    return sorted(
        {
            row["pair_id"]
            for row in trials
            if row.get("variant") in ("target_memory", "social_memory")
            and (fit_split == "all" or row.get("split") == fit_split)
        }
    )


def eval_splits(trials: list[dict[str, Any]]) -> list[str]:
    splits = sorted({str(row.get("split", "")) for row in trials if row.get("split")})
    return ["all", *splits]


def add_projections_to_trials(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    directions: dict[int, np.ndarray],
    layers: list[int],
) -> None:
    for layer in layers:
        direction = directions.get(layer)
        if direction is None:
            continue
        unit = normalize(direction)
        for row in trials:
            vec = vectors.get(str(row.get("trial_id")), {}).get(layer)
            if vec is not None:
                row[f"projection_layer_{layer}"] = float(np.dot(vec, unit))


def projection_distribution_rows(trials: list[dict[str, Any]], layers: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        for layer in layers:
            key = f"projection_layer_{layer}"
            if key not in trial:
                continue
            rows.append(
                {
                    "layer": layer,
                    "split": trial.get("split", ""),
                    "case_id": trial.get("case_id", ""),
                    "pair_id": trial.get("pair_id", ""),
                    "m": trial.get("m", ""),
                    "memory_strength": trial.get("memory_strength", ""),
                    "variant": trial.get("variant", ""),
                    "projection": trial[key],
                    "private_target": trial.get("private_target", ""),
                    "social_number": trial.get("social_number", ""),
                    "social_minus_private_number_logprob_margin": trial.get(
                        "social_minus_private_number_logprob_margin", ""
                    ),
                    "social_minus_private_logit_margin": trial.get("social_minus_private_logit_margin", ""),
                }
            )
    return rows


def data_scaling_curve(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    fit_split: str,
    sizes: list[int],
    seed: int,
    direction_method: str,
    direction_quantile: float,
    subspace_rank: int,
) -> list[dict[str, Any]]:
    if not sizes:
        return []
    fit_case_ids = sorted(
        {
            row["case_id"]
            for row in trials
            if row["variant"] in ("target_memory", "social_memory")
            and (fit_split == "all" or row.get("split") == fit_split)
        }
    )
    eval_rows_by_split = {
        split: [
            row
            for row in trials
            if row["variant"] in ("target_memory", "social_memory")
            and (split == "all" or row.get("split") == split)
        ]
        for split in eval_splits(trials)
    }
    rng = random.Random(seed)
    shuffled_cases = list(fit_case_ids)
    rng.shuffle(shuffled_cases)
    rows: list[dict[str, Any]] = []
    pair_ids_by_case: dict[str, set[str]] = {}
    for row in trials:
        if row["variant"] in ("target_memory", "social_memory"):
            pair_ids_by_case.setdefault(str(row["case_id"]), set()).add(str(row["pair_id"]))
    for requested_size in sorted(set(size for size in sizes if size > 0)):
        selected_cases = shuffled_cases[: min(requested_size, len(shuffled_cases))]
        selected_pairs = sorted({pair_id for case_id in selected_cases for pair_id in pair_ids_by_case.get(case_id, set())})
        if len(selected_cases) < requested_size:
            rows.append(
                {
                    "requested_case_count": requested_size,
                    "fit_case_count": len(selected_cases),
                    "fit_pair_count": len(selected_pairs),
                    "status": "insufficient_collected_cases",
                }
            )
            continue
        for layer in layers:
            fit = fit_direction(
                trials=trials,
                vectors=vectors,
                layer=layer,
                fit_split=fit_split,
                direction_method=direction_method,
                direction_quantile=direction_quantile,
                subspace_rank=subspace_rank,
                pair_ids=selected_pairs,
            )
            if fit is None:
                continue
            direction = fit["direction"]
            pair_norms = fit["pair_norms"]
            unit = normalize(direction)
            for eval_split, eval_rows in eval_rows_by_split.items():
                projections: list[float] = []
                margins: list[float] = []
                social_proj: list[float] = []
                target_proj: list[float] = []
                for row in eval_rows:
                    vec = vectors[row["trial_id"]].get(layer)
                    if vec is None:
                        continue
                    projection = float(np.dot(vec, unit))
                    projections.append(projection)
                    margins.append(float(row["social_minus_private_number_logprob_margin"]))
                    if row["variant"] == "social_memory":
                        social_proj.append(projection)
                    elif row["variant"] == "target_memory":
                        target_proj.append(projection)
                rows.append(
                    {
                        "requested_case_count": requested_size,
                        "fit_case_count": len(selected_cases),
                        "fit_pair_count": len(diffs),
                        "status": "ok",
                        "direction_method": direction_method,
                        "direction_quantile": direction_quantile,
                        "subspace_rank": subspace_rank,
                        "layer": layer,
                        "fit_split": fit_split,
                        "eval_split": eval_split,
                        "direction_norm": float(np.linalg.norm(direction)),
                        "mean_pair_norm": mean(pair_norms),
                        "fit_row_count": fit.get("fit_row_count", fit.get("fit_count")),
                        "social_projection_gap": mean(social_proj) - mean(target_proj),
                        "projection_logprob_margin_pearson": pearson(projections, margins),
                    }
                )
    return rows


def vector_stability_rows(
    *,
    trials: list[dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    fit_split: str,
    sizes: list[int],
    subset_count: int,
    seeds: list[int],
    direction_method: str,
    direction_quantile: float,
    subspace_rank: int,
) -> list[dict[str, Any]]:
    if subset_count <= 0:
        return []
    fit_case_ids = sorted(
        {
            row["case_id"]
            for row in trials
            if row["variant"] in ("target_memory", "social_memory")
            and (fit_split == "all" or row.get("split") == fit_split)
        }
    )
    if len(fit_case_ids) < 2:
        return []
    active_sizes = sorted(set(size for size in sizes if 1 < size <= len(fit_case_ids))) or [len(fit_case_ids)]
    active_seeds = seeds or [0, 1, 2]
    pair_ids_by_case: dict[str, set[str]] = {}
    for row in trials:
        if row["variant"] in ("target_memory", "social_memory"):
            pair_ids_by_case.setdefault(str(row["case_id"]), set()).add(str(row["pair_id"]))
    rows: list[dict[str, Any]] = []
    for requested_size in active_sizes:
        subset_vectors: dict[int, list[tuple[str, np.ndarray]]] = {layer: [] for layer in layers}
        for seed in active_seeds:
            rng = random.Random(seed)
            for subset_index in range(subset_count):
                selected_cases = list(fit_case_ids)
                rng.shuffle(selected_cases)
                selected_cases = selected_cases[:requested_size]
                selected_pairs = sorted({pair_id for case_id in selected_cases for pair_id in pair_ids_by_case.get(case_id, set())})
                for layer in layers:
                    fit = fit_direction(
                        trials=trials,
                        vectors=vectors,
                        layer=layer,
                        fit_split=fit_split,
                        direction_method=direction_method,
                        direction_quantile=direction_quantile,
                        subspace_rank=subspace_rank,
                        pair_ids=selected_pairs,
                    )
                    if fit is not None:
                        subset_vectors[layer].append((f"seed{seed}_subset{subset_index}", normalize(fit["direction"])))
        for layer, vectors_for_layer in subset_vectors.items():
            cosines: list[float] = []
            for left_index in range(len(vectors_for_layer)):
                for right_index in range(left_index + 1, len(vectors_for_layer)):
                    cosines.append(cosine(vectors_for_layer[left_index][1], vectors_for_layer[right_index][1]))
            rows.append(
                {
                    "requested_case_count": requested_size,
                    "fit_split": fit_split,
                    "direction_method": direction_method,
                    "direction_quantile": direction_quantile,
                    "subspace_rank": subspace_rank,
                    "layer": layer,
                    "subset_case_count": requested_size,
                    "subset_count": len(vectors_for_layer),
                    "mean_pairwise_cosine": mean(cosines),
                    "min_pairwise_cosine": min(cosines) if cosines else float("nan"),
                    "max_pairwise_cosine": max(cosines) if cosines else float("nan"),
                }
            )
    return rows


def direction_for_pairs(
    pair_ids: list[str],
    by_pair_variant: dict[tuple[str, str], dict[str, Any]],
    vectors: dict[str, dict[int, np.ndarray]],
    layer: int,
) -> np.ndarray | None:
    diffs: list[np.ndarray] = []
    for pair_id in pair_ids:
        target = by_pair_variant.get((pair_id, "target_memory"))
        social = by_pair_variant.get((pair_id, "social_memory"))
        if target is None or social is None:
            continue
        target_vec = vectors[target["trial_id"]].get(layer)
        social_vec = vectors[social["trial_id"]].get(layer)
        if target_vec is None or social_vec is None:
            continue
        diffs.append(social_vec - target_vec)
    if not diffs:
        return None
    return np.mean(np.stack(diffs, axis=0), axis=0)


def run_alpha_sweep(
    *,
    backend: Any,
    trials: list[dict[str, Any]],
    directions: dict[int, np.ndarray],
    direction_summary: list[dict[str, Any]],
    layers: list[int],
    alphas: list[float],
    max_alpha_trials: int | None,
    eval_variants: set[str] | None = None,
) -> list[dict[str, Any]]:
    scale_by_layer: dict[int, float] = {}
    for row in direction_summary:
        scale_by_layer.setdefault(int(row["layer"]), float(row["mean_pair_norm"]))
    rows: list[dict[str, Any]] = []
    active_variants = eval_variants or {"target_memory", "social_memory"}
    eval_trials = [
        row
        for row in trials
        if row.get("variant") in active_variants
        and row.get("private_target") is not None
        and row.get("social_number") is not None
    ]
    if max_alpha_trials is not None:
        eval_trials = eval_trials[:max_alpha_trials]
    for layer in layers:
        direction = directions.get(layer)
        if direction is None:
            continue
        unit = normalize(direction)
        scale = scale_by_layer.get(layer, float(np.linalg.norm(direction)))
        for alpha in alphas:
            margins: list[float] = []
            private_logprobs: list[float] = []
            social_logprobs: list[float] = []
            token_counts: list[int] = []
            for row in eval_trials:
                prompt_text = str(row["prompt"])
                private_logprob, private_token_count = number_sequence_logprob_after_prefix(
                    backend,
                    prompt_text,
                    int(row["private_target"]),
                    layer=layer,
                    steer_vector=unit * scale * alpha,
                )
                social_logprob, social_token_count = number_sequence_logprob_after_prefix(
                    backend,
                    prompt_text,
                    int(row["social_number"]),
                    layer=layer,
                    steer_vector=unit * scale * alpha,
                )
                private_logprobs.append(private_logprob)
                social_logprobs.append(social_logprob)
                margins.append(social_logprob - private_logprob)
                token_counts.append(private_token_count + social_token_count)
            rows.append(
                {
                    "direction": "social_minus_private_memory",
                    "layer": layer,
                    "alpha": alpha,
                    "n": len(margins),
                    "mean_social_minus_private_number_logprob_margin": mean(margins),
                    "mean_social_minus_private_number_logit_margin": mean(margins),
                    "mean_private_number_sequence_logprob": mean(private_logprobs),
                    "mean_social_number_sequence_logprob": mean(social_logprobs),
                    "mean_scored_number_token_count": mean([float(value) for value in token_counts]),
                    "mean_social_number_probability": mean_logistic(margins),
                }
            )
    return rows


def calibrate_steering_signs(steering_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in steering_rows:
        rows_by_layer.setdefault(int(row["layer"]), []).append(row)
    sign_rows: list[dict[str, Any]] = []
    for layer, rows in sorted(rows_by_layer.items()):
        xs = [float(row["alpha"]) for row in rows]
        ys = [float(row["mean_social_minus_private_number_logprob_margin"]) for row in rows]
        slope = linear_slope(xs, ys)
        social_sign = 1 if math.isnan(slope) or slope >= 0 else -1
        for row in rows:
            row["raw_alpha"] = row["alpha"]
            row["empirical_social_sign"] = social_sign
            row["calibrated_alpha"] = float(row["alpha"]) * social_sign
            row["calibration_note"] = (
                "positive calibrated_alpha increases social-minus-private margin"
                if social_sign == 1
                else "raw vector is causally flipped; negative raw_alpha increases social-minus-private margin"
            )
        sign_rows.append(
            {
                "layer": layer,
                "raw_alpha_social_logprob_margin_slope": slope,
                "raw_alpha_social_margin_slope": slope,
                "empirical_social_sign": social_sign,
                "raw_positive_alpha_effect": "more_social" if social_sign == 1 else "more_private",
                "calibrated_positive_alpha_effect": "more_social",
                "n_alpha_points": len(rows),
            }
        )
    return sign_rows


def apply_existing_sign_calibration(steering_rows: list[dict[str, Any]], sign_rows: list[dict[str, Any]]) -> None:
    sign_by_layer = {int(row["layer"]): int(row["empirical_social_sign"]) for row in sign_rows}
    for row in steering_rows:
        sign = sign_by_layer.get(int(row["layer"]), 1)
        row["raw_alpha"] = row["alpha"]
        row["empirical_social_sign"] = sign
        row["calibrated_alpha"] = float(row["alpha"]) * sign
        row["calibration_note"] = "sign inherited from synthetic contrast calibration"


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    denom = float(np.sum((x - float(np.mean(x))) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum((x - float(np.mean(x))) * (y - float(np.mean(y)))) / denom)


def number_logits_after_prefix(
    backend: Any,
    prompt_text: str,
    private_token_id: int,
    social_token_id: int,
    *,
    layer: int | None,
    steer_vector: np.ndarray | None,
) -> tuple[float, float]:
    torch = backend.torch
    prompt = backend._format_prompt(prompts.openai_messages(prompt_text))
    text = prompt + '{"number":'
    inputs = backend.tokenizer(text, return_tensors="pt")
    inputs = move_inputs_to_model(backend, inputs)

    handle = None
    if layer is not None and steer_vector is not None and layer > 0:
        module = layer_module(backend.model_obj, layer)
        if module is not None:
            vector = torch.tensor(steer_vector, device=inputs["input_ids"].device)

            def hook(_module: Any, _args: Any, output: Any) -> Any:
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    hidden[:, -1, :] = hidden[:, -1, :] + vector.to(device=hidden.device, dtype=hidden.dtype)
                    return (hidden, *output[1:])
                hidden = output.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + vector.to(device=hidden.device, dtype=hidden.dtype)
                return hidden

            handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = backend.model_obj(**inputs)
        logits = output.logits[0, -1].detach().float().cpu()
    finally:
        if handle is not None:
            handle.remove()
    return float(logits[private_token_id]), float(logits[social_token_id])


def number_sequence_logprob_after_prefix(
    backend: Any,
    prompt_text: str,
    number: int,
    *,
    layer: int | None,
    steer_vector: np.ndarray | None,
) -> tuple[float, int]:
    prompt = backend._format_prompt(prompts.openai_messages(prompt_text))
    return continuation_logprob(
        backend,
        prompt + '{"number":',
        str(number),
        layer=layer,
        steer_vector=steer_vector,
    )


def completion_logprob_after_prompt(
    backend: Any,
    prompt_text: str,
    completion_text: str,
    *,
    layer: int | None,
    steer_vector: np.ndarray | None,
) -> tuple[float, int]:
    prompt = backend._format_prompt(prompts.openai_messages(prompt_text))
    return continuation_logprob(
        backend,
        prompt,
        completion_text,
        layer=layer,
        steer_vector=steer_vector,
    )


def continuation_logprob(
    backend: Any,
    prefix_text: str,
    continuation_text: str,
    *,
    layer: int | None,
    steer_vector: np.ndarray | None,
) -> tuple[float, int]:
    torch = backend.torch
    prefix = backend.tokenizer(prefix_text, return_tensors="pt")
    continuation_ids = backend.tokenizer(continuation_text, add_special_tokens=False)["input_ids"]
    continuation_ids = [int(token_id) for token_id in continuation_ids]
    if not continuation_ids:
        return 0.0, 0
    prefix = move_inputs_to_model(backend, prefix)
    input_ids = prefix["input_ids"]
    attention_mask = prefix.get("attention_mask")
    extra = torch.tensor([continuation_ids], device=input_ids.device, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, extra], dim=1)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = torch.cat([attention_mask, torch.ones_like(extra)], dim=1)
    model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    prefix_len = int(prefix["input_ids"].shape[-1])
    positions = list(range(max(prefix_len - 1, 0), prefix_len + len(continuation_ids) - 1))
    handle = register_steering_hook(
        backend,
        layer=layer,
        steer_vector=steer_vector,
        positions=positions,
    )
    try:
        with torch.no_grad():
            output = backend.model_obj(**model_inputs)
        logits = output.logits[0].detach().float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        total = 0.0
        for offset, token_id in enumerate(continuation_ids):
            logit_position = prefix_len + offset - 1
            total += float(log_probs[logit_position, token_id].cpu())
    finally:
        if handle is not None:
            handle.remove()
    return total, len(continuation_ids)


def register_steering_hook(
    backend: Any,
    *,
    layer: int | None,
    steer_vector: np.ndarray | None,
    positions: list[int] | None = None,
) -> Any | None:
    if layer is None or steer_vector is None or layer <= 0:
        return None
    module = layer_module(backend.model_obj, layer)
    if module is None:
        return None
    torch = backend.torch
    vector = torch.tensor(steer_vector)

    def hook(_module: Any, _args: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0].clone()
            rest = output[1:]
        else:
            hidden = output.clone()
            rest = None
        active_vector = vector.to(device=hidden.device, dtype=hidden.dtype)
        if positions is None:
            hidden[:, -1, :] = hidden[:, -1, :] + active_vector
        else:
            valid_positions = [position for position in positions if 0 <= position < hidden.shape[1]]
            if valid_positions:
                hidden[:, valid_positions, :] = hidden[:, valid_positions, :] + active_vector
        if rest is not None:
            return (hidden, *rest)
        return hidden

    return module.register_forward_hook(hook)


def number_token_ids(backend: Any, number: int) -> list[int]:
    ids = backend.tokenizer(str(number), add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"Tokenizer returned no ids for number {number}")
    return [int(token_id) for token_id in ids]


def move_inputs_to_model(backend: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    model_device = getattr(backend.model_obj, "device", None)
    if model_device is not None:
        return {key: value.to(model_device) for key, value in inputs.items()}
    return inputs


def get_transformer_layers(model: Any) -> list[Any]:
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return list(base.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Could not locate transformer blocks for steering hooks.")


def layer_module(model: Any, hidden_state_layer: int) -> Any | None:
    blocks = get_transformer_layers(model)
    block_index = hidden_state_layer - 1
    if block_index < 0 or block_index >= len(blocks):
        return None
    return blocks[block_index]


def resolve_hidden_index(layer: int, hidden_state_count: int) -> int | None:
    index = layer if layer >= 0 else hidden_state_count + layer
    if index < 0 or index >= hidden_state_count:
        return None
    return index


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def mean_logistic(margins: list[float]) -> float:
    if not margins:
        return float("nan")
    probs = [1.0 / (1.0 + math.exp(-max(min(value, 60.0), -60.0))) for value in margins]
    return mean(probs)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def run_generation_steering(
    *,
    backend: Any,
    trials: list[dict[str, Any]],
    directions: dict[int, np.ndarray],
    direction_summary: list[dict[str, Any]],
    sign_rows: list[dict[str, Any]],
    layers: list[int],
    alphas: list[float],
    max_trials: int,
    dataset_name: str,
) -> list[dict[str, Any]]:
    scale_by_layer: dict[int, float] = {}
    for row in direction_summary:
        scale_by_layer.setdefault(int(row["layer"]), float(row["mean_pair_norm"]))
    sign_by_layer = {int(row["layer"]): int(row["empirical_social_sign"]) for row in sign_rows}
    eval_trials = [
        row
        for row in trials
        if row.get("variant") in ("target_memory", "social_memory", "ood_actual_social")
        and row.get("prompt")
        and row.get("private_target") is not None
        and row.get("social_number") is not None
    ][: max(max_trials, 0)]
    rows: list[dict[str, Any]] = []
    for layer in layers:
        direction = directions.get(layer)
        if direction is None:
            continue
        unit = normalize(direction) * sign_by_layer.get(layer, 1)
        scale = scale_by_layer.get(layer, float(np.linalg.norm(direction)))
        for alpha in alphas:
            steer_vector = unit * scale * alpha
            for trial in eval_trials:
                prompt_text = str(trial["prompt"])
                raw_text, completion_tokens = generate_text_with_steering(
                    backend,
                    prompt_text,
                    layer=layer,
                    steer_vector=steer_vector,
                )
                parsed_number: int | None = None
                parsed_reason: str | None = None
                parse_error: str | None = None
                valid = False
                try:
                    parsed = parse_number_message(raw_text, allowed_numbers=[], m=int(trial.get("m", 1)))
                    parsed_number = parsed.number
                    parsed_reason = parsed.reason
                    valid = True
                except ParseError as exc:
                    parse_error = str(exc)
                strict_json = is_strict_json_object(raw_text)
                clue_ok = (
                    clue_matches_number(str(trial.get("private_clue")), parsed_number)
                    if parsed_number is not None
                    else None
                )
                base_completion_logprob, base_completion_tokens = completion_logprob_after_prompt(
                    backend,
                    prompt_text,
                    raw_text,
                    layer=None,
                    steer_vector=None,
                )
                steered_completion_logprob, steered_completion_tokens = completion_logprob_after_prompt(
                    backend,
                    prompt_text,
                    raw_text,
                    layer=layer,
                    steer_vector=steer_vector,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "trial_id": trial.get("trial_id"),
                        "case_id": trial.get("case_id"),
                        "split": trial.get("split"),
                        "variant": trial.get("variant"),
                        "m": trial.get("m"),
                        "memory_strength": trial.get("memory_strength"),
                        "layer": layer,
                        "calibrated_alpha": alpha,
                        "private_clue": trial.get("private_clue"),
                        "private_target": trial.get("private_target"),
                        "social_number": trial.get("social_number"),
                        "raw_output": raw_text,
                        "parsed_number": parsed_number,
                        "parsed_reason": parsed_reason,
                        "valid": valid,
                        "strict_json": strict_json,
                        "format_damage": not strict_json,
                        "parse_error": parse_error,
                        "satisfies_private_clue": clue_ok,
                        "choice_category": choice_category(
                            parsed_number,
                            private_target=maybe_int(trial.get("private_target")),
                            social_number=maybe_int(trial.get("social_number")),
                            private_clue=str(trial.get("private_clue")),
                        ),
                        "completion_tokens": completion_tokens,
                        "base_completion_token_count": base_completion_tokens,
                        "base_completion_logprob": base_completion_logprob,
                        "base_completion_per_token_nll": per_token_nll(
                            base_completion_logprob,
                            base_completion_tokens,
                        ),
                        "base_completion_perplexity": perplexity_from_logprob(
                            base_completion_logprob,
                            base_completion_tokens,
                        ),
                        "steered_completion_token_count": steered_completion_tokens,
                        "steered_completion_logprob": steered_completion_logprob,
                        "steered_completion_per_token_nll": per_token_nll(
                            steered_completion_logprob,
                            steered_completion_tokens,
                        ),
                        "steered_completion_perplexity": perplexity_from_logprob(
                            steered_completion_logprob,
                            steered_completion_tokens,
                        ),
                        "prompt": prompt_text,
                    }
                )
    return rows


def generate_text_with_steering(
    backend: Any,
    prompt_text: str,
    *,
    layer: int,
    steer_vector: np.ndarray,
) -> tuple[str, int]:
    torch = backend.torch
    prompt = backend._format_prompt(prompts.openai_messages(prompt_text))
    inputs = backend.tokenizer(prompt, return_tensors="pt")
    inputs = move_inputs_to_model(backend, inputs)
    input_token_count = int(inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": backend.max_tokens,
        "pad_token_id": backend.tokenizer.eos_token_id,
    }
    if backend.temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": backend.temperature, "top_p": backend.top_p})
    else:
        generation_kwargs["do_sample"] = False
    handle = register_steering_hook(backend, layer=layer, steer_vector=steer_vector, positions=None)
    try:
        with torch.no_grad():
            generated = backend.model_obj.generate(**inputs, **generation_kwargs)
    finally:
        if handle is not None:
            handle.remove()
    completion_ids = generated[0][input_token_count:]
    text = backend.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text, int(completion_ids.shape[-1])


def generation_side_effect_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("layer"), row.get("calibrated_alpha"))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, layer, alpha), group in sorted(groups.items(), key=lambda item: (str(item[0][0]), int(item[0][1]), float(item[0][2]))):
        valid = [row for row in group if bool(row.get("valid"))]
        summary.append(
            {
                "dataset": dataset,
                "layer": layer,
                "calibrated_alpha": alpha,
                "n": len(group),
                "valid_rate": sum(bool(row.get("valid")) for row in group) / max(len(group), 1),
                "strict_json_rate": sum(bool(row.get("strict_json")) for row in group) / max(len(group), 1),
                "format_damage_rate": sum(bool(row.get("format_damage")) for row in group) / max(len(group), 1),
                "satisfies_private_clue_rate": mean(
                    [1.0 if row.get("satisfies_private_clue") else 0.0 for row in valid if row.get("satisfies_private_clue") is not None]
                ),
                "social_choice_rate": sum(row.get("choice_category") == "social" for row in valid) / max(len(valid), 1),
                "private_target_choice_rate": sum(row.get("choice_category") == "private_target" for row in valid) / max(len(valid), 1),
                "other_clue_compatible_rate": sum(row.get("choice_category") == "other_clue_compatible" for row in valid) / max(len(valid), 1),
                "incompatible_rate": sum(row.get("choice_category") == "incompatible" for row in valid) / max(len(valid), 1),
                "mean_base_completion_perplexity": mean(
                    [
                        float(row["base_completion_perplexity"])
                        for row in group
                        if row.get("base_completion_perplexity") not in (None, "") and not math.isnan(float(row["base_completion_perplexity"]))
                    ]
                ),
                "mean_steered_completion_perplexity": mean(
                    [
                        float(row["steered_completion_perplexity"])
                        for row in group
                        if row.get("steered_completion_perplexity") not in (None, "") and not math.isnan(float(row["steered_completion_perplexity"]))
                    ]
                ),
            }
        )
    return summary


def behavioral_steering_effect_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset_layer: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset_layer.setdefault((row.get("dataset"), row.get("layer")), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, layer), group in sorted(by_dataset_layer.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))):
        sorted_rows = sorted(group, key=lambda row: float(row["calibrated_alpha"]))
        baseline = nearest_alpha_row(sorted_rows, 0.0)
        positive = max((row for row in sorted_rows if float(row["calibrated_alpha"]) > 0), key=lambda row: float(row["calibrated_alpha"]), default=None)
        negative = min((row for row in sorted_rows if float(row["calibrated_alpha"]) < 0), key=lambda row: float(row["calibrated_alpha"]), default=None)
        best_social = best_social_behavior_row(sorted_rows)
        if baseline is None:
            continue
        out.append(
            {
                "dataset": dataset,
                "layer": layer,
                "baseline_alpha": baseline.get("calibrated_alpha"),
                "positive_alpha": positive.get("calibrated_alpha") if positive is not None else None,
                "negative_alpha": negative.get("calibrated_alpha") if negative is not None else None,
                "best_social_alpha": best_social.get("calibrated_alpha") if best_social is not None else None,
                "baseline_social_choice_rate": baseline.get("social_choice_rate"),
                "positive_social_choice_rate": positive.get("social_choice_rate") if positive is not None else None,
                "negative_social_choice_rate": negative.get("social_choice_rate") if negative is not None else None,
                "best_social_choice_rate": best_social.get("social_choice_rate") if best_social is not None else None,
                "positive_social_choice_delta": numeric_delta(positive, baseline, "social_choice_rate"),
                "negative_social_choice_delta": numeric_delta(negative, baseline, "social_choice_rate"),
                "best_social_choice_delta": numeric_delta(best_social, baseline, "social_choice_rate"),
                "baseline_private_target_choice_rate": baseline.get("private_target_choice_rate"),
                "positive_private_target_choice_delta": numeric_delta(positive, baseline, "private_target_choice_rate"),
                "best_private_target_choice_delta": numeric_delta(best_social, baseline, "private_target_choice_rate"),
                "baseline_satisfies_private_clue_rate": baseline.get("satisfies_private_clue_rate"),
                "positive_satisfies_private_clue_delta": numeric_delta(positive, baseline, "satisfies_private_clue_rate"),
                "best_satisfies_private_clue_delta": numeric_delta(best_social, baseline, "satisfies_private_clue_rate"),
                "baseline_valid_rate": baseline.get("valid_rate"),
                "positive_valid_rate_delta": numeric_delta(positive, baseline, "valid_rate"),
                "best_valid_rate_delta": numeric_delta(best_social, baseline, "valid_rate"),
                "baseline_format_damage_rate": baseline.get("format_damage_rate"),
                "positive_format_damage_delta": numeric_delta(positive, baseline, "format_damage_rate"),
                "best_format_damage_delta": numeric_delta(best_social, baseline, "format_damage_rate"),
            }
        )
    return out


def nearest_alpha_row(rows: list[dict[str, Any]], target_alpha: float) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: abs(float(row["calibrated_alpha"]) - target_alpha))


def best_social_behavior_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [
        row
        for row in rows
        if float(row.get("valid_rate", 0.0) or 0.0) >= 0.95
        and float(row.get("format_damage_rate", 1.0) or 1.0) <= 0.05
    ]
    active = viable or rows
    if not active:
        return None
    return max(
        active,
        key=lambda row: (
            float(row.get("social_choice_rate", 0.0) or 0.0),
            -float(row.get("format_damage_rate", 1.0) or 1.0),
            float(row.get("valid_rate", 0.0) or 0.0),
        ),
    )


def numeric_delta(row: dict[str, Any] | None, baseline: dict[str, Any], key: str) -> float | None:
    if row is None or row.get(key) in (None, "") or baseline.get(key) in (None, ""):
        return None
    try:
        return float(row[key]) - float(baseline[key])
    except (TypeError, ValueError):
        return None


def is_strict_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def choice_category(
    number: int | None,
    *,
    private_target: int | None,
    social_number: int | None,
    private_clue: str,
) -> str:
    if number is None:
        return "invalid"
    if private_target is not None and number == private_target:
        return "private_target"
    if social_number is not None and number == social_number:
        return "social"
    if clue_matches_number(private_clue, number):
        return "other_clue_compatible"
    return "incompatible"


def per_token_nll(logprob: float, token_count: int) -> float:
    if token_count <= 0:
        return float("nan")
    return -float(logprob) / float(token_count)


def perplexity_from_logprob(logprob: float, token_count: int) -> float:
    nll = per_token_nll(logprob, token_count)
    if math.isnan(nll):
        return float("nan")
    return float(math.exp(min(nll, 60.0)))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def write_qualitative_examples(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    examples_per_alpha: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    selected: list[dict[str, Any]] = []
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("layer"), row.get("calibrated_alpha"))
        groups.setdefault(key, []).append(row)
    for key in sorted(groups, key=lambda item: (str(item[0]), int(item[1]), float(item[2]))):
        selected.extend(groups[key][: max(examples_per_alpha, 0)])
    lines = ["# Qualitative Steering Examples", ""]
    for row in selected:
        lines.append(
            "## {dataset} layer={layer} alpha={alpha} case={case}".format(
                dataset=row.get("dataset"),
                layer=row.get("layer"),
                alpha=row.get("calibrated_alpha"),
                case=row.get("case_id"),
            )
        )
        lines.append("")
        lines.append(f"- variant: `{row.get('variant')}`")
        lines.append(f"- private clue: {row.get('private_clue')}")
        lines.append(f"- private target: {row.get('private_target')}")
        lines.append(f"- social number: {row.get('social_number')}")
        lines.append(f"- parsed number: {row.get('parsed_number')}")
        lines.append(f"- valid JSON/schema: {row.get('valid')}")
        lines.append(f"- strict JSON only: {row.get('strict_json')}")
        lines.append(f"- satisfies private clue: {row.get('satisfies_private_clue')}")
        if row.get("parsed_reason"):
            lines.append(f"- parsed reason: {row.get('parsed_reason')}")
        lines.append("")
        lines.append("Prompt:")
        lines.append("")
        lines.append("```text")
        lines.append(str(row.get("prompt", "")))
        lines.append("```")
        lines.append("")
        lines.append("Raw output:")
        lines.append("")
        lines.append("```text")
        lines.append(str(row.get("raw_output", "")))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_vectors(path: Path, directions: dict[int, np.ndarray]) -> None:
    payload = {f"layer_{layer}": vector for layer, vector in directions.items()}
    np.savez_compressed(path, **payload)


def save_calibrated_vectors(path: Path, directions: dict[int, np.ndarray], sign_rows: list[dict[str, Any]]) -> None:
    sign_by_layer = {int(row["layer"]): int(row["empirical_social_sign"]) for row in sign_rows}
    payload = {
        f"layer_{layer}": vector * sign_by_layer.get(layer, 1)
        for layer, vector in directions.items()
    }
    np.savez_compressed(path, **payload)


def write_plots(
    out_dir: Path,
    direction_summary: list[dict[str, Any]],
    steering_rows: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    generation_summary_rows: list[dict[str, Any]],
    behavioral_effect_rows: list[dict[str, Any]],
    ood_steering_rows: list[dict[str, Any]],
    ood_generation_summary_rows: list[dict[str, Any]],
    ood_behavioral_effect_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    if direction_summary:
        rows = preferred_summary_rows(direction_summary)
        rows = sorted(rows, key=lambda row: int(row["layer"]))
        layers = [int(row["layer"]) for row in rows]
        corr = [float(row["projection_logprob_margin_pearson"]) for row in rows]
        gap = [float(row["social_projection_gap"]) for row in rows]
        fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
        ax1.plot(layers, corr, marker="o", color="#2E6FBB", label="projection/logprob corr")
        ax1.axhline(0.0, color="#A6ACB8", linewidth=0.8)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Pearson r")
        ax2 = ax1.twinx()
        ax2.plot(layers, gap, marker="s", color="#E07A3F", label="social-private projection gap")
        ax2.set_ylabel("Projection gap")
        eval_split = str(rows[0].get("eval_split", "all")) if rows else "all"
        fit_split = str(rows[0].get("fit_split", "all")) if rows else "all"
        fig.suptitle(f"Social-vs-private direction by layer ({fit_split} fit, {eval_split} eval)", fontweight="bold")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "steering_direction_layer_summary.svg")
        plt.close(fig)

    if steering_rows:
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for row in steering_rows:
            by_layer.setdefault(int(row["layer"]), []).append(row)
        keep_layers = sorted(by_layer)
        if len(keep_layers) > 6:
            keep_layers = keep_layers[:: max(len(keep_layers) // 6, 1)]
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        for layer in keep_layers:
            rows = sorted(by_layer[layer], key=lambda row: float(row.get("calibrated_alpha", row["alpha"])))
            ax.plot(
                [float(row.get("calibrated_alpha", row["alpha"])) for row in rows],
                [float(row["mean_social_minus_private_number_logprob_margin"]) for row in rows],
                marker="o",
                linewidth=1.4,
                label=f"layer {layer}",
            )
        ax.axhline(0.0, color="#A6ACB8", linewidth=0.8)
        ax.set_xlabel("Empirically calibrated alpha (positive = more social)")
        ax.set_ylabel("Mean social - private number logprob margin")
        ax.set_title("Activation steering sanity check", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "steering_alpha_sweep.svg")
        plt.close(fig)
        write_alpha_heatmap(
            out_dir / "plots" / "layer_alpha_heatmap.svg",
            steering_rows,
            title="Layer x alpha steering effect",
        )
    if projection_rows:
        write_projection_distribution_plot(out_dir / "plots" / "projection_distributions.svg", projection_rows)
    if scaling_rows:
        write_data_scaling_plot(out_dir / "plots" / "data_scaling_curve.svg", scaling_rows)
    if stability_rows:
        write_vector_stability_plot(out_dir / "plots" / "vector_stability.svg", stability_rows)
    if generation_summary_rows:
        write_choice_composition_plot(
            out_dir / "plots" / "generation_choice_composition.svg",
            generation_summary_rows,
            behavioral_effect_rows,
            title="Generation-time steering choice composition",
        )
        write_side_effect_plot(
            out_dir / "plots" / "generation_side_effects.svg",
            generation_summary_rows,
            title="Generation-time steering side effects",
        )
    if ood_steering_rows:
        write_alpha_heatmap(
            out_dir / "plots" / "ood_generalization_layer_alpha_heatmap.svg",
            ood_steering_rows,
            title="OOD actual social prompts: layer x alpha",
        )
    if ood_generation_summary_rows:
        write_choice_composition_plot(
            out_dir / "plots" / "ood_social_choice_composition.svg",
            ood_generation_summary_rows,
            ood_behavioral_effect_rows,
            title="OOD actual social choice composition",
        )
        write_side_effect_plot(
            out_dir / "plots" / "ood_social_generation_side_effects.svg",
            ood_generation_summary_rows,
            title="OOD actual social generation side effects",
        )


def write_projection_distribution_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    best_layer_rows = select_best_projection_layer_rows(rows)
    if not best_layer_rows:
        path.write_text("")
        return
    splits = [split for split in ("train", "test", "ood_actual_social", "all") if any(row.get("split") == split for row in best_layer_rows)]
    if not splits:
        splits = sorted({str(row.get("split", "all")) for row in best_layer_rows})[:3] or ["all"]
    fig, axes = plt.subplots(len(splits), 2, figsize=(10.0, max(3.2, 3.0 * len(splits))), squeeze=False)
    colors = {"target_memory": "#2E6FBB", "social_memory": "#E07A3F", "ood_actual_social": "#16A34A"}
    labels = {"target_memory": "private/target memory", "social_memory": "social memory", "ood_actual_social": "actual social prompt"}
    layer = best_layer_rows[0].get("layer")
    for row_index, split in enumerate(splits):
        split_rows = [row for row in best_layer_rows if row.get("split") == split or split == "all"]
        ax_hist = axes[row_index][0]
        ax_scatter = axes[row_index][1]
        for variant in ("target_memory", "social_memory", "ood_actual_social"):
            variant_rows = [row for row in split_rows if row.get("variant") == variant]
            values = [float(row["projection"]) for row in variant_rows if row.get("projection") not in (None, "")]
            if values:
                ax_hist.hist(values, bins=min(18, max(5, len(values) // 2)), alpha=0.55, color=colors[variant], label=labels[variant])
                ax_scatter.scatter(
                    values,
                    [float(row["social_minus_private_number_logprob_margin"]) for row in variant_rows],
                    s=20,
                    alpha=0.75,
                    color=colors[variant],
                    label=labels[variant],
                )
        ax_hist.axvline(0.0, color="#A6ACB8", linewidth=0.8)
        ax_hist.set_title(f"{split}: projection histogram")
        ax_hist.set_xlabel("Projection on social-minus-private direction")
        ax_hist.set_ylabel("Count")
        ax_scatter.axhline(0.0, color="#A6ACB8", linewidth=0.8)
        ax_scatter.axvline(0.0, color="#A6ACB8", linewidth=0.8)
        ax_scatter.set_title(f"{split}: projection vs logprob margin")
        ax_scatter.set_xlabel("Projection")
        ax_scatter.set_ylabel("Social - private number logprob")
        ax_hist.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Projection distributions at layer {layer}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def select_best_projection_layer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("projection") in (None, ""):
            continue
        by_layer.setdefault(int(row["layer"]), []).append(row)
    if not by_layer:
        return []
    def score(layer: int) -> float:
        value = pearson(
            [float(row["projection"]) for row in by_layer[layer]],
            [float(row["social_minus_private_number_logprob_margin"]) for row in by_layer[layer]],
        )
        return 0.0 if math.isnan(value) else abs(value)

    best_layer = max(by_layer, key=score)
    return by_layer[best_layer]


def write_alpha_heatmap(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [
        row
        for row in rows
        if row.get("calibrated_alpha") not in (None, "")
        and row.get("mean_social_minus_private_number_logprob_margin") not in (None, "")
    ]
    if not ok_rows:
        path.write_text("")
        return
    layers = sorted({int(row["layer"]) for row in ok_rows})
    alphas = sorted({float(row["calibrated_alpha"]) for row in ok_rows})
    matrix = np.full((len(layers), len(alphas)), np.nan)
    for row in ok_rows:
        layer_index = layers.index(int(row["layer"]))
        alpha_index = alphas.index(float(row["calibrated_alpha"]))
        matrix[layer_index, alpha_index] = float(row["mean_social_minus_private_number_logprob_margin"])
    fig, ax = plt.subplots(figsize=(9.0, max(3.6, 0.35 * len(layers))))
    vmax = np.nanmax(np.abs(matrix)) if np.any(~np.isnan(matrix)) else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{alpha:g}" for alpha in alphas], rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xlabel("Calibrated alpha (positive = more social)")
    ax.set_ylabel("Layer")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Mean social - private number logprob")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_data_scaling_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    eval_split = "test" if any(row.get("eval_split") == "test" for row in ok_rows) else "all"
    ok_rows = [row for row in ok_rows if row.get("eval_split") == eval_split]
    if not ok_rows:
        path.write_text("")
        return
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    layer_scores = sorted(
        by_layer,
        key=lambda layer: max(abs(float(row["projection_logprob_margin_pearson"])) for row in by_layer[layer]),
        reverse=True,
    )[:5]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for layer in layer_scores:
        layer_rows = sorted(by_layer[layer], key=lambda row: int(row["requested_case_count"]))
        ax.plot(
            [int(row["requested_case_count"]) for row in layer_rows],
            [float(row["projection_logprob_margin_pearson"]) for row in layer_rows],
            marker="o",
            linewidth=1.5,
            label=f"layer {layer}",
        )
    ax.axhline(0.0, color="#A6ACB8", linewidth=0.8)
    ax.axvline(80, color="#64748B", linestyle=":", linewidth=1.0, label="~80-case CAE saturation check")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Contrast cases used to fit vector")
    ax.set_ylabel(f"Projection/logprob Pearson r ({eval_split})")
    ax.set_title("Data-scaling curve", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_vector_stability_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ok_rows = [row for row in rows if row.get("mean_pairwise_cosine") not in (None, "")]
    if not ok_rows:
        path.write_text("")
        return
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in ok_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    def stability_score(layer: int) -> float:
        values = [
            float(row["mean_pairwise_cosine"])
            for row in by_layer[layer]
            if not math.isnan(float(row["mean_pairwise_cosine"]))
        ]
        return max(values) if values else -1.0

    layer_scores = sorted(by_layer, key=stability_score, reverse=True)[:5]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for layer in layer_scores:
        layer_rows = sorted(by_layer[layer], key=lambda row: int(row["requested_case_count"]))
        ax.plot(
            [int(row["requested_case_count"]) for row in layer_rows],
            [float(row["mean_pairwise_cosine"]) for row in layer_rows],
            marker="o",
            linewidth=1.5,
            label=f"layer {layer}",
        )
    ax.axhline(0.0, color="#A6ACB8", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Contrast cases per subset")
    ax.set_ylabel("Mean pairwise cosine")
    ax.set_title("Vector stability across case subsets", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_choice_composition_plot(
    path: Path,
    rows: list[dict[str, Any]],
    behavioral_effect_rows: list[dict[str, Any]],
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    layer = choose_behavior_layer(rows, behavioral_effect_rows)
    layer_rows = sorted(
        [row for row in rows if int(row["layer"]) == layer],
        key=lambda row: float(row["calibrated_alpha"]),
    )
    if not layer_rows:
        path.write_text("")
        return
    alphas = [float(row["calibrated_alpha"]) for row in layer_rows]
    series = [
        ("social_choice_rate", "social number", "#E07A3F", "o"),
        ("private_target_choice_rate", "private target", "#2E6FBB", "s"),
        ("other_clue_compatible_rate", "other clue-compatible", "#16A34A", "^"),
        ("incompatible_rate", "incompatible", "#C2410C", "d"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for key, label, color, marker in series:
        ax.plot(
            alphas,
            [float(row.get(key, 0.0) or 0.0) for row in layer_rows],
            marker=marker,
            linewidth=1.7,
            label=label,
            color=color,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(0.0, color="#A6ACB8", linewidth=0.9)
    ax.set_xlabel("Calibrated alpha (positive = more social)")
    ax.set_ylabel("Choice rate among valid generations")
    ax.set_title(f"{title} (layer {layer})", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def choose_behavior_layer(rows: list[dict[str, Any]], behavioral_effect_rows: list[dict[str, Any]]) -> int:
    scored = [
        row
        for row in behavioral_effect_rows
        if row.get("best_social_choice_delta") not in (None, "")
    ]
    if scored:
        return int(
            max(
                scored,
                key=lambda row: (
                    float(row.get("best_social_choice_delta", 0.0) or 0.0),
                    -abs(float(row.get("best_format_damage_delta", 0.0) or 0.0)),
                ),
            )["layer"]
        )
    counts: dict[int, int] = {}
    for row in rows:
        counts[int(row["layer"])] = counts.get(int(row["layer"]), 0) + 1
    return max(counts, key=counts.get)


def write_side_effect_plot(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)
    layer = max(by_layer, key=lambda layer_id: len(by_layer[layer_id]))
    layer_rows = sorted(by_layer[layer], key=lambda row: float(row["calibrated_alpha"]))
    alphas = [float(row["calibrated_alpha"]) for row in layer_rows]
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.plot(alphas, [float(row["valid_rate"]) for row in layer_rows], marker="o", label="valid JSON/schema", color="#2E6FBB")
    ax1.plot(alphas, [float(row["format_damage_rate"]) for row in layer_rows], marker="s", label="format damage", color="#C2410C")
    ax1.plot(
        alphas,
        [
            float(row["satisfies_private_clue_rate"])
            if row.get("satisfies_private_clue_rate") not in (None, "") and not math.isnan(float(row["satisfies_private_clue_rate"]))
            else np.nan
            for row in layer_rows
        ],
        marker="^",
        label="satisfies private clue",
        color="#16A34A",
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Calibrated alpha (positive = more social)")
    ax1.set_ylabel("Rate")
    ax2 = ax1.twinx()
    ax2.plot(
        alphas,
        [
            float(row["mean_base_completion_perplexity"])
            if row.get("mean_base_completion_perplexity") not in (None, "") and not math.isnan(float(row["mean_base_completion_perplexity"]))
            else np.nan
            for row in layer_rows
        ],
        marker="d",
        label="base PPL of output",
        color="#7C3AED",
    )
    ax2.set_ylabel("Completion perplexity")
    ax1.set_title(f"{title} (layer {layer})", fontweight="bold")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_index(
    out_dir: Path,
    model: str,
    cases: list[SteeringCase],
    layers: list[int],
    memory_strengths: list[int],
    m_values: list[int],
    alphas: list[float],
    fit_split: str,
    direction_method: str,
    direction_quantile: float,
    subspace_rank: int,
) -> None:
    split_counts: dict[str, int] = {}
    for case in cases:
        split_counts[case.split] = split_counts.get(case.split, 0) + 1
    lines = [
        "# Number Game Local Steering Prep",
        "",
        f"- model: `{model}`",
        f"- case count: `{len(cases)}`",
        f"- split counts: `{split_counts}`",
        f"- fit split: `{fit_split}`",
        f"- direction method: `{direction_method}`",
        f"- direction quantile: `{direction_quantile}`",
        f"- subspace rank: `{subspace_rank}`",
        f"- layers: `{layers}`",
        f"- memory strengths: `{memory_strengths}`",
        f"- m values: `{m_values}`",
        f"- steering alphas: `{alphas}`",
        "",
        "Files:",
        "",
        "- `steering_prep_trials.csv`: prompt metadata, memory variant, and unsteered private/social number logprobs.",
        "- `steering_cases.csv`: generated private/social contrast cases and train/test split labels.",
        "- `steering_direction_summary.csv`: layerwise social-minus-private direction quality.",
        "- `projection_distributions.csv`: social/private projection rows for train/test histograms and scatter plots.",
        "- `data_scaling_curve.csv`: 8/16/32/64/128 contrast-case scaling diagnostics when enough cases are collected.",
        "- `vector_stability.csv`: pairwise cosine stability across case subsets.",
        "- `steering_alpha_sweep.csv`: full-sequence number-logprob response when adding/subtracting each layer direction.",
        "- `steering_sign_summary.csv`: empirical sign calibration for each layer's raw contrast vector.",
        "- `generation_steering_outputs.jsonl`: actual generated JSON/text under steering, when `--run-generation-steering` is used.",
        "- `generation_side_effect_summary.csv`: validity, clue-satisfaction, format-damage, and perplexity side effects by alpha.",
        "- `behavioral_steering_effect_summary.csv`: baseline-vs-steered social-choice deltas, private-clue deltas, and format/validity deltas.",
        "- `ood_social_*`: synthetic-train / actual-social-prompt generalization diagnostics when `--ood-social-dir` is supplied.",
        "- `steering_vectors_social_minus_private.npz`: compressed average direction vector by layer.",
        "- `steering_vectors_empirical_social.npz`: sign-calibrated vectors where positive alpha means more social when calibration exists.",
        "- `plots/steering_direction_layer_summary.svg`: layer localization view.",
        "- `plots/steering_alpha_sweep.svg`: first activation-steering sanity check.",
        "- `plots/projection_distributions.svg`: train/test projection distributions and projection-vs-logprob scatter.",
        "- `plots/layer_alpha_heatmap.svg`: layer x alpha effect-size heatmap.",
        "- `plots/data_scaling_curve.svg`: contrast-case sample-size curve.",
        "- `plots/vector_stability.svg`: vector cosine stability across random case subsets.",
        "- `plots/generation_choice_composition.svg`: social/private/other/incompatible choice rates versus alpha.",
        "- `plots/generation_side_effects.svg`: generation validity/perplexity/format damage versus alpha.",
        "",
        "Interpretation: the raw contrast vector is diagnostic. Use `calibrated_alpha` and `steering_vectors_empirical_social.npz` for causal steering, because raw signs can flip under intervention.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def best_layer(direction_summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not direction_summary:
        return None
    rows = preferred_summary_rows(direction_summary)
    rows = [row for row in rows if not math.isnan(float(row["projection_logprob_margin_pearson"]))]
    if not rows:
        rows = preferred_summary_rows(direction_summary) or direction_summary
    return max(
        rows,
        key=lambda row: (
            abs(float(row.get("projection_logprob_margin_pearson", 0.0))),
            float(row.get("social_projection_gap", 0.0)),
        ),
    )


def preferred_summary_rows(direction_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [
        row
        for row in direction_summary
        if row.get("eval_split") == "test" and not math.isnan(float(row["projection_logprob_margin_pearson"]))
    ]
    if test_rows:
        return test_rows
    all_rows = [row for row in direction_summary if row.get("eval_split", "all") == "all"]
    return all_rows or direction_summary


if __name__ == "__main__":
    main()

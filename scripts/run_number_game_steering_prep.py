from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nnd.number_game import prompts
from nnd.number_game.backend import build_backend
from nnd.number_game.config import apply_overrides, load_number_game_config, save_resolved_config
from nnd.number_game.domain import DEFAULT_CLUES, candidate_numbers, matching_clues


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
        description="Local steering-vector prep for the number game: contrast hidden states and steer answer logits."
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
    parser.add_argument("--skip-alpha-sweep", action="store_true")
    parser.add_argument("--max-alpha-trials", default=None, type=int)
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
    active_cases = build_active_cases(
        numbers=candidate_numbers(config.min_number, config.max_number),
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
    directions, direction_summary = compute_direction_summary(trials, vectors, layers, fit_split=args.fit_split)
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

    write_csv(out_dir / "steering_prep_trials.csv", trials)
    write_csv(out_dir / "steering_direction_summary.csv", direction_summary)
    write_csv(out_dir / "steering_alpha_sweep.csv", steering_rows)
    save_vectors(out_dir / "steering_vectors_social_minus_private.npz", directions)
    write_plots(out_dir, direction_summary, steering_rows)
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
    )

    print(f"Steering prep complete. Output saved to {out_dir}")
    best = best_layer(direction_summary)
    if best:
        print(
            "Best local layer: {layer} "
            "(projection/logit corr={projection_logit_margin_pearson:.3f}, "
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
                            "prompt": prompt_text,
                        }
                    )
    return trials, vectors


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
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    by_pair_variant = {(row["pair_id"], row["variant"]): row for row in trials}
    pair_ids = sorted({row["pair_id"] for row in trials})
    if fit_split != "all":
        pair_ids_for_fit = sorted(
            {
                row["pair_id"]
                for row in trials
                if row.get("split") == fit_split and row["variant"] in ("target_memory", "social_memory")
            }
        )
    else:
        pair_ids_for_fit = pair_ids
    directions: dict[int, np.ndarray] = {}
    summary: list[dict[str, Any]] = []

    for layer in layers:
        diffs: list[np.ndarray] = []
        pair_norms: list[float] = []
        for pair_id in pair_ids_for_fit:
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
            continue
        direction = np.mean(np.stack(diffs, axis=0), axis=0)
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
                    float(row["social_minus_private_logit_margin"]),
                    str(row["variant"]),
                )
                for row in eval_rows
                if f"projection_layer_{layer}" in row
            ]
            social_proj = [p for p, _, variant in projection_rows if variant == "social_memory"]
            target_proj = [p for p, _, variant in projection_rows if variant == "target_memory"]
            summary.append(
                {
                    "direction": "social_minus_private_memory",
                    "layer": layer,
                    "fit_split": fit_split,
                    "eval_split": eval_split,
                    "n_fit_pairs": len(diffs),
                    "n_eval_rows": len(projection_rows),
                    "mean_pair_norm": mean(pair_norms),
                    "direction_norm": float(np.linalg.norm(direction)),
                    "social_projection_mean": mean(social_proj),
                    "target_projection_mean": mean(target_proj),
                    "social_projection_gap": mean(social_proj) - mean(target_proj),
                    "projection_logit_margin_pearson": pearson(
                        [p for p, _, _ in projection_rows],
                        [m for _, m, _ in projection_rows],
                    ),
                }
            )
    return directions, summary


def eval_splits(trials: list[dict[str, Any]]) -> list[str]:
    splits = sorted({str(row.get("split", "")) for row in trials if row.get("split")})
    return ["all", *splits]


def run_alpha_sweep(
    *,
    backend: Any,
    trials: list[dict[str, Any]],
    directions: dict[int, np.ndarray],
    direction_summary: list[dict[str, Any]],
    layers: list[int],
    alphas: list[float],
    max_alpha_trials: int | None,
) -> list[dict[str, Any]]:
    scale_by_layer: dict[int, float] = {}
    for row in direction_summary:
        scale_by_layer.setdefault(int(row["layer"]), float(row["mean_pair_norm"]))
    rows: list[dict[str, Any]] = []
    eval_trials = [row for row in trials if row["variant"] in ("target_memory", "social_memory")]
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
            for row in eval_trials:
                prompt_text = str(row["prompt"])
                private_logit, social_logit = number_logits_after_prefix(
                    backend,
                    prompt_text,
                    int(row["private_number_first_token_id"]),
                    int(row["social_number_first_token_id"]),
                    layer=layer,
                    steer_vector=unit * scale * alpha,
                )
                margins.append(social_logit - private_logit)
            rows.append(
                {
                    "direction": "social_minus_private_memory",
                    "layer": layer,
                    "alpha": alpha,
                    "n": len(margins),
                    "mean_social_minus_private_number_logit_margin": mean(margins),
                    "mean_social_number_probability": mean_logistic(margins),
                }
            )
    return rows


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


def write_plots(out_dir: Path, direction_summary: list[dict[str, Any]], steering_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if direction_summary:
        rows = preferred_summary_rows(direction_summary)
        rows = sorted(rows, key=lambda row: int(row["layer"]))
        layers = [int(row["layer"]) for row in rows]
        corr = [float(row["projection_logit_margin_pearson"]) for row in rows]
        gap = [float(row["social_projection_gap"]) for row in rows]
        fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
        ax1.plot(layers, corr, marker="o", color="#2E6FBB", label="projection/logit corr")
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
            rows = sorted(by_layer[layer], key=lambda row: float(row["alpha"]))
            ax.plot(
                [float(row["alpha"]) for row in rows],
                [float(row["mean_social_minus_private_number_logit_margin"]) for row in rows],
                marker="o",
                linewidth=1.4,
                label=f"layer {layer}",
            )
        ax.axhline(0.0, color="#A6ACB8", linewidth=0.8)
        ax.set_xlabel("Steering alpha along social-minus-private direction")
        ax.set_ylabel("Mean social - private number logit margin")
        ax.set_title("Activation steering sanity check", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "steering_alpha_sweep.svg")
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
        f"- layers: `{layers}`",
        f"- memory strengths: `{memory_strengths}`",
        f"- m values: `{m_values}`",
        f"- steering alphas: `{alphas}`",
        "",
        "Files:",
        "",
        "- `steering_prep_trials.csv`: prompt metadata, memory variant, and unsteered private/social number logits.",
        "- `steering_cases.csv`: generated private/social contrast cases and train/test split labels.",
        "- `steering_direction_summary.csv`: layerwise social-minus-private direction quality.",
        "- `steering_alpha_sweep.csv`: answer-logit response when adding/subtracting each layer direction.",
        "- `steering_vectors_social_minus_private.npz`: compressed average direction vector by layer.",
        "- `plots/steering_direction_layer_summary.svg`: layer localization view.",
        "- `plots/steering_alpha_sweep.svg`: first activation-steering sanity check.",
        "",
        "Interpretation: positive alpha adds the social-memory direction; negative alpha pushes toward private/target memory.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")


def best_layer(direction_summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not direction_summary:
        return None
    rows = preferred_summary_rows(direction_summary)
    rows = [row for row in rows if not math.isnan(float(row["projection_logit_margin_pearson"]))]
    if not rows:
        rows = preferred_summary_rows(direction_summary) or direction_summary
    return max(
        rows,
        key=lambda row: (
            abs(float(row.get("projection_logit_margin_pearson", 0.0))),
            float(row.get("social_projection_gap", 0.0)),
        ),
    )


def preferred_summary_rows(direction_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [
        row
        for row in direction_summary
        if row.get("eval_split") == "test" and not math.isnan(float(row["projection_logit_margin_pearson"]))
    ]
    if test_rows:
        return test_rows
    all_rows = [row for row in direction_summary if row.get("eval_split", "all") == "all"]
    return all_rows or direction_summary


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

from nnd.number_game import prompts
from nnd.number_game.backend import build_backend
from nnd.number_game.config import apply_overrides, load_number_game_config, save_resolved_config
from nnd.number_game.domain import candidate_numbers, clue_matches_number, filter_candidates
from nnd.number_game.parsing import ParseError


@dataclass(frozen=True)
class EvidenceCondition:
    condition_id: str
    relation: str
    private_clue: str
    private_target: int
    memory_number: int
    reason_clues: tuple[str, ...]
    reason_lines: tuple[str, ...]


EVIDENCE_CONDITIONS: tuple[EvidenceCondition, ...] = (
    EvidenceCondition(
        condition_id="contradictory_digit_sum_7_vs_12",
        relation="contradictory",
        private_clue="the digits sum to 7",
        private_target=7,
        memory_number=12,
        reason_clues=("the number is even",),
        reason_lines=("The number is even.", "My clue points to an even number."),
    ),
    EvidenceCondition(
        condition_id="compatible_weak_under20_vs_14",
        relation="compatible_weak",
        private_clue="the number is under 20",
        private_target=7,
        memory_number=14,
        reason_clues=("the number is under 20",),
        reason_lines=("The number is under 20.", "My clue also says the number is under 20."),
    ),
    EvidenceCondition(
        condition_id="complementary_prime_to_7",
        relation="complementary",
        private_clue="the number is prime",
        private_target=7,
        memory_number=7,
        reason_clues=("the digits sum to 7", "the number is under 20", "the number has one digit"),
        reason_lines=(
            "The digits sum to 7.",
            "The number is under 20.",
            "The number has one digit.",
        ),
    ),
    EvidenceCondition(
        condition_id="decisive_odd_to_25",
        relation="decisive",
        private_clue="the number is odd",
        private_target=7,
        memory_number=25,
        reason_clues=("the number is a multiple of 5", "the digits sum to 7", "the number is over 10"),
        reason_lines=(
            "The number is a multiple of 5.",
            "The digits sum to 7.",
            "The number is over 10.",
        ),
    ),
)


OWNERSHIP_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "mine_prime_yours_even",
        "private_clue": "the number is prime",
        "memory_lines": ["12 | The number is even."],
        "private_candidate": 7,
        "social_number": 12,
    },
    {
        "case_id": "mine_even_yours_prime",
        "private_clue": "the number is even",
        "memory_lines": ["7 | The number is prime."],
        "private_candidate": 12,
        "social_number": 7,
    },
    {
        "case_id": "mine_digit_sum_yours_under20",
        "private_clue": "the digits sum to 7",
        "memory_lines": ["14 | The number is under 20."],
        "private_candidate": 7,
        "social_number": 14,
    },
    {
        "case_id": "mine_under20_yours_digit_sum",
        "private_clue": "the number is under 20",
        "memory_lines": ["7 | The digits sum to 7."],
        "private_candidate": 14,
        "social_number": 7,
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact local pre-RunPod number-game probes.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start-seed", default=0, type=int)
    parser.add_argument("--num-seeds", default=5, type=int)
    parser.add_argument("--max-strength", default=4, type=int)
    parser.add_argument("--hidden-layer", action="append", type=int, default=[0, 4, 8, 12, 16, 20, 24, 28])
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_number_game_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)
    config = config.model_copy(
        update={
            "backend": "transformers",
            "interaction_m": 3,
            "capture_hidden_states": True,
            "hidden_state_layers": args.hidden_layer,
            "max_tokens": max(config.max_tokens, 80),
        }
    )
    save_resolved_config(config, out_dir)
    numbers = candidate_numbers(config.min_number, config.max_number)
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    backend = build_backend(
        backend_name=config.backend,
        model=config.model,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        debug_dir=out_dir / "debug",
        seed=args.start_seed,
        social_susceptibility=config.social_susceptibility,
        prompt_social_susceptibility=config.prompt_social_susceptibility,
        prompt_number_range=config.prompt_number_range,
        capture_hidden_states=config.capture_hidden_states,
        hidden_state_layers=config.hidden_state_layers,
        use_response_format=config.use_response_format,
        api_base_url=config.api_base_url,
        api_key=config.api_key,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        enable_thinking=config.enable_thinking,
    )

    evidence_rows, evidence_prompt_rows, hidden_rows, logprob_rows = run_evidence_sweep(
        backend=backend,
        numbers=numbers,
        seeds=seeds,
        max_strength=args.max_strength,
    )
    ownership_rows, ownership_prompt_rows, ownership_hidden_rows = run_ownership_swap(
        backend=backend,
        numbers=numbers,
        seeds=seeds,
    )
    hidden_rows.extend(ownership_hidden_rows)

    evidence_summary = summarize_rows(evidence_rows, ["probe_family", "m", "relation", "memory_strength"])
    ownership_summary = summarize_rows(ownership_rows, ["probe_family", "case_id"])

    write_csv(out_dir / "evidence_sweep_trials.csv", evidence_rows)
    write_csv(out_dir / "evidence_sweep_summary.csv", evidence_summary)
    write_csv(out_dir / "ownership_swap_trials.csv", ownership_rows)
    write_csv(out_dir / "ownership_swap_summary.csv", ownership_summary)
    write_csv(out_dir / "number_logprob_probe.csv", logprob_rows)
    write_jsonl(out_dir / "probe_prompts.jsonl", evidence_prompt_rows + ownership_prompt_rows)
    write_jsonl(out_dir / "probe_hidden_states.jsonl", hidden_rows)
    write_response_grid(evidence_summary, out_dir / "plots" / "evidence_strength_response_grid.svg")
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "model": config.model,
                "seeds": seeds,
                "trial_count": len(evidence_rows) + len(ownership_rows),
                "evidence_summary": evidence_summary,
                "ownership_summary": ownership_summary,
                "backend_usage": backend.usage_summary() if hasattr(backend, "usage_summary") else {},
                "notes": {
                    "memory_format": "bare entries: number or number | reason",
                    "bayes_candidate_range": [config.min_number, config.max_number],
                },
            },
            handle,
            indent=2,
        )


def run_evidence_sweep(
    *,
    backend: Any,
    numbers: list[int],
    seeds: list[int],
    max_strength: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    logprob_rows: list[dict[str, Any]] = []
    prompt_index = 0
    logprob_probe_budget = 12

    for seed in seeds:
        for condition in EVIDENCE_CONDITIONS:
            for m in (1, 3):
                for strength in range(max_strength + 1):
                    memory_lines = build_memory_lines(condition, m=m, strength=strength, seed=seed)
                    call_id = f"evidence_seed{seed:04d}_{condition.condition_id}_m{m}_s{strength}"
                    prompt_text = prompts.interaction_text(
                        numbers=numbers,
                        private_clue=condition.private_clue,
                        memory_lines=memory_lines,
                        m=m,
                        social_susceptibility=0.5,
                        prompt_social_susceptibility=False,
                        prompt_number_range=getattr(backend, "prompt_number_range", False),
                    )
                    prompt_rows.append(
                        {
                            "call_id": call_id,
                            "probe_family": "evidence_sweep",
                            "seed": seed,
                            "m": m,
                            "condition_id": condition.condition_id,
                            "relation": condition.relation,
                            "memory_strength": strength,
                            "private_clue": condition.private_clue,
                            "private_target": condition.private_target,
                            "memory_number": condition.memory_number,
                            "memory_lines": memory_lines,
                            "prompt": prompt_text,
                        }
                    )
                    set_backend_seed(backend, seed * 1000 + prompt_index)
                    before_hidden = len(getattr(backend, "hidden_state_rows", []))
                    try:
                        message = backend.message(
                            numbers=numbers,
                            private_clue=condition.private_clue,
                            memory_lines=memory_lines,
                            m=m,
                        )
                        number = message.number
                        reason = message.reason
                        valid = True
                        error = None
                    except ParseError as exc:
                        number = None
                        reason = None
                        valid = False
                        error = str(exc)
                    after_hidden = getattr(backend, "hidden_state_rows", [])[before_hidden:]
                    for hidden_index, hidden_row in enumerate(after_hidden):
                        hidden_rows.append(
                            {
                                "call_id": call_id,
                                "probe_family": "evidence_sweep",
                                "seed": seed,
                                "m": m,
                                "condition_id": condition.condition_id,
                                "relation": condition.relation,
                                "memory_strength": strength,
                                "hidden_index": hidden_index,
                                **hidden_row,
                            }
                        )
                    bayes_clues = bayes_clues_for(condition, m=m, strength=strength)
                    bayes_candidates = filter_candidates(numbers, [condition.private_clue, *bayes_clues])
                    row = {
                        "call_id": call_id,
                        "probe_family": "evidence_sweep",
                        "seed": seed,
                        "m": m,
                        "condition_id": condition.condition_id,
                        "relation": condition.relation,
                        "memory_strength": strength,
                        "memory_lines": memory_lines,
                        "private_clue": condition.private_clue,
                        "private_target": condition.private_target,
                        "memory_number": condition.memory_number,
                        "valid": valid,
                        "number": number,
                        "reason": reason,
                        "error": error,
                        "chose_private_target": number == condition.private_target if valid else None,
                        "chose_memory_number": number == condition.memory_number if valid else None,
                        "satisfies_private_clue": (
                            clue_matches_number(condition.private_clue, int(number)) if valid and number is not None else None
                        ),
                        "bayes_clues": bayes_clues,
                        "bayes_candidate_count": len(bayes_candidates),
                        "bayes_candidates": bayes_candidates,
                        "chosen_in_bayes_candidates": number in bayes_candidates if valid else None,
                        "memory_number_in_bayes_candidates": condition.memory_number in bayes_candidates,
                        "response_category": response_category(
                            number=number,
                            valid=valid,
                            condition=condition,
                            bayes_candidates=bayes_candidates,
                        ),
                    }
                    rows.append(row)
                    if prompt_index < logprob_probe_budget and m == 1:
                        logprob_rows.extend(
                            number_logprob_probe(
                                backend=backend,
                                numbers=numbers,
                                prompt_text=prompt_text,
                                call_id=call_id,
                                condition=condition,
                                bayes_candidates=bayes_candidates,
                            )
                        )
                    prompt_index += 1
    return rows, prompt_rows, hidden_rows, logprob_rows


def run_ownership_swap(
    *,
    backend: Any,
    numbers: list[int],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    prompt_index = 0
    for seed in seeds:
        for case in OWNERSHIP_CASES:
            call_id = f"ownership_seed{seed:04d}_{case['case_id']}"
            prompt_text = prompts.interaction_text(
                numbers=numbers,
                private_clue=case["private_clue"],
                memory_lines=case["memory_lines"],
                m=3,
                social_susceptibility=0.5,
                prompt_social_susceptibility=False,
                prompt_number_range=getattr(backend, "prompt_number_range", False),
            )
            prompt_rows.append(
                {
                    "call_id": call_id,
                    "probe_family": "ownership_swap",
                    "seed": seed,
                    "case_id": case["case_id"],
                    "private_clue": case["private_clue"],
                    "memory_lines": case["memory_lines"],
                    "prompt": prompt_text,
                }
            )
            set_backend_seed(backend, seed * 2000 + prompt_index)
            before_hidden = len(getattr(backend, "hidden_state_rows", []))
            try:
                message = backend.message(
                    numbers=numbers,
                    private_clue=case["private_clue"],
                    memory_lines=case["memory_lines"],
                    m=3,
                )
                number = message.number
                reason = message.reason
                valid = True
                error = None
            except ParseError as exc:
                number = None
                reason = None
                valid = False
                error = str(exc)
            for hidden_index, hidden_row in enumerate(getattr(backend, "hidden_state_rows", [])[before_hidden:]):
                hidden_rows.append(
                    {
                        "call_id": call_id,
                        "probe_family": "ownership_swap",
                        "seed": seed,
                        "case_id": case["case_id"],
                        "hidden_index": hidden_index,
                        **hidden_row,
                    }
                )
            rows.append(
                {
                    "call_id": call_id,
                    "probe_family": "ownership_swap",
                    "seed": seed,
                    "case_id": case["case_id"],
                    "private_clue": case["private_clue"],
                    "memory_lines": case["memory_lines"],
                    "private_candidate": case["private_candidate"],
                    "social_number": case["social_number"],
                    "valid": valid,
                    "number": number,
                    "reason": reason,
                    "error": error,
                    "chose_private_candidate": number == case["private_candidate"] if valid else None,
                    "chose_social_number": number == case["social_number"] if valid else None,
                    "satisfies_private_clue": (
                        clue_matches_number(case["private_clue"], int(number)) if valid and number is not None else None
                    ),
                }
            )
            prompt_index += 1
    return rows, prompt_rows, hidden_rows


def build_memory_lines(condition: EvidenceCondition, *, m: int, strength: int, seed: int) -> list[str]:
    if strength <= 0:
        return []
    if m == 1:
        lines = [str(condition.memory_number) for _ in range(strength)]
    else:
        lines = [
            f"{condition.memory_number} | {condition.reason_lines[index % len(condition.reason_lines)]}"
            for index in range(strength)
        ]
    rng = random.Random(seed + strength)
    rng.shuffle(lines)
    return lines


def bayes_clues_for(condition: EvidenceCondition, *, m: int, strength: int) -> list[str]:
    if m != 3 or strength <= 0:
        return []
    return [condition.reason_clues[index % len(condition.reason_clues)] for index in range(strength)]


def response_category(
    *,
    number: int | None,
    valid: bool,
    condition: EvidenceCondition,
    bayes_candidates: list[int],
) -> str:
    if not valid or number is None:
        return "invalid"
    if number == condition.memory_number:
        return "memory_number"
    if number == condition.private_target:
        return "private_target"
    if number in bayes_candidates:
        return "bayes_compatible_other"
    if clue_matches_number(condition.private_clue, number):
        return "private_compatible_other"
    return "private_incompatible"


def number_logprob_probe(
    *,
    backend: Any,
    numbers: list[int],
    prompt_text: str,
    call_id: str,
    condition: EvidenceCondition,
    bayes_candidates: list[int],
) -> list[dict[str, Any]]:
    if not hasattr(backend, "model_obj") or not hasattr(backend, "tokenizer"):
        return []
    scores = []
    for number in numbers:
        completion = f'{{"number":{number}}}'
        score = sequence_logprob(backend, prompt_text, completion)
        scores.append((number, score))
    max_score = max(score for _, score in scores)
    denom = sum(math.exp(score - max_score) for _, score in scores)
    return [
        {
            "call_id": call_id,
            "condition_id": condition.condition_id,
            "relation": condition.relation,
            "candidate_number": number,
            "sequence_logprob": score,
            "normalized_probability": math.exp(score - max_score) / denom if denom > 0 else 0.0,
            "is_private_target": number == condition.private_target,
            "is_memory_number": number == condition.memory_number,
            "is_bayes_candidate": number in bayes_candidates,
            "satisfies_private_clue": clue_matches_number(condition.private_clue, number),
        }
        for number, score in scores
    ]


def sequence_logprob(backend: Any, prompt_text: str, completion: str) -> float:
    torch = backend.torch
    messages = prompts.openai_messages(prompt_text)
    prompt = backend._format_prompt(messages)
    prompt_ids = backend.tokenizer(prompt, return_tensors="pt")["input_ids"]
    full_ids = backend.tokenizer(prompt + completion, return_tensors="pt")["input_ids"]
    model_device = getattr(backend.model_obj, "device", None)
    if model_device is not None:
        prompt_ids = prompt_ids.to(model_device)
        full_ids = full_ids.to(model_device)
    prompt_len = int(prompt_ids.shape[-1])
    with torch.no_grad():
        logits = backend.model_obj(full_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits[0, prompt_len - 1 : -1], dim=-1)
    target_ids = full_ids[0, prompt_len:]
    if target_ids.numel() == 0:
        return float("-inf")
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(token_log_probs.sum().detach().cpu())


def set_backend_seed(backend: Any, seed: int) -> None:
    if hasattr(backend, "torch"):
        backend.torch.manual_seed(seed)
        if getattr(backend.torch, "cuda", None) and backend.torch.cuda.is_available():
            backend.torch.cuda.manual_seed_all(seed)


def summarize_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    summary: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        valid_rows = [row for row in group_rows if bool(row.get("valid", False))]
        out = {key: value for key, value in zip(keys, group_key, strict=True)}
        out.update(
            {
                "n": len(group_rows),
                "valid_rate": len(valid_rows) / max(len(group_rows), 1),
                "memory_number_rate": rate_category(valid_rows, "memory_number"),
                "private_target_rate": rate_category(valid_rows, "private_target"),
                "bayes_compatible_other_rate": rate_category(valid_rows, "bayes_compatible_other"),
                "private_compatible_other_rate": rate_category(valid_rows, "private_compatible_other"),
                "private_incompatible_rate": rate_category(valid_rows, "private_incompatible"),
                "chosen_in_bayes_candidates_rate": rate_bool(valid_rows, "chosen_in_bayes_candidates"),
                "satisfies_private_clue_rate": rate_bool(valid_rows, "satisfies_private_clue"),
                "chose_social_number_rate": rate_bool(valid_rows, "chose_social_number"),
                "chose_private_candidate_rate": rate_bool(valid_rows, "chose_private_candidate"),
            }
        )
        summary.append(out)
    return summary


def rate_category(rows: list[dict[str, Any]], category: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("response_category") == category) / float(len(rows))


def rate_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_response_grid(summary_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in summary_rows if row.get("probe_family") == "evidence_sweep"]
    width, height = 980, 620
    left, top = 88, 74
    panel_w, panel_h = 370, 205
    gap_x, gap_y = 72, 72
    colors = [
        ("memory_number_rate", "#f47c20", "Memory number"),
        ("private_target_rate", "#2f80ed", "Private target"),
        ("bayes_compatible_other_rate", "#2fb56e", "Other Bayes-compatible"),
        ("private_incompatible_rate", "#cfd8dc", "Private-incompatible"),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#0f172a}.muted{fill:#64748b;font-size:12px}.title{font-size:21px;font-weight:700}.panel{font-size:13px;font-weight:700}.axis{stroke:#334155;stroke-width:1}.grid{stroke:#e2e8f0}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="34" class="title">Number Game Evidence Sweep</text>',
        f'<text x="{left}" y="55" class="muted">Response composition by memory strength and evidence relation.</text>',
    ]

    def y_for(value: float, y0: float) -> float:
        return y0 + (1.0 - value) * panel_h

    relations = ["contradictory", "compatible_weak", "complementary", "decisive"]
    for idx, relation in enumerate(relations):
        x0 = left + (idx % 2) * (panel_w + gap_x)
        y0 = top + (idx // 2) * (panel_h + gap_y)
        rel_rows = sorted(
            [row for row in rows if row.get("relation") == relation and int(row.get("m", 0)) == 3],
            key=lambda row: int(row["memory_strength"]),
        )
        if not rel_rows:
            continue
        x_values = [
            x0 + i * panel_w / max(len(rel_rows) - 1, 1)
            for i in range(len(rel_rows))
        ]
        for tick in (0.0, 0.5, 1.0):
            y = y_for(tick, y0)
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" class="grid"/>')
            svg.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="muted">{tick:.1f}</text>')
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" class="axis"/>')
        svg.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" class="axis"/>')
        svg.append(f'<text x="{x0}" y="{y0 - 14}" class="panel">{relation}</text>')
        lower = [0.0 for _ in rel_rows]
        for metric, color, _label in colors:
            upper = [lower[i] + float(rel_rows[i].get(metric, 0.0) or 0.0) for i in range(len(rel_rows))]
            top_points = [f"{x_values[i]:.1f},{y_for(upper[i], y0):.1f}" for i in range(len(rel_rows))]
            bottom_points = [f"{x_values[i]:.1f},{y_for(lower[i], y0):.1f}" for i in range(len(rel_rows) - 1, -1, -1)]
            svg.append(f'<polygon points="{" ".join(top_points + bottom_points)}" fill="{color}" opacity="0.94"/>')
            lower = upper
        for i, row in enumerate(rel_rows):
            x = x_values[i]
            svg.append(f'<text x="{x:.1f}" y="{y0 + panel_h + 20}" text-anchor="middle" class="muted">{row["memory_strength"]}</text>')
    svg.append(f'<text x="{left + panel_w + gap_x / 2:.1f}" y="{height - 26}" text-anchor="middle" class="muted">Memory strength (m=3)</text>')
    legend_x, legend_y = 790, 90
    for i, (_metric, color, label) in enumerate(colors):
        y = legend_y + i * 26
        svg.append(f'<rect x="{legend_x}" y="{y - 12}" width="18" height="14" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 26}" y="{y}" class="muted">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()

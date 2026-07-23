from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from nnd.number_game.backend import build_backend
from nnd.number_game.config import NumberGameConfig, save_resolved_config
from nnd.number_game.domain import (
    DEFAULT_CLUES,
    candidate_numbers,
    clue_candidates,
    clue_information_bits,
    clue_information_phase,
    clue_matches_number,
)
from nnd.number_game.parsing import ParseError
from nnd.number_game import prompts


@dataclass(frozen=True)
class ConflictCase:
    case_id: str
    truth_number: int
    private_clue: str
    social_number: int


DEFAULT_CONFLICT_CASES: tuple[ConflictCase, ...] = (
    ConflictCase("digit_sum_7_vs_12", 7, "the digits sum to 7", 12),
    ConflictCase("prime_vs_12", 3, "the number is prime", 12),
    ConflictCase("odd_vs_8", 5, "the number is odd", 8),
    ConflictCase("multiple_5_vs_12", 5, "the number is a multiple of 5", 12),
    ConflictCase("square_vs_8", 9, "the number is a perfect square", 8),
    ConflictCase("one_digit_vs_15", 7, "the number has one digit", 15),
)


def run_social_conflict_battery(
    config: NumberGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
    m_values: list[int] | None = None,
    memory_strengths: list[int] | None = None,
    hidden_state_layers: list[int] | None = None,
    ratio_total: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, out_dir)
    numbers = candidate_numbers(config.min_number, config.max_number)
    active_m_values = m_values or [1, 3]
    active_memory_strengths = memory_strengths or (
        list(range(ratio_total + 1)) if ratio_total is not None else [0, 1, 2, 3]
    )
    active_layers = hidden_state_layers or [0, 4, 8, 12, 16, 20, 24, 28]
    result_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []

    for seed in seeds:
        rng = random.Random(seed)
        backend = build_backend(
            backend_name=config.backend,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=max(config.max_tokens, 96),
            debug_dir=out_dir / "debug" / _safe_model_name(config.model) / f"seed_{seed:04d}",
            seed=seed,
            social_susceptibility=config.social_susceptibility,
            prompt_social_susceptibility=config.prompt_social_susceptibility,
            prompt_number_range=config.prompt_number_range,
            capture_hidden_states=config.capture_hidden_states,
            hidden_state_layers=active_layers if config.capture_hidden_states else config.hidden_state_layers,
            use_response_format=config.use_response_format,
            api_base_url=config.api_base_url,
            api_key=config.api_key,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            enable_thinking=config.enable_thinking,
        )
        trials = [
            (m, memory_strength, case, _target_social_counts(memory_strength, ratio_total))
            for m in active_m_values
            for memory_strength in active_memory_strengths
            for case in DEFAULT_CONFLICT_CASES
        ]
        rng.shuffle(trials)
        for trial_index, (m, memory_strength, case, (target_count, social_count)) in enumerate(trials):
            memory_lines = _memory_lines(
                truth_number=case.truth_number,
                social_number=case.social_number,
                m=m,
                target_count=target_count,
                social_count=social_count,
                rng=rng,
            )
            memory_relation = _memory_relation(target_count=target_count, social_count=social_count)
            memory_ratio_label = f"{target_count}:{social_count}"
            clue_info = _clue_info_fields(numbers, case.private_clue)
            prompt_text = prompts.interaction_text(
                numbers=numbers,
                private_clue=case.private_clue,
                memory_lines=memory_lines,
                m=m,
                social_susceptibility=config.social_susceptibility,
                prompt_social_susceptibility=config.prompt_social_susceptibility,
                prompt_number_range=config.prompt_number_range,
            )
            call_id = (
                f"seed{seed:04d}_trial{trial_index:04d}_m{m}_{case.case_id}_"
                f"mem{memory_ratio_label.replace(':', '_')}"
            )
            prompt_rows.append(
                {
                    "call_id": call_id,
                    "seed": seed,
                    "trial_index": trial_index,
                    "m": m,
                    "case_id": case.case_id,
                    "truth_number": case.truth_number,
                    "private_clue": case.private_clue,
                    "social_number": case.social_number,
                    "memory_strength": memory_strength,
                    "target_memory_count": target_count,
                    "social_memory_count": social_count,
                    "memory_total": target_count + social_count,
                    "memory_ratio_label": memory_ratio_label,
                    "memory_relation": memory_relation,
                    **clue_info,
                    "memory_lines": memory_lines,
                    "prompt": prompt_text,
                }
            )
            before_hidden_count = len(getattr(backend, "hidden_state_rows", []))
            try:
                message = backend.message(
                    numbers=numbers,
                    private_clue=case.private_clue,
                    memory_lines=memory_lines,
                    m=m,
                )
                row = {
                    "call_id": call_id,
                    "seed": seed,
                    "trial_index": trial_index,
                    "m": m,
                    "case_id": case.case_id,
                    "truth_number": case.truth_number,
                    "private_clue": case.private_clue,
                    "social_number": case.social_number,
                    "memory_strength": memory_strength,
                    "target_memory_count": target_count,
                    "social_memory_count": social_count,
                    "memory_total": target_count + social_count,
                    "memory_ratio_label": memory_ratio_label,
                    "memory_relation": memory_relation,
                    **clue_info,
                    "valid": True,
                    "number": message.number,
                    "reason": message.reason,
                    "correct": message.number == case.truth_number,
                    "satisfies_private_clue": clue_matches_number(case.private_clue, message.number),
                    "chose_social_number": message.number == case.social_number,
                    "response_category": _response_category(
                        number=message.number,
                        truth_number=case.truth_number,
                        social_number=case.social_number,
                        private_clue=case.private_clue,
                    ),
                    "choice_type": _choice_type(
                        number=message.number,
                        social_number=case.social_number,
                        private_clue=case.private_clue,
                    ),
                    "error": None,
                }
            except ParseError as exc:
                row = {
                    "call_id": call_id,
                    "seed": seed,
                    "trial_index": trial_index,
                    "m": m,
                    "case_id": case.case_id,
                    "truth_number": case.truth_number,
                    "private_clue": case.private_clue,
                    "social_number": case.social_number,
                    "memory_strength": memory_strength,
                    "target_memory_count": target_count,
                    "social_memory_count": social_count,
                    "memory_total": target_count + social_count,
                    "memory_ratio_label": memory_ratio_label,
                    "memory_relation": memory_relation,
                    **clue_info,
                    "valid": False,
                    "number": None,
                    "reason": None,
                    "correct": None,
                    "satisfies_private_clue": None,
                    "chose_social_number": None,
                    "response_category": "invalid",
                    "choice_type": "invalid",
                    "error": str(exc),
                }
            after_hidden = getattr(backend, "hidden_state_rows", [])[before_hidden_count:]
            for hidden_index, hidden_row in enumerate(after_hidden):
                hidden_rows.append(
                    {
                        "call_id": call_id,
                        "seed": seed,
                        "trial_index": trial_index,
                        "m": m,
                        "case_id": case.case_id,
                        "truth_number": case.truth_number,
                        "private_clue": case.private_clue,
                        "social_number": case.social_number,
                        "memory_strength": memory_strength,
                        "target_memory_count": target_count,
                        "social_memory_count": social_count,
                        "memory_total": target_count + social_count,
                        "memory_ratio_label": memory_ratio_label,
                        "memory_relation": memory_relation,
                        **clue_info,
                        "response_category": row["response_category"],
                        "choice_type": row["choice_type"],
                        "number": row["number"],
                        "satisfies_private_clue": row["satisfies_private_clue"],
                        "chose_social_number": row["chose_social_number"],
                        "hidden_index": hidden_index,
                        **hidden_row,
                    }
                )
            result_rows.append(row)

    summary_rows = _summarize_conflict_rows(result_rows)
    phase_summary_rows = _summarize_conflict_rows(result_rows, by_phase=True)
    clue_info_rows = _clue_information_rows(numbers)
    _write_csv(out_dir / "conflict_trials.csv", result_rows)
    _write_csv(out_dir / "conflict_summary.csv", summary_rows)
    _write_csv(out_dir / "conflict_phase_summary.csv", phase_summary_rows)
    _write_csv(out_dir / "clue_information.csv", clue_info_rows)
    _write_jsonl(out_dir / "conflict_prompts.jsonl", prompt_rows)
    _write_jsonl(out_dir / "conflict_hidden_states.jsonl", hidden_rows)
    _write_conflict_plot(summary_rows, out_dir / "plots" / "social_conflict_rates.svg")
    _write_response_composition_plot(summary_rows, out_dir / "plots" / "memory_conflict_composition.svg")
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(
            {
                "model": config.model,
                "seeds": seeds,
                "m_values": active_m_values,
                "memory_strengths": active_memory_strengths,
                "ratio_total": ratio_total,
                "hidden_state_layers": active_layers if config.capture_hidden_states else [],
                "trial_count": len(result_rows),
                "clue_information": clue_info_rows,
                "summary": summary_rows,
                "phase_summary": phase_summary_rows,
            },
            handle,
            indent=2,
        )
    return {"trials": result_rows, "summary": summary_rows, "hidden_rows": hidden_rows}


def _target_social_counts(memory_strength: int, ratio_total: int | None) -> tuple[int, int]:
    if ratio_total is None:
        return 0, memory_strength
    social_count = memory_strength
    target_count = max(ratio_total - social_count, 0)
    return target_count, social_count


def _memory_relation(*, target_count: int, social_count: int) -> str:
    if target_count > 0 and social_count == 0:
        return "target_only"
    if target_count == 0 and social_count > 0:
        return "social_only"
    if target_count > 0 and social_count > 0:
        return "conflict_mixture"
    return "private_only"


def _memory_lines(
    *,
    truth_number: int,
    social_number: int,
    m: int,
    target_count: int,
    social_count: int,
    rng: random.Random,
) -> list[str]:
    if target_count <= 0 and social_count <= 0:
        return []
    if m == 1:
        lines = [str(truth_number) for _ in range(target_count)]
        lines.extend(str(social_number) for _ in range(social_count))
        rng.shuffle(lines)
        return lines
    social_reasons = [
        f"{social_number} | My clue points to {social_number}.",
        f"{social_number} | I am confident the answer is {social_number}.",
        f"{social_number} | The evidence I saw supports {social_number}.",
    ]
    target_reasons = [
        f"{truth_number} | My clue points to {truth_number}.",
        f"{truth_number} | I am confident the answer is {truth_number}.",
        f"{truth_number} | The evidence I saw supports {truth_number}.",
    ]
    lines = [target_reasons[index % len(target_reasons)] for index in range(target_count)]
    lines.extend(social_reasons[index % len(social_reasons)] for index in range(social_count))
    rng.shuffle(lines)
    return lines


def _choice_type(*, number: int, social_number: int, private_clue: str) -> str:
    if number == social_number:
        return "social_over_private"
    if clue_matches_number(private_clue, number):
        return "private_resists_social"
    return "neither_private_nor_social"


def _response_category(*, number: int, truth_number: int, social_number: int, private_clue: str) -> str:
    if number == truth_number:
        return "private_target"
    if number == social_number:
        return "social_evidence"
    if clue_matches_number(private_clue, number):
        return "other_compatible"
    return "incompatible"


def _clue_info_fields(numbers: list[int], clue_text: str) -> dict[str, Any]:
    candidates = clue_candidates(numbers, clue_text)
    bits = clue_information_bits(numbers, clue_text)
    return {
        "private_clue_candidate_count": len(candidates),
        "private_clue_prior_probability": len(candidates) / float(len(numbers)) if numbers else None,
        "private_clue_info_bits": bits,
        "private_clue_info_phase": clue_information_phase(bits),
    }


def _clue_information_rows(numbers: list[int]) -> list[dict[str, Any]]:
    rows = []
    for clue in DEFAULT_CLUES:
        fields = _clue_info_fields(numbers, clue.text)
        rows.append(
            {
                "clue_name": clue.name,
                "clue_text": clue.text,
                "candidate_range_count": len(numbers),
                **fields,
            }
        )
    return rows


def _summarize_conflict_rows(rows: list[dict[str, Any]], *, by_phase: bool = False) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = [
            int(row["m"]),
            int(row.get("target_memory_count", 0) or 0),
            int(row.get("social_memory_count", row["memory_strength"]) or 0),
            str(row["memory_relation"]),
        ]
        if by_phase:
            key.insert(1, str(row.get("private_clue_info_phase") or "unknown"))
        groups[tuple(key)].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: _summary_sort_key(item[0], by_phase=by_phase),
    ):
        if by_phase:
            m, phase, target_count, social_count, memory_relation = key
        else:
            m, target_count, social_count, memory_relation = key
            phase = None
        valid = [row for row in group if bool(row.get("valid", False))]
        summary_row = {
            "m": m,
            "memory_strength": social_count,
            "target_memory_count": target_count,
            "social_memory_count": social_count,
            "memory_total": target_count + social_count,
            "memory_ratio_label": f"{target_count}:{social_count}",
            "memory_relation": memory_relation,
            "n": len(group),
            "valid_rate": len(valid) / max(len(group), 1),
            "private_clue_info_bits_mean": _numeric_mean(group, "private_clue_info_bits"),
            "private_clue_info_bits_min": _numeric_min(group, "private_clue_info_bits"),
            "private_clue_info_bits_max": _numeric_max(group, "private_clue_info_bits"),
            "private_target_rate": _response_category_rate(valid, "private_target"),
            "social_evidence_rate": _response_category_rate(valid, "social_evidence"),
            "other_compatible_rate": _response_category_rate(valid, "other_compatible"),
            "incompatible_rate": _response_category_rate(valid, "incompatible"),
            "satisfies_private_clue_rate": _rate(valid, "satisfies_private_clue"),
            "social_over_private_rate": _choice_rate(valid, "social_over_private"),
            "private_resists_social_rate": _choice_rate(valid, "private_resists_social"),
            "neither_private_nor_social_rate": _choice_rate(valid, "neither_private_nor_social"),
            "correct_rate": _rate(valid, "correct"),
        }
        if by_phase:
            summary_row["private_clue_info_phase"] = phase
        summary_rows.append(summary_row)
    return summary_rows


def _summary_sort_key(key: tuple[Any, ...], *, by_phase: bool) -> tuple[Any, ...]:
    if by_phase:
        m, phase, target_count, social_count, memory_relation = key
        return (m, _phase_sort_key(phase), -int(target_count), int(social_count), str(memory_relation))
    m, target_count, social_count, memory_relation = key
    return (m, -int(target_count), int(social_count), str(memory_relation))


def _phase_sort_key(phase: Any) -> int:
    return {"weak": 0, "medium": 1, "strong": 2, "unknown": 3}.get(str(phase), 4)


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        values.append(float(raw))
    return values


def _numeric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_values(rows, key)
    if not values:
        return None
    return sum(values) / float(len(values))


def _numeric_min(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_values(rows, key)
    if not values:
        return None
    return min(values)


def _numeric_max(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_values(rows, key)
    if not values:
        return None
    return max(values)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _choice_rate(rows: list[dict[str, Any]], choice_type: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("choice_type") == choice_type) / float(len(rows))


def _response_category_rate(rows: list[dict[str, Any]], response_category: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("response_category") == response_category) / float(len(rows))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_conflict_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 880, 520
    left, right, top, bottom = 78, 190, 58, 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    rows = [row for row in summary_rows if row["memory_relation"] == "conflict"]
    strengths = sorted({int(row["memory_strength"]) for row in rows})
    if not strengths:
        path.write_text("")
        return

    def x_for(strength: int) -> float:
        if len(strengths) == 1:
            return left + plot_width / 2
        return left + (strength - min(strengths)) * plot_width / float(max(strengths) - min(strengths))

    def y_for(value: float) -> float:
        return top + (1.0 - value) * plot_height

    series = [
        ("m=1 social overwrite", 1, "social_over_private_rate", "#dc2626"),
        ("m=1 private resists", 1, "private_resists_social_rate", "#2563eb"),
        ("m=3 social overwrite", 3, "social_over_private_rate", "#ea580c"),
        ("m=3 private resists", 3, "private_resists_social_rate", "#16a34a"),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#0f172a}.muted{fill:#64748b;font-size:12px}.label{fill:#334155;font-size:13px}.title{font-size:21px;font-weight:700}.grid{stroke:#e2e8f0}.axis{stroke:#334155;stroke-width:1.2}.line{fill:none;stroke-linecap:round;stroke-linejoin:round}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="34" class="title">Contradictory Social Memory vs Private Clue</text>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_for(tick)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="muted">{tick:.2f}</text>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>')
    svg.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>')
    for strength in strengths:
        x = x_for(strength)
        svg.append(f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#334155"/>')
        svg.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" class="muted">{strength}</text>')
    svg.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">Contradictory memory entries</text>')
    svg.append(f'<text x="23" y="{top + plot_height / 2:.1f}" transform="rotate(-90 23 {top + plot_height / 2:.1f})" text-anchor="middle" class="label">Rate</text>')

    by_key = {(int(row["m"]), int(row["memory_strength"])): row for row in rows}
    legend_x = left + plot_width + 34
    legend_y = top + 14
    for index, (label, m, metric, color) in enumerate(series):
        points = []
        for strength in strengths:
            row = by_key.get((m, strength))
            if row is not None:
                points.append((x_for(strength), y_for(float(row[metric]))))
        if points:
            data = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
            svg.append(f'<path d="{data}" class="line" stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        y = legend_y + index * 24
        svg.append(f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 24}" y2="{y - 4}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 32}" y="{y}" class="muted">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


def _write_response_composition_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    ratio_rows = [row for row in summary_rows if int(row.get("memory_total", 0) or 0) > 0]
    if not ratio_rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 520
    margin_left, margin_right, top, bottom = 74, 150, 76, 84
    panel_gap = 54
    panel_width = (width - margin_left - margin_right - panel_gap) / 2.0
    panel_height = height - top - bottom
    panel_origins = {1: margin_left, 3: margin_left + panel_width + panel_gap}
    categories = [
        ("private_target_rate", "Private target number", "#2f80ed"),
        ("social_evidence_rate", "Social evidence number", "#f47c20"),
        ("other_compatible_rate", "Other compatible number", "#2fb56e"),
        ("incompatible_rate", "Incompatible", "#cfd8dc"),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#0f172a}.muted{fill:#64748b;font-size:12px}.label{fill:#334155;font-size:13px}.title{font-size:21px;font-weight:700}.paneltitle{font-size:13px;font-weight:700}.grid{stroke:#e2e8f0}.axis{stroke:#334155;stroke-width:1.1}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="34" class="title">Memory-Conflict Probe: Response Composition</text>',
        f'<text x="{margin_left}" y="55" class="muted">Memory shifts from private-target-heavy to contradictory-social-heavy.</text>',
    ]

    def y_for(value: float) -> float:
        return top + (1.0 - value) * panel_height

    for m in (1, 3):
        rows = sorted(
            [row for row in ratio_rows if int(row["m"]) == m],
            key=lambda row: (-int(row["target_memory_count"]), int(row["social_memory_count"])),
        )
        if not rows:
            continue
        x0 = panel_origins[m]
        x_values = []
        for index, row in enumerate(rows):
            if len(rows) == 1:
                x = x0 + panel_width / 2.0
            else:
                x = x0 + index * panel_width / float(len(rows) - 1)
            x_values.append(x)
        for tick in (0.0, 0.5, 1.0):
            y = y_for(tick)
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}" class="grid"/>')
            if m == 1:
                svg.append(f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" class="muted">{tick:.1f}</text>')
        svg.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{top + panel_height}" class="axis"/>')
        svg.append(f'<line x1="{x0}" y1="{top + panel_height}" x2="{x0 + panel_width}" y2="{top + panel_height}" class="axis"/>')
        title = "m=1: number only" if m == 1 else "m=3: number + reason"
        svg.append(f'<text x="{x0}" y="{top - 14}" class="paneltitle">{title}</text>')

        lower = [0.0 for _ in rows]
        for metric, _label, color in categories:
            upper = [lower[index] + float(row.get(metric, 0.0) or 0.0) for index, row in enumerate(rows)]
            top_points = [f"{x_values[index]:.1f},{y_for(upper[index]):.1f}" for index in range(len(rows))]
            bottom_points = [
                f"{x_values[index]:.1f},{y_for(lower[index]):.1f}" for index in range(len(rows) - 1, -1, -1)
            ]
            svg.append(
                f'<polygon points="{" ".join(top_points + bottom_points)}" fill="{color}" opacity="0.95"/>'
            )
            lower = upper

        for index, row in enumerate(rows):
            label = str(row["memory_ratio_label"])
            x = x_values[index]
            svg.append(f'<line x1="{x:.1f}" y1="{top + panel_height}" x2="{x:.1f}" y2="{top + panel_height + 5}" stroke="#334155"/>')
            svg.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 22}" text-anchor="middle" class="muted" transform="rotate(45 {x:.1f} {top + panel_height + 22})">{label}</text>'
            )

    svg.append(
        f'<text x="{margin_left + (width - margin_left - margin_right) / 2:.1f}" y="{height - 22}" text-anchor="middle" class="label">Memory entries (private target : contradictory social evidence)</text>'
    )
    svg.append(
        f'<text x="23" y="{top + panel_height / 2:.1f}" transform="rotate(-90 23 {top + panel_height / 2:.1f})" text-anchor="middle" class="label">Agent response probability</text>'
    )
    legend_x = width - margin_right + 24
    legend_y = top + 4
    for index, (_metric, label, color) in enumerate(categories):
        y = legend_y + index * 27
        svg.append(f'<rect x="{legend_x}" y="{y - 12}" width="18" height="14" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 26}" y="{y}" class="muted">{label}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n")


def _safe_model_name(model: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in model).strip("._") or "model"

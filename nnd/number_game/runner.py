from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

from nnd.number_game.backend import build_backend
from nnd.number_game.config import NumberGameConfig, save_resolved_config
from nnd.number_game.domain import candidate_numbers, clue_matches_number, sample_truth_and_clues
from nnd.number_game.parsing import ParseError
from nnd.number_game.viz import plot_number_share_trajectories


@dataclass(frozen=True)
class NumberRecord:
    round: int
    t: int
    agent_id: int
    role: str
    model: str
    m: int
    valid: bool
    number: int | None
    reason: str | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True)
class DecisionRecord:
    round: int
    agent_id: int
    model: str
    valid: bool
    initial_number: int | None
    number: int | None
    reason: str | None
    influential_agent_ids: list[int]
    changed_mind: bool | None
    correct: bool | None
    error: str | None = None


@dataclass(frozen=True)
class ProbeRecord:
    t: int
    agent_id: int
    model: str
    m: int
    valid: bool
    number: int | None
    reason: str | None
    correct: bool | None
    memory_length: int
    changed_from_previous_probe: bool | None
    changed_from_initial_probe: bool | None
    error: str | None = None


def choose_default_backend() -> str:
    return "openai_compatible" if _has_openai_target() else "scripted"


def _has_openai_target() -> bool:
    import os

    return bool(
        os.environ.get("NND_MODEL_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_KEY")
    )


def _resolve_agent_models(config: NumberGameConfig) -> list[str]:
    total_agents = config.N + 1 if config.protocol == "org" else config.N
    if config.agent_models is not None:
        return list(config.agent_models)
    return [config.model for _ in range(total_agents)]


def _model_debug_dir(debug_root: Path, model: str) -> Path:
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._")
    return debug_root / (suffix or "model")


def _build_agent_backends(config: NumberGameConfig, *, out_dir: Path, seed: int, agent_models: list[str]) -> list[Any]:
    cache: dict[str, Any] = {}
    for model in agent_models:
        if model not in cache:
            cache[model] = build_backend(
                backend_name=config.backend,
                model=model,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                debug_dir=_model_debug_dir(out_dir / "debug", model),
                seed=seed,
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
    return [cache[model] for model in agent_models]


def _summarize(rows: list[dict[str, Any]], *, truth_number: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {"final_accuracy": 0.0, "final_consensus_number": None}
    per_round: list[dict[str, Any]] = []
    for round_idx in sorted({int(row["round"]) for row in rows}):
        round_rows = [row for row in rows if int(row["round"]) == round_idx and bool(row.get("valid", False))]
        counts = Counter(int(row["number"]) for row in round_rows if row.get("number") is not None)
        if counts:
            number, count = counts.most_common(1)[0]
            top_share = count / float(len(round_rows))
        else:
            number, top_share = None, 0.0
        per_round.append(
            {
                "round": round_idx,
                "valid_count": len(round_rows),
                "consensus_number": number,
                "top1_share": top_share,
                "accuracy": sum(1 for row in round_rows if row.get("number") == truth_number) / max(len(round_rows), 1),
            }
        )
    final = per_round[-1]
    return per_round, {
        "truth_number": truth_number,
        "final_consensus_number": final["consensus_number"],
        "final_consensus_correct": final["consensus_number"] == truth_number,
        "final_accuracy": final["accuracy"],
        "final_top1_share": final["top1_share"],
    }


def _stable_consensus(
    per_round_rows: list[dict[str, Any]],
    window: int,
    *,
    consensus_threshold: float = 1.0,
) -> tuple[bool, int | None]:
    if window <= 0 or len(per_round_rows) < window:
        return False, None
    recent = per_round_rows[-window:]
    values = [row["consensus_number"] for row in recent]
    if any(value is None for value in values):
        return False, None
    if len(set(int(value) for value in values)) != 1:
        return False, None
    if not all(float(row["top1_share"]) >= consensus_threshold for row in recent):
        return False, None
    return True, int(values[0])


def _stable_probe_consensus(
    probe_rows: list[dict[str, Any]],
    *,
    n_agents: int,
    window: int,
    consensus_threshold: float,
) -> tuple[bool, int | None]:
    if window <= 0:
        return False, None
    valid = [row for row in probe_rows if bool(row.get("valid", False)) and row.get("number") is not None]
    probe_times = sorted({int(row["t"]) for row in valid})
    if len(probe_times) < window:
        return False, None
    recent_times = probe_times[-window:]
    consensus_numbers: list[int] = []
    for t in recent_times:
        rows_at_t = [row for row in valid if int(row["t"]) == t]
        if len(rows_at_t) < n_agents:
            return False, None
        counts = Counter(int(row["number"]) for row in rows_at_t)
        number, count = counts.most_common(1)[0]
        if count / float(n_agents) < consensus_threshold:
            return False, None
        consensus_numbers.append(number)
    if len(set(consensus_numbers)) != 1:
        return False, None
    return True, consensus_numbers[0]


def _line_from_message(row: dict[str, Any]) -> str:
    if bool(row.get("valid", False)) and row.get("number") is not None:
        if row.get("reason"):
            return f"agent {row['agent_id']} | number {row['number']} | reason {row['reason']}"
        return f"agent {row['agent_id']} | number {row['number']}"
    return f"agent {row['agent_id']} | invalid"


def _add_private_clue_metrics(row: dict[str, Any], private_clue: str | None) -> None:
    row["private_clue"] = private_clue
    value = (
        clue_matches_number(private_clue, int(row["number"]))
        if private_clue is not None and row.get("number") is not None
        else None
    )
    row["satisfies_private_clue"] = value
    row["private_clue_consistent"] = value


def _satisfies_private_clue_rate(rows: list[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        value = row.get("satisfies_private_clue")
        if value is None:
            value = row.get("private_clue_consistent")
        if value is not None:
            values.append(bool(value))
    if not values:
        return None
    return sum(values) / float(len(values))


def run_number_game_experiment(config: NumberGameConfig, *, out_dir: Path, seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, out_dir)
    rng = random.Random(seed)
    numbers = candidate_numbers(config.min_number, config.max_number)
    observer_count = config.N
    total_agents = config.N + 1 if config.protocol == "org" else config.N
    truth_number, private_clues = sample_truth_and_clues(
        rng=rng,
        n_agents=observer_count,
        min_number=config.min_number,
        max_number=config.max_number,
        fixed_truth_number=config.fixed_truth_number,
    )
    private_clue_texts = [clue.text for clue in private_clues]
    agent_models = _resolve_agent_models(config)
    backends = _build_agent_backends(config, out_dir=out_dir, seed=seed, agent_models=agent_models)

    if config.protocol == "pairwise":
        result = _run_pairwise(config, backends, numbers, private_clue_texts, agent_models, truth_number, seed)
    elif config.protocol == "broadcast":
        result = _run_broadcast(config, backends, numbers, private_clue_texts, agent_models, truth_number)
    else:
        result = _run_org(config, backends, numbers, private_clue_texts, agent_models, truth_number)

    result["summary"].update(
        {
            "protocol": config.protocol,
            "seed": seed,
            "truth_number": truth_number,
            "agent_private_clues": private_clue_texts,
        }
    )
    _write_outputs(out_dir, result, backends)
    return result


def run_pairwise_m_comparison(
    config: NumberGameConfig,
    *,
    out_dir: Path,
    seeds: list[int],
    m_values: list[int] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    active_m_values = m_values or [1, 3]
    condition_rows: list[dict[str, Any]] = []

    for m in active_m_values:
        condition_config = config.model_copy(
            update={
                "protocol": "pairwise",
                "interaction_m": m,
                "max_tokens": max(config.max_tokens, 96) if m == 3 else config.max_tokens,
            }
        )
        for seed in seeds:
            run_dir = out_dir / f"m{m}" / f"seed_{seed:04d}"
            result = run_number_game_experiment(condition_config, out_dir=run_dir, seed=seed)
            messages = result["messages"]
            probes = result.get("probes", [])
            valid_count = sum(1 for row in messages if bool(row.get("valid", False)))
            valid_messages = [row for row in messages if bool(row.get("valid", False))]
            message_satisfies_rate = _satisfies_private_clue_rate(valid_messages)
            condition_rows.append(
                {
                    "m": m,
                    "seed": seed,
                    "model": condition_config.model,
                    "N": condition_config.N,
                    "T": condition_config.T,
                    "H": condition_config.H,
                    "min_number": condition_config.min_number,
                    "max_number": condition_config.max_number,
                    "truth_number": result["summary"].get("truth_number"),
                    "final_consensus_number": result["summary"].get("final_consensus_number"),
                    "final_consensus_correct": result["summary"].get("final_consensus_correct"),
                    "final_accuracy": result["summary"].get("final_accuracy"),
                    "final_top1_share": result["summary"].get("final_top1_share"),
                    "valid_rate": valid_count / max(len(messages), 1),
                    "message_satisfies_private_clue_rate": message_satisfies_rate,
                    "message_private_clue_consistency_rate": message_satisfies_rate,
                    "probe_count": len(probes),
                    "probe_change_rate": result["summary"].get("probe_change_rate"),
                    "initial_probe_accuracy": result["summary"].get("initial_probe_accuracy"),
                    "final_probe_accuracy": result["summary"].get("final_probe_accuracy"),
                    "initial_probe_satisfies_private_clue_rate": result["summary"].get(
                        "initial_probe_satisfies_private_clue_rate"
                    ),
                    "final_probe_satisfies_private_clue_rate": result["summary"].get(
                        "final_probe_satisfies_private_clue_rate"
                    ),
                    "initial_probe_private_clue_consistency_rate": result["summary"].get(
                        "initial_probe_private_clue_consistency_rate"
                    ),
                    "final_probe_private_clue_consistency_rate": result["summary"].get(
                        "final_probe_private_clue_consistency_rate"
                    ),
                    "changed_to_truth_count": result["summary"].get("changed_to_truth_count"),
                    "changed_away_from_truth_count": result["summary"].get("changed_away_from_truth_count"),
                    "run_dir": str(run_dir),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for m in active_m_values:
        rows = [row for row in condition_rows if int(row["m"]) == m]
        if not rows:
            continue
        summary_rows.append(
            {
                "m": m,
                "num_seeds": len(rows),
                "mean_final_accuracy": _mean(float(row["final_accuracy"]) for row in rows),
                "mean_final_top1_share": _mean(float(row["final_top1_share"]) for row in rows),
                "mean_valid_rate": _mean(float(row["valid_rate"]) for row in rows),
                "mean_message_satisfies_private_clue_rate": _mean(
                    float(row["message_satisfies_private_clue_rate"])
                    for row in rows
                    if row.get("message_satisfies_private_clue_rate") not in (None, "")
                ),
                "mean_message_private_clue_consistency_rate": _mean(
                    float(row["message_private_clue_consistency_rate"])
                    for row in rows
                    if row.get("message_private_clue_consistency_rate") not in (None, "")
                ),
                "consensus_correct_rate": _mean(1.0 if row["final_consensus_correct"] else 0.0 for row in rows),
                "mean_probe_change_rate": _mean(
                    float(row["probe_change_rate"])
                    for row in rows
                    if row.get("probe_change_rate") not in (None, "")
                ),
                "mean_initial_probe_accuracy": _mean(
                    float(row["initial_probe_accuracy"])
                    for row in rows
                    if row.get("initial_probe_accuracy") not in (None, "")
                ),
                "mean_final_probe_accuracy": _mean(
                    float(row["final_probe_accuracy"])
                    for row in rows
                    if row.get("final_probe_accuracy") not in (None, "")
                ),
                "mean_initial_probe_satisfies_private_clue_rate": _mean(
                    float(row["initial_probe_satisfies_private_clue_rate"])
                    for row in rows
                    if row.get("initial_probe_satisfies_private_clue_rate") not in (None, "")
                ),
                "mean_final_probe_satisfies_private_clue_rate": _mean(
                    float(row["final_probe_satisfies_private_clue_rate"])
                    for row in rows
                    if row.get("final_probe_satisfies_private_clue_rate") not in (None, "")
                ),
                "mean_initial_probe_private_clue_consistency_rate": _mean(
                    float(row["initial_probe_private_clue_consistency_rate"])
                    for row in rows
                    if row.get("initial_probe_private_clue_consistency_rate") not in (None, "")
                ),
                "mean_final_probe_private_clue_consistency_rate": _mean(
                    float(row["final_probe_private_clue_consistency_rate"])
                    for row in rows
                    if row.get("final_probe_private_clue_consistency_rate") not in (None, "")
                ),
            }
        )

    _write_csv(out_dir / "pairwise_m_condition_results.csv", condition_rows)
    _write_csv(out_dir / "pairwise_m_summary.csv", summary_rows)
    with open(out_dir / "pairwise_m_summary.json", "w") as handle:
        json.dump({"conditions": condition_rows, "summary": summary_rows}, handle, indent=2)
    return {"conditions": condition_rows, "summary": summary_rows}


def _run_pairwise(
    config: NumberGameConfig,
    backends: list[Any],
    numbers: list[int],
    private_clues: list[str],
    agent_models: list[str],
    truth_number: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    memories = [deque(maxlen=config.H) for _ in range(config.N)]
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    previous_probe_numbers: dict[int, int | None] = {}
    initial_probe_numbers: dict[int, int | None] = {}

    def _probe_all(t: int) -> None:
        for agent_id in range(config.N):
            memory_lines = list(memories[agent_id])
            try:
                message = backends[agent_id].message(
                    numbers=numbers,
                    private_clue=private_clues[agent_id],
                    memory_lines=memory_lines,
                    m=config.interaction_m,
                )
                previous = previous_probe_numbers.get(agent_id)
                initial = initial_probe_numbers.get(agent_id)
                if t == 0:
                    initial_probe_numbers[agent_id] = message.number
                row = asdict(
                    ProbeRecord(
                        t=t,
                        agent_id=agent_id,
                        model=agent_models[agent_id],
                        m=config.interaction_m,
                        valid=True,
                        number=message.number,
                        reason=message.reason,
                        correct=message.number == truth_number,
                        memory_length=len(memory_lines),
                        changed_from_previous_probe=(
                            None if agent_id not in previous_probe_numbers else message.number != previous
                        ),
                        changed_from_initial_probe=(
                            None if initial is None else message.number != initial
                        ),
                    )
                )
                previous_probe_numbers[agent_id] = message.number
            except ParseError as exc:
                row = asdict(
                    ProbeRecord(
                        t=t,
                        agent_id=agent_id,
                        model=agent_models[agent_id],
                        m=config.interaction_m,
                        valid=False,
                        number=None,
                        reason=None,
                        correct=None,
                        memory_length=len(memory_lines),
                        changed_from_previous_probe=None,
                        changed_from_initial_probe=None,
                        error=str(exc),
                    )
                )
            _add_private_clue_metrics(row, private_clues[agent_id])
            probe_rows.append(row)

    if config.probe_every is not None:
        _probe_all(0)
    early_stop_number: int | None = None
    early_stop_t: int | None = None
    for t in range(1, config.T + 1):
        speaker = rng.randrange(config.N)
        listener = rng.randrange(config.N - 1)
        if listener >= speaker:
            listener += 1
        speaker_memory_before = list(memories[speaker])
        listener_memory_before = list(memories[listener])
        try:
            message = backends[speaker].message(
                numbers=numbers,
                private_clue=private_clues[speaker],
                memory_lines=speaker_memory_before,
                m=config.interaction_m,
            )
            row = asdict(
                NumberRecord(
                    round=t,
                    t=t,
                    agent_id=speaker,
                    role="speaker",
                    model=agent_models[speaker],
                    m=config.interaction_m,
                    valid=True,
                    number=message.number,
                    reason=message.reason,
                    correct=message.number == truth_number,
                )
            )
            if config.H > 0:
                memories[listener].append(message.normalized_memory_entry())
        except ParseError as exc:
            row = asdict(NumberRecord(t, t, speaker, "speaker", agent_models[speaker], config.interaction_m, False, None, None, None, str(exc)))
        row["listener_id"] = listener
        row["speaker_private_clue"] = private_clues[speaker]
        row["listener_private_clue"] = private_clues[listener]
        _add_private_clue_metrics(row, private_clues[speaker])
        row["speaker_memory_before"] = list(speaker_memory_before)
        row["listener_memory_before"] = list(listener_memory_before)
        row["listener_memory_after"] = list(memories[listener])
        rows.append(row)
        if config.probe_every is not None and (t % config.probe_every == 0 or t == config.T):
            _probe_all(t)
            stop, stop_number = _stable_probe_consensus(
                probe_rows,
                n_agents=config.N,
                window=config.early_stop_window,
                consensus_threshold=config.consensus_threshold,
            )
            if stop:
                early_stop_number = stop_number
                early_stop_t = t
                break
    per_round_df, summary = _summarize(rows, truth_number=truth_number)
    social_summary, change_rows = _summarize_social_change(
        probes=probe_rows,
        messages=rows,
        truth_number=truth_number,
        n_agents=config.N,
    )
    if social_summary.get("final_probe_consensus_number") is not None:
        summary["final_message_consensus_number"] = summary.get("final_consensus_number")
        summary["final_message_consensus_correct"] = summary.get("final_consensus_correct")
        summary["final_message_accuracy"] = summary.get("final_accuracy")
        summary["final_message_top1_share"] = summary.get("final_top1_share")
        summary["final_consensus_number"] = social_summary["final_probe_consensus_number"]
        summary["final_consensus_correct"] = social_summary["final_probe_consensus_correct"]
        summary["final_accuracy"] = social_summary["final_probe_accuracy"]
        summary["final_top1_share"] = social_summary["final_probe_top1_share"]
    summary.update(social_summary)
    summary["early_stopped"] = early_stop_t is not None
    summary["early_stop_t"] = early_stop_t
    summary["early_stop_number"] = early_stop_number
    return {
        "messages": rows,
        "decisions": [],
        "probes": probe_rows,
        "probe_changes": change_rows,
        "per_round": per_round_df,
        "summary": summary,
    }


def _run_broadcast(
    config: NumberGameConfig,
    backends: list[Any],
    numbers: list[int],
    private_clues: list[str],
    agent_models: list[str],
    truth_number: int,
) -> dict[str, Any]:
    memories = [deque(maxlen=config.H) for _ in range(config.N)]
    message_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=config.agent_workers) if config.agent_workers > 1 else None

    def one_message(agent_id: int, round_idx: int, snapshot: list[str]) -> dict[str, Any]:
        try:
            message = backends[agent_id].message(numbers=numbers, private_clue=private_clues[agent_id], memory_lines=snapshot, m=config.interaction_m)
            row = asdict(NumberRecord(round_idx, round_idx, agent_id, "broadcaster", agent_models[agent_id], config.interaction_m, True, message.number, message.reason, message.number == truth_number))
        except ParseError as exc:
            row = asdict(NumberRecord(round_idx, round_idx, agent_id, "broadcaster", agent_models[agent_id], config.interaction_m, False, None, None, None, str(exc)))
        _add_private_clue_metrics(row, private_clues[agent_id])
        return row

    try:
        for round_idx in range(1, config.rounds + 1):
            snapshots = [list(memory) for memory in memories]
            if executor is None:
                round_messages = [one_message(agent_id, round_idx, snapshots[agent_id]) for agent_id in range(config.N)]
            else:
                round_messages = list(executor.map(lambda agent_id: one_message(agent_id, round_idx, snapshots[agent_id]), range(config.N)))
            message_rows.extend(round_messages)
            message_map = {int(row["agent_id"]): row for row in round_messages}
            sorted_rows = sorted(round_messages, key=lambda row: int(row["agent_id"]))
            for agent_id in range(config.N):
                visible = [row for row in sorted_rows if int(row["agent_id"]) != agent_id]
                try:
                    decision = backends[agent_id].final_decision(
                        numbers=numbers,
                        private_clue=private_clues[agent_id],
                        memory_lines=snapshots[agent_id],
                        broadcast_lines=[_line_from_message(row) for row in visible],
                        m=config.interaction_m,
                        max_influential_agents=config.max_influential_agents,
                        valid_agent_ids={int(row["agent_id"]) for row in visible if bool(row.get("valid", False))},
                    )
                    initial = message_map[agent_id].get("number")
                    decision_rows.append(
                        asdict(
                            DecisionRecord(
                                round_idx,
                                agent_id,
                                agent_models[agent_id],
                                True,
                                initial,
                                decision.number,
                                decision.reason,
                                list(decision.influential_agent_ids),
                                decision.number != initial if initial is not None else None,
                                decision.number == truth_number,
                            )
                        )
                    )
                    _add_private_clue_metrics(decision_rows[-1], private_clues[agent_id])
                    if config.H > 0:
                        memories[agent_id].append(decision.normalized_memory_entry())
                except ParseError as exc:
                    row = asdict(DecisionRecord(round_idx, agent_id, agent_models[agent_id], False, None, None, None, [], None, None, str(exc)))
                    _add_private_clue_metrics(row, private_clues[agent_id])
                    decision_rows.append(row)
            per_round_df, _ = _summarize(decision_rows, truth_number=truth_number)
            stop, _ = _stable_consensus(
                per_round_df,
                config.early_stop_window,
                consensus_threshold=config.consensus_threshold,
            )
            if stop:
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    per_round_df, summary = _summarize(decision_rows, truth_number=truth_number)
    return {"messages": message_rows, "decisions": decision_rows, "probes": [], "probe_changes": [], "per_round": per_round_df, "summary": summary}


def _run_org(
    config: NumberGameConfig,
    backends: list[Any],
    numbers: list[int],
    private_clues: list[str],
    agent_models: list[str],
    truth_number: int,
) -> dict[str, Any]:
    aggregator_id = config.aggregator_agent_id
    observer_ids = [agent_id for agent_id in range(config.N + 1) if agent_id != aggregator_id]
    clues_by_agent = dict(zip(observer_ids, private_clues, strict=True))
    shared_memory: deque[str] = deque(maxlen=config.H)
    message_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for round_idx in range(1, config.rounds + 1):
        snapshot = list(shared_memory)
        round_messages = []
        for agent_id in observer_ids:
            try:
                message = backends[agent_id].message(numbers=numbers, private_clue=clues_by_agent[agent_id], memory_lines=snapshot, m=config.interaction_m)
                row = asdict(NumberRecord(round_idx, round_idx, agent_id, "observer", agent_models[agent_id], config.interaction_m, True, message.number, message.reason, message.number == truth_number))
            except ParseError as exc:
                row = asdict(NumberRecord(round_idx, round_idx, agent_id, "observer", agent_models[agent_id], config.interaction_m, False, None, None, None, str(exc)))
            _add_private_clue_metrics(row, clues_by_agent[agent_id])
            round_messages.append(row)
        message_rows.extend(round_messages)
        try:
            decision = backends[aggregator_id].organization_decision(
                numbers=numbers,
                memory_lines=snapshot,
                observer_statement_lines=[_line_from_message(row) for row in round_messages],
                m=config.interaction_m,
            )
            decision_rows.append(asdict(DecisionRecord(round_idx, aggregator_id, agent_models[aggregator_id], True, None, decision.number, decision.reason, [], None, decision.number == truth_number)))
            _add_private_clue_metrics(decision_rows[-1], None)
            if config.H > 0:
                shared_memory.append(decision.normalized_memory_entry())
        except ParseError as exc:
            row = asdict(DecisionRecord(round_idx, aggregator_id, agent_models[aggregator_id], False, None, None, None, [], None, None, str(exc)))
            _add_private_clue_metrics(row, None)
            decision_rows.append(row)
        per_round_df, _ = _summarize(decision_rows, truth_number=truth_number)
        stop, _ = _stable_consensus(
            per_round_df,
            config.early_stop_window,
            consensus_threshold=config.consensus_threshold,
        )
        if stop:
            break
    per_round_df, summary = _summarize(decision_rows, truth_number=truth_number)
    return {"messages": message_rows, "decisions": decision_rows, "probes": [], "probe_changes": [], "per_round": per_round_df, "summary": summary}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(out_dir: Path, result: dict[str, Any], backends: list[Any]) -> None:
    _write_csv(out_dir / "messages.csv", result["messages"])
    _write_csv(out_dir / "decisions.csv", result["decisions"])
    _write_csv(out_dir / "probes.csv", result.get("probes", []))
    _write_csv(out_dir / "probe_changes.csv", result.get("probe_changes", []))
    _write_csv(out_dir / "per_round.csv", result["per_round"])
    if result.get("probes"):
        (out_dir / "probe_agent_timeline.txt").write_text(_render_probe_agent_timeline(result["probes"]))
        (out_dir / "number_share_timeline.txt").write_text(_render_number_share_timeline(result["probes"]))
        plot_number_share_trajectories(
            result["probes"],
            truth_number=result["summary"].get("truth_number"),
            out_dir=out_dir,
        )
    if result.get("messages"):
        (out_dir / "dialogues.md").write_text(_render_dialogues(result["messages"]))
    if result.get("probe_changes"):
        with open(out_dir / "social_influence_summary.json", "w") as handle:
            json.dump(
                {
                    key: value
                    for key, value in result["summary"].items()
                    if key.startswith("probe_")
                    or key.startswith("initial_probe_")
                    or key.startswith("final_probe_")
                    or key.startswith("changed_")
                },
                handle,
                indent=2,
            )
    usage = [backend.usage_summary() for backend in {id(backend): backend for backend in backends}.values() if hasattr(backend, "usage_summary")]
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({**result["summary"], "backend_usage": usage}, handle, indent=2)
    hidden_rows = []
    for backend in {id(backend): backend for backend in backends}.values():
        hidden_rows.extend(getattr(backend, "hidden_state_rows", []))
    if hidden_rows:
        with open(out_dir / "hidden_states.jsonl", "w") as handle:
            for row in hidden_rows:
                handle.write(json.dumps(row) + "\n")


def _mean(values: Any) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return sum(collected) / float(len(collected))


def _summarize_social_change(
    *,
    probes: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    truth_number: int,
    n_agents: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid_probes = [row for row in probes if bool(row.get("valid", False)) and row.get("number") is not None]
    if not valid_probes:
        return {
            "probe_count": 0,
            "probe_change_rate": None,
            "initial_probe_accuracy": None,
            "final_probe_accuracy": None,
            "initial_probe_satisfies_private_clue_rate": None,
            "final_probe_satisfies_private_clue_rate": None,
            "initial_probe_private_clue_consistency_rate": None,
            "final_probe_private_clue_consistency_rate": None,
            "changed_agent_count": None,
            "changed_agent_rate": None,
            "changed_to_truth_count": None,
            "changed_away_from_truth_count": None,
        }, []

    probe_times = sorted({int(row["t"]) for row in valid_probes})
    first_t = probe_times[0]
    last_t = probe_times[-1]
    first_rows = [row for row in valid_probes if int(row["t"]) == first_t]
    last_rows = [row for row in valid_probes if int(row["t"]) == last_t]
    first_by_agent = {int(row["agent_id"]): int(row["number"]) for row in first_rows}
    last_by_agent = {int(row["agent_id"]): int(row["number"]) for row in last_rows}
    first_counts = Counter(first_by_agent.values())
    last_counts = Counter(last_by_agent.values())
    first_consensus_number, first_consensus_count = first_counts.most_common(1)[0]
    last_consensus_number, last_consensus_count = last_counts.most_common(1)[0]

    change_rows: list[dict[str, Any]] = []
    prev_by_agent: dict[int, dict[str, Any]] = {}
    messages_by_listener: dict[int, list[dict[str, Any]]] = {}
    for message in messages:
        listener = message.get("listener_id")
        if isinstance(listener, int):
            messages_by_listener.setdefault(listener, []).append(message)

    for row in valid_probes:
        agent_id = int(row["agent_id"])
        previous = prev_by_agent.get(agent_id)
        if previous is not None and previous.get("number") != row.get("number"):
            t = int(row["t"])
            received_since_previous = [
                message
                for message in messages_by_listener.get(agent_id, [])
                if int(previous["t"]) < int(message["t"]) <= t
            ]
            change_rows.append(
                {
                    "agent_id": agent_id,
                    "from_t": previous["t"],
                    "to_t": t,
                    "from_number": previous["number"],
                    "to_number": row["number"],
                    "to_truth": row["number"] == truth_number,
                    "away_from_truth": previous["number"] == truth_number and row["number"] != truth_number,
                    "private_clue": row.get("private_clue"),
                    "received_message_count": len(received_since_previous),
                    "received_numbers": [
                        message.get("number")
                        for message in received_since_previous
                        if message.get("number") is not None
                    ],
                    "received_reasons": [
                        message.get("reason")
                        for message in received_since_previous
                        if message.get("reason")
                    ],
                }
            )
        prev_by_agent[agent_id] = row

    changed_agents = {
        agent_id
        for agent_id, first_number in first_by_agent.items()
        if agent_id in last_by_agent and last_by_agent[agent_id] != first_number
    }
    initial_accuracy = sum(number == truth_number for number in first_by_agent.values()) / max(len(first_by_agent), 1)
    final_accuracy = sum(number == truth_number for number in last_by_agent.values()) / max(len(last_by_agent), 1)
    initial_satisfies_rate = _satisfies_private_clue_rate(first_rows)
    final_satisfies_rate = _satisfies_private_clue_rate(last_rows)
    return {
        "probe_count": len(valid_probes),
        "probe_round_count": len(probe_times),
        "probe_change_count": len(change_rows),
        "probe_change_rate": len(change_rows) / max(len(valid_probes) - n_agents, 1),
        "initial_probe_t": first_t,
        "final_probe_t": last_t,
        "initial_probe_accuracy": initial_accuracy,
        "final_probe_accuracy": final_accuracy,
        "initial_probe_satisfies_private_clue_rate": initial_satisfies_rate,
        "final_probe_satisfies_private_clue_rate": final_satisfies_rate,
        "initial_probe_private_clue_consistency_rate": initial_satisfies_rate,
        "final_probe_private_clue_consistency_rate": final_satisfies_rate,
        "initial_probe_consensus_number": first_consensus_number,
        "initial_probe_consensus_correct": first_consensus_number == truth_number,
        "initial_probe_top1_share": first_consensus_count / max(len(first_by_agent), 1),
        "final_probe_consensus_number": last_consensus_number,
        "final_probe_consensus_correct": last_consensus_number == truth_number,
        "final_probe_top1_share": last_consensus_count / max(len(last_by_agent), 1),
        "changed_agent_count": len(changed_agents),
        "changed_agent_rate": len(changed_agents) / max(n_agents, 1),
        "changed_to_truth_count": sum(1 for row in change_rows if row["to_truth"]),
        "changed_away_from_truth_count": sum(1 for row in change_rows if row["away_from_truth"]),
    }, change_rows


def _render_probe_agent_timeline(probes: list[dict[str, Any]]) -> str:
    valid = [row for row in probes if bool(row.get("valid", False))]
    times = sorted({int(row["t"]) for row in valid})
    agents = sorted({int(row["agent_id"]) for row in valid})
    by_key = {(int(row["agent_id"]), int(row["t"])): row for row in valid}
    width = max(2, max((len(str(row.get("number"))) for row in valid if row.get("number") is not None), default=2))
    lines = ["# Probe Agent Timeline", "", "Each row is one agent's probed belief over time.", ""]
    header = "agent clue".ljust(34) + " | " + " ".join(f"{t:>{width}}" for t in times)
    lines.append(header)
    lines.append("-" * len(header))
    for agent_id in agents:
        clue = str(by_key.get((agent_id, times[0]), {}).get("private_clue", ""))
        cells: list[str] = []
        previous = None
        for t in times:
            row = by_key.get((agent_id, t))
            value = "?"
            if row is not None and row.get("number") is not None:
                value = str(row["number"])
            marker = "*" if previous is not None and value != previous else " "
            cells.append(f"{value:>{width}}{marker}")
            previous = value
        lines.append(f"A{agent_id:02d} {clue}".ljust(34) + " | " + " ".join(cells))
    lines.append("")
    lines.append("* marks a changed probed number from the previous probe.")
    return "\n".join(lines) + "\n"


def _render_number_share_timeline(probes: list[dict[str, Any]]) -> str:
    valid = [row for row in probes if bool(row.get("valid", False)) and row.get("number") is not None]
    times = sorted({int(row["t"]) for row in valid})
    lines = ["# Number Share Timeline", "", "Each bar shows the share of probed agents choosing each number.", ""]
    for t in times:
        round_rows = [row for row in valid if int(row["t"]) == t]
        counts = Counter(int(row["number"]) for row in round_rows)
        total = max(sum(counts.values()), 1)
        parts = []
        for number, count in sorted(counts.items()):
            bar = "#" * max(1, round(20 * count / total))
            parts.append(f"{number}: {bar} {count}/{total}")
        lines.append(f"t={t:04d} | " + " | ".join(parts))
    return "\n".join(lines) + "\n"


def _render_dialogues(messages: list[dict[str, Any]]) -> str:
    lines = ["# Pairwise Dialogue Transcript", ""]
    for row in messages:
        t = row.get("t")
        speaker = row.get("agent_id")
        listener = row.get("listener_id")
        lines.append(f"## t={t} A{int(speaker):02d} -> A{int(listener):02d}" if isinstance(speaker, int) and isinstance(listener, int) else f"## t={t}")
        lines.append("")
        lines.append(f"- speaker clue: {row.get('speaker_private_clue', '')}")
        lines.append(f"- listener clue: {row.get('listener_private_clue', '')}")
        if bool(row.get("valid", False)):
            lines.append(f"- message number: {row.get('number')}")
            if row.get("satisfies_private_clue") is not None:
                lines.append(f"- satisfies private clue: {row.get('satisfies_private_clue')}")
            reason = row.get("reason")
            if reason:
                lines.append(f"- rationale: {reason}")
        else:
            lines.append(f"- invalid message: {row.get('error')}")
        before = row.get("listener_memory_before")
        after = row.get("listener_memory_after")
        lines.append(f"- listener memory before: {before if before else []}")
        lines.append(f"- listener memory after: {after if after else []}")
        lines.append("")
    return "\n".join(lines)

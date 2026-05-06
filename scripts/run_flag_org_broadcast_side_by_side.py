#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from nnd.flag_game_broadcast.config import (
    apply_overrides as apply_broadcast_overrides,
    load_broadcast_flag_game_config,
)
from nnd.flag_game_broadcast.runner import run_broadcast_flag_game_batch
from nnd.flag_game.config import (
    apply_overrides as apply_pairwise_overrides,
    load_flag_game_config,
)
from nnd.flag_game.runner import run_flag_game_batch
from nnd.flag_game_org.config import (
    apply_overrides as apply_org_overrides,
    load_org_flag_game_config,
)
from nnd.flag_game_org.runner import run_org_flag_game_batch


DEFAULT_POOLS = ("stripe_plus_real_triangle_28",)
DEFAULT_ORG_CONFIG = Path("configs/flag_game_org/rectangle_open_highres_m3_v1.yaml")
DEFAULT_BROADCAST_CONFIG = Path("configs/flag_game_broadcast/stripe_expanded_highres_m3_v1.yaml")
DEFAULT_PAIRWISE_CONFIG = Path("configs/flag_game/rectangle_open_highres_m3_v1.yaml")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _pool_slug(value: str) -> str:
    if value == "stripe_expanded_24":
        return "stripe"
    if value == "stripe_plus_real_triangle_28":
        return "full"
    return _slug(value)


def _short_model_label(model: str) -> str:
    if model.startswith("gpt-"):
        return model.removeprefix("gpt-")
    return model


def _balanced_model_counts(n_agents: int) -> tuple[int, int]:
    prestige_count = n_agents // 2
    comparison_count = n_agents - prestige_count
    return prestige_count, comparison_count


def _balanced_population_models(comparison_model: str, prestige_model: str, *, n_agents: int) -> list[str]:
    prestige_count, comparison_count = _balanced_model_counts(n_agents)
    return [prestige_model] * prestige_count + [comparison_model] * comparison_count


def _balanced_population_condition_name(comparison_model: str, prestige_model: str, *, n_agents: int) -> str:
    prestige_count, comparison_count = _balanced_model_counts(n_agents)
    return f"balanced_{prestige_count}_{_slug(prestige_model)}_{comparison_count}_{_slug(comparison_model)}"


def _balanced_population_condition_label(comparison_model: str, prestige_model: str, *, n_agents: int) -> str:
    prestige_count, comparison_count = _balanced_model_counts(n_agents)
    return (
        f"{prestige_count} {_short_model_label(prestige_model)} + "
        f"{comparison_count} {_short_model_label(comparison_model)} agents"
    )


def _role_slot_conditions(
    comparison_model: str,
    prestige_model: str,
    *,
    n_agents: int,
) -> list[dict[str, Any]]:
    comparison_slug = _slug(comparison_model)
    prestige_slug = _slug(prestige_model)
    conditions = [
        {
            "condition_name": f"all_{comparison_slug}",
            "condition_label": f"All {comparison_model}",
            "condition_order": 0,
            "org_condition_name": f"all_{comparison_slug}",
            "broadcast_agent_models": [comparison_model] * n_agents,
            "slot0_model": comparison_model,
            "rest_model": comparison_model,
            "condition_kind": "pure_comparison",
        },
        {
            "condition_name": f"all_{prestige_slug}",
            "condition_label": f"All {prestige_model}",
            "condition_order": 1,
            "org_condition_name": f"all_{prestige_slug}",
            "broadcast_agent_models": [prestige_model] * n_agents,
            "slot0_model": prestige_model,
            "rest_model": prestige_model,
            "condition_kind": "pure_prestige",
        },
    ]
    balanced_name = _balanced_population_condition_name(
        comparison_model,
        prestige_model,
        n_agents=n_agents,
    )
    balanced_models = _balanced_population_models(
        comparison_model,
        prestige_model,
        n_agents=n_agents,
    )
    conditions.append(
        {
            "condition_name": balanced_name,
            "condition_label": _balanced_population_condition_label(
                comparison_model,
                prestige_model,
                n_agents=n_agents,
            ),
            "condition_order": 2,
            "org_condition_name": "",
            "broadcast_agent_models": balanced_models,
            "slot0_model": balanced_models[0],
            "rest_model": "mixed",
            "condition_kind": "balanced_population",
        }
    )
    return conditions


def _org_condition_name_for_manager(
    *,
    manager_model: str,
    comparison_model: str,
    prestige_model: str,
    n_observers: int,
) -> str:
    prestige_count, comparison_count = _balanced_model_counts(n_observers)
    return (
        f"manager_{_slug(manager_model)}_observers_"
        f"{prestige_count}_{_slug(prestige_model)}_{comparison_count}_{_slug(comparison_model)}"
    )


def _paired_protocol_conditions(
    comparison_model: str,
    prestige_model: str,
    *,
    n_agents: int,
) -> list[dict[str, Any]]:
    comparison_slug = _slug(comparison_model)
    prestige_slug = _slug(prestige_model)
    balanced_name = _balanced_population_condition_name(
        comparison_model,
        prestige_model,
        n_agents=n_agents,
    )
    balanced_label = _balanced_population_condition_label(
        comparison_model,
        prestige_model,
        n_agents=n_agents,
    )
    population_conditions = {
        f"all_{comparison_slug}": [comparison_model] * n_agents,
        f"all_{prestige_slug}": [prestige_model] * n_agents,
        balanced_name: _balanced_population_models(
            comparison_model,
            prestige_model,
            n_agents=n_agents,
        ),
    }
    condition_specs = [
        {
            "condition_name": f"observers_{comparison_slug}_manager_{comparison_slug}",
            "condition_label": f"{_short_model_label(comparison_model)} manager, {_short_model_label(comparison_model)} observers",
            "condition_order": 0,
            "org_condition_name": f"all_{comparison_slug}",
            "population_condition_name": f"all_{comparison_slug}",
            "condition_kind": "comparison_observers_comparison_manager",
            "manager_model": comparison_model,
        },
        {
            "condition_name": f"observers_{comparison_slug}_manager_{prestige_slug}",
            "condition_label": f"{_short_model_label(prestige_model)} manager, {_short_model_label(comparison_model)} observers",
            "condition_order": 1,
            "org_condition_name": f"manager_{prestige_slug}_observers_{comparison_slug}",
            "population_condition_name": f"all_{comparison_slug}",
            "condition_kind": "comparison_observers_prestige_manager",
            "manager_model": prestige_model,
        },
        {
            "condition_name": f"observers_{prestige_slug}_manager_{prestige_slug}",
            "condition_label": f"{_short_model_label(prestige_model)} manager, {_short_model_label(prestige_model)} observers",
            "condition_order": 2,
            "org_condition_name": f"all_{prestige_slug}",
            "population_condition_name": f"all_{prestige_slug}",
            "condition_kind": "prestige_observers_prestige_manager",
            "manager_model": prestige_model,
        },
        {
            "condition_name": f"observers_{prestige_slug}_manager_{comparison_slug}",
            "condition_label": f"{_short_model_label(comparison_model)} manager, {_short_model_label(prestige_model)} observers",
            "condition_order": 3,
            "org_condition_name": f"manager_{comparison_slug}_observers_{prestige_slug}",
            "population_condition_name": f"all_{prestige_slug}",
            "condition_kind": "prestige_observers_comparison_manager",
            "manager_model": comparison_model,
        },
        {
            "condition_name": f"observers_{balanced_name}_manager_{comparison_slug}",
            "condition_label": f"{_short_model_label(comparison_model)} manager, {balanced_label}",
            "condition_order": 4,
            "org_condition_name": _org_condition_name_for_manager(
                manager_model=comparison_model,
                comparison_model=comparison_model,
                prestige_model=prestige_model,
                n_observers=n_agents,
            ),
            "population_condition_name": balanced_name,
            "condition_kind": "balanced_observers_comparison_manager",
            "manager_model": comparison_model,
        },
        {
            "condition_name": f"observers_{balanced_name}_manager_{prestige_slug}",
            "condition_label": f"{_short_model_label(prestige_model)} manager, {balanced_label}",
            "condition_order": 5,
            "org_condition_name": _org_condition_name_for_manager(
                manager_model=prestige_model,
                comparison_model=comparison_model,
                prestige_model=prestige_model,
                n_observers=n_agents,
            ),
            "population_condition_name": balanced_name,
            "condition_kind": "balanced_observers_prestige_manager",
            "manager_model": prestige_model,
        },
    ]
    conditions: list[dict[str, Any]] = []
    for spec in condition_specs:
        agent_models = population_conditions[str(spec["population_condition_name"])]
        conditions.append(
            {
                **spec,
                "broadcast_agent_models": agent_models,
                "slot0_model": agent_models[0],
                "rest_model": agent_models[1] if len(set(agent_models[1:])) == 1 else "mixed",
            }
        )
    return conditions


def _agent_model_signature(agent_models: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    ordered: list[str] = []
    for model in agent_models:
        if model not in counts:
            ordered.append(model)
            counts[model] = 0
        counts[model] += 1
    if len(ordered) == 1:
        return f"all:{ordered[0]}"
    return " + ".join(f"{model} x{counts[model]}" for model in ordered)


def _build_org_agent_models(
    *,
    n_observers: int,
    aggregator_agent_id: int,
    manager_model: str,
    observer_models: list[str],
) -> list[str]:
    total_agents = n_observers + 1
    if len(observer_models) != n_observers:
        raise ValueError(f"observer_models must have length N={n_observers}, got {len(observer_models)}")
    if aggregator_agent_id < 0 or aggregator_agent_id >= total_agents:
        raise ValueError("aggregator_agent_id must be in [0, n_observers]")
    agent_models = [manager_model for _ in range(total_agents)]
    observer_iter = iter(observer_models)
    for agent_id in range(total_agents):
        if agent_id == aggregator_agent_id:
            agent_models[agent_id] = manager_model
        else:
            agent_models[agent_id] = next(observer_iter)
    return agent_models


def _org_side_by_side_conditions(
    comparison_model: str,
    prestige_model: str,
    *,
    n_observers: int,
) -> list[dict[str, Any]]:
    comparison_slug = _slug(comparison_model)
    prestige_slug = _slug(prestige_model)
    comparison_short = _short_model_label(comparison_model)
    prestige_short = _short_model_label(prestige_model)
    balanced_observers = _balanced_population_models(
        comparison_model,
        prestige_model,
        n_agents=n_observers,
    )
    prestige_count, comparison_count = _balanced_model_counts(n_observers)
    balanced_short = f"{prestige_count} {prestige_short} + {comparison_count} {comparison_short}"
    return [
        {
            "condition_name": f"all_{comparison_slug}",
            "condition_label": f"{comparison_model} manager, {comparison_model} observers",
            "condition_short_label": f"all\n{comparison_short}",
            "condition_order": 0,
            "condition_kind": "all_comparison",
            "manager_model": comparison_model,
            "observer_models": [comparison_model] * n_observers,
        },
        {
            "condition_name": f"all_{prestige_slug}",
            "condition_label": f"{prestige_model} manager, {prestige_model} observers",
            "condition_short_label": f"all\n{prestige_short}",
            "condition_order": 2,
            "condition_kind": "all_prestige",
            "manager_model": prestige_model,
            "observer_models": [prestige_model] * n_observers,
        },
        {
            "condition_name": f"manager_{prestige_slug}_observers_{comparison_slug}",
            "condition_label": f"{prestige_model} manager, {comparison_model} observers",
            "condition_short_label": f"{prestige_short} mgr\n{comparison_short} obs",
            "condition_order": 1,
            "condition_kind": "prestige_manager_comparison_observers",
            "manager_model": prestige_model,
            "observer_models": [comparison_model] * n_observers,
        },
        {
            "condition_name": f"manager_{comparison_slug}_observers_{prestige_slug}",
            "condition_label": f"{comparison_model} manager, {prestige_model} observers",
            "condition_short_label": f"{comparison_short} mgr\n{prestige_short} obs",
            "condition_order": 3,
            "condition_kind": "comparison_manager_prestige_observers",
            "manager_model": comparison_model,
            "observer_models": [prestige_model] * n_observers,
        },
        {
            "condition_name": _org_condition_name_for_manager(
                manager_model=comparison_model,
                comparison_model=comparison_model,
                prestige_model=prestige_model,
                n_observers=n_observers,
            ),
            "condition_label": f"{comparison_model} manager, {balanced_short} observers",
            "condition_short_label": f"{comparison_short} mgr\n{balanced_short} obs",
            "condition_order": 4,
            "condition_kind": "comparison_manager_balanced_observers",
            "manager_model": comparison_model,
            "observer_models": balanced_observers,
        },
        {
            "condition_name": _org_condition_name_for_manager(
                manager_model=prestige_model,
                comparison_model=comparison_model,
                prestige_model=prestige_model,
                n_observers=n_observers,
            ),
            "condition_label": f"{prestige_model} manager, {balanced_short} observers",
            "condition_short_label": f"{prestige_short} mgr\n{balanced_short} obs",
            "condition_order": 5,
            "condition_kind": "prestige_manager_balanced_observers",
            "manager_model": prestige_model,
            "observer_models": balanced_observers,
        },
    ]


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _condition_complete(rows: list[dict[str, Any]], condition_name: str, seeds: list[int]) -> bool:
    present = {
        int(row["seed"])
        for row in rows
        if row.get("condition_name") == condition_name and not pd.isna(row.get("seed"))
    }
    return all(seed in present for seed in seeds)


def _drop_condition_seed_rows(
    rows: list[dict[str, Any]],
    *,
    condition_name: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    seed_set = set(seeds)
    kept: list[dict[str, Any]] = []
    for row in rows:
        row_seed = row.get("seed")
        if (
            row.get("condition_name") == condition_name
            and row_seed is not None
            and not pd.isna(row_seed)
            and int(row_seed) in seed_set
        ):
            continue
        kept.append(row)
    return kept


def _aggregate_condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    metrics = [
        "initial_accuracy",
        "final_accuracy",
        "collaboration_gain_over_initial_accuracy",
        "final_consensus_correct",
        "oracle_accuracy",
        "oracle_gap_over_final_accuracy",
        "executed_rounds",
        "estimated_cost_usd",
    ]
    summary_rows: list[dict[str, Any]] = []
    for condition_name, group in frame.groupby("condition_name", sort=False):
        first = group.sort_values("condition_order").iloc[0]
        row: dict[str, Any] = {
            "condition_name": condition_name,
            "condition_label": first.get("condition_label"),
            "condition_order": int(first.get("condition_order", 0)),
            "condition_kind": first.get("condition_kind"),
            "slot0_model": first.get("slot0_model"),
            "rest_model": first.get("rest_model"),
            "comparison_model": first.get("comparison_model"),
            "prestige_model": first.get("prestige_model"),
            "n_trials": int(len(group)),
        }
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return sorted(summary_rows, key=lambda item: int(item["condition_order"]))


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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate_org_condition_rows(condition_df: pd.DataFrame) -> list[dict[str, Any]]:
    if condition_df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, group in condition_df.groupby("condition_name", sort=False):
        first = group.sort_values("condition_order").iloc[0]
        final_correct = group["final_consensus_correct"].map(_coerce_bool)
        rows.append(
            {
                "condition_name": first["condition_name"],
                "condition_label": first["condition_label"],
                "condition_short_label": first["condition_short_label"],
                "condition_order": int(first["condition_order"]),
                "condition_kind": first["condition_kind"],
                "condition_dir": first["condition_dir"],
                "manager_model": first["manager_model"],
                "observer_model": first["observer_model"],
                "observer_model_signature": first.get("observer_model_signature", first.get("observer_model")),
                "comparison_model": first["comparison_model"],
                "prestige_model": first["prestige_model"],
                "manager_is_prestige": _coerce_bool(first["manager_is_prestige"]),
                "observer_prestige_count": _safe_int(first.get("observer_prestige_count"), 0),
                "observer_comparison_count": _safe_int(first.get("observer_comparison_count"), 0),
                "observer_prestige_fraction": _safe_float(first.get("observer_prestige_fraction"), 0.0),
                "agent_model_signature": first["agent_model_signature"],
                "n_trials": int(len(group)),
                "seed_min": int(group["seed"].min()) if "seed" in group else None,
                "seed_max": int(group["seed"].max()) if "seed" in group else None,
                "initial_accuracy_mean": _series_mean(group, "initial_accuracy"),
                "initial_accuracy_std": _series_std(group, "initial_accuracy"),
                "final_accuracy_mean": _series_mean(group, "final_accuracy"),
                "final_accuracy_std": _series_std(group, "final_accuracy"),
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
    return sorted(rows, key=lambda item: int(item["condition_order"]))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_org_role_mix_report(
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
        "- balanced observer conditions assign the first half of observer crop slots to the prestige model and the second half to the comparison model",
        "",
        "| condition | manager | observers | n | IIQ | CIQ manager correct | CIQ - IIQ | oracle gap |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['condition_label']} | "
            f"{row['manager_model']} | "
            f"{row['observer_model_signature']} | "
            f"{row['n_trials']} | "
            f"{_fmt(row.get('initial_accuracy_mean'))} | "
            f"{_fmt(row.get('manager_correct_rate'))} | "
            f"{_fmt(row.get('collaboration_gain_mean'))} | "
            f"{_fmt(row.get('oracle_gap_mean'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _run_org_side_by_side_comparison(
    base_config: Any,
    *,
    out_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _org_side_by_side_conditions(
        comparison_model,
        prestige_model,
        n_observers=base_config.N,
    )
    condition_rows = _read_records(out_dir / "role_mix_condition_results.csv")
    skipped_conditions: list[str] = []

    for condition in conditions:
        condition_name = str(condition["condition_name"])
        manager_model = str(condition["manager_model"])
        observer_models = list(condition["observer_models"])
        observer_signature = _agent_model_signature(observer_models)
        observer_model = observer_models[0] if len(set(observer_models)) == 1 else "mixed"
        observer_prestige_count = sum(1 for model in observer_models if model == prestige_model)
        observer_comparison_count = sum(1 for model in observer_models if model == comparison_model)
        agent_models = _build_org_agent_models(
            n_observers=base_config.N,
            aggregator_agent_id=base_config.aggregator_agent_id,
            manager_model=manager_model,
            observer_models=observer_models,
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

        condition_rows = _drop_condition_seed_rows(
            condition_rows,
            condition_name=condition_name,
            seeds=seeds,
        )
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
                    "observer_model_signature": observer_signature,
                    "comparison_model": comparison_model,
                    "prestige_model": prestige_model,
                    "manager_is_prestige": manager_model == prestige_model,
                    "observer_is_prestige": observer_model == prestige_model,
                    "observer_prestige_count": observer_prestige_count,
                    "observer_comparison_count": observer_comparison_count,
                    "observer_prestige_fraction": observer_prestige_count / float(base_config.N),
                    "agent_model_signature": _agent_model_signature(agent_models),
                }
            )
            condition_rows.append(row)

    condition_df = pd.DataFrame(condition_rows)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["condition_order", "seed"]).reset_index(drop=True)
    summary_rows = _aggregate_org_condition_rows(condition_df)
    summary_df = pd.DataFrame(summary_rows)
    condition_df.to_csv(out_dir / "role_mix_condition_results.csv", index=False)
    summary_df.to_csv(out_dir / "role_mix_summary.csv", index=False)
    _write_org_role_mix_report(
        out_dir / "role_mix_report.md",
        summary_rows=summary_rows,
        comparison_model=comparison_model,
        prestige_model=prestige_model,
    )
    return {
        "condition_results": condition_df.to_dict(orient="records"),
        "summary": summary_rows,
        "skipped_conditions": skipped_conditions,
    }


def _write_broadcast_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Broadcast Population-Mix Pilot",
        "",
        "This is the broadcast half of the org-vs-broadcast side-by-side pilot.",
        "Every broadcast agent is crop-bearing; conditions match the org observer population, not the crop-blind manager.",
        "",
        "| condition | n | initial | final | consensus | oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {n} | {initial:.3f} | {final:.3f} | {consensus:.3f} | {oracle:.3f} |".format(
                condition=row.get("condition_label") or row["condition_name"],
                n=int(row.get("n_trials", 0)),
                initial=float(row.get("initial_accuracy_mean", float("nan"))),
                final=float(row.get("final_accuracy_mean", float("nan"))),
                consensus=float(row.get("final_consensus_correct_mean", float("nan"))),
                oracle=float(row.get("oracle_accuracy_mean", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _run_broadcast_role_slot_comparison(
    base_config: Any,
    *,
    out_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _role_slot_conditions(
        comparison_model,
        prestige_model,
        n_agents=base_config.N,
    )
    condition_rows = _read_records(out_dir / "broadcast_role_slot_condition_results.csv")
    skipped_conditions: list[str] = []

    for condition in conditions:
        condition_name = str(condition["condition_name"])
        agent_models = list(condition["broadcast_agent_models"])
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

        condition_rows = _drop_condition_seed_rows(
            condition_rows,
            condition_name=condition_name,
            seeds=seeds,
        )
        results = run_broadcast_flag_game_batch(condition_config, out_dir=condition_dir, seeds=seeds)
        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            row.update(
                {
                    "seed": seed,
                    "condition_name": condition_name,
                    "condition_label": condition["condition_label"],
                    "condition_order": condition["condition_order"],
                    "condition_kind": condition["condition_kind"],
                    "condition_dir": str(condition_dir),
                    "slot0_model": condition["slot0_model"],
                    "rest_model": condition["rest_model"],
                    "comparison_model": comparison_model,
                    "prestige_model": prestige_model,
                    "slot0_is_prestige": condition["slot0_model"] == prestige_model,
                    "rest_is_prestige": condition["rest_model"] == prestige_model,
                    "agent_model_signature": _agent_model_signature(agent_models),
                }
            )
            condition_rows.append(row)

    condition_df = pd.DataFrame(condition_rows)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["condition_order", "seed"]).reset_index(drop=True)
    summary_rows = _aggregate_condition_rows(condition_df.to_dict(orient="records"))
    summary_df = pd.DataFrame(summary_rows)
    condition_df.to_csv(out_dir / "broadcast_role_slot_condition_results.csv", index=False)
    summary_df.to_csv(out_dir / "broadcast_role_slot_summary.csv", index=False)
    _write_broadcast_report(out_dir / "broadcast_role_slot_report.md", summary_rows)
    return {
        "condition_results": condition_df.to_dict(orient="records"),
        "summary": summary_rows,
        "skipped_conditions": skipped_conditions,
    }


def _write_pairwise_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pairwise Population-Mix Pilot",
        "",
        "This is the pairwise half of the protocol side-by-side pilot.",
        "Every pairwise agent is crop-bearing; conditions match the org observer population, not the crop-blind manager.",
        "",
        "| condition | n | initial | final | consensus | oracle |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {n} | {initial:.3f} | {final:.3f} | {consensus:.3f} | {oracle:.3f} |".format(
                condition=row.get("condition_label") or row["condition_name"],
                n=int(row.get("n_trials", 0)),
                initial=float(row.get("initial_accuracy_mean", float("nan"))),
                final=float(row.get("final_accuracy_mean", float("nan"))),
                consensus=float(row.get("final_consensus_correct_mean", float("nan"))),
                oracle=float(row.get("oracle_accuracy_mean", float("nan"))),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _run_pairwise_role_slot_comparison(
    base_config: Any,
    *,
    out_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _role_slot_conditions(
        comparison_model,
        prestige_model,
        n_agents=base_config.N,
    )
    condition_rows = _read_records(out_dir / "pairwise_role_slot_condition_results.csv")
    skipped_conditions: list[str] = []

    for condition in conditions:
        condition_name = str(condition["condition_name"])
        agent_models = list(condition["broadcast_agent_models"])
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

        condition_rows = _drop_condition_seed_rows(
            condition_rows,
            condition_name=condition_name,
            seeds=seeds,
        )
        results = run_flag_game_batch(condition_config, out_dir=condition_dir, seeds=seeds)
        for seed, result in zip(seeds, results, strict=False):
            row = dict(result["summary"])
            row.update(
                {
                    "seed": seed,
                    "condition_name": condition_name,
                    "condition_label": condition["condition_label"],
                    "condition_order": condition["condition_order"],
                    "condition_kind": condition["condition_kind"],
                    "condition_dir": str(condition_dir),
                    "slot0_model": condition["slot0_model"],
                    "rest_model": condition["rest_model"],
                    "comparison_model": comparison_model,
                    "prestige_model": prestige_model,
                    "slot0_is_prestige": condition["slot0_model"] == prestige_model,
                    "rest_is_prestige": condition["rest_model"] == prestige_model,
                    "agent_model_signature": _agent_model_signature(agent_models),
                }
            )
            condition_rows.append(row)

    condition_df = pd.DataFrame(condition_rows)
    if not condition_df.empty:
        condition_df = condition_df.sort_values(["condition_order", "seed"]).reset_index(drop=True)
    summary_rows = _aggregate_condition_rows(condition_df.to_dict(orient="records"))
    summary_df = pd.DataFrame(summary_rows)
    condition_df.to_csv(out_dir / "pairwise_role_slot_condition_results.csv", index=False)
    summary_df.to_csv(out_dir / "pairwise_role_slot_summary.csv", index=False)
    _write_pairwise_report(out_dir / "pairwise_role_slot_report.md", summary_rows)
    return {
        "condition_results": condition_df.to_dict(orient="records"),
        "summary": summary_rows,
        "skipped_conditions": skipped_conditions,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _crop_signature(manifest: dict[str, Any], *, protocol: str) -> list[tuple[int, int, int, int, int]]:
    assignments = manifest.get("assignments") or []
    if protocol == "org":
        assignments = sorted(assignments, key=lambda item: int(item["agent_id"]))
    else:
        assignments = sorted(assignments, key=lambda item: int(item["agent_id"]))
    signature: list[tuple[int, int, int, int, int]] = []
    for item in assignments:
        signature.append(
            (
                int(item.get("crop_index", len(signature))),
                int(item["top"]),
                int(item["left"]),
                int(item["height"]),
                int(item["width"]),
            )
        )
    return signature


def _protocol_trial_path(pool_dir: Path, protocol: str, condition: dict[str, Any], seed: int) -> Path:
    if protocol == "org":
        condition_name = str(condition["org_condition_name"])
    else:
        condition_name = str(condition.get("population_condition_name") or condition["condition_name"])
    return pool_dir / protocol / condition_name / f"seed_{seed:04d}"


def _compare_pool_trials(
    *,
    pool: str,
    pool_dir: Path,
    seeds: list[int],
    comparison_model: str,
    prestige_model: str,
    n_agents: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in _paired_protocol_conditions(
        comparison_model,
        prestige_model,
        n_agents=n_agents,
    ):
        for seed in seeds:
            org_path = _protocol_trial_path(pool_dir, "org", condition, seed)
            broadcast_path = _protocol_trial_path(pool_dir, "broadcast", condition, seed)
            pairwise_path = _protocol_trial_path(pool_dir, "pairwise", condition, seed)
            org_summary_path = org_path / "summary.json"
            broadcast_summary_path = broadcast_path / "summary.json"
            pairwise_summary_path = pairwise_path / "summary.json"
            org_manifest_path = org_path / "trial_manifest.json"
            broadcast_manifest_path = broadcast_path / "trial_manifest.json"
            pairwise_manifest_path = pairwise_path / "trial_manifest.json"

            row: dict[str, Any] = {
                "pool": pool,
                "pool_label": "Full flag set" if pool == "stripe_plus_real_triangle_28" else "Stripe-only flag set",
                "seed": seed,
                "condition_name": condition["condition_name"],
                "condition_label": condition["condition_label"],
                "condition_order": condition["condition_order"],
                "org_condition_name": condition["org_condition_name"],
                "population_condition_name": condition.get("population_condition_name", condition["condition_name"]),
                "org_trial_dir": str(org_path),
                "broadcast_trial_dir": str(broadcast_path),
                "pairwise_trial_dir": str(pairwise_path),
                "complete": all(
                    path.exists()
                    for path in (
                        org_summary_path,
                        broadcast_summary_path,
                        pairwise_summary_path,
                        org_manifest_path,
                        broadcast_manifest_path,
                        pairwise_manifest_path,
                    )
                ),
            }
            if not row["complete"]:
                rows.append(row)
                continue

            org_summary = _load_json(org_summary_path)
            broadcast_summary = _load_json(broadcast_summary_path)
            pairwise_summary = _load_json(pairwise_summary_path)
            org_manifest = _load_json(org_manifest_path)
            broadcast_manifest = _load_json(broadcast_manifest_path)
            pairwise_manifest = _load_json(pairwise_manifest_path)
            org_crops = _crop_signature(org_manifest, protocol="org")
            broadcast_crops = _crop_signature(broadcast_manifest, protocol="broadcast")
            pairwise_crops = _crop_signature(pairwise_manifest, protocol="pairwise")
            truth_match = (
                org_manifest.get("truth_country")
                == broadcast_manifest.get("truth_country")
                == pairwise_manifest.get("truth_country")
            )
            broadcast_crop_match = org_crops == broadcast_crops
            pairwise_crop_match = org_crops == pairwise_crops
            crop_match = broadcast_crop_match and pairwise_crop_match

            row.update(
                {
                    "truth_match": truth_match,
                    "crop_match": crop_match,
                    "broadcast_crop_match": broadcast_crop_match,
                    "pairwise_crop_match": pairwise_crop_match,
                    "org_truth_country": org_manifest.get("truth_country"),
                    "broadcast_truth_country": broadcast_manifest.get("truth_country"),
                    "pairwise_truth_country": pairwise_manifest.get("truth_country"),
                    "org_initial_accuracy": org_summary.get("initial_accuracy"),
                    "broadcast_initial_accuracy": broadcast_summary.get("initial_accuracy"),
                    "pairwise_initial_accuracy": pairwise_summary.get("initial_accuracy"),
                    "initial_accuracy_delta_org_minus_broadcast": _maybe_delta(
                        org_summary.get("initial_accuracy"),
                        broadcast_summary.get("initial_accuracy"),
                    ),
                    "initial_accuracy_delta_org_minus_pairwise": _maybe_delta(
                        org_summary.get("initial_accuracy"),
                        pairwise_summary.get("initial_accuracy"),
                    ),
                    "org_final_accuracy": org_summary.get("final_accuracy"),
                    "broadcast_final_accuracy": broadcast_summary.get("final_accuracy"),
                    "pairwise_final_accuracy": pairwise_summary.get("final_accuracy"),
                    "final_accuracy_delta_org_minus_broadcast": _maybe_delta(
                        org_summary.get("final_accuracy"),
                        broadcast_summary.get("final_accuracy"),
                    ),
                    "final_accuracy_delta_org_minus_pairwise": _maybe_delta(
                        org_summary.get("final_accuracy"),
                        pairwise_summary.get("final_accuracy"),
                    ),
                    "final_accuracy_delta_broadcast_minus_pairwise": _maybe_delta(
                        broadcast_summary.get("final_accuracy"),
                        pairwise_summary.get("final_accuracy"),
                    ),
                    "org_final_consensus_correct": org_summary.get("final_consensus_correct"),
                    "broadcast_final_consensus_correct": broadcast_summary.get("final_consensus_correct"),
                    "pairwise_final_consensus_correct": pairwise_summary.get("final_consensus_correct"),
                    "final_consensus_delta_org_minus_broadcast": _maybe_delta(
                        org_summary.get("final_consensus_correct"),
                        broadcast_summary.get("final_consensus_correct"),
                    ),
                    "final_consensus_delta_org_minus_pairwise": _maybe_delta(
                        org_summary.get("final_consensus_correct"),
                        pairwise_summary.get("final_consensus_correct"),
                    ),
                    "org_final_outcome": org_summary.get("final_outcome"),
                    "broadcast_final_outcome": broadcast_summary.get("final_outcome"),
                    "pairwise_final_outcome": pairwise_summary.get("final_outcome"),
                    "org_oracle_accuracy": org_summary.get("oracle_accuracy"),
                    "broadcast_oracle_accuracy": broadcast_summary.get("oracle_accuracy"),
                    "pairwise_oracle_accuracy": pairwise_summary.get("oracle_accuracy"),
                    "org_executed_rounds": org_summary.get("executed_rounds"),
                    "broadcast_executed_rounds": broadcast_summary.get("executed_rounds"),
                    "pairwise_executed_T": pairwise_summary.get("executed_T"),
                    "org_estimated_cost_usd": org_summary.get("estimated_cost_usd"),
                    "broadcast_estimated_cost_usd": broadcast_summary.get("estimated_cost_usd"),
                    "pairwise_estimated_cost_usd": pairwise_summary.get("estimated_cost_usd"),
                }
            )
            rows.append(row)
    return rows


def _maybe_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _write_paired_readme(path: Path, *, pools: list[str], seeds: list[int], args: argparse.Namespace) -> None:
    prestige_count, comparison_count = _balanced_model_counts(args.n)
    lines = [
        "# Flag Game Protocol Side-by-Side Pilot",
        "",
        "This run pairs the org manager protocol, broadcast protocol, and pairwise protocol on the same seeds and crop geometry.",
        "For each seed, org observer 1..N is compared to broadcast agent 0..N-1, so the observer crop order is identical when `crop_match=true`.",
        "Pairwise agent 0..N-1 uses the same crop order as broadcast.",
        "",
        "## Design",
        "",
        f"- N: `{args.n}`",
        f"- m: `{args.interaction_m}`",
        f"- rounds: `{args.rounds}`",
        f"- pairwise T: `{args.pairwise_t or args.rounds * args.n}`",
        f"- pairwise probe_every: `{args.pairwise_probe_every or args.n}`",
        f"- H: `{args.memory_capacity}`",
        f"- seeds: `{min(seeds)}..{max(seeds)}`",
        f"- pools: `{', '.join(pools)}`",
        f"- comparison model: `{args.comparison_model}`",
        f"- prestige model: `{args.prestige_model}`",
        f"- social susceptibility prompt: `{args.prompt_social_susceptibility}`",
        "",
        "## Conditions",
        "",
        "| org | broadcast analogue | pairwise analogue |",
        "| --- | --- | --- |",
        f"| {args.comparison_model} manager, {args.comparison_model} observers | all {args.comparison_model} | all {args.comparison_model} |",
        f"| {args.prestige_model} manager, {args.comparison_model} observers | all {args.comparison_model} | all {args.comparison_model} |",
        f"| {args.prestige_model} manager, {args.prestige_model} observers | all {args.prestige_model} | all {args.prestige_model} |",
        f"| {args.comparison_model} manager, {args.prestige_model} observers | all {args.prestige_model} | all {args.prestige_model} |",
        f"| {args.comparison_model} manager, {prestige_count} {args.prestige_model} + {comparison_count} {args.comparison_model} observers | {prestige_count} {args.prestige_model} + {comparison_count} {args.comparison_model} agents | same as broadcast |",
        f"| {args.prestige_model} manager, {prestige_count} {args.prestige_model} + {comparison_count} {args.comparison_model} observers | {prestige_count} {args.prestige_model} + {comparison_count} {args.comparison_model} agents | same as broadcast |",
        "",
        "Broadcast and pairwise agents all receive crops. Org adds one crop-blind manager, so manager-model variants reuse the same broadcast/pairwise observer-population trial.",
        "",
        "## Outputs",
        "",
        "- `paired_summary_all_pools.csv`: seed-level paired protocol comparison.",
        "- `<pool>/paired_summary.csv`: pool-specific paired comparison.",
        "- `<pool>/org/`: org role-mix outputs, including balanced-observer manager variants.",
        "- `<pool>/broadcast/`: exact broadcast population-mix outputs.",
        "- `<pool>/pairwise/`: exact pairwise population-mix outputs.",
        "- `side_by_side_final_accuracy.png/pdf` and `side_by_side_delta_heatmap.png/pdf`: pilot visual summaries when visuals are enabled.",
    ]
    path.write_text("\n".join(lines) + "\n")


def _make_visuals(summary_csv: Path, out_dir: Path) -> None:
    frame = pd.read_csv(summary_csv)
    frame = frame[frame["complete"] == True].copy()  # noqa: E712
    if frame.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    condition_rows = (
        frame[["condition_order", "condition_label"]]
        .drop_duplicates()
        .sort_values("condition_order")
        .to_dict(orient="records")
    )
    pool_rows = (
        frame[["pool", "pool_label"]]
        .drop_duplicates()
        .sort_values("pool")
        .to_dict(orient="records")
    )
    condition_labels = [row["condition_label"] for row in condition_rows]
    x = np.arange(len(condition_labels))
    width = 0.25
    org_color = "#0F766E"
    broadcast_color = "#7C3AED"
    pairwise_color = "#D97706"

    fig, axes = plt.subplots(1, len(pool_rows), figsize=(5.2 * len(pool_rows), 4.2), sharey=True)
    if len(pool_rows) == 1:
        axes = [axes]
    for ax, pool_row in zip(axes, pool_rows, strict=True):
        pool_frame = frame[frame["pool"] == pool_row["pool"]]
        org_values = []
        broadcast_values = []
        pairwise_values = []
        org_err = []
        broadcast_err = []
        pairwise_err = []
        for condition in condition_rows:
            group = pool_frame[pool_frame["condition_order"] == condition["condition_order"]]
            org = pd.to_numeric(group["org_final_accuracy"], errors="coerce").dropna()
            broadcast = pd.to_numeric(group["broadcast_final_accuracy"], errors="coerce").dropna()
            pairwise = pd.to_numeric(group["pairwise_final_accuracy"], errors="coerce").dropna()
            org_values.append(float(org.mean()) if not org.empty else np.nan)
            broadcast_values.append(float(broadcast.mean()) if not broadcast.empty else np.nan)
            pairwise_values.append(float(pairwise.mean()) if not pairwise.empty else np.nan)
            org_err.append(float(org.std(ddof=1) / np.sqrt(len(org))) if len(org) > 1 else 0.0)
            broadcast_err.append(float(broadcast.std(ddof=1) / np.sqrt(len(broadcast))) if len(broadcast) > 1 else 0.0)
            pairwise_err.append(float(pairwise.std(ddof=1) / np.sqrt(len(pairwise))) if len(pairwise) > 1 else 0.0)
        ax.bar(x - width, pairwise_values, width, yerr=pairwise_err, label="Pairwise", color=pairwise_color)
        ax.bar(x, broadcast_values, width, yerr=broadcast_err, label="Broadcast", color=broadcast_color)
        ax.bar(x + width, org_values, width, yerr=org_err, label="Manager", color=org_color)
        ax.set_title(str(pool_row["pool_label"]))
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, rotation=25, ha="right")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Final accuracy")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Pairwise vs Broadcast vs Manager, Same Seeds and Crops", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"side_by_side_final_accuracy.{suffix}", dpi=300)
    plt.close(fig)

    delta_rows: list[dict[str, Any]] = []
    for pool_row in pool_rows:
        pool_frame = frame[frame["pool"] == pool_row["pool"]]
        for contrast, column in [
            ("Manager - Broadcast", "final_accuracy_delta_org_minus_broadcast"),
            ("Manager - Pairwise", "final_accuracy_delta_org_minus_pairwise"),
            ("Broadcast - Pairwise", "final_accuracy_delta_broadcast_minus_pairwise"),
        ]:
            row: dict[str, Any] = {"contrast": f"{pool_row['pool_label']}: {contrast}"}
            for condition in condition_rows:
                group = pool_frame[pool_frame["condition_order"] == condition["condition_order"]]
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                row[condition["condition_label"]] = float(values.mean()) if not values.empty else np.nan
            delta_rows.append(row)
    pivot = pd.DataFrame(delta_rows).set_index("contrast").reindex(columns=condition_labels)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    values = pivot.to_numpy(dtype=float)
    limit = max(0.25, float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.25)
    image = ax.imshow(values, cmap="BrBG", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(condition_labels)))
    ax.set_xticklabels(condition_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
            else:
                label = f"{value:+.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9)
    ax.set_title("Final Accuracy Protocol Deltas")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03, label="Delta")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"side_by_side_delta_heatmap.{suffix}", dpi=300)
    plt.close(fig)


def _build_org_config(args: argparse.Namespace, pool: str) -> Any:
    config = load_org_flag_game_config(args.org_config)
    overrides = [
        f"N={args.n}",
        f"rounds={args.rounds}",
        f"H={args.memory_capacity}",
        f"interaction_m={args.interaction_m}",
        f"country_pool={pool}",
        f"early_stop_round_window={args.early_stop_window}",
        f"canvas_width={args.canvas_width}",
        f"canvas_height={args.canvas_height}",
        f"tile_width={args.tile_width}",
        f"tile_height={args.tile_height}",
        f"render_scale={args.render_scale}",
        f"image_detail={args.image_detail}",
        f"output.save_crop_images={str(args.save_crop_images).lower()}",
        f"output.make_plots={str(args.trial_plots).lower()}",
    ]
    if args.fixed_truth_country:
        overrides.append(f"fixed_truth_country={args.fixed_truth_country}")
    config = apply_org_overrides(config, overrides)
    return config.model_copy(
        update={
            "backend": args.backend,
            "agent_workers": args.agent_workers,
            "seed_workers": args.seed_workers,
        }
    )


def _build_broadcast_config(args: argparse.Namespace, pool: str) -> Any:
    config = load_broadcast_flag_game_config(args.broadcast_config)
    overrides = [
        f"N={args.n}",
        f"rounds={args.rounds}",
        f"H={args.memory_capacity}",
        f"interaction_m={args.interaction_m}",
        f"country_pool={pool}",
        f"early_stop_round_window={args.early_stop_window}",
        f"canvas_width={args.canvas_width}",
        f"canvas_height={args.canvas_height}",
        f"tile_width={args.tile_width}",
        f"tile_height={args.tile_height}",
        f"render_scale={args.render_scale}",
        f"image_detail={args.image_detail}",
        f"social_susceptibility={args.social_susceptibility}",
        f"prompt_social_susceptibility={str(args.prompt_social_susceptibility).lower()}",
        f"max_influential_agents={args.max_influential_agents}",
        f"output.save_crop_images={str(args.save_crop_images).lower()}",
        f"output.make_plots={str(args.trial_plots).lower()}",
    ]
    if args.fixed_truth_country:
        overrides.append(f"fixed_truth_country={args.fixed_truth_country}")
    config = apply_broadcast_overrides(config, overrides)
    return config.model_copy(
        update={
            "backend": args.backend,
            "agent_workers": args.agent_workers,
            "seed_workers": args.seed_workers,
            "comparison_model_label": args.comparison_model,
            "prestige_model_label": args.prestige_model,
        }
    )


def _build_pairwise_config(args: argparse.Namespace, pool: str) -> Any:
    config = load_flag_game_config(args.pairwise_config)
    pairwise_t = args.pairwise_t if args.pairwise_t is not None else args.rounds * args.n
    pairwise_probe_every = args.pairwise_probe_every if args.pairwise_probe_every is not None else args.n
    overrides = [
        f"N={args.n}",
        f"T={pairwise_t}",
        f"H={args.memory_capacity}",
        f"interaction_m={args.interaction_m}",
        f"country_pool={pool}",
        f"early_stop_probe_window={args.early_stop_window}",
        f"probe_every={pairwise_probe_every}",
        f"canvas_width={args.canvas_width}",
        f"canvas_height={args.canvas_height}",
        f"tile_width={args.tile_width}",
        f"tile_height={args.tile_height}",
        f"render_scale={args.render_scale}",
        f"image_detail={args.image_detail}",
        f"social_susceptibility={args.social_susceptibility}",
        f"prompt_social_susceptibility={str(args.prompt_social_susceptibility).lower()}",
        f"output.save_crop_images={str(args.save_crop_images).lower()}",
        f"output.make_plots={str(args.trial_plots).lower()}",
    ]
    if args.fixed_truth_country:
        overrides.append(f"fixed_truth_country={args.fixed_truth_country}")
    config = apply_pairwise_overrides(config, overrides)
    return config.model_copy(
        update={
            "backend": args.backend,
            "probe_workers": args.agent_workers,
            "seed_workers": args.seed_workers,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an apples-to-apples pairwise vs broadcast vs org-manager Flag Game pilot."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/flag_game_side_by_side/protocol_side_by_side_N8_m3_full_10seed"),
    )
    parser.add_argument("--org-config", type=Path, default=DEFAULT_ORG_CONFIG)
    parser.add_argument("--broadcast-config", type=Path, default=DEFAULT_BROADCAST_CONFIG)
    parser.add_argument("--pairwise-config", type=Path, default=DEFAULT_PAIRWISE_CONFIG)
    parser.add_argument("--pool", action="append", default=None, help="Country pool to include. Repeatable.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--backend", choices=("openai", "scripted"), default="openai")
    parser.add_argument("--comparison-model", default="gpt-4o")
    parser.add_argument("--prestige-model", default="gpt-5.4")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--pairwise-t", type=int, default=None, help="Pairwise interaction budget. Defaults to rounds * N.")
    parser.add_argument("--pairwise-probe-every", type=int, default=None, help="Pairwise probe cadence. Defaults to N.")
    parser.add_argument("--memory-capacity", type=int, default=8)
    parser.add_argument("--interaction-m", type=int, default=3)
    parser.add_argument("--canvas-width", type=int, default=24)
    parser.add_argument("--canvas-height", type=int, default=16)
    parser.add_argument("--tile-width", type=int, default=6)
    parser.add_argument("--tile-height", type=int, default=4)
    parser.add_argument("--render-scale", type=int, default=25)
    parser.add_argument("--image-detail", default="high")
    parser.add_argument("--social-susceptibility", type=float, default=0.5)
    parser.add_argument("--prompt-social-susceptibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stop-window", type=int, default=3)
    parser.add_argument("--max-influential-agents", type=int, default=3)
    parser.add_argument("--fixed-truth-country", default=None)
    parser.add_argument("--agent-workers", type=int, default=4)
    parser.add_argument("--seed-workers", type=int, default=1)
    parser.add_argument("--save-crop-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trial-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-crop-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = args.pool or list(DEFAULT_POOLS)
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    if args.backend == "openai" and args.run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it, or use --backend scripted for a plumbing smoke test."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for pool in pools:
        pool_dir = args.out / _pool_slug(pool)
        pool_dir.mkdir(parents=True, exist_ok=True)
        org_config = _build_org_config(args, pool)
        broadcast_config = _build_broadcast_config(args, pool)
        pairwise_config = _build_pairwise_config(args, pool)

        if args.run:
            print(f"Running org role-mix for pool={pool} seeds={seeds}")
            _run_org_side_by_side_comparison(
                org_config,
                out_dir=pool_dir / "org",
                seeds=seeds,
                comparison_model=args.comparison_model,
                prestige_model=args.prestige_model,
            )
            print(f"Running broadcast population mix for pool={pool} seeds={seeds}")
            _run_broadcast_role_slot_comparison(
                broadcast_config,
                out_dir=pool_dir / "broadcast",
                seeds=seeds,
                comparison_model=args.comparison_model,
                prestige_model=args.prestige_model,
            )
            print(f"Running pairwise population mix for pool={pool} seeds={seeds}")
            _run_pairwise_role_slot_comparison(
                pairwise_config,
                out_dir=pool_dir / "pairwise",
                seeds=seeds,
                comparison_model=args.comparison_model,
                prestige_model=args.prestige_model,
            )

        rows = _compare_pool_trials(
            pool=pool,
            pool_dir=pool_dir,
            seeds=seeds,
            comparison_model=args.comparison_model,
            prestige_model=args.prestige_model,
            n_agents=args.n,
        )
        pool_frame = pd.DataFrame(rows)
        pool_frame.to_csv(pool_dir / "paired_summary.csv", index=False)
        (pool_dir / "paired_summary.json").write_text(json.dumps(rows, indent=2))
        all_rows.extend(rows)

    all_frame = pd.DataFrame(all_rows)
    all_csv = args.out / "paired_summary_all_pools.csv"
    all_frame.to_csv(all_csv, index=False)
    (args.out / "paired_summary_all_pools.json").write_text(json.dumps(all_rows, indent=2))
    _write_paired_readme(args.out / "README.md", pools=list(pools), seeds=seeds, args=args)

    complete = all_frame[all_frame.get("complete", False) == True] if not all_frame.empty else all_frame  # noqa: E712
    if not complete.empty:
        crop_mismatches = complete[complete["crop_match"] != True]  # noqa: E712
        truth_mismatches = complete[complete["truth_match"] != True]  # noqa: E712
        if args.make_visuals:
            _make_visuals(all_csv, args.out)
        if (not crop_mismatches.empty or not truth_mismatches.empty) and not args.allow_crop_mismatch:
            raise SystemExit(
                "Paired run finished, but at least one truth/crop mismatch was detected. "
                "Inspect paired_summary_all_pools.csv."
            )

    print(f"Wrote paired comparison: {all_csv}")
    if args.make_visuals:
        print(f"Wrote visuals under: {args.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any

import pandas as pd

from nnd.flag_game.analysis import classify_probe_distribution


def summarize_org_rounds(
    observer_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    countries: list[str],
    truth_country: str,
    consensus_threshold: float,
    polarization_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not observer_rows or not decision_rows:
        return pd.DataFrame(), {}

    rounds = sorted(
        set(int(row["round"]) for row in observer_rows)
        | set(int(row["round"]) for row in decision_rows)
    )
    per_round_rows: list[dict[str, Any]] = []
    time_to_final_any_consensus: int | None = None
    time_to_final_correct_consensus: int | None = None

    for round_idx in rounds:
        round_observers = [row for row in observer_rows if int(row["round"]) == round_idx]
        round_decisions = [row for row in decision_rows if int(row["round"]) == round_idx]
        initial_summary = _summarize_stage(
            round_observers,
            countries=countries,
            truth_country=truth_country,
            consensus_threshold=consensus_threshold,
            polarization_threshold=polarization_threshold,
            country_field="country",
        )
        final_summary = _summarize_stage(
            round_decisions,
            countries=countries,
            truth_country=truth_country,
            consensus_threshold=consensus_threshold,
            polarization_threshold=polarization_threshold,
            country_field="country",
        )

        if time_to_final_any_consensus is None and final_summary["classification"]["consensus_country"] is not None:
            time_to_final_any_consensus = round_idx
        if time_to_final_correct_consensus is None and final_summary["classification"]["consensus_correct"]:
            time_to_final_correct_consensus = round_idx

        row = {
            "round": round_idx,
            "truth_country": truth_country,
            "initial_accuracy": initial_summary["accuracy"],
            "initial_valid_count": initial_summary["valid_count"],
            "initial_invalid_count": initial_summary["invalid_count"],
            "initial_support_size": initial_summary["support_size"],
            "initial_top1_share": initial_summary["classification"]["top1_share"],
            "initial_top2_share": initial_summary["classification"]["top2_share"],
            "initial_outcome": initial_summary["classification"]["outcome"],
            "initial_consensus_country": initial_summary["classification"]["consensus_country"],
            "initial_consensus_correct": initial_summary["classification"]["consensus_correct"],
            "initial_vote_country": initial_summary["classification"]["top_vote_country"],
            "initial_vote_correct": initial_summary["classification"]["top_vote_correct"],
            "final_accuracy": final_summary["accuracy"],
            "final_valid_count": final_summary["valid_count"],
            "final_invalid_count": final_summary["invalid_count"],
            "final_support_size": final_summary["support_size"],
            "final_top1_share": final_summary["classification"]["top1_share"],
            "final_top2_share": final_summary["classification"]["top2_share"],
            "final_outcome": final_summary["classification"]["outcome"],
            "final_consensus_country": final_summary["classification"]["consensus_country"],
            "final_consensus_correct": final_summary["classification"]["consensus_correct"],
            "final_vote_country": final_summary["classification"]["top_vote_country"],
            "final_vote_correct": final_summary["classification"]["top_vote_correct"],
        }
        for country in countries:
            row[f"initial_share_{country}"] = initial_summary["shares"].get(country, 0.0)
            row[f"final_share_{country}"] = final_summary["shares"].get(country, 0.0)
        per_round_rows.append(row)

    frame = pd.DataFrame(per_round_rows)
    initial_row = frame.iloc[0].to_dict()
    final_row = frame.iloc[-1].to_dict()
    summary = {
        "initial_accuracy": float(initial_row["initial_accuracy"]),
        "final_accuracy": float(final_row["final_accuracy"]),
        "initial_outcome": initial_row["initial_outcome"],
        "final_outcome": final_row["final_outcome"],
        "final_consensus_country": final_row["final_consensus_country"],
        "final_consensus_correct": bool(final_row["final_consensus_correct"]),
        "initial_vote_country": initial_row["initial_vote_country"],
        "initial_vote_correct": bool(initial_row["initial_vote_correct"]),
        "initial_vote_accuracy": 1.0 if bool(initial_row["initial_vote_correct"]) else 0.0,
        "final_vote_country": final_row["final_vote_country"],
        "final_vote_correct": bool(final_row["final_vote_correct"]),
        "final_vote_accuracy": 1.0 if bool(final_row["final_vote_correct"]) else 0.0,
        "time_to_final_any_consensus": time_to_final_any_consensus,
        "time_to_final_correct_consensus": time_to_final_correct_consensus,
        "collaboration_gain_over_initial_accuracy": float(
            final_row["final_accuracy"] - initial_row["initial_accuracy"]
        ),
        "collaboration_gain_over_initial_vote_accuracy": float(
            (1.0 if bool(final_row["final_vote_correct"]) else 0.0)
            - (1.0 if bool(initial_row["initial_vote_correct"]) else 0.0)
        ),
        "invalid_observation_count": int(sum(1 for row in observer_rows if not bool(row.get("valid", False)))),
        "invalid_decision_count": int(sum(1 for row in decision_rows if not bool(row.get("valid", False)))),
    }
    return frame, summary


def _summarize_stage(
    rows: list[dict[str, Any]],
    *,
    countries: list[str],
    truth_country: str,
    consensus_threshold: float,
    polarization_threshold: float,
    country_field: str,
) -> dict[str, Any]:
    valid_rows = [
        row
        for row in rows
        if bool(row.get("valid", False)) and isinstance(row.get(country_field), str)
    ]
    valid_count = len(valid_rows)
    invalid_count = len(rows) - valid_count
    counts = pd.Series(
        {country: sum(1 for row in valid_rows if row[country_field] == country) for country in countries},
        dtype=float,
    )
    denominator = float(valid_count) if valid_count > 0 else 1.0
    shares = {country: float(counts[country] / denominator) for country in countries}
    classification = classify_probe_distribution(
        shares,
        truth_country=truth_country,
        consensus_threshold=consensus_threshold,
        polarization_threshold=polarization_threshold,
    )
    return {
        "accuracy": float(sum(1 for row in valid_rows if row[country_field] == truth_country) / denominator)
        if valid_count > 0
        else 0.0,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "support_size": int(sum(1 for share in shares.values() if share > 0.0)),
        "shares": shares,
        "classification": classification,
    }

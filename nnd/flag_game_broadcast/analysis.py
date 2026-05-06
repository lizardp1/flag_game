from __future__ import annotations

from typing import Any

import pandas as pd

from nnd.flag_game.analysis import classify_probe_distribution


def summarize_broadcast_rounds(
    broadcast_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    *,
    countries: list[str],
    truth_country: str,
    consensus_threshold: float,
    polarization_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not broadcast_rows or not decision_rows:
        return pd.DataFrame(), {}

    rounds = sorted(
        set(int(row["round"]) for row in broadcast_rows)
        | set(int(row["round"]) for row in decision_rows)
    )
    per_round_rows: list[dict[str, Any]] = []
    time_to_final_any_consensus: int | None = None
    time_to_final_correct_consensus: int | None = None

    for round_idx in rounds:
        round_broadcasts = [row for row in broadcast_rows if int(row["round"]) == round_idx]
        round_decisions = [row for row in decision_rows if int(row["round"]) == round_idx]
        initial_summary = _summarize_stage(
            round_broadcasts,
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

        decision_df = pd.DataFrame(round_decisions)
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
            "changed_mind_fraction": _mean_or_zero(decision_df, "changed_mind"),
            "toward_prestige_majority_switches": _sum_or_zero(decision_df, "switched_toward_prestige_majority"),
            "toward_comparison_majority_switches": _sum_or_zero(decision_df, "switched_toward_comparison_majority"),
            "to_prestige_only_country_count": _sum_or_zero(decision_df, "aligns_with_prestige_only_country"),
            "to_comparison_only_country_count": _sum_or_zero(decision_df, "aligns_with_comparison_only_country"),
            "mean_influential_prestige_fraction": _mean_or_none(
                decision_df, "influential_prestige_fraction"
            ),
            "mean_influential_comparison_fraction": _mean_or_none(
                decision_df, "influential_comparison_fraction"
            ),
        }
        for country in countries:
            row[f"initial_share_{country}"] = initial_summary["shares"].get(country, 0.0)
            row[f"final_share_{country}"] = final_summary["shares"].get(country, 0.0)
        per_round_rows.append(row)

    frame = pd.DataFrame(per_round_rows)
    initial_row = frame.iloc[0].to_dict()
    final_row = frame.iloc[-1].to_dict()
    decision_frame = pd.DataFrame(decision_rows)

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
        "mean_changed_mind_fraction": float(frame["changed_mind_fraction"].mean()),
        "total_toward_prestige_majority_switches": int(frame["toward_prestige_majority_switches"].sum()),
        "total_toward_comparison_majority_switches": int(frame["toward_comparison_majority_switches"].sum()),
        "switch_delta_toward_prestige": int(
            frame["toward_prestige_majority_switches"].sum()
            - frame["toward_comparison_majority_switches"].sum()
        ),
        "mean_influential_prestige_fraction": float(
            decision_frame["influential_prestige_fraction"].dropna().mean()
        )
        if "influential_prestige_fraction" in decision_frame
        and not decision_frame["influential_prestige_fraction"].dropna().empty
        else None,
        "mean_influential_comparison_fraction": float(
            decision_frame["influential_comparison_fraction"].dropna().mean()
        )
        if "influential_comparison_fraction" in decision_frame
        and not decision_frame["influential_comparison_fraction"].dropna().empty
        else None,
        "decision_alignment_prestige_only_rate": _mean_or_zero(
            decision_frame, "aligns_with_prestige_only_country"
        ),
        "decision_alignment_comparison_only_rate": _mean_or_zero(
            decision_frame, "aligns_with_comparison_only_country"
        ),
        "self_report_mismatch_count": int(
            sum(
                1
                for row in broadcast_rows
                if bool(row.get("valid", False))
                and not bool(row.get("self_report_matches_assigned", False))
            )
        ),
        "invalid_broadcast_count": int(sum(1 for row in broadcast_rows if not bool(row.get("valid", False)))),
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


def _mean_or_zero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    series = frame[column].dropna()
    if series.empty:
        return 0.0
    return float(series.astype(float).mean())


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return float(series.astype(float).mean())


def _sum_or_zero(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    series = frame[column].dropna()
    if series.empty:
        return 0
    return int(series.astype(int).sum())

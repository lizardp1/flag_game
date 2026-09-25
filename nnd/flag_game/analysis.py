from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from nnd.metrics import entropy, polarization_u


T0_INFORMATIVENESS_LABELS = ("unique", "narrow", "moderate", "ambiguous", "invalid")


def classify_probe_distribution(
    shares: dict[str, float],
    *,
    truth_country: str,
    consensus_threshold: float,
    polarization_threshold: float,
) -> dict[str, Any]:
    if not shares:
        return {
            "outcome": "fragmentation",
            "consensus_country": None,
            "consensus_correct": False,
            "top_vote_country": None,
            "top_vote_correct": False,
            "top1_share": 0.0,
            "top2_share": 0.0,
        }
    ordered = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
    top_country, top_share = ordered[0]
    top2_share = ordered[1][1] if len(ordered) > 1 else 0.0
    top_vote_country = _unique_top_country(shares)
    top_vote_correct = top_vote_country == truth_country
    if top_share >= consensus_threshold:
        return {
            "outcome": "correct_consensus" if top_country == truth_country else "wrong_consensus",
            "consensus_country": top_country,
            "consensus_correct": top_country == truth_country,
            "top_vote_country": top_vote_country,
            "top_vote_correct": top_vote_correct,
            "top1_share": top_share,
            "top2_share": top2_share,
        }
    substantial = sum(1 for _, share in ordered if share >= polarization_threshold)
    if substantial >= 2:
        outcome = "polarization"
    else:
        outcome = "fragmentation"
    return {
        "outcome": outcome,
        "consensus_country": None,
        "consensus_correct": False,
        "top_vote_country": top_vote_country,
        "top_vote_correct": top_vote_correct,
        "top1_share": top_share,
        "top2_share": top2_share,
    }


def summarize_probe_rows(
    probe_rows: list[dict[str, Any]],
    *,
    countries: list[str],
    truth_country: str,
    consensus_threshold: float,
    polarization_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not probe_rows:
        return pd.DataFrame(), {}

    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in probe_rows:
        grouped.setdefault(int(row["t"]), []).append(row)

    time_to_any_consensus: int | None = None
    time_to_correct_consensus: int | None = None

    for t in sorted(grouped):
        records = grouped[t]
        valid_records = [
            record
            for record in records
            if bool(record.get("valid", True)) and isinstance(record.get("country"), str)
        ]
        counts = Counter(str(record["country"]) for record in valid_records)
        valid_count = sum(counts.values())
        invalid_count = len(records) - valid_count
        total = valid_count or 1
        shares = {country: counts.get(country, 0) / total for country in countries} if valid_count > 0 else {country: 0.0 for country in countries}
        distribution = pd.Series([shares[country] for country in countries], index=countries, dtype=float)
        classification = classify_probe_distribution(
            shares,
            truth_country=truth_country,
            consensus_threshold=consensus_threshold,
            polarization_threshold=polarization_threshold,
        )
        if valid_count > 0 and time_to_any_consensus is None and classification["consensus_country"] is not None:
            time_to_any_consensus = t
        if valid_count > 0 and time_to_correct_consensus is None and classification["consensus_correct"]:
            time_to_correct_consensus = t
        accuracy = (counts.get(truth_country, 0) / valid_count) if valid_count > 0 else 0.0
        row = {
            "t": t,
            "truth_country": truth_country,
            "truth_mass": accuracy,
            "mean_accuracy": accuracy,
            "valid_probe_count": valid_count,
            "invalid_probe_count": invalid_count,
            "support_size": int(sum(1 for share in shares.values() if share > 0.0)),
            "entropy": entropy(distribution.to_numpy()),
            "U": polarization_u(distribution.to_numpy()),
            "top1_share": classification["top1_share"],
            "top2_share": classification["top2_share"],
            "outcome": classification["outcome"],
            "consensus_country": classification["consensus_country"],
            "consensus_correct": classification["consensus_correct"],
            "top_vote_country": classification["top_vote_country"],
            "top_vote_correct": classification["top_vote_correct"],
        }
        for country in countries:
            row[country] = shares[country]
        rows.append(row)

    frame = pd.DataFrame(rows)
    initial_row = frame.iloc[0].to_dict()
    final_row = frame.iloc[-1].to_dict()

    summary = {
        "initial_accuracy": float(initial_row["mean_accuracy"]),
        "final_accuracy": float(final_row["mean_accuracy"]),
        "final_support_size": int(final_row["support_size"]),
        "initial_truth_mass": float(initial_row["truth_mass"]),
        "final_truth_mass": float(final_row["truth_mass"]),
        "initial_majority_country": initial_row["consensus_country"] or _argmax_country(initial_row, countries),
        "initial_vote_country": initial_row["top_vote_country"],
        "initial_vote_correct": bool(initial_row["top_vote_correct"]),
        "initial_vote_accuracy": 1.0 if bool(initial_row["top_vote_correct"]) else 0.0,
        "final_vote_country": final_row["top_vote_country"],
        "final_vote_correct": bool(final_row["top_vote_correct"]),
        "final_vote_accuracy": 1.0 if bool(final_row["top_vote_correct"]) else 0.0,
        "final_consensus_country": final_row["consensus_country"],
        "final_outcome": final_row["outcome"],
        "final_consensus_correct": bool(final_row["consensus_correct"]),
        "time_to_any_consensus": time_to_any_consensus,
        "time_to_correct_consensus": time_to_correct_consensus,
        "collaboration_gain_over_initial_accuracy": float(final_row["mean_accuracy"] - initial_row["mean_accuracy"]),
        "collaboration_gain_over_initial_vote": float(
            (1.0 if (final_row["consensus_country"] or _argmax_country(final_row, countries)) == truth_country else 0.0)
            - (1.0 if (initial_row["consensus_country"] or _argmax_country(initial_row, countries)) == truth_country else 0.0)
        ),
        "collaboration_gain_over_initial_vote_accuracy": float(
            (1.0 if bool(final_row["top_vote_correct"]) else 0.0)
            - (1.0 if bool(initial_row["top_vote_correct"]) else 0.0)
        ),
    }
    return frame, summary


def summarize_initial_probe_rows(
    probe_rows: list[dict[str, Any]],
    *,
    crop_diagnostics: list[dict[str, Any]],
    truth_country: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    crop_by_agent = {
        int(row["agent_id"]): row
        for row in crop_diagnostics
        if "agent_id" in row
    }
    t0_rows = [
        row
        for row in probe_rows
        if int(row.get("t", -1)) == 0
    ]

    rows: list[dict[str, Any]] = []
    for record in sorted(t0_rows, key=lambda row: int(row.get("agent_id", -1))):
        agent_id = int(record["agent_id"])
        crop_diag = crop_by_agent.get(agent_id, {})
        compatible_countries = [str(country) for country in crop_diag.get("compatible_countries", [])]
        predicted_country = record.get("country")
        valid = bool(record.get("valid", True)) and isinstance(predicted_country, str)
        predicted_text = str(predicted_country) if valid else None
        correct = bool(record.get("correct", False)) if valid else False
        compatible_with_crop = bool(valid and predicted_text in compatible_countries)
        informativeness_label = str(crop_diag.get("informativeness_label", "invalid"))

        rows.append(
            {
                "agent_id": agent_id,
                "model": str(record.get("model", crop_diag.get("model", ""))),
                "truth_country": truth_country,
                "predicted_country": predicted_text,
                "valid": valid,
                "correct": correct,
                "compatible_with_crop": compatible_with_crop,
                "compatible_but_wrong": bool(compatible_with_crop and not correct),
                "compatible_country_count": int(crop_diag.get("compatible_country_count", 0) or 0),
                "compatible_countries_json": compatible_countries,
                "informativeness_label": informativeness_label,
                "informativeness_bits": crop_diag.get("informativeness_bits"),
                "informativeness_score": crop_diag.get("informativeness_score"),
                "is_unique": bool(crop_diag.get("is_unique", False)),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {
            "initial_probe_count": 0,
            "initial_valid_probe_count": 0,
            "initial_valid_probe_rate": 0.0,
            "initial_crop_compatibility_rate": 0.0,
            "initial_crop_compatibility_rate_given_valid": 0.0,
            "initial_compatible_but_wrong_rate": 0.0,
            "initial_mean_compatible_country_count": 0.0,
            **{
                f"initial_{label}_probe_count": 0
                for label in T0_INFORMATIVENESS_LABELS
            },
            **{
                f"initial_{label}_accuracy": None
                for label in T0_INFORMATIVENESS_LABELS
            },
            **{
                f"initial_{label}_crop_compatibility_rate": None
                for label in T0_INFORMATIVENESS_LABELS
            },
        }

    summary: dict[str, Any] = {
        "initial_probe_count": int(len(frame)),
        "initial_valid_probe_count": int(frame["valid"].sum()),
        "initial_valid_probe_rate": float(frame["valid"].mean()),
        "initial_crop_compatibility_rate": float(frame["compatible_with_crop"].mean()),
        "initial_compatible_but_wrong_rate": float(frame["compatible_but_wrong"].mean()),
        "initial_mean_compatible_country_count": float(frame["compatible_country_count"].mean()),
    }

    valid_frame = frame[frame["valid"]]
    if valid_frame.empty:
        summary["initial_crop_compatibility_rate_given_valid"] = 0.0
    else:
        summary["initial_crop_compatibility_rate_given_valid"] = float(valid_frame["compatible_with_crop"].mean())

    for label in T0_INFORMATIVENESS_LABELS:
        subset = frame[frame["informativeness_label"] == label]
        summary[f"initial_{label}_probe_count"] = int(len(subset))
        summary[f"initial_{label}_accuracy"] = (
            float(subset["correct"].mean()) if not subset.empty else None
        )
        summary[f"initial_{label}_crop_compatibility_rate"] = (
            float(subset["compatible_with_crop"].mean()) if not subset.empty else None
        )

    return frame, summary


def _argmax_country(row: dict[str, Any], countries: list[str]) -> str | None:
    best_country: str | None = None
    best_value = float("-inf")
    for country in countries:
        value = float(row.get(country, 0.0))
        if value > best_value:
            best_value = value
            best_country = country
    return best_country


def _unique_top_country(shares: dict[str, float]) -> str | None:
    if not shares:
        return None
    max_share = max(shares.values())
    winners = [country for country, share in shares.items() if share == max_share]
    if len(winners) != 1 or max_share <= 0.0:
        return None
    return winners[0]

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from nnd.flag_game.backend import FlagGameOpenAIBackend, ScriptedFlagGameBackend
from nnd.flag_game.catalog import FlagSpec
from nnd.flag_game_org.parsing import (
    ObserverStatement,
    OrganizationDecision,
    parse_observer_statement,
    parse_organization_decision,
)
from nnd.flag_game_org.prompts import (
    aggregator_decision_retry_text,
    aggregator_decision_text,
    observer_statement_retry_text,
    observer_statement_text,
    openai_multimodal_messages,
    openai_text_messages,
)


@dataclass
class OrgFlagGameOpenAIBackend(FlagGameOpenAIBackend):
    def observer_statement(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        m: int,
    ) -> ObserverStatement:
        messages = openai_multimodal_messages(
            text=observer_statement_text(
                countries=countries,
                memory_lines=memory_lines,
                m=m,
            ),
            crop_data_uri=prepared_crop,
            image_detail=self.image_detail,
        )
        return self._call_with_retries(
            messages,
            lambda text: parse_observer_statement(
                text,
                countries=countries,
                m=m,
            ),
            retry_builder=lambda exc: observer_statement_retry_text(
                countries=countries,
                m=m,
                error_text=str(exc),
            ),
        )

    def organization_decision(
        self,
        *,
        countries: list[str],
        memory_lines: list[str],
        observer_statement_lines: list[str],
        m: int,
    ) -> OrganizationDecision:
        messages = openai_text_messages(
            text=aggregator_decision_text(
                countries=countries,
                memory_lines=memory_lines,
                observer_statement_lines=observer_statement_lines,
                m=m,
            )
        )
        return self._call_with_retries(
            messages,
            lambda text: parse_organization_decision(
                text,
                countries=countries,
                m=m,
            ),
            retry_builder=lambda exc: aggregator_decision_retry_text(
                countries=countries,
                m=m,
                error_text=str(exc),
            ),
        )


@dataclass
class OrgFlagGameScriptedBackend(ScriptedFlagGameBackend):
    def observer_statement(
        self,
        *,
        countries: list[str],
        prepared_crop: np.ndarray,
        memory_lines: list[str],
        m: int,
    ) -> ObserverStatement:
        if m != 3:
            raise ValueError("flag_game_org currently uses interaction m=3")
        scores = {country: self._country_score(country, prepared_crop, memory_lines) for country in countries}
        best_country = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return ObserverStatement(
            country=best_country,
            reason=self._reason(best_country, prepared_crop, memory_lines),
        )

    def organization_decision(
        self,
        *,
        countries: list[str],
        memory_lines: list[str],
        observer_statement_lines: list[str],
        m: int,
    ) -> OrganizationDecision:
        if m != 3:
            raise ValueError("flag_game_org currently uses interaction m=3")
        observer_votes = _parse_observer_vote_lines(observer_statement_lines)
        vote_counts = Counter(
            payload["country"]
            for payload in observer_votes.values()
            if payload["country"] in countries
        )
        memory_votes = _memory_vote_counts(memory_lines)
        scores: dict[str, float] = {}
        memory_weight = 0.5
        for country in countries:
            scores[country] = float(vote_counts.get(country, 0))
            if memory_votes:
                scores[country] += memory_weight * memory_votes.get(country, 0)
        best_country = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        matching_support = sum(
            1
            for payload in observer_votes.values()
            if payload["country"] == best_country
        )
        if matching_support:
            reason = f"The strongest observer support points to {best_country}."
        elif memory_votes:
            reason = f"Observer support was weak, so I kept the prior organization decision near {best_country}."
        else:
            reason = f"No observer majority was clear, so I chose {best_country} deterministically."
        return OrganizationDecision(
            country=best_country,
            reason=reason,
        )


def _memory_vote_counts(memory_lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in memory_lines:
        match = re.match(r"([^|]+)", line)
        if match:
            counts[match.group(1).strip()] += 1
    return counts


def _parse_observer_vote_lines(lines: list[str]) -> dict[int, dict[str, str]]:
    payloads: dict[int, dict[str, str]] = {}
    for idx, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        country = payload.get("country")
        if not isinstance(country, str) or not country.strip():
            continue
        payloads[idx] = {
            "country": country.strip(),
        }
    return payloads


def build_backend(
    *,
    backend_name: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    debug_dir: Path,
    image_detail: str,
    seed: int,
    country_lookup: dict[str, FlagSpec] | None = None,
) -> OrgFlagGameOpenAIBackend | OrgFlagGameScriptedBackend:
    if backend_name == "scripted":
        return OrgFlagGameScriptedBackend(
            seed=seed,
            country_lookup=country_lookup,
        )
    return OrgFlagGameOpenAIBackend(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        debug_dir=debug_dir,
        image_detail=image_detail,
    )

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nnd.flag_game.backend import FlagGameOpenAIBackend, ScriptedFlagGameBackend
from nnd.flag_game.catalog import FlagSpec
from nnd.flag_game_broadcast.parsing import (
    BroadcastStatement,
    FinalDecision,
    parse_broadcast_statement,
    parse_final_decision,
)
from nnd.flag_game_broadcast.prompts import (
    decision_retry_text,
    decision_text,
    openai_multimodal_messages,
    statement_retry_text,
    statement_text,
)


@dataclass
class BroadcastFlagGameOpenAIBackend(FlagGameOpenAIBackend):
    assigned_model_identity: str = ""

    def broadcast_statement(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        m: int,
    ) -> BroadcastStatement:
        messages = openai_multimodal_messages(
            text=statement_text(
                countries=countries,
                memory_lines=memory_lines,
                m=m,
                model_identity=self.assigned_model_identity or self.model,
            ),
            crop_data_uri=prepared_crop,
            image_detail=self.image_detail,
        )
        expected_model_identity = self.assigned_model_identity or self.model
        return self._call_with_retries(
            messages,
            lambda text: parse_broadcast_statement(
                text,
                countries=countries,
                m=m,
                expected_model_identity=expected_model_identity,
            ),
            retry_builder=lambda exc: statement_retry_text(
                countries=countries,
                model_identity=expected_model_identity,
                m=m,
                error_text=str(exc),
            ),
        )

    def final_decision(
        self,
        *,
        countries: list[str],
        prepared_crop: str,
        memory_lines: list[str],
        round_broadcast_lines: list[str],
        m: int,
        max_influential_agents: int,
        valid_agent_ids: set[int],
    ) -> FinalDecision:
        messages = openai_multimodal_messages(
            text=decision_text(
                countries=countries,
                memory_lines=memory_lines,
                round_broadcast_lines=round_broadcast_lines,
                m=m,
                max_influential_agents=max_influential_agents,
                social_susceptibility=self.social_susceptibility,
                prompt_social_susceptibility=self.prompt_social_susceptibility,
            ),
            crop_data_uri=prepared_crop,
            image_detail=self.image_detail,
        )
        return self._call_with_retries(
            messages,
            lambda text: parse_final_decision(
                text,
                countries=countries,
                m=m,
                max_influential_agents=max_influential_agents,
                valid_agent_ids=valid_agent_ids,
            ),
            retry_builder=lambda exc: decision_retry_text(
                countries=countries,
                m=m,
                max_influential_agents=max_influential_agents,
                error_text=str(exc),
            ),
        )


@dataclass
class BroadcastFlagGameScriptedBackend(ScriptedFlagGameBackend):
    assigned_model_identity: str = ""

    def broadcast_statement(
        self,
        *,
        countries: list[str],
        prepared_crop: np.ndarray,
        memory_lines: list[str],
        m: int,
    ) -> BroadcastStatement:
        scores = {country: self._country_score(country, prepared_crop, memory_lines) for country in countries}
        best_country = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        reason = None
        if m > 1:
            reason = self._reason(best_country, prepared_crop, memory_lines)
            if m == 2:
                reason = _clip_to_short_phrase(reason)
        return BroadcastStatement(
            model_identity=self.assigned_model_identity or "scripted",
            country=best_country,
            reason=reason,
        )

    def final_decision(
        self,
        *,
        countries: list[str],
        prepared_crop: np.ndarray,
        memory_lines: list[str],
        round_broadcast_lines: list[str],
        m: int,
        max_influential_agents: int,
        valid_agent_ids: set[int],
    ) -> FinalDecision:
        base_scores = {country: self._country_score(country, prepared_crop, memory_lines) for country in countries}
        broadcast_votes = _parse_broadcast_vote_lines(round_broadcast_lines, valid_agent_ids=valid_agent_ids)
        for agent_id, payload in broadcast_votes.items():
            country = payload["country"]
            if country not in base_scores:
                continue
            base_scores[country] += 1.5

        ordered = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))
        best_country = ordered[0][0]
        matching_influential_ids = [
            agent_id
            for agent_id, payload in broadcast_votes.items()
            if payload["country"] == best_country
        ][:max_influential_agents]
        reason = None
        if m > 1:
            if matching_influential_ids:
                reason = f"I updated toward the broadcast support for {best_country}."
            else:
                reason = self._reason(best_country, prepared_crop, memory_lines)
            if m == 2:
                reason = _clip_to_short_phrase(reason)
        return FinalDecision(
            country=best_country,
            influential_agent_ids=tuple(matching_influential_ids),
            reason=reason,
        )


def _clip_to_short_phrase(text: str) -> str:
    words = text.strip().split()
    if not words:
        return "limited evidence"
    return " ".join(words[:6])


def _parse_broadcast_vote_lines(
    lines: list[str],
    *,
    valid_agent_ids: set[int],
) -> dict[int, dict[str, str]]:
    payloads: dict[int, dict[str, str]] = {}
    for line in lines:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        try:
            agent_id = int(parts[0].removeprefix("agent "))
        except ValueError:
            continue
        if agent_id not in valid_agent_ids:
            continue
        payloads[agent_id] = {
            "model_identity": parts[1].removeprefix("model ").strip(),
            "country": parts[2].removeprefix("country ").strip(),
        }
    return payloads


def build_backend(
    *,
    backend_name: str,
    model: str,
    assigned_model_identity: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    debug_dir: Path,
    image_detail: str,
    seed: int,
    social_susceptibility: float,
    prompt_social_susceptibility: bool,
    country_lookup: dict[str, FlagSpec] | None = None,
) -> BroadcastFlagGameOpenAIBackend | BroadcastFlagGameScriptedBackend:
    if backend_name == "scripted":
        return BroadcastFlagGameScriptedBackend(
            seed=seed,
            social_susceptibility=social_susceptibility,
            country_lookup=country_lookup,
            assigned_model_identity=assigned_model_identity,
        )
    return BroadcastFlagGameOpenAIBackend(
        model=model,
        assigned_model_identity=assigned_model_identity,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        debug_dir=debug_dir,
        image_detail=image_detail,
        social_susceptibility=social_susceptibility,
        prompt_social_susceptibility=prompt_social_susceptibility,
    )

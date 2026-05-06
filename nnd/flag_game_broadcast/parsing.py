from __future__ import annotations

from dataclasses import dataclass

from nnd.backends.parsing import ParseError, _fuzzy_match, _load_json_strict


@dataclass(frozen=True)
class BroadcastStatement:
    model_identity: str
    country: str
    reason: str | None = None

    def normalized_broadcast(self) -> str:
        if self.reason:
            return f"{self.model_identity} | {self.country} | {self.reason.strip()}"
        return f"{self.model_identity} | {self.country}"


@dataclass(frozen=True)
class FinalDecision:
    country: str
    influential_agent_ids: tuple[int, ...]
    reason: str | None = None

    def normalized_memory_entry(self) -> str:
        if self.reason:
            return f"{self.country} | {self.reason.strip()}"
        return self.country


def parse_broadcast_statement(
    text: str,
    *,
    countries: list[str],
    m: int,
    expected_model_identity: str,
) -> BroadcastStatement:
    if m not in (1, 2, 3):
        raise ValueError("interaction m must be in {1, 2, 3}")
    obj = _load_json_strict(text)
    expected_keys = {"model_identity", "country"} if m == 1 else {"model_identity", "country", "reason"}
    if set(obj) != expected_keys:
        raise ParseError(f"response must contain exactly {sorted(expected_keys)!r}")

    model_identity = obj.get("model_identity")
    if not isinstance(model_identity, str) or model_identity.strip() != expected_model_identity:
        raise ParseError(f'model_identity must be exactly "{expected_model_identity}"')

    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")

    if m == 1:
        return BroadcastStatement(model_identity=expected_model_identity, country=parsed_country)

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("reason must be a non-empty string")
    return BroadcastStatement(
        model_identity=expected_model_identity,
        country=parsed_country,
        reason=reason.strip(),
    )


def parse_final_decision(
    text: str,
    *,
    countries: list[str],
    m: int,
    max_influential_agents: int,
    valid_agent_ids: set[int],
) -> FinalDecision:
    if m not in (1, 2, 3):
        raise ValueError("interaction m must be in {1, 2, 3}")
    obj = _load_json_strict(text)
    expected_keys = {"country", "influential_agent_ids"} if m == 1 else {"country", "reason", "influential_agent_ids"}
    if set(obj) != expected_keys:
        raise ParseError(f"response must contain exactly {sorted(expected_keys)!r}")

    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")

    raw_ids = obj.get("influential_agent_ids")
    if not isinstance(raw_ids, list):
        raise ParseError("influential_agent_ids must be a JSON list")
    influential_agent_ids: list[int] = []
    for item in raw_ids:
        if not isinstance(item, int):
            raise ParseError("influential_agent_ids entries must be integers")
        if item not in valid_agent_ids:
            raise ParseError(f"influential_agent_ids entry {item} is not in the current broadcast")
        influential_agent_ids.append(item)
    if len(influential_agent_ids) != len(set(influential_agent_ids)):
        raise ParseError("influential_agent_ids must be unique")
    if len(influential_agent_ids) > max_influential_agents:
        raise ParseError(
            f"influential_agent_ids may contain at most {max_influential_agents} entries"
        )

    if m == 1:
        return FinalDecision(
            country=parsed_country,
            influential_agent_ids=tuple(influential_agent_ids),
        )

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("reason must be a non-empty string")
    return FinalDecision(
        country=parsed_country,
        influential_agent_ids=tuple(influential_agent_ids),
        reason=reason.strip(),
    )

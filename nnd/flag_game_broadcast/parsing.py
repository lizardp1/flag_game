from __future__ import annotations

from dataclasses import dataclass

from nnd.backends.parsing import ParseError, _fuzzy_match, _load_json_strict


@dataclass(frozen=True)
class BroadcastStatement:
    country: str
    reason: str | None = None

    def normalized_broadcast(self) -> str:
        if self.reason:
            return f"{self.country} | {self.reason.strip()}"
        return self.country


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
) -> BroadcastStatement:
    if m not in (1, 2, 3):
        raise ValueError("interaction m must be in {1, 2, 3}")
    obj = _load_json_strict(text)
    expected_keys = {"country"} if m == 1 else {"country", "reason"}
    if set(obj) != expected_keys:
        raise ParseError(f"response must contain exactly {sorted(expected_keys)!r}")

    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")

    if m == 1:
        return BroadcastStatement(country=parsed_country)

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("reason must be a non-empty string")
    return BroadcastStatement(
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
    expected_keys = {"country"} if m == 1 else {"country", "reason"}
    if set(obj) != expected_keys:
        raise ParseError(f"response must contain exactly {sorted(expected_keys)!r}")

    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")

    if m == 1:
        return FinalDecision(
            country=parsed_country,
            influential_agent_ids=(),
        )

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("reason must be a non-empty string")
    return FinalDecision(
        country=parsed_country,
        influential_agent_ids=(),
        reason=reason.strip(),
    )

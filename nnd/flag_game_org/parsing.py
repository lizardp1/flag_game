from __future__ import annotations

from dataclasses import dataclass

from nnd.backends.parsing import ParseError, _fuzzy_match, _load_json_strict


@dataclass(frozen=True)
class ObserverStatement:
    country: str
    reason: str = ""

    def normalized_statement(self) -> str:
        return f"{self.country} | {self.reason.strip()}" if self.reason else self.country

    def normalized_memory_entry(self) -> str:
        return f"{self.country} | {self.reason.strip()}" if self.reason else self.country


@dataclass(frozen=True)
class OrganizationDecision:
    country: str
    reason: str = ""

    def normalized_memory_entry(self) -> str:
        return f"{self.country} | {self.reason.strip()}" if self.reason else self.country


def _require_bandwidth(m: int) -> None:
    if m not in (1, 2, 3):
        raise ValueError("interaction m must be one of {1, 2, 3}")


def parse_observer_statement(
    text: str,
    *,
    countries: list[str],
    m: int,
) -> ObserverStatement:
    _require_bandwidth(m)
    obj = _load_json_strict(text)
    expected_keys = {"country"} if m == 1 else {"country", "reason"}
    if set(obj) != expected_keys:
        raise ParseError(f"response must contain exactly {sorted(expected_keys)!r}")

    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")

    reason = obj.get("reason", "")
    if m != 1 and (not isinstance(reason, str) or not reason.strip()):
        raise ParseError("reason must be a non-empty string")
    return ObserverStatement(
        country=parsed_country,
        reason=reason.strip(),
    )


def parse_organization_decision(
    text: str,
    *,
    countries: list[str],
    m: int,
) -> OrganizationDecision:
    _require_bandwidth(m)
    obj = _load_json_strict(text)
    expected_keys = {"country"} if m == 1 else {"country", "reason"}
    if set(obj) != expected_keys:
        raise ParseError(f"response must contain exactly {sorted(expected_keys)!r}")

    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")

    reason = obj.get("reason", "")
    if m != 1 and (not isinstance(reason, str) or not reason.strip()):
        raise ParseError("reason must be a non-empty string")
    return OrganizationDecision(
        country=parsed_country,
        reason=reason.strip(),
    )

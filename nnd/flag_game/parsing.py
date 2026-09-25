from __future__ import annotations

from dataclasses import dataclass

from nnd.backends.parsing import ParseError, _load_json_strict, _fuzzy_match
from nnd.flag_game.catalog import canonical_country_name


@dataclass(frozen=True)
class InteractionMessage:
    country: str
    clue: str | None = None
    reason: str | None = None

    def normalized_memory_entry(self) -> str:
        if self.reason:
            return f"{self.country} | {self.reason.strip()}"
        if self.clue:
            return f"{self.country} | {self.clue.strip()}"
        return self.country

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"country": self.country}
        if self.clue is not None:
            payload["clue"] = self.clue
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


def parse_probe_response(
    text: str,
    countries: list[str],
    m: int = 1,
) -> InteractionMessage:
    return parse_interaction_response(text, countries, m)


def parse_interaction_response(
    text: str,
    countries: list[str],
    m: int,
) -> InteractionMessage:
    if m not in (1, 2, 3):
        raise ValueError("interaction m must be in {1, 2, 3}")
    obj = _load_json_strict(text)
    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _fuzzy_match(country.strip(), countries, name="country")
    if m == 1:
        expected = {"country"}
        if set(obj) != expected:
            raise ParseError(f"m=1 response must contain exactly {sorted(expected)!r}")
        return InteractionMessage(country=parsed_country)

    if m == 2:
        expected = {"country", "clue"}
        if set(obj) != expected:
            raise ParseError(f"m=2 response must contain exactly {sorted(expected)!r}")
        clue = obj.get("clue")
        if not isinstance(clue, str) or not clue.strip():
            raise ParseError("m=2 clue must be a non-empty string")
        return InteractionMessage(country=parsed_country, clue=clue.strip())

    expected = {"country", "reason"}
    if set(obj) != expected:
        raise ParseError(f"m=3 response must contain exactly {sorted(expected)!r}")
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("m=3 reason must be a non-empty string")
    return InteractionMessage(country=parsed_country, reason=reason.strip())


def parse_open_country_probe_response(
    text: str,
    countries: list[str],
    m: int = 1,
    *,
    universe_name: str = "world_rectangle",
) -> InteractionMessage:
    return parse_open_country_interaction_response(
        text,
        countries,
        m,
        universe_name=universe_name,
    )


def parse_open_country_interaction_response(
    text: str,
    countries: list[str],
    m: int,
    *,
    universe_name: str = "world_rectangle",
) -> InteractionMessage:
    if m not in (1, 2, 3):
        raise ValueError("interaction m must be one of {1, 2, 3}")
    obj = _load_json_strict(text)
    country = obj.get("country")
    if not isinstance(country, str) or not country.strip():
        raise ParseError("Missing or invalid 'country'")
    parsed_country = _parse_open_country_name(
        country.strip(),
        countries,
        universe_name=universe_name,
    )
    if m == 1:
        expected = {"country"}
        if set(obj) != expected:
            raise ParseError(f"m=1 response must contain exactly {sorted(expected)!r}")
        return InteractionMessage(country=parsed_country)

    if m == 2:
        expected = {"country", "clue"}
        if set(obj) != expected:
            raise ParseError(f"m=2 response must contain exactly {sorted(expected)!r}")
        clue = obj.get("clue")
        if not isinstance(clue, str) or not clue.strip():
            raise ParseError("m=2 clue must be a non-empty string")
        return InteractionMessage(country=parsed_country, clue=clue.strip())

    expected = {"country", "reason"}
    if set(obj) != expected:
        raise ParseError(f"m=3 response must contain exactly {sorted(expected)!r}")
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("m=3 reason must be a non-empty string")
    return InteractionMessage(country=parsed_country, reason=reason.strip())


def _parse_open_country_name(
    value: str,
    countries: list[str],
    *,
    universe_name: str,
) -> str:
    canonical = canonical_country_name(value, universe_name=universe_name)
    if canonical is not None:
        if canonical in countries:
            return canonical
        raise ParseError(
            f"country value '{value}' is recognized as '{canonical}' but is not in this trial's country pool"
        )
    try:
        return _fuzzy_match(value, countries, name="country")
    except ParseError as exc:
        raise ParseError(f"country value '{value}' is not a recognized country name") from exc

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class NumberMessage:
    number: int
    reason: str | None = None

    def normalized_memory_entry(self) -> str:
        if self.reason:
            return f"{self.number} | {self.reason.strip()}"
        return str(self.number)


@dataclass(frozen=True)
class NumberDecision:
    number: int
    influential_agent_ids: tuple[int, ...] = ()
    reason: str | None = None

    def normalized_memory_entry(self) -> str:
        if self.reason:
            return f"{self.number} | {self.reason.strip()}"
        return str(self.number)


def _load_json_strict(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ParseError(f"Response is not valid JSON: {exc}") from exc
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as inner_exc:
            raise ParseError(f"Response is not valid JSON: {inner_exc}") from inner_exc
    if not isinstance(obj, dict):
        raise ParseError("Response must be a JSON object")
    return obj


def _parse_number(value: Any) -> int:
    if isinstance(value, bool):
        raise ParseError("Missing or invalid 'number'")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ParseError("Missing or invalid 'number'")
    return parsed


def parse_number_message(text: str, *, allowed_numbers: list[int], m: int) -> NumberMessage:
    if m not in (1, 3):
        raise ValueError("number game supports m=1 or m=3")
    obj = _load_json_strict(text)
    expected = {"number"} if m == 1 else {"number", "reason"}
    if set(obj) != expected:
        raise ParseError(f"response must contain exactly {sorted(expected)!r}")
    number = _parse_number(obj.get("number"))
    if m == 1:
        return NumberMessage(number=number)
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("m=3 reason must be a non-empty string")
    return NumberMessage(number=number, reason=reason.strip())


def parse_number_decision(
    text: str,
    *,
    allowed_numbers: list[int],
    m: int,
    max_influential_agents: int,
    valid_agent_ids: set[int],
) -> NumberDecision:
    obj = _load_json_strict(text)
    expected = {"number", "influential_agent_ids"} if m == 1 else {"number", "reason", "influential_agent_ids"}
    if set(obj) != expected:
        raise ParseError(f"response must contain exactly {sorted(expected)!r}")
    number = _parse_number(obj.get("number"))
    raw_ids = obj.get("influential_agent_ids")
    if not isinstance(raw_ids, list):
        raise ParseError("influential_agent_ids must be a JSON list")
    influential_ids: list[int] = []
    for item in raw_ids:
        if not isinstance(item, int):
            raise ParseError("influential_agent_ids entries must be integers")
        if item not in valid_agent_ids:
            raise ParseError(f"influential_agent_ids entry {item} is not visible in this round")
        influential_ids.append(item)
    if len(influential_ids) != len(set(influential_ids)):
        raise ParseError("influential_agent_ids must be unique")
    if len(influential_ids) > max_influential_agents:
        raise ParseError(f"influential_agent_ids may contain at most {max_influential_agents} entries")
    if m == 1:
        return NumberDecision(number=number, influential_agent_ids=tuple(influential_ids))
    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ParseError("m=3 reason must be a non-empty string")
    return NumberDecision(number=number, influential_agent_ids=tuple(influential_ids), reason=reason.strip())

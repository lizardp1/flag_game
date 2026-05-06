from __future__ import annotations

import json
from typing import Any


def observer_system_prompt() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, "
        "and no text outside the JSON object.\n"
        "You are an observer in a flag identification game.\n"
        "You receive a private crop of the flag and report what you see to the manager.\n"
        "The manager's country decisions are shared back to observers as memory.\n"
        "Follow the exact output schema given in the user message."
    )


def manager_system_prompt() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, "
        "and no text outside the JSON object.\n"
        "You are the manager in a flag identification game.\n"
        "Use observer JSON reports and your prior country decisions to choose the flag country.\n"
        "Follow the exact output schema given in the user message."
    )


def system_prompt() -> str:
    return observer_system_prompt()


def _memory_block(memory_lines: list[str], *, label: str) -> str:
    if not memory_lines:
        return f"{label} (oldest -> newest): []"
    joined = "\n".join(f"- {line}" for line in memory_lines)
    return f"{label} (oldest -> newest):\n{joined}"


def _observer_block(observer_statement_lines: list[str]) -> str:
    if not observer_statement_lines:
        return "[]"
    joined = ",\n  ".join(observer_statement_lines)
    return f"[\n  {joined}\n]"


def _observer_schema(m: int) -> str:
    if m != 3:
        raise ValueError("flag_game_org currently uses interaction m=3")
    return (
        'Output JSON exactly: '
        '{"country":"<one allowed country>","reason":"<one sentence describing what you see>"}'
    )


def _decision_schema(m: int) -> str:
    if m != 3:
        raise ValueError("flag_game_org currently uses interaction m=3")
    return (
        'Output JSON exactly: '
        '{"country":"<one allowed country>","reason":"<one sentence>"}'
    )


def observer_statement_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    m: int,
) -> str:
    lines = [
        "All observers are looking at private crops from the same underlying flag.",
        "You are an observer. You see a private crop.",
        "The manager cannot see the flag or any private crop images.",
        "Your report is the only visual evidence the manager gets from your crop.",
        "Report your best country guess.",
        f"Allowed countries: {json.dumps(countries, ensure_ascii=True)}",
        _memory_block(memory_lines, label="Manager's country decisions"),
        _observer_schema(m),
    ]
    return "\n".join(lines)


def aggregator_decision_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    observer_statement_lines: list[str],
    m: int,
) -> str:
    lines = [
        "You are the manager and final decision maker.",
        "All observers are looking at private crops from the same underlying flag.",
        "Use their different crop reports together to identify that one flag country.",
        f"Allowed countries: {json.dumps(countries, ensure_ascii=True)}",
        _memory_block(memory_lines, label="Your prior decisions"),
        "Observer JSON:",
        f"{_observer_block(observer_statement_lines)}",
    ]
    lines.append(_decision_schema(m))
    return "\n".join(lines)


def observer_statement_retry_text(
    *,
    countries: list[str],
    m: int,
    error_text: str,
) -> str:
    return (
        f"Invalid answer: {error_text}\n"
        f"Allowed countries are exactly: {json.dumps(countries, ensure_ascii=True)}\n"
        "Choose exactly one allowed country from that list. Any other country is invalid.\n"
        f"{_observer_schema(m)}"
    )


def aggregator_decision_retry_text(
    *,
    countries: list[str],
    m: int,
    error_text: str,
) -> str:
    return (
        f"Invalid answer: {error_text}\n"
        f"Allowed countries are exactly: {json.dumps(countries, ensure_ascii=True)}\n"
        "Choose exactly one allowed country from that list. Any other country is invalid.\n"
        f"{_decision_schema(m)}"
    )


def openai_text_messages(*, text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": manager_system_prompt()},
        {"role": "user", "content": text},
    ]


def openai_multimodal_messages(
    *,
    text: str,
    crop_data_uri: str,
    image_detail: str = "original",
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": observer_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": crop_data_uri, "detail": image_detail}},
            ],
        },
    ]

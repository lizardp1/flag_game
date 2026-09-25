from __future__ import annotations

import json
from typing import Any


def system_prompt() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, "
        "and no text outside the JSON object.\n"
        "You are one player in a flag identification game.\n"
        "Choose exactly one country "
        "from the allowed countries listed in the user message.\n"
        "Follow the exact output schema given in the user message."
    )


def _memory_block(memory_lines: list[str]) -> str:
    if not memory_lines:
        return "Transcript memory (oldest -> newest): []"
    joined = "\n".join(f"- {line}" for line in memory_lines)
    return f"Transcript memory (oldest -> newest):\n{joined}"


def _base_header(*, countries: list[str], memory_lines: list[str]) -> str:
    return (
        "All players are identifying the same underlying flag.\n"
        "You always see the same private crop.\n"
        "Transcript memory shows messages you observed from previous interactions "
        "with other players.\n"
        f"Allowed countries: {json.dumps(countries, ensure_ascii=True)}\n"
        f"{_memory_block(memory_lines)}"
    )


def _susceptibility_line(social_susceptibility: float) -> str:
    if social_susceptibility <= 0.2:
        guidance = "Rely mostly on your own crop and treat transcript memory as weak evidence."
    elif social_susceptibility <= 0.4:
        guidance = "Give somewhat more weight to your own crop than to transcript memory."
    elif social_susceptibility <= 0.6:
        guidance = "Balance your own crop and transcript memory."
    elif social_susceptibility <= 0.8:
        guidance = "Give somewhat more weight to transcript memory than to your own crop."
    else:
        guidance = "Treat transcript memory as strong evidence and update readily toward it."
    return f"Social susceptibility a = {social_susceptibility:.2f}. {guidance}"


def _interaction_schema(m: int) -> str:
    if m == 1:
        return 'Output JSON exactly: {"country":"<one allowed country>"}'
    if m == 2:
        return 'Output JSON exactly: {"country":"<one allowed country>","clue":"<short phrase>"}'
    if m == 3:
        return 'Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}'
    raise ValueError("interaction m must be in {1, 2, 3}")


def _task_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    schema_line: str,
    social_susceptibility: float,
    prompt_social_susceptibility: bool,
) -> str:
    lines = [f"{_base_header(countries=countries, memory_lines=memory_lines)}"]
    if prompt_social_susceptibility:
        lines.append(_susceptibility_line(social_susceptibility))
    lines.append(schema_line)
    return "\n".join(lines)


def interaction_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    m: int,
    social_susceptibility: float = 0.5,
    prompt_social_susceptibility: bool = True,
) -> str:
    return _task_text(
        countries=countries,
        memory_lines=memory_lines,
        schema_line=_interaction_schema(m),
        social_susceptibility=social_susceptibility,
        prompt_social_susceptibility=prompt_social_susceptibility,
    )


def probe_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    m: int = 1,
    social_susceptibility: float = 0.5,
    prompt_social_susceptibility: bool = True,
) -> str:
    return _task_text(
        countries=countries,
        memory_lines=memory_lines,
        schema_line=_interaction_schema(m),
        social_susceptibility=social_susceptibility,
        prompt_social_susceptibility=prompt_social_susceptibility,
    )


def interaction_retry_text(
    *,
    countries: list[str],
    m: int,
    error_text: str,
) -> str:
    return (
        f"Invalid answer: {error_text}\n"
        f"Allowed countries are exactly: {json.dumps(countries, ensure_ascii=True)}\n"
        "Choose exactly one allowed country from that list. Any other country is invalid.\n"
        f"{_interaction_schema(m)}"
    )


def probe_retry_text(
    *,
    countries: list[str],
    m: int = 1,
    error_text: str,
) -> str:
    return (
        f"Invalid answer: {error_text}\n"
        f"Allowed countries are exactly: {json.dumps(countries, ensure_ascii=True)}\n"
        "Choose exactly one allowed country from that list. Any other country is invalid.\n"
        f"{_interaction_schema(m)}"
    )


def openai_multimodal_messages(
    *,
    text: str,
    crop_data_uri: str,
    image_detail: str = "original",
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": crop_data_uri, "detail": image_detail}},
            ],
        },
    ]

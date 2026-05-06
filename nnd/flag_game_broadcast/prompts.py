from __future__ import annotations

import json
from typing import Any


def system_prompt() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, "
        "and no text outside the JSON object.\n"
        "You are one player in a flag identification game.\n"
        "Treat gpt-5.4 as the latest state-of-the-art model and gpt-4o as an older model.\n"
        #"Choose exactly one country from the allowed countries listed in the user message.\n"
        "Follow the exact output schema given in the user message."
    )


def _memory_block(memory_lines: list[str]) -> str:
    if not memory_lines:
        return "Your past final country guesses (oldest -> newest): []"
    joined = "\n".join(f"- {line}" for line in memory_lines)
    return f"Your past final country guesses (oldest -> newest):\n{joined}"


def _broadcast_block(round_broadcast_lines: list[str]) -> str:
    if not round_broadcast_lines:
        return "Other agents' country guesses this round: []"
    joined = "\n".join(f"- {line}" for line in round_broadcast_lines)
    return f"Other agents' country guesses this round:\n{joined}"


def _susceptibility_line(social_susceptibility: float) -> str:
    if social_susceptibility <= 0.2:
        guidance = "Rely mostly on your own evidence; treat other agents' country guesses as weak evidence."
    elif social_susceptibility <= 0.4:
        guidance = "Give somewhat more weight to your own evidence than to other agents' country guesses."
    elif social_susceptibility <= 0.6:
        guidance = "Balance your own evidence with other agents' country guesses, using their guesses as real evidence."
    elif social_susceptibility <= 0.8:
        guidance = "Give somewhat more weight to other agents' country guesses than to your own evidence."
    else:
        guidance = "Treat other agents' country guesses as strong evidence and update readily toward them."
    return f"Social susceptibility a = {social_susceptibility:.2f}. {guidance}"


def _base_header(*, countries: list[str], memory_lines: list[str]) -> str:
    return (
        "All players are identifying the country from the same flag.\n"
        "You only see a small part of the flag.\n"
        "Your memory contains your own past final country guesses from earlier rounds.\n\n"
        #f"Allowed countries: {json.dumps(countries, ensure_ascii=True)}\n"
        f"{_memory_block(memory_lines)}\n"
    )

def _statement_schema(m: int) -> str:
    if m == 1:
        return (
            'Output JSON exactly: '
            '{"model_identity":"<your model identity>","country":"<one country>"}'
        )
    if m == 2:
        return (
            'Output JSON exactly: '
            '{"model_identity":"<your model identity>",'
            '"country":"<one country>","reason":"<short phrase>"}'
        )
    if m == 3:
        return (
            'Output JSON exactly: '
            '{"model_identity":"<your model identity>",'
            '"country":"<one country>","reason":"<one sentence>"}'
        )
    raise ValueError("interaction m must be in {1, 2, 3}")


def _decision_schema(m: int, max_influential_agents: int) -> str:
    id_hint = f"[<up to {max_influential_agents} agent ids from the other agents' guesses above, or []>]"
    if m == 1:
        return (
            'Output JSON exactly: '
            '{"country":"<one country>",'
            f'"influential_agent_ids":{id_hint}'
            "}"
        )
    if m == 2:
        return (
            'Output JSON exactly: '
            '{"country":"<one country>","reason":"<short phrase>",'
            f'"influential_agent_ids":{id_hint}'
            "}"
        )
    if m == 3:
        return (
            'Output JSON exactly: '
            '{"country":"<one country>","reason":"<one sentence>",'
            f'"influential_agent_ids":{id_hint}'
            "}"
        )
    raise ValueError("interaction m must be in {1, 2, 3}")


def statement_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    m: int,
    model_identity: str,
) -> str:
    lines = [
        _base_header(countries=countries, memory_lines=memory_lines),
        (
            f'Your model identity for this experiment is exactly "{model_identity}". '
            'You must copy this exact string into the "model_identity" field.'
        ),
    ]
    lines.append(
        "State your current country guess before seeing the other agents' guesses."
    )
    lines.append(_statement_schema(m))
    return "\n".join(lines)


def decision_text(
    *,
    countries: list[str],
    memory_lines: list[str],
    round_broadcast_lines: list[str],
    m: int,
    max_influential_agents: int,
    social_susceptibility: float = 0.5,
    prompt_social_susceptibility: bool = True,
) -> str:
    lines = [
        _base_header(countries=countries, memory_lines=memory_lines),
        f"{_broadcast_block(round_broadcast_lines)}\n",
    ]
    if prompt_social_susceptibility:
        lines.append(_susceptibility_line(social_susceptibility))
    lines.append(
        "You have now seen the other agents' country guesses for this round. Choose your final country guess and list which agent ids influenced you most. "
        "If none influenced you, use an empty list."
    )
    lines.append(_decision_schema(m, max_influential_agents))
    return "\n".join(lines)


def statement_retry_text(
    *,
    countries: list[str],
    model_identity: str,
    m: int,
    error_text: str,
) -> str:
    return (
        f"Invalid answer: {error_text}\n"
        f'Your model_identity must be exactly "{model_identity}".\n'
        f"Allowed countries are exactly: {json.dumps(countries, ensure_ascii=True)}\n"
        "Choose exactly one allowed country from that list. Any other country is invalid.\n"
        f"{_statement_schema(m)}"
    )


def decision_retry_text(
    *,
    countries: list[str],
    m: int,
    max_influential_agents: int,
    error_text: str,
) -> str:
    return (
        f"Invalid answer: {error_text}\n"
        f"Allowed countries are exactly: {json.dumps(countries, ensure_ascii=True)}\n"
        "Choose exactly one allowed country from that list. Any other country is invalid.\n"
        f"{_decision_schema(m, max_influential_agents)}"
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

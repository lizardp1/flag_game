from __future__ import annotations


def system_prompt() -> str:
    return (
        "You must output only valid JSON. No extra keys, no markdown, and no text outside the JSON object.\n"
        "You are one player in a collaborative number identification game.\n"
        "Choose exactly one integer answer.\n"
        "Follow the exact output schema given in the user message."
    )


def _memory_block(memory_lines: list[str]) -> str:
    if not memory_lines:
        return "Transcript memory (oldest -> newest): []"
    return "Transcript memory (oldest -> newest):\n" + "\n".join(f"- {line}" for line in memory_lines)


def _validate_memory_lines(memory_lines: list[str], m: int) -> None:
    if m == 1:
        bad = [line for line in memory_lines if " | " in line]
        if bad:
            raise ValueError("m=1 transcript memory must contain number-only entries")
        return
    if m == 3:
        bad = [line for line in memory_lines if " | " not in line]
        if bad:
            raise ValueError("m=3 transcript memory must contain entries formatted as 'number | reason'")
        return
    raise ValueError("number game supports m=1 or m=3")


def _range_line(numbers: list[int]) -> str:
    if not numbers:
        return "The hidden integer is in the experiment's configured integer range."
    return f"The hidden integer is in the range {min(numbers)} through {max(numbers)}, inclusive."


def _base_header(
    *,
    numbers: list[int],
    private_clue: str,
    memory_lines: list[str],
    prompt_number_range: bool = False,
) -> str:
    lines = [
        "All players are identifying the same hidden integer.",
        "Each player receives a different private clue that is true of the hidden integer.",
    ]
    if prompt_number_range:
        lines.append(_range_line(numbers))
    lines.extend(
        [
            f"Your private clue: {private_clue}.",
            "Transcript memory shows messages you observed from previous interactions with other players.",
            _memory_block(memory_lines),
        ]
    )
    return "\n".join(lines)


def _susceptibility_line(social_susceptibility: float) -> str:
    if social_susceptibility <= 0.2:
        guidance = "Rely mostly on your private clue and treat transcript memory as weak evidence."
    elif social_susceptibility <= 0.4:
        guidance = "Give somewhat more weight to your private clue than to transcript memory."
    elif social_susceptibility <= 0.6:
        guidance = "Balance your private clue and transcript memory."
    elif social_susceptibility <= 0.8:
        guidance = "Give somewhat more weight to transcript memory than to your private clue."
    else:
        guidance = "Treat transcript memory as strong evidence and update readily toward it."
    return f"Social susceptibility a = {social_susceptibility:.2f}. {guidance}"


def message_schema(m: int) -> str:
    if m == 1:
        return 'Output JSON exactly: {"number":<integer>}'
    if m == 3:
        return 'Output JSON exactly: {"number":<integer>,"reason":"<one sentence>"}'
    raise ValueError("number game supports m=1 or m=3")


def decision_schema(m: int) -> str:
    if m == 1:
        return 'Output JSON exactly: {"number":<integer>,"influential_agent_ids":[<integer agent ids you used>]}'
    if m == 3:
        return 'Output JSON exactly: {"number":<integer>,"reason":"<one sentence>","influential_agent_ids":[<integer agent ids you used>]}'
    raise ValueError("number game supports m=1 or m=3")


def interaction_text(
    *,
    numbers: list[int],
    private_clue: str,
    memory_lines: list[str],
    m: int,
    social_susceptibility: float = 0.5,
    prompt_social_susceptibility: bool = True,
    prompt_number_range: bool = False,
) -> str:
    _validate_memory_lines(memory_lines, m)
    lines = [
        _base_header(
            numbers=numbers,
            private_clue=private_clue,
            memory_lines=memory_lines,
            prompt_number_range=prompt_number_range,
        )
    ]
    if prompt_social_susceptibility:
        lines.append(_susceptibility_line(social_susceptibility))
    lines.append(message_schema(m))
    return "\n".join(lines)


def broadcast_text(**kwargs: object) -> str:
    return interaction_text(**kwargs)  # type: ignore[arg-type]


def final_decision_text(
    *,
    numbers: list[int],
    private_clue: str,
    memory_lines: list[str],
    broadcast_lines: list[str],
    m: int,
    max_influential_agents: int,
    social_susceptibility: float = 0.5,
    prompt_social_susceptibility: bool = True,
    prompt_number_range: bool = False,
) -> str:
    _validate_memory_lines(memory_lines, m)
    lines = [
        _base_header(
            numbers=numbers,
            private_clue=private_clue,
            memory_lines=memory_lines,
            prompt_number_range=prompt_number_range,
        ),
        "Current round broadcasts from other agents:\n"
        + ("\n".join(f"- {line}" for line in broadcast_lines) if broadcast_lines else "[]"),
        f"You may list at most {max_influential_agents} influential_agent_ids.",
    ]
    if prompt_social_susceptibility:
        lines.append(_susceptibility_line(social_susceptibility))
    lines.append(decision_schema(m))
    return "\n".join(lines)


def organization_decision_text(
    *,
    numbers: list[int],
    memory_lines: list[str],
    observer_statement_lines: list[str],
    m: int,
    prompt_number_range: bool = False,
) -> str:
    _validate_memory_lines(memory_lines, m)
    lines = [
        "You are the manager in a number identification organization.",
        "Observers each saw a private clue about the same hidden integer.",
        "Choose exactly one integer answer.",
    ]
    if prompt_number_range:
        lines.append(_range_line(numbers))
    lines.extend(
        [
            _memory_block(memory_lines),
            "Current observer statements:\n"
            + ("\n".join(f"- {line}" for line in observer_statement_lines) if observer_statement_lines else "[]"),
            message_schema(m),
        ]
    )
    return "\n".join(lines)


def retry_text(*, numbers: list[int], m: int, error_text: str, decision: bool = False) -> str:
    schema = decision_schema(m) if decision else message_schema(m)
    return (
        f"Invalid answer: {error_text}\n"
        "Return one integer answer using the required JSON schema.\n"
        f"{schema}"
    )


def openai_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": text},
    ]

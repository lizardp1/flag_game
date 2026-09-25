"""Backward-compatible alias for the canonical anonymous closed-list prompt.

The former variant was created only to turn the country list back on.  The
canonical broadcast prompt now has that constraint, so keeping a distinct
module would risk reintroducing prompt differences in future diagnostics.
"""

from nnd.flag_game_broadcast.prompts import (
    _base_header,
    _broadcast_block,
    _decision_schema,
    _memory_block,
    _statement_schema,
    _susceptibility_line,
    decision_retry_text,
    decision_text,
    openai_multimodal_messages,
    statement_retry_text,
    statement_text,
    system_prompt,
)

__all__ = [
    "_base_header",
    "_broadcast_block",
    "_decision_schema",
    "_memory_block",
    "_statement_schema",
    "_susceptibility_line",
    "decision_retry_text",
    "decision_text",
    "openai_multimodal_messages",
    "statement_retry_text",
    "statement_text",
    "system_prompt",
]

"""Turn-finalization helpers shared by tests and runtime utilities."""

from __future__ import annotations

import time
from typing import Any

from agent.models import MessageRole
from agent.state import AgentState


def finalize_assistant_turn_delta(state: AgentState) -> dict[str, Any]:
    """Return the state delta that appends the assistant reply to transcript."""

    response_text = str(state.get("response_text", "") or "").strip()
    finalize_done_at_monotonic = time.monotonic()

    if not response_text:
        return {
            "diagnostics": {
                "finalize_done_at_monotonic": finalize_done_at_monotonic,
            }
        }

    assistant_turn = {
        "role": MessageRole.ASSISTANT.value,
        "content": response_text,
        "response_style": state.get("response_style") or None,
    }
    return {
        "transcript": [assistant_turn],
        "diagnostics": {
            "finalize_done_at_monotonic": finalize_done_at_monotonic,
        },
    }


__all__ = ["finalize_assistant_turn_delta"]

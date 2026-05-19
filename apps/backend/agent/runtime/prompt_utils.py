"""Shared OpenAI runtime utility helpers."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState


def include_prompt_history(session: Any | None) -> bool:
    """Return whether prompts must carry recent transcript history."""

    return session is None


def strip_recent_history_from_prompt(prompt: str) -> str:
    marker = "Recent conversation:\n"
    current_marker = "\nCurrent user message:\n"
    start = prompt.find(marker)
    if start == -1:
        return prompt
    history_start = start + len(marker)
    end = prompt.find(current_marker, history_start)
    if end == -1:
        return prompt

    middle = prompt[history_start:end]
    preserved = ""
    for context_marker in (
        "\nRelevant context requested by the user:",
        "\nRelevant context from past sessions:",
        "\nPrivate memory context is available",
    ):
        context_start = middle.find(context_marker)
        if context_start != -1:
            preserved = middle[context_start:]
            break

    replacement = "(conversation history is provided by the SDK session)"
    return f"{prompt[:history_start]}{replacement}{preserved}{prompt[end:]}"


def final_output_text(output: Any, *, fallback: str = "") -> str:
    text = output if isinstance(output, str) and output else str(output or fallback)
    if not text:
        raise ValueError("OpenAI Agents SDK returned an empty text response.")
    return text


def chunk_from_sdk_event(event: Any) -> str | None:
    if getattr(event, "type", None) != "raw_response_event":
        return None

    data = getattr(event, "data", None)
    event_type = (
        data.get("type") if isinstance(data, dict) else getattr(data, "type", None)
    )
    if event_type != "response.output_text.delta":
        return None

    delta = (
        data.get("delta") if isinstance(data, dict) else getattr(data, "delta", None)
    )
    return str(delta or "") or None


def state_without_prompt_history(state: AgentState) -> AgentState:
    prompt_state = dict(state)
    prompt_state["transcript"] = []
    prompt_state["history"] = []
    return prompt_state

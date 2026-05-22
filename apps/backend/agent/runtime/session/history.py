"""Transcript and SDK-session history boundary helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from agent.models import Message, MessageRole
from agent.state import AgentState

PUBLIC_CONVERSATION_ROLES = {MessageRole.USER.value, MessageRole.ASSISTANT.value}


def include_prompt_history(session: Any | None) -> bool:
    """Return whether prompts must carry recent transcript history."""

    return session is None


def strip_recent_history_from_prompt(prompt: str) -> str:
    """Remove prompt-local chat history when the SDK session owns history."""

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


def state_without_prompt_history(state: AgentState) -> AgentState:
    """Return a state copy with prompt-local transcript/history removed."""

    prompt_state = dict(state)
    prompt_state["transcript"] = []
    prompt_state["history"] = []
    return cast(AgentState, prompt_state)


def messages_from_transcript(
    transcript: list[dict[str, Any]],
) -> list[Message]:
    """Materialize public user/assistant messages from app transcript entries."""

    messages: list[Message] = []
    for turn in transcript:
        if not isinstance(turn, Mapping):
            continue
        message = _message_from_mapping(turn)
        if message is not None:
            messages.append(message)
    return messages


def messages_from_sdk_session_items(items: list[Any]) -> list[Message]:
    """Convert SDK session items into public user/assistant chat messages."""

    messages: list[Message] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        message = _message_from_mapping(item)
        if message is not None:
            messages.append(message)
    return messages


def content_to_text(content: Any) -> str:
    """Normalize SDK string/list content into plain text."""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part.strip() for part in parts if part.strip())


def _message_from_mapping(item: Mapping[str, Any]) -> Message | None:
    role = str(item.get("role") or "")
    if role not in PUBLIC_CONVERSATION_ROLES:
        return None
    content = content_to_text(item.get("content")).strip()
    if not content:
        return None
    style = item.get("response_style") if role == MessageRole.ASSISTANT.value else None
    return Message(
        role=MessageRole(role),
        content=content,
        response_style=str(style) if isinstance(style, str) and style else None,
    )


__all__ = [
    "content_to_text",
    "include_prompt_history",
    "messages_from_sdk_session_items",
    "messages_from_transcript",
    "state_without_prompt_history",
    "strip_recent_history_from_prompt",
]

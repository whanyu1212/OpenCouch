"""Voice tool registry and dispatch orchestration."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from agent.memory.modes import resolve_effective_memory_mode
from agent.observability.decorators import trace_event, trace_span
from agent.observability.events import (
    VOICE_TOOL_COMPLETED,
    VOICE_TOOL_DISPATCH,
    VOICE_TOOL_FAILED,
)
from agent.voice.tools.context import VoiceToolDispatchContext
from agent.voice.tools.specs import VOICE_TOOL_SPECS_BY_NAME
from llm.base import BaseLLMClient

_RECENT_USER_TURN_LIMIT = 3
_MIN_USER_QUOTE_LENGTH = 8


def _normalize_voice_tool_result(result: object) -> dict[str, object]:
    if isinstance(result, BaseModel):
        return dict(result.model_dump(mode="json"))
    if isinstance(result, dict):
        return dict(result)
    return {"result": str(result)}


def _safe_trace_tool_name(tool_name: object) -> str:
    if isinstance(tool_name, str) and tool_name in VOICE_TOOL_SPECS_BY_NAME:
        return tool_name
    return "unsupported"


def _voice_tool_trace_attrs(
    _args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
    return {
        "voice_runtime": "openai_realtime",
        "tool_name": _safe_trace_tool_name(kwargs.get("tool_name")),
        "memory_mode": kwargs.get("memory_mode"),
    }


@trace_span(
    VOICE_TOOL_DISPATCH,
    attrs=_voice_tool_trace_attrs,
    record_error_message=False,
)
async def execute_voice_tool_call(
    *,
    runtime: Any,
    tool_name: str,
    arguments: dict[str, object],
    thread_id: str,
    user_id: str | None,
    current_user_message: str,
    transcript: list[dict[str, object]],
    client_turn_id: str | None = None,
    memory_mode: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, object]:
    """Execute one app-owned voice function tool call."""

    if tool_name not in VOICE_TOOL_SPECS_BY_NAME:
        trace_event(
            VOICE_TOOL_FAILED,
            {"tool_name": "unsupported", "error_type": "unsupported_tool"},
        )
        raise ValueError(f"Unsupported voice tool: {tool_name!r}")

    spec = VOICE_TOOL_SPECS_BY_NAME.get(tool_name)
    if spec is None:
        trace_event(
            VOICE_TOOL_FAILED,
            {"tool_name": tool_name, "error_type": "unhandled_tool"},
        )
        raise AssertionError(f"Unhandled voice tool: {tool_name!r}")

    effective_memory_mode = _effective_memory_mode(runtime, memory_mode)
    if effective_memory_mode == "incognito":
        if tool_name == "show_memory_status":
            trace_event(
                VOICE_TOOL_COMPLETED,
                {
                    "tool_name": tool_name,
                    "status": "completed",
                    "result_type": "incognito_memory_status",
                },
            )
            return _incognito_memory_status_result()
        if spec.persistent_only:
            trace_event(
                VOICE_TOOL_FAILED,
                {"tool_name": tool_name, "error_type": "incognito_unavailable"},
            )
            raise ValueError(f"{tool_name!r} is not available in incognito voice mode.")

    if spec.memory_mutator and not _has_owner_or_session_id(
        user_id=user_id,
        thread_id=thread_id,
    ):
        trace_event(
            VOICE_TOOL_COMPLETED,
            {
                "tool_name": tool_name,
                "status": "refused",
                "reason": "owner_or_session_missing",
            },
        )
        return _voice_mutator_refusal(
            reason="owner_or_session_missing",
            response_text=(
                "I can't change saved memory because this voice session does "
                "not include a user or session identifier."
            ),
        )

    if spec.intent_gated_mutator and not _user_quote_matches_turn(
        arguments=arguments,
        current_user_message=current_user_message,
        transcript=transcript,
    ):
        trace_event(
            VOICE_TOOL_COMPLETED,
            {
                "tool_name": tool_name,
                "status": "refused",
                "reason": "user_intent_not_verified",
            },
        )
        return _voice_mutator_refusal(
            reason="user_intent_not_verified",
            response_text=(
                "I can't change saved memory from voice unless the tool call "
                "includes exact recent user words that asked for that change."
            ),
        )

    if tool_name in {"start_guided_exercise", "record_guided_exercise_progress"}:
        await runtime.voice.prepare_voice_guided_exercise_tool(
            thread_id=thread_id,
            llm_client=llm_client,
        )

    tool_context = None
    if spec.requires_context:
        tool_context = await runtime.voice.build_voice_tool_context(
            thread_id=thread_id,
            user_id=user_id,
            current_user_message=current_user_message,
            transcript=transcript,
            client_turn_id=client_turn_id,
            llm_client=llm_client,
        )

    try:
        result = await spec.handler(
            VoiceToolDispatchContext(
                runtime=runtime,
                tool_context=tool_context,
                thread_id=thread_id,
                user_id=user_id,
                client_turn_id=client_turn_id,
            ),
            arguments,
        )
        normalized_result = _normalize_voice_tool_result(result)
        if spec.memory_mutator:
            await runtime.voice.persist_voice_memory_tool_result(
                thread_id=thread_id,
                user_id=user_id,
                current_user_message=current_user_message,
                transcript=transcript,
                result=normalized_result,
            )
        if tool_name in {"start_guided_exercise", "record_guided_exercise_progress"}:
            normalized_result = (
                await runtime.voice.persist_voice_guided_exercise_result(
                    thread_id=thread_id,
                    user_id=user_id,
                    current_user_message=current_user_message,
                    transcript=transcript,
                    result=normalized_result,
                    memory_mode=effective_memory_mode,
                )
            )
    except Exception as exc:
        trace_event(
            VOICE_TOOL_FAILED,
            {"tool_name": tool_name, "error_type": type(exc).__name__},
        )
        raise
    trace_event(
        VOICE_TOOL_COMPLETED,
        {
            "tool_name": tool_name,
            "status": "completed",
            "result_type": type(result).__name__,
            "result_key_count": len(normalized_result),
        },
    )
    return normalized_result


def _effective_memory_mode(runtime: Any, memory_mode: str | None) -> str:
    """Return the resolved binary memory mode for a tool dispatch.

    Delegates to ``resolve_effective_memory_mode`` so incognito is the
    floor: if the runtime is incognito, the request cannot escalate to
    persistent. Routes resolve this upstream; the helper stays as
    defense-in-depth for direct callers (tests, future channels).
    """

    return resolve_effective_memory_mode(
        getattr(runtime, "memory_mode", None), memory_mode
    )


def _incognito_memory_status_result() -> dict[str, object]:
    return {
        "response_text": (
            "Memory is off for this incognito voice session. I won't save or "
            "use durable memory for this voice session."
        ),
        "memory_mode": "incognito",
        "memory_control": {"memory_mode": "incognito"},
        "side_effect": "none",
        "retry_safe": True,
    }


def _has_owner_or_session_id(*, user_id: str | None, thread_id: str) -> bool:
    return bool((user_id or "").strip() or thread_id.strip())


def _user_quote_matches_turn(
    *,
    arguments: dict[str, object],
    current_user_message: str,
    transcript: list[dict[str, object]],
) -> bool:
    raw_quote = arguments.get("user_quote")
    user_quote = raw_quote if isinstance(raw_quote, str) else ""
    normalized_quote = _normalize_user_quote_text(user_quote)
    if len(normalized_quote) < _MIN_USER_QUOTE_LENGTH:
        return False

    evidence_text = _recent_user_evidence_text(
        current_user_message=current_user_message,
        transcript=transcript,
    )
    normalized_evidence = _normalize_user_quote_text(evidence_text)
    return bool(normalized_evidence and normalized_quote in normalized_evidence)


def _recent_user_evidence_text(
    *,
    current_user_message: str,
    transcript: list[dict[str, object]],
) -> str:
    parts: list[str] = []
    current = current_user_message.strip()
    if current:
        parts.append(current)

    user_turns: list[str] = []
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = turn.get("content")
        if content is None:
            continue
        text = str(content).strip()
        if text:
            user_turns.append(text)
    parts.extend(user_turns[-_RECENT_USER_TURN_LIMIT:])
    return " ".join(parts)


def _normalize_user_quote_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned.strip().lower())


def _voice_mutator_refusal(*, reason: str, response_text: str) -> dict[str, object]:
    return {
        "response_text": response_text,
        "refused": True,
        "reason": reason,
        "side_effect": "none",
        "retry_safe": True,
    }

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
from agent.voice.tools.context import VoiceToolDefinition, VoiceToolDispatchContext
from agent.voice.tools.handlers import (
    _handle_answer_grounded_lookup,
    _handle_cancel_memory_deletion,
    _handle_confirm_memory_deletion,
    _handle_get_crisis_support_template,
    _handle_list_guided_exercise_skills,
    _handle_load_guided_exercise_skill,
    _handle_load_therapeutic_response_skill,
    _handle_lookup_crisis_resources,
    _handle_prepare_memory_deletion_by_index,
    _handle_prepare_memory_deletion_by_query,
    _handle_recall_saved_memory,
    _handle_record_guided_exercise_progress,
    _handle_save_response_preference,
    _handle_set_proactive_memory_recall,
    _handle_show_memory_status,
    _handle_show_saved_memory,
    _handle_wait_for_user,
)
from agent.voice.tools.schemas import (
    _INTENT_GATED_MUTATOR_TOOL_NAMES,
    _PERSISTENT_ONLY_TOOL_NAMES,
    _SUPPORTED_VOICE_TOOL_NAMES,
    _VOICE_MEMORY_MUTATOR_TOOL_NAMES,
)
from llm.base import BaseLLMClient

_RECENT_USER_TURN_LIMIT = 3
_MIN_USER_QUOTE_LENGTH = 8

_VOICE_TOOL_REGISTRY: dict[str, VoiceToolDefinition] = {
    "wait_for_user": VoiceToolDefinition(
        name="wait_for_user",
        handler=_handle_wait_for_user,
        requires_context=False,
    ),
    "show_memory_status": VoiceToolDefinition(
        name="show_memory_status",
        handler=_handle_show_memory_status,
    ),
    "show_saved_memory": VoiceToolDefinition(
        name="show_saved_memory",
        handler=_handle_show_saved_memory,
    ),
    "recall_saved_memory": VoiceToolDefinition(
        name="recall_saved_memory",
        handler=_handle_recall_saved_memory,
    ),
    "set_proactive_memory_recall": VoiceToolDefinition(
        name="set_proactive_memory_recall",
        handler=_handle_set_proactive_memory_recall,
    ),
    "save_response_preference": VoiceToolDefinition(
        name="save_response_preference",
        handler=_handle_save_response_preference,
    ),
    "prepare_memory_deletion_by_index": VoiceToolDefinition(
        name="prepare_memory_deletion_by_index",
        handler=_handle_prepare_memory_deletion_by_index,
    ),
    "prepare_memory_deletion_by_query": VoiceToolDefinition(
        name="prepare_memory_deletion_by_query",
        handler=_handle_prepare_memory_deletion_by_query,
    ),
    "confirm_memory_deletion": VoiceToolDefinition(
        name="confirm_memory_deletion",
        handler=_handle_confirm_memory_deletion,
    ),
    "cancel_memory_deletion": VoiceToolDefinition(
        name="cancel_memory_deletion",
        handler=_handle_cancel_memory_deletion,
    ),
    "answer_grounded_lookup": VoiceToolDefinition(
        name="answer_grounded_lookup",
        handler=_handle_answer_grounded_lookup,
    ),
    "lookup_crisis_resources": VoiceToolDefinition(
        name="lookup_crisis_resources",
        handler=_handle_lookup_crisis_resources,
    ),
    "get_crisis_support_template": VoiceToolDefinition(
        name="get_crisis_support_template",
        handler=_handle_get_crisis_support_template,
    ),
    "list_guided_exercise_skills": VoiceToolDefinition(
        name="list_guided_exercise_skills",
        handler=_handle_list_guided_exercise_skills,
    ),
    "load_therapeutic_response_skill": VoiceToolDefinition(
        name="load_therapeutic_response_skill",
        handler=_handle_load_therapeutic_response_skill,
    ),
    "load_guided_exercise_skill": VoiceToolDefinition(
        name="load_guided_exercise_skill",
        handler=_handle_load_guided_exercise_skill,
    ),
    "record_guided_exercise_progress": VoiceToolDefinition(
        name="record_guided_exercise_progress",
        handler=_handle_record_guided_exercise_progress,
    ),
}


def _registered_voice_tool_names() -> set[str]:
    return set(_VOICE_TOOL_REGISTRY)


def _normalize_voice_tool_result(result: object) -> dict[str, object]:
    if isinstance(result, BaseModel):
        return dict(result.model_dump(mode="json"))
    if isinstance(result, dict):
        return dict(result)
    return {"result": str(result)}


def _safe_trace_tool_name(tool_name: object) -> str:
    if isinstance(tool_name, str) and tool_name in _SUPPORTED_VOICE_TOOL_NAMES:
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
    memory_mode: str | None = None,
    llm_client: BaseLLMClient | None = None,
) -> dict[str, object]:
    """Execute one app-owned voice function tool call."""

    if tool_name not in _SUPPORTED_VOICE_TOOL_NAMES:
        trace_event(
            VOICE_TOOL_FAILED,
            {"tool_name": "unsupported", "error_type": "unsupported_tool"},
        )
        raise ValueError(f"Unsupported voice tool: {tool_name!r}")

    definition = _VOICE_TOOL_REGISTRY.get(tool_name)
    if definition is None:
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
        if tool_name in _PERSISTENT_ONLY_TOOL_NAMES:
            trace_event(
                VOICE_TOOL_FAILED,
                {"tool_name": tool_name, "error_type": "incognito_unavailable"},
            )
            raise ValueError(f"{tool_name!r} is not available in incognito voice mode.")

    if tool_name in _VOICE_MEMORY_MUTATOR_TOOL_NAMES and not _has_owner_or_session_id(
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

    if tool_name in _INTENT_GATED_MUTATOR_TOOL_NAMES and not _user_quote_matches_turn(
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

    if tool_name == "record_guided_exercise_progress":
        await runtime.voice.prepare_voice_guided_exercise_progress(
            thread_id=thread_id,
            llm_client=llm_client,
        )

    tool_context = None
    if definition.requires_context:
        tool_context = await runtime.voice.build_voice_tool_context(
            thread_id=thread_id,
            user_id=user_id,
            current_user_message=current_user_message,
            transcript=transcript,
            llm_client=llm_client,
        )

    try:
        result = await definition.handler(
            VoiceToolDispatchContext(
                runtime=runtime,
                tool_context=tool_context,
                thread_id=thread_id,
                user_id=user_id,
            ),
            arguments,
        )
        normalized_result = _normalize_voice_tool_result(result)
        if tool_name in _VOICE_MEMORY_MUTATOR_TOOL_NAMES:
            await runtime.voice.persist_voice_memory_tool_result(
                thread_id=thread_id,
                user_id=user_id,
                current_user_message=current_user_message,
                transcript=transcript,
                result=normalized_result,
            )
        if tool_name == "record_guided_exercise_progress":
            await runtime.voice.persist_voice_guided_exercise_progress(
                thread_id=thread_id,
                user_id=user_id,
                current_user_message=current_user_message,
                transcript=transcript,
                result=normalized_result,
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

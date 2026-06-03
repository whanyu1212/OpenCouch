"""Shared turn-state helpers for the OpenAI text runtime."""

from __future__ import annotations

import time
from hashlib import sha256
from typing import Any, cast

from agent.models import MessageRole
from agent.observability.decorators import trace_event
from agent.observability.diagnostics import diagnostics_from_state, merge_diagnostics
from agent.observability.events import RUNTIME_TEXT_TURN_FINALIZED
from agent.runtime.types import TextRuntimeShadowResult, TextRuntimeShadowStatus
from agent.state import AgentState


DICT_REDUCER_KEYS = {
    "session_memory",
    "procedural_profile",
    "session_progress",
    "exercise_state",
    "memory_control",
    "grounded_lookup",
    "diagnostics",
}


def apply_state_delta(state: AgentState, delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        if key in DICT_REDUCER_KEYS:
            state[key] = cast(
                Any,
                {
                    **dict(state.get(key, {}) or {}),
                    **dict(value or {}),
                },
            )
        else:
            state[key] = cast(Any, value)


def route_for_runtime_mode(runtime_mode: str) -> str | None:
    if runtime_mode in {"safe_therapeutic", "crisis_clarification"}:
        return "therapeutic"
    if runtime_mode == "memory_control":
        return "memory_control"
    if runtime_mode == "grounded_lookup":
        return "grounded_lookup"
    if runtime_mode == "crisis_response":
        return "crisis"
    return None


def finalize_openai_turn(
    state: AgentState,
    *,
    response_text: str,
    runtime_mode: str,
    response_style: str,
    selected_agent: str | None,
    sdk_duration_ms: float | None,
    streamed: bool,
) -> AgentState:
    assistant_turn = {
        "role": MessageRole.ASSISTANT.value,
        "content": response_text,
        "response_style": response_style,
    }
    diagnostics = merge_diagnostics(
        diagnostics_from_state(state),
        {
            "text_agent_runtime": "openai",
            "openai_text_runtime_mode": runtime_mode,
            "openai_selected_agent": selected_agent,
            "openai_streamed": streamed,
            "finalize_done_at_monotonic": time.monotonic(),
        },
    )
    if sdk_duration_ms is not None:
        diagnostics["openai_sdk_ms"] = round(sdk_duration_ms, 2)

    trace_event(
        RUNTIME_TEXT_TURN_FINALIZED,
        {
            "runtime_mode": runtime_mode,
            "response_style": response_style,
            "selected_agent": selected_agent,
            "streamed": streamed,
            "sdk_duration_ms": round(sdk_duration_ms, 2)
            if sdk_duration_ms is not None
            else None,
        },
    )
    route = route_for_runtime_mode(runtime_mode)
    final_values: dict[str, Any] = {
        **dict(state),
        "response_text": response_text,
        "response_style": response_style,
        "diagnostics": diagnostics,
        "transcript": [*list(state.get("transcript", [])), assistant_turn],
    }
    if route is not None:
        final_values["route"] = route
    if runtime_mode == "safe_therapeutic":
        final_values.update(
            {
                "therapeutic_approach": state.get("therapeutic_approach") or "none",
                "session_action": "none",
                "should_persist_memory": False,
            }
        )
    elif runtime_mode == "crisis_clarification":
        final_values.update(
            {
                "therapeutic_approach": "none",
                "session_action": "none",
                "should_persist_memory": False,
            }
        )
    return cast(AgentState, final_values)


def build_shadow_result(
    prepared: Any,
    *,
    status: TextRuntimeShadowStatus,
    selected_agent: str | None = None,
    sdk_duration_ms: float | None = None,
    shadow_duration_ms: float | None = None,
    response_text: str | None = None,
) -> TextRuntimeShadowResult:
    assessment = prepared.state.get("crisis")
    summary = response_text_summary(response_text)
    memory_reference = prepared.state.get("memory_reference", {}) or {}
    memory_reference_mode = (
        memory_reference.get("mode") if isinstance(memory_reference, dict) else None
    )
    grounded_lookup = prepared.state.get("grounded_lookup", {}) or {}
    grounded_lookup_query = (
        grounded_lookup.get("query") if isinstance(grounded_lookup, dict) else None
    )
    return TextRuntimeShadowResult(
        runtime="openai",
        status=status,
        eligible=prepared.eligible,
        fallback_reason=prepared.fallback_reason or None,
        route=prepared.state.get("route"),
        memory_reference_mode=(
            str(memory_reference_mode) if memory_reference_mode is not None else None
        ),
        grounded_lookup_query=(
            str(grounded_lookup_query) if grounded_lookup_query else None
        ),
        crisis_level=getattr(assessment, "level", None),
        needs_crisis_response=getattr(assessment, "needs_crisis_response", None),
        needs_crisis_clarification=getattr(assessment, "needs_clarification", None),
        selected_agent=selected_agent,
        sdk_duration_ms=sdk_duration_ms,
        shadow_duration_ms=shadow_duration_ms,
        **summary,
    )


def response_text_summary(text: str | None) -> dict[str, Any]:
    if not text:
        return {
            "response_text_length": None,
            "response_text_preview": None,
            "response_text_sha256": None,
        }
    return {
        "response_text_length": len(text),
        "response_text_preview": text[:160],
        "response_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
    }

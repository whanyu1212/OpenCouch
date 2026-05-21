"""Therapeutic execution path helpers for the OpenAI text runtime."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

from openai import APIConnectionError, AuthenticationError, OpenAIError

from agent.observability.timing import elapsed_ms
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.specialists.therapeutic_prompts import (
    _format_working_memory,
    build_clarifying_system_prompt,
    build_supportive_system_prompt,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.session.state import format_recent_history
from agent.runtime.prompt_utils import (
    chunk_from_sdk_event,
    final_output_text,
    include_prompt_history,
)
from agent.runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient


_OPENAI_API_KEY_FALLBACK_REASON = "missing_openai_api_key"
_OPENAI_CONNECTION_FALLBACK_REASON = "openai_api_connection_error"


@dataclass(frozen=True)
class TherapeuticAgentResult:
    response_text: str
    runtime_mode: str
    response_style: str
    sdk_duration_ms: float


async def run_therapeutic_response_llm_turn(
    runtime: Any,
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
    session: Any | None,
    fallback_reason: str | None = None,
) -> TherapeuticAgentResult:
    from agent.runtime.state_ops import apply_state_delta

    run_start = time.monotonic()
    response_text = await llm_client.generate_text(
        prompt=runtime._input_text_for_state(
            state,
            include_recent_history=include_prompt_history(session),
        ),
        system_instruction=therapeutic_system_prompt_for_state(state),
    )
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_response_llm_override": True,
    }
    if fallback_reason is not None:
        diagnostics["openai_sdk_fallback_reason"] = fallback_reason
    apply_state_delta(state, {"diagnostics": diagnostics})
    return TherapeuticAgentResult(
        response_text=response_text,
        runtime_mode="safe_therapeutic",
        response_style=response_style_from_state(state),
        sdk_duration_ms=elapsed_ms(run_start),
    )


async def run_therapeutic_response_llm_stream(
    runtime: Any,
    state: AgentState,
    *,
    config: Any,
    llm_client: BaseLLMClient,
    session: Any | None,
    fallback_reason: str | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    from agent.runtime.state_ops import apply_state_delta

    run_start = time.monotonic()
    chunks: list[str] = []
    async for chunk in llm_client.generate_text_stream(
        prompt=runtime._input_text_for_state(
            state,
            include_recent_history=include_prompt_history(session),
        ),
        system_instruction=therapeutic_system_prompt_for_state(state),
    ):
        chunks.append(chunk)
        if chunk:
            yield TextRuntimeChunkEvent(text=chunk)
    response_text = "".join(chunks)
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_response_llm_override": True,
    }
    if fallback_reason is not None:
        diagnostics["openai_sdk_fallback_reason"] = fallback_reason
    apply_state_delta(state, {"diagnostics": diagnostics})
    final_state = await runtime._finalize_openai_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode="safe_therapeutic",
        response_style=response_style_from_state(state),
        selected_agent=THERAPEUTIC_AGENT_NAME,
        sdk_duration_ms=elapsed_ms(run_start),
        streamed=True,
    )
    yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    yield TextRuntimeStateEvent(state=final_state)


async def run_therapeutic_turn(
    runtime: Any,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    session: Any | None = None,
) -> TherapeuticAgentResult:
    if context.response_llm is not None:
        return await run_therapeutic_response_llm_turn(
            runtime,
            state,
            llm_client=context.response_llm,
            session=session,
        )

    run_context = runtime._run_context_for_state(state, config, context)
    agent = runtime._build_agent(state)
    input_text = runtime._input_text_for_state(
        state,
        include_recent_history=include_prompt_history(session),
    )
    try:
        response_text, sdk_duration_ms = await runtime._run_openai_agent_with(
            state,
            agent=agent,
            input_text=input_text,
            run_context=run_context,
            session=session,
        )
    except Exception as exc:
        if not can_fallback_to_control_response(exc, context):
            raise
        return await run_therapeutic_response_llm_turn(
            runtime,
            state,
            llm_client=cast(BaseLLMClient, context.llm_client),
            session=session,
            fallback_reason=cast(str, openai_sdk_fallback_reason(exc)),
        )
    return resolve_therapeutic_result(
        state,
        run_context=run_context,
        response_text=response_text,
        sdk_duration_ms=sdk_duration_ms,
    )


async def run_therapeutic_turn_stream(
    runtime: Any,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    session: Any | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    if context.response_llm is not None:
        yield TextRuntimeStatusEvent(stage="therapeutic")
        async for event in run_therapeutic_response_llm_stream(
            runtime,
            state,
            config=config,
            llm_client=context.response_llm,
            session=session,
        ):
            yield event
        return

    run_context = runtime._run_context_for_state(state, config, context)
    agent = runtime._build_agent(state)
    input_text = runtime._input_text_for_state(
        state,
        include_recent_history=include_prompt_history(session),
    )

    yield TextRuntimeStatusEvent(stage="therapeutic")
    run_start = time.monotonic()
    chunks: list[str] = []
    try:
        stream = runtime._runner.run_streamed(
            agent=agent,
            input_text=input_text,
            context=run_context,
            session=session,
        )
        async for sdk_event in stream.stream_events():
            chunk = chunk_from_sdk_event(sdk_event)
            if chunk:
                chunks.append(chunk)
                yield TextRuntimeChunkEvent(text=chunk)
    except Exception as exc:
        if not can_fallback_to_control_response(exc, context):
            raise
        async for event in run_therapeutic_response_llm_stream(
            runtime,
            state,
            config=config,
            llm_client=cast(BaseLLMClient, context.llm_client),
            session=session,
            fallback_reason=cast(str, openai_sdk_fallback_reason(exc)),
        ):
            yield event
        return

    response_text = final_output_text(
        getattr(stream, "final_output", None),
        fallback="".join(chunks),
    )
    result = resolve_therapeutic_result(
        state,
        run_context=run_context,
        response_text=response_text,
        sdk_duration_ms=elapsed_ms(run_start),
    )
    final_state = await runtime._finalize_openai_turn(
        state,
        response_text=result.response_text,
        config=config,
        runtime_mode=result.runtime_mode,
        response_style=result.response_style,
        selected_agent=THERAPEUTIC_AGENT_NAME,
        sdk_duration_ms=result.sdk_duration_ms,
        streamed=True,
    )
    yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    yield TextRuntimeStateEvent(state=final_state)


def resolve_therapeutic_result(
    state: AgentState,
    *,
    run_context: OpenAITextRunContext,
    response_text: str,
    sdk_duration_ms: float,
) -> TherapeuticAgentResult:
    runtime_mode, response_style, resolved_text = merge_therapeutic_tool_results(
        state,
        run_context=run_context,
        response_text=response_text,
    )
    return TherapeuticAgentResult(
        response_text=resolved_text,
        runtime_mode=runtime_mode,
        response_style=response_style,
        sdk_duration_ms=sdk_duration_ms,
    )


def therapeutic_system_prompt_for_state(state: AgentState) -> str:
    if response_style_from_state(state) == "clarifying":
        return build_clarifying_system_prompt(state)
    return build_supportive_system_prompt(state)


def therapeutic_agent_prompt_for_state(state: AgentState) -> str:
    memory_block = _format_working_memory(state)
    return (
        "Write the next assistant message for a mental health support "
        "conversation.\n\n"
        "For an ordinary therapeutic reply, first call "
        "load_therapeutic_response_skill with the response_style that best fits "
        "this turn, then use the returned skill_context as private guidance. "
        "Do not expose internal style names unless the user asks how the system "
        "works. Do not start or continue guided exercises here; the runtime "
        "routes those turns to GuidedExerciseAgent.\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
    )


def can_fallback_to_control_response(
    exc: Exception,
    context: WorkflowContext,
) -> bool:
    return (
        context.llm_client is not None and openai_sdk_fallback_reason(exc) is not None
    )


def openai_sdk_fallback_reason(exc: Exception) -> str | None:
    if _is_missing_openai_api_key_error(exc):
        return _OPENAI_API_KEY_FALLBACK_REASON
    if isinstance(exc, APIConnectionError):
        return _OPENAI_CONNECTION_FALLBACK_REASON
    return None


def operational_context_for_prompt(state: AgentState) -> str:
    lines = [
        "Operational context:",
        "- The current turn has already passed the app-owned crisis gate.",
        "- For ordinary therapeutic replies, call "
        "load_therapeutic_response_skill before answering and use the returned "
        "skill_context as private style guidance.",
        "- Use memory or grounded lookup tools instead when the user explicitly "
        "asks for saved-memory management or grounded lookup.",
    ]
    memory_control = state.get("memory_control", {}) or {}
    pending_action = (
        memory_control.get("pending_action")
        if isinstance(memory_control, Mapping)
        else None
    )
    if isinstance(pending_action, Mapping):
        preview = ""
        target = pending_action.get("target")
        if isinstance(target, Mapping):
            preview = str(target.get("preview") or "").strip()
        pending_line = (
            "- Pending memory deletion exists. Call confirm_memory_deletion only "
            "if the user clearly confirms; call cancel_memory_deletion only if "
            "the user clearly declines."
        )
        if preview:
            pending_line = f"{pending_line} Target preview: {preview}"
        lines.append(pending_line)

    turn_lifecycle = state.get("turn_lifecycle", {}) or {}
    if (
        isinstance(turn_lifecycle, Mapping)
        and turn_lifecycle.get("triage_confidence") == "low"
    ):
        tentative_route = str(turn_lifecycle.get("tentative_route") or "").strip()
        if tentative_route:
            lines.append(
                "- The user's intent is ambiguous. Triage tentatively suggested "
                f"'{tentative_route}'. Clarify whether the user wants to proceed "
                "with that intent or continue the current flow before taking "
                "route-specific action."
            )

    memory_reference = state.get("memory_reference", {}) or {}
    if (
        isinstance(memory_reference, Mapping)
        and memory_reference.get("mode") == "explicit"
    ):
        lines.append(
            "- The user explicitly asked to use prior conversation context; use "
            "retrieved memory context when it is available."
        )

    return "\n".join(lines)


def response_style_from_state(state: Mapping[str, Any]) -> str:
    style = str(state.get("response_style") or "").strip()
    if style and style != "pending":
        return style
    return "supportive"


def merge_therapeutic_tool_results(
    state: AgentState,
    *,
    run_context: OpenAITextRunContext,
    response_text: str,
) -> tuple[str, str, str]:
    from agent.runtime.state_ops import apply_state_delta

    memory_calls = list(run_context.memory_tool_calls)
    grounded_calls = list(run_context.grounded_tool_calls)
    therapeutic_skill_calls = list(run_context.therapeutic_response_skill_tool_calls)
    diagnostics: dict[str, Any] = {
        **dict(state.get("diagnostics", {}) or {}),
    }

    for call in memory_calls:
        delta: dict[str, Any] = {"memory_control": call.memory_control}
        if call.procedural_profile is not None:
            delta["procedural_profile"] = call.procedural_profile
        apply_state_delta(state, delta)

    if memory_calls:
        latest_memory_call = memory_calls[-1]
        diagnostics.update(
            {
                "openai_memory_tool_expected": latest_memory_call.tool_name,
                "openai_memory_tool_selected": latest_memory_call.tool_name,
                "openai_memory_tool_calls": [call.tool_name for call in memory_calls],
                "openai_memory_tool_side_effects": [
                    call.side_effect for call in memory_calls
                ],
                "openai_memory_tool_fallback": False,
            }
        )

    for call in grounded_calls:
        apply_state_delta(state, {"grounded_lookup": call.grounded_lookup})

    if grounded_calls:
        latest_grounded_call = grounded_calls[-1]
        diagnostics.update(
            {
                "openai_grounded_tool_expected": latest_grounded_call.tool_name,
                "openai_grounded_tool_selected": latest_grounded_call.tool_name,
                "openai_grounded_tool_calls": [
                    call.tool_name for call in grounded_calls
                ],
                "openai_grounded_tool_fallback": False,
            }
        )

    if therapeutic_skill_calls:
        latest_therapeutic_skill_call = therapeutic_skill_calls[-1]
        apply_state_delta(
            state,
            {
                "response_style": latest_therapeutic_skill_call.response_style,
                "therapeutic_approach": (
                    latest_therapeutic_skill_call.therapeutic_approach
                ),
            },
        )
        diagnostics.update(
            {
                "openai_therapeutic_skill_tool_expected": (
                    "load_therapeutic_response_skill"
                ),
                "openai_therapeutic_skill_tool_selected": (
                    latest_therapeutic_skill_call.tool_name
                ),
                "openai_therapeutic_skill_tool_calls": [
                    call.tool_name for call in therapeutic_skill_calls
                ],
                "openai_therapeutic_skill_response_style": (
                    latest_therapeutic_skill_call.response_style
                ),
                "openai_therapeutic_skill_approach": (
                    latest_therapeutic_skill_call.therapeutic_approach
                ),
                "openai_therapeutic_skill_tool_fallback": False,
            }
        )

    apply_state_delta(state, {"diagnostics": diagnostics})

    if grounded_calls:
        apply_state_delta(state, {"route": "grounded_lookup"})
        return "grounded_lookup", "grounded_lookup", grounded_calls[-1].response_text
    if memory_calls:
        apply_state_delta(state, {"route": "memory_control"})
        return "memory_control", "memory_control", memory_calls[-1].response_text

    apply_state_delta(state, {"route": "therapeutic"})
    return "safe_therapeutic", response_style_from_state(state), response_text


def _is_missing_openai_api_key_error(exc: Exception) -> bool:
    if isinstance(exc, AuthenticationError):
        return True
    if not isinstance(exc, OpenAIError):
        return False
    message = str(exc)
    return (
        "OPENAI_API_KEY" in message and "api_key" in message
    ) or "Missing bearer or basic authentication in header" in message

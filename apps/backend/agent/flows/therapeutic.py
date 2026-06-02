"""Therapeutic execution path helpers for the OpenAI text runtime."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from pydantic import BaseModel, Field

from agent.flows.sdk_fallback import (
    can_fallback_to_control_response,
    openai_sdk_fallback_reason,
)
from agent.observability.timing import elapsed_ms
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.specialists.therapeutic_response.prompts import (
    _format_working_memory,
    build_clarifying_system_prompt,
    build_supportive_system_prompt,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.session.state import format_recent_history
from agent.runtime.prompt_utils import (
    chunk_from_sdk_event,
    final_output_text,
)
from agent.runtime.session.history import include_prompt_history
from agent.runtime.session.history import state_without_prompt_history
from agent.runtime.services import TextRuntimeServices
from agent.runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime.workflow_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient


@dataclass(frozen=True)
class TherapeuticAgentResult:
    response_text: str
    runtime_mode: str
    response_style: str
    sdk_duration_ms: float


@dataclass(frozen=True)
class TherapeuticToolMergeResult:
    runtime_mode: str
    response_style: str
    response_text: str


class TherapeuticResponseLLMOutput(BaseModel):
    response_text: str = Field(
        description=(
            "Final user-facing assistant message only. Do not include tool calls, "
            "JSON, XML tags, internal style names, or implementation traces."
        )
    )


@dataclass(frozen=True)
class ResponseLLMText:
    text: str
    sanitized: bool
    raw_text: str


@dataclass(frozen=True)
class TherapeuticResponseLLMRequest:
    prompt: str
    system_instruction: str


async def run_therapeutic_response_llm_turn(
    services: TextRuntimeServices,
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
    session: Any | None,
    fallback_reason: str | None = None,
) -> TherapeuticAgentResult:
    from agent.runtime.state_ops import apply_state_delta

    run_start = time.monotonic()
    request = therapeutic_response_llm_request_for_state(state, session=session)
    structured_output = await llm_client.generate_structured(
        prompt=request.prompt,
        response_schema=TherapeuticResponseLLMOutput,
        system_instruction=request.system_instruction,
    )
    response_text = response_llm_text_from_structured_output(structured_output)
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        **response_llm_diagnostics(
            response_text,
            structured=True,
            fallback_reason=fallback_reason,
        ),
    }
    apply_state_delta(state, {"diagnostics": diagnostics})
    return TherapeuticAgentResult(
        response_text=response_text.text,
        runtime_mode="safe_therapeutic",
        response_style=response_style_from_state(state),
        sdk_duration_ms=elapsed_ms(run_start),
    )


async def run_therapeutic_response_llm_stream(
    services: TextRuntimeServices,
    state: AgentState,
    *,
    config: Any,
    llm_client: BaseLLMClient,
    session: Any | None,
    fallback_reason: str | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    from agent.runtime.state_ops import apply_state_delta

    run_start = time.monotonic()
    request = therapeutic_response_llm_request_for_state(state, session=session)
    chunks: list[str] = []
    async for chunk in llm_client.generate_text_stream(
        prompt=request.prompt,
        system_instruction=request.system_instruction,
    ):
        chunks.append(chunk)
    response_text = sanitize_response_llm_text("".join(chunks))
    if response_text.text:
        yield TextRuntimeChunkEvent(text=response_text.text)
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        **response_llm_diagnostics(
            response_text,
            structured=False,
            fallback_reason=fallback_reason,
        ),
    }
    apply_state_delta(state, {"diagnostics": diagnostics})
    final_state = await services.finalize_turn(
        state,
        response_text=response_text.text,
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
    services: TextRuntimeServices,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    session: Any | None = None,
) -> TherapeuticAgentResult:
    if context.response_llm is not None:
        return await run_therapeutic_response_llm_turn(
            services,
            state,
            llm_client=context.response_llm,
            session=session,
        )

    run_context = services.build_run_context(state, config, context)
    agent = services.build_agent(state)
    input_text = services.input_text_for_state(
        state,
        include_recent_history=include_prompt_history(session),
    )
    try:
        response_text, sdk_duration_ms = await services.run_openai_agent_with(
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
            services,
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
    services: TextRuntimeServices,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    session: Any | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    if context.response_llm is not None:
        yield TextRuntimeStatusEvent(stage="therapeutic")
        async for event in run_therapeutic_response_llm_stream(
            services,
            state,
            config=config,
            llm_client=context.response_llm,
            session=session,
        ):
            yield event
        return

    run_context = services.build_run_context(state, config, context)
    agent = services.build_agent(state)
    input_text = services.input_text_for_state(
        state,
        include_recent_history=include_prompt_history(session),
    )

    yield TextRuntimeStatusEvent(stage="therapeutic")
    run_start = time.monotonic()
    chunks: list[str] = []
    try:
        stream = services.runner.run_streamed(
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
            services,
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
    final_state = await services.finalize_turn(
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
    merge_result = merge_therapeutic_tool_results(
        state,
        run_context=run_context,
        response_text=response_text,
    )
    return TherapeuticAgentResult(
        response_text=merge_result.response_text,
        runtime_mode=merge_result.runtime_mode,
        response_style=merge_result.response_style,
        sdk_duration_ms=sdk_duration_ms,
    )


def therapeutic_system_prompt_for_state(state: AgentState) -> str:
    if response_style_from_state(state) == "clarifying":
        return build_clarifying_system_prompt(state)
    return build_supportive_system_prompt(state)


def response_llm_prompt_for_state(
    state: AgentState,
    *,
    include_recent_history: bool = True,
) -> str:
    """Build the plain response-writer prompt for response LLM overrides."""

    prompt_state = (
        state if include_recent_history else state_without_prompt_history(state)
    )
    memory_block = _format_working_memory(prompt_state)
    return (
        "Write the next assistant message for a mental health support "
        "conversation.\n\n"
        "You are writing final user-facing text only. You do not have access "
        "to tools in this response-writing path. Do not emit tool calls, "
        "function names, JSON arguments, XML tags, internal style names, or "
        "implementation traces. Use any private context silently.\n\n"
        f"Recent conversation:\n{format_recent_history(prompt_state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {prompt_state['message']}"
    )


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


def therapeutic_response_llm_request_for_state(
    state: AgentState,
    *,
    session: Any | None,
) -> TherapeuticResponseLLMRequest:
    return TherapeuticResponseLLMRequest(
        prompt=response_llm_prompt_for_state(
            state,
            include_recent_history=include_prompt_history(session),
        ),
        system_instruction=therapeutic_system_prompt_for_state(state),
    )


def normalize_response_llm_text(text: str) -> str:
    return str(text or "").strip()


def response_llm_text_from_structured_output(
    output: TherapeuticResponseLLMOutput,
) -> ResponseLLMText:
    return sanitize_response_llm_text(normalize_response_llm_text(output.response_text))


def sanitize_response_llm_text(raw_text: str) -> ResponseLLMText:
    """Strip leading pseudo tool-call text from response-LLM output."""

    cleaned = str(raw_text or "").strip()
    sanitized = False
    while cleaned:
        stripped = _strip_leading_pseudo_tool_call(cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped.lstrip()
        sanitized = True
    if not cleaned:
        cleaned = str(raw_text or "").strip()
    return ResponseLLMText(text=cleaned, sanitized=sanitized, raw_text=raw_text)


def response_llm_sanitization_diagnostics(
    response_text: ResponseLLMText,
) -> dict[str, Any]:
    if not response_text.sanitized:
        return {"openai_response_llm_output_sanitized": False}
    raw_text = str(response_text.raw_text or "")
    return {
        "openai_response_llm_output_sanitized": True,
        "openai_response_llm_raw_text_length": len(raw_text),
        "openai_response_llm_raw_text_preview": raw_text[:160],
        "openai_response_llm_raw_text_sha256": sha256(raw_text.encode()).hexdigest(),
    }


def response_llm_diagnostics(
    response_text: ResponseLLMText,
    *,
    structured: bool,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "openai_response_llm_override": True,
        "openai_response_llm_output_structured": structured,
        "openai_response_llm_response_text_length": len(response_text.text),
    }
    diagnostics.update(response_llm_sanitization_diagnostics(response_text))
    if fallback_reason is not None:
        diagnostics["openai_sdk_fallback_reason"] = fallback_reason
    return diagnostics


def _strip_leading_pseudo_tool_call(text: str) -> str:
    stripped = text.lstrip()
    lower = stripped.lower()
    if lower.startswith("<tool_call"):
        close_index = lower.find("</tool_call>")
        if close_index != -1:
            return stripped[close_index + len("</tool_call>") :]

    marker = "load_therapeutic_response_skill"
    marker_index = stripped.find(marker)
    if marker_index == -1 or marker_index > 40:
        return text

    prefix = stripped[:marker_index].strip().lower()
    if prefix and not prefix.startswith("to="):
        return text

    json_start = stripped.find("{", marker_index)
    if json_start != -1:
        json_end = _find_matching_json_object_end(stripped, json_start)
        if json_end != -1:
            remainder = stripped[json_end + 1 :]
            if remainder.startswith(")"):
                remainder = remainder[1:]
            return remainder

    paren_end = stripped.find(")", marker_index)
    if paren_end != -1 and paren_end <= marker_index + 240:
        return stripped[paren_end + 1 :]
    return text


def _find_matching_json_object_end(text: str, start_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def merge_therapeutic_tool_results(
    state: AgentState,
    *,
    run_context: OpenAITextRunContext,
    response_text: str,
) -> TherapeuticToolMergeResult:
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
        if call.clear_session_buffer:
            delta["session_memory"] = {
                "held_semantic_candidates": [],
                "held_procedural_candidates": [],
            }
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
        return TherapeuticToolMergeResult(
            runtime_mode="grounded_lookup",
            response_style="grounded_lookup",
            response_text=grounded_calls[-1].response_text,
        )
    if memory_calls:
        apply_state_delta(state, {"route": "memory_control"})
        return TherapeuticToolMergeResult(
            runtime_mode="memory_control",
            response_style="memory_control",
            response_text=memory_calls[-1].response_text,
        )

    apply_state_delta(state, {"route": "therapeutic"})
    return TherapeuticToolMergeResult(
        runtime_mode="safe_therapeutic",
        response_style=response_style_from_state(state),
        response_text=response_text,
    )

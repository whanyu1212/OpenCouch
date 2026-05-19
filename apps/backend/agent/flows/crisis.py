"""Crisis execution path helpers for the OpenAI text runtime."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from agent.audit.crisis_log import write_crisis_log
from agent.observability.timing import elapsed_ms
from agent.runtime.prompt_utils import chunk_from_sdk_event, final_output_text
from agent.specialists.crisis import CRISIS_AGENT_NAME, build_runtime_crisis_agent
from agent.guardrails.prompts import build_crisis_response_system_prompt
from agent.specialists.therapeutic_prompts import build_clarifying_system_prompt
from agent.tools.crisis import (
    build_crisis_resource_lookup_delta,
    crisis_response_delta,
)
from agent.runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_crisis_turn(
    runtime: Any,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    runtime_mode: str,
    streamed: bool,
    session: Any | None = None,
) -> AgentState:
    if runtime_mode == "crisis_clarification":
        state = await runtime._load_turn_memory(state, context)
    elif runtime_mode != "crisis_response":
        raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

    if context.response_llm is not None:
        return await run_crisis_response_llm_turn(
            runtime,
            state,
            config=config,
            context=context,
            runtime_mode=runtime_mode,
            streamed=streamed,
            session=session,
        )

    run_context = runtime._run_context_for_state(state, config, context)
    agent = build_runtime_crisis_agent(
        state=state,
        runtime_mode=runtime_mode,
        base_agent=runtime._roster.crisis_agent,
    )
    tool_call_count = len(run_context.crisis_resource_tool_calls)
    input_text = runtime._crisis_input_text_for_state(
        state,
        runtime_mode=runtime_mode,
        include_recent_history=session is None,
        require_resource_tool=runtime_mode == "crisis_response",
    )
    response_text, sdk_duration_ms = await runtime._run_openai_agent_with(
        state,
        agent=agent,
        input_text=input_text,
        run_context=run_context,
        session=session,
    )

    response_style = _response_style_for_crisis_mode(runtime_mode)
    if runtime_mode == "crisis_response":
        if not _crisis_resource_tool_called(
            run_context,
            tool_call_count=tool_call_count,
        ):
            lookup_delta = await build_crisis_resource_lookup_delta(state, context)
            _apply_lookup_delta(state, lookup_delta)
            _apply_crisis_resource_fallback_diagnostics(state, run_context)
            response_text, sdk_duration_ms = await runtime._run_openai_agent_with(
                state,
                agent=build_runtime_crisis_agent(
                    state=state,
                    runtime_mode=runtime_mode,
                    base_agent=runtime._roster.crisis_agent,
                    enable_resource_tools=False,
                ),
                input_text=runtime._crisis_input_text_for_state(
                    state,
                    runtime_mode=runtime_mode,
                    include_recent_history=session is None,
                    require_resource_tool=False,
                ),
                run_context=run_context,
                session=session,
            )
        else:
            _apply_crisis_resource_tool_result(state, run_context)
        _apply_lookup_delta(state, crisis_response_delta(response_text))
        await write_crisis_log(state, context)
    else:
        state["route"] = "therapeutic"
        state["response_style"] = response_style
        state["response_text"] = response_text

    return await runtime._finalize_openai_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode=runtime_mode,
        response_style=response_style,
        selected_agent=CRISIS_AGENT_NAME,
        sdk_duration_ms=sdk_duration_ms,
        streamed=streamed,
    )


async def run_crisis_turn_stream(
    runtime: Any,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    runtime_mode: str,
    session: Any | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    if runtime_mode == "crisis_clarification":
        state = await runtime._load_turn_memory(state, context)
    elif runtime_mode != "crisis_response":
        raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

    if context.response_llm is not None:
        yield TextRuntimeStatusEvent(stage=runtime_mode)
        final_state = await run_crisis_response_llm_turn(
            runtime,
            state,
            config=config,
            context=context,
            runtime_mode=runtime_mode,
            streamed=True,
            session=session,
        )
        response_text = str(final_state.get("response_text") or "")
        if response_text:
            yield TextRuntimeChunkEvent(text=response_text)
        if runtime_mode == "crisis_response":
            yield TextRuntimeStatusEvent(stage="crisis_log")
        yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
        yield TextRuntimeStateEvent(state=final_state)
        return

    run_context = runtime._run_context_for_state(state, config, context)
    agent = build_runtime_crisis_agent(
        state=state,
        runtime_mode=runtime_mode,
        base_agent=runtime._roster.crisis_agent,
    )
    tool_call_count = len(run_context.crisis_resource_tool_calls)
    input_text = runtime._crisis_input_text_for_state(
        state,
        runtime_mode=runtime_mode,
        include_recent_history=session is None,
        require_resource_tool=runtime_mode == "crisis_response",
    )

    yield TextRuntimeStatusEvent(stage=runtime_mode)
    run_start = time.monotonic()

    stream = runtime._runner.run_streamed(
        agent=agent,
        input_text=input_text,
        context=run_context,
        session=session,
    )
    chunks: list[str] = []
    async for sdk_event in stream.stream_events():
        chunk = chunk_from_sdk_event(sdk_event)
        if chunk:
            chunks.append(chunk)

    response_text = final_output_text(
        getattr(stream, "final_output", None),
        fallback="".join(chunks),
    )
    sdk_duration_ms = elapsed_ms(run_start)
    response_style = _response_style_for_crisis_mode(runtime_mode)

    if runtime_mode == "crisis_response":
        if not _crisis_resource_tool_called(
            run_context,
            tool_call_count=tool_call_count,
        ):
            lookup_delta = await build_crisis_resource_lookup_delta(state, context)
            _apply_lookup_delta(state, lookup_delta)
            _apply_crisis_resource_fallback_diagnostics(state, run_context)
            response_text, sdk_duration_ms = await runtime._run_openai_agent_with(
                state,
                agent=build_runtime_crisis_agent(
                    state=state,
                    runtime_mode=runtime_mode,
                    base_agent=runtime._roster.crisis_agent,
                    enable_resource_tools=False,
                ),
                input_text=runtime._crisis_input_text_for_state(
                    state,
                    runtime_mode=runtime_mode,
                    include_recent_history=session is None,
                    require_resource_tool=False,
                ),
                run_context=run_context,
                session=session,
            )
            chunks = [response_text]
        else:
            _apply_crisis_resource_tool_result(state, run_context)
        for chunk in chunks:
            yield TextRuntimeChunkEvent(text=chunk)
        if response_text and not chunks:
            yield TextRuntimeChunkEvent(text=response_text)
        _apply_lookup_delta(state, crisis_response_delta(response_text))
        yield TextRuntimeStatusEvent(stage="crisis_log")
        await write_crisis_log(state, context)
    else:
        for chunk in chunks:
            yield TextRuntimeChunkEvent(text=chunk)
        if response_text and not chunks:
            yield TextRuntimeChunkEvent(text=response_text)
        state["route"] = "therapeutic"
        state["response_style"] = response_style
        state["response_text"] = response_text

    final_state = await runtime._finalize_openai_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode=runtime_mode,
        response_style=response_style,
        selected_agent=CRISIS_AGENT_NAME,
        sdk_duration_ms=sdk_duration_ms,
        streamed=True,
    )
    yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    yield TextRuntimeStateEvent(state=final_state)


async def run_crisis_response_llm_turn(
    runtime: Any,
    state: AgentState,
    *,
    config: Any,
    context: WorkflowContext,
    runtime_mode: str,
    streamed: bool,
    session: Any | None = None,
) -> AgentState:
    llm_client = context.response_llm
    if llm_client is None:
        raise RuntimeError("crisis response override requires response_llm.")

    if runtime_mode == "crisis_response":
        lookup_delta = await build_crisis_resource_lookup_delta(state, context)
        _apply_lookup_delta(state, lookup_delta)
        prompt = runtime._crisis_input_text_for_state(
            state,
            runtime_mode=runtime_mode,
            include_recent_history=session is None,
            require_resource_tool=False,
        )
        system_instruction = build_crisis_response_system_prompt()
    elif runtime_mode == "crisis_clarification":
        prompt = runtime._crisis_input_text_for_state(
            state,
            runtime_mode=runtime_mode,
            include_recent_history=session is None,
        )
        system_instruction = build_clarifying_system_prompt(state)
    else:
        raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")

    run_start = time.monotonic()
    response_text = await llm_client.generate_text(
        prompt=prompt,
        system_instruction=system_instruction,
    )
    sdk_duration_ms = elapsed_ms(run_start)
    response_style = _response_style_for_crisis_mode(runtime_mode)
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_response_llm_override": True,
    }
    if runtime_mode == "crisis_response":
        diagnostics["openai_crisis_tool_fallback"] = True
        _apply_lookup_delta(state, crisis_response_delta(response_text))
        await write_crisis_log(state, context)
    else:
        state["route"] = "therapeutic"
        state["response_style"] = response_style
        state["response_text"] = response_text
    state["diagnostics"] = diagnostics
    return await runtime._finalize_openai_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode=runtime_mode,
        response_style=response_style,
        selected_agent=CRISIS_AGENT_NAME,
        sdk_duration_ms=sdk_duration_ms,
        streamed=streamed,
    )


def _apply_lookup_delta(state: AgentState, delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        if key == "diagnostics":
            state["diagnostics"] = {
                **dict(state.get("diagnostics", {}) or {}),
                **dict(value or {}),
            }
        else:
            state[key] = value


def _crisis_resource_tool_called(run_context: Any, *, tool_call_count: int) -> bool:
    return len(run_context.crisis_resource_tool_calls) > tool_call_count


def _apply_crisis_resource_tool_result(state: AgentState, run_context: Any) -> None:
    result = run_context.latest_crisis_resource_tool_result()
    if result is None:
        return
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_crisis_tool_expected": "lookup_crisis_resources",
        "openai_crisis_tool_calls": [
            call.tool_name for call in run_context.crisis_resource_tool_calls
        ],
        "openai_crisis_tool_fallback": False,
    }
    state["inferred_location"] = result.inferred_location
    state["found_resources"] = result.found_resources
    state["resource_lookup_status"] = result.resource_lookup_status
    state["diagnostics"] = diagnostics


def _apply_crisis_resource_fallback_diagnostics(
    state: AgentState,
    run_context: Any,
) -> None:
    state["diagnostics"] = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_crisis_tool_expected": "lookup_crisis_resources",
        "openai_crisis_tool_calls": [
            call.tool_name for call in run_context.crisis_resource_tool_calls
        ],
        "openai_crisis_tool_fallback": True,
    }


def _response_style_for_crisis_mode(runtime_mode: str) -> str:
    if runtime_mode == "crisis_response":
        return "crisis_response"
    if runtime_mode == "crisis_clarification":
        return "clarifying"
    raise ValueError(f"Unsupported OpenAI crisis runtime mode: {runtime_mode}")


__all__ = [
    "run_crisis_response_llm_turn",
    "run_crisis_turn",
    "run_crisis_turn_stream",
]

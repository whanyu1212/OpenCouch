"""Grounded lookup execution path for the OpenAI text runtime."""

from __future__ import annotations

import json
from typing import Any

from agent.flows.therapeutic import (
    can_fallback_to_control_response,
    openai_sdk_fallback_reason,
)
from agent.runtime.services import TextRuntimeServices
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded import build_grounded_lookup_delta


async def run_grounded_lookup_turn(
    services: TextRuntimeServices,
    state: AgentState,
    *,
    query: str,
    config: Any,
    context: WorkflowContext,
    streamed: bool,
    session: Any | None = None,
) -> AgentState:
    """Run a grounded lookup turn through the SDK tool path."""

    from agent.runtime.state_ops import apply_state_delta

    run_context = services.build_run_context(state, config, context)
    agent = services.build_agent(state)
    sdk_duration_ms: float | None
    fallback_reason: str | None = None
    try:
        _, sdk_duration_ms = await services.run_openai_agent_with(
            state,
            agent=agent,
            input_text=grounded_lookup_input_text_for_state(state, query),
            run_context=run_context,
            session=session,
        )
    except Exception as exc:
        if not can_fallback_to_control_response(exc, context):
            raise
        sdk_duration_ms = None
        fallback_reason = openai_sdk_fallback_reason(exc)
    tool_result = run_context.latest_grounded_tool_result()
    diagnostics: dict[str, Any] = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_grounded_tool_expected": "answer_grounded_lookup",
        "openai_grounded_tool_calls": [
            call.tool_name for call in run_context.grounded_tool_calls
        ],
    }
    if fallback_reason is not None:
        diagnostics["openai_sdk_fallback_reason"] = fallback_reason

    if tool_result is None:
        fallback_delta = await build_grounded_lookup_delta(state, context)
        response_text = str(fallback_delta.get("response_text") or "")
        if not response_text:
            raise ValueError("grounded_lookup returned an empty response.")
        diagnostics["openai_grounded_tool_fallback"] = True
        apply_state_delta(
            state,
            {
                **dict(fallback_delta),
                "route": "grounded_lookup",
                "diagnostics": diagnostics,
            },
        )
    else:
        response_text = tool_result.response_text
        diagnostics["openai_grounded_tool_fallback"] = False
        apply_state_delta(
            state,
            {
                "grounded_lookup": tool_result.grounded_lookup,
                "diagnostics": diagnostics,
            },
        )

    return await services.finalize_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode="grounded_lookup",
        response_style="grounded_lookup",
        selected_agent=agent.name,
        sdk_duration_ms=sdk_duration_ms,
        streamed=streamed,
    )


def grounded_lookup_input_text_for_state(
    state: AgentState,
    query: str,
) -> str:
    """Build the strict tool-forcing grounded lookup prompt."""

    return (
        "The current user turn is an explicit grounded lookup request "
        "selected by the OpenCouch runtime.\n\n"
        "Required tool: answer_grounded_lookup\n"
        f"Required tool arguments: {json.dumps({'query': query}, sort_keys=True)}\n"
        "Call the required tool exactly once before answering. Then answer "
        "using only the tool result's response_text. Do not provide "
        "ungrounded factual claims.\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


__all__ = ["grounded_lookup_input_text_for_state", "run_grounded_lookup_turn"]

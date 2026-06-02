"""Triage turn-dispatch policy for the OpenAI text runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent.flows.sdk_fallback import (
    can_fallback_to_control_response,
    openai_sdk_fallback_reason,
)
from agent.memory.types import TurnDispatchDecision
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.session.state import format_recent_history
from agent.runtime.state_ops import apply_state_delta
from agent.runtime.turn_dispatch import state_delta_for_turn_dispatch
from agent.runtime.types import TextRuntimeConfig
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient


async def apply_triage_turn_dispatch(
    state: AgentState,
    *,
    config: TextRuntimeConfig,
    context: WorkflowContext,
    run_context_factory: Callable[[], OpenAITextRunContext],
    runner: Any,
    triage_agent: Any,
) -> AgentState:
    """Run triage dispatch and apply the resulting route state."""

    del config
    llm_client = context.llm_client
    if llm_client is None:
        return state

    triage_input = triage_input_text_for_state(state)
    fallback_reason: str | None = None
    try:
        result = await runner.run_triage(
            agent=triage_agent,
            input_text=triage_input,
            context=run_context_factory(),
        )
        decision = turn_dispatch_decision_from_output(
            getattr(result, "final_output", None)
        )
    except Exception as exc:
        if not can_fallback_to_control_response(exc, context):
            raise
        fallback_reason = openai_sdk_fallback_reason(exc)
        decision = await _generate_fallback_decision(
            llm_client,
            triage_input=triage_input,
            triage_agent=triage_agent,
        )

    apply_triage_decision_to_state(
        state,
        decision,
        fallback_reason=fallback_reason,
    )
    return state


def apply_triage_decision_to_state(
    state: AgentState,
    decision: TurnDispatchDecision,
    *,
    fallback_reason: str | None = None,
) -> AgentState:
    """Apply triage policy decisions without mutating the source decision object."""

    clarification_kind = decision.clarification_kind
    needs_blocking_clarification = (
        decision.clarification_needed and clarification_kind == "blocking"
    )
    legacy_low_confidence_clarification = (
        decision.confidence == "low" and not decision.clarification_needed
    )
    should_route_to_clarification = (
        needs_blocking_clarification or legacy_low_confidence_clarification
    )
    tentative_route = decision.route if should_route_to_clarification else None
    effective_decision = (
        decision.model_copy(update={"route": "therapeutic"})
        if should_route_to_clarification
        else decision
    )
    apply_state_delta(
        state,
        state_delta_for_turn_dispatch(state, effective_decision),
    )
    if fallback_reason is not None:
        apply_state_delta(
            state,
            {
                "diagnostics": {
                    "openai_triage_sdk_fallback_reason": fallback_reason,
                }
            },
        )
    if should_route_to_clarification:
        apply_state_delta(
            state,
            {
                "response_style": "clarifying",
                "turn_lifecycle": {
                    "active_flow": active_flow_for_state(state),
                    "action": decision.active_flow_action,
                    "tentative_route": tentative_route,
                    "triage_confidence": decision.confidence,
                    "clarification_needed": True,
                    "clarification_kind": clarification_kind,
                    "secondary_route": decision.secondary_route,
                    "intent_summary": decision.intent_summary,
                    "clarification_question": decision.clarification_question,
                    "no_clarification_reason": decision.no_clarification_reason,
                },
                "diagnostics": {
                    "openai_triage_tentative_route": tentative_route,
                },
            },
        )
    return state


def active_flow_for_state(state: AgentState) -> str:
    active_flow = "none"
    exercise_state = state.get("exercise_state", {}) or {}
    if (
        isinstance(exercise_state, Mapping)
        and exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    ):
        active_flow = "guided_exercise"
    memory_control = state.get("memory_control", {}) or {}
    if (
        isinstance(memory_control, Mapping)
        and memory_control.get("pending_action") is not None
    ):
        active_flow = "pending_memory_action"
    return active_flow


def triage_input_text_for_state(state: AgentState) -> str:
    active_flow = active_flow_for_state(state)
    memory_reference = state.get("memory_reference", {}) or {}
    memory_reference_mode = (
        memory_reference.get("mode")
        if isinstance(memory_reference, Mapping)
        else "none"
    )
    turn_lifecycle = state.get("turn_lifecycle", {}) or {}
    prior_clarification = ""
    if (
        isinstance(turn_lifecycle, Mapping)
        and turn_lifecycle.get("triage_confidence") == "low"
    ):
        tentative_route = str(turn_lifecycle.get("tentative_route") or "").strip()
        if tentative_route:
            prior_clarification = (
                "Prior low-confidence clarification: the previous turn asked the "
                f"user to clarify ambiguous intent; tentative route was "
                f'"{tentative_route}". Treat the current message as a possible '
                "answer to that clarification, but do not force the tentative "
                "route if the user changed topics.\n"
            )
    return (
        f"Active flow: {active_flow}\n"
        f"Memory reference mode: {memory_reference_mode}\n"
        f"{prior_clarification}"
        f"Recent conversation:\n{format_recent_history(state)}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def turn_dispatch_decision_from_output(output: Any) -> TurnDispatchDecision:
    if isinstance(output, TurnDispatchDecision):
        return output
    if isinstance(output, Mapping):
        return TurnDispatchDecision.model_validate(dict(output))
    raise TypeError(
        "OpenAI triage agent returned unsupported output "
        f"{type(output).__name__}; expected TurnDispatchDecision."
    )


async def _generate_fallback_decision(
    llm_client: BaseLLMClient,
    *,
    triage_input: str,
    triage_agent: Any,
) -> TurnDispatchDecision:
    return await llm_client.generate_structured(
        prompt=triage_input,
        response_schema=TurnDispatchDecision,
        system_instruction=triage_agent.instructions,
    )


__all__ = [
    "active_flow_for_state",
    "apply_triage_decision_to_state",
    "apply_triage_turn_dispatch",
    "triage_input_text_for_state",
    "turn_dispatch_decision_from_output",
]

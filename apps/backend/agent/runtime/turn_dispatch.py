"""Turn-dispatch state mapping for the OpenAI text runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.memory.types import TurnDispatchDecision
from agent.specialists.triage import TRIAGE_AGENT_NAME
from agent.state import AgentState


def state_delta_for_turn_dispatch(
    state: AgentState,
    decision: TurnDispatchDecision,
) -> dict[str, Any]:
    """Map a structured triage decision into app-owned runtime state."""

    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_triage_agent": TRIAGE_AGENT_NAME,
        "openai_triage_route": decision.route,
        "openai_triage_active_flow_action": decision.active_flow_action,
        "openai_triage_confidence": decision.confidence,
        "openai_triage_clarification_needed": decision.clarification_needed,
        "openai_triage_clarification_kind": decision.clarification_kind,
        "openai_triage_secondary_route": decision.secondary_route,
        "openai_triage_no_clarification_reason": decision.no_clarification_reason,
    }
    existing_memory_reference = state.get("memory_reference", {}) or {}
    existing_memory_reference_mode = (
        existing_memory_reference.get("mode")
        if isinstance(existing_memory_reference, Mapping)
        else None
    )
    memory_reference_mode = (
        "explicit"
        if existing_memory_reference_mode == "explicit"
        and decision.memory_reference_mode == "none"
        else decision.memory_reference_mode
    )
    turn_lifecycle: dict[str, Any] = {
        "active_flow": (
            "guided_exercise" if decision.route == "guided_exercise" else "none"
        ),
        "action": decision.active_flow_action,
    }
    if decision.clarification_needed or decision.no_clarification_reason != "none":
        turn_lifecycle.update(
            {
                "triage_confidence": decision.confidence,
                "clarification_needed": decision.clarification_needed,
                "clarification_kind": decision.clarification_kind,
                "secondary_route": decision.secondary_route,
                "intent_summary": decision.intent_summary,
                "clarification_question": decision.clarification_question,
                "no_clarification_reason": decision.no_clarification_reason,
            }
        )
    delta: dict[str, Any] = {
        "route": decision.route,
        "memory_reference": {"mode": memory_reference_mode},
        "turn_lifecycle": turn_lifecycle,
        "diagnostics": diagnostics,
    }
    if decision.route == "grounded_lookup":
        delta["grounded_lookup"] = {
            **dict(state.get("grounded_lookup", {}) or {}),
            "query": decision.query or str(state.get("message") or "").strip(),
        }
        delta["response_style"] = "grounded_lookup"
    if decision.route == "guided_exercise":
        delta["response_style"] = "guided_exercise"
        delta["therapeutic_approach"] = state.get("therapeutic_approach") or "none"
        diagnostics["openai_guided_exercise_selection_basis"] = (
            decision.exercise_start_basis
        )
        if decision.exercise_type:
            exercise_state = dict(state.get("exercise_state", {}) or {})
            exercise_state.setdefault("exercise_type", decision.exercise_type)
            delta["exercise_state"] = exercise_state
    return delta


__all__ = ["state_delta_for_turn_dispatch"]

"""LLM-only crisis gate node for the OpenCouch graph."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.audit.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.graph_constants import (
    CRISIS_RESOURCE_LOOKUP_NODE,
    TURN_DISPATCH_NODE,
    CrisisGateNextNode,
)
from agent.models import CrisisAssessment
from agent.observability.routing_trace import append_routing_trace
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.gates.safety.service import CrisisRiskService
from agent.state import AgentState, cleared_exercise_state


def _build_crisis_delta(
    assessment: CrisisAssessment,
    *,
    override_kind: CrisisOverrideOutcome,
    classifier_path: CrisisClassifierPath,
    llm_failure_occurred: bool,
    duration_ms: float,
    prior_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the state delta for one crisis-gate decision.

    Args:
        assessment (CrisisAssessment): Final assessment chosen for this turn.
        override_kind (CrisisOverrideOutcome): Override outcome recorded for audit metadata.
        classifier_path (CrisisClassifierPath): Classifier path that produced the assessment.
        llm_failure_occurred (bool): Whether the LLM path was attempted and failed.
        duration_ms (float): Total wall-clock time spent in the crisis gate.
        prior_diagnostics (dict[str, Any] | None): Existing diagnostics used
            only to preserve earlier routing-trace entries.

    Returns:
        dict[str, Any]: Partial state update containing crisis, crisis-audit,
            turn-scoped routing/output channels, and diagnostics data.
    """

    route = "crisis" if assessment.needs_crisis_response else "therapeutic"
    if assessment.needs_crisis_response:
        decision = "crisis"
    elif assessment.needs_clarification:
        decision = "check"
    elif assessment.level >= 1:
        decision = "distress"
    else:
        decision = "normal"

    diagnostics = {
        "crisis_gate_ms": round(duration_ms, 2),
        "crisis_classifier_path": classifier_path,
        "crisis_level": assessment.level,
        **append_routing_trace(
            prior_diagnostics,
            {
                "stage": "safety",
                "decision": decision,
                "source": classifier_path,
                "reason": assessment.reason or "No crisis signal detected.",
                "confidence": assessment.confidence,
            },
        ),
    }

    delta: dict[str, Any] = {
        "crisis": assessment,
        "route": route,
        "crisis_audit": {
            "crisis_override_kind": override_kind,
            "crisis_classifier_path": classifier_path,
            "crisis_llm_failure_occurred": llm_failure_occurred,
        },
        "diagnostics": diagnostics,
    }
    if assessment.needs_crisis_response:
        delta["exercise_state"] = cleared_exercise_state()
    return delta


async def run_crisis_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[CrisisGateNextNode]:
    """Run the crisis gate for the current turn.

    Args:
        state: Current graph state for the turn being processed.
        runtime: LangGraph runtime carrying the workflow context.

    Returns:
        State update plus the next node to run.
    """

    gate_start = time.monotonic()

    result = await CrisisRiskService().assess_turn(
        state,
        llm_client=runtime.context.llm_client,
    )

    gate_duration_ms = elapsed_ms(gate_start)
    delta = _build_crisis_delta(
        result.assessment,
        override_kind=result.override_kind,
        classifier_path=result.classifier_path,
        llm_failure_occurred=result.llm_failure_occurred,
        duration_ms=gate_duration_ms,
        prior_diagnostics=state.get("diagnostics"),
    )
    assessment = result.assessment
    next_node: CrisisGateNextNode = (
        CRISIS_RESOURCE_LOOKUP_NODE
        if assessment.needs_crisis_response
        else TURN_DISPATCH_NODE
    )
    return Command(update=delta, goto=next_node)

"""Application-owned crisis gate helpers shared by text runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent.audit.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.guardrails.service import CrisisRiskService
from agent.models import CrisisAssessment
from agent.observability.decorators import trace_event, trace_span
from agent.observability.diagnostics import merge_diagnostics
from agent.observability.events import SAFETY_ASSESS, SAFETY_ASSESS_COMPLETED
from agent.observability.routing_trace import append_routing_trace
from agent.observability.timing import elapsed_ms
from agent.state import AgentState
from llm.base import BaseLLMClient


@dataclass(frozen=True)
class CrisisGateResult:
    """Application-owned crisis-gate result for one turn."""

    assessment: CrisisAssessment
    delta: dict[str, Any]


def build_crisis_gate_delta(
    assessment: CrisisAssessment,
    *,
    override_kind: CrisisOverrideOutcome,
    classifier_path: CrisisClassifierPath,
    llm_failure_occurred: bool,
    duration_ms: float,
    prior_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the state delta for one crisis-gate decision."""

    route = "crisis" if assessment.needs_crisis_response else "therapeutic"
    if assessment.needs_crisis_response:
        decision = "crisis"
    elif assessment.needs_clarification:
        decision = "check"
    elif assessment.level >= 1:
        decision = "distress"
    else:
        decision = "normal"

    diagnostics = merge_diagnostics(
        {
            "crisis_gate_ms": round(duration_ms, 2),
            "crisis_classifier_path": classifier_path,
            "crisis_level": assessment.level,
        },
        append_routing_trace(
            prior_diagnostics,
            {
                "stage": "safety",
                "decision": decision,
                "source": classifier_path,
                "reason": assessment.reason or "No crisis signal detected.",
                "confidence": assessment.confidence,
            },
        ),
    )

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
        delta.update(
            {
                "turn_lifecycle": {"active_flow": "none", "action": "none"},
            }
        )
    return delta


@trace_span(SAFETY_ASSESS, attrs={"runtime_mode": "text"})
async def assess_crisis_gate(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    service: CrisisRiskService | None = None,
) -> CrisisGateResult:
    """Run the application-owned crisis gate for the current turn."""

    gate_start = time.monotonic()
    result = await (service or CrisisRiskService()).assess_turn(
        state,
        llm_client=llm_client,
    )

    trace_event(
        SAFETY_ASSESS_COMPLETED,
        {
            "level": result.assessment.level,
            "needs_crisis_response": result.assessment.needs_crisis_response,
            "needs_clarification": result.assessment.needs_clarification,
            "classifier_path": result.classifier_path,
            "override_kind": result.override_kind,
            "llm_failure_occurred": result.llm_failure_occurred,
        },
    )
    delta = build_crisis_gate_delta(
        result.assessment,
        override_kind=result.override_kind,
        classifier_path=result.classifier_path,
        llm_failure_occurred=result.llm_failure_occurred,
        duration_ms=elapsed_ms(gate_start),
        prior_diagnostics=state.get("diagnostics"),
    )
    return CrisisGateResult(assessment=result.assessment, delta=delta)

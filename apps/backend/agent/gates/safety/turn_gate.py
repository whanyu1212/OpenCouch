"""Application-owned crisis gate helpers shared by text runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent.active_flow import clear_all_active_flows_delta
from agent.audit.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.gates.safety.service import CrisisRiskService
from agent.models import CrisisAssessment
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
        delta.update(clear_all_active_flows_delta())
    return delta


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

    delta = build_crisis_gate_delta(
        result.assessment,
        override_kind=result.override_kind,
        classifier_path=result.classifier_path,
        llm_failure_occurred=result.llm_failure_occurred,
        duration_ms=elapsed_ms(gate_start),
        prior_diagnostics=state.get("diagnostics"),
    )
    return CrisisGateResult(assessment=result.assessment, delta=delta)

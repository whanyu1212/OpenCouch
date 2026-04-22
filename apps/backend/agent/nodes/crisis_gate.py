"""Hybrid crisis gate node for the OpenCouch graph."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.memory.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.models import CrisisAssessment, ModeType, ResponseCategory
from agent.prompts import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
)
from agent.runtime_context import WorkflowContext
from agent.safety.crisis_rules import (
    assess_crisis_risk_deterministically,
    detect_crisis_override,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class CrisisAssessmentSchema(BaseModel):
    """Structured schema for crisis-classification model output."""

    level: int = Field(
        ge=0,
        le=3,
        description=(
            "Risk level: 0=no risk, 1=ambiguous/distress, "
            "2=clear self-harm or suicidal ideation, "
            "3=imminent risk with plan, means, or timing."
        ),
    )
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(description="Short explanation of the classification.")
    needs_crisis_response: bool = Field(description="Expected true for levels 2 and 3.")
    needs_clarification: bool = Field(description="Expected true for level 1.")


async def assess_crisis_risk_with_llm(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> CrisisAssessment:
    """Assess crisis risk with the structured LLM classifier.

    Args:
        state (AgentState): Current graph state for the turn being classified.
        llm_client (BaseLLMClient): Configured LLM client used for structured output.

    Returns:
        CrisisAssessment: Classifier output converted into the shared crisis assessment model.
    """

    raw = await llm_client.generate_structured(
        prompt=build_crisis_classifier_prompt(state),
        response_schema=CrisisAssessmentSchema,
        system_instruction=build_crisis_classifier_system_prompt(),
    )

    return CrisisAssessment(**raw.model_dump())


def enforce_crisis_truth_table(assessment: CrisisAssessment) -> CrisisAssessment:
    """Enforce the crisis level-to-flag truth table on an assessment.

    Args:
        assessment (CrisisAssessment): Assessment whose boolean flags should match its level.

    Returns:
        CrisisAssessment: Copy of the assessment with truth-table-consistent flags.
    """

    return assessment.model_copy(
        update={
            "needs_crisis_response": assessment.level >= 2,
            "needs_clarification": assessment.level == 1,
        }
    )


def _build_crisis_delta(
    state: AgentState,
    assessment: CrisisAssessment,
    *,
    override_kind: CrisisOverrideOutcome,
    classifier_path: CrisisClassifierPath,
    llm_failure_occurred: bool,
    duration_ms: float,
) -> dict[str, Any]:
    """Build the state delta for one crisis-gate decision.

    Args:
        state (AgentState): Current graph state before the crisis update is applied.
        assessment (CrisisAssessment): Final assessment chosen for this turn.
        override_kind (CrisisOverrideOutcome): Override outcome recorded for audit metadata.
        classifier_path (CrisisClassifierPath): Classifier path that produced the assessment.
        llm_failure_occurred (bool): Whether the LLM path was attempted and failed.
        duration_ms (float): Total wall-clock time spent in the crisis gate.

    Returns:
        dict[str, Any]: Partial state update containing crisis, routing, response, and diagnostics data.
    """

    route = "crisis" if assessment.needs_crisis_response else "therapeutic"
    routing = state.get("routing", {})
    response = state.get("response", {})

    return {
        "crisis": assessment,
        "routing": {
            **routing,
            "route": route,
            "response_style": (
                "safety_check" if route == "crisis" else routing.get("response_style")
            ),
            "response_style_source": "crisis_gate",
            "response_style_type": (
                ModeType.CRISIS
                if route == "crisis"
                else routing.get("response_style_type")
            ),
            "crisis_override_kind": override_kind,
            "crisis_classifier_path": classifier_path,
            "crisis_llm_failure_occurred": llm_failure_occurred,
        },
        "response": {
            **response,
            "kind": (
                ResponseCategory.CRISIS if route == "crisis" else response.get("kind")
            ),
        },
        "diagnostics": {
            "crisis_gate_ms": round(duration_ms, 2),
            "crisis_classifier_path": classifier_path,
            "crisis_level": assessment.level,
        },
    }


async def run_crisis_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[Literal["crisis_response_node", "load_memory_node"]]:
    """Run the crisis gate for the current turn.

    Args:
        state (AgentState): Current graph state for the turn being processed.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying the workflow context.

    Returns:
        Command[Literal["crisis_response_node", "load_memory_node"]]: State update plus the next node to run.
    """

    llm_client = runtime.context.llm_client
    gate_start = time.monotonic()

    override_kind: CrisisOverrideOutcome = "none"
    classifier_path: CrisisClassifierPath
    llm_failure_occurred = False

    override = detect_crisis_override(state)
    if override is not None:
        override_kind_detected, override_assessment = override
        override_kind = override_kind_detected
        classifier_path = "override"
        assessment = enforce_crisis_truth_table(override_assessment)

    else:
        if llm_client is None:
            classifier_path = "deterministic"
            deterministic = assess_crisis_risk_deterministically(state)
            assessment = enforce_crisis_truth_table(deterministic)
        else:
            try:
                llm_assessment = await assess_crisis_risk_with_llm(
                    state, llm_client=llm_client
                )
                classifier_path = "llm_primary"
                assessment = enforce_crisis_truth_table(llm_assessment)
            except Exception:
                logger.warning(
                    "Crisis LLM classifier failed; using deterministic fallback.",
                    exc_info=True,
                )
                classifier_path = "deterministic"
                llm_failure_occurred = True
                deterministic = assess_crisis_risk_deterministically(state)
                assessment = enforce_crisis_truth_table(deterministic)

    gate_duration_ms = (time.monotonic() - gate_start) * 1000
    delta = _build_crisis_delta(
        state,
        assessment,
        override_kind=override_kind,
        classifier_path=classifier_path,
        llm_failure_occurred=llm_failure_occurred,
        duration_ms=gate_duration_ms,
    )
    next_node = (
        "crisis_response_node"
        if assessment.needs_crisis_response
        else "load_memory_node"
    )
    return Command(update=delta, goto=next_node)

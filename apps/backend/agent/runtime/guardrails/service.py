"""Service for turn-level crisis assessment and safety classification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.audit.models import CrisisClassifierPath, CrisisOverrideOutcome
from agent.models import CrisisAssessment
from agent.runtime.guardrails.prompts import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
)
from agent.state import AgentState
from llm.base import BaseLLMClient


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


class CrisisRiskResult(BaseModel):
    """Structured crisis assessment result returned by the safety service."""

    assessment: CrisisAssessment
    override_kind: CrisisOverrideOutcome = "none"
    classifier_path: CrisisClassifierPath
    llm_failure_occurred: bool = False


async def assess_crisis_risk_with_llm(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> CrisisAssessment:
    """Assess crisis risk with the structured LLM classifier."""

    raw = await llm_client.generate_structured(
        prompt=build_crisis_classifier_prompt(state),
        response_schema=CrisisAssessmentSchema,
        system_instruction=build_crisis_classifier_system_prompt(),
    )
    return CrisisAssessment(**raw.model_dump())


def enforce_crisis_truth_table(assessment: CrisisAssessment) -> CrisisAssessment:
    """Enforce the crisis level-to-flag truth table on an assessment."""

    return assessment.model_copy(
        update={
            "needs_crisis_response": assessment.level >= 2,
            "needs_clarification": assessment.level == 1,
        }
    )


class CrisisRiskService:
    """Orchestrate LLM-backed crisis assessment for one turn."""

    async def assess_turn(
        self,
        state: AgentState,
        *,
        llm_client: BaseLLMClient | None,
    ) -> CrisisRiskResult:
        """Return the final crisis assessment for one turn."""

        if llm_client is None:
            raise RuntimeError(
                "CrisisRiskService requires an LLM client for crisis classification."
            )

        llm_assessment = await assess_crisis_risk_with_llm(
            state,
            llm_client=llm_client,
        )
        return CrisisRiskResult(
            assessment=enforce_crisis_truth_table(llm_assessment),
            classifier_path="llm_primary",
        )

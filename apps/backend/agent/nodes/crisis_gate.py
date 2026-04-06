"""Hybrid crisis gate for the MVP agent.

The design is intentionally split into three layers:
- deterministic overrides for obvious boundary cases
- optional LLM classification for the gray area
- policy normalization to produce one final assessment
"""

from __future__ import annotations

import re
from typing import Literal

from agent.models import CrisisAssessment
from agent.prompts import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
)
from agent.state import AgentState
from pydantic import BaseModel

from services.llm.base import BaseLLMClient

OverrideKind = Literal["imminent_risk", "idiomatic_safe"]

IMMINENT_PATTERNS = (
    r"\b(plan to|going to|gonna|will)\s+(kill myself|end my life|hurt myself)\b",
    r"\b(kill myself|end my life|hurt myself)\s+(tonight|today|right now|soon)\b",
    r"\b(i have|got)\s+(pills|a gun|a rope|a knife)\b",
    r"\b(tonight|today|right now)\b.*\b(kill myself|end my life|hurt myself)\b",
)

IDIOMATIC_SAFE_PATTERNS = (
    r"\bwork is killing me\b",
    r"\bthis homework is killing me\b",
    r"\bi could kill my\b",
)

CLEAR_SELF_HARM_PATTERNS = (
    r"\b(kill myself|suicide|end my life|want to die)\b",
    r"\b(hurt myself|self harm|self-harm|cutting again|been cutting)\b",
    r"\b(don't want to be here anymore|nobody would miss me if i was gone)\b",
    r"\b(thinking about ending it all|thinking about killing myself)\b",
)

AMBIGUOUS_PATTERNS = (
    r"\b(wish i could disappear|wish i wasn't here|wish i could go away)\b",
    r"\b(i can't do this anymore|i am done|i'm done|give up)\b",
    r"\b(dark thoughts|not sure i can keep going)\b",
    r"\b(i want to disappear|i don't want to exist)\b",
)

DISTRESS_PATTERNS = (r"\b(hopeless|empty|worthless|trapped|completely overwhelmed)\b",)


class CrisisAssessmentSchema(BaseModel):
    """Structured schema for crisis-classification model output."""

    level: int
    confidence: str
    reason: str
    needs_crisis_response: bool
    needs_clarification: bool


def _combined_user_text(state: AgentState) -> str:
    """Combine recent user turns into a single lowercase text blob."""

    recent_user_turns = [
        turn["content"]
        for turn in state["history"][-4:]
        if turn.get("role") == "user" and turn.get("content")
    ]
    recent_user_turns.append(state["message"])
    return " ".join(recent_user_turns).lower()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any pattern in the provided tuple."""

    return any(re.search(pattern, text) for pattern in patterns)


def detect_crisis_override(
    state: AgentState,
) -> tuple[OverrideKind, CrisisAssessment] | None:
    """Return a hard override for obvious crisis-boundary cases.

    Args:
        state: The current agent graph state.

    Returns:
        A tuple of override kind and assessment when a hard override applies, or
        `None` when the message should continue to a classifier or fallback path.
    """

    text = _combined_user_text(state)

    if _matches_any(text, IDIOMATIC_SAFE_PATTERNS):
        return (
            "idiomatic_safe",
            CrisisAssessment(
                level=0,
                confidence="high",
                reason="Matched common idiomatic language without self-harm intent.",
                needs_crisis_response=False,
                needs_clarification=False,
            ),
        )

    if _matches_any(text, IMMINENT_PATTERNS):
        return (
            "imminent_risk",
            CrisisAssessment(
                level=3,
                confidence="high",
                reason="Detected imminent self-harm language with plan, means, or timing.",
                needs_crisis_response=True,
                needs_clarification=False,
            ),
        )

    return None


def assess_crisis_risk_deterministically(state: AgentState) -> CrisisAssessment:
    """Assess crisis risk using deterministic fallback rules.

    Args:
        state: The current agent graph state.

    Returns:
        A crisis assessment derived only from deterministic rules.
    """

    text = _combined_user_text(state)

    override = detect_crisis_override(state)
    if override is not None:
        _, assessment = override
        return assessment

    if _matches_any(text, CLEAR_SELF_HARM_PATTERNS):
        return CrisisAssessment(
            level=2,
            confidence="high",
            reason="Detected clear self-harm or suicidal ideation language.",
            needs_crisis_response=True,
            needs_clarification=False,
        )

    if _matches_any(text, AMBIGUOUS_PATTERNS):
        return CrisisAssessment(
            level=1,
            confidence="medium",
            reason="Detected ambiguous but concerning self-harm-adjacent language.",
            needs_crisis_response=False,
            needs_clarification=True,
        )

    if _matches_any(text, DISTRESS_PATTERNS):
        return CrisisAssessment(
            level=1,
            confidence="medium",
            reason="Detected high-distress language without explicit self-harm signal.",
            needs_crisis_response=False,
            needs_clarification=True,
        )

    return CrisisAssessment(
        level=0,
        confidence="high",
        reason="No self-harm signal detected by deterministic fallback rules.",
        needs_crisis_response=False,
        needs_clarification=False,
    )


async def assess_crisis_risk_with_llm(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> CrisisAssessment:
    """Assess crisis risk with a configured model client.

    Args:
        state: The current agent graph state.
        llm_client: The provider-backed client used to make the structured model call.

    Returns:
        A crisis assessment derived from model output.
    """

    raw = await llm_client.generate_structured(
        prompt=build_crisis_classifier_prompt(state),
        response_schema=CrisisAssessmentSchema,
        system_instruction=build_crisis_classifier_system_prompt(),
        temperature=0,
    )

    level = max(0, min(3, int(raw.level)))
    confidence = (
        raw.confidence if raw.confidence in {"low", "medium", "high"} else "medium"
    )

    return CrisisAssessment(
        level=level,
        confidence=confidence,
        reason=raw.reason,
        needs_crisis_response=raw.needs_crisis_response,
        needs_clarification=raw.needs_clarification,
    )


def normalize_crisis_assessment(assessment: CrisisAssessment) -> CrisisAssessment:
    """Normalize crisis assessment fields into an internally consistent shape.

    Args:
        assessment: The raw crisis assessment.

    Returns:
        A normalized crisis assessment with bounded levels and consistent flags.
    """

    level = max(0, min(3, int(assessment.level)))
    confidence = (
        assessment.confidence
        if assessment.confidence in {"low", "medium", "high"}
        else "medium"
    )
    needs_crisis_response = assessment.needs_crisis_response or level >= 2

    # If we are already in crisis-response territory, clarification can still be useful,
    # but it should not weaken the urgency. Keep the flag, but the route will still be crisis.
    needs_clarification = assessment.needs_clarification

    return CrisisAssessment(
        level=level,
        confidence=confidence,
        reason=assessment.reason,
        needs_crisis_response=needs_crisis_response,
        needs_clarification=needs_clarification,
    )


def apply_crisis_result_to_state(
    state: AgentState,
    assessment: CrisisAssessment,
) -> AgentState:
    """Write the final crisis decision back into graph state.

    Args:
        state: The current agent graph state.
        assessment: The final normalized crisis assessment.

    Returns:
        The updated graph state.
    """

    state["crisis"] = assessment
    state["route"] = "crisis" if assessment.needs_crisis_response else "therapeutic"
    return state


async def run_crisis_gate(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Run the hybrid crisis gate.

    Args:
        state: The current agent graph state.
        llm_client: Optional provider-backed client for nuanced classification.

    Returns:
        The updated graph state with crisis fields and route populated.

    Notes:
        Flow:
        - apply deterministic override checks first
        - otherwise use the configured LLM classifier when available
        - otherwise fall back to deterministic assessment
        - normalize the result and store it in graph state
    """

    override = detect_crisis_override(state)
    if override is not None:
        _, override_assessment = override
        return apply_crisis_result_to_state(
            state,
            normalize_crisis_assessment(override_assessment),
        )

    if llm_client is not None:
        try:
            assessment = await assess_crisis_risk_with_llm(state, llm_client=llm_client)
        except Exception:
            assessment = assess_crisis_risk_deterministically(state)
    else:
        assessment = assess_crisis_risk_deterministically(state)

    return apply_crisis_result_to_state(
        state,
        normalize_crisis_assessment(assessment),
    )

"""Therapeutic response-style models used by memory-aware response planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.types.primitives import ConfidenceLevel
from agent.models import (
    GuidancePermission,
    SessionIntent,
    TherapeuticApproach,
)

TherapeuticResponseStyle = Literal[
    "supportive",
    "reflective",
    "psychoeducation",
    "guided_exercise",
    "closing",
    "clarifying",
    "technique",
]

SessionStage = Literal["opening", "deepening", "stabilizing", "closing"]

ExerciseStartBasis = Literal[
    "explicit_user_request",
    "accepted_assistant_offer",
    "ambiguous_or_none",
]
TurnRoute = Literal[
    "therapeutic",
    "memory_control",
    "grounded_lookup",
    "guided_exercise",
]
ActiveFlowAction = Literal["none", "continue", "preserve", "clear"]
ClarificationKind = Literal["none", "blocking", "soft"]
NoClarificationReason = Literal[
    "none",
    "safety_precedence",
    "explicit_privacy_control",
    "explicit_action_request",
    "clear_single_intent",
]


class TurnDispatchDecision(BaseModel):
    """Structured turn-level routing decision for runtime-owned dispatch."""

    route: TurnRoute
    active_flow_action: ActiveFlowAction = "none"
    clarification_needed: bool = False
    clarification_kind: ClarificationKind = "none"
    secondary_route: TurnRoute | None = None
    intent_summary: str = Field(
        default="",
        max_length=300,
        description=(
            "Compact private summary of mixed or ambiguous user intent for "
            "clarification-aware response planning. Do not script user-visible text."
        ),
    )
    clarification_question: str = Field(
        default="",
        max_length=240,
        description=(
            "Optional concise user-facing clarification question when "
            "clarification_kind is blocking."
        ),
    )
    no_clarification_reason: NoClarificationReason = "none"
    memory_reference_mode: Literal["none", "explicit"] = "none"
    query: str = Field(
        default="",
        description=(
            "Lookup query for grounded_lookup when the user's request needs "
            "external or source-backed information."
        ),
    )
    exercise_start_basis: ExerciseStartBasis = "ambiguous_or_none"
    exercise_type: str = ""
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: ConfidenceLevel


class DispatchDecision(BaseModel):
    """Legacy structured response-style decision used by test fixtures."""

    response_style: TherapeuticResponseStyle
    therapeutic_approach: TherapeuticApproach = "none"
    session_intent: SessionIntent = Field(
        default="vent",
        description=(
            "The user's conversational intent for this turn: vent, understand, "
            "reflect, work, regulate, repair, or close."
        ),
    )
    session_stage: SessionStage = Field(
        default="opening",
        description=(
            "The session arc stage for shaping the next reply: opening, "
            "deepening, stabilizing, or closing."
        ),
    )
    guidance_permission: GuidancePermission = Field(
        default="unknown",
        description=(
            "Whether the user has invited guidance: unknown, not_yet, or granted. "
            "Use not_yet when the user mainly needs to be heard; use granted when "
            "they ask for advice, next steps, structured work, or an exercise."
        ),
    )
    response_guidance: str = Field(
        default="",
        max_length=900,
        description=(
            "Compact private guidance for the next assistant reply. Describe "
            "the posture and one useful next move; do not script user-visible text."
        ),
    )
    exercise_start_basis: ExerciseStartBasis = Field(
        description=(
            "Whether the user explicitly authorized starting a guided exercise "
            "on this turn. Use ambiguous_or_none unless the user directly asks "
            "to do an exercise or cleanly accepts a specific assistant offer. "
            "Broad requests for help calming down are not explicit exercise "
            "requests."
        )
    )
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: ConfidenceLevel


__all__ = [
    "TherapeuticResponseStyle",
    "TherapeuticApproach",
    "SessionIntent",
    "SessionStage",
    "GuidancePermission",
    "ExerciseStartBasis",
    "TurnRoute",
    "ActiveFlowAction",
    "ClarificationKind",
    "NoClarificationReason",
    "TurnDispatchDecision",
    "DispatchDecision",
]

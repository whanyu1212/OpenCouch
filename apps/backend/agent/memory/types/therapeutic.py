"""Therapeutic response-style models used by memory-aware response planning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.types.primitives import ConfidenceLevel
from agent.therapeutic_policy import (
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


class TurnDispatchDecision(BaseModel):
    """Structured turn-level routing decision for runtime-owned dispatch."""

    route: TurnRoute
    active_flow_action: ActiveFlowAction = "none"
    memory_reference_mode: Literal["none", "explicit"] = "none"
    memory_action_type: (
        Literal[
            "status",
            "list",
            "set_recall",
            "save_preference",
            "forget_by_index",
            "forget_by_query",
            "confirm_pending",
            "cancel_pending",
        ]
        | None
    ) = None
    query: str = Field(
        default="",
        description=(
            "Lookup query for grounded_lookup or query text for query-based memory "
            "actions when relevant."
        ),
    )
    enabled: bool | None = None
    target_kind: str | None = None
    target_index: int | None = None
    preference_text: str = ""
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
    "TurnDispatchDecision",
    "DispatchDecision",
]

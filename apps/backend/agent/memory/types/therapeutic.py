"""Therapeutic-dispatch models used by the memory-aware subgraph."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.memory.types.primitives import ConfidenceLevel

TherapeuticResponseStyle = Literal[
    "supportive",
    "reflective",
    "psychoeducation",
    "guided_exercise",
    "closing",
    "clarifying",
    "technique",
]

TherapeuticApproach = Literal[
    "motivational_interviewing",
    "cbt",
    "act",
    "dbt_skills",
    "grief_support",
    "interpersonal_therapy",
    "pfa",
    "none",
]

SessionIntent = Literal[
    "vent",
    "understand",
    "reflect",
    "work",
    "regulate",
    "repair",
    "close",
]

SessionStage = Literal["opening", "deepening", "stabilizing", "closing"]

GuidancePermission = Literal["unknown", "not_yet", "granted"]

ExerciseStartBasis = Literal[
    "explicit_user_request",
    "accepted_assistant_offer",
    "ambiguous_or_none",
]


class DispatchDecision(BaseModel):
    """The structured output of the therapeutic_dispatch_node LLM call."""

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
    "DispatchDecision",
]

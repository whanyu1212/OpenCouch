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


class DispatchDecision(BaseModel):
    """The structured output of the therapeutic_dispatch_node LLM call."""

    response_style: TherapeuticResponseStyle
    therapeutic_approach: TherapeuticApproach = "none"
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: ConfidenceLevel


__all__ = [
    "TherapeuticResponseStyle",
    "TherapeuticApproach",
    "DispatchDecision",
]

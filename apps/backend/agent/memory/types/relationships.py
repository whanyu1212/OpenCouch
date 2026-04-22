"""Relationship and graph-edge models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent.memory.types.primitives import ConfidenceLevel

RelationshipKind = Literal[
    "mother",
    "father",
    "parent",
    "step_mother",
    "step_father",
    "sister",
    "brother",
    "sibling",
    "child",
    "grandparent",
    "grandchild",
    "other_family",
    "partner",
    "spouse",
    "ex_partner",
    "ex_spouse",
    "friend",
    "close_friend",
    "estranged_friend",
    "colleague",
    "boss",
    "subordinate",
    "client",
    "therapist",
    "doctor",
    "caregiver",
    "dependent",
    "other",
]

FAMILY_KINDS: frozenset[RelationshipKind] = frozenset(
    {
        "mother",
        "father",
        "parent",
        "step_mother",
        "step_father",
        "sister",
        "brother",
        "sibling",
        "child",
        "grandparent",
        "grandchild",
        "other_family",
    }
)
ROMANTIC_KINDS: frozenset[RelationshipKind] = frozenset(
    {"partner", "spouse", "ex_partner", "ex_spouse"}
)
FRIENDSHIP_KINDS: frozenset[RelationshipKind] = frozenset(
    {"friend", "close_friend", "estranged_friend"}
)
PROFESSIONAL_KINDS: frozenset[RelationshipKind] = frozenset(
    {"colleague", "boss", "subordinate", "client"}
)
CARE_KINDS: frozenset[RelationshipKind] = frozenset(
    {"therapist", "doctor", "caregiver", "dependent"}
)


class RelatesToEdge(BaseModel):
    """An inter-entity relationship edge with controlled-vocabulary kind."""

    kind: RelationshipKind
    kind_raw: str | None = None
    first_observed_at: str
    last_observed_at: str
    confidence: ConfidenceLevel


__all__ = [
    "RelationshipKind",
    "FAMILY_KINDS",
    "ROMANTIC_KINDS",
    "FRIENDSHIP_KINDS",
    "PROFESSIONAL_KINDS",
    "CARE_KINDS",
    "RelatesToEdge",
]

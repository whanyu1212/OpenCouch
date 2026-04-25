"""Shared primitive types for memory-layer models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ConfidenceLevel = Literal["low", "medium", "high"]
MemoryWriteTiming = Literal["immediate", "session_end", "promotion"]

EntityType = Literal[
    "User",
    "Person",
    "Concern",
    "Event",
    "CopingStrategy",
    "Goal",
    "Session",
    "Turn",
]


class EntityRef(BaseModel):
    """A reference to a graph entity by type and canonical identifier."""

    type: EntityType
    identifier: str = Field(min_length=1, max_length=200)


HotPathEdgeType = Literal[
    "KNOWS",
    "WORRIES_ABOUT",
    "EXPERIENCED",
    "USES",
    "WANTS",
    "PARTICIPATED_IN",
    "MENTIONED_IN",
]


__all__ = [
    "ConfidenceLevel",
    "MemoryWriteTiming",
    "EntityType",
    "EntityRef",
    "HotPathEdgeType",
]

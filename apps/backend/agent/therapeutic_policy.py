"""Shared therapeutic policy labels for text and voice runtimes."""

from __future__ import annotations

from typing import Literal

SessionIntent = Literal[
    "vent",
    "understand",
    "reflect",
    "work",
    "regulate",
    "repair",
    "close",
]

GuidancePermission = Literal["unknown", "not_yet", "granted"]

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

__all__ = [
    "SessionIntent",
    "GuidancePermission",
    "TherapeuticApproach",
]

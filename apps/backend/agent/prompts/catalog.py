"""Catalog of prompt composition inputs for response modes and modalities."""

from __future__ import annotations

from typing import Literal

ResponseMode = Literal[
    "supportive_conversation",
    "safety_check",
    "crisis_response",
    "orientation",
    "pattern_reflection",
    "guided_exercise",
    "psychoeducation",
    "out_of_scope",
    "realignment",
    "crisis_classifier",
]

Modality = Literal[
    "pfa",
    "cbt",
    "grief_support",
    "act",
]

MODE_FILES: dict[ResponseMode, tuple[str, ...]] = {
    "supportive_conversation": ("response_modes/support.md",),
    "safety_check": ("response_modes/safety_check.md",),
    "crisis_response": (
        "policy/crisis.md",
        "response_modes/crisis_response.md",
    ),
    "orientation": ("response_modes/orientation.md",),
    "pattern_reflection": ("response_modes/reflection.md",),
    "guided_exercise": ("response_modes/guided_exercise.md",),
    "psychoeducation": ("response_modes/psychoeducation.md",),
    "out_of_scope": ("response_modes/out_of_scope.md",),
    "realignment": ("response_modes/realignment.md",),
    "crisis_classifier": ("policy/crisis.md",),
}

MODALITY_FILES: dict[Modality, tuple[str, ...]] = {
    "pfa": ("modalities/pfa.md", "modalities/dbt_skills.md"),
    "cbt": ("modalities/cbt.md",),
    "grief_support": ("modalities/grief_support.md",),
    "act": ("modalities/act.md",),
}

MODE_BASELINE_FILES: dict[ResponseMode, tuple[str, ...]] = {
    "supportive_conversation": ("modalities/motivational_interviewing.md",),
    "orientation": ("modalities/motivational_interviewing.md",),
    "pattern_reflection": ("modalities/motivational_interviewing.md",),
    "realignment": ("modalities/motivational_interviewing.md",),
}

ALLOWED_MODALITIES: dict[ResponseMode, tuple[Modality, ...]] = {
    "supportive_conversation": (
        "cbt",
        "grief_support",
        "pfa",
        "act",
    ),
    "safety_check": ("pfa",),
    "crisis_response": ("pfa",),
    "orientation": (),
    "pattern_reflection": (
        "grief_support",
        "cbt",
        "act",
    ),
    "guided_exercise": ("cbt", "pfa", "act"),
    "psychoeducation": ("cbt", "act", "pfa", "grief_support"),
    "out_of_scope": (),
    "realignment": (),
    "crisis_classifier": (),
}

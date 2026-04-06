"""Catalog of prompt composition inputs for response modes and modalities."""

from __future__ import annotations

from typing import Literal

ResponseMode = Literal[
    "support",
    "safety_check",
    "crisis_response",
    "orientation",
    "reflection",
    "guided_exercise",
    "out_of_scope",
    "realignment",
    "crisis_classifier",
]

Modality = Literal[
    "pfa",
    "motivational_interviewing",
    "cbt",
    "grief_support",
]

MODE_FILES: dict[ResponseMode, tuple[str, ...]] = {
    "support": ("response_modes/support.md",),
    "safety_check": ("response_modes/safety_check.md",),
    "crisis_response": (
        "policy/crisis.md",
        "response_modes/crisis_response.md",
    ),
    "orientation": ("response_modes/orientation.md",),
    "reflection": ("response_modes/reflection.md",),
    "guided_exercise": ("response_modes/guided_exercise.md",),
    "out_of_scope": ("response_modes/out_of_scope.md",),
    "realignment": ("response_modes/realignment.md",),
    "crisis_classifier": ("policy/crisis.md",),
}

MODALITY_FILES: dict[Modality, tuple[str, ...]] = {
    "pfa": ("modalities/pfa.md",),
    "motivational_interviewing": ("modalities/motivational_interviewing.md",),
    "cbt": ("modalities/cbt.md",),
    "grief_support": ("modalities/grief_support.md",),
}

ALLOWED_MODALITIES: dict[ResponseMode, tuple[Modality, ...]] = {
    "support": ("motivational_interviewing", "cbt", "grief_support", "pfa"),
    "safety_check": ("pfa",),
    "crisis_response": ("pfa",),
    "orientation": ("motivational_interviewing",),
    "reflection": ("motivational_interviewing", "grief_support", "cbt"),
    "guided_exercise": ("cbt", "pfa"),
    "out_of_scope": (),
    "realignment": ("motivational_interviewing",),
    "crisis_classifier": (),
}

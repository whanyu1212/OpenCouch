"""Prompt source selection for therapeutic response modes."""

from __future__ import annotations

from agent.prompts.shared import CORE_SOURCES


_MODE_BASE_KNOWLEDGE: dict[str, tuple[str, ...]] = {
    "supportive": (*CORE_SOURCES, "response_modes/support.md"),
    "reflective": (*CORE_SOURCES, "response_modes/reflection.md"),
    "clarifying": CORE_SOURCES,
    "psychoeducation": (*CORE_SOURCES, "response_modes/psychoeducation.md"),
    "closing": (*CORE_SOURCES, "response_modes/closing.md"),
    "guided_exercise": (*CORE_SOURCES, "response_modes/guided_exercise.md"),
    # Technique mode uses ONLY core + the approach overlay. The approach
    # knowledge IS the mode-specific guidance — no separate response_mode file.
    "technique": CORE_SOURCES,
}

_MODALITY_FILES: dict[str, tuple[str, ...]] = {
    "motivational_interviewing": ("modalities/motivational_interviewing.md",),
    "cbt": ("modalities/cbt.md", "modalities/cbt_arc.md"),
    "act": ("modalities/act.md",),
    "dbt_skills": ("modalities/dbt_skills.md",),
    "grief_support": ("modalities/grief_support.md",),
    "interpersonal_therapy": ("modalities/interpersonal_therapy.md",),
    "pfa": ("modalities/pfa.md",),
}


def _knowledge_for_mode(mode: str, modality: str | None = None) -> tuple[str, ...]:
    """Compose the knowledge file list for a mode + modality combination.

    Returns the base knowledge for the mode, plus the modality overlay
    file(s) if a valid modality is specified. When modality is None or
    "none", only the base mode knowledge is returned.

    Args:
        mode: Therapeutic response mode name.
        modality: Optional therapeutic approach selected by the dispatcher.

    Returns:
        Prompt source paths to compose for the mode and modality.
    """

    base = _MODE_BASE_KNOWLEDGE.get(mode, CORE_SOURCES)
    if modality and modality != "none" and modality in _MODALITY_FILES:
        return (*base, *_MODALITY_FILES[modality])
    return base

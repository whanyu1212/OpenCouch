"""Prompt source selection for therapeutic response styles."""

from __future__ import annotations

from agent.prompts import CORE_SOURCES


_RESPONSE_STYLE_BASE_KNOWLEDGE: dict[str, tuple[str, ...]] = {
    "supportive": (*CORE_SOURCES, "response_styles/support.md"),
    "reflective": (*CORE_SOURCES, "response_styles/reflection.md"),
    "clarifying": CORE_SOURCES,
    "psychoeducation": (*CORE_SOURCES, "response_styles/psychoeducation.md"),
    "closing": (*CORE_SOURCES, "response_styles/closing.md"),
    "guided_exercise": (*CORE_SOURCES, "response_styles/guided_exercise.md"),
    # Technique uses ONLY core + the approach overlay. The approach knowledge
    # IS the style-specific guidance — no separate response style file.
    "technique": CORE_SOURCES,
}

_THERAPEUTIC_APPROACH_FILES: dict[str, tuple[str, ...]] = {
    "motivational_interviewing": (
        "therapeutic_approaches/motivational_interviewing.md",
    ),
    "cbt": ("therapeutic_approaches/cbt.md", "therapeutic_approaches/cbt_arc.md"),
    "act": ("therapeutic_approaches/act.md",),
    "dbt_skills": ("therapeutic_approaches/dbt_skills.md",),
    "grief_support": ("therapeutic_approaches/grief_support.md",),
    "interpersonal_therapy": ("therapeutic_approaches/interpersonal_therapy.md",),
    "pfa": ("therapeutic_approaches/pfa.md",),
}


def _knowledge_for_response_style(
    response_style: str,
    therapeutic_approach: str | None = None,
) -> tuple[str, ...]:
    """Compose the source files for a response style and therapeutic approach.

    Returns the base knowledge for the response style plus the approach
    overlay files when a valid therapeutic approach is specified.

    Args:
        response_style: Therapeutic response style name.
        therapeutic_approach: Optional approach selected by runtime context.

    Returns:
        Prompt source paths to compose.
    """

    base = _RESPONSE_STYLE_BASE_KNOWLEDGE.get(response_style, CORE_SOURCES)
    if (
        therapeutic_approach
        and therapeutic_approach != "none"
        and therapeutic_approach in _THERAPEUTIC_APPROACH_FILES
    ):
        return (*base, *_THERAPEUTIC_APPROACH_FILES[therapeutic_approach])
    return base

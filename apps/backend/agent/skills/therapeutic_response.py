"""Prompt-local skills for therapeutic response generation."""

from __future__ import annotations

from typing import cast

from agent.state import AgentState
from agent.specialists.therapeutic_prompts import (
    build_clarifying_system_prompt,
    build_closing_system_prompt,
    build_psychoeducation_system_prompt,
    build_reflective_system_prompt,
    build_supportive_system_prompt,
    build_technique_system_prompt,
)

THERAPEUTIC_RESPONSE_SKILL_STYLES = (
    "supportive",
    "reflective",
    "clarifying",
    "psychoeducation",
    "closing",
    "technique",
)

_SYSTEM_PROMPT_BUILDERS = {
    "supportive": build_supportive_system_prompt,
    "reflective": build_reflective_system_prompt,
    "clarifying": build_clarifying_system_prompt,
    "psychoeducation": build_psychoeducation_system_prompt,
    "closing": build_closing_system_prompt,
    "technique": build_technique_system_prompt,
}


def render_therapeutic_response_skill_context(
    state: AgentState,
    *,
    response_style: str,
    therapeutic_approach: str | None,
) -> str:
    """Render response-style guidance as a bounded prompt-local skill block."""

    style = _normalize_response_style(response_style)
    prompt_state = dict(state)
    approach = _normalize_therapeutic_approach(
        therapeutic_approach
        if therapeutic_approach is not None
        else prompt_state.get("therapeutic_approach")
    )
    prompt_state["therapeutic_approach"] = approach
    system_prompt = _SYSTEM_PROMPT_BUILDERS[style](cast(AgentState, prompt_state))
    return "\n".join(
        [
            "Therapeutic response skill:",
            f"- skill_id: therapeutic_response/{style}",
            "- version: 1",
            f"- response_style: {style}",
            f"- therapeutic_approach: {approach}",
            "- side_effect: none",
            "- retry_safe: true",
            "Operating boundaries:",
            "- Use this skill only for ordinary non-crisis therapeutic replies.",
            "- Crisis classification and crisis-resource lookup are not owned "
            "by this skill.",
            "- Guided-exercise step state is owned by GuidedExerciseAgent skills.",
            "- Apply the skill guidance to the current user message without "
            "naming the skill.",
            "",
            "Skill guidance:",
            system_prompt,
        ]
    )


def _normalize_response_style(response_style: str) -> str:
    style = " ".join(response_style.strip().lower().split())
    if style not in THERAPEUTIC_RESPONSE_SKILL_STYLES:
        return "supportive"
    return style


def _normalize_therapeutic_approach(value: object) -> str:
    approach = str(value or "").strip()
    return approach if approach else "none"


__all__ = [
    "THERAPEUTIC_RESPONSE_SKILL_STYLES",
    "render_therapeutic_response_skill_context",
]

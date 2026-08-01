"""OpenAI Agents SDK therapeutic response skill tools."""

from __future__ import annotations

from typing import Any, cast

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.runtime.context import OpenAITextRunContext
from agent.state import AgentState
from agent.specialists.therapeutic_response.style_guidance import (
    THERAPEUTIC_RESPONSE_STYLE_GUIDANCE_STYLES,
    render_therapeutic_response_skill_context,
)


class TherapeuticResponseSkillToolResult(BaseModel):
    """Structured result returned by therapeutic response skill tools."""

    skill_context: str = Field(
        description="Prompt-ready therapeutic response-style skill context."
    )
    response_style: str = Field(description="Selected therapeutic response style.")
    therapeutic_approach: str = Field(
        default="none",
        description="Therapeutic approach overlay used by the skill.",
    )
    side_effect: str = Field(
        default="none",
        description="Skill loading does not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the skill load can duplicate side effects.",
    )


async def execute_therapeutic_response_skill_tool(
    context: OpenAITextRunContext,
    *,
    response_style: str,
    therapeutic_approach: str | None = None,
) -> TherapeuticResponseSkillToolResult:
    """Render one therapeutic response skill for the current turn."""

    state = cast(AgentState, context.agent_state or {})
    style = " ".join(str(response_style or "supportive").strip().lower().split())
    if style not in THERAPEUTIC_RESPONSE_STYLE_GUIDANCE_STYLES:
        style = "supportive"
    approach = (
        str(therapeutic_approach).strip()
        if therapeutic_approach is not None
        else str(state.get("therapeutic_approach") or "none")
    )
    if not approach:
        approach = "none"
    skill_context = render_therapeutic_response_skill_context(
        state,
        response_style=style,
        therapeutic_approach=approach,
        prompt_appendix=context.workflow_context.prompt_appendix,
    )
    result = TherapeuticResponseSkillToolResult(
        skill_context=skill_context,
        response_style=style,
        therapeutic_approach=approach,
    )
    context.record_therapeutic_response_skill_tool_result(
        response_style=result.response_style,
        therapeutic_approach=result.therapeutic_approach,
        skill_context=result.skill_context,
    )
    return result


@function_tool(
    name_override="load_therapeutic_response_skill",
    description_override=(
        "Load a side-effect-free therapeutic response-style skill block for "
        "ordinary non-crisis replies. Use this before drafting a normal "
        "TherapeuticResponseAgent answer when no specialist operational tool "
        "has taken over the reply. Parameters: response_style is one of "
        "supportive, reflective, clarifying, psychoeducation, closing, or "
        "technique; therapeutic_approach is an optional overlay such as cbt, "
        "act, dbt_skills, motivational_interviewing, grief_support, "
        "interpersonal_therapy, pfa, or none. Side effects: none. Retry "
        "safety: safe."
    ),
)
async def load_therapeutic_response_skill(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    response_style: str,
    therapeutic_approach: str | None = None,
) -> TherapeuticResponseSkillToolResult:
    """Load one therapeutic response skill for the current turn."""

    return await execute_therapeutic_response_skill_tool(
        wrapper.context,
        response_style=response_style,
        therapeutic_approach=therapeutic_approach,
    )


def build_therapeutic_response_tools() -> list[Any]:
    """Return therapeutic response skill tools for the primary agent."""

    return [load_therapeutic_response_skill]


__all__ = [
    "TherapeuticResponseSkillToolResult",
    "build_therapeutic_response_tools",
    "execute_therapeutic_response_skill_tool",
    "load_therapeutic_response_skill",
]

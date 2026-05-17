"""OpenAI Agents SDK guided-exercise tools."""

from __future__ import annotations

from typing import Any

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, Field

from agent.text_runtime.openai_agents.context import OpenAITextRunContext
from agent.therapeutic.exercises.skills import render_exercise_skill_context


class GuidedExerciseSkillToolResult(BaseModel):
    """Structured result returned by guided-exercise skill tools."""

    skill_context: str = Field(
        description="Prompt-ready exercise skill context selected by the runtime."
    )
    exercise_type: str = Field(description="Registered exercise identifier.")
    current_step_index: int | None = Field(
        default=None,
        description="Current runtime step index, when one applies.",
    )
    runtime_action: str = Field(
        description="Runtime-owned action such as start, hold, advance, or exit."
    )
    side_effect: str = Field(
        default="none",
        description="Skill loading does not mutate durable state.",
    )
    retry_safe: bool = Field(
        default=True,
        description="Whether retrying the skill load can duplicate side effects.",
    )


async def execute_guided_exercise_skill_tool(
    context: OpenAITextRunContext,
    *,
    exercise_type: str,
    current_step_index: int | None,
    runtime_action: str,
) -> GuidedExerciseSkillToolResult:
    """Render one runtime-selected exercise skill through the catalog."""

    exercise_id = exercise_type.strip()
    if not exercise_id:
        raise ValueError("load_guided_exercise_skill requires exercise_type.")
    action = runtime_action.strip()
    if not action:
        raise ValueError("load_guided_exercise_skill requires runtime_action.")

    try:
        skill_context = render_exercise_skill_context(
            exercise_id,
            current_step_index=current_step_index,
            runtime_action=action,
        )
    except KeyError:
        skill_context = (
            "Exercise skill:\n"
            f"- skill_id: {exercise_id}\n"
            f"- runtime_action: {action}\n"
            "- registry_status: unavailable\n"
            "Operating boundaries:\n"
            "- Follow the runtime task exactly and do not invent extra steps."
        )
    result = GuidedExerciseSkillToolResult(
        skill_context=skill_context,
        exercise_type=exercise_id,
        current_step_index=current_step_index,
        runtime_action=action,
    )
    context.record_guided_exercise_skill_tool_result(
        exercise_type=result.exercise_type,
        current_step_index=result.current_step_index,
        runtime_action=result.runtime_action,
        skill_context=result.skill_context,
    )
    return result


@function_tool(
    name_override="load_guided_exercise_skill",
    description_override=(
        "Load the runtime-selected guided-exercise skill block for the current "
        "step and action. Use only when the runtime prompt requires it for a "
        "GuidedExerciseAgent turn. Side effects: none. Retry safety: safe."
    ),
)
async def load_guided_exercise_skill(
    wrapper: RunContextWrapper[OpenAITextRunContext],
    exercise_type: str,
    runtime_action: str,
    current_step_index: int | None = None,
) -> GuidedExerciseSkillToolResult:
    """Load one guided-exercise skill selected by the app runtime."""

    return await execute_guided_exercise_skill_tool(
        wrapper.context,
        exercise_type=exercise_type,
        current_step_index=current_step_index,
        runtime_action=runtime_action,
    )


def build_guided_exercise_tools() -> list[Any]:
    """Return guided-exercise tools for the OpenAI specialist."""

    return [load_guided_exercise_skill]


__all__ = [
    "GuidedExerciseSkillToolResult",
    "build_guided_exercise_tools",
    "execute_guided_exercise_skill_tool",
    "load_guided_exercise_skill",
]

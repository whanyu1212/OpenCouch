"""Structured prompt directives for guided-exercise response generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.prompts.tool_forcing import force_tool_directive
from agent.skills.guided_exercises.rendering.skill_context import (
    render_exercise_skill_context,
)


@dataclass(frozen=True)
class GuidedExerciseDirective:
    """Runtime-owned directive for one guided-exercise response turn."""

    exercise_type: str
    runtime_action: str
    current_step_index: int | None
    runtime_task: str


def render_full_guided_exercise_directive(directive: GuidedExerciseDirective) -> str:
    """Render a directive with full prompt-local skill context."""

    return (
        f"{_render_skill_context(directive)}\n\nRuntime task:\n{directive.runtime_task}"
    )


def render_tool_forced_guided_exercise_directive(
    directive: GuidedExerciseDirective,
) -> str:
    """Render a directive that requires loading skill context by tool."""

    arguments: dict[str, Any] = {
        "exercise_type": directive.exercise_type,
        "runtime_action": directive.runtime_action,
    }
    if directive.current_step_index is not None:
        arguments["current_step_index"] = directive.current_step_index
    replacement = (
        "Exercise skill:\n"
        "(skill context is owned by GuidedExerciseAgent tools)\n"
        + force_tool_directive("load_guided_exercise_skill", arguments)
        + "Use only the returned skill_context plus the Runtime task below. "
        "Do not invent exercise steps, switch exercises, or offer a menu."
    )
    return f"{replacement}\n\nRuntime task:\n{directive.runtime_task}"


def _render_skill_context(directive: GuidedExerciseDirective) -> str:
    try:
        return render_exercise_skill_context(
            directive.exercise_type,
            current_step_index=directive.current_step_index,
            runtime_action=directive.runtime_action,
        )
    except KeyError:
        return (
            "Exercise skill:\n"
            f"- skill_id: {directive.exercise_type}\n"
            f"- runtime_action: {directive.runtime_action}\n"
            "- registry_status: unavailable\n"
            "Operating boundaries:\n"
            "- Follow the runtime task exactly and do not invent extra steps."
        )


__all__ = [
    "GuidedExerciseDirective",
    "render_full_guided_exercise_directive",
    "render_tool_forced_guided_exercise_directive",
]

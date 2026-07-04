"""Prompt-to-tool instruction helpers for guided exercise execution."""

from __future__ import annotations

from dataclasses import dataclass

from agent.runtime.context import OpenAITextRunContext
from agent.skills.guided_exercises.rendering.directives import (
    GuidedExerciseDirective,
    render_tool_forced_guided_exercise_directive,
)


@dataclass(frozen=True)
class _ExerciseSkillToolRequest:
    exercise_type: str
    runtime_action: str
    current_step_index: int | None


def _replace_exercise_skill_context_with_tool_instruction(
    prompt: str,
) -> tuple[str, _ExerciseSkillToolRequest | None]:
    skill_start = prompt.find("Exercise skill:")
    runtime_task_marker = "\n\nRuntime task:"
    runtime_task_start = prompt.find(runtime_task_marker, skill_start)
    if skill_start == -1 or runtime_task_start == -1:
        return prompt, None

    skill_block = prompt[skill_start:runtime_task_start]
    exercise_type = _skill_block_value(skill_block, "skill_id")
    runtime_action = _skill_block_value(skill_block, "runtime_action")
    if not exercise_type or not runtime_action:
        return prompt, None

    current_step_index = _parse_optional_int(_skill_block_value(skill_block, "index"))
    request = _ExerciseSkillToolRequest(
        exercise_type=exercise_type,
        runtime_action=runtime_action,
        current_step_index=current_step_index,
    )
    replacement = _render_tool_forced_skill_block(request)
    return (
        f"{prompt[:skill_start]}{replacement}{prompt[runtime_task_start:]}",
        request,
    )


def _render_tool_forced_skill_block(request: _ExerciseSkillToolRequest) -> str:
    """Render the forced-tool skill block while preserving the legacy suffix."""

    directive = GuidedExerciseDirective(
        exercise_type=request.exercise_type,
        runtime_action=request.runtime_action,
        current_step_index=request.current_step_index,
        runtime_task="",
    )
    rendered = render_tool_forced_guided_exercise_directive(directive)
    return rendered.removesuffix("\n\nRuntime task:\n")


def _skill_block_value(block: str, key: str) -> str:
    prefix = f"- {key}:"
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()
    return ""


def _parse_optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _guided_exercise_skill_tool_called(
    run_context: OpenAITextRunContext,
    *,
    tool_call_count: int,
) -> bool:
    return len(run_context.guided_exercise_skill_tool_calls) > tool_call_count


__all__ = [
    "_ExerciseSkillToolRequest",
    "_guided_exercise_skill_tool_called",
    "_parse_optional_int",
    "_replace_exercise_skill_context_with_tool_instruction",
    "_skill_block_value",
]

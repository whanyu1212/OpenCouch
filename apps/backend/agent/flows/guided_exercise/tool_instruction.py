"""Prompt-to-tool instruction helpers for guided exercise execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.flows.tool_forcing import force_tool_directive
from agent.runtime.context import OpenAITextRunContext


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
    arguments: dict[str, Any] = {
        "exercise_type": exercise_type,
        "runtime_action": runtime_action,
    }
    if current_step_index is not None:
        arguments["current_step_index"] = current_step_index

    replacement = (
        "Exercise skill:\n"
        "(skill context is owned by GuidedExerciseAgent tools)\n"
        + force_tool_directive("load_guided_exercise_skill", arguments)
        + "Use only the "
        "returned skill_context plus the Runtime task below. Do not invent "
        "exercise steps, switch exercises, or offer a menu."
    )
    return (
        f"{prompt[:skill_start]}{replacement}{prompt[runtime_task_start:]}",
        _ExerciseSkillToolRequest(
            exercise_type=exercise_type,
            runtime_action=runtime_action,
            current_step_index=current_step_index,
        ),
    )


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

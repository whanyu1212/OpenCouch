"""Exercise skill rendering for guided-exercise response generation.

OpenAI's Skills guidance treats skills as bounded, reviewed bundles the model
can read when a workflow needs specialized procedure knowledge. OpenCouch keeps
the exercise catalog application-owned, so this module renders one selected
catalog entry as a prompt-local skill block instead of exposing a user-selected
skill catalog to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.skills.guided_exercises.rendering.skill_docs import (
    get_guided_exercise_skill_doc,
)
from agent.skills.guided_exercises.registry import get_exercise_definition
from agent.skills.guided_exercises.types import CompletionMode, ExerciseDefinition


@dataclass(frozen=True)
class ExerciseSkillStep:
    """Prompt-ready view of one exercise step."""

    index: int
    step_id: str
    instruction: str
    completion_mode: CompletionMode
    completion_criteria: str
    target_items: int | None
    min_items: int | None


@dataclass(frozen=True)
class ExerciseSkill:
    """Prompt-ready bounded skill generated from an exercise definition."""

    skill_id: str
    version: int
    name: str
    when_to_use: str
    category: str
    tags: tuple[str, ...]
    intensity: str
    duration_seconds: int | None
    supported_channels: tuple[str, ...]
    required_capability: str | None
    steps: tuple[ExerciseSkillStep, ...]


def build_exercise_skill(exercise_type: str) -> ExerciseSkill:
    """Build a bounded exercise skill from the registered catalog entry.

    Args:
        exercise_type: Registered exercise identifier.

    Returns:
        ExerciseSkill generated from the matching definition.

    Raises:
        KeyError: If ``exercise_type`` is not registered.
    """

    definition = get_exercise_definition(exercise_type)
    if definition is None:
        raise KeyError(exercise_type)
    return _skill_from_definition(definition)


def render_exercise_skill_context(
    exercise_type: str,
    *,
    current_step_index: int | None,
    runtime_action: str,
) -> str:
    """Render the selected exercise as a compact prompt-local skill block.

    Args:
        exercise_type: Registered exercise identifier.
        current_step_index: Current runtime step, when one exists.
        runtime_action: Runtime-owned action such as start, hold, advance, or exit.

    Returns:
        Skill block suitable for the guided-exercise task prompt.

    Raises:
        KeyError: If ``exercise_type`` is not registered.
    """

    skill = build_exercise_skill(exercise_type)
    lines = [
        "Exercise skill:",
        f"- skill_id: {skill.skill_id}",
        f"- version: {skill.version}",
        f"- name: {skill.name}",
        f"- runtime_action: {runtime_action}",
        f"- when_to_use: {skill.when_to_use}",
        f"- category: {skill.category or 'general'}",
        f"- tags: {_format_tuple(skill.tags)}",
        f"- intensity: {skill.intensity}",
        f"- duration_seconds: {_format_optional_int(skill.duration_seconds)}",
        f"- supported_channels: {_format_tuple(skill.supported_channels)}",
        f"- required_capability: {skill.required_capability or 'none'}",
    ]
    lines.extend(_skill_doc_guidance(exercise_type))
    lines.extend(
        [
            "Operating boundaries:",
            "- Use this skill only because the application runtime selected it.",
            "- Follow the current runtime step exactly; do not skip, reorder, or add steps.",
            "- Rephrase step instructions naturally without changing the task.",
            "- Keep the reply brief and paced for one exercise turn.",
            "- If the runtime action is exit or complete, do not start another exercise.",
        ]
    )

    if current_step_index is not None:
        step = _step_at(skill, current_step_index)
        if step is not None:
            lines.extend(
                [
                    "Current runtime step:",
                    f"- index: {step.index}",
                    f"- step_id: {step.step_id}",
                    f"- completion_mode: {step.completion_mode}",
                    f"- completion_criteria: {step.completion_criteria or 'default'}",
                    f"- target_items: {_format_optional_int(step.target_items)}",
                    f"- min_items: {_format_optional_int(step.min_items)}",
                    f'- canonical_instruction: "{step.instruction}"',
                ]
            )

    lines.append("Step map:")
    for step in skill.steps:
        details = [f"{step.index}. {step.step_id}", step.completion_mode]
        if step.target_items is not None:
            details.append(f"target_items={step.target_items}")
        if step.min_items is not None:
            details.append(f"min_items={step.min_items}")
        lines.append(f"- {'; '.join(details)}")
    return "\n".join(lines)


def _skill_doc_guidance(exercise_type: str) -> list[str]:
    skill_doc = get_guided_exercise_skill_doc(exercise_type)
    if skill_doc is None:
        return []
    return [
        "Skill document guidance:",
        skill_doc.body,
    ]


def _skill_from_definition(definition: ExerciseDefinition) -> ExerciseSkill:
    return ExerciseSkill(
        skill_id=definition.id,
        version=definition.version,
        name=definition.display_name,
        when_to_use=definition.selection_use_case,
        category=definition.category,
        tags=definition.tags,
        intensity=definition.intensity,
        duration_seconds=definition.duration_seconds,
        supported_channels=definition.channels,
        required_capability=definition.required_skill,
        steps=tuple(
            ExerciseSkillStep(
                index=index,
                step_id=step.id,
                instruction=step.instruction,
                completion_mode=step.completion_mode,
                completion_criteria=step.completion_criteria,
                target_items=step.target_items,
                min_items=step.min_items,
            )
            for index, step in enumerate(definition.steps)
        ),
    )


def _step_at(skill: ExerciseSkill, index: int) -> ExerciseSkillStep | None:
    if index < 0 or index >= len(skill.steps):
        return None
    return skill.steps[index]


def _format_tuple(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


def _format_optional_int(value: int | None) -> str:
    return str(value) if value is not None else "none"


__all__ = [
    "ExerciseSkill",
    "ExerciseSkillStep",
    "build_exercise_skill",
    "render_exercise_skill_context",
]

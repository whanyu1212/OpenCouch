"""LiveKit AgentTask for structured therapeutic exercises.

GroundingTask wraps the shared therapeutic exercise registry into a LiveKit
AgentTask that can be ``await``ed from a ``@function_tool`` on the
TherapeuticAgent.

Voice mode keeps exercise choice LLM-owned: the parent realtime model passes a
supported ``exercise_type`` id to the tool. The task only validates capability
constraints, builds instructions from the definition, and exposes
``complete_step()`` / ``exit_exercise()`` tools so the model can conduct the
exercise naturally.

Usage from TherapeuticAgent::

    @function_tool()
    async def start_grounding_exercise(self, context: RunContext, exercise_type: str):
        result = await GroundingTask(exercise_type=exercise_type)
        return f"Exercise finished: {result}"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from livekit.agents import AgentTask, RunContext, function_tool

from agent.therapeutic.exercises.registry import (
    get_exercise_display_name,
    get_exercise_steps,
    iter_exercise_definitions,
    voice_exercise_ids,
)
from agent.therapeutic.exercises.types import ExerciseStep
from agent.voice.session_data import SessionData

logger = logging.getLogger(__name__)


VOICE_EXERCISES: set[str] = set(voice_exercise_ids())

TEXT_EXERCISES: set[str] = {definition.id for definition in iter_exercise_definitions()}


@dataclass
class ExerciseResult:
    """Result returned when a GroundingTask completes."""

    exercise_type: str
    display_name: str
    steps_completed: int
    total_steps: int
    outcome: Literal["completed", "exited"]


def supported_exercise_ids(
    input_modality: Literal["voice", "text"] = "voice",
) -> tuple[str, ...]:
    """Return exercise ids available for a voice task.

    Args:
        input_modality: Current user input modality.

    Returns:
        Sorted supported exercise ids for the modality.
    """

    allowed_exercises = TEXT_EXERCISES if input_modality == "text" else VOICE_EXERCISES
    return tuple(sorted(allowed_exercises))


def _resolve_exercise(
    exercise_type: str,
    *,
    input_modality: Literal["voice", "text"] = "voice",
) -> tuple[str, tuple[ExerciseStep, ...]]:
    """Validate an LLM-selected exercise id and return its steps.

    Args:
        exercise_type: Exact exercise id chosen by the voice model.
        input_modality: Current user input modality.

    Returns:
        Exercise id and ordered steps.
    """

    normalized_exercise_type = exercise_type.strip()
    allowed_exercises = set(supported_exercise_ids(input_modality))

    if normalized_exercise_type not in allowed_exercises:
        raise ValueError(
            f"Unsupported {input_modality} exercise_type {normalized_exercise_type!r}."
        )

    steps = get_exercise_steps(normalized_exercise_type)
    if steps is None:
        raise KeyError(normalized_exercise_type)
    return normalized_exercise_type, steps


def _build_exercise_instructions(
    exercise_type: str,
    steps: tuple[ExerciseStep, ...],
) -> str:
    """Build the agent instructions for a grounding exercise.

    Gives the LLM the full exercise plan so it can guide the user
    naturally, but with explicit rules about pacing.
    """
    display_name = get_exercise_display_name(exercise_type, default="this exercise")
    total = len(steps)

    step_plan = []
    for i, step in enumerate(steps):
        if step.completion_mode == "confirmation":
            mode_label = (
                "Ask the user to tell you when they have done it, then wait "
                "for that confirmation."
            )
        elif step.completion_mode == "items":
            mode_label = (
                f"Wait for the user to name at least {step.min_items or 1} item(s)."
            )
        else:
            mode_label = (
                "Wait for a meaningful response that satisfies the step, then advance."
            )
            if step.completion_criteria:
                mode_label = f"{mode_label} Criteria: {step.completion_criteria}"
        step_plan.append(f"Step {i + 1}: {step.instruction}\n  -> {mode_label}")

    plan_text = "\n\n".join(step_plan)

    return f"""You are guiding the user through {display_name} ({total} steps).

RULES:
- Deliver ONE step at a time. Do NOT skip ahead or combine steps.
- After delivering a step, WAIT for the user to respond before moving on.
- You cannot see whether the user has done a body, breathing, or imagery action. For confirmation steps, explicitly ask them to tell you when they have done it.
- When the user has engaged with the current step (named items, confirmed, or responded meaningfully), call complete_step() to advance.
- If the user says they want to stop, skip, or can't continue, call exit_exercise().
- Keep your voice warm, patient, and unhurried. Brief encouragement between steps is good.
- Rephrase the step instructions in natural spoken language. Do not read them like a script.
- Do not announce step numbers unless it helps orientation.
- The exercise should feel like you are doing it with the user, not administering instructions.
- If the user gives a partial response (e.g. names 1 item when asked for 3), gently encourage them to continue the same step. Do NOT call complete_step() yet.
- If the user seems stuck, frustrated, or self-conscious, normalize that briefly and offer a simpler version of the same step.

EXERCISE PLAN:
{plan_text}

Start with Step 1 now."""


class GroundingTask(AgentTask[ExerciseResult]):
    """A bounded voice exercise that returns to the parent agent on completion.

    Args:
        exercise_type: Exact exercise id chosen by the parent voice model.
        chat_ctx: Optional chat context to carry over from the parent.
    """

    def __init__(
        self,
        exercise_type: str,
        chat_ctx=None,
        input_modality: Literal["voice", "text"] = "voice",
    ) -> None:
        exercise_type, steps = _resolve_exercise(
            exercise_type,
            input_modality=input_modality,
        )
        self._exercise_type = exercise_type
        self._steps = steps
        self._current_step = 0
        self._total_steps = len(steps)
        self._display_name = get_exercise_display_name(
            exercise_type,
            default="this exercise",
        )

        super().__init__(
            instructions=_build_exercise_instructions(exercise_type, steps),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        logger.info(
            "GroundingTask: starting exercise=%s steps=%d",
            self._exercise_type,
            self._total_steps,
        )
        self.session.generate_reply(
            instructions=f"Guide the user into Step 1 of {self._display_name}. "
            "Use warm, spoken language. Do not say 'Step 1' literally. "
            "If the step asks the user to do an action, end by asking them "
            "to tell you when they have done it."
        )

    @function_tool()
    async def complete_step(self, context: RunContext[SessionData]) -> str:
        """Mark the current step as complete and advance to the next one.

        Call this when the user has meaningfully engaged with the
        current step — named the requested items, confirmed they
        did the action, or responded with substance.
        """
        self._current_step += 1

        if self._current_step >= self._total_steps:
            logger.info(
                "GroundingTask: exercise completed exercise=%s",
                self._exercise_type,
            )
            self.complete(
                ExerciseResult(
                    exercise_type=self._exercise_type,
                    display_name=self._display_name,
                    steps_completed=self._total_steps,
                    total_steps=self._total_steps,
                    outcome="completed",
                )
            )
            return (
                f"The exercise is now complete. Acknowledge what the user "
                f"just did, mention they finished {self._display_name}, and "
                f"gently ask how it felt for them."
            )

        next_step = self._steps[self._current_step]
        if next_step.completion_mode == "confirmation":
            completion_hint = (
                "Because you cannot see the user, ask them to tell you when "
                "they have done it."
            )
        elif next_step.completion_mode == "items":
            completion_hint = "Wait for the user to answer with the requested items."
        else:
            completion_hint = (
                "Wait for the user to answer with a meaningful response before "
                "calling complete_step()."
            )
        return (
            f"Step {self._current_step} is done. Moving to step "
            f"{self._current_step + 1} of {self._total_steps}. "
            f"Next step instruction: {next_step.instruction} "
            f"Rephrase this naturally and deliver it to the user. {completion_hint}"
        )

    @function_tool()
    async def exit_exercise(self, context: RunContext[SessionData]) -> str:
        """Exit the exercise early because the user wants to stop.

        Call this when the user says they want to stop, skip, quit,
        or indicates the exercise is not helping.
        """
        logger.info(
            "GroundingTask: user exited exercise=%s at step=%d/%d",
            self._exercise_type,
            self._current_step + 1,
            self._total_steps,
        )
        self.complete(
            ExerciseResult(
                exercise_type=self._exercise_type,
                display_name=self._display_name,
                steps_completed=self._current_step,
                total_steps=self._total_steps,
                outcome="exited",
            )
        )
        return (
            "The exercise has been stopped. Warmly acknowledge that "
            "stopping is completely fine, and ask what would feel "
            "most helpful right now."
        )

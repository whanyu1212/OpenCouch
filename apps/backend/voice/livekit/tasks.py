"""LiveKit AgentTask for structured therapeutic exercises.

GroundingTask wraps the exercise registry from
``agent/therapeutic/guided_exercise.py`` into a LiveKit AgentTask
that can be ``await``ed from a ``@function_tool`` on the
TherapeuticAgent.

The existing exercise engine uses a deterministic state-machine
classifier (complete/hold/stuck/exit) to drive step transitions.
In voice mode the RealtimeModel handles this more naturally: the
LLM judges when the user has engaged with a step and calls
``complete_step()`` to advance.  ``exit_exercise()`` lets the user
bail out at any time.

Usage from TherapeuticAgent::

    @function_tool()
    async def start_grounding_exercise(self, context: RunContext, technique: str):
        result = await GroundingTask(technique=technique)
        return f"Exercise finished: {result}"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from livekit.agents import AgentTask, RunContext, function_tool

from agent.therapeutic.exercises.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_GRATITUDE,
    EXERCISE_IMPROVE,
    EXERCISE_MUSCLE_RELAXATION,
    EXERCISE_SELF_COMPASSION,
    EXERCISE_STOP_TECHNIQUE,
    get_exercise_display_name,
    get_exercise_steps,
    iter_exercise_definitions,
    iter_exercise_selectors,
    voice_exercise_ids,
)
from agent.therapeutic.exercises.types import ExerciseStep
from voice.livekit.session_data import SessionData

logger = logging.getLogger(__name__)


# Subset of exercises suitable for voice-guided delivery.
# Sequential worksheet-like exercises (thought records and behavioral
# experiments) work better in text mode where the user can re-read prompts.
# Voice mode favors body-based, verbal, and low-visual-load exercises.
VOICE_EXERCISES: set[str] = set(voice_exercise_ids())

TEXT_EXERCISES: set[str] = {definition.id for definition in iter_exercise_definitions()}

_GENERIC_VOICE_EXERCISE_ROTATION: tuple[str, ...] = (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_STOP_TECHNIQUE,
    EXERCISE_MUSCLE_RELAXATION,
    EXERCISE_SELF_COMPASSION,
    EXERCISE_IMPROVE,
    EXERCISE_GRATITUDE,
)


@dataclass
class ExerciseResult:
    """Result returned when a GroundingTask completes."""

    exercise_type: str
    display_name: str
    steps_completed: int
    total_steps: int
    outcome: Literal["completed", "exited"]


def _match_requested_exercise(technique: str) -> str | None:
    """Return the exercise directly implied by the technique text, if any."""
    lowered = technique.lower()
    for keywords, exercise_type in iter_exercise_selectors():
        for kw in keywords:
            if re.search(kw, lowered):
                return exercise_type
    return None


def _pick_diversified_generic_exercise(recent_exercise_types: tuple[str, ...]) -> str:
    """Rotate generic grounding requests away from recently used exercises."""
    recent = set(recent_exercise_types[-3:])
    for exercise_type in _GENERIC_VOICE_EXERCISE_ROTATION:
        if exercise_type not in recent:
            return exercise_type
    return _GENERIC_VOICE_EXERCISE_ROTATION[0]


def _resolve_exercise(
    technique: str,
    *,
    recent_exercise_types: tuple[str, ...] = (),
    input_modality: Literal["voice", "text"] = "voice",
) -> tuple[str, tuple[ExerciseStep, ...]]:
    """Map a free-text technique request to an exercise type and steps.

    Uses the existing keyword selector from guided_exercise.py.
    Falls back to a rotated voice-friendly exercise for underspecified
    requests so the session does not keep repeating the same default.
    """
    requested_exercise = _match_requested_exercise(technique)
    if requested_exercise is None:
        exercise_type = _pick_diversified_generic_exercise(recent_exercise_types)
    else:
        exercise_type = requested_exercise

    allowed_exercises = TEXT_EXERCISES if input_modality == "text" else VOICE_EXERCISES

    # Typed turns can use the full registry. Spoken turns stay on the
    # voice-safe subset to avoid awkward read-aloud cognitive exercises.
    if exercise_type not in allowed_exercises:
        exercise_type = EXERCISE_5_4_3_2_1

    steps = get_exercise_steps(exercise_type)
    if steps is None:
        raise KeyError(exercise_type)
    return exercise_type, steps


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
        mode_label = (
            "Ask the user to tell you when they have done it, then wait for that confirmation."
            if step.completion_mode == "user_confirmation"
            else f"Wait for the user to name at least {step.min_count_for_completion} item(s)."
        )
        step_plan.append(f"Step {i + 1}: {step.prompt_fallback}\n  -> {mode_label}")

    plan_text = "\n\n".join(step_plan)

    return f"""You are guiding the user through {display_name} ({total} steps).

RULES:
- Deliver ONE step at a time. Do NOT skip ahead or combine steps.
- After delivering a step, WAIT for the user to respond before moving on.
- You cannot see whether the user has done a body, breathing, or imagery action. For user-confirmation steps, explicitly ask them to tell you when they have done it.
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
        technique: Free-text description of what the user wants
            (e.g. "breathing", "grounding", "body scan"). Mapped to
            the closest exercise via keyword matching.
        chat_ctx: Optional chat context to carry over from the parent.
    """

    def __init__(
        self,
        technique: str = "grounding",
        chat_ctx=None,
        recent_exercise_types: tuple[str, ...] = (),
        input_modality: Literal["voice", "text"] = "voice",
    ) -> None:
        exercise_type, steps = _resolve_exercise(
            technique,
            recent_exercise_types=recent_exercise_types,
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
        completion_hint = (
            "Because you cannot see the user, ask them to tell you when they "
            "have done it."
            if next_step.completion_mode == "user_confirmation"
            else "Wait for the user to answer with the requested item or items."
        )
        return (
            f"Step {self._current_step} is done. Moving to step "
            f"{self._current_step + 1} of {self._total_steps}. "
            f"Next step instruction: {next_step.prompt_fallback} "
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

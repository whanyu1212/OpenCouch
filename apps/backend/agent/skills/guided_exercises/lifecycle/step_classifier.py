"""Classify progress within an active guided exercise step."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from agent.skills.guided_exercises.catalog.registry import get_exercise_display_name
from agent.skills.guided_exercises.catalog.types import (
    ExerciseStep,
    ExerciseStepDecision,
    StepState,
)

# ── Step-state classifier ──────────────────────────────────────────────


def _default_completion_criteria(current_step: ExerciseStep) -> str:
    """Return generic completion criteria when a step does not specify one.

    Args:
        current_step: Exercise step the user is responding to.

    Returns:
        Natural-language criteria for the classifier prompt.
    """

    if current_step.completion_criteria:
        return current_step.completion_criteria
    if current_step.completion_mode == "items":
        min_items = current_step.min_items or 1
        target_hint = (
            f" out of the requested {current_step.target_items}"
            if current_step.target_items is not None
            else ""
        )
        return f"Complete when the user names at least {min_items} relevant item(s){target_hint}."
    if current_step.completion_mode == "confirmation":
        return "Complete when the user indicates they performed the private action or are ready to continue."
    if current_step.completion_mode == "llm_judged":
        return "Complete when the user's reply satisfies the step instruction in substance, even if imperfect."
    return "Complete when the user gives a substantive answer to the step instruction."


def _build_step_classifier_prompt(
    *,
    state: AgentState,
    exercise_type: str,
    step_index: int,
    current_step: ExerciseStep,
) -> str:
    """Build the LLM prompt for guided-exercise step classification.

    Args:
        state: Current runtime state.
        exercise_type: Active exercise identifier.
        step_index: Current zero-based exercise step.
        current_step: Exercise step the user is responding to.

    Returns:
        Prompt asking for a structured step-state decision.
    """

    message = state.get("message", "")
    exercise_name = get_exercise_display_name(exercise_type)
    return (
        "Classify the user's latest reply to the current guided-exercise step. "
        "Return exactly one step_state:\n"
        "- complete: the user appears to have completed the requested step, "
        "including natural confirmations like 'done that' or equivalent wording.\n"
        "- hold: the user is tentative, partial, off-step, or still engaging but "
        "has not clearly completed the step.\n"
        "- stuck: the user says they cannot do the step, nothing comes to mind, "
        "or the exercise feels confusing/frustrating.\n"
        "- exit: the user wants to stop, cancel, switch away, or just talk.\n\n"
        "If uncertain between complete and hold, choose hold. If the reply "
        "clearly opts out, choose exit.\n\n"
        f"Exercise: {exercise_name} ({exercise_type})\n"
        f"Current step index: {step_index}\n"
        f"Completion mode: {current_step.completion_mode}\n"
        f"Completion criteria: {_default_completion_criteria(current_step)}\n"
        f"Target items: {current_step.target_items if current_step.target_items is not None else '(none)'}\n"
        f"Minimum items: {current_step.min_items if current_step.min_items is not None else '(none)'}\n"
        f'Step instruction: "{current_step.instruction}"\n'
        f'User reply: "{message}"'
    )


def _message_requests_next_confirmation_step(
    *,
    state: AgentState,
    current_step: ExerciseStep,
) -> bool:
    if current_step.completion_mode != "confirmation":
        return False
    text = " ".join(
        str(state.get("message") or "")
        .casefold()
        .replace("’", "'")
        .replace("`", "'")
        .split()
    )
    if not text:
        return False
    decline_cues = (
        "not ready",
        "not able",
        "not yet",
        "not sure i can",
        "not sure i'm ready",
        "can't",
        "cant",
        "cannot",
        "can not",
        "unable",
        "don't want",
        "do not want",
        "won't",
        "wouldn't",
        "rather not",
        "need more time",
    )
    if any(cue in text for cue in decline_cues):
        return False
    hold_cues = (
        "distracted",
        "where we left off",
        "where i left off",
        "same step",
        "repeat",
        "restart",
        "start over",
    )
    if any(cue in text for cue in hold_cues):
        return False
    advance_cues = (
        "next step",
        "one more step",
        "another step",
        "move on",
        "move to the next",
        "keep going",
    )
    return any(cue in text for cue in advance_cues)


async def classify_step_state(
    *,
    state: AgentState,
    classifier_llm: Any,
    exercise_type: str,
    step_index: int,
    current_step: ExerciseStep,
) -> StepState:
    """Classify step progress with the control-plane LLM.

    Args:
        state: Current runtime state.
        classifier_llm: Control-plane LLM client.
        exercise_type: Active exercise identifier.
        step_index: Current zero-based exercise step.
        current_step: Exercise step the user is responding to.

    Returns:
        Step-state classification for the current exercise turn.
    """

    if classifier_llm is None:
        raise RuntimeError("Guided exercise step classification requires an LLM.")

    if _message_requests_next_confirmation_step(
        state=state,
        current_step=current_step,
    ):
        return "complete"

    decision: ExerciseStepDecision = await classifier_llm.generate_structured(
        prompt=_build_step_classifier_prompt(
            state=state,
            exercise_type=exercise_type,
            step_index=step_index,
            current_step=current_step,
        ),
        response_schema=ExerciseStepDecision,
        system_instruction=(
            "You are a strict state classifier for a therapeutic guided "
            "exercise. Do not write user-facing text. Classify only the "
            "latest user reply against the current step."
        ),
    )
    return decision.step_state

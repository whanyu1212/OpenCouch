"""Behavioral activation exercise definitions."""

from __future__ import annotations

from agent.skills.guided_exercises.types import (
    ExerciseDefinition,
    ExerciseStep,
)


# ── Tiny action experiment ───────────────────────────────────────────
# A behavioral activation exercise: identify one small action, plan
# when/where, anticipate obstacles, check feasibility. Sequential.

EXERCISE_TINY_ACTION = "behavioral_activation_tiny_action"

_TINY_ACTION_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's find one small thing you could try — not a plan, "
            "just an experiment. What's something you've been meaning "
            "to do or used to enjoy, even a little? It can be very small."
        ),
        id="choose_action",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names an activity, task, or small action "
            "they might try."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Good. Now let's make it smaller and more specific. When "
            "could you do it today or tomorrow, and where? Just a rough "
            "picture — no pressure to commit yet."
        ),
        id="time_place",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user gives either a time, place, or concrete "
            "setup for the action."
        ),
    ),
    ExerciseStep(
        instruction=(
            "What might get in the way? Not to solve it in advance — "
            "just to notice it, so it doesn't surprise you."
        ),
        id="obstacles",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names at least one possible obstacle, "
            "barrier, or friction point."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Last thing. On a scale from 'no way' to 'I could probably "
            "do that,' how doable does this feel? And is there anything "
            "that would make it even one notch more doable?"
        ),
        id="feasibility",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user gives a feasibility judgment, adjustment, "
            "or way to make the action easier."
        ),
    ),
)


TINY_ACTION_DEFINITION = ExerciseDefinition(
    id=EXERCISE_TINY_ACTION,
    display_name="a tiny action experiment",
    selection_use_case=(
        "behavioral activation when the user feels stuck, depleted, "
        "avoidant, or unable to start"
    ),
    steps=_TINY_ACTION_STEPS,
    category="activation",
    tags=("behavioral_activation", "motivation", "avoidance", "stuck", "small_step"),
    duration_seconds=420,
    intensity="medium",
    selection_aliases=("tiny action", "small action", "motivation"),
    voice_supported=True,
)


DEFINITIONS: tuple[ExerciseDefinition, ...] = (TINY_ACTION_DEFINITION,)

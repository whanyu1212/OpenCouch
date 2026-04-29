"""Behavioral activation exercise definitions."""

from __future__ import annotations

from agent.therapeutic.exercises.types import (
    ExerciseDefinition,
    ExerciseSelectorGroup,
    ExerciseStep,
)


# ── Tiny action experiment ───────────────────────────────────────────
# A behavioral activation exercise: identify one small action, plan
# when/where, anticipate obstacles, check feasibility. Sequential.

EXERCISE_TINY_ACTION = "behavioral_activation_tiny_action"

_TINY_ACTION_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's find one small thing you could try — not a plan, "
            "just an experiment. What's something you've been meaning "
            "to do or used to enjoy, even a little? It can be very small."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Good. Now let's make it smaller and more specific. When "
            "could you do it today or tomorrow, and where? Just a rough "
            "picture — no pressure to commit yet."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "What might get in the way? Not to solve it in advance — "
            "just to notice it, so it doesn't surprise you."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Last thing. On a scale from 'no way' to 'I could probably "
            "do that,' how doable does this feel? And is there anything "
            "that would make it even one notch more doable?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
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
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "stuck",
                "can't start",
                "motivation",
                "depleted",
                "can't do anything",
                "small action",
                "tiny action",
            ),
            priority=80,
        ),
    ),
    selection_aliases=("tiny action", "small action", "motivation"),
    voice_supported=True,
)

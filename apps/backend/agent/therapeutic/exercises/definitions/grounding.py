"""Grounding and body-based exercise definitions."""

from __future__ import annotations

from agent.therapeutic.exercises.types import (
    ExerciseDefinition,
    ExerciseSelectorGroup,
    ExerciseStep,
)


# The exercise_type value for the 5-4-3-2-1 grounding exercise. Kept as
# a module-level constant so the dispatcher and the node's
# internal branching both reference the same string.
EXERCISE_5_4_3_2_1 = "grounding_5_4_3_2_1"


# 5-4-3-2-1 grounding: a sensory exercise that anchors the user in the
# present moment by asking them to identify items across five senses.
# The steps are independent: the order does not strictly matter, but
# the standard order is see -> hear -> feel -> smell -> taste. Each step
# uses a lenient completion threshold so the user can advance without
# matching the requested count perfectly.
_GROUNDING_5_4_3_2_1_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's try a quick grounding exercise called 5-4-3-2-1. "
            "Take a breath. Can you name five things you can see around "
            "you right now? Just describe them — no right or wrong answer."
        ),
        expected_count=5,
        min_count_for_completion=3,
    ),
    ExerciseStep(
        prompt_fallback=(
            "Good. Now four things you can hear. They can be loud or "
            "quiet — the hum of a fridge, traffic, your own breathing."
        ),
        expected_count=4,
        min_count_for_completion=2,
    ),
    ExerciseStep(
        prompt_fallback=(
            "Nice. Three things you can feel — the texture of your "
            "clothes, the floor under your feet, the temperature of "
            "the air."
        ),
        expected_count=3,
        min_count_for_completion=2,
    ),
    ExerciseStep(
        prompt_fallback=(
            "Two things you can smell. If nothing stands out, you can "
            "cup your hands and smell them, or imagine a smell you like."
        ),
        expected_count=2,
        min_count_for_completion=1,
    ),
    ExerciseStep(
        prompt_fallback=(
            "And one thing you can taste — the last thing you ate or "
            "drank, or just the inside of your mouth."
        ),
        expected_count=1,
        min_count_for_completion=1,
    ),
)

# ── Box breathing ─────────────────────────────────────────────────────
# A structured 4-phase breathing cycle. Each step is a single breathing
# action that the user confirms completing. Steps use
# user_confirmation mode.

EXERCISE_BOX_BREATHING = "grounding_box_breathing"

_BOX_BREATHING_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's try box breathing. Breathe in slowly through your "
            "nose for a count of four. Just focus on the air coming in. "
            "Let me know when you've done that."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Good. Now hold that breath gently for another count of "
            "four. No strain — just a soft pause. Tell me when you're "
            "ready."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Now breathe out slowly through your mouth for a count of "
            "four. Let the air go completely. Let me know when you've "
            "exhaled."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "One more hold — empty lungs, count of four. Just sit with "
            "the stillness for a moment. Tell me when you're done."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
)


# ── STOP technique ───────────────────────────────────────────────────
# A DBT-informed distress tolerance skill. Each letter in STOP is a
# discrete step. Steps 0-1 use confirmation; steps 2-3 use item_count
# (the user names an observation or action).

EXERCISE_STOP_TECHNIQUE = "grounding_stop_technique"

_STOP_TECHNIQUE_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's try the STOP technique. S is for Stop — just pause "
            "whatever you're doing right now. Hands in your lap, feet "
            "on the floor. Take a second. Let me know when you've paused."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "T is for Take a breath. One slow, deliberate breath — in "
            "through your nose, out through your mouth. Tell me when "
            "you've taken it."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "O is for Observe. What are you noticing right now — in "
            "your body, your thoughts, or your surroundings? Just name "
            "what's there."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "P is for Proceed. Now that you've paused and noticed, "
            "what feels like the most useful next thing you could do? "
            "Even something small."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


# ── Progressive muscle relaxation ─────────────────────────────────────
# A body-focused relaxation exercise: tense and release 5 muscle groups.
# Each step uses user_confirmation mode.

EXERCISE_MUSCLE_RELAXATION = "grounding_muscle_relaxation"

_MUSCLE_RELAXATION_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's release some tension from your body. Start with your "
            "hands — clench both fists as tight as you can. Hold for "
            "about five seconds, then let go all at once. Notice the "
            "difference. Tell me when you've done that."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Good. Now your shoulders — shrug them up toward your ears, "
            "hold them there for five seconds, then drop them. Let them "
            "fall all the way down. Let me know when you've released."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Now your face — scrunch everything up: squeeze your eyes, "
            "clench your jaw, furrow your brow. Hold it for five "
            "seconds, then release. Let your face go completely slack. "
            "Tell me when you're done."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Your stomach now — pull it in tight, like you're bracing "
            "for something. Hold for five seconds, then let it go soft. "
            "Let me know when you've released."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Last one — your legs and feet. Press your feet into the "
            "floor and tense your thighs. Hold for five seconds, then "
            "release everything. Let your legs go heavy. Tell me when "
            "you're done."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
)


GROUNDING_5_4_3_2_1_DEFINITION = ExerciseDefinition(
    id=EXERCISE_5_4_3_2_1,
    display_name="a grounding moment",
    selection_use_case=(
        "sensory grounding for acute anxiety, panic, dissociation, "
        "or needing to orient to the room"
    ),
    steps=_GROUNDING_5_4_3_2_1_STEPS,
    selection_aliases=("grounding", "ground me", "5-4-3-2-1"),
    fallback_suggestion_rank=10,
    voice_supported=True,
)

BOX_BREATHING_DEFINITION = ExerciseDefinition(
    id=EXERCISE_BOX_BREATHING,
    display_name="a box breathing cycle",
    selection_use_case=(
        "paced breathing for stress, body activation, racing heart, "
        "or needing to slow down"
    ),
    steps=_BOX_BREATHING_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=("breath", "breathe", "breathing", "box breath"),
            priority=10,
        ),
    ),
    selection_aliases=("breathing", "box breathing", "breath"),
    fallback_suggestion_rank=20,
    voice_supported=True,
)

STOP_TECHNIQUE_DEFINITION = ExerciseDefinition(
    id=EXERCISE_STOP_TECHNIQUE,
    display_name="the STOP technique",
    selection_use_case=(
        "a quick pause for urges, impulsive reactions, or needing to stop "
        "and choose the next action"
    ),
    steps=_STOP_TECHNIQUE_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=("stop technique", "stop method", "s.t.o.p"),
            priority=30,
        ),
        ExerciseSelectorGroup(
            keywords=("stop", "pause", "slow down"),
            priority=120,
        ),
    ),
    selection_aliases=("STOP technique", "S.T.O.P.", "pause technique"),
    voice_supported=True,
)

MUSCLE_RELAXATION_DEFINITION = ExerciseDefinition(
    id=EXERCISE_MUSCLE_RELAXATION,
    display_name="a muscle relaxation exercise",
    selection_use_case=(
        "body tension, muscle tightness, restlessness, or wanting to relax physically"
    ),
    steps=_MUSCLE_RELAXATION_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "muscle",
                "tense",
                "tension",
                "relax my body",
                "pmr",
                "progressive",
            ),
            priority=20,
        ),
    ),
    selection_aliases=(
        "muscle relaxation",
        "progressive muscle relaxation",
        "PMR",
    ),
    voice_supported=True,
)

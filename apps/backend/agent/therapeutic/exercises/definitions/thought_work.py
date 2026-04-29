"""Thought-work exercise definitions."""

from __future__ import annotations

from agent.therapeutic.exercises.types import (
    ExerciseDefinition,
    ExerciseSelectorGroup,
    ExerciseStep,
)


# ── Simple thought record ────────────────────────────────────────────
# A simplified 4-step CBT thought record: situation → thought →
# evidence → alternative. Sequential — each step depends on prior
# steps. Exit is the only valid off-ramp (no skip).

EXERCISE_THOUGHT_RECORD = "thought_work_simple_record"

_THOUGHT_RECORD_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's slow down and look at one thought that's been "
            "pulling at you. Can you describe the situation — what was "
            "happening when this thought showed up?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Got it. Now, what's the specific thought or belief that "
            "came with that moment? Try to put it in one sentence if "
            "you can — the exact words your mind was saying."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Okay. Now let's look at that thought from the outside for "
            "a moment. What evidence do you have that it might not be "
            "the full picture? Even small things count."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Last step. Given what you just noticed, is there a more "
            "balanced way to hold that thought? Not a fake positive — "
            "just something that accounts for the full picture."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


# ── Behavioral experiment ────────────────────────────────────────────
# A CBT exercise for testing beliefs in the real world. Sequential —
# each step depends on prior. The gap between step 2 and 3 may span
# hours or days (the user does something IRL).

EXERCISE_BEHAVIORAL_EXPERIMENT = "thought_work_behavioral_experiment"

_BEHAVIORAL_EXPERIMENT_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's test a belief. What's a thought or prediction you "
            "keep making that causes you distress? Try to state it as "
            "clearly as you can — something like 'If I speak up in the "
            "meeting, people will think I'm stupid.'"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Got it. Now — what's a small, manageable way you could "
            "test whether that's actually true? Not a huge leap, just "
            "something that would give you real information. What could "
            "you try?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Before you try it, let's write down your prediction. What "
            "exactly do you think will happen? Be specific — what will "
            "people do, how will you feel, what's the worst-case scenario "
            "you're expecting?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Now — what actually happened? Or if you haven't tried it "
            "yet, come back when you have. How did the reality compare "
            "to what you predicted? What surprised you?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


# ── Gratitude inventory ──────────────────────────────────────────────
# A short positive-psychology exercise for building positive affect.
# 3 steps, item_count mode. Good session closer.

EXERCISE_CONTINUUM = "thought_work_continuum"

# The continuum technique targets rigid all-or-nothing beliefs by
# converting an absolute label into a 0-100 dimension, then placing
# the user on it honestly. Most users discover they're mid-range, not
# at zero — which is already a shift from the absolute framing.
_CONTINUUM_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's look at that belief more closely. Can you state it "
            "as an absolute — the all-or-nothing version? Something like "
            "'I'm a terrible [X]' or 'I always [Y].'"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Now let's turn that into a scale. If we put that quality on "
            "a 0-to-100 spectrum — what would a 0 look like? The absolute "
            "worst-case version, someone who truly has none of that quality?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "And what would 100 look like? The impossibly perfect version "
            "— which nobody actually is?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Where would you honestly place yourself on that scale right "
            "now? Just a number — there's no wrong answer."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "That's not zero. What's one small thing that would move you "
            "about 5 points up from where you are? Something concrete and "
            "doable this week."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


THOUGHT_RECORD_DEFINITION = ExerciseDefinition(
    id=EXERCISE_THOUGHT_RECORD,
    display_name="a thought record",
    selection_use_case=(
        "examining a distressing thought, belief, assumption, or "
        "self-critical interpretation"
    ),
    steps=_THOUGHT_RECORD_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "thought record",
                "thought check",
                "examine.*thought",
                "look at.*thought",
                "belief",
            ),
            priority=70,
        ),
    ),
    selection_aliases=("thought record", "thought check", "belief"),
)

BEHAVIORAL_EXPERIMENT_DEFINITION = ExerciseDefinition(
    id=EXERCISE_BEHAVIORAL_EXPERIMENT,
    display_name="a behavioral experiment",
    selection_use_case=(
        "testing a fear, prediction, or belief with a small real-world experiment"
    ),
    steps=_BEHAVIORAL_EXPERIMENT_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "behavioral experiment",
                "test this belief",
                "is this.*true",
                "prove it",
                "check if",
            ),
            priority=50,
        ),
    ),
    selection_aliases=("behavioral experiment", "test this belief", "test a belief"),
)

CONTINUUM_DEFINITION = ExerciseDefinition(
    id=EXERCISE_CONTINUUM,
    display_name="a continuum exercise",
    selection_use_case=(
        "softening all-or-nothing labels like total failure, terrible person, "
        "or 100 percent bad"
    ),
    steps=_CONTINUUM_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "continuum",
                "all.or.nothing",
                "black.and.white",
                r"i'?m (?:a )?(?:terrible|horrible|worst|complete|total)\b",
                r"i (?:always|never) (?:fail|mess|ruin|screw|disappoint|let)",
                r"100\s*%",
            ),
            priority=60,
        ),
    ),
    selection_aliases=("continuum", "all-or-nothing", "black-and-white"),
    voice_supported=True,
)

"""Thought-work exercise definitions."""

from __future__ import annotations

from agent.skills.guided_exercises.types import (
    ExerciseDefinition,
    ExerciseStep,
)


# ── Simple thought record ────────────────────────────────────────────
# A simplified 4-step CBT thought record: situation → thought →
# evidence → alternative. Sequential — each step depends on prior
# steps. Exit is the only valid off-ramp (no skip).

EXERCISE_THOUGHT_RECORD = "thought_work_simple_record"

_THOUGHT_RECORD_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's slow down and look at one thought that's been "
            "pulling at you. Can you describe the situation — what was "
            "happening when this thought showed up?"
        ),
        id="situation",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user describes a concrete situation, moment, "
            "or context where the thought showed up."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Got it. Now, what's the specific thought or belief that "
            "came with that moment? Try to put it in one sentence if "
            "you can — the exact words your mind was saying."
        ),
        id="thought",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a specific thought, belief, "
            "prediction, assumption, or self-judgment."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Okay. Now let's look at that thought from the outside for "
            "a moment. What evidence do you have that it might not be "
            "the full picture? Even small things count."
        ),
        id="evidence_against",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user offers an exception, missing information, "
            "alternative interpretation, or evidence that softens the thought."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Last step. Given what you just noticed, is there a more "
            "balanced way to hold that thought? Not a fake positive — "
            "just something that accounts for the full picture."
        ),
        id="balanced_thought",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user drafts a more balanced or less absolute "
            "version of the thought, even if it is imperfect."
        ),
    ),
)


# ── Behavioral experiment ────────────────────────────────────────────
# A CBT exercise for testing beliefs in the real world. Sequential —
# each step depends on prior. The gap between step 2 and 3 may span
# hours or days (the user does something IRL).

EXERCISE_BEHAVIORAL_EXPERIMENT = "thought_work_behavioral_experiment"

_BEHAVIORAL_EXPERIMENT_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's test a belief. What's a thought or prediction you "
            "keep making that causes you distress? Try to state it as "
            "clearly as you can — something like 'If I speak up in the "
            "meeting, people will think I'm stupid.'"
        ),
        id="belief",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user states a belief, fear, or prediction "
            "that could be tested."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Got it. Now — what's a small, manageable way you could "
            "test whether that's actually true? Not a huge leap, just "
            "something that would give you real information. What could "
            "you try?"
        ),
        id="experiment",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a small real-world action or "
            "experiment that could test the belief."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Before you try it, let's write down your prediction. What "
            "exactly do you think will happen? Be specific — what will "
            "people do, how will you feel, what's the worst-case scenario "
            "you're expecting?"
        ),
        id="prediction",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user records a specific expected outcome, "
            "reaction, feeling, or worst-case prediction."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Now — what actually happened? Or if you haven't tried it "
            "yet, come back when you have. How did the reality compare "
            "to what you predicted? What surprised you?"
        ),
        id="outcome",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user compares what happened with what they "
            "predicted, or clearly says they have not tried it yet."
        ),
    ),
)


# ── Gratitude inventory ──────────────────────────────────────────────
# A short positive-psychology exercise for building positive affect.
# Good for softening rigid self-labels.

EXERCISE_CONTINUUM = "thought_work_continuum"

# The continuum technique targets rigid all-or-nothing beliefs by
# converting an absolute label into a 0-100 dimension, then placing
# the user on it honestly. Most users discover they're mid-range, not
# at zero — which is already a shift from the absolute framing.
_CONTINUUM_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's look at that belief more closely. Can you state it "
            "as an absolute — the all-or-nothing version? Something like "
            "'I'm a terrible [X]' or 'I always [Y].'"
        ),
        id="absolute_belief",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names an absolute, all-or-nothing belief or label."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Now let's turn that into a scale. If we put that quality on "
            "a 0-to-100 spectrum — what would a 0 look like? The absolute "
            "worst-case version, someone who truly has none of that quality?"
        ),
        id="low_anchor",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user describes the low end of the continuum."
        ),
    ),
    ExerciseStep(
        instruction=(
            "And what would 100 look like? The impossibly perfect version "
            "— which nobody actually is?"
        ),
        id="high_anchor",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user describes the high end of the continuum."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Where would you honestly place yourself on that scale right "
            "now? Just a number — there's no wrong answer."
        ),
        id="self_rating",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user gives a number or approximate position "
            "on the scale."
        ),
    ),
    ExerciseStep(
        instruction=(
            "That's not zero. What's one small thing that would move you "
            "about 5 points up from where you are? Something concrete and "
            "doable this week."
        ),
        id="small_shift",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a small concrete action that could "
            "move them a little on the scale."
        ),
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
    category="thought_work",
    tags=("cbt", "thought_record", "belief", "self_criticism", "reframe"),
    duration_seconds=600,
    intensity="high",
    selection_aliases=("thought record", "thought check", "belief"),
)

BEHAVIORAL_EXPERIMENT_DEFINITION = ExerciseDefinition(
    id=EXERCISE_BEHAVIORAL_EXPERIMENT,
    display_name="a behavioral experiment",
    selection_use_case=(
        "testing a fear, prediction, or belief with a small real-world experiment"
    ),
    steps=_BEHAVIORAL_EXPERIMENT_STEPS,
    category="thought_work",
    tags=("cbt", "behavioral_experiment", "prediction", "fear", "testing_beliefs"),
    duration_seconds=600,
    intensity="high",
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
    category="thought_work",
    tags=("cbt", "continuum", "all_or_nothing", "black_and_white", "labels"),
    duration_seconds=480,
    intensity="medium",
    selection_aliases=("continuum", "all-or-nothing", "black-and-white"),
    voice_supported=True,
)


DEFINITIONS: tuple[ExerciseDefinition, ...] = (
    THOUGHT_RECORD_DEFINITION,
    BEHAVIORAL_EXPERIMENT_DEFINITION,
    CONTINUUM_DEFINITION,
)

"""ACT defusion and values exercise definitions."""

from __future__ import annotations

from agent.therapeutic.exercises.types import (
    ExerciseDefinition,
    ExerciseSelectorGroup,
    ExerciseStep,
)


# ── Leaves on a stream ───────────────────────────────────────────────
# An ACT defusion exercise. The user names a sticky thought, places it
# on an imagined leaf, watches it float away, notices what remains,
# then identifies a values-aligned step. Mixed completion modes.

EXERCISE_LEAVES_ON_STREAM = "defusion_leaves_on_stream"

_LEAVES_ON_STREAM_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's try something different with the thought that keeps "
            "showing up. First — can you tell me the thought, in the "
            "exact words your mind uses? Not the story around it, just "
            "the sentence."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Now imagine a slow stream in front of you, with leaves "
            "floating on the surface. Take that thought and place it "
            "on one of the leaves. Watch it sit there for a moment. "
            "Let me know when you can picture it."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Good. Now let the leaf drift downstream. You don't have "
            "to push it — just let the current take it. If your mind "
            "pulls you back to the thought, that's fine — just notice "
            "that, and gently put the new thought on another leaf. "
            "Tell me when you've watched it drift for a moment."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "What do you notice right now? Not whether the thought is "
            "gone — it probably isn't — but what's the feeling in your "
            "body or the space in your mind like, compared to a few "
            "minutes ago?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "The thought might float back. That's normal — thoughts do "
            "that. But now you know you can set it down without having "
            "to argue with it or fix it first. What's one small thing "
            "you could do next that matters to you, even with this "
            "thought still in the background?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


# ── Values compass ───────────────────────────────────────────────────
# ACT values clarification. Helps users identify what matters and take
# one step toward it. Complements defusion exercises (letting go) with
# direction (moving toward).

EXERCISE_VALUES_COMPASS = "defusion_values_compass"

_VALUES_COMPASS_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's check your values compass. Think about the big areas "
            "of your life — relationships, work, health, personal growth, "
            "fun. Which area feels most important to you right now, or "
            "most neglected?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Why does that area matter to you? Not what you 'should' "
            "care about — what genuinely pulls at you when you're honest "
            "with yourself?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "On a scale of 1 to 10, how aligned do you feel your "
            "current actions are with what you just described? 1 is "
            "'completely off track,' 10 is 'living it fully.' Just a "
            "gut feeling — no wrong answer."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "What's one small thing you could do this week — even "
            "today — that would move that number up by one? Not a "
            "big overhaul, just one step closer."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


LEAVES_ON_STREAM_DEFINITION = ExerciseDefinition(
    id=EXERCISE_LEAVES_ON_STREAM,
    display_name="a defusion exercise",
    selection_use_case=(
        "defusion or acceptance when the user is caught in thoughts and "
        "wants distance from them"
    ),
    steps=_LEAVES_ON_STREAM_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "accept",
                "let go",
                "defusion",
                "leaves",
                "step back from",
                "stop fighting",
                "fused",
                "fusion",
            ),
            priority=110,
        ),
    ),
)

VALUES_COMPASS_DEFINITION = ExerciseDefinition(
    id=EXERCISE_VALUES_COMPASS,
    display_name="a values compass exercise",
    selection_use_case="values, meaning, direction, purpose, or deciding what matters next",
    steps=_VALUES_COMPASS_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "values",
                "what matters",
                "meaning",
                "purpose",
                "direction",
                "compass",
                "life direction",
            ),
            priority=100,
        ),
    ),
    voice_supported=True,
)

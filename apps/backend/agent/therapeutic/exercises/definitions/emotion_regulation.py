"""Emotion regulation and self-compassion exercise definitions."""

from __future__ import annotations

from agent.therapeutic.exercises.types import (
    ExerciseDefinition,
    ExerciseSelectorGroup,
    ExerciseStep,
)


# ── Self-compassion break ────────────────────────────────────────────
# Kristin Neff's 3-component model: mindfulness of suffering, common
# humanity, self-kindness. Very short (3 steps), confirmation mode.

EXERCISE_SELF_COMPASSION = "self_compassion_break"

_SELF_COMPASSION_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's take a moment to be gentle with yourself. First — "
            "just acknowledge what you're going through. Say it simply, "
            "like 'This is really hard right now' or 'I'm struggling.' "
            "You can say it out loud or just in your mind. Let me know "
            "when you've done that."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Good. Now remind yourself that you're not alone in this. "
            "Other people feel this way too — it's part of being human, "
            "not a sign that something is wrong with you. Try saying "
            "something like 'Everyone struggles sometimes.' Let me know "
            "when you've sat with that for a moment."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Last step. Place a hand on your chest if that feels "
            "comfortable, and say something kind to yourself — the way "
            "you'd talk to a friend who was hurting. Something like "
            "'May I be kind to myself' or 'May I give myself what I "
            "need.' What feels right to you?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


# ── IMPROVE the moment ───────────────────────────────────────────────
# DBT distress tolerance skill. We use 4 of the 7 letters:
# Imagery, Meaning, One thing, Encouragement. Mixed completion modes.

EXERCISE_IMPROVE = "emotion_regulation_improve"

_IMPROVE_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's IMPROVE this moment. I is for Imagery — close your "
            "eyes if you can, and picture a place where you feel safe "
            "or calm. It can be real or imagined. Spend a few seconds "
            "there. Let me know when you have it."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "M is for Meaning. Even in difficult moments, there's "
            "sometimes something to be learned or a reason to keep "
            "going. Can you name one thing — even small — that makes "
            "this struggle worth enduring?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "O is for One thing in the moment. Instead of thinking "
            "about everything at once, focus on just one thing you can "
            "do right now. What's one manageable task or focus point "
            "for the next few minutes? Let me know when you've picked one."
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="user_confirmation",
    ),
    ExerciseStep(
        prompt_fallback=(
            "E is for Encouragement. Say something supportive to "
            "yourself — not toxic positivity, just honest encouragement. "
            "Something like 'I've gotten through hard things before' or "
            "'I'm doing the best I can right now.' What feels true?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


EXERCISE_GRATITUDE = "emotion_regulation_gratitude"

_GRATITUDE_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        prompt_fallback=(
            "Let's shift gears for a moment. Can you name three things "
            "you're grateful for today? They can be big or small — a "
            "good cup of coffee counts as much as a good friend."
        ),
        expected_count=3,
        min_count_for_completion=2,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Pick the one that resonates most. Why does it matter to "
            "you? Not just 'it's nice' — what does it give you or mean "
            "to you?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
    ExerciseStep(
        prompt_fallback=(
            "Take a moment and notice what's happening in your body "
            "right now, after focusing on that. Does anything feel "
            "different compared to a few minutes ago — even slightly?"
        ),
        expected_count=1,
        min_count_for_completion=1,
        completion_mode="item_count",
    ),
)


SELF_COMPASSION_DEFINITION = ExerciseDefinition(
    id=EXERCISE_SELF_COMPASSION,
    display_name="a self-compassion break",
    selection_use_case=(
        "self-criticism, shame, harsh self-talk, or wanting to be kinder to oneself"
    ),
    steps=_SELF_COMPASSION_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "self.?compassion",
                "kinder to myself",
                "hard on myself",
                "self.?critical",
                "hate myself",
                "compassion break",
            ),
            priority=90,
        ),
    ),
    selection_aliases=("self-compassion", "compassion break", "kinder to myself"),
    fallback_suggestion_rank=30,
    voice_supported=True,
)

IMPROVE_DEFINITION = ExerciseDefinition(
    id=EXERCISE_IMPROVE,
    display_name="an IMPROVE the moment exercise",
    selection_use_case="distress tolerance for getting through an overwhelming moment",
    steps=_IMPROVE_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "improve the moment",
                "improve",
                "cope",
                "get through this",
                "emotion regulation",
            ),
            priority=40,
        ),
        ExerciseSelectorGroup(
            keywords=("overwhelmed", "too much"),
            priority=130,
        ),
    ),
    selection_aliases=("IMPROVE", "IMPROVE the moment", "emotion regulation"),
    voice_supported=True,
)

GRATITUDE_DEFINITION = ExerciseDefinition(
    id=EXERCISE_GRATITUDE,
    display_name="a gratitude inventory",
    selection_use_case=(
        "noticing positive moments, appreciation, or broadening attention "
        "beyond distress"
    ),
    steps=_GRATITUDE_STEPS,
    selector_groups=(
        ExerciseSelectorGroup(
            keywords=(
                "grateful",
                "gratitude",
                "thankful",
                "something good",
                "positive",
                "appreciate",
            ),
            priority=115,
        ),
    ),
    selection_aliases=("gratitude", "gratitude exercise", "grateful"),
    voice_supported=True,
)

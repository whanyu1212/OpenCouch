"""Emotion regulation and self-compassion exercise definitions."""

from __future__ import annotations

from agent.skills.guided_exercises.types import (
    ExerciseDefinition,
    ExerciseStep,
)


# ── Self-compassion break ────────────────────────────────────────────
# Kristin Neff's 3-component model: mindfulness of suffering, common
# humanity, self-kindness. Very short (3 steps), mixed modes.

EXERCISE_SELF_COMPASSION = "self_compassion_break"

_SELF_COMPASSION_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's take a moment to be gentle with yourself. First — "
            "just acknowledge what you're going through. Say it simply, "
            "like 'This is really hard right now' or 'I'm struggling.' "
            "You can say it out loud or just in your mind. Let me know "
            "when you've done that."
        ),
        id="acknowledge_suffering",
        completion_mode="confirmation",
    ),
    ExerciseStep(
        instruction=(
            "Good. Now remind yourself that you're not alone in this. "
            "Other people feel this way too — it's part of being human, "
            "not a sign that something is wrong with you. Try saying "
            "something like 'Everyone struggles sometimes.' Let me know "
            "when you've sat with that for a moment."
        ),
        id="common_humanity",
        completion_mode="confirmation",
    ),
    ExerciseStep(
        instruction=(
            "Last step. Place a hand on your chest if that feels "
            "comfortable, and say something kind to yourself — the way "
            "you'd talk to a friend who was hurting. Something like "
            "'May I be kind to myself' or 'May I give myself what I "
            "need.' What feels right to you?"
        ),
        id="kind_phrase",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user offers a kind phrase, wish, or supportive "
            "self-directed sentence."
        ),
    ),
)


# ── IMPROVE the moment ───────────────────────────────────────────────
# DBT distress tolerance skill. We use 4 of the 7 letters:
# Imagery, Meaning, One thing, Encouragement. Mixed completion modes.

EXERCISE_IMPROVE = "emotion_regulation_improve"

_IMPROVE_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's IMPROVE this moment. I is for Imagery — close your "
            "eyes if you can, and picture a place where you feel safe "
            "or calm. It can be real or imagined. Spend a few seconds "
            "there. Let me know when you have it."
        ),
        id="imagery",
        completion_mode="confirmation",
    ),
    ExerciseStep(
        instruction=(
            "M is for Meaning. Even in difficult moments, there's "
            "sometimes something to be learned or a reason to keep "
            "going. Can you name one thing — even small — that makes "
            "this struggle worth enduring?"
        ),
        id="meaning",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a reason, value, lesson, or "
            "meaning that helps them endure the moment."
        ),
    ),
    ExerciseStep(
        instruction=(
            "O is for One thing in the moment. Instead of thinking "
            "about everything at once, focus on just one thing you can "
            "do right now. What's one manageable task or focus point "
            "for the next few minutes? Let me know when you've picked one."
        ),
        id="one_thing",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names one manageable task, focus point, "
            "or next action for the immediate moment."
        ),
    ),
    ExerciseStep(
        instruction=(
            "E is for Encouragement. Say something supportive to "
            "yourself — not toxic positivity, just honest encouragement. "
            "Something like 'I've gotten through hard things before' or "
            "'I'm doing the best I can right now.' What feels true?"
        ),
        id="encouragement",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user offers an honest encouraging statement "
            "or supportive self-talk phrase."
        ),
    ),
)


EXERCISE_GRATITUDE = "emotion_regulation_gratitude"

_GRATITUDE_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's shift gears for a moment. Can you name three things "
            "you're grateful for today? They can be big or small — a "
            "good cup of coffee counts as much as a good friend."
        ),
        id="name_items",
        completion_mode="items",
        target_items=3,
        min_items=2,
    ),
    ExerciseStep(
        instruction=(
            "Pick the one that resonates most. Why does it matter to "
            "you? Not just 'it's nice' — what does it give you or mean "
            "to you?"
        ),
        id="why_matters",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user explains why one gratitude item matters "
            "or what it gives them."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Take a moment and notice what's happening in your body "
            "right now, after focusing on that. Does anything feel "
            "different compared to a few minutes ago — even slightly?"
        ),
        id="body_check",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user describes a body sensation, emotional "
            "shift, or says there is no noticeable change."
        ),
    ),
)


SELF_COMPASSION_DEFINITION = ExerciseDefinition(
    id=EXERCISE_SELF_COMPASSION,
    display_name="a self-compassion break",
    selection_use_case=(
        "self-criticism, shame, harsh self-talk, or wanting to be kinder to oneself"
    ),
    steps=_SELF_COMPASSION_STEPS,
    category="emotion_regulation",
    tags=("self_compassion", "shame", "self_criticism", "kindness", "neff"),
    duration_seconds=240,
    intensity="medium",
    selection_aliases=("self-compassion", "compassion break", "kinder to myself"),
    voice_supported=True,
    text_fit="good",
    voice_fit="good",
    interaction_pattern="reflection",
    cognitive_load="medium",
)

IMPROVE_DEFINITION = ExerciseDefinition(
    id=EXERCISE_IMPROVE,
    display_name="an IMPROVE the moment exercise",
    selection_use_case="distress tolerance for getting through an overwhelming moment",
    steps=_IMPROVE_STEPS,
    category="emotion_regulation",
    tags=("dbt", "distress_tolerance", "overwhelmed", "emotion_regulation", "cope"),
    duration_seconds=420,
    intensity="medium",
    selection_aliases=("IMPROVE", "IMPROVE the moment", "emotion regulation"),
    voice_supported=True,
    text_fit="good",
    voice_fit="good",
    interaction_pattern="reflection",
    cognitive_load="medium",
)

GRATITUDE_DEFINITION = ExerciseDefinition(
    id=EXERCISE_GRATITUDE,
    display_name="a gratitude inventory",
    selection_use_case=(
        "noticing positive moments, appreciation, or broadening attention "
        "beyond distress"
    ),
    steps=_GRATITUDE_STEPS,
    category="emotion_regulation",
    tags=("gratitude", "positive_affect", "appreciation", "attention_shift"),
    duration_seconds=300,
    intensity="low",
    selection_aliases=("gratitude", "gratitude exercise", "grateful"),
    voice_supported=True,
    text_fit="good",
    voice_fit="good",
    interaction_pattern="item_collection",
    cognitive_load="low",
)


DEFINITIONS: tuple[ExerciseDefinition, ...] = (
    SELF_COMPASSION_DEFINITION,
    IMPROVE_DEFINITION,
    GRATITUDE_DEFINITION,
)

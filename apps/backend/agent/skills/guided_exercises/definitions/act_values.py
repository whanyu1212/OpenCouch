"""ACT defusion and values exercise definitions."""

from __future__ import annotations

from agent.skills.guided_exercises.types import (
    ExerciseDefinition,
    ExerciseStep,
)


# ── Leaves on a stream ───────────────────────────────────────────────
# An ACT defusion exercise. The user names a sticky thought, places it
# on an imagined leaf, watches it float away, notices what remains,
# then identifies a values-aligned step. Mixed completion modes.

EXERCISE_LEAVES_ON_STREAM = "defusion_leaves_on_stream"

_LEAVES_ON_STREAM_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's try something different with the thought that keeps "
            "showing up. First — can you tell me the thought, in the "
            "exact words your mind uses? Not the story around it, just "
            "the sentence."
        ),
        id="name_thought",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names the sticky thought or sentence "
            "their mind is repeating."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Now imagine a slow stream in front of you, with leaves "
            "floating on the surface. Take that thought and place it "
            "on one of the leaves. Watch it sit there for a moment. "
            "Let me know when you can picture it."
        ),
        id="place_on_leaf",
        completion_mode="confirmation",
    ),
    ExerciseStep(
        instruction=(
            "Good. Now let the leaf drift downstream. You don't have "
            "to push it — just let the current take it. If your mind "
            "pulls you back to the thought, that's fine — just notice "
            "that, and gently put the new thought on another leaf. "
            "Tell me when you've watched it drift for a moment."
        ),
        id="watch_drift",
        completion_mode="confirmation",
    ),
    ExerciseStep(
        instruction=(
            "What do you notice right now? Not whether the thought is "
            "gone — it probably isn't — but what's the feeling in your "
            "body or the space in your mind like, compared to a few "
            "minutes ago?"
        ),
        id="notice_shift",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user describes a body sensation, mental space, "
            "emotional shift, or lack of change."
        ),
    ),
    ExerciseStep(
        instruction=(
            "The thought might float back. That's normal — thoughts do "
            "that. But now you know you can set it down without having "
            "to argue with it or fix it first. What's one small thing "
            "you could do next that matters to you, even with this "
            "thought still in the background?"
        ),
        id="values_step",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a values-aligned next step, even "
            "if the thought is still present."
        ),
    ),
)


# ── Values compass ───────────────────────────────────────────────────
# ACT values clarification. Helps users identify what matters and take
# one step toward it. Complements defusion exercises (letting go) with
# direction (moving toward).

EXERCISE_VALUES_COMPASS = "defusion_values_compass"

_VALUES_COMPASS_STEPS: tuple[ExerciseStep, ...] = (
    ExerciseStep(
        instruction=(
            "Let's check your values compass. Think about the big areas "
            "of your life — relationships, work, health, personal growth, "
            "fun. Which area feels most important to you right now, or "
            "most neglected?"
        ),
        id="life_area",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a life area, value area, or area "
            "that feels important or neglected."
        ),
    ),
    ExerciseStep(
        instruction=(
            "Why does that area matter to you? Not what you 'should' "
            "care about — what genuinely pulls at you when you're honest "
            "with yourself?"
        ),
        id="why_matters",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user gives a reason, value, longing, or "
            "personal meaning behind that area."
        ),
    ),
    ExerciseStep(
        instruction=(
            "On a scale of 1 to 10, how aligned do you feel your "
            "current actions are with what you just described? 1 is "
            "'completely off track,' 10 is 'living it fully.' Just a "
            "gut feeling — no wrong answer."
        ),
        id="alignment_rating",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user gives a number or approximate alignment rating."
        ),
    ),
    ExerciseStep(
        instruction=(
            "What's one small thing you could do this week — even "
            "today — that would move that number up by one? Not a "
            "big overhaul, just one step closer."
        ),
        id="one_step",
        completion_mode="llm_judged",
        completion_criteria=(
            "Complete when the user names a small action that would move "
            "them closer to the value."
        ),
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
    category="act_values",
    tags=("act", "defusion", "acceptance", "sticky_thoughts", "letting_go"),
    duration_seconds=480,
    intensity="medium",
    selection_aliases=("defusion", "leaves exercise", "let go"),
    text_fit="okay",
    voice_fit="poor",
    interaction_pattern="imagery",
    cognitive_load="medium",
)

VALUES_COMPASS_DEFINITION = ExerciseDefinition(
    id=EXERCISE_VALUES_COMPASS,
    display_name="a values compass exercise",
    selection_use_case="values, meaning, direction, purpose, or deciding what matters next",
    steps=_VALUES_COMPASS_STEPS,
    category="act_values",
    tags=("act", "values", "meaning", "purpose", "direction"),
    duration_seconds=420,
    intensity="medium",
    selection_aliases=("values", "values compass", "what matters"),
    voice_supported=True,
    text_fit="good",
    voice_fit="okay",
    interaction_pattern="planning",
    cognitive_load="medium",
)


DEFINITIONS: tuple[ExerciseDefinition, ...] = (
    LEAVES_ON_STREAM_DEFINITION,
    VALUES_COMPASS_DEFINITION,
)

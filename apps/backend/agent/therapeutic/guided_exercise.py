"""Guided exercise response mode — multi-turn structured walkthroughs.

Guided exercise is the FIRST multi-turn response mode in the
codebase. Unlike supportive/reflective/clarifying/psychoeducation/
closing which all return a single response delta per turn and leave
no state behind, guided exercise tracks which exercise is running
and which step the user is on across multiple turns.

Architecture overview:

1. **State lives on ``progress``.** Two new fields on
   :class:`SessionProgressState`:
   - ``exercise_type`` — the exercise identifier, e.g.,
     "grounding_5_4_3_2_1"
   - ``exercise_step`` — the current 0-indexed step number

   When an exercise is not running, both fields are either ``None``
   or absent from the progress dict.

2. **Dispatcher fast-path.** The therapeutic dispatcher checks
   ``progress.exercise_type`` on every turn. If an exercise is
   active, it short-circuits to ``guided_exercise_response_node``
   without running the LLM classifier — otherwise the LLM classifier
   would re-route the user's step response ("I see my lamp") to
   supportive or clarifying, breaking the multi-turn flow. See
   ``agent/therapeutic/dispatcher.py`` for the fast-path rule.

3. **Deterministic step-state classifier.** The node inspects the
   user's message and classifies it as ``complete`` / ``hold`` /
   ``stuck`` / ``exit``. This is done with regex + simple heuristics
   rather than an inner LLM call, deliberately — the state machine
   should be explicit and testable, and when it gets things wrong
   during dogfood the fix is obvious (tweak the regex) rather than
   "tune the sub-prompt and hope."

4. **LLM generates response text; node owns state.** The node
   decides which step to advance to (via the classifier + the
   exercise registry), then calls the LLM to generate the
   response prose for that step. The LLM never decides the state
   transition — that's the node's job. This keeps the state machine
   auditable and the LLM call single-purpose (write the response,
   don't classify).

5. **One exercise in v0.6 Stage C.** The registry currently
   supports 5-4-3-2-1 grounding as the only exercise. Adding more
   exercises (box breathing, thought records, acceptance/defusion)
   is a future-stage concern; the registry is designed to grow
   without schema changes.

Design decisions and non-decisions:

- **Non-decision**: How to resume an exercise after CLI restart.
  The v0.8 SQLite checkpointer persists ``state["progress"]``
  across runtime restarts automatically, so resume works "for
  free" via existing infrastructure. This mode doesn't need to
  do anything special.
- **Decision**: Exit is ALWAYS cleaner than skip-to-next-step.
  Skipping mid-exercise only makes sense for exercises whose steps
  are independent (like 5-4-3-2-1, where the five senses are
  parallel), not for sequential exercises (thought records). The
  current implementation uses exit as the universal off-ramp; a
  future "skip" path can be added per-exercise later.
- **Decision**: Patience is a feature. Single turns of tentative
  engagement ("um, a plant?") trigger HOLD, not an escalation
  ladder step. Three+ turns of tentative engagement or an explicit
  "I can't" triggers escalation. This is encoded in the knowledge
  file and enforced in the state classifier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.models import EntityRef, SemanticFact
from agent.models import ModeType, ResponseKind
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id
from agent.therapeutic.prompts import (
    build_guided_exercise_system_prompt,
    build_therapeutic_response_prompt,
)

logger = logging.getLogger(__name__)

# The exercise_type value for the 5-4-3-2-1 grounding exercise. Kept as
# a module-level constant so the dispatcher fast-path and the node's
# internal branching both reference the same string.
EXERCISE_5_4_3_2_1 = "grounding_5_4_3_2_1"


StepState = Literal["complete", "hold", "stuck", "exit"]


CompletionMode = Literal["item_count", "user_confirmation"]


@dataclass(frozen=True)
class ExerciseStep:
    """One step of a multi-turn exercise.

    Each step has:

    - ``prompt_fallback``: the deterministic response text used when
      no LLM client is available. The LLM path uses the same prompt
      but can vary wording turn-to-turn.
    - ``expected_count``: for counting-based steps (e.g., "name 5
      things you can see"), the number of items the user should name
      to count as "complete." The state classifier uses this to
      distinguish COMPLETE ("I see a lamp, a book, a plant, my
      coffee, and the window") from HOLD ("I see... a lamp?").
    - ``min_count_for_completion``: the minimum number of items that
      still counts as complete. Leniency matters — a user naming 4
      things on a "name 5" step should be allowed to advance rather
      than being held back on a technicality.
    - ``completion_mode``: how the classifier determines completion.
      ``"item_count"`` (default) counts listed items; used for steps
      that ask the user to name things. ``"user_confirmation"`` matches
      confirmation phrases ("ok", "done", "yes"); used for steps where
      the user performs an action (breathing, visualization) and
      confirms they did it.
    """

    prompt_fallback: str
    expected_count: int
    min_count_for_completion: int
    completion_mode: CompletionMode = "item_count"


# 5-4-3-2-1 grounding: a sensory exercise that anchors the user in the
# present moment by asking them to identify items across five senses.
# The steps are INDEPENDENT — the order doesn't strictly matter, but
# the standard order is see → hear → feel → smell → taste.
#
# For Stage C, we use the standard 5-4-3-2-1 count pattern. Each step
# has a min_count_for_completion that's deliberately less strict than
# the expected_count — if a user names 4 things when asked for 5, they
# still advance.
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


# Registry mapping exercise_type → step sequence. Adding a new exercise
# means adding an entry here plus an ExerciseStep tuple; the dispatcher
# and node code don't need to change.
_EXERCISE_REGISTRY: dict[str, tuple[ExerciseStep, ...]] = {
    EXERCISE_5_4_3_2_1: _GROUNDING_5_4_3_2_1_STEPS,
    EXERCISE_BOX_BREATHING: _BOX_BREATHING_STEPS,
    EXERCISE_STOP_TECHNIQUE: _STOP_TECHNIQUE_STEPS,
    EXERCISE_THOUGHT_RECORD: _THOUGHT_RECORD_STEPS,
    EXERCISE_TINY_ACTION: _TINY_ACTION_STEPS,
    EXERCISE_LEAVES_ON_STREAM: _LEAVES_ON_STREAM_STEPS,
    EXERCISE_MUSCLE_RELAXATION: _MUSCLE_RELAXATION_STEPS,
    EXERCISE_BEHAVIORAL_EXPERIMENT: _BEHAVIORAL_EXPERIMENT_STEPS,
    EXERCISE_SELF_COMPASSION: _SELF_COMPASSION_STEPS,
    EXERCISE_IMPROVE: _IMPROVE_STEPS,
    EXERCISE_VALUES_COMPASS: _VALUES_COMPASS_STEPS,
    EXERCISE_GRATITUDE: _GRATITUDE_STEPS,
    EXERCISE_CONTINUUM: _CONTINUUM_STEPS,
}

# Display names for exercise-aware fallback messages.
_EXERCISE_DISPLAY_NAMES: dict[str, str] = {
    EXERCISE_5_4_3_2_1: "a grounding moment",
    EXERCISE_BOX_BREATHING: "a box breathing cycle",
    EXERCISE_STOP_TECHNIQUE: "the STOP technique",
    EXERCISE_THOUGHT_RECORD: "a thought record",
    EXERCISE_TINY_ACTION: "a tiny action experiment",
    EXERCISE_LEAVES_ON_STREAM: "a defusion exercise",
    EXERCISE_MUSCLE_RELAXATION: "a muscle relaxation exercise",
    EXERCISE_BEHAVIORAL_EXPERIMENT: "a behavioral experiment",
    EXERCISE_SELF_COMPASSION: "a self-compassion break",
    EXERCISE_IMPROVE: "an IMPROVE the moment exercise",
    EXERCISE_VALUES_COMPASS: "a values compass exercise",
    EXERCISE_GRATITUDE: "a gratitude inventory",
    EXERCISE_CONTINUUM: "a continuum exercise",
}


# ── Step-state classifier ──────────────────────────────────────────────

# Explicit exit signals. The user wants to STOP the exercise.
_EXIT_PATTERNS: tuple[str, ...] = (
    r"\b(?:stop|quit|cancel|never[\s-]?mind|nvm)\b",
    r"\bi (?:don'?t|do not) (?:want to|wanna)\b",
    r"\b(?:this|it) (?:isn'?t|is not|ain'?t) helping\b",
    r"\b(?:can|could) we just talk\b",
    r"\b(?:i need to|i have to|i should) (?:go|stop|step away)\b",
    r"\bnot (?:in the mood|into this|feeling (?:this|it))\b",
)

# Explicit "I can't" stuck signals. The user wants help with the step
# itself but can't engage — the escalation ladder should offer to
# rephrase or (after multiple turns) exit.
_STUCK_PATTERNS: tuple[str, ...] = (
    r"\bi can'?t\b",
    r"\bi (?:don'?t|do not) know\b",
    r"\b(?:nothing|none) (?:comes to mind|stands out)\b",
    r"\b(?:this is|that is|it'?s) (?:stupid|pointless|not working)\b",
    r"\bi (?:am|'?m) stuck\b",
)


# User confirmation patterns. Used for steps where the user does
# something (breathe, visualize, pause) and confirms they did it.
# These are intentionally strict: bare "ok" / "done" are full-message
# matches; longer confirmations require specific phrasing. This avoids
# false positives where "ok" appears mid-sentence (e.g., "ok but this
# isn't working" — which hits STUCK first anyway via pattern priority).
_CONFIRMATION_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:ok|okay|done|yes|yeah|yep|yup|got it|did it|ready|"
    r"finished|mhm|mhmm|alright|sure)\s*[.!]?\s*$",
    r"\b(?:i did|i'?ve done|i'?m done|i'?m ready|done with that|"
    r"i did that|that'?s done)\b",
    r"\b(?:took (?:a |the )?breath|breathed?|exhaled?|inhaled?|"
    r"held it|paused|i (?:can |do )?(?:see|picture|imagine) it)\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any of the patterns."""

    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _count_listed_items(message: str) -> int:
    """Count how many distinct items the user listed in their message.

    Used by the step-state classifier to decide whether a counting-based
    step is complete. The heuristic is simple: split on common list
    delimiters (commas, "and", newlines) and count non-trivial tokens.

    Examples:
        "a lamp, a plant, and my coffee cup" → 3
        "I see my keys and a book" → 2
        "um, a plant?" → 1
        "just a chair" → 1
        "" → 0

    This is deliberately lenient — users phrase lists in many different
    ways, and being too strict about the counting pattern means missing
    valid completions. The classifier uses this count + the step's
    ``min_count_for_completion`` to decide COMPLETE vs HOLD.
    """

    if not message.strip():
        return 0

    # Strip filler words and question marks so "um, a plant?" doesn't
    # get counted as 2 items ("um" and "a plant?").
    cleaned = re.sub(
        r"\b(?:um|uh|hmm+|well|maybe|like|just|i see|i hear|i feel|i smell|i taste)\b",
        " ",
        message,
        flags=re.IGNORECASE,
    )

    # Split on commas, semicolons, " and ", " & ", newlines. These are
    # the canonical list separators; anything else is either punctuation
    # noise or a single-item phrase.
    parts = re.split(r"(?:,|;|\band\b|&|\n)", cleaned, flags=re.IGNORECASE)

    # A part counts if it has at least one non-trivial word (length >= 2
    # after stripping whitespace and punctuation).
    non_trivial = 0
    for part in parts:
        stripped = re.sub(r"[^\w\s]", "", part).strip()
        if len(stripped) >= 2:
            non_trivial += 1

    return non_trivial


def _classify_step_state(
    message: str,
    current_step: ExerciseStep,
) -> StepState:
    """Classify the user's message as complete / hold / stuck / exit.

    Order of checks matters:
    1. EXIT first — explicit exit signals dominate everything else,
       even if the message happens to name items. If the user says
       "I can't, this isn't helping, let's stop" we exit, period.
    2. STUCK second — explicit "I can't" signals. The user is engaging
       but needs help, not advancement.
    3. COMPLETE third — did the user name at least
       ``min_count_for_completion`` items?
    4. HOLD as the fallback — the user is engaging tentatively or
       sharing something off-step. Give space, don't advance, don't
       escalate.

    Note: HOLD is the "safe default" — if the classifier is uncertain,
    it should HOLD rather than advance or exit. Advancing incorrectly
    rushes the user through a step they didn't finish; exiting
    incorrectly abandons an exercise they were engaging with. Holding
    wastes at most one turn of prompting.
    """

    if _matches_any(message, _EXIT_PATTERNS):
        return "exit"

    if _matches_any(message, _STUCK_PATTERNS):
        return "stuck"

    if current_step.completion_mode == "user_confirmation":
        if _matches_any(message, _CONFIRMATION_PATTERNS):
            return "complete"
    else:
        item_count = _count_listed_items(message)
        if item_count >= current_step.min_count_for_completion:
            return "complete"

    return "hold"


# ── State delta helpers ────────────────────────────────────────────────


def _start_exercise_delta(
    state: AgentState,
    *,
    exercise_type: str,
) -> dict[str, Any]:
    """Return the progress delta that starts a new exercise at step 0."""

    progress = state.get("progress", {})
    return {
        "progress": {
            **progress,
            "exercise_type": exercise_type,
            "exercise_step": 0,
        },
    }


def _advance_step_delta(state: AgentState) -> dict[str, Any]:
    """Return the progress delta that bumps the exercise step index."""

    progress = state.get("progress", {})
    current = progress.get("exercise_step") or 0
    return {
        "progress": {
            **progress,
            "exercise_step": current + 1,
        },
    }


def _clear_exercise_delta(state: AgentState) -> dict[str, Any]:
    """Return the progress delta that clears exercise state.

    Used on both exit and natural completion. Setting both fields to
    ``None`` is the marker for "no exercise running" that the
    dispatcher fast-path checks for.
    """

    progress = state.get("progress", {})
    return {
        "progress": {
            **progress,
            "exercise_type": None,
            "exercise_step": None,
        },
    }


# ── Main node function ────────────────────────────────────────────────


# ── Exercise selection ─────────────────────────────────────────────────

# Keyword patterns for selecting an exercise from the user's message.
# Checked in order; the first match wins. The default fallback is
# 5-4-3-2-1 grounding.
_EXERCISE_SELECTORS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Grounding
    (("breath", "breathe", "breathing", "box breath"), EXERCISE_BOX_BREATHING),
    (
        ("muscle", "tense", "tension", "relax my body", "pmr", "progressive"),
        EXERCISE_MUSCLE_RELAXATION,
    ),
    (("stop technique", "stop method", "s.t.o.p"), EXERCISE_STOP_TECHNIQUE),
    # Emotion regulation (must come before thought work — "improve"
    # contains "prove" which would match behavioral experiment)
    (
        (
            "improve the moment",
            "improve",
            "cope",
            "get through this",
            "emotion regulation",
        ),
        EXERCISE_IMPROVE,
    ),
    # Thought work
    (
        (
            "behavioral experiment",
            "test this belief",
            "is this.*true",
            "prove it",
            "check if",
        ),
        EXERCISE_BEHAVIORAL_EXPERIMENT,
    ),
    # Continuum: targets explicit all-or-nothing self-labels and direct
    # requests. Triggers are intentionally narrow — "always" and "never"
    # alone are too common. We match patterns that combine absolute
    # language with self-reference ("I'm a terrible", "I always fail")
    # plus explicit exercise requests.
    (
        (
            "continuum",
            "all.or.nothing",
            "black.and.white",
            r"i'?m (?:a )?(?:terrible|horrible|worst|complete|total)",
            r"i (?:always|never) (?:fail|mess|ruin|screw|disappoint|let)",
            r"100\s*%",
        ),
        EXERCISE_CONTINUUM,
    ),
    (
        (
            "thought record",
            "thought check",
            "examine.*thought",
            "look at.*thought",
            "belief",
        ),
        EXERCISE_THOUGHT_RECORD,
    ),
    # Behavioral activation
    (
        (
            "stuck",
            "can't start",
            "motivation",
            "depleted",
            "can't do anything",
            "small action",
            "tiny action",
        ),
        EXERCISE_TINY_ACTION,
    ),
    # Self-compassion (must come before values compass — "compassion"
    # contains "compass" as a substring)
    (
        (
            "self.?compassion",
            "kinder to myself",
            "hard on myself",
            "self.?critical",
            "hate myself",
            "compassion break",
        ),
        EXERCISE_SELF_COMPASSION,
    ),
    # Acceptance / defusion / values
    (
        (
            "values",
            "what matters",
            "meaning",
            "purpose",
            "direction",
            "compass",
            "life direction",
        ),
        EXERCISE_VALUES_COMPASS,
    ),
    (
        (
            "accept",
            "let go",
            "defusion",
            "leaves",
            "step back from",
            "stop fighting",
            "fused",
            "fusion",
        ),
        EXERCISE_LEAVES_ON_STREAM,
    ),
    (
        (
            "grateful",
            "gratitude",
            "thankful",
            "something good",
            "positive",
            "appreciate",
        ),
        EXERCISE_GRATITUDE,
    ),
    # Generic grounding triggers (lower priority)
    (("stop", "pause", "slow down"), EXERCISE_STOP_TECHNIQUE),
    (("overwhelmed", "too much"), EXERCISE_IMPROVE),
)


def _select_exercise(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Pick an exercise type based on the current message plus recent context.

    The selector still prioritizes explicit keywords in the current message,
    but short acceptance turns like "can we work through that?" need a small
    amount of recent user-turn context to avoid falling back to generic
    grounding when the prior turn clearly named a cognitive belief pattern.

    Returns the exercise_type constant. Falls back to 5-4-3-2-1
    grounding when no keyword matches — the most established
    exercise and the safest default.
    """

    lowered = message.lower()
    for keywords, exercise_type in _EXERCISE_SELECTORS:
        for kw in keywords:
            if re.search(kw, lowered):
                return exercise_type

    if re.search(r"\bwork through (?:that|this|it)\b", lowered):
        recent_user_text = " ".join(
            turn.get("content", "")
            for turn in (history or [])[-6:]
            if turn.get("role") == "user" and turn.get("content")
        ).lower()
        if any(
            marker in recent_user_text
            for marker in (
                "i always assume",
                "one mistake means",
                "i'm incompetent",
                "im incompetent",
                "everyone will see",
                "i'm about to fail",
                "im about to fail",
                "this belief",
                "this thought",
            )
        ):
            return EXERCISE_THOUGHT_RECORD

    return EXERCISE_5_4_3_2_1


# Deterministic fallback strings used when no LLM client is available.
_FALLBACK_HOLD = "Take your time — even one counts. There's no rush."
_FALLBACK_STUCK_REPHRASE = (
    "That's okay. Let's make it smaller — just one thing you can "
    "notice right now, whatever stands out."
)
_FALLBACK_EXIT = "Of course, let's stop. What would feel most helpful right now?"


def _get_current_step(
    exercise_type: str | None,
    step_index: int | None,
) -> ExerciseStep | None:
    """Return the current ExerciseStep, or None if out of range / invalid."""

    if exercise_type is None or step_index is None:
        return None
    steps = _EXERCISE_REGISTRY.get(exercise_type)
    if steps is None:
        return None
    if step_index < 0 or step_index >= len(steps):
        return None
    return steps[step_index]


def _is_last_step(exercise_type: str, step_index: int) -> bool:
    """Return whether the given step is the last one in the exercise."""

    steps = _EXERCISE_REGISTRY.get(exercise_type)
    if steps is None:
        return False
    return step_index >= len(steps) - 1


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Drive a multi-turn guided exercise.

    Two entry conditions:

    1. **Starting an exercise** — ``progress.exercise_type`` is None
       (no exercise running). The dispatcher's LLM classifier picked
       this mode based on the user's current message. The node starts
       the default exercise (5-4-3-2-1 grounding) at step 0.

    2. **Continuing an exercise** — ``progress.exercise_type`` is set
       from a prior turn. The dispatcher's active-exercise fast-path
       routed here without re-classifying. The node classifies the
       user's message as complete/hold/stuck/exit and acts accordingly.

    The node ALWAYS returns a response + routing delta, and may also
    return a progress delta when the exercise state changes (start,
    advance, clear).

    Falls back to deterministic templates when no LLM client is
    available. The fallbacks are comprehensive enough to drive the
    full 5-4-3-2-1 exercise end-to-end with no LLM — not just start.
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client
    memory_store = runtime.context.memory_store
    memory_mode = runtime.context.memory_mode
    progress = state.get("progress", {})
    exercise_type = progress.get("exercise_type")
    step_index = progress.get("exercise_step")

    # ── Entry condition 1: starting a new exercise ─────────────────
    if exercise_type is None or step_index is None:
        return _handle_start(state, llm_client)

    # ── Entry condition 2: continuing an existing exercise ─────────
    current_step = _get_current_step(exercise_type, step_index)
    if current_step is None:
        # Invalid state (unknown exercise_type, or step_index out of
        # range). Clear and fall back to a fresh start — this is a
        # defensive path that should never fire in normal operation
        # but prevents lockup if the state gets corrupted.
        logger.warning(
            "run_guided_exercise_response_node: invalid exercise state "
            "exercise_type=%r step_index=%r; clearing and restarting",
            exercise_type,
            step_index,
        )
        cleared = _clear_exercise_delta(state)
        start_delta = _handle_start(state, llm_client)
        # Merge: the start delta's progress update wins over the clear
        return {**cleared, **start_delta}

    return await _handle_continue(
        state=state,
        llm_client=llm_client,
        memory_store=memory_store,
        memory_mode=memory_mode,
        exercise_type=exercise_type,
        step_index=step_index,
        current_step=current_step,
    )


def _handle_start(
    state: AgentState,
    llm_client: Any,  # BaseLLMClient | None, but Any avoids an import
) -> dict[str, Any]:
    """Start a new exercise at step 0.

    Selects the exercise based on keywords in the user's message.
    Falls back to 5-4-3-2-1 grounding when no keyword matches.
    """

    message = state.get("message", "")
    selected = _select_exercise(message, history=state.get("history", []))
    steps = _EXERCISE_REGISTRY[selected]
    response_text = steps[0].prompt_fallback
    # The start step uses the deterministic fallback directly —
    # variation doesn't add much for instructions. The LLM is
    # used on subsequent turns (hold/complete/exit) where wording
    # variation helps more.
    _ = llm_client  # unused for the start step; flagged for future use

    start_progress_delta = _start_exercise_delta(state, exercise_type=selected)
    return {
        **start_progress_delta,
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "guided_exercise",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }


async def _handle_continue(
    *,
    state: AgentState,
    llm_client: Any,  # BaseLLMClient | None
    memory_store: Any,  # MemoryStore | None
    memory_mode: Any,  # MemoryMode
    exercise_type: str,
    step_index: int,
    current_step: ExerciseStep,
) -> dict[str, Any]:
    """Continue an exercise based on the user's current message.

    Classifies the message as complete/hold/stuck/exit, builds the
    appropriate progress + response delta, and returns it. The LLM
    (if available) generates the response text for the new state;
    the deterministic fallback is used when the LLM is unavailable
    or errors.
    """

    message = state.get("message", "")
    step_state = _classify_step_state(message, current_step)

    logger.debug(
        "guided_exercise continue: exercise_type=%s step_index=%d step_state=%s",
        exercise_type,
        step_index,
        step_state,
    )

    if step_state == "exit":
        return _build_exit_delta(state, llm_client=llm_client)

    if step_state == "stuck":
        return await _build_stuck_delta(state, llm_client=llm_client)

    if step_state == "hold":
        return await _build_hold_delta(state, llm_client=llm_client)

    # step_state == "complete" → advance or finish
    if _is_last_step(exercise_type, step_index):
        return await _build_complete_delta(
            state,
            llm_client=llm_client,
            memory_store=memory_store,
            memory_mode=memory_mode,
        )

    return await _build_advance_delta(
        state=state,
        llm_client=llm_client,
        exercise_type=exercise_type,
        next_step_index=step_index + 1,
    )


def _build_exit_delta(
    state: AgentState,
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the delta for an exit — clear state, warm landing."""

    # Exit responses are short and specific enough that the
    # deterministic fallback is good enough. The LLM path would
    # marginally improve wording but not change behavior, and the
    # priority here is that the exercise state gets cleared
    # reliably. We still return the response text from the fallback
    # string to keep exit behavior deterministic.
    _ = llm_client  # unused for exit; flagged for future use

    cleared = _clear_exercise_delta(state)
    return {
        **cleared,
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": _FALLBACK_EXIT,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "guided_exercise",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_stuck_delta(
    state: AgentState,
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the delta for a stuck classification — offer to simplify."""

    progress = state.get("progress", {})
    step_index = progress.get("exercise_step", 0)
    exercise_type = progress.get("exercise_type", EXERCISE_5_4_3_2_1)
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.prompt_fallback if current_step else ""

    directive = (
        f"The user is STUCK on step {step_index} of the exercise. "
        f'The step asked: "{step_ref}"\n'
        f"Offer a simpler version of the same step — make it smaller and "
        f"more concrete. Do NOT advance to the next step or repeat the "
        f"original instruction verbatim."
    )

    response_text = _FALLBACK_STUCK_REPHRASE
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(
                    state,
                    mode="guided_exercise",
                    step_directive=directive,
                ),
                system_instruction=build_guided_exercise_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Guided exercise stuck-path LLM call failed; "
                "using deterministic fallback.",
                exc_info=True,
            )

    return {
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "guided_exercise",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_hold_delta(
    state: AgentState,
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the delta for a hold classification — space, no advancement."""

    progress = state.get("progress", {})
    step_index = progress.get("exercise_step", 0)
    exercise_type = progress.get("exercise_type", EXERCISE_5_4_3_2_1)
    current_step = _get_current_step(exercise_type, step_index)
    step_ref = current_step.prompt_fallback if current_step else ""

    directive = (
        f"The user gave a tentative or partial response to step {step_index}. "
        f'The step asked: "{step_ref}"\n'
        f"Give brief encouragement to continue this same step. Do NOT "
        f"advance to the next step or re-explain the full instruction."
    )

    response_text = _FALLBACK_HOLD
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(
                    state,
                    mode="guided_exercise",
                    step_directive=directive,
                ),
                system_instruction=build_guided_exercise_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Guided exercise hold-path LLM call failed; "
                "using deterministic fallback.",
                exc_info=True,
            )

    return {
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "guided_exercise",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_advance_delta(
    *,
    state: AgentState,
    llm_client: Any,
    exercise_type: str,
    next_step_index: int,
) -> dict[str, Any]:
    """Build the delta for advancing to the next step.

    Updates ``progress.exercise_step`` and returns the next step's
    prompt as the response text (via LLM or fallback).
    """

    steps = _EXERCISE_REGISTRY[exercise_type]
    next_step = steps[next_step_index]
    response_text = next_step.prompt_fallback

    total_steps = len(steps)
    directive = (
        f"The user completed step {next_step_index - 1} of {total_steps - 1}. "
        f"Briefly acknowledge what they shared, then move to step "
        f"{next_step_index}.\n"
        f'Step {next_step_index} instruction: "{next_step.prompt_fallback}"\n'
        f"Rephrase naturally in your own words — do NOT repeat this "
        f"instruction verbatim. Do NOT repeat any earlier step."
    )

    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(
                    state,
                    mode="guided_exercise",
                    step_directive=directive,
                ),
                system_instruction=build_guided_exercise_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Guided exercise advance-path LLM call failed; "
                "using deterministic fallback.",
                exc_info=True,
            )

    advance_progress = _advance_step_delta(state)
    return {
        **advance_progress,
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "guided_exercise",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_complete_delta(
    state: AgentState,
    *,
    llm_client: Any,
    memory_store: Any = None,  # MemoryStore | None
    memory_mode: Any = None,  # MemoryMode | None
) -> dict[str, Any]:
    """Build the delta for natural completion of the exercise.

    Clears exercise state, writes a coping_strategy semantic fact
    (if memory is enabled), and returns a brief "you did it" response.
    """

    progress = state.get("progress", {})
    exercise_type = progress.get("exercise_type", EXERCISE_5_4_3_2_1)
    display_name = _EXERCISE_DISPLAY_NAMES.get(exercise_type, "that exercise")

    directive = (
        f"The user just finished the LAST step of the exercise. "
        f"Briefly acknowledge what they shared, name what they just did "
        f"({display_name}), and invite them to notice how their body "
        f"feels now. Do NOT start a new exercise or ask a new question."
    )

    fallback_complete = (
        f"You just walked yourself through {display_name}. "
        f"Notice how your body feels now compared to when we started."
    )
    response_text = fallback_complete
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(
                    state,
                    mode="guided_exercise",
                    step_directive=directive,
                ),
                system_instruction=build_guided_exercise_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Guided exercise complete-path LLM call failed; "
                "using deterministic fallback.",
                exc_info=True,
            )

    # ── Write exercise completion as a coping_strategy fact ──────────
    # This runs BEFORE clearing exercise state, while exercise_type
    # is still available. Only writes in non-incognito mode with a
    # valid memory store.
    await _write_exercise_completion_fact(
        state=state,
        exercise_type=exercise_type,
        display_name=display_name,
        memory_store=memory_store,
        memory_mode=memory_mode,
    )

    cleared = _clear_exercise_delta(state)
    return {
        **cleared,
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
            "should_persist_memory": True,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "guided_exercise",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }


async def _write_exercise_completion_fact(
    *,
    state: AgentState,
    exercise_type: str,
    display_name: str,
    memory_store: Any,
    memory_mode: Any,
) -> None:
    """Write a semantic fact recording that the user completed an exercise.

    This is a deterministic write — no LLM involved. The fact is
    written as a coping_strategy with predicate USES, which the
    retrieval system will surface on future turns when the user's
    context overlaps with coping strategies.

    Skips silently when:
    - memory_store is None (no store configured)
    - memory_mode is INCOGNITO (no persistent writes allowed)
    - any error occurs (logged, never raised)
    """

    if memory_store is None or memory_mode == MemoryMode.INCOGNITO:
        return

    owner_id = resolve_owner_id(state)
    session_id = state.get("session_id") or owner_id
    turn_count = state.get("progress", {}).get("turn_count", 0)

    now = datetime.now(timezone.utc).isoformat()
    fact = SemanticFact(
        id=str(uuid4()),
        category="coping_strategy",
        subject=EntityRef(type="User", identifier=owner_id),
        predicate="USES",
        object=EntityRef(type="CopingStrategy", identifier=exercise_type),
        evidence_quote=f"Completed {display_name} exercise.",
        confidence="high",
        source_session_id=session_id,
        source_turn_index=turn_count,
        created_at=now,
        last_referenced_at=now,
        dormant_at=None,
        superseded_by=None,
        user_visible=True,
    )

    try:
        namespace = (owner_id, "semantic")
        await memory_store.aput(
            namespace,
            key=fact.id,
            value=fact.model_dump(mode="json"),
        )
        logger.info(
            "Wrote exercise completion fact: exercise_type=%s owner=%s",
            exercise_type,
            owner_id,
        )
    except Exception:
        logger.warning(
            "Failed to write exercise completion fact; skipping.",
            exc_info=True,
        )

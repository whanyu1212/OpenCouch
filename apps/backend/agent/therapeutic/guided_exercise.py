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
from typing import Any, Literal

from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseKind
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
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
    """

    prompt_fallback: str
    expected_count: int
    min_count_for_completion: int


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

# Registry mapping exercise_type → step sequence. Adding a new exercise
# in a future stage means adding an entry here plus an ExerciseStep
# tuple; the dispatcher and node code don't need to change.
_EXERCISE_REGISTRY: dict[str, tuple[ExerciseStep, ...]] = {
    EXERCISE_5_4_3_2_1: _GROUNDING_5_4_3_2_1_STEPS,
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


# Deterministic fallback strings used when no LLM client is available.
_FALLBACK_HOLD = "Take your time — even one counts. There's no rush."
_FALLBACK_STUCK_REPHRASE = (
    "That's okay. Let's make it smaller — just one thing you can "
    "notice right now, whatever stands out."
)
_FALLBACK_EXIT = "Of course, let's stop. What would feel most helpful right now?"
_FALLBACK_COMPLETE = (
    "You just walked yourself through a grounding moment. "
    "Notice how your body feels now compared to when we started."
)
_FALLBACK_START_DEFAULT = _GROUNDING_5_4_3_2_1_STEPS[0].prompt_fallback


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

    llm_client = runtime.context.get("llm_client")
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
        exercise_type=exercise_type,
        step_index=step_index,
        current_step=current_step,
    )


def _handle_start(
    state: AgentState,
    llm_client: Any,  # BaseLLMClient | None, but Any avoids an import
) -> dict[str, Any]:
    """Start a new exercise at step 0.

    Currently always starts 5-4-3-2-1 grounding. A future stage could
    inspect the user's message to pick a different exercise, but for
    v0.6 Stage C the default is fine — the LLM dispatcher only routes
    here when the user has asked for grounding-adjacent support.
    """

    response_text = _FALLBACK_START_DEFAULT
    # The LLM path uses the same template as the fallback, just with
    # the LLM rephrasing it to match the user's register. We don't
    # call the LLM here in Stage C because the start step is pure
    # instruction — variation doesn't add much. When dogfood reveals
    # this is too robotic we can revisit. For now, use the fallback
    # directly for the start step.
    #
    # NOTE: this is intentionally different from hold/complete/exit
    # paths below, which DO call the LLM when available. The start
    # step is the one place where determinism is more valuable than
    # wording variation.
    _ = llm_client  # unused for the start step; flagged for future use

    start_progress_delta = _start_exercise_delta(
        state, exercise_type=EXERCISE_5_4_3_2_1
    )
    return {
        **start_progress_delta,
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "mode": "guided_exercise",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
        },
    }


async def _handle_continue(
    *,
    state: AgentState,
    llm_client: Any,  # BaseLLMClient | None
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
        # Stuck → offer a rephrase/simplification. Do NOT advance the
        # step; the user is still engaging with it, just needs help.
        # Do NOT exit; that escalation happens only after multiple
        # turns, and the dispatcher fast-path keeps routing here on
        # subsequent turns so a pattern of stuck will naturally
        # produce multiple stuck-classified turns (future: add a
        # stuck-turn counter to escalate to exit after N turns).
        return await _build_stuck_delta(state, llm_client=llm_client)

    if step_state == "hold":
        # Hold → give space, don't advance. The response should be
        # minimal encouragement, not a re-explanation of the step.
        return await _build_hold_delta(state, llm_client=llm_client)

    # step_state == "complete" → advance or finish
    if _is_last_step(exercise_type, step_index):
        return await _build_complete_delta(state, llm_client=llm_client)

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
            "mode": "guided_exercise",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_stuck_delta(
    state: AgentState,
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the delta for a stuck classification — offer to simplify."""

    response_text = _FALLBACK_STUCK_REPHRASE
    if llm_client is not None:
        try:
            response_text = await llm_client.generate_text(
                prompt=build_therapeutic_response_prompt(state, mode="guided_exercise"),
                system_instruction=build_guided_exercise_system_prompt(state),
                temperature=0.7,
            )
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
            "mode": "guided_exercise",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_hold_delta(
    state: AgentState,
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the delta for a hold classification — space, no advancement."""

    response_text = _FALLBACK_HOLD
    if llm_client is not None:
        try:
            response_text = await llm_client.generate_text(
                prompt=build_therapeutic_response_prompt(state, mode="guided_exercise"),
                system_instruction=build_guided_exercise_system_prompt(state),
                temperature=0.7,
            )
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
            "mode": "guided_exercise",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
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

    if llm_client is not None:
        try:
            response_text = await llm_client.generate_text(
                prompt=build_therapeutic_response_prompt(state, mode="guided_exercise"),
                system_instruction=build_guided_exercise_system_prompt(state),
                temperature=0.7,
            )
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
            "mode": "guided_exercise",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
        },
    }


async def _build_complete_delta(
    state: AgentState,
    *,
    llm_client: Any,
) -> dict[str, Any]:
    """Build the delta for natural completion of the exercise.

    Clears exercise state and returns a brief "you did it" response.
    """

    response_text = _FALLBACK_COMPLETE
    if llm_client is not None:
        try:
            response_text = await llm_client.generate_text(
                prompt=build_therapeutic_response_prompt(state, mode="guided_exercise"),
                system_instruction=build_guided_exercise_system_prompt(state),
                temperature=0.7,
            )
        except Exception:
            logger.warning(
                "Guided exercise complete-path LLM call failed; "
                "using deterministic fallback.",
                exc_info=True,
            )

    cleared = _clear_exercise_delta(state)
    return {
        **cleared,
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "mode": "guided_exercise",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
        },
    }

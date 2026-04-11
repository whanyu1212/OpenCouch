"""Therapeutic dispatch node — picks which mode handles the current turn.

Phase 1 v0.1 uses a **hybrid dispatcher**: high-precision regex patterns
short-circuit the decision for obvious cases, and an LLM structured-output
classifier handles everything else when a provider client is available.
If no LLM client is present, the regex pathway is the sole source of
truth (default-to-supportive for anything unmatched).

Dispatch flow:
    1. REFLECTIVE fast path — if a high-precision pattern-recognition
       regex matches, go straight to reflective. Bypass the LLM.
    2. Explicit CONFUSION markers — "huh?", "what do you mean?" etc.
       route to clarifying without consulting the LLM.
    3. LLM classifier — if an llm_client is available and no fast path
       fired, call ``generate_structured`` with the ``DispatchDecision``
       schema and the last ~6 turns of history. Trust the LLM's pick.
    4. Regex fallback — no LLM client, or LLM call failed: apply the
       short-message-without-self-report heuristic for clarifying, and
       default to supportive otherwise.

This mirrors the hybrid pattern in ``agent/nodes/crisis_gate.py``:
deterministic rules first, LLM fallback for ambiguous cases, graceful
degradation when the LLM is unavailable.

The dispatcher returns ``Command(goto=<node_name>)`` with an empty
update dict — the individual mode nodes are responsible for setting
``routing.mode``, ``routing.mode_source``, and ``routing.mode_type``
in their own deltas. This keeps each mode self-documenting and makes
the LangSmith trace clearly show which mode ran.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.memory.models import DispatchDecision
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


# ─── Node names (must match agent/therapeutic/graph.py registration) ───────

SUPPORTIVE_NODE = "supportive_response_node"
REFLECTIVE_NODE = "reflective_response_node"
CLARIFYING_NODE = "clarifying_response_node"


# ─── Dispatch patterns ─────────────────────────────────────────────────────

# Reflective mode is triggered when the user is naming a recurring pattern,
# asking "why does this keep happening" type questions, or surfacing a
# theme across multiple turns. These patterns are deliberately specific —
# we want high precision (few false positives) more than high recall,
# because a false-positive reflective response ("I notice you keep doing
# X" when the user hasn't been doing X) is the worst failure mode.
#
# A regex hit on these patterns is treated as a HIGH-CONFIDENCE fast path
# that bypasses the LLM dispatcher entirely.
REFLECTIVE_PATTERNS: tuple[str, ...] = (
    r"\bwhy do(?:es)? (?:i|this|it) keep\b",
    r"\bwhy does (?:this|it) (?:keep|always) happen\b",
    # "I always ... doing/saying/feeling/ending up/end up" — accepts
    # both the present participle ("ending up apologizing") and the
    # bare infinitive ("end up apologizing"). v0.5 eval surfaced that
    # "I always end up apologizing first" wasn't matching because the
    # original alternation only listed "ending up". Regression pin:
    # the reflective_i_always test case in therapeutic_routing_v0.json.
    r"\bi (?:always|keep)\b.{0,20}\b(?:doing|saying|feeling|ending up|end up)\b",
    r"\bevery time i\b",
    r"\bsame (?:thing|pattern|story|cycle)\b",
    r"\bthis (?:keeps|always) happen(?:ing|s)\b",
    r"\bi('m| am) stuck in (?:this|the same|a) (?:pattern|cycle|loop)\b",
    r"\bi notice (?:i|myself) (?:always|keep|often)\b",
    r"\bis there a pattern\b",
)

# Explicit confusion markers — the user is signaling they didn't
# understand something, or their message is pure noise. These are
# unambiguous and also fast-path (bypass the LLM).
CONFUSION_PATTERNS: tuple[str, ...] = (
    r"^\s*huh\??\s*$",
    r"^\s*what\??\s*$",
    r"^\s*what do you mean\b",
    r"\bi don'?t (?:understand|get it|follow)\b",
    r"\bcan you (?:explain|clarify)\b",
    r"\bi'?m (?:not sure|confused)\b",
)

# A short message routes to clarifying in the FALLBACK path only (when
# no LLM is available) ONLY IF it's not a complete self-report. A
# self-report has a first-person subject + a feeling or state verb —
# "I feel overwhelmed", "I am tired", "I'm anxious" — and should route
# to supportive even when brief. Truly sparse messages without this
# structure ("ok", "huh?", "thanks") still route to clarifying.
#
# When an LLM client is available, this heuristic is skipped; the LLM
# makes the call with full context.
CLARIFYING_MAX_WORD_COUNT = 5
SELF_REPORT_PATTERNS: tuple[str, ...] = (
    # "I feel", "I am", "I was", "I have", "I had" (with space separator)
    r"\bi (?:feel|am|was|have|had)\b",
    # "I'm", "I've", "I'd" (contracted forms — no space before the suffix)
    r"\bi'(?:m|ve|d)\b",
    # "my <body-part> is/feels/hurts" — somatic self-reports
    r"\bmy (?:head|heart|body|chest|stomach|mind) (?:is|feels|hurts)\b",
    # "it hurts", "it feels", etc. — impersonal self-reports
    r"\bit (?:hurts|feels|sucks|is tough|is hard)\b",
)


# ─── Regex helpers ─────────────────────────────────────────────────────────


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any of the patterns."""

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _word_count(text: str) -> int:
    """Count words in text, ignoring punctuation-only tokens."""

    return len([w for w in re.findall(r"\w+", text) if w])


def pick_therapeutic_mode(
    message: str,
) -> Literal["supportive", "reflective", "clarifying"]:
    """Return the therapeutic mode name for a given user message (regex-only).

    Pure function — takes only the message, returns the mode name.
    Exposed as a module-level function so it can be unit-tested in
    isolation from the LangGraph runtime.

    This is the **regex-only** fallback path used when no LLM client is
    available, or when the LLM call fails. When an LLM client is
    available, ``run_therapeutic_dispatch_node`` uses a hybrid approach
    that first checks the high-precision fast-path patterns below and
    then consults the LLM for anything else.
    """

    lowered = message.lower()

    # Reflective wins first — pattern recognition beats terseness.
    # "Why do I keep doing this?" is short but clearly a pattern question.
    if _matches_any(lowered, REFLECTIVE_PATTERNS):
        return "reflective"

    # Explicit confusion markers always route to clarifying, regardless
    # of length. "I don't really understand what you mean by that" is
    # a long message that still needs clarification.
    if _matches_any(lowered, CONFUSION_PATTERNS):
        return "clarifying"

    # Short message without a self-report → clarifying. A self-report
    # like "I feel overwhelmed" is a complete statement even when brief,
    # so it falls through to supportive.
    is_short = _word_count(message) <= CLARIFYING_MAX_WORD_COUNT
    if is_short and not _matches_any(lowered, SELF_REPORT_PATTERNS):
        return "clarifying"

    # Default: supportive
    return "supportive"


# ─── LLM classifier path ───────────────────────────────────────────────────


def build_therapeutic_dispatch_system_prompt() -> str:
    """System prompt for the LLM-backed therapeutic dispatcher.

    Kept inline (not in ``agent/therapeutic/prompts.py``) because it is
    specific to the dispatcher's classification role and doesn't benefit
    from the knowledge-file composition that the mode prompts use.
    Dispatching is a classification task, not a response-generation task.
    """

    return (
        "You are the dispatcher for a mental health support conversation. "
        "Your only job is to pick the single best therapeutic response mode "
        "for the next turn, based on what the user just said and the recent "
        "conversation history.\n\n"
        "The three modes are:\n"
        "- supportive: default warm validation. Use when the user is sharing "
        "feelings, venting, or describing a situation without asking a "
        "pattern question. Also use for session-opening greetings and "
        "general capability questions like 'Hi, what can you do for me?' — "
        "these are warm-up signals from someone reaching out for help, not "
        "literal requests for tool documentation. This is the most common "
        "mode and the right default when in doubt.\n"
        "- reflective: pattern-naming and gentle probing. Use when the user "
        "is describing a recurring pattern, asking 'why does this keep "
        "happening?' type questions, or surfacing a theme. Only pick this "
        "mode when the user has ALREADY shown evidence of a pattern. Never "
        "introduce a pattern the user hasn't described.\n"
        "- clarifying: ask one focused question. Use only when the user's "
        "message is genuinely too ambiguous to respond to meaningfully "
        "(e.g., a bare 'ok' with no context, or an unclear pronoun reference "
        "to something the conversation hasn't covered), AND the user is not "
        "reporting a feeling or state. A short message like 'I feel sad' is "
        "a complete self-report and should NOT route to clarifying. A "
        "session-opening greeting is NOT clarifying territory — route those "
        "to supportive.\n\n"
        "Pick one mode. Return your decision in the structured schema. "
        "Keep the reasoning to one short sentence — it's for debugging, "
        "not for the user."
    )


def build_therapeutic_dispatch_prompt(state: AgentState) -> str:
    """User/task prompt for the LLM-backed therapeutic dispatcher.

    Injects the current message, the last ~6 turns of history, and a
    compact summary of working memory (if any) so the classifier has
    enough context to pick the right mode.
    """

    history = state.get("history", [])[-6:]
    history_block = (
        "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
            for turn in history
            if turn.get("content")
        )
        or "(no prior history)"
    )

    working_memory = state.get("working_memory", [])
    if working_memory:
        memory_block = "Relevant context from past sessions:\n" + "\n".join(
            f"- {snippet}" for snippet in working_memory[:3]
        )
    else:
        memory_block = "(no working memory for this turn)"

    return (
        f"Recent conversation:\n{history_block}\n\n"
        f"{memory_block}\n\n"
        f"Current user message:\nuser: {state['message']}\n\n"
        "Which therapeutic mode should handle this turn?"
    )


async def _pick_mode_with_llm(
    state: AgentState,
    llm_client,
) -> Literal["supportive", "reflective", "clarifying"]:
    """Call the structured-output LLM classifier to pick a mode.

    Returns the picked mode string. On any error, raises; callers are
    responsible for falling back to the regex pathway.
    """

    raw: DispatchDecision = await llm_client.generate_structured(
        prompt=build_therapeutic_dispatch_prompt(state),
        response_schema=DispatchDecision,
        system_instruction=build_therapeutic_dispatch_system_prompt(),
        temperature=0,
    )

    # The DispatchDecision.mode Literal is ["supportive", "reflective",
    # "psychoeducation", "guided_exercise", "closing", "clarifying"]. v0.1
    # only has three modes wired, so if the LLM returns one of the
    # three deferred modes, we normalize it to the closest v0.1 equivalent:
    # - psychoeducation → supportive (it's the safest fallback for
    #   educational questions; the user still gets a warm reply)
    # - guided_exercise → supportive (same — we don't have the exercise
    #   infrastructure yet)
    # - closing → supportive (closing is a phase-2 concern; supportive
    #   is the closest warm alternative)
    if raw.mode in ("supportive", "reflective", "clarifying"):
        return raw.mode  # type: ignore[return-value]
    logger.debug(
        "LLM dispatcher picked deferred mode %s; normalizing to supportive", raw.mode
    )
    return "supportive"


# ─── Node-name mapping ─────────────────────────────────────────────────────

# Mapping from mode name → subgraph node name. Kept as a dict so the
# dispatcher's logic stays pure (pick_therapeutic_mode returns a name)
# and the routing layer does the name-to-node translation.
_MODE_NODE_MAP: dict[str, str] = {
    "supportive": SUPPORTIVE_NODE,
    "reflective": REFLECTIVE_NODE,
    "clarifying": CLARIFYING_NODE,
}


# ─── Dispatch node ─────────────────────────────────────────────────────────


async def run_therapeutic_dispatch_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[
    Literal[
        "supportive_response_node",
        "reflective_response_node",
        "clarifying_response_node",
    ]
]:
    """Dispatch the turn to the right therapeutic mode node.

    Hybrid dispatch: high-precision regex fast paths first, then LLM
    classifier if available, then regex fallback as last resort.

    The update dict is empty — the individual mode nodes are responsible
    for setting ``routing.mode``, ``routing.mode_source``, and
    ``routing.mode_type`` in their own deltas.
    """

    message = state.get("message", "")
    lowered = message.lower()
    llm_client = runtime.context.get("llm_client")

    # ── Fast path 1: high-precision reflective patterns ─────────────────
    # These patterns only match when the user has explicitly used
    # pattern-recognition language. Skip the LLM — the regex is more
    # reliable for these specific phrasings.
    if _matches_any(lowered, REFLECTIVE_PATTERNS):
        logger.debug("therapeutic_dispatch: reflective fast path")
        return Command(update={}, goto=REFLECTIVE_NODE)

    # ── Fast path 2: explicit confusion markers ──────────────────────────
    # "Huh?", "what do you mean?" — unambiguous clarification requests.
    if _matches_any(lowered, CONFUSION_PATTERNS):
        logger.debug("therapeutic_dispatch: clarifying confusion-marker fast path")
        return Command(update={}, goto=CLARIFYING_NODE)

    # ── LLM classifier path ──────────────────────────────────────────────
    # Everything else that has an LLM client goes through the classifier.
    # This catches subtle cases the regex can't handle: implicit pattern
    # questions, emotionally nuanced messages, context-dependent routing.
    if llm_client is not None:
        try:
            mode = await _pick_mode_with_llm(state, llm_client)
            logger.debug("therapeutic_dispatch: LLM picked mode=%s", mode)
            return Command(update={}, goto=_MODE_NODE_MAP[mode])
        except Exception:
            logger.warning(
                "therapeutic_dispatch LLM classifier failed; falling back to regex.",
                exc_info=True,
            )

    # ── Regex fallback ───────────────────────────────────────────────────
    # No LLM client, or LLM call failed: apply the short-message-without-
    # self-report heuristic for clarifying, and default to supportive.
    mode = pick_therapeutic_mode(message)
    logger.debug("therapeutic_dispatch: regex fallback picked mode=%s", mode)
    return Command(update={}, goto=_MODE_NODE_MAP[mode])

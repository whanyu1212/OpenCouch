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
PSYCHOEDUCATION_NODE = "psychoeducation_response_node"
CLOSING_NODE = "closing_response_node"
GUIDED_EXERCISE_NODE = "guided_exercise_response_node"


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
    # v0.6 Stage A: exclude the "I don't understand WHY X" narrative form.
    # "I don't understand why that happened" is a narrative about the
    # user's own confusion with their reaction, not a clarification
    # request. The negative lookahead (?!\s+why\b) preserves the
    # short-clarification form ("I don't understand") and the
    # assistant-directed form ("I don't understand what you said")
    # while excluding narrative confusion that belongs in
    # psychoeducation. Regression pin: the
    # psychoeducation_grief_confusion case in therapeutic_routing_v0.json.
    r"\bi don'?t (?:understand|get it|follow)(?!\s+why\b)\b",
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


def _has_active_exercise(state: AgentState) -> bool:
    """Return whether a guided exercise is currently in progress.

    Checks ``state["progress"]["exercise_type"]`` and
    ``state["progress"]["exercise_step"]``. Both must be non-None for
    the exercise to count as active. The fields are cleared (set to
    None) by the guided_exercise node on both natural completion and
    user-initiated exit, so this check correctly returns False in
    both cases.

    This is the mechanism that keeps the dispatcher routing to
    guided_exercise across multiple turns without needing the LLM
    classifier to read history and infer "we're mid-exercise." See
    ``agent/therapeutic/guided_exercise.py`` for the state lifecycle.
    """

    progress = state.get("progress", {}) or {}
    return (
        progress.get("exercise_type") is not None
        and progress.get("exercise_step") is not None
    )


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
        "The modes are:\n"
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
        "- psychoeducation: short, normalizing framing. Use when the user "
        "DESCRIBES a specific reaction (a bodily sensation, an emotional "
        "response, a behavior they don't recognize in themselves) AND is "
        "seeking a frame for it. Both pieces must be present: a described "
        "experience AND a request for understanding. Examples: 'Why am I "
        "crying over this?' (described: crying; request: why), 'Is it "
        "normal to feel both angry and relieved?' (described: mixed "
        "emotions; request: is this normal), 'My heart starts racing for "
        "no reason — I don't get what's happening' (described: racing "
        "heart; request: what's happening). "
        "Counter-examples that should route to supportive: "
        "(a) bare self-reports without a 'help me understand' framing — "
        "'My chest feels tight', 'I feel angry', 'I can't sleep' — these "
        "are expressions, not questions about the reaction. "
        "(b) present-tense emotional expressions — 'I'm so angry right "
        "now', 'I cried again today and I hate it' — the user wants to be "
        "heard, not explained to. "
        "Counter-examples that should route to clarifying: "
        "(c) ambiguous confusion with NO described experience — 'It just "
        "doesn't make sense to me', 'I don't know what to think' — the "
        "agent doesn't know what 'it' refers to and needs to ask. "
        "Psychoeducation requires a concrete experience to frame; if the "
        "experience itself is unclear, route to clarifying first. "
        "Counter-examples that should route to reflective: "
        "(d) questions where the user is NAMING THEIR OWN pattern — 'I "
        "always apologize first in arguments, why?', 'I keep finding "
        "myself in the same fight' — the user has ALREADY identified the "
        "pattern and wants the agent to help them examine it. "
        "Distinguishing reflective from psychoeducation when behavior is "
        "involved: if the user is CONFUSED about their own behavior "
        "('I don't know why I'm so short with everyone lately') they want "
        "a FRAME — that's psychoeducation. If the user has NAMED the "
        "pattern themselves ('I always end up being the one who gives in') "
        "they want REFLECTION — that's reflective. The tell is whether "
        "the user is asking for understanding ('why am I doing this?') "
        "or inviting pattern exploration ('here's what I keep doing').\n"
        "- closing: short, warm farewell. Use ONLY when the user is "
        "explicitly signaling they're winding down or want to stop — "
        "'I should go', 'thanks, this helped', 'I need to step away', "
        "'I'm going to head out', 'I have to run'. The trigger is an "
        "explicit wind-down signal, not just a polite acknowledgment "
        "mid-conversation. A turn that says 'thanks, that helps' in the "
        "middle of a flowing conversation is NOT closing — it's a "
        "natural acknowledgment and the session continues, so route to "
        "supportive. Use closing only when the user is clearly leaving. "
        "False-positive closings ('oh, I thought you were done') are "
        "user-trust-damaging in a way that other false-positive mode "
        "choices aren't, so err toward supportive when uncertain.\n"
        "- guided_exercise: start a structured exercise. Use when the "
        "user explicitly asks for an exercise or technique — grounding, "
        "breathing, muscle relaxation, thought work, behavioral "
        "experiments, behavioral activation, acceptance/defusion, values "
        "work, self-compassion, emotion regulation, or gratitude. "
        "Trigger phrases include: 'ground me', 'breathing exercise', "
        "'help me calm down', 'relax my body', 'release tension', "
        "'let's do a thought record', 'examine this belief', 'test "
        "this thought', 'I can't start anything', 'I need to let go "
        "of this', 'leaves exercise', 'STOP technique', 'values "
        "compass', 'what matters to me', 'self-compassion', 'I'm so "
        "hard on myself', 'IMPROVE the moment', 'help me cope', "
        "'gratitude exercise', 'something I'm thankful for'. The "
        "trigger is a REQUEST for a structured intervention, not a "
        "general description of distress. "
        "Counter-examples that should route to supportive: "
        "'I can't calm down' (expressing distress, not asking for an "
        "exercise), 'I'm so anxious right now' (expressing, not "
        "requesting), 'nothing is helping me feel better' (expressing "
        "frustration). The distinction is: is the user asking the "
        "agent to DO something structured with them, or sharing how "
        "they feel? Only the former is guided_exercise. "
        "Counter-examples that should route to psychoeducation: "
        "'why does grounding even work?' (asking about the mechanism, "
        "not asking to do it). "
        "When uncertain, route to supportive — the user can always "
        "ask again more explicitly if they want the structured path.\n"
        "- clarifying: ask one focused question. Use only when the user's "
        "message is genuinely too ambiguous to respond to meaningfully "
        "(e.g., a bare 'ok' with no context, or an unclear pronoun reference "
        "to something the conversation hasn't covered), AND the user is not "
        "reporting a feeling or state. A short message like 'I feel sad' is "
        "a complete self-report and should NOT route to clarifying. A "
        "session-opening greeting is NOT clarifying territory — route those "
        "to supportive.\n\n"
        "Pick one mode. "
        "Additionally, pick the therapeutic modality that best fits this "
        "turn's content. The modality determines which therapeutic "
        "framework informs the response:\n"
        "- motivational_interviewing: user exploring change, ambivalence, "
        "autonomy, stuck between options\n"
        "- cbt: user examining thoughts, beliefs, cognitive patterns, "
        "wanting practical structure or behavioral change\n"
        "- act: user fighting or avoiding internal experiences, ruminating, "
        "needing acceptance or values reconnection\n"
        "- dbt_skills: user in acute emotional overwhelm, needing "
        "distress tolerance or emotion regulation skills\n"
        "- grief_support: user processing loss, bereavement, missing "
        "someone, anniversary reactions\n"
        "- interpersonal_therapy: user struggling with relationships, "
        "role transitions, communication breakdowns, loneliness\n"
        "- pfa: user in acute distress needing stabilization and "
        "practical support, not deep exploration\n"
        "- none: clarifying or closing turns, or when no specific "
        "modality fits better than the default\n\n"
        "Return your decision in the structured schema. "
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


async def _pick_mode_and_modality_with_llm(
    state: AgentState,
    llm_client,
) -> tuple[str, str]:
    """Call the structured-output LLM classifier to pick mode + modality.

    Returns ``(mode, modality)`` as strings. On any error, raises;
    callers are responsible for falling back to the regex pathway.
    """

    raw: DispatchDecision = await llm_client.generate_structured(
        prompt=build_therapeutic_dispatch_prompt(state),
        response_schema=DispatchDecision,
        system_instruction=build_therapeutic_dispatch_system_prompt(),
        temperature=0,
    )

    return raw.mode, raw.modality  # type: ignore[return-value]


# ─── Node-name mapping ─────────────────────────────────────────────────────

# Mapping from mode name → subgraph node name. Kept as a dict so the
# dispatcher's logic stays pure (pick_therapeutic_mode returns a name)
# and the routing layer does the name-to-node translation.
_MODE_NODE_MAP: dict[str, str] = {
    "supportive": SUPPORTIVE_NODE,
    "reflective": REFLECTIVE_NODE,
    "clarifying": CLARIFYING_NODE,
    "psychoeducation": PSYCHOEDUCATION_NODE,
    "closing": CLOSING_NODE,
    "guided_exercise": GUIDED_EXERCISE_NODE,
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
        "psychoeducation_response_node",
        "closing_response_node",
        "guided_exercise_response_node",
    ]
]:
    """Dispatch the turn to the right therapeutic mode node.

    Hybrid dispatch with an active-exercise override layered on top:

    1. Active-exercise fast path — if ``progress.exercise_type`` and
       ``progress.exercise_step`` are both set, a multi-turn exercise
       is in progress and the user's message is a step response.
       Route directly to ``guided_exercise_response_node`` without
       classifying the message. This is the mechanism that keeps
       multi-turn exercises coherent across turns (v0.6 Stage C).
    2. Regex fast paths — high-precision patterns for reflective and
       clarifying modes.
    3. LLM classifier — for everything else that doesn't match a
       fast path, the LLM picks one of the six modes.
    4. Regex fallback — when no LLM client is available, or the LLM
       call errored.

    The update dict is empty — the individual mode nodes are responsible
    for setting ``routing.mode``, ``routing.mode_source``, and
    ``routing.mode_type`` in their own deltas.
    """

    message = state.get("message", "")
    lowered = message.lower()
    llm_client = runtime.context.get("llm_client")

    # Helper to build routing update with modality.
    def _routing_update(modality: str) -> dict:
        return {"routing": {**state.get("routing", {}), "modality": modality}}

    # ── Fast path 0: active multi-turn exercise ──────────────────────────
    # Preserve the modality from the turn that started the exercise
    # rather than overwriting with "none". The first turn picks the
    # modality via the LLM classifier; continuation turns keep it.
    if _has_active_exercise(state):
        logger.debug("therapeutic_dispatch: active-exercise fast path")
        existing_modality = state.get("routing", {}).get("modality") or "none"
        return Command(
            update=_routing_update(existing_modality), goto=GUIDED_EXERCISE_NODE
        )

    # ── Fast path 1: high-precision reflective patterns ─────────────────
    if _matches_any(lowered, REFLECTIVE_PATTERNS):
        logger.debug("therapeutic_dispatch: reflective fast path")
        return Command(
            update=_routing_update("interpersonal_therapy"),
            goto=REFLECTIVE_NODE,
        )

    # ── Fast path 2: explicit confusion markers ──────────────────────────
    if _matches_any(lowered, CONFUSION_PATTERNS):
        logger.debug("therapeutic_dispatch: clarifying confusion-marker fast path")
        return Command(update=_routing_update("none"), goto=CLARIFYING_NODE)

    # ── LLM classifier path ──────────────────────────────────────────────
    if llm_client is not None:
        try:
            mode, modality = await _pick_mode_and_modality_with_llm(state, llm_client)
            logger.debug(
                "therapeutic_dispatch: LLM picked mode=%s modality=%s",
                mode,
                modality,
            )
            return Command(
                update=_routing_update(modality),
                goto=_MODE_NODE_MAP[mode],
            )
        except Exception:
            logger.warning(
                "therapeutic_dispatch LLM classifier failed; falling back to regex.",
                exc_info=True,
            )

    # ── Regex fallback ───────────────────────────────────────────────────
    mode = pick_therapeutic_mode(message)
    logger.debug("therapeutic_dispatch: regex fallback picked mode=%s", mode)
    # Default modality for regex: MI for supportive, none for others.
    fallback_modality = "motivational_interviewing" if mode == "supportive" else "none"
    return Command(
        update=_routing_update(fallback_modality),
        goto=_MODE_NODE_MAP[mode],
    )

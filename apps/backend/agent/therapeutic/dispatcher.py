"""Therapeutic dispatch node — picks which mode handles the current turn.

Uses an **LLM-primary** architecture: the LLM structured-output
classifier is the default decision path for all routing, including
mid-exercise continuation/exit, closing detection, and mode/modality
selection. Regex patterns are demoted to fallback-only, used when no
LLM client is available or the LLM call fails.

Dispatch flow:
    1. Exercise exit overrides (deterministic) — narrow, high-precision
       regex for unambiguous exercise opt-out ("quit", "cancel",
       "never mind"). Fire before the LLM to honor clear exit intent
       instantly.
    2. LLM classifier (primary) — handles all other routing. Picks one
       of six modes plus a therapeutic modality. For mid-exercise turns,
       the LLM sees exercise context in the prompt and decides whether
       to continue or exit.
    3. Regex fallback (degraded) — when no LLM client is available or
       the call fails. Uses narrow reflective/confusion patterns and
       defaults to supportive.

This mirrors the LLM-primary pattern in ``agent/nodes/crisis_gate.py``:
deterministic overrides for critical boundary cases, LLM as primary
classifier, regex as graceful degradation.

The dispatcher returns ``Command(goto=<node_name>)`` with a
``therapeutic_approach`` update. The individual response style nodes
set ``response_style``, ``response_style_source``, and
``response_style_type`` in their own deltas.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, TypeAlias

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.memory.models import DispatchDecision
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.working_memory import format_working_memory_entries

logger = logging.getLogger(__name__)


TherapeuticNodeName: TypeAlias = Literal[
    "supportive_response_node",
    "reflective_response_node",
    "clarifying_response_node",
    "psychoeducation_response_node",
    "closing_response_node",
    "guided_exercise_response_node",
    "technique_response_node",
]

SUPPORTIVE_NODE: TherapeuticNodeName = "supportive_response_node"
REFLECTIVE_NODE: TherapeuticNodeName = "reflective_response_node"
CLARIFYING_NODE: TherapeuticNodeName = "clarifying_response_node"
PSYCHOEDUCATION_NODE: TherapeuticNodeName = "psychoeducation_response_node"
CLOSING_NODE: TherapeuticNodeName = "closing_response_node"
GUIDED_EXERCISE_NODE: TherapeuticNodeName = "guided_exercise_response_node"
TECHNIQUE_NODE: TherapeuticNodeName = "technique_response_node"


# Reflective fallback patterns — ONLY for forms where the user is
# explicitly naming their OWN recurring behavioral pattern with
# first-person framing + repetition marker. Broader patterns like
# "same thing" and "is there a pattern" are demoted to the LLM
# classifier because they produce harmful false positives on
# non-pattern contexts. "Every time I" is kept only when followed by a
# second first-person consequence marker:
#   - "Every time I try your grounding exercise it helps" → NOT reflective
#   - "Is there a pattern to panic attacks?" → psychoeducation, not reflective
#   - "Same thing my therapist said" → NOT reflective
#
# The LLM dispatcher prompt already has the right reflective/psychoeducation
# distinction, so demoting broad patterns to LLM-primary improves accuracy
# without losing recall.
REFLECTIVE_PATTERNS: tuple[str, ...] = (
    r"\bwhy do(?:es)? (?:i|this|it) keep\b",
    r"\bwhy does (?:this|it) (?:keep|always) happen\b",
    r"\bevery time i\b.{0,80}\bi (?:end up|start|keep|always)\b",
    # "I always ... doing/saying/feeling/ending up/end up" — accepts
    # both the present participle ("ending up apologizing") and the
    # bare infinitive ("end up apologizing").
    r"\bi (?:always|keep)\b.{0,20}\b(?:doing|saying|feeling|ending up|end up)\b",
    r"\bthis (?:keeps|always) happen(?:ing|s)\b",
    r"\bi('m| am) stuck in (?:this|the same|a) (?:pattern|cycle|loop)\b",
    r"\bi notice (?:i|myself) (?:always|keep|often)\b",
)

# Explicit confusion markers — ONLY ultra-short, unambiguous,
# assistant-directed signals. Broader patterns like "can you explain",
# broad "I don't understand" and "I'm confused" forms are demoted to
# the LLM classifier because they produce harmful false positives on
# psychoeducation requests. Assistant-directed variants stay here:
#   - "Can you explain why my body does this?" → psychoeducation, not clarifying
#   - "I don't understand what's happening in my body" → psychoeducation
#   - "I'm confused about why my chest gets tight" → psychoeducation
#
# The LLM dispatcher prompt already has the right clarifying/psychoeducation
# distinction (including the "confusion about reaction" vs "assistant-directed
# confusion" boundary), so demoting broad patterns improves accuracy.
CONFUSION_PATTERNS: tuple[str, ...] = (
    r"^\s*huh\??\s*$",
    r"^\s*what\??\s*$",
    r"^\s*what do you mean\b",
    r"^\s*i don'?t understand what (?:you'?re|you are) getting at\b",
)

# Explicit exit signals for mid-exercise turns. These fire ONLY when
# an exercise is active (checked by the caller), so they're scoped to
# exercise context. Patterns are tightened to avoid false positives on
# non-exit language used during exercises:
#   - "I need to stop spiraling" → NOT an exit request, user is processing
#   - "I don't want to feel this way anymore" → NOT an exit, expressing distress
#   - "stop, let me think" → NOT an exit, user is pausing within the exercise
#
# The patterns now require either a clear opt-out verb ("quit", "cancel",
# "never mind") or an explicit "don't want to do this/continue" framing.
# Deterministic exercise exit patterns — ONLY unambiguous opt-out
# signals that should fire instantly without waiting for the LLM.
# All other exit signals (ambiguous stops, soft wrap-ups, topic changes)
# are handled by the LLM classifier, which sees exercise context and
# can distinguish "I want to stop the exercise" from "I need to stop
# spiraling" or "I should quit my job."
EXERCISE_EXIT_PATTERNS: tuple[str, ...] = (
    r"\bnever[\s-]?mind\b",
    r"\bnvm\b",
    r"\b(?:stop|end|skip|quit|cancel) (?:the |this )?(?:exercise|activity|technique)\b",
    r"\b(?:can|could) we just talk\b",
    # Bare "quit", "cancel", "stop" are intentionally excluded — they
    # match non-exit content like "I should quit my job" or "I had to
    # cancel dinner" or "I need to stop spiraling."
    # Bare "I don't want to" is excluded — it matches "I don't want to
    # feel this way anymore" which is distress, not an exit request.
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

COPING_ADVICE_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\b(?:what are|what're|what is|what's|give me|can you give me|could you give me|do you have|any)\b.{0,80}\b(?:tips|strategies|ways|ideas|options|suggestions|advice|skills|tools)\b",
    r"\b(?:tips|strategies|ways|ideas|options|suggestions|advice|skills|tools)\b.{0,80}\b(?:cope|coping|manage|handle|deal with|calm|anxiety|panic|stress|overwhelm)\b",
    r"\bhow (?:can|do|should) i\b.{0,80}\b(?:cope|manage|handle|deal with|calm|respond)\b",
    r"\bwhat (?:can|should) i do\b.{0,80}\b(?:when|if|about|for|to)\b",
    r"\bdifferent severity levels\b",
)

# Bare "walk|guide me through" was removed in favor of the noun-gated
# WALKTHROUGH_CONSENT_PATTERN below — bare walkthrough requests can be
# informational ("walk me through why this happens") and must not count as
# exercise consent on their own.
EXPLICIT_EXERCISE_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\b(?:let'?s|can we|could we|shall we|help me)\b.{0,60}\b(?:do|try|practice|start|begin)\b",
    r"\b(?:do|try|practice|start|begin)\b.{0,50}\b(?:exercise|grounding|breathing|skill|technique|thought record|behavioral experiment|values|defusion|self-compassion|gratitude|relaxation)\b",
    r"\b(?:grounding exercise|breathing exercise|box breathing|muscle relaxation|progressive muscle relaxation|thought record|behavioral experiment|values compass|leaves exercise|stop technique|improve the moment|gratitude exercise)\b",
    r"\bcan we do something\b",
    r"\bis there something we can do\b",
)


# ─── Canonical guided_exercise consent triggers ──────────────────────────────
#
# Single source of truth for triggers that appear verbatim in the dispatcher
# prompt's guided_exercise example list AND must be recognized by
# EXERCISE_CONSENT_PATTERNS. To add a trigger: append to this tuple. Both the
# rendered prompt sentence and the consent regex pick it up automatically.
# The contract test ``test_dispatcher_prompt_trigger_sentence_is_mechanically_rendered``
# enforces that the prompt prose cannot drift from this list.
_PROMPT_GUIDED_EXERCISE_TRIGGERS: tuple[str, ...] = (
    "ground me",
    "breathing exercise",
    "guide me through a grounding exercise",
    "let's do a thought record",
    "can we figure out a way to test it",
    "behavioral experiment",
    "can we look at what actually matters to me",
    "is there something we can do about that",
    "values compass",
    "leaves exercise",
    "STOP technique",
    "IMPROVE the moment",
    "gratitude exercise",
)


# Trailing words that complete an exercise noun phrase. If a tool noun is
# followed by something OTHER than end/punctuation/these completers, the user
# is talking ABOUT the noun ("grounding theory"), not asking to do it.
_NOUN_PHRASE_COMPLETERS: tuple[str, ...] = (
    "exercise",
    "practice",
    "technique",
    "session",
    "skill",
    "skills",
)


# Exercise/tool nouns recognized as the direct object of "walk/guide me through".
_WALKTHROUGH_NOUNS: tuple[str, ...] = (
    "grounding",
    "breathing",
    "thought record",
    "behavioral experiment",
    "values",
    "defusion",
    "self-compassion",
    "gratitude",
    "mindfulness",
    "exercise",
    "technique",
    "skill",
    "skills",
    "practice",
)


# Real terminator: end-of-message OR (whitespace +) punctuation.
_TERMINATOR = r"(?:[\s\.\,\?\!]*$|\s*[\.\,\?\!])"


# Walkthrough consent: "walk/guide me through (det/mod)? NOUN (completer)?
# TERMINATOR". The completer is consumed (not just lookahead) so an additional
# trailing word cannot sneak past as informational content. See
# UNCONSENTED_EXERCISE_FIX_PLAN.md Constraint 4 + Codex iter-11/12 for the
# rationale.
WALKTHROUGH_CONSENT_PATTERN = (
    rf"\b(?:walk|guide) me through "
    rf"(?:(?:a|an|the|some|my|that|this|your|your favorite|a short|a quick|my usual)\s+){{0,3}}"
    rf"(?:{'|'.join(re.escape(n) for n in _WALKTHROUGH_NOUNS)})"
    rf"(?:\s+(?:{'|'.join(_NOUN_PHRASE_COMPLETERS)}))?"
    rf"{_TERMINATOR}"
)


# How-to consent variant: "walk/guide me through how to (verb) NOUN".
WALKTHROUGH_HOWTO_CONSENT_PATTERN = (
    rf"\b(?:walk|guide) me through how to "
    rf"(?:do|use|practice|start|run|try|work through|go through|fill out|complete|begin|apply|get started with) "
    rf"(?:(?:a|an|the|some|my|that|this|your)\s+){{0,2}}"
    rf"(?:{'|'.join(re.escape(n) for n in _WALKTHROUGH_NOUNS)})\b"
)


# Informational walkthrough — wh-question form. Does not require a tool noun.
INFORMATIONAL_WALKTHROUGH_PATTERN = (
    r"\b(?:walk|guide) me through\b.{0,40}\b(?:why|what|how|whether)\b"
)


# Informational walkthrough — tool noun + non-terminating trailing content.
# Catches "walk me through grounding theory", "walk me through breathing
# problems", "walk me through grounding exercise theory". This pattern is NOT
# strictly disjoint from WALKTHROUGH_CONSENT_PATTERN; routing correctness comes
# from the consent-first ordering in
# _is_advice_request_without_exercise_consent. See Constraint 7 in
# UNCONSENTED_EXERCISE_FIX_PLAN.md.
INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN = (
    rf"\b(?:walk|guide) me through "
    rf"(?:(?:a|an|the|some|my|that|this|your)\s+){{0,3}}"
    rf"(?:{'|'.join(re.escape(n) for n in _WALKTHROUGH_NOUNS)})"
    rf"(?:\s+(?:{'|'.join(_NOUN_PHRASE_COMPLETERS)}))?"
    rf"\s+\w+"
)


def _trigger_to_regex(trigger: str) -> str:
    """Compile a literal trigger phrase to a word-bounded case-insensitive regex.

    Trailing 'it' is generalized to 'it/that/this/the X' for natural object
    substitution — e.g. "can we figure out a way to test it" also matches
    "can we figure out a way to test the thought".

    Args:
        trigger: The literal canonical trigger phrase.

    Returns:
        A regex string for case-insensitive substring matching.
    """

    base = re.escape(trigger.lower())
    if trigger.lower().endswith(" it"):
        base = re.escape(trigger.lower()[:-3]) + r"\s+(?:it|that|this|the\s+\w+)"
    return rf"\b{base}\b"


def _format_prompt_trigger_phrases() -> str:
    """Format the canonical trigger list as a quoted, comma-separated string."""

    return ", ".join(f"'{t}'" for t in _PROMPT_GUIDED_EXERCISE_TRIGGERS)


# Delimited canonical trigger sentence rendered into the dispatcher prompt.
# The HTML-comment delimiters are LLM-tolerated formatting and provide a
# deterministic span for the contract test
# ``test_dispatcher_prompt_trigger_sentence_is_mechanically_rendered``.
_TRIGGER_LIST_SENTENCE = (
    f"<!-- triggers:start -->Trigger phrases include: "
    f"{_format_prompt_trigger_phrases()}.<!-- triggers:end -->"
)


# Canonical exercise-consent regex set, derived from the explicit pattern set,
# the canonical trigger list, and the walkthrough consent patterns. Both the
# existing _is_coping_advice_without_exercise_consent guard and the new
# _is_advice_request_without_exercise_consent guard consult this set.
EXERCISE_CONSENT_PATTERNS: tuple[str, ...] = (
    *EXPLICIT_EXERCISE_REQUEST_PATTERNS,
    *(_trigger_to_regex(t) for t in _PROMPT_GUIDED_EXERCISE_TRIGGERS),
    WALKTHROUGH_CONSENT_PATTERN,
    WALKTHROUGH_HOWTO_CONSENT_PATTERN,
)


# Anaphoric guidance — bare "how do I X this/that/it/the pattern" questions
# whose object is a behavior pattern, not a content topic. Six branches; see
# UNCONSENTED_EXERCISE_FIX_PLAN.md Step 3f for design.
ANAPHORIC_GUIDANCE_PATTERNS: tuple[str, ...] = (
    # Branch 1: phrasal verb + bare pronoun, terminal/punctuation.
    # "break out of it", "get out of this", "snap out of that". Lookahead
    # blocks "out of this lease/relationship/college".
    r"\bhow (?:do|can|should|would|to)\s*(?:i|we|you)?\s*(?:\w+\s+){0,2}"
    r"(?:break (?:out of|free of|away from)|get (?:out of|away from|into)|"
    r"snap out of)"
    r"(?:\s+(?:doing|fighting|avoiding))?\s+"
    r"(?:this|that|it)(?=[\s\.\,\?\!]*$|\s*[\.\,\?\!])",
    # Branch 2: phrasal verb + pronoun + EXPLICIT pattern noun.
    # "get out of this loop", "break out of this cycle", "snap out of this spiral".
    r"\bhow (?:do|can|should|would|to)\s*(?:i|we|you)?\s*(?:\w+\s+){0,2}"
    r"(?:break (?:out of|free of|away from)|get (?:out of|away from|into)|"
    r"snap out of)"
    r"\s+(?:this|that|the)\s+(?:pattern|cycle|habit|behaviou?r|loop|spiral)\b",
    # Branch 3: bare verb + (optional gerund) + EXPLICIT pattern noun.
    # "break this cycle", "stop this pattern", "interrupt the loop".
    r"\bhow (?:do|can|should|would|to)\s*(?:i|we|you)?\s*(?:\w+\s+){0,2}"
    r"(?:break|stop|change|interrupt|fix|undo|escape)"
    r"(?:\s+(?:doing|fighting|avoiding))?\s+"
    r"(?:this|that|the)\s+(?:pattern|cycle|habit|behaviou?r|loop|spiral)\b",
    # Branch 4: "stop doing this/that/it" — terminal pronoun, narrow softeners
    # allowed.
    r"\bhow (?:do|can|should|would|to)\s*(?:i|we|you)?\s*(?:\w+\s+){0,2}"
    r"stop\s+doing\s+(?:this|that|it)"
    r"(?:\s+(?:in general|for good|anymore|once and for all))?"
    r"(?=[\s\.\,\?\!]*$|\s*[\.\,\?\!])",
    # Branch 5: "what do/can/should/would I do (about|with) this/that/it"
    r"\bwhat (?:do|can|should|would) i do (?:about|with) (?:this|that|it)\b",
    # Branch 6: "what now"
    r"\bwhat now\b\??\s*$",
)


# Acceptance regex set — end-anchored to the full message. An acknowledgment
# plus a new question (e.g. "yes, that makes sense, how do I stop this?") does
# NOT match acceptance.
ACCEPTANCE_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:yes|yeah|yep|sure|ok|okay|please|alright|absolutely)[\.\!]?\s*$",
    r"^\s*(?:yes|yeah|yep|sure|ok|okay)[,\s]+(?:please|let'?s(?:\s+(?:do|try)(?:\s+it)?)?|do it|try it)[\.\!]?\s*$",
    r"^\s*(?:let'?s|please)\s+(?:do|try)\s+(?:it|that)[\.\!]?\s*$",
    r"^\s*(?:i'?m|i am)\s+(?:ready|in|game)[\.\!]?\s*$",
    r"^\s*(?:go ahead|sounds good)[\.\!]?\s*$",
)


# Patterns that detect an assistant-side exercise offer in the prior turn.
EXERCISE_OFFER_PATTERNS: tuple[str, ...] = (
    r"\bwould you like to (?:try|do)\b.{0,40}\b(?:exercise|grounding|breathing|practice|technique|thought record|behavioral experiment|values|defusion|self-compassion)\b",
    r"\b(?:we could|i could)\b.{0,40}\b(?:try|do|walk through|practice|run through)\b.{0,40}\b(?:exercise|grounding|breathing|technique|thought record|behavioral experiment|values|defusion|self-compassion)\b",
    r"\bwant to (?:try|do)\b.{0,40}\b(?:grounding|breathing|exercise|technique|thought record|behavioral experiment)\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any regex pattern.

    Args:
        text: The text to test.
        patterns: The regex patterns to evaluate.

    Returns:
        ``True`` when any pattern matches ``text``.
    """

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _word_count(text: str) -> int:
    """Count words in a message.

    Args:
        text: The input text to tokenize.

    Returns:
        The number of word-like tokens in ``text``.
    """

    return len([w for w in re.findall(r"\w+", text) if w])


def _has_active_exercise(state: AgentState) -> bool:
    """Return whether an exercise is active in exercise state.

    Args:
        state: The current agent state.

    Returns:
        ``True`` when both ``exercise_type`` and ``exercise_step`` are set.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    return (
        exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )


def _has_pending_exercise_selection(state: AgentState) -> bool:
    """Return whether the user has pending guided-exercise options.

    Args:
        state: The current agent state.

    Returns:
        ``True`` when the prior guided-exercise turn offered selectable options.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    return bool(exercise_state.get("exercise_selection_options"))


def _looks_like_pending_exercise_choice(message: str) -> bool:
    """Return whether a message looks like an exercise-option choice.

    Args:
        message: The current user message.

    Returns:
        ``True`` for short numeric/ordinal choices or explicit exercise names.
    """

    lowered = message.lower().strip()
    if re.match(r"^(?:option\s*)?[1-9](?:[.)])?\s*$", lowered):
        return True
    return _matches_any(
        lowered,
        (
            r"\b(?:one|two|three|first|second|third)\b",
            r"\b(?:grounding|ground me|5-4-3-2-1)\b",
            r"\b(?:breath|breathe|breathing|box breathing)\b",
            r"\b(?:self.?compassion|compassion break|kinder to myself)\b",
            r"\b(?:thought record|thought check|belief)\b",
            r"\b(?:values|what matters|purpose|compass)\b",
            r"\b(?:gratitude|grateful|thankful)\b",
            r"\b(?:muscle|relaxation|pmr)\b",
            r"\b(?:stop technique|s\.t\.o\.p)\b",
            r"\b(?:improve|overwhelmed|too much)\b",
            r"\b(?:defusion|leaves|let go)\b",
            r"\b(?:behavioral experiment|test this belief)\b",
            r"\b(?:continuum|all.or.nothing)\b",
        ),
    )


def _active_exercise_modality(state: AgentState) -> str | None:
    """Return the pinned modality for an active exercise.

    Args:
        state: The current agent state.

    Returns:
        The exercise modality when present, otherwise the current
        turn's top-level ``therapeutic_approach``.
    """

    exercise_state = state.get("exercise_state", {}) or {}
    modality = exercise_state.get("exercise_modality")
    if modality:
        return modality
    return state.get("therapeutic_approach")


def _is_coping_advice_without_exercise_consent(message: str) -> bool:
    """Return whether a message asks for advice rather than guided practice.

    Args:
        message: The current user message.

    Returns:
        ``True`` when the user is asking for tips/options/strategies
        and has not explicitly opted into a structured exercise.
    """

    lowered = message.lower()
    return _matches_any(
        lowered,
        COPING_ADVICE_REQUEST_PATTERNS,
    ) and not _matches_any(lowered, EXERCISE_CONSENT_PATTERNS)


def _message_is_acceptance_of_offer(state: AgentState, message: str) -> bool:
    """Return whether the prior assistant turn offered an exercise AND the
    current message is a clean direct acceptance of that offer.

    Args:
        state: The current agent state, with conversation ``history``.
        message: The current user message.

    Returns:
        ``True`` when the most recent assistant turn contained an exercise
        offer and the current message matches an acceptance pattern. Returns
        ``False`` when the message is an acknowledgment plus a new question
        (which doesn't end-anchor to acceptance).
    """

    history = state.get("history", []) or []
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            offered = _matches_any(
                turn.get("content", "").lower(), EXERCISE_OFFER_PATTERNS
            )
            return offered and _matches_any(message.lower(), ACCEPTANCE_PATTERNS)
    return False


def _is_advice_request_without_exercise_consent(
    state: AgentState, message: str
) -> bool:
    """Return whether the message is a bare anaphoric advice request OR an
    informational walkthrough, with no consent signal.

    LOAD-BEARING ORDER: the consent check runs BEFORE the trigger checks.
    ``INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN`` deliberately overlaps with
    ``WALKTHROUGH_CONSENT_PATTERN`` on edge cases like
    "walk me through grounding exercise" — disjointness is not enforced at
    the regex level. Routing correctness depends on consent winning when both
    patterns could match.

    Reordering these checks would change observed behavior. See Constraint 7
    in ``UNCONSENTED_EXERCISE_FIX_PLAN.md``.

    Args:
        state: The current agent state, with conversation ``history``.
        message: The current user message.

    Returns:
        ``True`` when an LLM ``guided_exercise`` pick should be rewritten to
        psychoeducation; ``False`` when the LLM's pick should stand.
    """

    lowered = message.lower()

    # Consent FIRST — load-bearing.
    if _matches_any(lowered, EXERCISE_CONSENT_PATTERNS):
        return False
    if _has_active_exercise(state):
        return False
    if _message_is_acceptance_of_offer(state, message):
        return False

    # Trigger conditions: anaphoric guidance OR informational walkthrough.
    if _matches_any(lowered, ANAPHORIC_GUIDANCE_PATTERNS):
        return True
    if _matches_any(
        lowered,
        (INFORMATIONAL_WALKTHROUGH_PATTERN, INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN),
    ):
        return True
    return False


def pick_therapeutic_mode(
    message: str,
) -> Literal["supportive", "reflective", "clarifying"]:
    """Pick a fallback therapeutic mode from regex heuristics.

    Args:
        message: The current user message.

    Returns:
        The fallback mode name for regex-only dispatch.
    """

    lowered = message.lower()

    if _matches_any(lowered, REFLECTIVE_PATTERNS):
        return "reflective"

    if _matches_any(lowered, CONFUSION_PATTERNS):
        return "clarifying"

    is_short = _word_count(message) <= CLARIFYING_MAX_WORD_COUNT
    if is_short and not _matches_any(lowered, SELF_REPORT_PATTERNS):
        return "clarifying"

    return "supportive"


def build_therapeutic_dispatch_system_prompt() -> str:
    """Build the system prompt for the LLM dispatcher.

    Returns:
        The full system prompt string for mode and modality classification.
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
        "Also use psychoeducation, not guided_exercise, when the user asks "
        "for general tips, options, strategies, or severity-level guidance "
        "about coping and has not explicitly asked to practice one now. "
        "This is an informational request even if it mentions coping skills. "
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
        "- technique: the user wants structured therapeutic work, but is "
        "NOT asking to start a named exercise track. Use when the user wants "
        "to examine a thought, belief, or dilemma in a collaborative "
        "step-by-step way without launching a formal exercise like a thought "
        "record, behavioral experiment, or values clarification flow. The "
        "therapeutic_approach knowledge drives the response shape in this "
        "style. "
        "Signals that technique is right: the user has identified a "
        "specific thought and is ready to examine it, the user wants to "
        "look at evidence for and against a belief, the user wants help "
        "thinking through a belief from different angles, or the user "
        "wants collaborative therapeutic structure without asking to "
        "start a named tool. "
        "Signals that technique is wrong: the user is venting or "
        "expressing emotion (use supportive), the user is noticing a "
        "pattern but not ready to work on it (use reflective), the user "
        "is asking 'why does this happen?' (use psychoeducation), OR the "
        "user is explicitly asking to START a specific exercise track from "
        "the canonical guided_exercise trigger list above — those are "
        "guided_exercise turns because the agent should begin the matching "
        "stepwise exercise. Also do NOT use technique just because the user "
        "wants to 'talk it through' or remember what went better. If they "
        "are consolidating progress, naming strengths, or asking what they "
        "did differently in a hard moment, prefer supportive or reflective "
        "unless they explicitly ask for structured step-by-step thought work. "
        "Likewise, an opening disclosure like 'I keep avoiding work tasks "
        "because I get anxious and start spiraling before I even begin' is "
        "supportive or reflective, not technique. The agent can choose ACT "
        "as the therapeutic_approach for that turn without switching the "
        "response_style to technique. "
        "Technique requires an active therapeutic_approach — if no "
        "approach fits, do not use technique.\n"
        "- closing: short, warm farewell. Use ONLY when the user is "
        "explicitly signaling they're winding down or want to stop — "
        "'I should go', 'thanks, I need to head out', "
        "'I need to step away', 'I'm going to head out', "
        "'I have to run'. Also use closing "
        "when the user pairs wrap-up language with a takeaway request, "
        "such as 'before we wrap up, what's the main takeaway?', "
        "'what should I remember from this?', or 'put the main thing "
        "in one sentence'. The trigger is an explicit wind-down signal, "
        "not just a polite acknowledgment mid-conversation. Do NOT infer "
        "closing from thanks/helped language alone. A turn that says "
        "'thanks, that helps' in the middle of a flowing conversation is "
        "NOT closing — it's a natural acknowledgment and the session "
        "continues, so route to supportive. Use closing only when the user "
        "is clearly leaving, stopping, pausing, or wrapping up. "
        "False-positive closings ('oh, I thought you were done') are "
        "user-trust-damaging in a way that other false-positive mode "
        "choices aren't, so err toward supportive when uncertain.\n"
        "- guided_exercise: start a structured exercise. Use when the "
        "user explicitly asks for an exercise or technique — grounding, "
        "breathing, muscle relaxation, thought work, behavioral "
        "experiments, behavioral activation, acceptance/defusion, values "
        "work, self-compassion, emotion regulation, or gratitude. "
        f"{_TRIGGER_LIST_SENTENCE} The "
        "trigger is a REQUEST for a structured intervention, not a "
        "general description of distress. When the user names "
        "self-criticism or another concrete pain and then asks if there's "
        "something to be done about it, that 'do something about it' style "
        "trigger is consent to a self-compassion exercise. "
        "If the user is explicitly asking to START one of the supported "
        "exercise tracks, choose guided_exercise even if the content "
        "involves thought work, testing a belief, or values exploration. "
        "Those explicit starts belong here, not in technique. "
        "Counter-examples that should route to supportive: "
        "'I can't start anything' or a bare 'I'm so hard on myself' "
        "(expressing pain, not requesting an exercise), "
        "'I can't calm down' (expressing distress, not asking for an "
        "exercise), 'I'm so anxious right now' (expressing, not "
        "requesting), 'nothing is helping me feel better' (expressing "
        "frustration). The distinction is: is the user asking the "
        "agent to DO something structured with them, or sharing how "
        "they feel? Only the former is guided_exercise. "
        "If the user names self-criticism AND explicitly asks to do "
        "something about it together, that is a self-compassion exercise "
        "request and should route to guided_exercise. "
        "Counter-examples that should route to psychoeducation: "
        "'why does grounding even work?' (asking about the mechanism, "
        "not asking to do it), 'what are some tips to cope at different "
        "severity levels?' (asking for guidance/options, not to practice "
        "a skill now), 'how do I break out of this?', 'how do I stop "
        "doing this?', 'how do I break this cycle?', 'how do I get out "
        "of this loop?', 'what do I do about this?', 'what now?' (short "
        "anaphoric requests for guidance on changing a behavior or "
        "pattern — the user wants a frame or one or two options, not a "
        "structured exercise like grounding, breathing, or a thought "
        "record). "
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
        "Pick one response_style. "
        "The active therapeutic_approach from a prior turn does NOT, by "
        "itself, authorize starting that approach's named exercises. To "
        "pick guided_exercise, one of the following must hold: (i) the "
        "current user message contains a request to do something structured "
        "(matches the guided_exercise trigger phrases above), OR (ii) the "
        "assistant's previous turn explicitly offered a specific exercise "
        "AND the current user message is a clean direct acceptance ('yes', "
        "'yes please', 'sure', 'let's try it'). An acknowledgment-plus-"
        "question like 'yes, that makes sense, but how do I stop doing "
        "this?' is NOT an acceptance. If neither holds — for example, the "
        "prior modality is dbt_skills and the user asks 'how do I stop "
        "doing this' — route to psychoeducation, not guided_exercise.\n\n"
        "Additionally, pick the therapeutic_approach that best fits this "
        "turn's content. The therapeutic approach determines which "
        "framework informs the response:\n"
        "- motivational_interviewing: user exploring change, ambivalence, "
        "autonomy, stuck between options\n"
        "- cbt: user examining thoughts, beliefs, cognitive patterns, "
        "wanting practical structure or behavioral change. "
        "Concrete CBT signals: 'let's look at the evidence', 'I want "
        "to examine this thought', 'help me test this belief', 'what "
        "would be a realistic step'. The user wants to WORK ON the "
        "thought or behavior, not escape it.\n"
        "- act: user fighting or avoiding internal experiences, ruminating, "
        "needing acceptance or values reconnection. "
        "Concrete ACT signals: 'I keep fighting this feeling', 'the more "
        "I try to make it go away the worse it gets', 'I want to step "
        "back from this thought', 'I keep avoiding because of anxiety', "
        "'I'm exhausted from battling my own head', 'what do I do with "
        "this instead of fighting it'. The tell: the user is struggling "
        "WITH the experience itself — the avoidance, the rumination, "
        "or the control effort is the problem, not a specific distorted "
        "thought. If avoidance is driven by fighting internal states "
        "rather than wanting to restructure a belief, pick act over cbt.\n"
        "- dbt_skills: user in acute emotional overwhelm, needing "
        "distress tolerance or emotion regulation skills\n"
        "- grief_support: user processing loss, bereavement, missing "
        "someone, anniversary reactions\n"
        "- interpersonal_therapy: user struggling with relationships, "
        "role transitions, communication breakdowns, loneliness\n"
        "- pfa: user in acute distress needing stabilization and "
        "practical support, not deep exploration\n"
        "- none: clarifying or closing turns, or when no specific "
        "approach fits better than the default\n\n"
        "Return your decision in the structured schema. "
        "Keep the reasoning to one short sentence — it's for debugging, "
        "not for the user."
    )


def build_therapeutic_dispatch_prompt(state: AgentState) -> str:
    """Build the user prompt for the LLM dispatcher.

    Args:
        state: The current agent state.

    Returns:
        The user/task prompt containing recent history, memory, and the
        current message.
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

    working_memory = format_working_memory_entries(
        state.get("working_memory", []),
        limit=3,
    )
    if working_memory:
        memory_block = "Relevant context from past sessions:\n" + "\n".join(
            f"- {snippet}" for snippet in working_memory[:3]
        )
    else:
        memory_block = "(no working memory for this turn)"

    exercise_state = state.get("exercise_state", {}) or {}
    exercise_type = exercise_state.get("exercise_type")
    if exercise_type:
        exercise_block = (
            f"\nActive exercise: {exercise_type} "
            f"(step {exercise_state.get('exercise_step', '?')}). "
            "If the user is responding to the exercise, pick guided_exercise. "
            "If the user is exiting, wrapping up, or changing topic, pick the "
            "appropriate non-exercise mode.\n"
        )
    else:
        exercise_block = ""

    return (
        f"Recent conversation:\n{history_block}\n\n"
        f"{memory_block}\n"
        f"{exercise_block}\n"
        f"Current user message:\nuser: {state['message']}\n\n"
        "Which therapeutic mode should handle this turn?"
    )


async def _pick_mode_and_modality_with_llm(
    state: AgentState,
    llm_client,
) -> tuple[str, str]:
    """Call the structured-output classifier for mode and modality.

    Args:
        state: The current agent state.
        llm_client: The configured control-plane LLM client.

    Returns:
        A ``(mode, modality)`` tuple from the structured classifier response.

    Raises:
        Exception: Propagates any classifier error to the caller.
    """

    raw: DispatchDecision = await llm_client.generate_structured(
        prompt=build_therapeutic_dispatch_prompt(state),
        response_schema=DispatchDecision,
        system_instruction=build_therapeutic_dispatch_system_prompt(),
    )

    return raw.response_style, raw.therapeutic_approach  # type: ignore[return-value]


# Mapping from mode name → subgraph node name. Kept as a dict so the
# dispatcher's logic stays pure (pick_therapeutic_mode returns a name)
# and the routing layer does the name-to-node translation.
_MODE_NODE_MAP: dict[str, TherapeuticNodeName] = {
    "supportive": SUPPORTIVE_NODE,
    "reflective": REFLECTIVE_NODE,
    "clarifying": CLARIFYING_NODE,
    "psychoeducation": PSYCHOEDUCATION_NODE,
    "closing": CLOSING_NODE,
    "guided_exercise": GUIDED_EXERCISE_NODE,
    "technique": TECHNIQUE_NODE,
}


async def run_therapeutic_dispatch_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[TherapeuticNodeName]:
    """Route the current turn to the correct therapeutic response node.

    Args:
        state: The current agent state.
        runtime: The LangGraph runtime carrying injected dependencies.

    Returns:
        A ``Command`` pointing at the next therapeutic mode node, with any
        required routing or exercise-state updates.
    """

    message = state.get("message", "")
    lowered = message.lower()
    llm_client = runtime.context.llm_client

    def _routing_update(modality: str) -> dict:
        """Build the routing metadata update for the selected modality.

        Args:
            modality: Therapeutic approach chosen for this turn.

        Returns:
            State delta containing the top-level therapeutic approach.
        """

        return {"therapeutic_approach": modality}

    def _clear_active_exercise_update(modality: str) -> dict:
        """Build a routing update that also clears active exercise state.

        Args:
            modality: Therapeutic approach chosen for the non-exercise turn.

        Returns:
            State delta containing routing metadata and a cleared exercise state.
        """

        return {
            **_routing_update(modality),
            "exercise_state": {
                "exercise_type": None,
                "exercise_step": None,
                "exercise_modality": None,
                "exercise_selection_options": None,
            },
        }

    exercise_active = _has_active_exercise(state)
    exercise_selection_pending = _has_pending_exercise_selection(state)

    # Honor explicit exercise opt-outs without waiting for the LLM.
    if exercise_active and _matches_any(lowered, EXERCISE_EXIT_PATTERNS):
        logger.debug("therapeutic_dispatch: active-exercise exit override")
        return Command(
            update=_clear_active_exercise_update("none"),
            goto=SUPPORTIVE_NODE,
        )

    if exercise_selection_pending and _looks_like_pending_exercise_choice(message):
        logger.debug("therapeutic_dispatch: pending exercise selection choice")
        existing_modality = state.get("therapeutic_approach") or "none"
        return Command(
            update=_routing_update(existing_modality),
            goto=GUIDED_EXERCISE_NODE,
        )

    if llm_client is not None:
        try:
            mode, modality = await _pick_mode_and_modality_with_llm(state, llm_client)
            logger.debug(
                "therapeutic_dispatch: LLM picked mode=%s modality=%s",
                mode,
                modality,
            )

            if exercise_active:
                if mode == "guided_exercise":
                    existing_modality = _active_exercise_modality(state) or modality
                    return Command(
                        update=_routing_update(existing_modality),
                        goto=GUIDED_EXERCISE_NODE,
                    )

                if mode == "clarifying":
                    existing_modality = _active_exercise_modality(state) or modality
                    logger.debug(
                        "therapeutic_dispatch: mid-exercise clarifying "
                        "(exercise state preserved, modality=%s)",
                        existing_modality,
                    )
                    return Command(
                        update=_routing_update(existing_modality),
                        goto=_MODE_NODE_MAP["clarifying"],
                    )

                if mode == "psychoeducation":
                    logger.debug(
                        "therapeutic_dispatch: mid-exercise psychoeducation "
                        "(exercise state preserved, modality=%s)",
                        modality,
                    )
                    return Command(
                        update=_routing_update(modality),
                        goto=_MODE_NODE_MAP["psychoeducation"],
                    )

                logger.debug(
                    "therapeutic_dispatch: LLM exit from active exercise -> %s",
                    mode,
                )
                return Command(
                    update=_clear_active_exercise_update(modality),
                    goto=_MODE_NODE_MAP[mode],
                )

            if mode == "guided_exercise" and _is_coping_advice_without_exercise_consent(
                message
            ):
                logger.debug(
                    "therapeutic_dispatch: guided_exercise advice guard -> "
                    "psychoeducation"
                )
                return Command(
                    update=_routing_update(modality),
                    goto=PSYCHOEDUCATION_NODE,
                )

            if (
                mode == "guided_exercise"
                and _is_advice_request_without_exercise_consent(state, message)
            ):
                logger.debug(
                    "therapeutic_dispatch: anaphoric/walkthrough guidance guard -> "
                    "psychoeducation"
                )
                return Command(
                    update=_routing_update(modality),
                    goto=PSYCHOEDUCATION_NODE,
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

    # Without an LLM, active exercises continue unless a deterministic exit fired.
    if exercise_active:
        logger.debug("therapeutic_dispatch: regex fallback - continuing exercise")
        existing_modality = _active_exercise_modality(state) or "none"
        return Command(
            update=_routing_update(existing_modality), goto=GUIDED_EXERCISE_NODE
        )

    mode = pick_therapeutic_mode(message)
    logger.debug("therapeutic_dispatch: regex fallback picked mode=%s", mode)
    fallback_modality = "motivational_interviewing" if mode == "supportive" else "none"
    return Command(
        update=_routing_update(fallback_modality),
        goto=_MODE_NODE_MAP[mode],
    )

"""Regex constants and derived regexes for therapeutic dispatch."""

from __future__ import annotations

import re


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
    r"\b(?:can|could) we stop\b",
    r"\b(?:stop|end|skip|quit|cancel) (?:the |this )?(?:exercise|activity|technique)\b",
    r"\b(?:can|could) we just talk\b",
    r"\bi don'?t want to (?:do|continue) "
    r"(?:this|the exercise|the activity|the technique)\b",
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

_BARE_ACKNOWLEDGMENT_PATTERNS: tuple[str, ...] = (
    r"^\s*(?:yes|yeah|yep|yup|ok|okay|sure|mhm|mhmm)\s*[.!]?\s*$",
)

_OPEN_QUESTION_PATTERNS: tuple[str, ...] = (
    r"\b(?:what|what's|what is|how|why|where|when|who|which)\b.{0,120}\?",
    r"\b(?:tell me|say more|could you say more|can you say more)\b",
)

_ACTIVE_EXERCISE_CLARIFICATION_PATTERNS: tuple[str, ...] = (
    r"\bdo you mean\b",
    r"\bwhat do you mean\b",
    r"\bare you asking\b",
    r"\bam i supposed to\b",
    r"\bshould i\b",
    r"\bwhat counts\b",
    r"\b(?:right now|around me|in general)\b.{0,80}\?",
)

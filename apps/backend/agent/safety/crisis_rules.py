"""Deterministic crisis-classification policy and regex rules."""

from __future__ import annotations

import re
from typing import Literal

from agent.conversation import get_recent_history
from agent.models import CrisisAssessment
from agent.state import AgentState

# High-confidence immediate danger signals (plan/means/timing/finality).
IMMINENT_PATTERNS = (
    r"\b(?:tonight|today|right now|this (?:morning|afternoon|evening)|soon)\b.{0,35}\b(?:kill myself|end my life|end it|overdose|hang myself|jump)\b",
    r"\b(?:plan to|going to|gonna|will)\s+(?:kill myself|end my life|hurt myself|harm myself|overdose|jump)\b",
    r"\b(?:have|got)\s+(?:a|the|my)?\s*(?:gun|rope|knife|pills?|razor|blade)\b.{0,35}\b(?:for|to)\b.{0,20}\b(?:myself|end it|kill myself)\b",
    r"\b(?:this is my last|last message|goodbye everyone|you won't hear from me)\b",
    r"\b(?:swallow|take)\b.{0,20}\b(?:all|whole bottle|entire bottle)\b.{0,20}\b(?:pills?|tablets?)\b",
    # Location-based imminent signals can omit explicit self-harm verbs.
    r"\b(?:going to|headed to|driving to|walking to)\s+(?:the\s+)?(?:bridge|roof|railroad|tracks|river)\b",
    # Stockpiling language is urgent even before a self-harm verb appears.
    r"\b(?:saved up|stockpil(?:ed|ing)|collected|got enough)\s+(?:the\s+|my\s+|enough\s+)*(?:pills?|meds|medication|tablets?|sleeping pills?)\b",
    # Means plus proximate timing is enough for imminent classification.
    r"\b(?:gun|rope|knife|pills?|razor|blade|rifle|firearm|overdose)\b.{0,40}\b(?:tomorrow|tonight|today|this (?:morning|afternoon|evening|week)|soon|right now|when\b.{0,20}\bleaves?)\b",
    r"\b(?:tomorrow|tonight|today|this (?:morning|afternoon|evening|week)|soon|right now)\b.{0,40}\b(?:gun|rope|knife|pills?|razor|blade|rifle|firearm|overdose)\b",
)

# Explicit self-harm/suicidal language, including colloquialisms/evasion slang.
CLEAR_SELF_HARM_PATTERNS = (
    r"\b(?:kill myself|k[i1]ll m(?:yself|y self)|suicid(?:e|al)|end my life|want to die|die by suicide)\b",
    # Include gerund forms such as "hurting myself" and "harming myself".
    r"\b(?:hurt(?:ing)? myself|harm(?:ing)? myself|self[- ]?harm|cut(?:ting)? again|burn(?:ing)? myself)\b",
    r"\b(?:off myself|unalive myself|kms)\b",
    r"\b(?:don't want to be here anymore|nobody would miss me if i was gone|better off dead)\b",
    r"\b(?:go to sleep and (?:never|not) wake up|wish i didn't exist)\b",
    # Common evasions can omit the explicit "myself" object.
    r"\b(?:end(?:ing)? it(?: all)?|want it to be over|ready for it to end)\b",
    # "No reason to live" variants are explicit suicidal ideation.
    r"\b(?:no|any)\s+(?:reason|point)\s+(?:to|in)\s+liv(?:e|ing)\b",
    r"\b(?:nothing (?:left )?to live for|can't see a (?:future|reason))\b",
)

# Possible risk language requiring clarification.
AMBIGUOUS_PATTERNS = (
    r"\b(?:wish i could disappear|wish i wasn't here|wish i could go away)\b",
    r"\b(?:i can't do this anymore|i am done|i'm done|give up)\b",
    r"\b(?:dark thoughts|not sure i can keep going|can't keep going)\b",
    r"\b(?:i want to disappear|i don't want to exist)\b",
    r"\b(?:better off without me|i'm a burden|everyone would be better without me)\b",
    # Trapped-feeling language should clarify rather than escalate.
    r"\bno way out\b",
    # Passive "no point" framing is level 1 without explicit self-harm.
    r"\b(?:don't see the point|no point)\b.{0,30}\b(?:getting (?:out of )?bed|waking up|going on|anything|even trying)\b",
)

# Severe distress without explicit self-harm.
DISTRESS_PATTERNS = (
    r"\b(?:hopeless|empty|worthless|trapped|completely overwhelmed|breaking point)\b",
)

# Benign hyperbole and idioms to prevent false positives.
IDIOMATIC_SAFE_PATTERNS = (
    r"\b(?:work|school|homework|deadlines?)\s+(?:is|are)\s+killing me\b",
    r"\b(?:i'm|im)\s+dead(?:\s*$|\s+[😂🤣💀])",
    r"\b(?:dead tired|dying of laughter|bored to death|to die for|killing it)\b",
    r"\b(?:i could kill for)\b",
)

# Benign planning or sorting language that can resemble "I have a plan" when
# stripped from context, but is not self-harm content on its own.
BENIGN_PLANNING_SAFE_PATTERNS = (
    r"\bplanned it within my head\b",
    r"\bfigured it out\b",
    r"\bsorted it\b",
)

# Denials after a safety-check turn.
SAFETY_DENIAL_PATTERNS = (
    r"\bnot suicidal\b",
    r"\bnot thinking about\s+(?:self[- ]?harm|suicide|hurting myself|harming myself|killing myself|ending it)\b",
    r"\bnot\s+(?:planning|going|gonna|intending)\s+to\s+(?:hurt|harm|kill)\s+myself\b",
    r"\bno\s+(?:plan|intent|intention)\s+to\s+(?:hurt|harm|kill)\s+myself\b",
    r"\b(?:i'?m safe|i am safe|i would never do that|i won't do that|i won't do anything)\b",
    r"\b(?:just venting|figure of speech|didn't mean it literally)\b",
    # Require explicit negation-of-intent, not bare "no" + self-harm
    # co-occurrence. "No, I'm still thinking about ending it" confirms risk.
    r"\bno(?:,| -)?\s*(?:i'?m not|not)\s+(?:hurting myself|harming myself|killing myself|ending it|thinking about it)\b",
)


def _combined_user_text(state: AgentState) -> str:
    """Combine recent user turns into a lowercase text blob.

    Args:
        state: Current graph state.

    Returns:
        Lowercase text containing recent user turns plus the current message.
    """

    recent_user_turns = [
        turn["content"]
        for turn in get_recent_history(state, limit=6)
        if turn.get("role") == "user" and turn.get("content")
    ]
    recent_user_turns.append(state["message"])
    return " ".join(recent_user_turns).lower()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether text matches any regex pattern.

    Args:
        text: Text to search.
        patterns: Regex patterns to test.

    Returns:
        Whether any pattern matches.
    """

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _redact_matches(text: str, patterns: tuple[str, ...]) -> str:
    """Remove matched spans from text before follow-up pattern scans.

    Args:
        text: Text to redact.
        patterns: Regex patterns whose matched spans should be removed.

    Returns:
        Text with matched spans replaced by spaces.
    """

    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, " ", redacted, flags=re.IGNORECASE)
    return redacted


def _previous_mode_was_safety_check(state: AgentState) -> bool:
    """Return whether the most recent assistant turn appears to be a safety check.

    The phrase list is intentionally broad to catch LLM-generated variations.

    The crisis assessment state is reset each turn by ``build_initial_state``
    (it's not persisted via checkpoint merge), so state-based detection
    isn't viable.

    Args:
        state: Current graph state.

    Returns:
        Whether the latest assistant turn looks like a safety check.
    """

    for turn in reversed(get_recent_history(state, limit=4)):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "").lower()
        if any(
            phrase in content
            for phrase in (
                "thinking about hurting yourself",
                "thinking about harming yourself",
                "thoughts of hurting",
                "thoughts of harming",
                "thoughts of self-harm",
                "are you safe",
                "check on your safety",
                "having thoughts of hurting",
                "having thoughts of harming",
                "thoughts about ending",
                "thinking about ending",
                "hurting yourself",
                "harming yourself",
                "suicidal thought",
                "suicidal ideation",
                "your safety matters most right now",
                "move away from anything you could use to hurt yourself",
                "contact a trusted person who can stay with you",
                "local crisis line",
                "nearest emergency department",
            )
        ):
            return True
        break
    return False


def detect_crisis_override(
    state: AgentState,
) -> tuple[Literal["imminent_risk", "idiomatic_safe"], CrisisAssessment] | None:
    """Return hard overrides for obvious boundary cases.

    Imminent risk takes priority over safe idioms. Safe idioms only match
    the current message so old benign language cannot suppress new risk.

    Args:
        state: Current graph state.

    Returns:
        Override kind plus assessment, or ``None`` when normal classification
        should run.
    """

    text = _combined_user_text(state)
    current_message = state["message"].lower()

    if _matches_any(text, IMMINENT_PATTERNS):
        return (
            "imminent_risk",
            CrisisAssessment(
                level=3,
                confidence="high",
                reason="Detected imminent self-harm language with plan, means, timing, or finality cues.",
                needs_crisis_response=True,
                needs_clarification=False,
            ),
        )

    safe_patterns = IDIOMATIC_SAFE_PATTERNS + BENIGN_PLANNING_SAFE_PATTERNS
    if _matches_any(current_message, safe_patterns) and not (
        _matches_any(current_message, CLEAR_SELF_HARM_PATTERNS)
        or _matches_any(current_message, AMBIGUOUS_PATTERNS)
    ):
        return (
            "idiomatic_safe",
            CrisisAssessment(
                level=0,
                confidence="high",
                reason="Matched safe idiomatic or benign planning language without self-harm intent.",
                needs_crisis_response=False,
                needs_clarification=False,
            ),
        )

    if _previous_mode_was_safety_check(state):
        has_new_crisis = _matches_any(
            current_message, CLEAR_SELF_HARM_PATTERNS
        ) or _matches_any(current_message, IMMINENT_PATTERNS)
        has_new_ambiguous = _matches_any(current_message, AMBIGUOUS_PATTERNS)
        has_new_distress = _matches_any(current_message, DISTRESS_PATTERNS)
        has_denial = _matches_any(current_message, SAFETY_DENIAL_PATTERNS)

        if has_denial or (
            not has_new_crisis and not has_new_ambiguous and not has_new_distress
        ):
            return (
                "idiomatic_safe",
                CrisisAssessment(
                    level=0,
                    confidence="high",
                    reason=(
                        "User denied risk after a safety check."
                        if has_denial
                        else "No new crisis signals after a safety check."
                    ),
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )

    return None


def assess_crisis_risk_deterministically(state: AgentState) -> CrisisAssessment:
    """Assess crisis risk using deterministic rules.

    Args:
        state: Current graph state.

    Returns:
        Deterministic crisis assessment for the turn.
    """

    text = _combined_user_text(state)

    override = detect_crisis_override(state)
    if override is not None:
        _, assessment = override
        return assessment

    risk_text = _redact_matches(text, SAFETY_DENIAL_PATTERNS)

    if _matches_any(risk_text, CLEAR_SELF_HARM_PATTERNS):
        return CrisisAssessment(
            level=2,
            confidence="high",
            reason="Detected clear self-harm or suicidal ideation language.",
            needs_crisis_response=True,
            needs_clarification=False,
        )

    if _matches_any(risk_text, AMBIGUOUS_PATTERNS):
        return CrisisAssessment(
            level=1,
            confidence="medium",
            reason="Detected ambiguous self-harm-adjacent language requiring clarification.",
            needs_crisis_response=False,
            needs_clarification=True,
        )

    if _matches_any(text, DISTRESS_PATTERNS):
        return CrisisAssessment(
            level=1,
            confidence="medium",
            reason="Detected severe distress language without explicit self-harm signal.",
            needs_crisis_response=False,
            needs_clarification=True,
        )

    return CrisisAssessment(
        level=0,
        confidence="high",
        reason="No self-harm signal detected by deterministic rules.",
        needs_crisis_response=False,
        needs_clarification=False,
    )


__all__ = [
    "detect_crisis_override",
    "assess_crisis_risk_deterministically",
]

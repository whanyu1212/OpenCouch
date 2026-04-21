"""Hybrid crisis gate node for the OpenCouch graph."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

from agent.memory.models import CrisisClassifierPath
from agent.models import CrisisAssessment, ModeType, ResponseKind
from agent.prompts import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

OverrideKind = Literal["imminent_risk", "idiomatic_safe"]

# High-confidence immediate danger signals (plan/means/timing/finality).
IMMINENT_PATTERNS = (
    r"\b(?:tonight|today|right now|this (?:morning|afternoon|evening)|soon)\b.{0,35}\b(?:kill myself|end my life|end it|overdose|hang myself|jump)\b",
    r"\b(?:plan to|going to|gonna|will)\s+(?:kill myself|end my life|hurt myself|harm myself|overdose|jump)\b",
    r"\b(?:have|got)\s+(?:a|the|my)?\s*(?:gun|rope|knife|pills?|razor|blade)\b.{0,35}\b(?:for|to)\b.{0,20}\b(?:myself|end it|kill myself)\b",
    r"\b(?:this is my last|last message|goodbye everyone|you won't hear from me)\b",
    r"\b(?:swallow|take)\b.{0,20}\b(?:all|whole bottle|entire bottle)\b.{0,20}\b(?:pills?|tablets?)\b",
    # v0.5: "going to the bridge" + timing — location-based imminent signals.
    # The original `plan to|going to` pattern required explicit self-harm verbs
    # after it ("going to kill myself"), but "going to the bridge tonight" is
    # just as urgent without using the literal words. Locations associated
    # with suicide (bridge, roof, tracks, garage with car running) paired
    # with timing (tonight, tomorrow, right now) are high-specificity
    # imminent signals. Regression pin: level_3_bridge_tonight test case.
    r"\b(?:going to|headed to|driving to|walking to)\s+(?:the\s+)?(?:bridge|roof|railroad|tracks|river)\b",
    # v0.5: stockpiling + preparation language. "Saved up pills" + "wrote
    # letters" / "goodbye letters" is a textbook imminent-preparation
    # cluster per policy/crisis.md but wasn't captured in regex. This
    # pattern catches the stockpile half; the preparation half is in the
    # "last message / goodbye" pattern above. Regression pin:
    # level_3_saved_up_pills.
    r"\b(?:saved up|stockpil(?:ed|ing)|collected|got enough)\s+(?:the\s+|my\s+|enough\s+)*(?:pills?|meds|medication|tablets?|sleeping pills?)\b",
    # v0.5: means + proximate timing, generalized form. Catches
    # "gun tomorrow morning" and similar "means word + tomorrow/
    # tonight/today/soon/this week" combinations. The `have/got` pattern
    # above required a 'for/to myself' suffix which was too strict —
    # most real imminent disclosures don't add that suffix, they just
    # name the means and the time. Regression pin:
    # level_3_tomorrow_morning_gun.
    r"\b(?:gun|rope|knife|pills?|razor|blade|rifle|firearm|overdose)\b.{0,40}\b(?:tomorrow|tonight|today|this (?:morning|afternoon|evening|week)|soon|right now|when\b.{0,20}\bleaves?)\b",
    r"\b(?:tomorrow|tonight|today|this (?:morning|afternoon|evening|week)|soon|right now)\b.{0,40}\b(?:gun|rope|knife|pills?|razor|blade|rifle|firearm|overdose)\b",
)

# Explicit self-harm/suicidal language, including colloquialisms/evasion slang.
CLEAR_SELF_HARM_PATTERNS = (
    r"\b(?:kill myself|k[i1]ll m(?:yself|y self)|suicid(?:e|al)|end my life|want to die|die by suicide)\b",
    # v0.5: added gerund forms ("hurting myself", "harming myself") in
    # addition to the bare infinitive. Same bug class as the dispatcher's
    # Stage B "end up" / "ending up" gap. "I keep thinking about hurting
    # myself" was silently falling through to level 0 because the regex
    # only matched "hurt myself" exact. Regression pin:
    # level_2_hurting_myself_thought.
    r"\b(?:hurt(?:ing)? myself|harm(?:ing)? myself|self[- ]?harm|cut(?:ting)? again|burn(?:ing)? myself)\b",
    r"\b(?:off myself|unalive myself|kms)\b",
    r"\b(?:don't want to be here anymore|nobody would miss me if i was gone|better off dead)\b",
    r"\b(?:go to sleep and (?:never|not) wake up|wish i didn't exist)\b",
    # Suicidal ideation phrased as "ending it" or "wanting it to be over" — these
    # are common evasions that omit the explicit "myself" object but signal the
    # same ideation. Captured here so the deterministic tier flags them at L2.
    r"\b(?:end(?:ing)? it(?: all)?|want it to be over|ready for it to end)\b",
    # v0.5: "no reason to live" and close variants. This phrase is
    # explicitly listed in policy/crisis.md as a warning sign but wasn't
    # captured in regex. It's clearly level 2 content — stating there's
    # no reason to live is an explicit suicidal ideation disclosure.
    # Two sub-patterns:
    #  1. Direct "no/any reason/point to live" (covers "no reason to live",
    #     "don't see any reason to live", "there's no point in living").
    #  2. "nothing to live for" and the bleak "can't see a future".
    # Regression pin: level_2_no_reason_to_live_explicit ("I don't see any
    # reason to live anymore") and the canonical "no reason to live".
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
    # v0.5: "no way out" is a trapped-feeling warning sign per
    # policy/crisis.md but wasn't captured. "It feels like there's no way
    # out of this" is ambiguous enough that we clarify rather than
    # escalate to crisis response — but we must at least clarify.
    # Regression pin: level_1_no_way_out.
    r"\bno way out\b",
    # v0.5: passive anhedonia with "no point" framing — "don't see the
    # point in getting out of bed", "no point in (doing) anything". These
    # are warning signs but without explicit self-harm they're level 1,
    # not level 2. Regression pin: level_1_no_reason_to_wake_up.
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

# Denials after a safety-check turn.
SAFETY_DENIAL_PATTERNS = (
    r"\bnot suicidal\b",
    # Require explicit self-harm object after "not thinking about".
    r"\bnot thinking about\s+(?:self[- ]?harm|suicide|hurting myself|harming myself|killing myself|ending it)\b",
    r"\b(?:i'?m safe|i am safe|i would never do that|i won't do that|i won't do anything)\b",
    r"\b(?:just venting|figure of speech|didn't mean it literally)\b",
    # Require explicit negation-of-intent, not bare "no" + self-harm
    # co-occurrence. "No, I'm still thinking about ending it" must NOT
    # match — that's confirming risk, not denying it.
    r"\bno(?:,| -)?\s*(?:i'?m not|not)\s+(?:hurting myself|harming myself|killing myself|ending it|thinking about it)\b",
)


class CrisisAssessmentSchema(BaseModel):
    """Structured schema for crisis-classification model output."""

    level: int
    confidence: str
    reason: str
    needs_crisis_response: bool
    needs_clarification: bool


def _combined_user_text(state: AgentState) -> str:
    """Combine recent user turns into a single lowercase text blob."""

    recent_user_turns = [
        turn["content"]
        for turn in state.get("history", [])[-6:]
        if turn.get("role") == "user" and turn.get("content")
    ]
    recent_user_turns.append(state["message"])
    return " ".join(recent_user_turns).lower()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any pattern in the provided tuple."""

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _previous_mode_was_safety_check(state: AgentState) -> bool:
    """Return whether the most recent assistant turn appears to be a safety check.

    Uses two detection strategies:

    1. **User denial in current message**: if the user's current message
       contains safety-denial language AND the assistant's prior turn
       contained safety-related language, we're in a post-safety-check
       context.
    2. **Phrase-based**: check the assistant's most recent response for
       known safety-check phrases. The list is intentionally broad to
       catch LLM-generated variations.

    The crisis assessment state is reset each turn by ``build_initial_state``
    (it's not persisted via checkpoint merge), so state-based detection
    isn't viable.
    """

    history = state.get("history", [])
    for turn in reversed(history[-4:]):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "").lower()
        # Narrow to safety-specific phrases only. Broad phrases like
        # "want to check" and "need to ask" can false-positive on
        # normal conversation and create incorrect post-denial context.
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
            )
        ):
            return True
        break
    return False


def detect_crisis_override(
    state: AgentState,
) -> tuple[OverrideKind, CrisisAssessment] | None:
    """Return hard overrides for obvious boundary cases.

    Override precedence (highest to lowest):
    1. Imminent risk — ALWAYS wins. A prior idiom must never suppress
       a current imminent disclosure.
    2. Idiomatic safe — only matches the CURRENT message, not combined
       history. This prevents a prior "work is killing me" from
       suppressing crisis detection on a later message.
    3. Safety-denial context — post-safety-check de-escalation.
    """

    text = _combined_user_text(state)
    current_message = state["message"].lower()

    # Imminent risk checked FIRST — must never be suppressed by idioms.
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

    # Idiomatic safe checked on CURRENT message only — a prior idiom
    # in history must not suppress crisis detection on the current turn.
    # GUARD: do NOT return idiomatic-safe if the same message also
    # matches CLEAR_SELF_HARM or AMBIGUOUS patterns. A message like
    # "work is killing me and I want to kill myself" contains both an
    # idiom AND a real self-harm signal — the signal must win.
    if _matches_any(current_message, IDIOMATIC_SAFE_PATTERNS) and not (
        _matches_any(current_message, CLEAR_SELF_HARM_PATTERNS)
        or _matches_any(current_message, AMBIGUOUS_PATTERNS)
    ):
        return (
            "idiomatic_safe",
            CrisisAssessment(
                level=0,
                confidence="high",
                reason="Matched common idiomatic language without self-harm intent.",
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
    """Assess crisis risk using deterministic rules."""

    text = _combined_user_text(state)

    override = detect_crisis_override(state)
    if override is not None:
        _, assessment = override
        return assessment

    if _matches_any(text, CLEAR_SELF_HARM_PATTERNS):
        return CrisisAssessment(
            level=2,
            confidence="high",
            reason="Detected clear self-harm or suicidal ideation language.",
            needs_crisis_response=True,
            needs_clarification=False,
        )

    if _matches_any(text, AMBIGUOUS_PATTERNS):
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


async def assess_crisis_risk_with_llm(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> CrisisAssessment:
    """Assess crisis risk with a structured LLM classifier."""

    raw = await llm_client.generate_structured(
        prompt=build_crisis_classifier_prompt(state),
        response_schema=CrisisAssessmentSchema,
        system_instruction=build_crisis_classifier_system_prompt(),
    )

    level = max(0, min(3, int(raw.level)))
    confidence = (
        raw.confidence if raw.confidence in {"low", "medium", "high"} else "medium"
    )

    # Let normalize_crisis_assessment enforce the truth table rather
    # than trusting the LLM's raw flag values.
    return CrisisAssessment(
        level=level,
        confidence=confidence,
        reason=raw.reason,
        needs_crisis_response=level >= 2,
        needs_clarification=level == 1,
    )


def normalize_crisis_assessment(assessment: CrisisAssessment) -> CrisisAssessment:
    """Normalize crisis assessment fields to enforce the level/flag truth table.

    Truth table (enforced regardless of what the classifier returned):
        Level 0: needs_crisis_response=False, needs_clarification=False
        Level 1: needs_crisis_response=False, needs_clarification=True
        Level 2: needs_crisis_response=True,  needs_clarification=False
        Level 3: needs_crisis_response=True,  needs_clarification=False

    The classifier's own flag values are overridden when they violate
    this table. This prevents a miscalibrated LLM from setting
    needs_crisis_response=True on a level-0 message, or forgetting
    needs_clarification on a level-1 message.
    """

    level = max(0, min(3, int(assessment.level)))
    confidence = (
        assessment.confidence
        if assessment.confidence in {"low", "medium", "high"}
        else "medium"
    )

    # Enforce the truth table strictly.
    needs_crisis_response = level >= 2
    needs_clarification = level == 1

    return CrisisAssessment(
        level=level,
        confidence=confidence,
        reason=assessment.reason,
        needs_crisis_response=needs_crisis_response,
        needs_clarification=needs_clarification,
    )


def _build_crisis_delta(
    state: AgentState,
    assessment: CrisisAssessment,
    *,
    override_kind: Literal["imminent_risk", "idiomatic_safe", "none"],
    classifier_path: CrisisClassifierPath,
    llm_failure_occurred: bool,
    duration_ms: float,
    shadow_deterministic_level: int | None = None,
) -> dict[str, Any]:
    """Build the state-delta dict for one crisis-gate decision.

    Returns only the keys the gate updated: ``crisis``, ``routing`` (with
    the crisis debug-metadata fields so ``crisis_log_node`` can record an
    accurate audit trail), the crisis-tagged ``response.kind`` so
    downstream nodes can rely on the response slot already being marked,
    and a ``diagnostics`` entry with this turn's crisis-gate timing.

    The three debug-metadata kwargs are required (not defaulted) so each
    call site in :func:`run_crisis_gate_node` has to think explicitly
    about which code path it's on. Missing metadata would silently
    corrupt the safety audit log.
    """

    route = "crisis" if assessment.needs_crisis_response else "therapeutic"
    routing = state.get("routing", {})
    response = state.get("response", {})

    return {
        "crisis": assessment,
        "routing": {
            **routing,
            "route": route,
            "response_style": "safety_check"
            if route == "crisis"
            else routing.get("response_style"),
            "response_style_source": "crisis_gate",
            "response_style_type": ModeType.CRISIS
            if route == "crisis"
            else routing.get("response_style_type"),
            # Crisis debug metadata — read by crisis_log_node for the
            # audit record. Always populated on crisis-gate runs so the
            # audit trail reflects the actual code path taken.
            "crisis_override_kind": override_kind,
            "crisis_classifier_path": classifier_path,
            "crisis_llm_failure_occurred": llm_failure_occurred,
        },
        "response": {
            **response,
            "kind": ResponseKind.CRISIS if route == "crisis" else response.get("kind"),
        },
        # v0.8 observability: per-stage timing for the crisis gate.
        # The CLI renders this alongside load_memory_ms and the
        # other stage timings in the post-turn diagnostics panel.
        "diagnostics": {
            "crisis_gate_ms": round(duration_ms, 2),
            "crisis_classifier_path": classifier_path,
            "crisis_level": assessment.level,
            "crisis_shadow_deterministic_level": shadow_deterministic_level,
        },
    }


async def run_crisis_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[Literal["crisis_response_node", "load_memory_node"]]:
    """Run the crisis gate with LLM-primary, deterministic-fallback design.

    Decision flow:

    1. **Deterministic overrides** — imminent-risk (level 3 with
       plan/means/timing) and idiomatic-safe patterns fire instantly.
       These are too critical to wait for a network call.
    2. **LLM classifier** (primary) — for all other messages, the LLM
       evaluates crisis risk with full conversation context. This handles
       negation, sarcasm, quoted speech, and other nuances that regex
       cannot reliably parse.
    3. **Deterministic regex ladder** (fallback) — only used when the LLM
       provider is unavailable or the call fails. Provides degraded but
       functional safety coverage during outages.

    Returns a :class:`Command` that combines the assessment state update
    with the routing decision. Routes to ``crisis_response_node`` when
    ``needs_crisis_response`` is set; otherwise routes to
    ``load_memory_node`` for the therapeutic path.
    """

    llm_client = runtime.context.llm_client

    # v0.8 observability: time the whole gate call so the CLI can
    # render it in the post-turn diagnostics panel. The timer covers
    # both the override checks and the LLM/deterministic paths.
    gate_start = time.monotonic()

    # Debug metadata tracked across the decision tree. Every path below
    # MUST set all three before reaching _build_crisis_delta so the
    # safety audit log reflects the actual code path taken.
    override_kind: Literal["imminent_risk", "idiomatic_safe", "none"] = "none"
    classifier_path: CrisisClassifierPath
    llm_failure_occurred = False
    shadow_deterministic_level: int | None = None

    # ── Path 1: deterministic overrides (instant, no network) ────────
    # Imminent-risk and idiomatic-safe patterns are high-precision and
    # must fire before any LLM call. Imminent risk cannot wait 1-2s
    # for a network round-trip. Idiomatic-safe prevents false alarms
    # on "work is killing me" etc.
    override = detect_crisis_override(state)
    if override is not None:
        override_kind_detected, override_assessment = override
        override_kind = override_kind_detected
        classifier_path = "override"
        assessment = normalize_crisis_assessment(override_assessment)

    # ── Path 2: LLM classifier (primary) ─────────────────────────────
    # The LLM handles negation, context, sarcasm, quoted speech, and
    # all the nuances that regex cannot reliably parse. This is the
    # default path for all non-override messages.
    elif llm_client is not None:
        deterministic = assess_crisis_risk_deterministically(state)
        try:
            llm_assessment = await assess_crisis_risk_with_llm(
                state, llm_client=llm_client
            )
            classifier_path = "llm_primary"
            assessment = normalize_crisis_assessment(llm_assessment)

            # Shadow monitoring: compare LLM result against deterministic.
            # Log disagreements so drift can be detected. If deterministic
            # sees level >= 2 but LLM says level 0, that's a potential
            # false-negative worth investigating.
            shadow = normalize_crisis_assessment(deterministic)
            shadow_deterministic_level = shadow.level
            if shadow.level != assessment.level:
                logger.info(
                    "Crisis gate disagreement: LLM level=%d vs deterministic level=%d "
                    "(message=%r). LLM reason: %s",
                    assessment.level,
                    shadow.level,
                    state["message"][:100],
                    assessment.reason,
                )
            if shadow.level >= 2 and assessment.level == 0:
                logger.warning(
                    "Crisis gate: deterministic flagged level %d but LLM returned level 0. "
                    "Potential false negative. Message: %r",
                    shadow.level,
                    state["message"][:100],
                )
        except Exception:
            # LLM call failed — fall back to deterministic regex ladder.
            logger.warning(
                "Crisis LLM classifier failed; using deterministic fallback.",
                exc_info=True,
            )
            classifier_path = "deterministic"
            llm_failure_occurred = True
            assessment = normalize_crisis_assessment(deterministic)

    # ── Path 3: deterministic regex ladder (no LLM available) ────────
    # Graceful degradation: when no LLM client is configured, the
    # regex ladder provides functional (if less nuanced) safety
    # coverage. This preserves CI-time deterministic eval behavior.
    else:
        deterministic = assess_crisis_risk_deterministically(state)
        classifier_path = "deterministic"
        assessment = normalize_crisis_assessment(deterministic)

    gate_duration_ms = (time.monotonic() - gate_start) * 1000
    delta = _build_crisis_delta(
        state,
        assessment,
        override_kind=override_kind,
        classifier_path=classifier_path,
        llm_failure_occurred=llm_failure_occurred,
        duration_ms=gate_duration_ms,
        shadow_deterministic_level=shadow_deterministic_level,
    )
    next_node = (
        "crisis_response_node"
        if assessment.needs_crisis_response
        else "load_memory_node"
    )
    return Command(update=delta, goto=next_node)

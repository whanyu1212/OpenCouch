"""Hybrid crisis gate node for the OpenCouch graph."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel

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
    r"\b(?:plan to|going to|gonna|will)\s+(?:kill myself|end my life|hurt myself|overdose|jump)\b",
    r"\b(?:have|got)\s+(?:a|the|my)?\s*(?:gun|rope|knife|pills?|razor|blade)\b.{0,35}\b(?:for|to)\b.{0,20}\b(?:myself|end it|kill myself)\b",
    r"\b(?:this is my last|last message|goodbye everyone|you won't hear from me)\b",
    r"\b(?:swallow|take)\b.{0,20}\b(?:all|whole bottle|entire bottle)\b.{0,20}\b(?:pills?|tablets?)\b",
)

# Explicit self-harm/suicidal language, including colloquialisms/evasion slang.
CLEAR_SELF_HARM_PATTERNS = (
    r"\b(?:kill myself|k[i1]ll m(?:yself|y self)|suicid(?:e|al)|end my life|want to die|die by suicide)\b",
    r"\b(?:hurt myself|harm myself|self[- ]?harm|cut(?:ting)? again|burn myself)\b",
    r"\b(?:off myself|unalive myself|kms)\b",
    r"\b(?:don't want to be here anymore|nobody would miss me if i was gone|better off dead)\b",
    r"\b(?:go to sleep and (?:never|not) wake up|wish i didn't exist)\b",
    # Suicidal ideation phrased as "ending it" or "wanting it to be over" — these
    # are common evasions that omit the explicit "myself" object but signal the
    # same ideation. Captured here so the deterministic tier flags them at L2.
    r"\b(?:end(?:ing)? it(?: all)?|want it to be over|ready for it to end)\b",
)

# Possible risk language requiring clarification.
AMBIGUOUS_PATTERNS = (
    r"\b(?:wish i could disappear|wish i wasn't here|wish i could go away)\b",
    r"\b(?:i can't do this anymore|i am done|i'm done|give up)\b",
    r"\b(?:dark thoughts|not sure i can keep going|can't keep going)\b",
    r"\b(?:i want to disappear|i don't want to exist)\b",
    r"\b(?:better off without me|i'm a burden|everyone would be better without me)\b",
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
    r"\b(?:not suicidal|not thinking about(?:\s+self[- ]?harm|\s+suicide)?)\b",
    r"\b(?:i'?m safe|i am safe|i would never do that|i won't do that)\b",
    r"\b(?:just venting|figure of speech|didn't mean it literally)\b",
    r"\b(?:no)\b.{0,25}\b(?:hurting myself|harming myself|killing myself|ending it)\b",
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
    """Return whether the most recent assistant turn appears to be a safety check."""

    history = state.get("history", [])
    for turn in reversed(history[-4:]):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "").lower()
        if any(
            phrase in content
            for phrase in (
                "thinking about hurting yourself",
                "thinking about harming yourself",
                "are you safe right now",
                "check on your safety",
                "check something important",
            )
        ):
            return True
        break
    return False


def detect_crisis_override(
    state: AgentState,
) -> tuple[OverrideKind, CrisisAssessment] | None:
    """Return hard overrides for obvious boundary cases."""

    text = _combined_user_text(state)
    current_message = state["message"].lower()

    if _matches_any(text, IDIOMATIC_SAFE_PATTERNS):
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

    if _previous_mode_was_safety_check(state):
        has_new_crisis = _matches_any(
            current_message, CLEAR_SELF_HARM_PATTERNS
        ) or _matches_any(current_message, IMMINENT_PATTERNS)
        has_new_ambiguous = _matches_any(current_message, AMBIGUOUS_PATTERNS)
        has_denial = _matches_any(current_message, SAFETY_DENIAL_PATTERNS)

        if has_denial or (not has_new_crisis and not has_new_ambiguous):
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
        temperature=0,
    )

    level = max(0, min(3, int(raw.level)))
    confidence = (
        raw.confidence if raw.confidence in {"low", "medium", "high"} else "medium"
    )

    return CrisisAssessment(
        level=level,
        confidence=confidence,
        reason=raw.reason,
        needs_crisis_response=raw.needs_crisis_response,
        needs_clarification=raw.needs_clarification,
    )


def normalize_crisis_assessment(assessment: CrisisAssessment) -> CrisisAssessment:
    """Normalize crisis assessment fields into a consistent shape."""

    level = max(0, min(3, int(assessment.level)))
    confidence = (
        assessment.confidence
        if assessment.confidence in {"low", "medium", "high"}
        else "medium"
    )
    needs_crisis_response = assessment.needs_crisis_response or level >= 2
    needs_clarification = assessment.needs_clarification and not needs_crisis_response

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
    classifier_path: Literal["deterministic", "llm_fallback", "override"],
    llm_failure_occurred: bool,
) -> dict[str, Any]:
    """Build the state-delta dict for one crisis-gate decision.

    Returns only the keys the gate updated: ``crisis``, ``routing`` (with
    the crisis debug-metadata fields so ``crisis_log_node`` can record an
    accurate audit trail), and the crisis-tagged ``response.kind`` so
    downstream nodes can rely on the response slot already being marked.

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
            "mode": "safety_check" if route == "crisis" else routing.get("mode"),
            "mode_source": "crisis_gate",
            "mode_type": ModeType.CRISIS
            if route == "crisis"
            else routing.get("mode_type"),
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
    }


async def run_crisis_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[Literal["crisis_response_node", "therapeutic_subgraph"]]:
    """Run the hybrid crisis gate (deterministic + optional LLM fallback).

    Returns a :class:`Command` that combines the assessment state update
    with the routing decision in a single step. Routes to the crisis
    response node when the assessment marks ``needs_crisis_response``;
    otherwise routes to the therapeutic subgraph which picks the right
    response mode (supportive, reflective, or clarifying in v0.1).
    """

    llm_client = runtime.context.get("llm_client")

    # Debug metadata tracked across the decision tree. Every path below
    # MUST set all three before reaching _build_crisis_delta so the
    # safety audit log reflects the actual code path taken.
    override_kind: Literal["imminent_risk", "idiomatic_safe", "none"] = "none"
    classifier_path: Literal["deterministic", "llm_fallback", "override"]
    llm_failure_occurred = False

    override = detect_crisis_override(state)
    if override is not None:
        # Path 1: deterministic override — "imminent_risk" or "idiomatic_safe"
        override_kind_detected, override_assessment = override
        override_kind = override_kind_detected
        classifier_path = "override"
        assessment = normalize_crisis_assessment(override_assessment)
    else:
        deterministic = assess_crisis_risk_deterministically(state)
        if deterministic.level >= 2:
            # Path 2: deterministic ladder returned high confidence — skip LLM
            classifier_path = "deterministic"
            assessment = normalize_crisis_assessment(deterministic)
        elif llm_client is not None:
            try:
                llm_assessment = await assess_crisis_risk_with_llm(
                    state, llm_client=llm_client
                )
                # Path 3: LLM classifier succeeded
                classifier_path = "llm_fallback"
                assessment = normalize_crisis_assessment(llm_assessment)
            except Exception:
                # Path 4: LLM was called but raised — fall back to deterministic.
                # Classifier_path stays "deterministic" because that's what
                # we actually used; llm_failure_occurred distinguishes this
                # case from path 5 where the LLM was never called.
                logger.warning(
                    "Crisis LLM classifier failed; using deterministic fallback.",
                    exc_info=True,
                )
                classifier_path = "deterministic"
                llm_failure_occurred = True
                assessment = normalize_crisis_assessment(deterministic)
        else:
            # Path 5: no LLM client available — deterministic only
            classifier_path = "deterministic"
            assessment = normalize_crisis_assessment(deterministic)

    delta = _build_crisis_delta(
        state,
        assessment,
        override_kind=override_kind,
        classifier_path=classifier_path,
        llm_failure_occurred=llm_failure_occurred,
    )
    next_node = (
        "crisis_response_node"
        if assessment.needs_crisis_response
        else "therapeutic_subgraph"
    )
    return Command(update=delta, goto=next_node)

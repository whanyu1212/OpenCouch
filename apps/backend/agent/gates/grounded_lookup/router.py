"""Routing policy for explicit grounded factual lookup requests."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from agent.gates.grounded_lookup.patterns import (
    AMBIGUOUS_LOOKUP_SIGNAL_RE as _AMBIGUOUS_LOOKUP_SIGNAL_RE,
    CHECK_IF_RE as _CHECK_IF_RE,
    CURRENT_INFO_RE as _CURRENT_INFO_RE,
    LOOKUP_VERB_RE as _LOOKUP_VERB_RE,
    THERAPEUTIC_SUBJECTIVE_RE as _THERAPEUTIC_SUBJECTIVE_RE,
    VERIFY_RE as _VERIFY_RE,
)
from agent.gates.grounded_lookup.prompts import (
    build_grounded_lookup_prompt as _build_grounded_lookup_prompt,
    build_grounded_lookup_system_prompt as _build_grounded_lookup_system_prompt,
)
from agent.state import AgentState
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundedLookupAction:
    """Resolved grounded lookup action."""

    query: str


@dataclass(frozen=True)
class GroundedLookupRoute:
    """Resolved grounded lookup route decision."""

    action: GroundedLookupAction | None
    classifier_path: str
    llm_failure_occurred: bool
    reason: str
    confidence: str | None = None


class GroundedLookupDecision(BaseModel):
    """Structured output for grounded-lookup routing."""

    should_lookup: bool = Field(
        description="Whether the user is asking for external factual lookup."
    )
    query: str | None = Field(
        default=None,
        description="Search query to use when should_lookup is true.",
    )
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


def detect_grounded_lookup_action(message: str) -> GroundedLookupAction | None:
    """Detect an explicit factual/current lookup request.

    Args:
        message (str): Current user message.

    Returns:
        GroundedLookupAction | None: A resolved lookup action, or ``None`` for
            ordinary therapeutic routing.
    """

    stripped = message.strip()
    if not stripped:
        return None
    if _THERAPEUTIC_SUBJECTIVE_RE.search(stripped):
        return None

    has_lookup_verb = bool(_LOOKUP_VERB_RE.search(stripped))
    has_current_info = bool(_CURRENT_INFO_RE.search(stripped))
    has_check_or_verify = bool(
        _CHECK_IF_RE.search(stripped) or _VERIFY_RE.search(stripped)
    )
    has_numeric_or_url = bool(re.search(r"\d|https?://|www\.", stripped, re.I))
    is_question = stripped.endswith("?") or bool(
        re.match(r"\s*(what|which|where|how|is|are|can|do|does)\b", stripped, re.I)
    )

    if has_lookup_verb or (
        has_check_or_verify and (has_current_info or has_numeric_or_url)
    ):
        return GroundedLookupAction(query=stripped)
    if has_current_info and is_question:
        return GroundedLookupAction(query=stripped)
    return None


def _needs_lookup_classifier(message: str) -> bool:
    """Return whether a message is ambiguous enough to ask the classifier.

    Args:
        message (str): Current user message.

    Returns:
        bool: ``True`` when the message contains lookup-shaped factual signals
            that are not decisive enough for a hard deterministic route.
    """

    stripped = message.strip()
    if not stripped:
        return False
    if _THERAPEUTIC_SUBJECTIVE_RE.search(stripped):
        return False
    return bool(_AMBIGUOUS_LOOKUP_SIGNAL_RE.search(stripped))


async def _classify_grounded_lookup_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> tuple[GroundedLookupAction | None, str, str]:
    """Classify an ambiguous message as grounded lookup or ordinary support.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient): Configured control-plane LLM client.

    Returns:
        tuple[GroundedLookupAction | None, str, str]: Lookup action, classifier
            reasoning, and confidence.
    """

    decision: GroundedLookupDecision = await llm_client.generate_structured(
        prompt=_build_grounded_lookup_prompt(state),
        response_schema=GroundedLookupDecision,
        system_instruction=_build_grounded_lookup_system_prompt(),
    )

    if not decision.should_lookup or decision.confidence == "low":
        return None, decision.reasoning, decision.confidence

    query = (decision.query or "").strip() or state.get("message", "").strip()
    if not query:
        return None, decision.reasoning, decision.confidence
    return GroundedLookupAction(query=query), decision.reasoning, decision.confidence


async def resolve_grounded_lookup_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
) -> GroundedLookupRoute:
    """Resolve grounded lookup routing using hard rules plus LLM middle path.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient | None): Optional control-plane LLM client.

    Returns:
        GroundedLookupRoute: Resolved route decision with optional lookup
            action, classifier path, and LLM-failure flag.
    """

    message = state.get("message", "")
    hard_action = detect_grounded_lookup_action(message)
    if hard_action is not None:
        return GroundedLookupRoute(
            action=hard_action,
            classifier_path="deterministic",
            llm_failure_occurred=False,
            reason="Hard rule matched a factual lookup request.",
            confidence="high",
        )

    if not _needs_lookup_classifier(message):
        return GroundedLookupRoute(
            action=None,
            classifier_path="not_attempted",
            llm_failure_occurred=False,
            reason="No external factual lookup request detected.",
            confidence=None,
        )

    if llm_client is None:
        return GroundedLookupRoute(
            action=None,
            classifier_path="deterministic",
            llm_failure_occurred=False,
            reason="Grounded-lookup classifier unavailable; continuing normal route.",
            confidence=None,
        )

    try:
        action, reason, confidence = await _classify_grounded_lookup_action(
            state,
            llm_client=llm_client,
        )
    except Exception:
        logger.warning(
            "Grounded lookup LLM classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return GroundedLookupRoute(
            action=None,
            classifier_path="deterministic",
            llm_failure_occurred=True,
            reason="Grounded-lookup classifier failed; continuing normal route.",
            confidence=None,
        )

    return GroundedLookupRoute(
        action=action,
        classifier_path="llm_primary",
        llm_failure_occurred=False,
        reason=reason,
        confidence=confidence,
    )

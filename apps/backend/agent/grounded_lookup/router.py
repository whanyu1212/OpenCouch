"""Routing policy for explicit grounded factual lookup requests."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.conversation import format_recent_history
from agent.state import AgentState
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


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


_LOOKUP_VERB_RE = re.compile(
    r"\b(look up|search(?: for| online| the web)?|web search|google|"
    r"check official|check current|"
    r"find (?:official|current|verified|local|nearby|resources|services|"
    r"clinics|directories))\b",
    re.I,
)
_CHECK_IF_RE = re.compile(r"\bcan you check (?:if|whether)\b", re.I)
_VERIFY_RE = re.compile(r"\bverify(?: whether| if| that)?\b", re.I)
_CURRENT_INFO_RE = re.compile(
    r"\b(latest|current|up[- ]?to[- ]?date|still available|still works|"
    r"eligibility|official rules?|law|regulation|policy|price|cost|schedule)\b",
    re.I,
)
_THERAPEUTIC_SUBJECTIVE_RE = re.compile(
    r"\b(being unreasonable|overreacting|bad person|wrong for feeling|"
    r"should i feel|why do i feel|what does it mean that i|"
    r"is it normal to feel|am i wrong|am i bad)\b",
    re.I,
)
_AMBIGUOUS_LOOKUP_SIGNAL_RE = re.compile(
    r"\b(can you check|can you verify|verify whether|verify if|fact[- ]?check|"
    r"evidence[- ]?based|research|stud(?:y|ies)|clinical trials?|proven|"
    r"legit|reliable|source|sources|citation|citations|website|url|link|"
    r"wearables?|apps?|does .{0,40} work|is .{0,40} effective|"
    r"is .{0,40} safe)\b",
    re.I,
)


def _build_grounded_lookup_prompt(state: AgentState) -> str:
    """Build the LLM prompt for ambiguous grounded-lookup routing.

    Args:
        state (AgentState): Current graph state.

    Returns:
        str: Prompt asking for a structured lookup-routing decision.
    """

    recent_history = format_recent_history(state, limit=6, empty="(none)")
    return (
        "Decide whether the user's message should route to grounded web/current "
        "factual lookup before therapeutic response generation.\n\n"
        "Route to lookup only when the user is asking for external factual, "
        "current, official, research, evidence, price, eligibility, schedule, "
        "resource, URL, product, or service information that should be verified "
        "outside the conversation.\n\n"
        "Do not route to lookup for subjective therapeutic reassurance, emotional "
        "validation, relationship advice, or questions like 'am I overreacting?', "
        "'am I a bad person?', 'is it normal to feel this way?', or 'what should "
        "I do about this feeling?'.\n\n"
        "If lookup is needed, set should_lookup=true and provide a concise search "
        "query. If uncertain, set should_lookup=false.\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def _build_grounded_lookup_system_prompt() -> str:
    """Build the system prompt for the grounded-lookup classifier.

    Returns:
        str: System instruction for structured lookup routing.
    """

    return (
        "You are a strict routing classifier. Return only the structured "
        "decision. You do not answer the user."
    )


def detect_grounded_lookup_action(message: str) -> dict[str, Any] | None:
    """Detect an explicit factual/current lookup request.

    Args:
        message (str): Current user message.

    Returns:
        dict[str, Any] | None: A serializable lookup action, or ``None`` for
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
        return {"query": stripped}
    if has_current_info and is_question:
        return {"query": stripped}
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
) -> dict[str, Any] | None:
    """Classify an ambiguous message as grounded lookup or ordinary support.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient): Configured control-plane LLM client.

    Returns:
        dict[str, Any] | None: Grounded lookup action, or ``None`` when ordinary
            routing should handle the turn.
    """

    decision: GroundedLookupDecision = await llm_client.generate_structured(
        prompt=_build_grounded_lookup_prompt(state),
        response_schema=GroundedLookupDecision,
        system_instruction=_build_grounded_lookup_system_prompt(),
    )

    if not decision.should_lookup or decision.confidence == "low":
        return None

    query = (decision.query or "").strip() or state.get("message", "").strip()
    if not query:
        return None
    return {"query": query}


async def resolve_grounded_lookup_action(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Resolve grounded lookup routing using hard rules plus LLM middle path.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient | None): Optional control-plane LLM client.

    Returns:
        tuple[dict[str, Any] | None, str, bool]: Tuple of lookup action,
            classifier path, and whether an LLM failure occurred.
    """

    message = state.get("message", "")
    hard_action = detect_grounded_lookup_action(message)
    if hard_action is not None:
        return hard_action, "deterministic", False

    if not _needs_lookup_classifier(message):
        return None, "not_attempted", False

    if llm_client is None:
        return None, "deterministic", False

    try:
        action = await _classify_grounded_lookup_action(state, llm_client=llm_client)
    except Exception:
        logger.warning(
            "Grounded lookup LLM classifier failed; using deterministic fallback.",
            exc_info=True,
        )
        return None, "deterministic", True

    return action, "llm_primary", False

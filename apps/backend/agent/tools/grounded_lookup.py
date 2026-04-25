"""General grounded factual lookup helper."""

from __future__ import annotations

import logging
from typing import Literal

from agent.state import AgentState
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

GroundedLookupStatus = Literal[
    "not_attempted",
    "answered",
    "search_unavailable",
    "search_failed",
    "no_verified_answer",
]

_GROUNDING_SYSTEM = (
    "You answer explicit factual lookup requests for OpenCouch, a mental-health "
    "support agent. Use web search/grounding. Answer only the factual question "
    "the user asked. Prefer official, primary, or otherwise reputable sources. "
    "Do not invent facts, contact details, eligibility rules, prices, dates, or "
    "source names. If you cannot verify the answer, say that clearly. Keep the "
    "answer concise and include a short 'Sources:' section with source names or "
    "URLs when available."
)

_NO_VERIFIED_MARKERS = (
    "could not verify",
    "couldn't verify",
    "cannot verify",
    "can't verify",
    "unable to verify",
    "i don't know",
    "i do not know",
    "no verified",
    "not enough information",
)


async def answer_grounded_lookup(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
    query: str,
) -> tuple[str, GroundedLookupStatus]:
    """Answer one explicit factual lookup request with search grounding.

    Args:
        state: Current graph state. Recent history is included only to clarify
            references in the user's lookup request.
        llm_client: Provider client with search-grounded text generation.
        query: The factual/current-information request to answer.

    Returns:
        A ``(answer, status)`` tuple. ``answer`` is safe to show directly to the
        user when non-empty.
    """

    prompt = _build_grounded_lookup_prompt(state, query=query)
    try:
        raw = await llm_client.generate_text(
            prompt=prompt,
            system_instruction=_GROUNDING_SYSTEM,
            use_search=True,
        )
    except Exception:
        logger.warning("Grounded factual lookup failed.", exc_info=True)
        return "", "search_failed"

    answer = _normalize_grounded_answer(raw)
    if not answer:
        return "", "no_verified_answer"
    if _looks_unverified(answer):
        return answer, "no_verified_answer"
    return answer, "answered"


def _build_grounded_lookup_prompt(state: AgentState, *, query: str) -> str:
    """Build the user prompt for a grounded factual lookup.

    Args:
        state: Current graph state.
        query: User's factual lookup request.

    Returns:
        Prompt text for provider-native search grounding.
    """

    history = state.get("history", [])[-4:]
    history_text = "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in history
    )
    return (
        "The user explicitly asked for factual/current information. Use search "
        "grounding and answer only that request. Do not provide therapy advice, "
        "diagnosis, or crisis guidance in this answer. If the answer depends on "
        "location and the user did not provide one, say what location would be "
        "needed instead of guessing.\n\n"
        f"Recent conversation for reference:\n{history_text or '(none)'}\n\n"
        f"Lookup request:\n{query}"
    )


def _normalize_grounded_answer(raw: str) -> str:
    """Normalize a provider-grounded answer.

    Args:
        raw: Raw provider output.

    Returns:
        Cleaned answer text, or an empty string for unusable output.
    """

    answer = raw.strip()
    if not answer:
        return ""
    return "\n".join(line.rstrip() for line in answer.splitlines()).strip()


def _looks_unverified(answer: str) -> bool:
    """Return whether the answer says verification failed.

    Args:
        answer: Normalized provider answer.

    Returns:
        ``True`` when the answer is an explicit no-verified-answer response.
    """

    lowered = answer.lower()
    return any(marker in lowered for marker in _NO_VERIFIED_MARKERS)

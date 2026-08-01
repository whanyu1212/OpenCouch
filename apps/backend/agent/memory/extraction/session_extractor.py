"""Session-end memory extractor.

Runs ONCE at session end over the whole transcript, populating the
``SessionMemoryBuffer`` that ``commit_session_memory`` drains. This is the
intake front-half that was removed in the LangGraph teardown; it is rebuilt as a
whole-transcript pass (rather than the original per-turn pass) so durability can
be judged with the full session arc in view.

Flow: skip gates -> two structured LLM passes (semantic + procedural, run
concurrently, each degrading to empty on error) -> provenance attribution
(map each fact's evidence quote back to the user turn it came from) -> build
candidates -> hold on the buffer as ``commit_at_session_end``.

Extraction is a side-effect path: it must never raise out and break session
finalization.
"""

from __future__ import annotations

import asyncio
import logging
from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import (
    PolicyDecision,
    SessionMemoryBuffer,
    build_procedural_candidate,
    build_semantic_candidate,
)
from agent.memory.prompts.session_extraction import (
    build_session_procedural_system_prompt,
    build_session_procedural_user_prompt,
    build_session_semantic_system_prompt,
    build_session_semantic_user_prompt,
)
from agent.memory.types.procedural import (
    ProceduralExtractionResult,
    ProceduralRuleDraft,
)
from agent.memory.types.semantic import ExtractionResult, MemoryWrite
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# All session-end candidates are held with this decision: by definition we are
# already at session end, so the per-candidate "when to commit" timing decision
# the old write-policy classifier made is moot. Durability is judged by the
# extractor prompt (whole-arc context) and re-gated by commit-time corroboration
# scoring; the dropped decide_*_llm_primary classifier is not reintroduced.
_SESSION_END_DECISION = PolicyDecision(
    action="commit_at_session_end",
    reason="session-end extraction",
    policy_version="session_extract_v1",
)


async def extract_session_candidates(
    *,
    user_turn_texts: list[str],
    session_id: str,
    session_buffer: SessionMemoryBuffer,
    llm_client: BaseLLMClient | None,
    memory_mode: MemoryMode,
) -> None:
    """Extract durable memory candidates from a whole session into the buffer.

    Args:
        user_turn_texts: Canonical user messages from the completed session.
        session_id: Session identifier used for candidate provenance.
        session_buffer: The per-thread buffer to populate; mutated in place.
        llm_client: Structured-output client; extraction is skipped when ``None``.
        memory_mode: Skipped entirely in incognito mode (no durable writes).

    Returns:
        None. Populates ``session_buffer.held_*`` as a side effect.
    """

    if memory_mode == MemoryMode.INCOGNITO:
        return
    if llm_client is None:
        return
    user_texts = list(user_turn_texts)
    if not user_texts:
        return

    semantic_result, procedural_result = await asyncio.gather(
        _extract_semantic(user_texts=user_texts, session_id=session_id, llm=llm_client),
        _extract_procedural(user_texts=user_texts, llm=llm_client),
    )

    for write in semantic_result:
        semantic_candidate = build_semantic_candidate(
            write,
            message=_turn_text(user_texts, write.source_turn_index),
        )
        session_buffer.hold_semantic(semantic_candidate, _SESSION_END_DECISION)

    for draft, turn_index in procedural_result:
        procedural_candidate = build_procedural_candidate(
            draft,
            message=_turn_text(user_texts, turn_index),
            session_id=session_id,
            turn_index=turn_index,
        )
        session_buffer.hold_procedural(procedural_candidate, _SESSION_END_DECISION)


async def _extract_semantic(
    *,
    user_texts: list[str],
    session_id: str,
    llm: BaseLLMClient,
) -> list[MemoryWrite]:
    """Run the semantic extraction pass; degrade to empty on any error."""

    try:
        result = await llm.generate_structured(
            prompt=build_session_semantic_user_prompt(
                user_texts=user_texts,
                session_id=session_id,
            ),
            response_schema=ExtractionResult,
            system_instruction=build_session_semantic_system_prompt(),
        )
    except Exception:
        logger.warning("session semantic extraction failed; skipping", exc_info=True)
        return []

    facts: list[MemoryWrite] = []
    for fact in result.facts:
        # Re-anchor provenance to the actual session + the turn the evidence
        # quote came from; the LLM's self-reported index is corrected here.
        fact.source_session_id = session_id
        fact.source_turn_index = _attribute_turn(
            user_texts, fact.evidence_quote, fact.source_turn_index
        )
        facts.append(fact)
    return facts


async def _extract_procedural(
    *,
    user_texts: list[str],
    llm: BaseLLMClient,
) -> list[tuple[ProceduralRuleDraft, int]]:
    """Run the procedural extraction pass; degrade to empty on any error."""

    try:
        result = await llm.generate_structured(
            prompt=build_session_procedural_user_prompt(user_texts=user_texts),
            response_schema=ProceduralExtractionResult,
            system_instruction=build_session_procedural_system_prompt(),
        )
    except Exception:
        logger.warning("session procedural extraction failed; skipping", exc_info=True)
        return []

    rules: list[tuple[ProceduralRuleDraft, int]] = []
    for draft in result.rules:
        quote = draft.evidence[0] if draft.evidence else draft.rule
        rules.append((draft, _attribute_turn(user_texts, quote, len(user_texts) - 1)))
    return rules


def _attribute_turn(user_texts: list[str], quote: str, fallback: int) -> int:
    """Map an evidence quote back to the user turn it came from.

    Exact substring match first; then highest token-overlap turn; then the
    provided fallback clamped into range. Pure, LLM-free, and unit-testable.
    """

    if not user_texts:
        return 0
    needle = quote.strip().lower()
    if needle:
        for index, text in enumerate(user_texts):
            if needle in text.lower():
                return index
        best_index, best_overlap = -1, 0
        needle_tokens = set(needle.split())
        if needle_tokens:
            for index, text in enumerate(user_texts):
                overlap = len(needle_tokens & set(text.lower().split()))
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
        if best_index >= 0:
            return best_index
    return max(0, min(fallback, len(user_texts) - 1))


def _turn_text(user_texts: list[str], turn_index: int) -> str:
    """Return the user-turn text for an index, clamped into range."""

    if not user_texts:
        return ""
    return user_texts[max(0, min(turn_index, len(user_texts) - 1))]


__all__ = ["extract_session_candidates"]

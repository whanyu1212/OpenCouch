"""LLM-driven extraction services for semantic and procedural memory.

Per AGENTS.md §6, the LangGraph nodes that surface memory extraction to
the graph topology should stay thin — orchestration only. The actual
work of building prompts, calling the LLM, applying deterministic
backstops, and dispatching candidates through ``TurnWriteService`` lives
here as plain async services.

Two parallel functions live in this module:

- :func:`extract_semantic_facts` — wraps the semantic extractor LLM
  call, appends deterministic backstops, and dispatches the resulting
  facts through :class:`TurnWriteService` for policy-aware writes.
- :func:`extract_procedural_rules` — same shape for procedural rules,
  minus the backstops (procedural extraction has no backstop helper).

Both functions:

1. Silently skip when the runtime lacks an LLM client or is in
   ``MemoryMode.INCOGNITO``. The skip is reflected in the returned
   outcome's ``reason`` field; the caller decides how to surface it.
2. Catch all exceptions from the LLM call and downstream service
   dispatch, logging at WARNING level with ``exc_info=True`` and
   returning a degraded outcome rather than propagating. Memory
   extraction is a side-effect path; an extraction failure must not
   fail the parent turn.
3. Return a structured outcome carrying the duration, candidate
   counts, write counts, and a reason string. The node wrappers
   format these into LangGraph diagnostics deltas.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from agent.memory.models import ExtractionResult, ProceduralExtractionResult
from agent.memory.modes import MemoryMode
from agent.memory.policy.backstops import get_deterministic_semantic_backstops
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.policy.turn_routing import (
    get_session_turn_index,
    should_skip_memory_extraction,
)
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.prompts.extraction import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from agent.memory.prompts.procedural import (
    build_procedural_writer_system_prompt,
    build_procedural_writer_user_prompt,
)
from agent.memory.store import MemoryStore
from agent.memory.turn_write_service import TurnWriteService
from agent.observability.timing import elapsed_ms
from agent.state import AgentState, resolve_owner_id
from services.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SemanticExtractionOutcome:
    """Result of one semantic-extraction service call.

    Carries the per-call telemetry that the runtime merges into per-turn
    diagnostics via :meth:`as_diagnostics`.
    """

    duration_ms: float
    semantic_writes: int = 0
    semantic_bumps: int = 0
    semantic_candidates: int = 0
    semantic_commit_now_candidates: int = 0
    semantic_session_end_holds: int = 0
    semantic_repeat_required: int = 0
    semantic_policy_drops: int = 0
    reason: str = ""

    def as_diagnostics(self) -> dict[str, Any]:
        """Render this outcome as the dict the runtime merges into diagnostics.

        Returns:
            Diagnostics-shaped dict with the standard ``extract_facts_*``
            and ``semantic_*`` keys.
        """

        return {
            "extract_facts_ms": self.duration_ms,
            "semantic_writes": self.semantic_writes,
            "semantic_bumps": self.semantic_bumps,
            "semantic_candidates": self.semantic_candidates,
            "semantic_commit_now_candidates": self.semantic_commit_now_candidates,
            "semantic_session_end_holds": self.semantic_session_end_holds,
            "semantic_repeat_required": self.semantic_repeat_required,
            "semantic_policy_drops": self.semantic_policy_drops,
            "extract_facts_reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProceduralExtractionOutcome:
    """Result of one procedural-extraction service call."""

    duration_ms: float
    procedural_writes: int = 0
    procedural_candidates: int = 0
    procedural_commit_now_candidates: int = 0
    procedural_session_end_holds: int = 0
    procedural_policy_drops: int = 0
    reason: str = ""

    def as_diagnostics(self) -> dict[str, Any]:
        """Render this outcome as the dict the runtime merges into diagnostics.

        Returns:
            Diagnostics-shaped dict with the standard ``extract_procedural_*``
            and ``procedural_*`` keys.
        """

        return {
            "extract_procedural_ms": self.duration_ms,
            "procedural_writes": self.procedural_writes,
            "procedural_candidates": self.procedural_candidates,
            "procedural_commit_now_candidates": self.procedural_commit_now_candidates,
            "procedural_session_end_holds": self.procedural_session_end_holds,
            "procedural_policy_drops": self.procedural_policy_drops,
            "extract_procedural_reason": self.reason,
        }


async def extract_semantic_facts(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
    embedding_provider: EmbeddingProvider | None,
    session_buffer: SessionMemoryBuffer | None,
) -> SemanticExtractionOutcome:
    """Extract and persist zero or more semantic facts from the current turn.

    Args:
        state (AgentState): Current graph state after response generation.
        llm_client (BaseLLMClient | None): Control LLM used for extraction.
        memory_store (MemoryStore): Store to write semantic facts into.
        memory_mode (MemoryMode): Runtime memory mode; ``INCOGNITO`` skips writes.
        embedding_provider (EmbeddingProvider | None): Optional document embedding
            provider used by the write service.
        session_buffer (SessionMemoryBuffer | None): Runtime-managed buffer for
            candidates held until session end.

    Returns:
        SemanticExtractionOutcome: Counts and timing for the extraction call.
    """

    start = time.monotonic()

    skip_reason = should_skip_memory_extraction(state)
    if skip_reason is not None:
        logger.debug(
            "extract_semantic_facts: %s; skipping extraction",
            skip_reason,
        )
        return SemanticExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason=f"skipped: {skip_reason}",
        )

    if llm_client is None:
        logger.debug("extract_semantic_facts: no llm_client; skipping")
        return SemanticExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason="skipped: no llm_client",
        )
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_semantic_facts: incognito mode; skipping (no writes to "
            "persistent memory in incognito)"
        )
        return SemanticExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason="skipped: incognito",
        )

    owner_id = resolve_owner_id(state)
    turn_index = get_session_turn_index(state)

    try:
        extraction: ExtractionResult = await llm_client.generate_structured(
            prompt=build_extraction_user_prompt(state, turn_index=turn_index),
            response_schema=ExtractionResult,
            system_instruction=build_extraction_system_prompt(),
        )
    except Exception:
        logger.warning(
            "extract_semantic_facts: LLM structured-output call failed; "
            "skipping extraction for this turn.",
            exc_info=True,
        )
        return SemanticExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason="skipped: llm error",
        )

    logger.info(
        "extract_semantic_facts: %d facts, reason=%r",
        len(extraction.facts),
        extraction.reason,
    )

    backstop_facts = get_deterministic_semantic_backstops(
        message=state["message"],
        session_id=state.get("session_id"),
        turn_index=turn_index,
    )
    if backstop_facts:
        extraction.facts.extend(backstop_facts)

    if not extraction.facts:
        return SemanticExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason=extraction.reason,
        )

    result = await TurnWriteService().process_semantic_facts(
        writes=extraction.facts,
        message=state["message"],
        reason=extraction.reason,
        owner_id=owner_id,
        store=memory_store,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        session_buffer=session_buffer,
    )
    return SemanticExtractionOutcome(
        duration_ms=elapsed_ms(start),
        semantic_writes=result.written,
        semantic_bumps=result.bumped,
        semantic_candidates=result.candidates,
        semantic_commit_now_candidates=result.commit_now_candidates,
        semantic_session_end_holds=result.session_end_holds,
        semantic_repeat_required=result.repeat_required,
        semantic_policy_drops=result.policy_drops,
        reason=result.reason,
    )


async def extract_procedural_rules(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
    memory_store: MemoryStore,
    memory_mode: MemoryMode,
    session_buffer: SessionMemoryBuffer | None,
) -> ProceduralExtractionOutcome:
    """Extract and persist zero or more procedural rules from the current turn.

    Args:
        state (AgentState): Current graph state after response generation.
        llm_client (BaseLLMClient | None): Control LLM used for extraction.
        memory_store (MemoryStore): Store hosting the user's procedural profile.
        memory_mode (MemoryMode): Runtime memory mode; ``INCOGNITO`` skips writes.
        session_buffer (SessionMemoryBuffer | None): Runtime-managed buffer for
            candidates held until session end.

    Returns:
        ProceduralExtractionOutcome: Counts and timing for the extraction call.
    """

    start = time.monotonic()

    skip_reason = should_skip_memory_extraction(state)
    if skip_reason is not None:
        logger.debug(
            "extract_procedural_rules: %s; skipping extraction",
            skip_reason,
        )
        return ProceduralExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason=f"skipped: {skip_reason}",
        )

    if llm_client is None:
        logger.debug("extract_procedural_rules: no llm_client; skipping")
        return ProceduralExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason="skipped: no llm_client",
        )
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_procedural_rules: incognito mode; skipping (no "
            "writes to persistent memory in incognito)"
        )
        return ProceduralExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason="skipped: incognito",
        )

    owner_id = resolve_owner_id(state)
    turn_index = get_session_turn_index(state)

    try:
        result: ProceduralExtractionResult = await llm_client.generate_structured(
            prompt=build_procedural_writer_user_prompt(state),
            response_schema=ProceduralExtractionResult,
            system_instruction=build_procedural_writer_system_prompt(),
        )
    except Exception:
        logger.warning(
            "extract_procedural_rules: LLM structured-output call "
            "failed; skipping rule writes for this turn.",
            exc_info=True,
        )
        return ProceduralExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason="skipped: llm error",
        )

    logger.info(
        "extract_procedural_rules: %d rules, reason=%r",
        len(result.rules),
        result.reason,
    )

    if not result.rules:
        return ProceduralExtractionOutcome(
            duration_ms=elapsed_ms(start),
            reason=result.reason,
        )

    processing = await TurnWriteService().process_procedural_rules(
        drafts=result.rules,
        message=state["message"],
        reason=result.reason,
        session_id=state.get("session_id") or owner_id,
        turn_index=turn_index,
        owner_id=owner_id,
        store=memory_store,
        llm_client=llm_client,
        session_buffer=session_buffer,
    )
    return ProceduralExtractionOutcome(
        duration_ms=elapsed_ms(start),
        procedural_writes=processing.written,
        procedural_candidates=processing.candidates,
        procedural_commit_now_candidates=processing.commit_now_candidates,
        procedural_session_end_holds=processing.session_end_holds,
        procedural_policy_drops=processing.policy_drops,
        reason=processing.reason,
    )

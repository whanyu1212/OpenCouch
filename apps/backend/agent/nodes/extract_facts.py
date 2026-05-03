"""Semantic fact extraction node with hot-path policy and session buffering.

Runs after response generation on every turn and extracts zero or more
memory-worthy facts from the current user message. The node separates
"the extractor produced a fact" from "the store should persist it
immediately": each LLM output first becomes a semantic candidate, the
deterministic write policy decides whether it is safe to commit now,
and only ``commit_now`` candidates continue to dedup + store write.
Candidates marked for later review are buffered in the runtime-managed
session buffer and revisited at session end.

Design rules:

1. **Conservative extraction.** The LLM is told via system prompt that
   most turns should produce zero facts. Small talk, transient moods,
   speculation, and ambiguous statements are all filtered out.

2. **Silent skip on incognito or no LLM.** The node is always registered
   in the parent graph, but it no-ops when either (a) the memory mode
   is INCOGNITO, or (b) no LLM client is available. Both are legitimate
   states that should not trigger extraction-path machinery.

3. **Policy before persistence.** Extractor outputs become
   :class:`SemanticCandidate` instances first. The deterministic policy
   layer can downgrade a candidate to ``commit_at_session_end``,
   ``require_repetition``, or ``drop``. Session-end / repetition
   candidates are buffered for later review rather than written on the
   hot path.

4. **Hot-path dedup.** Every immediate-commit fact is checked against
   existing store records via :func:`find_near_duplicate`. Duplicates
   bump the matched record's ``last_referenced_at`` timestamp instead
   of writing a new row. Dedup uses token-set Jaccard similarity on
   evidence quotes.

5. **Failures degrade silently.** LLM errors, schema validation errors,
   and store write errors are all logged at WARNING level but never
   propagate. The extraction node is a side-effect node — a failure
   here must not fail the parent turn.

6. **Always return an empty state delta.** The write is a side effect;
   state isn't modified. Returning ``{}`` is the canonical "I touched
   nothing" signal for LangGraph delta-return nodes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.backstops import get_deterministic_semantic_backstops
from agent.memory.extraction_prompts import (
    build_extraction_system_prompt,
    build_extraction_user_prompt,
)
from agent.memory.models import ExtractionResult
from agent.memory.modes import MemoryMode
from agent.memory.orchestration import (
    get_session_turn_index,
    should_skip_memory_extraction,
)
from agent.memory.service import MemoryService
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)


async def run_extract_semantic_facts_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Extract and persist zero or more semantic facts from the current turn.

    Args:
        state: Current graph state after response generation.
        runtime: LangGraph runtime carrying memory dependencies.

    Returns:
        State delta containing extractor diagnostics.
    """

    # Time the full extraction call and report write counts via diagnostics.
    # Every return path composes its delta through ``_diagnostics_delta`` so
    # skipped turns are distinguishable from turns where the node never ran.
    start = time.monotonic()

    def _diagnostics_delta(
        *,
        semantic_writes: int = 0,
        semantic_bumps: int = 0,
        semantic_candidates: int = 0,
        semantic_commit_now_candidates: int = 0,
        semantic_session_end_holds: int = 0,
        semantic_repeat_required: int = 0,
        semantic_policy_drops: int = 0,
        semantic_written_items: list[Any] | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Return a state delta carrying extractor diagnostics and approved writes.

        Args:
            semantic_writes: Number of newly written semantic facts.
            semantic_bumps: Number of duplicate records timestamp-bumped.
            semantic_candidates: Total LLM candidate count.
            semantic_commit_now_candidates: Candidates eligible for immediate write.
            semantic_session_end_holds: Candidates buffered until session end.
            semantic_repeat_required: Candidates requiring repetition evidence.
            semantic_policy_drops: Candidates rejected by deterministic policy.
            semantic_written_items: Approved semantic facts written this turn.
            reason: Human-readable extraction or skip reason.

        Returns:
            State delta with diagnostics and approved writes.
        """

        return {
            "diagnostics": {
                "extract_facts_ms": round((time.monotonic() - start) * 1000, 2),
                "semantic_writes": semantic_writes,
                "semantic_bumps": semantic_bumps,
                "semantic_candidates": semantic_candidates,
                "semantic_commit_now_candidates": semantic_commit_now_candidates,
                "semantic_session_end_holds": semantic_session_end_holds,
                "semantic_repeat_required": semantic_repeat_required,
                "semantic_policy_drops": semantic_policy_drops,
                "extract_facts_reason": reason,
            },
            "written_items": list(semantic_written_items or []),
        }

    skip_reason = should_skip_memory_extraction(state)
    if skip_reason is not None:
        logger.debug(
            "extract_semantic_facts_node: %s; skipping extraction",
            skip_reason,
        )
        return _diagnostics_delta(reason=f"skipped: {skip_reason}")

    llm_client = runtime.context.llm_client
    memory_mode = runtime.context.memory_mode

    if llm_client is None:
        logger.debug("extract_semantic_facts_node: no llm_client; skipping")
        return _diagnostics_delta(reason="skipped: no llm_client")
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_semantic_facts_node: incognito mode; skipping (no writes to "
            "persistent memory in incognito)"
        )
        return _diagnostics_delta(reason="skipped: incognito")

    store = runtime.context.memory_store
    embedding_provider = runtime.context.embedding_provider
    session_buffer = runtime.context.session_memory_buffer
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
            "extract_semantic_facts_node: LLM structured-output call failed; "
            "skipping extraction for this turn.",
            exc_info=True,
        )
        return _diagnostics_delta(reason="skipped: llm error")

    # Log the reason regardless of whether facts were produced; the
    # conservative extractor rejects most turns, and the reason is the
    # fastest way to catch prompt drift.
    logger.info(
        "extract_semantic_facts_node: %d facts, reason=%r",
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
        return _diagnostics_delta(reason=extraction.reason)

    result = await MemoryService().process_semantic_facts(
        writes=extraction.facts,
        message=state["message"],
        reason=extraction.reason,
        owner_id=owner_id,
        store=store,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        session_buffer=session_buffer,
    )
    return _diagnostics_delta(
        semantic_writes=result.written,
        semantic_bumps=result.bumped,
        semantic_candidates=result.candidates,
        semantic_commit_now_candidates=result.commit_now_candidates,
        semantic_session_end_holds=result.session_end_holds,
        semantic_repeat_required=result.repeat_required,
        semantic_policy_drops=result.policy_drops,
        semantic_written_items=result.written_items,
        reason=result.reason,
    )

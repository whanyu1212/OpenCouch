"""Service for processing and persisting extracted memories.

This module owns memory policy, deduplication, reconciliation, and store writes
for semantic and procedural candidates produced during turn-level extraction or
session-end promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from agent.memory.policy.candidates import (
    SessionMemoryBuffer,
    build_procedural_candidate,
    build_semantic_candidate,
)
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.models import (
    MemoryWrite,
    ProceduralRule,
    ProceduralRuleDraft,
    SemanticFact,
)
from agent.memory.procedural import aupsert_procedural_rule, build_procedural_rule
from agent.memory.semantic_writes import (
    BatchWriteItem,
    apply_semantic_writes_batch,
)
from agent.memory.store import MemoryStore
from agent.memory.policy.write import (
    decide_procedural_candidate_llm_primary,
    decide_semantic_candidate_llm_primary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticProcessingResult:
    """Summary of semantic candidate processing for telemetry and diagnostics."""

    written: int
    bumped: int
    candidates: int
    commit_now_candidates: int
    session_end_holds: int
    repeat_required: int
    policy_drops: int
    reason: str
    written_items: list[SemanticFact] = field(default_factory=list)


@dataclass(frozen=True)
class ProceduralProcessingResult:
    """Summary of procedural candidate processing for telemetry and diagnostics."""

    written: int
    candidates: int
    commit_now_candidates: int
    session_end_holds: int
    policy_drops: int
    reason: str
    written_items: list[ProceduralRule] = field(default_factory=list)


class TurnWriteService:
    """Process extracted memory candidates and persist approved writes."""

    async def process_semantic_facts(
        self,
        *,
        writes: list[MemoryWrite],
        message: str,
        reason: str,
        owner_id: str,
        store: MemoryStore,
        llm_client: Any,
        embedding_provider: EmbeddingProvider | None = None,
        session_buffer: SessionMemoryBuffer | None = None,
    ) -> SemanticProcessingResult:
        """Apply policy, deduplication, reconciliation, and writes to facts.

        Args:
            writes: Candidate fact writes from the extractor and backstops.
            message: Current user message used for candidate context.
            reason: Extractor reason to preserve in diagnostics.
            owner_id: Owner whose memory namespace receives writes.
            store: Memory store for reads and writes.
            llm_client: Control LLM used by policy/reconciliation helpers.
            embedding_provider: Optional document embedding provider.
            session_buffer: Optional session-end candidate buffer.

        Returns:
            Processing counters for node diagnostics.
        """

        immediate_candidates: list[tuple[Any, Any]] = []
        session_end_holds = 0
        repeat_required = 0
        policy_drops = 0

        for write in writes:
            candidate = build_semantic_candidate(write, message=message)
            decision = await decide_semantic_candidate_llm_primary(
                candidate,
                llm_client=llm_client,
            )

            if decision.action == "commit_now":
                immediate_candidates.append((candidate, decision))
            elif decision.action == "commit_at_session_end":
                session_end_holds += 1
                if session_buffer is not None:
                    session_buffer.semantic_candidates.append(candidate)
            elif decision.action == "require_repetition":
                repeat_required += 1
                if session_buffer is not None:
                    session_buffer.semantic_candidates.append(candidate)
            else:
                policy_drops += 1

        if not immediate_candidates:
            logger.info(
                "memory_service: semantic policy held all %d facts "
                "(session_end=%d, repetition=%d, dropped=%d)",
                len(writes),
                session_end_holds,
                repeat_required,
                policy_drops,
            )
            return SemanticProcessingResult(
                written=0,
                bumped=0,
                candidates=len(writes),
                commit_now_candidates=0,
                session_end_holds=session_end_holds,
                repeat_required=repeat_required,
                policy_drops=policy_drops,
                reason=reason,
                written_items=[],
            )

        batch_items = [
            BatchWriteItem(
                candidate=candidate,
                write_timing="immediate",
                write_reason=decision.reason,
                policy_version=decision.policy_version,
            )
            for candidate, decision in immediate_candidates
        ]
        batch_outcome = await apply_semantic_writes_batch(
            store,
            owner_id=owner_id,
            items=batch_items,
            llm_client=llm_client,
            embedding_provider=embedding_provider,
            log_context="memory_service",
        )

        if batch_outcome.fetch_failed:
            return SemanticProcessingResult(
                written=0,
                bumped=0,
                candidates=len(writes),
                commit_now_candidates=len(immediate_candidates),
                session_end_holds=session_end_holds,
                repeat_required=repeat_required,
                policy_drops=policy_drops,
                reason="skipped: dedup fetch failed",
                written_items=[],
            )

        logger.info(
            "memory_service: semantic turn complete — %d written, %d bumped, "
            "%d immediate, %d held_for_session, %d repeat_required, %d dropped",
            batch_outcome.written,
            batch_outcome.bumped,
            len(immediate_candidates),
            session_end_holds,
            repeat_required,
            policy_drops,
        )
        return SemanticProcessingResult(
            written=batch_outcome.written,
            bumped=batch_outcome.bumped,
            candidates=len(writes),
            commit_now_candidates=len(immediate_candidates),
            session_end_holds=session_end_holds,
            repeat_required=repeat_required,
            policy_drops=policy_drops,
            reason=reason,
            written_items=batch_outcome.written_items,
        )

    async def process_procedural_rules(
        self,
        *,
        drafts: list[ProceduralRuleDraft],
        message: str,
        reason: str,
        session_id: str,
        turn_index: int,
        owner_id: str,
        store: MemoryStore,
        llm_client: Any,
        session_buffer: SessionMemoryBuffer | None = None,
    ) -> ProceduralProcessingResult:
        """Apply policy and profile upserts to procedural rule drafts.

        Args:
            drafts: Candidate rule drafts returned by the extractor.
            message: Current user message used for candidate context.
            reason: Extractor reason to preserve in diagnostics.
            session_id: Session id for candidate provenance.
            turn_index: Zero-based turn index for candidate provenance.
            owner_id: Owner whose procedural profile receives writes.
            store: Memory store for profile writes.
            llm_client: Control LLM used by policy/reconciliation helpers.
            session_buffer: Optional session-end candidate buffer.

        Returns:
            Processing counters for node diagnostics.
        """

        immediate_candidates: list[tuple[Any, Any]] = []
        session_end_holds = 0
        policy_drops = 0

        for draft in drafts:
            candidate = build_procedural_candidate(
                draft,
                message=message,
                session_id=session_id,
                turn_index=turn_index,
            )
            decision = await decide_procedural_candidate_llm_primary(
                candidate,
                llm_client=llm_client,
            )

            if decision.action == "commit_now":
                immediate_candidates.append((candidate, decision))
            elif decision.action == "commit_at_session_end":
                session_end_holds += 1
                if session_buffer is not None:
                    session_buffer.procedural_candidates.append(candidate)
            else:
                policy_drops += 1

        if not immediate_candidates:
            logger.info(
                "memory_service: procedural policy held all %d rules "
                "(session_end=%d, dropped=%d)",
                len(drafts),
                session_end_holds,
                policy_drops,
            )
            return ProceduralProcessingResult(
                written=0,
                candidates=len(drafts),
                commit_now_candidates=0,
                session_end_holds=session_end_holds,
                policy_drops=policy_drops,
                reason=reason,
                written_items=[],
            )

        written = 0
        written_items: list[ProceduralRule] = []
        for candidate, decision in immediate_candidates:
            draft = candidate.payload
            try:
                rule = build_procedural_rule(
                    rule_text=draft.rule,
                    evidence=draft.evidence,
                    confidence=draft.confidence,
                    source="explicit_user",
                    write_timing="immediate",
                    write_reason=decision.reason,
                    policy_version=decision.policy_version,
                )
                upsert = await aupsert_procedural_rule(
                    store,
                    user_id=owner_id,
                    rule=rule,
                    llm_client=llm_client,
                )
                if upsert.action != "skipped":
                    written += 1
                    written_items.append(rule)
            except Exception:
                logger.warning(
                    "memory_service: failed to write procedural draft %r; "
                    "continuing with other drafts.",
                    draft.rule[:60],
                    exc_info=True,
                )

        logger.info(
            "memory_service: procedural turn complete — %d written, %d immediate, "
            "%d held_for_session, %d dropped",
            written,
            len(immediate_candidates),
            session_end_holds,
            policy_drops,
        )
        return ProceduralProcessingResult(
            written=written,
            candidates=len(drafts),
            commit_now_candidates=len(immediate_candidates),
            session_end_holds=session_end_holds,
            policy_drops=policy_drops,
            reason=reason,
            written_items=written_items,
        )

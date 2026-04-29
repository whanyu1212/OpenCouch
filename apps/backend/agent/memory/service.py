"""Memory side-effect service for extraction nodes.

This module keeps memory policy, deduplication, reconciliation, and store writes
outside LangGraph nodes. Nodes remain responsible for graph/runtime concerns:
early skips, LLM extraction prompts, and diagnostics deltas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent.memory.candidates import (
    SessionMemoryBuffer,
    build_procedural_candidate,
    build_semantic_candidate,
)
from agent.memory.dedup import find_near_duplicate
from agent.memory.embeddings import EmbeddingProvider
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.models import (
    MemoryWrite,
    ProceduralRuleDraft,
    SemanticFact,
)
from agent.memory.procedural import aupsert_procedural_rule, build_procedural_rule
from agent.memory.reconciliation import (
    filter_semantic_collision_candidates,
    plan_semantic_write_llm_primary,
)
from agent.memory.store import MemoryStore, StoreRecord
from agent.memory.write_policy import (
    decide_procedural_candidate_llm_primary,
    decide_semantic_candidate_llm_primary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticProcessingResult:
    """Summary of semantic candidate processing for node diagnostics."""

    written: int
    bumped: int
    candidates: int
    commit_now_candidates: int
    session_end_holds: int
    repeat_required: int
    policy_drops: int
    reason: str


@dataclass(frozen=True)
class ProceduralProcessingResult:
    """Summary of procedural candidate processing for node diagnostics."""

    written: int
    candidates: int
    commit_now_candidates: int
    session_end_holds: int
    policy_drops: int
    reason: str


def _memory_write_to_semantic_fact(
    write: MemoryWrite,
    *,
    write_timing: str = "immediate",
    write_reason: str = "",
    policy_version: str = "phase1_v1",
) -> SemanticFact:
    """Convert an LLM-produced :class:`MemoryWrite` to a stored fact.

    Args:
        write: Structured memory write returned by the extractor.
        write_timing: Timing label for the stored fact.
        write_reason: Reason attached to the write policy decision.
        policy_version: Policy version label written into the record.

    Returns:
        Stored semantic fact model.
    """

    now = _iso_now()
    return SemanticFact(
        id=str(uuid4()),
        category=write.category,
        subject=write.subject,
        predicate=write.predicate,
        object=write.object,
        evidence_quote=write.evidence_quote,
        confidence=write.confidence,
        source_session_id=write.source_session_id,
        source_turn_index=write.source_turn_index,
        created_at=now,
        last_referenced_at=now,
        dormant_at=None,
        superseded_by=None,
        user_visible=True,
        write_timing=write_timing,  # type: ignore[arg-type]
        write_reason=write_reason,
        policy_version=policy_version,
    )


async def _fetch_existing_user_records(
    store: MemoryStore,
    *,
    owner_id: str,
) -> list[Any]:
    """Fetch all semantic-namespace records for a user.

    Args:
        store: Memory store to query.
        owner_id: Owner whose semantic namespace should be loaded.

    Returns:
        Semantic store records for the owner.
    """

    namespace = (owner_id, "semantic")
    record_count = await store.arecord_count(namespace)
    if record_count == 0:
        return []
    return await store.asearch(namespace, query=None, limit=record_count)


async def _write_new_fact(
    store: MemoryStore,
    *,
    owner_id: str,
    fact: SemanticFact,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> None:
    """Persist a freshly-extracted semantic fact to the store.

    Args:
        store: Memory store to write to.
        owner_id: Owner whose semantic namespace receives the fact.
        fact: Semantic fact to persist.
        embedding: Optional document embedding for hybrid retrieval.
        embedding_model: Optional embedding model identifier.
    """

    namespace = (owner_id, "semantic")
    await store.aput(
        namespace,
        key=fact.id,
        value=fact.model_dump(mode="json"),
        embedding=embedding,
        embedding_model=embedding_model,
    )


async def _bump_last_referenced_at(
    store: MemoryStore,
    *,
    matched_record: Any,
) -> None:
    """Update the matched record's ``last_referenced_at`` to now.

    Args:
        store: Memory store containing the matched record.
        matched_record: Existing semantic record to update.
    """

    updated_value = dict(matched_record.value)
    updated_value["last_referenced_at"] = _iso_now()
    await store.aput(
        matched_record.namespace,
        key=matched_record.key,
        value=updated_value,
    )


async def _mark_fact_superseded(
    store: MemoryStore,
    *,
    matched_record: Any,
    replacement_fact_id: str,
) -> None:
    """Mark one stored semantic fact as superseded by a newer fact.

    Args:
        store: Memory store containing the matched record.
        matched_record: Existing semantic record to mark dormant.
        replacement_fact_id: New fact id that supersedes the matched record.
    """

    updated_value = dict(matched_record.value)
    now = _iso_now()
    updated_value["last_referenced_at"] = now
    updated_value["dormant_at"] = now
    updated_value["superseded_by"] = replacement_fact_id
    await store.aput(
        matched_record.namespace,
        key=matched_record.key,
        value=updated_value,
        embedding=getattr(matched_record, "embedding", None),
        embedding_model=getattr(matched_record, "embedding_model", None),
    )


class MemoryService:
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
            )

        try:
            existing_records = await _fetch_existing_user_records(
                store, owner_id=owner_id
            )
        except Exception:
            logger.warning(
                "memory_service: failed to fetch existing semantic records for "
                "dedup; skipping all candidates for this turn.",
                exc_info=True,
            )
            return SemanticProcessingResult(
                written=0,
                bumped=0,
                candidates=len(writes),
                commit_now_candidates=len(immediate_candidates),
                session_end_holds=session_end_holds,
                repeat_required=repeat_required,
                policy_drops=policy_drops,
                reason="skipped: dedup fetch failed",
            )

        candidate_embeddings: list[list[float] | None] = [None] * len(
            immediate_candidates
        )
        embedding_model_name: str | None = None
        if embedding_provider is not None:
            try:
                quotes = [
                    candidate.payload.evidence_quote
                    for candidate, _ in immediate_candidates
                ]
                candidate_embeddings = await embedding_provider.aembed(
                    quotes,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                embedding_model_name = embedding_provider.model_name
                if all(e is None for e in candidate_embeddings):
                    embedding_model_name = None
            except Exception:
                logger.warning(
                    "memory_service: semantic embedding batch failed; writing facts "
                    "without embeddings for this turn.",
                    exc_info=True,
                )
                candidate_embeddings = [None] * len(immediate_candidates)
                embedding_model_name = None

        written = 0
        bumped = 0
        for candidate_index, (candidate, decision) in enumerate(immediate_candidates):
            write = candidate.payload
            collision_records = filter_semantic_collision_candidates(
                write,
                existing_records,
            )
            try:
                matched = find_near_duplicate(write, collision_records)
            except Exception:
                logger.warning(
                    "memory_service: semantic dedup check raised for candidate %r; "
                    "skipping this candidate.",
                    write.evidence_quote[:40],
                    exc_info=True,
                )
                continue

            if matched is not None:
                try:
                    await _bump_last_referenced_at(store, matched_record=matched)
                    bumped += 1
                except Exception:
                    logger.warning(
                        "memory_service: failed to bump last_referenced_at on "
                        "matched record %r; continuing with other candidates.",
                        matched.key,
                        exc_info=True,
                    )
                continue

            try:
                fact = _memory_write_to_semantic_fact(
                    write,
                    write_timing="immediate",
                    write_reason=decision.reason,
                    policy_version=decision.policy_version,
                )
                reconciliation = await plan_semantic_write_llm_primary(
                    fact,
                    collision_records,
                    llm_client=llm_client,
                )
                if reconciliation.bump_record is not None:
                    await _bump_last_referenced_at(
                        store,
                        matched_record=reconciliation.bump_record,
                    )
                    bumped += 1
                    continue

                this_embedding = candidate_embeddings[candidate_index]
                this_model = (
                    embedding_model_name if this_embedding is not None else None
                )
                await _write_new_fact(
                    store,
                    owner_id=owner_id,
                    fact=fact,
                    embedding=this_embedding,
                    embedding_model=this_model,
                )
                written += 1
                existing_records.append(
                    StoreRecord(
                        namespace=(owner_id, "semantic"),
                        key=fact.id,
                        value=fact.model_dump(mode="json"),
                        embedding=this_embedding,
                        embedding_model=this_model,
                    )
                )
                for superseded_record in reconciliation.supersede_records:
                    try:
                        await _mark_fact_superseded(
                            store,
                            matched_record=superseded_record,
                            replacement_fact_id=fact.id,
                        )
                        superseded_record.value["last_referenced_at"] = fact.created_at
                        superseded_record.value["dormant_at"] = fact.created_at
                        superseded_record.value["superseded_by"] = fact.id
                    except Exception:
                        logger.warning(
                            "memory_service: failed to mark stale fact %r as "
                            "superseded after writing replacement.",
                            superseded_record.key,
                            exc_info=True,
                        )
            except Exception:
                logger.warning(
                    "memory_service: failed to write semantic candidate %r; "
                    "continuing with other candidates.",
                    write.evidence_quote[:40],
                    exc_info=True,
                )

        logger.info(
            "memory_service: semantic turn complete — %d written, %d bumped, "
            "%d immediate, %d held_for_session, %d repeat_required, %d dropped",
            written,
            bumped,
            len(immediate_candidates),
            session_end_holds,
            repeat_required,
            policy_drops,
        )
        return SemanticProcessingResult(
            written=written,
            bumped=bumped,
            candidates=len(writes),
            commit_now_candidates=len(immediate_candidates),
            session_end_holds=session_end_holds,
            repeat_required=repeat_required,
            policy_drops=policy_drops,
            reason=reason,
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
            )

        written = 0
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
        )

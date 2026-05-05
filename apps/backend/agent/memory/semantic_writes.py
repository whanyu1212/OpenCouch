"""Shared semantic-memory write primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent.memory.dedup import find_near_duplicate
from agent.memory.hashing import iso_now as _iso_now
from agent.memory.models import MemoryWrite, SemanticFact
from agent.memory.reconciliation import (
    filter_semantic_collision_candidates,
    plan_semantic_write_llm_primary,
)
from agent.memory.store import MemoryStore, StoreRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SemanticWriteOutcome:
    """Result of applying one semantic write candidate."""

    written: int = 0
    bumped: int = 0
    skipped: int = 0
    fact: SemanticFact | None = None


def memory_write_to_semantic_fact(
    write: MemoryWrite,
    *,
    write_timing: str = "immediate",
    write_reason: str = "",
    policy_version: str = "phase1_v1",
) -> SemanticFact:
    """Convert an LLM-produced memory write to a stored semantic fact.

    Args:
        write (MemoryWrite): Structured memory write returned by the extractor.
        write_timing (str): Timing label for the stored fact.
        write_reason (str): Reason attached to the write policy decision.
        policy_version (str): Policy version label written into the record.

    Returns:
        SemanticFact: Stored semantic fact model.
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


async def fetch_existing_semantic_records(
    store: MemoryStore,
    *,
    owner_id: str,
) -> list[StoreRecord]:
    """Fetch all semantic-namespace records for a user.

    Args:
        store (MemoryStore): Memory store to query.
        owner_id (str): Owner whose semantic namespace should be loaded.

    Returns:
        list[StoreRecord]: Semantic store records for the owner.
    """

    namespace = (owner_id, "semantic")
    record_count = await store.arecord_count(namespace)
    if record_count == 0:
        return []
    return await store.asearch(namespace, query=None, limit=record_count)


async def write_new_semantic_fact(
    store: MemoryStore,
    *,
    owner_id: str,
    fact: SemanticFact,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> None:
    """Persist a freshly extracted semantic fact to the store.

    Args:
        store (MemoryStore): Memory store to write to.
        owner_id (str): Owner whose semantic namespace receives the fact.
        fact (SemanticFact): Semantic fact to persist.
        embedding (list[float] | None): Optional document embedding for hybrid retrieval.
        embedding_model (str | None): Optional embedding model identifier.

    Returns:
        None: Writes the semantic fact as a side effect.
    """

    namespace = (owner_id, "semantic")
    await store.aput(
        namespace,
        key=fact.id,
        value=fact.model_dump(mode="json"),
        embedding=embedding,
        embedding_model=embedding_model,
    )


async def bump_semantic_last_referenced_at(
    store: MemoryStore,
    *,
    matched_record: Any,
) -> None:
    """Update a matched semantic record's last-referenced timestamp.

    Args:
        store (MemoryStore): Memory store containing the matched record.
        matched_record (Any): Existing semantic record to update.

    Returns:
        None: Updates the stored record as a side effect.
    """

    updated_value = dict(matched_record.value)
    updated_value["last_referenced_at"] = _iso_now()
    await store.aput(
        matched_record.namespace,
        key=matched_record.key,
        value=updated_value,
    )


async def mark_semantic_fact_superseded(
    store: MemoryStore,
    *,
    matched_record: Any,
    replacement_fact_id: str,
) -> None:
    """Mark one stored semantic fact as superseded by a newer fact.

    Args:
        store (MemoryStore): Memory store containing the matched record.
        matched_record (Any): Existing semantic record to mark dormant.
        replacement_fact_id (str): New fact id that supersedes the matched record.

    Returns:
        None: Updates the stored record as a side effect.
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


async def apply_semantic_write(
    store: MemoryStore,
    *,
    owner_id: str,
    write: MemoryWrite,
    existing_records: list[StoreRecord],
    llm_client: Any,
    write_timing: str,
    write_reason: str,
    policy_version: str,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
    log_context: str = "semantic_writes",
) -> SemanticWriteOutcome:
    """Apply dedup, reconciliation, write, bump, and supersede for one fact.

    Args:
        store (MemoryStore): Memory store to update.
        owner_id (str): Owner whose semantic namespace receives writes.
        write (MemoryWrite): Semantic candidate payload to apply.
        existing_records (list[StoreRecord]): Mutable semantic record cache used
            by the caller for this write batch.
        llm_client (Any): Optional control LLM used by reconciliation.
        write_timing (str): Timing label for the stored fact.
        write_reason (str): Reason attached to the write.
        policy_version (str): Policy version label written into the record.
        embedding (list[float] | None): Optional document embedding.
        embedding_model (str | None): Optional embedding model identifier.
        log_context (str): Prefix used in warning log messages.

    Returns:
        SemanticWriteOutcome: Counters plus the written fact when a new record
        was persisted.
    """

    collision_records = filter_semantic_collision_candidates(
        write,
        existing_records,
    )
    try:
        matched = find_near_duplicate(write, collision_records)
    except Exception:
        logger.warning(
            "%s: semantic dedup check failed for candidate %r; skipping it.",
            log_context,
            write.evidence_quote[:60],
            exc_info=True,
        )
        return SemanticWriteOutcome(skipped=1)

    if matched is not None:
        try:
            await bump_semantic_last_referenced_at(store, matched_record=matched)
            return SemanticWriteOutcome(bumped=1)
        except Exception:
            logger.warning(
                "%s: failed to bump last_referenced_at on matched record %r.",
                log_context,
                matched.key,
                exc_info=True,
            )
            return SemanticWriteOutcome(skipped=1)

    try:
        fact = memory_write_to_semantic_fact(
            write,
            write_timing=write_timing,
            write_reason=write_reason,
            policy_version=policy_version,
        )
        reconciliation = await plan_semantic_write_llm_primary(
            fact,
            collision_records,
            llm_client=llm_client,
        )
        if reconciliation.bump_record is not None:
            await bump_semantic_last_referenced_at(
                store,
                matched_record=reconciliation.bump_record,
            )
            return SemanticWriteOutcome(bumped=1)

        await write_new_semantic_fact(
            store,
            owner_id=owner_id,
            fact=fact,
            embedding=embedding,
            embedding_model=embedding_model,
        )
        existing_records.append(
            StoreRecord(
                namespace=(owner_id, "semantic"),
                key=fact.id,
                value=fact.model_dump(mode="json"),
                embedding=embedding,
                embedding_model=embedding_model,
            )
        )
        for superseded_record in reconciliation.supersede_records:
            try:
                await mark_semantic_fact_superseded(
                    store,
                    matched_record=superseded_record,
                    replacement_fact_id=fact.id,
                )
                superseded_record.value["last_referenced_at"] = fact.created_at
                superseded_record.value["dormant_at"] = fact.created_at
                superseded_record.value["superseded_by"] = fact.id
            except Exception:
                logger.warning(
                    "%s: failed to mark stale fact %r as superseded after "
                    "writing replacement.",
                    log_context,
                    superseded_record.key,
                    exc_info=True,
                )
        return SemanticWriteOutcome(written=1, fact=fact)
    except Exception:
        logger.warning(
            "%s: failed to write semantic candidate %r.",
            log_context,
            write.evidence_quote[:60],
            exc_info=True,
        )
        return SemanticWriteOutcome(skipped=1)

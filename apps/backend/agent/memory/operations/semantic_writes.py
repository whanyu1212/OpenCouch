"""Shared semantic-memory write primitives."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent.memory.hashing import iso_now as _iso_now
from agent.memory.operations.dedup import find_near_duplicate
from agent.memory.operations.reconciliation import (
    filter_semantic_collision_candidates,
    plan_semantic_write_llm_primary,
)
from agent.memory.types import MemoryWrite, SemanticFact
from agent.memory.store import MemoryStore, StoreRecord

if TYPE_CHECKING:
    from agent.memory.policy.candidates import SemanticCandidate
    from agent.memory.providers.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SemanticWriteOutcome:
    """Result of applying one semantic write candidate."""

    written: int = 0
    bumped: int = 0
    skipped: int = 0
    fact: SemanticFact | None = None


@dataclass(frozen=True, slots=True)
class BatchWriteItem:
    """One semantic candidate plus its caller-decided write metadata."""

    candidate: "SemanticCandidate"
    write_timing: str
    write_reason: str
    policy_version: str


@dataclass(slots=True)
class BatchSemanticWriteOutcome:
    """Aggregate counters and written facts from a batch write.

    ``fetch_failed`` distinguishes the case where the existing-records fetch
    itself failed (no item could even be evaluated for dedup) from the case
    where every item was evaluated but each per-candidate write was skipped
    or threw. Both paths produce ``skipped == len(items)``; only the former
    means the batch never made it past the prelude.
    """

    written: int = 0
    bumped: int = 0
    skipped: int = 0
    failures: int = 0
    fetch_failed: bool = False
    written_items: list[SemanticFact] = field(default_factory=list)


def _canonicalize_user_subject_for_owner(
    write: MemoryWrite,
    *,
    owner_id: str,
) -> MemoryWrite:
    """Return a write whose user subject uses the namespace owner id.

    Session-end extractors may emit a placeholder user identifier (for example
    ``"test-user"`` in deterministic fakes). The memory namespace owner is the
    authoritative current-user id, so normalize user-subject writes before
    deduplication and persistence.
    """

    if write.subject.type != "User" or write.subject.identifier == owner_id:
        return write
    return write.model_copy(
        update={"subject": write.subject.model_copy(update={"identifier": owner_id})}
    )


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
        embedding=getattr(matched_record, "embedding", None),
        embedding_model=getattr(matched_record, "embedding_model", None),
    )


def build_superseded_value(
    matched_record: Any,
    *,
    replacement_fact_id: str,
    superseded_at: str,
) -> dict[str, Any]:
    """Return the stored value marking one fact superseded by a newer fact.

    Args:
        matched_record (Any): Existing semantic record to mark dormant.
        replacement_fact_id (str): New fact id that supersedes the matched record.
        superseded_at (str): Timestamp recorded for the supersede transition.

    Returns:
        dict[str, Any]: Updated record value; the input record is not mutated.
    """

    updated_value = dict(matched_record.value)
    updated_value["last_referenced_at"] = superseded_at
    updated_value["dormant_at"] = superseded_at
    updated_value["superseded_by"] = replacement_fact_id
    return updated_value


def _record_to_batch_item(
    matched_record: Any,
    value: dict[str, Any],
) -> tuple[Any, str, dict[str, Any], list[float] | None, str | None]:
    """Return one ``aput_batch`` item preserving a record's embedding metadata.

    Args:
        matched_record (Any): Record supplying namespace, key, and embeddings.
        value (dict[str, Any]): Serialized value to persist for the record.

    Returns:
        tuple: Item shaped as ``(namespace, key, value, embedding, embedding_model)``.
    """

    return (
        matched_record.namespace,
        matched_record.key,
        value,
        getattr(matched_record, "embedding", None),
        getattr(matched_record, "embedding_model", None),
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

    updated_value = build_superseded_value(
        matched_record,
        replacement_fact_id=replacement_fact_id,
        superseded_at=_iso_now(),
    )
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

    write = _canonicalize_user_subject_for_owner(write, owner_id=owner_id)
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

        # Prepare the replacement plus every supersede mutation before touching
        # the store, then commit them as one batch. A partial commit would leave
        # the replacement active alongside the facts it contradicts, so the plan
        # must succeed or fail as a unit.
        namespace = (owner_id, "semantic")
        fact_value = fact.model_dump(mode="json")
        superseded_values = [
            build_superseded_value(
                superseded_record,
                replacement_fact_id=fact.id,
                superseded_at=fact.created_at,
            )
            for superseded_record in reconciliation.supersede_records
        ]
        await store.aput_batch(
            [
                (namespace, fact.id, fact_value, embedding, embedding_model),
                *(
                    _record_to_batch_item(superseded_record, superseded_value)
                    for superseded_record, superseded_value in zip(
                        reconciliation.supersede_records,
                        superseded_values,
                        strict=True,
                    )
                ),
            ]
        )

        existing_records.append(
            StoreRecord(
                namespace=namespace,
                key=fact.id,
                value=fact_value,
                embedding=embedding,
                embedding_model=embedding_model,
            )
        )
        for superseded_record, superseded_value in zip(
            reconciliation.supersede_records,
            superseded_values,
            strict=True,
        ):
            superseded_record.value.update(superseded_value)
        return SemanticWriteOutcome(written=1, fact=fact)
    except Exception:
        logger.warning(
            "%s: failed to write semantic candidate %r; the reconciliation plan "
            "was not applied.",
            log_context,
            write.evidence_quote[:60],
            exc_info=True,
        )
        return SemanticWriteOutcome(skipped=1)


async def _embed_candidate_quotes(
    items: list[BatchWriteItem],
    *,
    embedding_provider: "EmbeddingProvider | None",
    log_context: str,
) -> tuple[list[list[float] | None], str | None]:
    """Embed candidate evidence quotes with safe fallback on failure.

    Args:
        items: Candidates to embed.
        embedding_provider: Optional document embedding provider.
        log_context: Prefix used in warning log messages.

    Returns:
        Per-candidate embeddings (``None`` on failure or when provider is absent)
        and the embedding model identifier (``None`` when no embedding succeeded).
    """

    if not items or embedding_provider is None:
        return [None] * len(items), None

    try:
        quotes = [item.candidate.payload.evidence_quote for item in items]
        embeddings = await embedding_provider.aembed(
            quotes,
            task_type="RETRIEVAL_DOCUMENT",
        )
        model_name: str | None = embedding_provider.model_name
        if all(embedding is None for embedding in embeddings):
            model_name = None
        return embeddings, model_name
    except Exception:
        logger.warning(
            "%s: semantic embedding batch failed; writing facts without "
            "embeddings for this batch.",
            log_context,
            exc_info=True,
        )
        return [None] * len(items), None


async def apply_semantic_writes_batch(
    store: MemoryStore,
    *,
    owner_id: str,
    items: list[BatchWriteItem],
    llm_client: Any,
    embedding_provider: "EmbeddingProvider | None" = None,
    log_context: str,
) -> BatchSemanticWriteOutcome:
    """Apply a batch of semantic candidates with shared dedup and embeddings.

    Centralizes the embed → fetch existing records → loop ``apply_semantic_write``
    pattern shared by turn-level and session-end semantic write paths.

    Args:
        store: Memory store to update.
        owner_id: Owner whose semantic namespace receives writes.
        items: Candidates with caller-decided write metadata.
        llm_client: Optional control LLM used by reconciliation.
        embedding_provider: Optional document embedding provider.
        log_context: Prefix used in warning log messages.

    Returns:
        Aggregate batch outcome with counters and written facts. When the
        existing-records fetch fails, all items are counted as failures.
    """

    outcome = BatchSemanticWriteOutcome()
    if not items:
        return outcome

    try:
        existing_records = await fetch_existing_semantic_records(
            store, owner_id=owner_id
        )
    except Exception:
        logger.warning(
            "%s: failed to fetch existing semantic records for dedup; marking "
            "all candidates in this batch as failed.",
            log_context,
            exc_info=True,
        )
        outcome.failures = len(items)
        outcome.fetch_failed = True
        return outcome

    embeddings, model_name = await _embed_candidate_quotes(
        items,
        embedding_provider=embedding_provider,
        log_context=log_context,
    )

    for index, item in enumerate(items):
        embedding = embeddings[index]
        embedding_model = model_name if embedding is not None else None
        item_outcome = await apply_semantic_write(
            store,
            owner_id=owner_id,
            write=item.candidate.payload,
            existing_records=existing_records,
            llm_client=llm_client,
            write_timing=item.write_timing,
            write_reason=item.write_reason,
            policy_version=item.policy_version,
            embedding=embedding,
            embedding_model=embedding_model,
            log_context=log_context,
        )
        outcome.written += item_outcome.written
        outcome.bumped += item_outcome.bumped
        outcome.skipped += item_outcome.skipped
        if item_outcome.fact is not None:
            outcome.written_items.append(item_outcome.fact)

    return outcome

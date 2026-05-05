"""Shared semantic-memory write primitives."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from agent.memory.hashing import iso_now as _iso_now
from agent.memory.models import MemoryWrite, SemanticFact
from agent.memory.store import MemoryStore


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
) -> list[Any]:
    """Fetch all semantic-namespace records for a user.

    Args:
        store (MemoryStore): Memory store to query.
        owner_id (str): Owner whose semantic namespace should be loaded.

    Returns:
        list[Any]: Semantic store records for the owner.
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

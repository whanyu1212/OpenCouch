"""Reusable memory-store fixtures for backend tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from agent.memory.procedural_profile import (
    aadd_procedural_rule,
    aset_proactive_recall,
)
from agent.memory.store import MemoryStore, Namespace
from agent.memory.types.episodic import MoodArc, StoredSessionArc
from agent.memory.types.primitives import ConfidenceLevel, EntityRef, HotPathEdgeType
from agent.memory.types.procedural import (
    ProceduralProfile,
    ProceduralRule,
    ProceduralRuleSource,
)
from agent.memory.types.semantic import SemanticCategory, SemanticFact


def utc_z(dt: datetime) -> str:
    """Return an ISO-8601 UTC timestamp using the store's ``Z`` suffix convention."""

    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fixed_utc(*, minutes: int = 0) -> str:
    """Return a deterministic UTC timestamp offset from the test fixture epoch."""

    return utc_z(datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes))


def semantic_namespace(user_id: str) -> Namespace:
    """Return the semantic-memory namespace for ``user_id``."""

    return (user_id, "semantic")


def episodic_namespace(user_id: str) -> Namespace:
    """Return the episodic-memory namespace for ``user_id``."""

    return (user_id, "episodic")


async def seed_semantic_fact(
    store: MemoryStore,
    user_id: str,
    evidence_quote: str,
    *,
    fact_id: str | None = None,
    category: SemanticCategory = "relationship",
    subject: EntityRef | dict[str, str] | None = None,
    predicate: HotPathEdgeType = "KNOWS",
    object: EntityRef | dict[str, str] | None = None,
    confidence: ConfidenceLevel = "high",
    source_session_id: str = "seed-session",
    source_turn_index: int = 0,
    created_at: str | None = None,
    last_referenced_at: str | None = None,
    dormant_at: str | None = None,
    superseded_by: str | None = None,
    user_visible: bool = True,
    write_reason: str = "seeded test fact",
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
    value_overrides: dict[str, Any] | None = None,
) -> SemanticFact:
    """Seed one validated semantic fact and return the stored model."""

    fact_key = fact_id or f"fact-{uuid4()}"
    timestamp = created_at or fixed_utc()
    fact = SemanticFact(
        id=fact_key,
        category=category,
        subject=EntityRef.model_validate(
            subject or {"type": "User", "identifier": user_id}
        ),
        predicate=predicate,
        object=EntityRef.model_validate(
            object or {"type": "Person", "identifier": "Sarah"}
        ),
        evidence_quote=evidence_quote,
        confidence=confidence,
        source_session_id=source_session_id,
        source_turn_index=source_turn_index,
        created_at=timestamp,
        last_referenced_at=last_referenced_at or timestamp,
        dormant_at=dormant_at,
        superseded_by=superseded_by,
        user_visible=user_visible,
        write_reason=write_reason,
    )
    value = fact.model_dump(mode="json")
    if value_overrides:
        value.update(value_overrides)

    await store.aput(
        semantic_namespace(user_id),
        fact.id,
        value,
        embedding=embedding,
        embedding_model=embedding_model,
    )
    return fact


async def seed_episodic_arc(
    store: MemoryStore,
    user_id: str,
    summary: str,
    *,
    arc_id: str | None = None,
    session_id: str = "seed-session",
    primary_themes: list[str] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    duration_seconds: int = 1800,
    turn_count: int = 8,
    mood_opened: str = "tense",
    mood_closed: str = "steadier",
    open_loops: list[str] | None = None,
    resolved_threads: list[str] | None = None,
    approach_used: str | None = None,
    created_at: str | None = None,
    last_referenced_at: str | None = None,
    user_visible: bool = True,
    write_reason: str = "seeded test episode",
    crisis_level_max: int = 0,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
    value_overrides: dict[str, Any] | None = None,
) -> StoredSessionArc:
    """Seed one validated episodic session arc and return the stored model."""

    ended = ended_at or fixed_utc(minutes=30)
    arc = StoredSessionArc(
        id=arc_id or f"episode-{uuid4()}",
        owner_id=user_id,
        session_id=session_id,
        started_at=started_at or fixed_utc(),
        ended_at=ended,
        duration_seconds=duration_seconds,
        turn_count=turn_count,
        primary_themes=primary_themes or [],
        summary=summary,
        mood_arc=MoodArc(opened=mood_opened, closed=mood_closed),
        open_loops=open_loops or [],
        resolved_threads=resolved_threads or [],
        approach_used=approach_used,
        approach_context=None,
        created_at=created_at or ended,
        last_referenced_at=last_referenced_at or ended,
        user_visible=user_visible,
        write_reason=write_reason,
        crisis_level_max=crisis_level_max,
    )
    value = arc.model_dump(mode="json")
    if value_overrides:
        value.update(value_overrides)

    await store.aput(
        episodic_namespace(user_id),
        arc.id,
        value,
        embedding=embedding,
        embedding_model=embedding_model,
    )
    return arc


async def seed_procedural_profile(
    store: MemoryStore,
    user_id: str,
    rules: list[str],
    *,
    proactive_recall_enabled: bool = False,
    evidence: list[str] | None = None,
    confidence: ConfidenceLevel = "high",
    source: ProceduralRuleSource = "manual",
    added_at: str | None = None,
    write_reason: str = "seeded test procedural rule",
) -> ProceduralProfile:
    """Seed procedural rules through production helpers and return the profile."""

    profile: ProceduralProfile | None = None
    for rule_text in rules:
        profile = await aadd_procedural_rule(
            store,
            user_id=user_id,
            rule=ProceduralRule(
                rule=rule_text,
                evidence=evidence or [],
                confidence=confidence,
                added_at=added_at or fixed_utc(),
                source=source,
                write_reason=write_reason,
            ),
        )

    profile = await aset_proactive_recall(
        store,
        user_id=user_id,
        enabled=proactive_recall_enabled,
    )
    return profile


__all__ = [
    "episodic_namespace",
    "fixed_utc",
    "seed_episodic_arc",
    "seed_procedural_profile",
    "seed_semantic_fact",
    "semantic_namespace",
    "utc_z",
]

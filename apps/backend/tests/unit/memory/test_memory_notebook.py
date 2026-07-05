"""Tests for the read-only memory notebook view."""

from __future__ import annotations

import pytest

from agent.memory.notebook import build_memory_notebook
from agent.memory.operations.procedural_profile import aput_procedural_profile
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.types import (
    EntityRef,
    MoodArc,
    ProceduralProfile,
    ProceduralRule,
    SemanticFact,
    StoredSessionArc,
)


def _entity(entity_type: str, identifier: str) -> EntityRef:
    return EntityRef(type=entity_type, identifier=identifier)  # type: ignore[arg-type]


def _semantic_fact(
    fact_id: str,
    *,
    category: str,
    predicate: str,
    obj: EntityRef,
    evidence_quote: str,
    user_visible: bool = True,
) -> SemanticFact:
    return SemanticFact(
        id=fact_id,
        category=category,  # type: ignore[arg-type]
        subject=_entity("User", "user"),
        predicate=predicate,  # type: ignore[arg-type]
        object=obj,
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id="session-1",
        source_turn_index=3,
        created_at="2026-07-01T00:00:00Z",
        last_referenced_at="2026-07-02T00:00:00Z",
        user_visible=user_visible,
        write_timing="session_end",
        write_reason="policy allowed session-end write",
        policy_version="phase1_v1",
    )


def _session_arc() -> StoredSessionArc:
    return StoredSessionArc(
        id="arc-1",
        owner_id="user-1",
        session_id="session-2",
        started_at="2026-07-03T00:00:00Z",
        ended_at="2026-07-03T00:30:00Z",
        duration_seconds=1800,
        turn_count=6,
        primary_themes=["work stress", "sleep"],
        summary="The user connected work stress with disrupted sleep.",
        mood_arc=MoodArc(opened="tense", closed="calmer"),
        open_loops=["try a wind-down routine"],
        resolved_threads=["named the stressor"],
        approach_used="cbt",
        created_at="2026-07-03T00:31:00Z",
        last_referenced_at="2026-07-04T00:00:00Z",
        write_reason="session summary",
        policy_version="phase5_v1",
        crisis_level_max=0,
    )


@pytest.mark.asyncio
async def test_memory_notebook_groups_visible_records_by_topic() -> None:
    store = OpenCouchMemoryStore()
    owner_id = "user-1"
    preference = _semantic_fact(
        "fact-preference",
        category="preference",
        predicate="WANTS",
        obj=_entity("Goal", "short step-by-step plans"),
        evidence_quote="I prefer short step-by-step plans.",
    )
    coping = _semantic_fact(
        "fact-coping",
        category="coping_strategy",
        predicate="USES",
        obj=_entity("CopingStrategy", "box breathing"),
        evidence_quote="Box breathing helps me settle.",
    )
    hidden_trigger = _semantic_fact(
        "fact-hidden",
        category="trigger",
        predicate="WORRIES_ABOUT",
        obj=_entity("Concern", "crowds"),
        evidence_quote="Crowds make me anxious.",
        user_visible=False,
    )
    await store.aput(
        (owner_id, "semantic"),
        preference.id,
        preference.model_dump(mode="json"),
    )
    await store.aput((owner_id, "semantic"), coping.id, coping.model_dump(mode="json"))
    await store.aput(
        (owner_id, "semantic"),
        hidden_trigger.id,
        hidden_trigger.model_dump(mode="json"),
    )

    notebook = await build_memory_notebook(store, owner_id=owner_id)

    assert notebook.owner_id == owner_id
    assert notebook.counts.semantic == 2
    assert notebook.counts.total_entries == 2
    assert [topic.id for topic in notebook.topics] == [
        "preferences",
        "coping_strategies",
    ]
    assert notebook.topics[0].entries[0].id == "fact-preference"
    assert notebook.topics[0].entries[0].summary == (
        "I prefer short step-by-step plans."
    )
    assert notebook.topics[1].entries[0].title == "user uses box breathing"


@pytest.mark.asyncio
async def test_memory_notebook_includes_provenance_and_metadata() -> None:
    store = OpenCouchMemoryStore()
    owner_id = "user-1"
    fact = _semantic_fact(
        "fact-goal",
        category="goal",
        predicate="WANTS",
        obj=_entity("Goal", "better sleep"),
        evidence_quote="I want to sleep better.",
    )
    await store.aput((owner_id, "semantic"), fact.id, fact.model_dump(mode="json"))

    notebook = await build_memory_notebook(store, owner_id=owner_id)
    entry = notebook.topics[0].entries[0]

    assert entry.category == "goal"
    assert entry.provenance.source_session_id == "session-1"
    assert entry.provenance.source_turn_index == 3
    assert entry.provenance.confidence == "high"
    assert entry.provenance.write_reason == "policy allowed session-end write"
    assert entry.metadata["predicate"] == "WANTS"
    assert entry.metadata["object"] == {"type": "Goal", "identifier": "better sleep"}


@pytest.mark.asyncio
async def test_memory_notebook_includes_procedural_rules_and_session_arcs() -> None:
    store = OpenCouchMemoryStore()
    owner_id = "user-1"
    await aput_procedural_profile(
        store,
        user_id=owner_id,
        profile=ProceduralProfile(
            proactive_recall_enabled=True,
            rules=[
                ProceduralRule(
                    id="rule-1",
                    rule="Use concise, grounded language.",
                    evidence=["user asked for concise replies"],
                    confidence="medium",
                    added_at="2026-07-01T00:00:00Z",
                    source="explicit_user",
                    write_timing="immediate",
                    write_reason="explicit preference",
                    policy_version="phase1_v1",
                )
            ],
        ),
    )
    arc = _session_arc()
    await store.aput((owner_id, "episodic"), arc.id, arc.model_dump(mode="json"))

    notebook = await build_memory_notebook(store, owner_id=owner_id)

    assert notebook.proactive_recall_enabled is True
    assert notebook.counts.procedural_rules == 1
    assert notebook.counts.episodic == 1
    assert notebook.counts.total_entries == 2
    assert [topic.id for topic in notebook.topics] == [
        "procedural_rules",
        "session_arcs",
    ]

    procedural_entry = notebook.topics[0].entries[0]
    assert procedural_entry.kind == "procedural"
    assert procedural_entry.summary == "Use concise, grounded language."
    assert procedural_entry.metadata["source"] == "explicit_user"
    assert procedural_entry.metadata["evidence"] == ["user asked for concise replies"]

    episodic_entry = notebook.topics[1].entries[0]
    assert episodic_entry.kind == "episodic"
    assert episodic_entry.title == "work stress, sleep"
    assert episodic_entry.summary == (
        "The user connected work stress with disrupted sleep."
    )
    assert episodic_entry.provenance.source_session_id == "session-2"
    assert episodic_entry.metadata["primary_themes"] == ["work stress", "sleep"]


@pytest.mark.asyncio
async def test_memory_notebook_returns_empty_view_for_unknown_owner() -> None:
    store = OpenCouchMemoryStore()

    notebook = await build_memory_notebook(store, owner_id="missing-user")

    assert notebook.owner_id == "missing-user"
    assert notebook.topics == []
    assert notebook.counts.total_entries == 0
    assert notebook.proactive_recall_enabled is False

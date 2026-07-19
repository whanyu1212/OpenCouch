"""Read-only Claude-style notebook view over typed memory records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.store import MemoryStore, StoreRecord
from agent.memory.types import ProceduralRule, SemanticFact, StoredSessionArc

NotebookEntryKind = Literal["semantic", "episodic", "procedural"]


class MemoryNotebookProvenance(BaseModel):
    """Source and policy metadata attached to a notebook entry."""

    source_session_id: str | None = None
    source_turn_index: int | None = None
    created_at: str | None = None
    last_referenced_at: str | None = None
    confidence: str | None = None
    write_timing: str | None = None
    write_reason: str | None = None
    policy_version: str | None = None


class MemoryNotebookEntry(BaseModel):
    """One user-visible item in the memory notebook read model."""

    id: str
    kind: NotebookEntryKind
    title: str
    summary: str
    category: str
    provenance: MemoryNotebookProvenance = Field(
        default_factory=MemoryNotebookProvenance
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryNotebookTopic(BaseModel):
    """Topic bucket in the memory notebook."""

    id: str
    label: str
    entries: list[MemoryNotebookEntry] = Field(default_factory=list)


class MemoryNotebookCounts(BaseModel):
    """Visible entry counts by memory kind."""

    semantic: int = 0
    episodic: int = 0
    procedural_rules: int = 0
    total_entries: int = 0


class MemoryNotebook(BaseModel):
    """Claude-style grouped memory view for one owner."""

    owner_id: str
    topics: list[MemoryNotebookTopic] = Field(default_factory=list)
    counts: MemoryNotebookCounts = Field(default_factory=MemoryNotebookCounts)
    proactive_recall_enabled: bool = False


_SEMANTIC_TOPIC_BY_CATEGORY = {
    "preference": "preferences",
    "coping_strategy": "coping_strategies",
    "relationship": "relationships",
    "goal": "goals",
    "trigger": "triggers",
    "loss": "losses",
    "context": "context",
}

_TOPIC_LABELS = {
    "preferences": "Preferences",
    "coping_strategies": "Coping strategies",
    "relationships": "Relationships",
    "goals": "Goals",
    "triggers": "Triggers and sensitivities",
    "losses": "Losses",
    "context": "Context",
    "session_arcs": "Session arcs",
    "procedural_rules": "Response preferences",
}

_TOPIC_ORDER = (
    "preferences",
    "procedural_rules",
    "coping_strategies",
    "relationships",
    "goals",
    "triggers",
    "losses",
    "context",
    "session_arcs",
)

_EDGE_LABELS = {
    "KNOWS": "knows",
    "WORRIES_ABOUT": "worries about",
    "EXPERIENCED": "experienced",
    "USES": "uses",
    "WANTS": "wants",
    "PARTICIPATED_IN": "participated in",
    "MENTIONED_IN": "mentioned in",
}


async def build_memory_notebook(
    store: MemoryStore,
    *,
    owner_id: str,
    semantic_limit: int = 100,
    episodic_limit: int = 50,
    include_hidden: bool = False,
) -> MemoryNotebook:
    """Build a read-only notebook view from existing memory records.

    This service does not mutate memory, change retrieval behavior, or introduce a
    new persistence format. It groups typed records already stored in
    ``MemoryStore`` into a Claude-style inspection/read model.
    """

    semantic_records = await _records_for_namespace(
        store,
        (owner_id, "semantic"),
        limit=semantic_limit,
    )
    episodic_records = await _records_for_namespace(
        store,
        (owner_id, "episodic"),
        limit=episodic_limit,
    )
    procedural_profile = await aget_procedural_profile(store, user_id=owner_id)

    topic_entries: dict[str, list[MemoryNotebookEntry]] = defaultdict(list)
    counts = MemoryNotebookCounts()

    for record in semantic_records:
        if counts.semantic >= semantic_limit:
            break
        try:
            fact = SemanticFact.model_validate(record.value)
        except ValidationError:
            continue
        if not include_hidden and not _is_visible_memory(fact):
            continue
        topic_id = _SEMANTIC_TOPIC_BY_CATEGORY.get(fact.category, "context")
        topic_entries[topic_id].append(_semantic_entry(record, fact))
        counts.semantic += 1

    procedural_rules = list(procedural_profile.rules)
    if include_hidden:
        procedural_rules.extend(procedural_profile.archived_rules)
    for rule in procedural_rules:
        if not include_hidden and not _is_visible_memory(rule):
            continue
        topic_entries["procedural_rules"].append(_procedural_entry(rule))
        counts.procedural_rules += 1

    for record in episodic_records:
        if counts.episodic >= episodic_limit:
            break
        try:
            arc = StoredSessionArc.model_validate(record.value)
        except ValidationError:
            continue
        if not include_hidden and not _is_visible_memory(arc):
            continue
        topic_entries["session_arcs"].append(_episodic_entry(record, arc))
        counts.episodic += 1

    counts.total_entries = counts.semantic + counts.episodic + counts.procedural_rules

    return MemoryNotebook(
        owner_id=owner_id,
        topics=[
            MemoryNotebookTopic(
                id=topic_id,
                label=_TOPIC_LABELS[topic_id],
                entries=topic_entries[topic_id],
            )
            for topic_id in _TOPIC_ORDER
            if topic_entries.get(topic_id)
        ],
        counts=counts,
        proactive_recall_enabled=procedural_profile.proactive_recall_enabled,
    )


async def _records_for_namespace(
    store: MemoryStore,
    namespace: tuple[str, str],
    *,
    limit: int,
) -> list[StoreRecord]:
    if limit <= 0:
        return []
    record_count = await store.arecord_count(namespace)
    if record_count == 0:
        return []
    # ``asearch(query=None)`` returns insertion-order slices. Fetch the whole
    # namespace, then apply the notebook's visible-entry limit after validation
    # and hidden/superseded filtering so an old hidden prefix does not mask later
    # visible replacement records.
    return await store.asearch(namespace, query=None, limit=record_count)


def _is_visible_memory(record: Any) -> bool:
    return (
        bool(getattr(record, "user_visible", True))
        and not getattr(
            record,
            "dormant_at",
            None,
        )
        and not getattr(record, "superseded_by", None)
    )


def _semantic_entry(record: StoreRecord, fact: SemanticFact) -> MemoryNotebookEntry:
    subject = fact.subject.identifier
    obj = fact.object.identifier
    edge = _EDGE_LABELS.get(fact.predicate, fact.predicate.lower())
    return MemoryNotebookEntry(
        id=fact.id or record.key,
        kind="semantic",
        title=f"{subject} {edge} {obj}",
        summary=fact.evidence_quote,
        category=fact.category,
        provenance=MemoryNotebookProvenance(
            source_session_id=fact.source_session_id,
            source_turn_index=fact.source_turn_index,
            created_at=fact.created_at,
            last_referenced_at=fact.last_referenced_at,
            confidence=fact.confidence,
            write_timing=fact.write_timing,
            write_reason=fact.write_reason or None,
            policy_version=fact.policy_version,
        ),
        metadata={
            "subject": fact.subject.model_dump(mode="json"),
            "predicate": fact.predicate,
            "object": fact.object.model_dump(mode="json"),
        },
    )


def _procedural_entry(rule: ProceduralRule) -> MemoryNotebookEntry:
    return MemoryNotebookEntry(
        id=rule.id,
        kind="procedural",
        title="Response preference",
        summary=rule.rule,
        category="procedural_rule",
        provenance=MemoryNotebookProvenance(
            created_at=rule.added_at,
            confidence=rule.confidence,
            write_timing=rule.write_timing,
            write_reason=rule.write_reason or None,
            policy_version=rule.policy_version,
        ),
        metadata={
            "source": rule.source,
            "evidence": list(rule.evidence),
        },
    )


def _episodic_entry(record: StoreRecord, arc: StoredSessionArc) -> MemoryNotebookEntry:
    title = _session_arc_title(arc)
    return MemoryNotebookEntry(
        id=arc.id or record.key,
        kind="episodic",
        title=title,
        summary=arc.summary,
        category="session_arc",
        provenance=MemoryNotebookProvenance(
            source_session_id=arc.session_id,
            created_at=arc.created_at,
            last_referenced_at=arc.last_referenced_at,
            write_timing=arc.write_timing,
            write_reason=arc.write_reason or None,
            policy_version=arc.policy_version,
        ),
        metadata={
            "session_id": arc.session_id,
            "started_at": arc.started_at,
            "ended_at": arc.ended_at,
            "turn_count": arc.turn_count,
            "primary_themes": list(arc.primary_themes),
            "open_loops": list(arc.open_loops),
            "resolved_threads": list(arc.resolved_threads),
            "mood_arc": arc.mood_arc.model_dump(mode="json"),
            "approach_used": arc.approach_used,
            "crisis_level_max": arc.crisis_level_max,
        },
    )


def _session_arc_title(arc: StoredSessionArc) -> str:
    if arc.primary_themes:
        return ", ".join(arc.primary_themes[:2])
    return f"Session {arc.session_id}"


__all__ = [
    "MemoryNotebook",
    "MemoryNotebookCounts",
    "MemoryNotebookEntry",
    "MemoryNotebookProvenance",
    "MemoryNotebookTopic",
    "build_memory_notebook",
]

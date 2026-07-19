"""User-directed memory-management helpers.

These helpers back conversational memory-management turns. They intentionally
operate below runtime services and above the raw store so nodes can stay small while
still keeping memory edits scoped to one owner.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from agent.memory.operations.procedural_profile import (
    adelete_procedural_rule,
    aget_procedural_profile,
    aset_proactive_recall,
    aupsert_procedural_rule,
    build_procedural_rule,
)
from agent.memory.operations.reconciliation import filter_active_semantic_records
from agent.memory.store import MemoryStore, Namespace
from agent.memory.text_tokens import tokenize_meaningful
from llm.base import BaseLLMClient

MemoryControlKind = Literal["fact", "session", "rule"]


class MemoryControlTarget(TypedDict):
    """Serializable reference to one user-visible memory item."""

    kind: MemoryControlKind
    key: str
    preview: str
    namespace: list[str]
    rule_id: str | None


def _entity_identifier(entity: object) -> str:
    """Return a readable identifier from a serialized memory entity.

    Args:
        entity: Serialized entity object from a memory payload.

    Returns:
        Entity identifier, or ``"unknown"`` when unavailable.
    """

    if isinstance(entity, dict):
        identifier = entity.get("identifier")
        if identifier:
            return str(identifier)
    return "unknown"


def _semantic_preview(value: dict[str, Any]) -> str:
    """Build a compact preview for a semantic fact.

    Args:
        value: Serialized semantic fact payload.

    Returns:
        Human-readable fact preview.
    """

    category = str(value.get("category", "fact"))
    predicate = str(value.get("predicate", ""))
    object_id = _entity_identifier(value.get("object"))
    quote = str(value.get("evidence_quote", "")).strip()
    if quote and not predicate:
        return quote
    if quote:
        return f'{category}: {predicate} {object_id} — "{quote}"'
    return f"{category}: {predicate} {object_id}"


def _episodic_preview(value: dict[str, Any]) -> str:
    """Build a compact preview for an episodic session arc.

    Args:
        value: Serialized session arc payload.

    Returns:
        Human-readable session preview.
    """

    summary = str(value.get("summary", "")).strip()
    ended_at = str(value.get("ended_at", ""))
    date = ended_at[:10] if len(ended_at) >= 10 else "session"
    return f"{date}: {summary}" if summary else date


def _target_from_record(
    *,
    kind: MemoryControlKind,
    namespace: Namespace,
    key: str,
    preview: str,
    rule_id: str | None = None,
) -> MemoryControlTarget:
    """Create a serializable target from a store record reference.

    Args:
        kind: Memory item kind.
        namespace: Store namespace containing the item.
        key: Store key for the item.
        preview: User-visible preview text.
        rule_id: Optional procedural rule id.

    Returns:
        Serializable memory-management target.
    """

    return {
        "kind": kind,
        "namespace": list(namespace),
        "key": key,
        "rule_id": rule_id,
        "preview": preview,
    }


async def list_memory_for_owner(
    store: MemoryStore,
    *,
    owner_id: str,
    limit_per_kind: int = 5,
) -> dict[str, list[str]]:
    """Return user-visible memory previews for one owner.

    Args:
        store: Memory store to inspect.
        owner_id: Owner namespace to read.
        limit_per_kind: Maximum previews returned for each memory kind.

    Returns:
        Dict with ``facts``, ``sessions``, and ``rules`` preview lists.
    """

    semantic_records = await store.asearch(
        (owner_id, "semantic"), query=None, limit=limit_per_kind
    )
    semantic_records = filter_active_semantic_records(semantic_records)
    episodic_records = await store.asearch(
        (owner_id, "episodic"), query=None, limit=limit_per_kind
    )
    profile = await aget_procedural_profile(store, user_id=owner_id)

    return {
        "facts": [_semantic_preview(record.value) for record in semantic_records],
        "sessions": [_episodic_preview(record.value) for record in episodic_records],
        "rules": [rule.rule for rule in profile.rules[:limit_per_kind]],
    }


async def set_memory_recall(
    store: MemoryStore,
    *,
    owner_id: str,
    enabled: bool,
) -> bool:
    """Set proactive recall for one owner.

    Args:
        store: Memory store to update.
        owner_id: Owner namespace to update.
        enabled: Desired proactive-recall state.

    Returns:
        The stored proactive-recall state.
    """

    await aset_proactive_recall(store, user_id=owner_id, enabled=enabled)
    return enabled


async def save_preference_rule(
    store: MemoryStore,
    *,
    owner_id: str,
    rule_text: str,
    evidence: str,
    llm_client: BaseLLMClient,
) -> str:
    """Save one explicit procedural preference.

    Args:
        store: Memory store to update.
        owner_id: Owner namespace to update.
        rule_text: User-facing rule text to persist.
        evidence: User message that justifies the rule.
        llm_client: Control LLM used for procedural reconciliation.

    Returns:
        The final stored rule text.
    """

    rule = build_procedural_rule(
        rule_text=rule_text,
        evidence=[evidence],
        confidence="high",
        source="explicit_user",
        write_reason="explicit conversational memory-management request",
    )
    result = await aupsert_procedural_rule(
        store,
        user_id=owner_id,
        rule=rule,
        llm_client=llm_client,
    )
    stored_rule = result.profile.rules[-1] if result.profile.rules else rule
    return stored_rule.rule


def _text_matches_query(text: str, query: str) -> bool:
    """Return whether ``text`` is a useful lexical match for ``query``.

    Args:
        text: Candidate text to inspect.
        query: User-supplied query text.

    Returns:
        ``True`` when query-token recall clears the match threshold.
    """

    query_tokens = tokenize_meaningful(query)
    if not query_tokens:
        return False
    text_tokens = tokenize_meaningful(text)
    if not text_tokens:
        return False
    overlap = query_tokens & text_tokens
    return bool(overlap) and (len(overlap) / len(query_tokens)) >= 0.33


async def find_memory_targets(
    store: MemoryStore,
    *,
    owner_id: str,
    query: str,
    limit: int = 5,
) -> list[MemoryControlTarget]:
    """Find user-visible memory targets matching a natural-language query.

    Args:
        store: Memory store to search.
        owner_id: Owner namespace to search.
        query: User-supplied query text.
        limit: Maximum targets to return.

    Returns:
        Matching targets across semantic facts, episodic arcs, and procedural
        rules.
    """

    targets: list[MemoryControlTarget] = []

    semantic_namespace = (owner_id, "semantic")
    semantic_records = await store.asearch(semantic_namespace, query=query, limit=limit)
    for record in filter_active_semantic_records(semantic_records):
        targets.append(
            _target_from_record(
                kind="fact",
                namespace=semantic_namespace,
                key=record.key,
                preview=_semantic_preview(record.value),
            )
        )

    episodic_namespace = (owner_id, "episodic")
    episodic_records = await store.asearch(episodic_namespace, query=query, limit=limit)
    for record in episodic_records:
        targets.append(
            _target_from_record(
                kind="session",
                namespace=episodic_namespace,
                key=record.key,
                preview=_episodic_preview(record.value),
            )
        )

    profile = await aget_procedural_profile(store, user_id=owner_id)
    for rule in profile.rules:
        haystack = " ".join([rule.rule, *rule.evidence])
        if not _text_matches_query(haystack, query):
            continue
        targets.append(
            _target_from_record(
                kind="rule",
                namespace=(owner_id, "procedural"),
                key=rule.id,
                rule_id=rule.id,
                preview=rule.rule,
            )
        )
        if len(targets) >= limit:
            break

    return targets[:limit]


async def find_memory_target_by_index(
    store: MemoryStore,
    *,
    owner_id: str,
    kind: MemoryControlKind,
    index_1based: int,
) -> MemoryControlTarget | None:
    """Return a memory target by displayed 1-based index.

    Args:
        store: Memory store to inspect.
        owner_id: Owner namespace to read.
        kind: Memory kind to index into.
        index_1based: User-facing index, starting at 1.

    Returns:
        Matching target, or ``None`` when the index is out of range.
    """

    if index_1based < 1:
        return None

    if kind == "rule":
        profile = await aget_procedural_profile(store, user_id=owner_id)
        if index_1based > len(profile.rules):
            return None
        rule = profile.rules[index_1based - 1]
        return _target_from_record(
            kind="rule",
            namespace=(owner_id, "procedural"),
            key=rule.id,
            rule_id=rule.id,
            preview=rule.rule,
        )

    namespace_kind = "semantic" if kind == "fact" else "episodic"
    namespace = (owner_id, namespace_kind)
    records = await store.asearch(namespace, query=None, limit=1000)
    if kind == "fact":
        records = filter_active_semantic_records(records)
    if index_1based > len(records):
        return None
    record = records[index_1based - 1]
    preview = (
        _semantic_preview(record.value)
        if kind == "fact"
        else _episodic_preview(record.value)
    )
    return _target_from_record(
        kind=kind,
        namespace=namespace,
        key=record.key,
        preview=preview,
    )


async def delete_memory_target(
    store: MemoryStore,
    *,
    owner_id: str,
    target: MemoryControlTarget,
) -> bool:
    """Delete a confirmed memory target.

    Args:
        store: Memory store to update.
        owner_id: Owner namespace expected for this deletion.
        target: Confirmed deletion target.

    Returns:
        ``True`` when a record/rule was deleted.
    """

    kind = target["kind"]
    if kind == "rule":
        rule_id = target.get("rule_id") or target.get("key")
        if not rule_id:
            return False
        return (
            await adelete_procedural_rule(store, user_id=owner_id, rule_id=rule_id)
        ) is not None

    namespace = tuple(target["namespace"])
    if not namespace or namespace[0] != owner_id:
        return False
    return await store.adelete(namespace, target["key"])

"""Memory inspection and deletion endpoints.

GET    /api/memory/status: user-facing memory counts and recall toggle.
PATCH  /api/memory/recall: set proactive recall on or off.
DELETE /api/memory/facts/{n}: delete one semantic fact by index.
DELETE /api/memory/sessions/{n}: delete one episodic arc by index.
DELETE /api/memory/rules/{n}: delete one procedural rule by index.

These endpoints mirror the CLI's ``/memory status``, ``/memory
forget fact <n>``, ``/memory forget session <n>``, and ``/memory
forget rule <n>`` commands. The indexes are 1-based and match the
order shown by ``/memory list`` (insertion order).

Bulk destructive ops (``/memory clear``, ``/memory purge-crisis``)
are intentionally not exposed over the API because they require typed
confirmation that a REST endpoint can't enforce safely. They remain
CLI-only until a frontend with a proper confirmation flow exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from agent.memory.reconciliation import filter_active_semantic_records
from agent.memory.procedural import (
    adelete_procedural_rule,
    aget_procedural_profile,
    aset_proactive_recall,
)
from agent.memory.store import StoreRecord
from agent.persistence import PersistentAgentRuntime
from api.dependencies import get_runtime
from api.models import (
    DeleteResponse,
    MemoryRecallUpdateRequest,
    MemoryRecallUpdateResponse,
    MemoryStatusResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])


def _resolve_owner_id(
    user_id: str | None,
    thread_id: str,
) -> str:
    """Resolve the effective owner_id from user_id or thread_id.

    Mirrors ``RunnerSession.owner_id()`` in the CLI: user_id if set,
    thread_id as fallback. Every memory operation is scoped to this
    owner so cross-user access is not reachable.

    Args:
        user_id: Optional stable owner identifier.
        thread_id: Thread identifier used as the fallback owner.

    Returns:
        Effective owner id for memory operations.
    """

    return user_id or thread_id


async def _list_active_semantic_records(
    runtime: PersistentAgentRuntime,
    *,
    owner_id: str,
) -> list[StoreRecord]:
    """Return the active semantic records for this owner.

    Args:
        runtime: Runtime that owns the memory store.
        owner_id: Owner namespace to query.

    Returns:
        Active semantic records after tombstone reconciliation.
    """

    store = runtime.memory_store
    records = await store.asearch((owner_id, "semantic"), query=None, limit=1000)
    return filter_active_semantic_records(records)


@router.get("/status", response_model=MemoryStatusResponse)
async def memory_status(
    thread_id: str = Query(description="Thread to scope the status to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> MemoryStatusResponse:
    """Return user-facing memory counts and the recall toggle state.

    Args:
        thread_id: Thread to scope the status to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        Memory status payload for the effective owner.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    store = runtime.memory_store
    crisis_log = runtime.crisis_log_backend
    session_feedback = runtime.session_feedback_backend

    crisis_count = await crisis_log.arecord_count()
    feedback_count = await session_feedback.arecord_count()
    profile = await aget_procedural_profile(store, user_id=owner_id)
    semantic_records = await _list_active_semantic_records(runtime, owner_id=owner_id)
    episodic_count = await store.arecord_count((owner_id, "episodic"))

    counts = {
        "semantic": len(semantic_records),
        "episodic": episodic_count,
        "procedural": len(profile.rules),
    }

    return MemoryStatusResponse(
        memory_mode=str(runtime.memory_mode.value),
        owner_id=owner_id,
        counts=counts,
        crisis_log_count=crisis_count,
        session_feedback_count=feedback_count,
        proactive_recall_enabled=profile.proactive_recall_enabled,
    )


@router.patch("/recall", response_model=MemoryRecallUpdateResponse)
async def update_memory_recall(
    payload: MemoryRecallUpdateRequest,
    thread_id: str = Query(description="Thread to scope the update to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> MemoryRecallUpdateResponse:
    """Set the proactive recall toggle for the effective owner.

    Args:
        payload: Desired proactive recall state.
        thread_id: Thread to scope the update to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        Updated proactive recall state for the effective owner.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    profile = await aset_proactive_recall(
        runtime.memory_store,
        user_id=owner_id,
        enabled=payload.enabled,
    )
    state_text = "on" if profile.proactive_recall_enabled else "off"
    return MemoryRecallUpdateResponse(
        owner_id=owner_id,
        proactive_recall_enabled=profile.proactive_recall_enabled,
        detail=f"Proactive recall is now {state_text}.",
    )


@router.get("/facts")
async def list_facts(
    thread_id: str = Query(description="Thread to scope the listing to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> list[dict]:
    """List all semantic facts for this owner.

    Args:
        thread_id: Thread to scope the listing to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        JSON-serializable semantic fact rows.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    records = await _list_active_semantic_records(runtime, owner_id=owner_id)
    return [
        {
            "index": i + 1,
            "key": r.key,
            "category": r.value.get("category", ""),
            "predicate": r.value.get("predicate", ""),
            "subject": r.value.get("subject", {}).get("identifier", ""),
            "object": r.value.get("object", {}).get("identifier", ""),
            "evidence_quote": r.value.get("evidence_quote", ""),
            "confidence": r.value.get("confidence", ""),
            "created_at": r.value.get("created_at", ""),
        }
        for i, r in enumerate(records)
    ]


@router.get("/sessions")
async def list_sessions(
    thread_id: str = Query(description="Thread to scope the listing to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> list[dict]:
    """List all episodic session arcs for this owner.

    Args:
        thread_id: Thread to scope the listing to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        JSON-serializable episodic session rows.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    store = runtime.memory_store
    namespace = (owner_id, "episodic")
    records = await store.asearch(namespace, query=None, limit=1000)
    return [
        {
            "index": i + 1,
            "key": r.key,
            "session_id": r.value.get("session_id", ""),
            "summary": r.value.get("summary", ""),
            "themes": r.value.get("primary_themes", []),
            "mood_opened": (r.value.get("mood_arc") or {}).get("opened", ""),
            "mood_closed": (r.value.get("mood_arc") or {}).get("closed", ""),
            "turn_count": r.value.get("turn_count", 0),
            "ended_at": r.value.get("ended_at", ""),
        }
        for i, r in enumerate(records)
    ]


@router.get("/rules")
async def list_rules(
    thread_id: str = Query(description="Thread to scope the listing to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> list[dict]:
    """List all procedural rules for this owner.

    Args:
        thread_id: Thread to scope the listing to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        JSON-serializable procedural rule rows.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    store = runtime.memory_store
    profile = await aget_procedural_profile(store, user_id=owner_id)
    return [
        {
            "index": i + 1,
            "rule": rule.rule,
            "evidence": rule.evidence,
            "confidence": rule.confidence,
            "added_at": rule.added_at,
        }
        for i, rule in enumerate(profile.rules)
    ]


@router.delete("/facts/{index}", response_model=DeleteResponse)
async def delete_fact(
    index: int,
    thread_id: str = Query(description="Thread to scope the deletion to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> DeleteResponse:
    """Delete one semantic fact by its 1-based index.

    The index matches the ``#`` column in ``/memory list`` (insertion
    order) for active, user-visible facts only. Returns 404 if the
    index is out of range or no active facts exist.

    Args:
        index: One-based fact index from the memory list.
        thread_id: Thread to scope the deletion to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        Delete result with a user-facing detail string.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    store = runtime.memory_store
    namespace = (owner_id, "semantic")

    records = await _list_active_semantic_records(runtime, owner_id=owner_id)
    if not records:
        raise HTTPException(
            status_code=404,
            detail="No active semantic facts for this owner.",
        )
    if index < 1 or index > len(records):
        raise HTTPException(
            status_code=404,
            detail=f"Fact #{index} does not exist (only {len(records)} fact(s)).",
        )

    target = records[index - 1]
    deleted = await store.adelete(namespace, target.key)
    return DeleteResponse(
        deleted=deleted,
        detail=f"Deleted fact #{index}. {len(records) - 1} remaining.",
    )


@router.delete("/sessions/{index}", response_model=DeleteResponse)
async def delete_session(
    index: int,
    thread_id: str = Query(description="Thread to scope the deletion to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> DeleteResponse:
    """Delete one episodic session arc by its 1-based index.

    Args:
        index: One-based session index from the memory list.
        thread_id: Thread to scope the deletion to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        Delete result with a user-facing detail string.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    store = runtime.memory_store
    namespace = (owner_id, "episodic")

    records = await store.asearch(namespace, query=None, limit=1000)
    if not records:
        raise HTTPException(
            status_code=404, detail="No episodic sessions for this owner."
        )
    if index < 1 or index > len(records):
        raise HTTPException(
            status_code=404,
            detail=f"Session #{index} does not exist (only {len(records)} arc(s)).",
        )

    target = records[index - 1]
    deleted = await store.adelete(namespace, target.key)
    return DeleteResponse(
        deleted=deleted,
        detail=f"Deleted session #{index}. {len(records) - 1} remaining.",
    )


@router.delete("/rules/{index}", response_model=DeleteResponse)
async def delete_rule(
    index: int,
    thread_id: str = Query(description="Thread to scope the deletion to."),
    user_id: str | None = Query(default=None, description="Optional owner override."),
    runtime: PersistentAgentRuntime = Depends(get_runtime),
) -> DeleteResponse:
    """Delete one procedural rule by its 1-based index.

    Args:
        index: One-based rule index from the memory list.
        thread_id: Thread to scope the deletion to.
        user_id: Optional owner override.
        runtime: Shared persistent agent runtime.

    Returns:
        Delete result with a user-facing detail string.
    """

    owner_id = _resolve_owner_id(user_id, thread_id)
    store = runtime.memory_store
    profile = await aget_procedural_profile(store, user_id=owner_id)

    if not profile.rules:
        raise HTTPException(
            status_code=404, detail="No procedural rules for this owner."
        )
    if index < 1 or index > len(profile.rules):
        raise HTTPException(
            status_code=404,
            detail=f"Rule #{index} does not exist (only {len(profile.rules)} rule(s)).",
        )

    deleted = await adelete_procedural_rule(
        store,
        user_id=owner_id,
        rule_id=profile.rules[index - 1].id,
    )
    if deleted is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rule #{index} no longer exists for this owner.",
        )
    profile, removed_rule = deleted
    return DeleteResponse(
        deleted=True,
        detail=f"Deleted rule #{index} ({removed_rule.rule[:60]}). "
        f"{len(profile.rules)} remaining.",
    )

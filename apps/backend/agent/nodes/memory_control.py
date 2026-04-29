"""Memory-control node for explicit user memory commands."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.control import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
    list_memory_for_owner,
    save_preference_rule,
    set_memory_recall,
)
from agent.memory.modes import MemoryMode
from agent.memory.procedural import aget_procedural_profile
from agent.models import ResponseStyleType, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id


def _base_delta(response_text: str, *, started_at: float) -> dict[str, Any]:
    """Return the shared response delta for memory-control turns.

    Args:
        response_text: User-facing operational reply.
        started_at: Monotonic start timestamp.

    Returns:
        Partial graph state update for memory-control turns.
    """

    return {
        "route": "memory_control",
        "response_style": "memory_control",
        "response_style_source": "memory_control_gate",
        "response_style_type": ResponseStyleType.OPERATIONAL,
        "response_kind": ResponseCategory.THERAPEUTIC,
        "response_text": response_text,
        "diagnostics": {
            "memory_control_ms": round((time.monotonic() - started_at) * 1000, 2)
        },
    }


def _empty_memory_reply() -> str:
    """Return a concise reply for an empty memory store.

    Returns:
        User-facing empty-memory reply.
    """

    return (
        "I don't have any saved facts, session summaries, or style rules for you "
        "right now."
    )


def _format_memory_overview(previews: dict[str, list[str]]) -> str:
    """Render memory previews into a short user-facing list.

    Args:
        previews: Memory preview rows grouped by facts, sessions, and rules.

    Returns:
        User-facing memory overview.
    """

    lines: list[str] = []
    labels = {
        "facts": "Saved facts",
        "sessions": "Session summaries",
        "rules": "Style preferences",
    }
    for key in ("facts", "sessions", "rules"):
        items = previews.get(key, [])
        if not items:
            continue
        lines.append(f"{labels[key]}:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))

    if not lines:
        return _empty_memory_reply()
    return "Here's what I currently have saved:\n\n" + "\n".join(lines)


def _pending_delete_reply(target: MemoryControlTarget) -> str:
    """Return confirmation wording for a pending memory deletion.

    Args:
        target: Memory target selected for deletion.

    Returns:
        User-facing deletion confirmation prompt.
    """

    return (
        f"I found this saved {target['kind']}:\n\n"
        f'"{target["preview"]}"\n\n'
        "Do you want me to delete it?"
    )


def _multiple_matches_reply(targets: list[MemoryControlTarget]) -> str:
    """Return disambiguation wording for multiple deletion matches.

    Args:
        targets: Candidate memory targets matching the deletion query.

    Returns:
        User-facing disambiguation prompt.
    """

    lines = [
        "I found multiple saved memories that might match. Which one should I delete?"
    ]
    lines.extend(
        f"{index}. {target['kind']}: {target['preview']}"
        for index, target in enumerate(targets, start=1)
    )
    return "\n\n".join([lines[0], "\n".join(lines[1:])])


def _incognito_reply() -> str:
    """Return the no-op reply for incognito memory mode.

    Returns:
        User-facing no-persistent-memory reply.
    """

    return (
        "You're in guest mode, so I don't have persistent memory to show or edit "
        "for this session."
    )


async def _owner_or_reply(state: AgentState) -> tuple[str | None, str | None]:
    """Resolve the memory owner or return a user-facing failure reply.

    Args:
        state: Current graph state containing user/session identity.

    Returns:
        ``(owner_id, None)`` on success, otherwise ``(None, reply)``.
    """

    try:
        return resolve_owner_id(state), None
    except ValueError:
        return (
            None,
            "I don't have a stable memory owner for this conversation, so I can't "
            "show or edit saved memory here.",
        )


async def _handle_status(
    *,
    owner_id: str,
    runtime: Runtime[WorkflowContext],
) -> str:
    """Return memory status text for one owner.

    Args:
        owner_id: Owner whose memory status should be loaded.
        runtime: LangGraph runtime carrying memory dependencies.

    Returns:
        User-facing memory status text.
    """

    store = runtime.context.memory_store
    profile = await aget_procedural_profile(store, user_id=owner_id)
    fact_count = await store.arecord_count((owner_id, "semantic"))
    session_count = await store.arecord_count((owner_id, "episodic"))
    return (
        "Memory status:\n\n"
        f"Saved facts: {fact_count}\n"
        f"Session summaries: {session_count}\n"
        f"Style preferences: {len(profile.rules)}\n"
        f"Proactive recall: {'on' if profile.proactive_recall_enabled else 'off'}"
    )


async def run_memory_control_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Execute an explicit memory-control action.

    Args:
        state: Current graph state with ``memory_control.action`` set by the gate.
        runtime: LangGraph runtime carrying memory dependencies.

    Returns:
        Partial state update containing an operational reply and any
        memory-control state changes.
    """

    started_at = time.monotonic()
    action = (state.get("memory_control", {}) or {}).get("action", {}) or {}
    action_type = action.get("type")

    if runtime.context.memory_mode == MemoryMode.INCOGNITO:
        delta = _base_delta(_incognito_reply(), started_at=started_at)
        delta["memory_control"] = {"pending_action": None}
        return delta

    owner_id, failure_reply = await _owner_or_reply(state)
    if owner_id is None:
        delta = _base_delta(
            failure_reply or _empty_memory_reply(), started_at=started_at
        )
        delta["memory_control"] = {"pending_action": None}
        return delta

    store = runtime.context.memory_store

    if action_type == "status":
        delta = _base_delta(
            await _handle_status(owner_id=owner_id, runtime=runtime),
            started_at=started_at,
        )
        delta["memory_control"] = {"pending_action": None}
        return delta

    if action_type == "list":
        previews = await list_memory_for_owner(store, owner_id=owner_id)
        delta = _base_delta(_format_memory_overview(previews), started_at=started_at)
        delta["memory_control"] = {"pending_action": None}
        return delta

    if action_type == "set_recall":
        enabled = bool(action.get("enabled", False))
        await set_memory_recall(store, owner_id=owner_id, enabled=enabled)
        state_text = "on" if enabled else "off"
        reply = (
            f"I turned proactive recall {state_text}. "
            "Style preferences can still shape how I respond, but I "
            f"{'may' if enabled else 'will not'} proactively bring up past sessions."
        )
        delta = _base_delta(reply, started_at=started_at)
        delta["procedural_profile"] = {"proactive_recall_enabled": enabled}
        delta["memory_control"] = {"pending_action": None}
        return delta

    if action_type == "save_preference":
        rule_text = str(action.get("rule_text", "")).strip()
        if not rule_text:
            delta = _base_delta(
                "I couldn't identify a clear preference to save.",
                started_at=started_at,
            )
            delta["memory_control"] = {"pending_action": None}
            return delta
        saved_rule = await save_preference_rule(
            store,
            owner_id=owner_id,
            rule_text=rule_text,
            evidence=state.get("message", ""),
        )
        delta = _base_delta(f"Saved: {saved_rule}", started_at=started_at)
        delta["memory_control"] = {"pending_action": None}
        return delta

    if action_type == "forget_by_index":
        kind = str(action.get("target_kind", "fact"))
        index = int(action.get("target_index", 0) or 0)
        if kind not in {"fact", "session", "rule"}:
            delta = _base_delta(
                "I can delete saved facts, session summaries, or style rules.",
                started_at=started_at,
            )
            delta["memory_control"] = {"pending_action": None}
            return delta
        target = await find_memory_target_by_index(
            store,
            owner_id=owner_id,
            kind=kind,  # type: ignore[arg-type]
            index_1based=index,
        )
        if target is None:
            delta = _base_delta(
                f"I couldn't find saved {kind} #{index}.",
                started_at=started_at,
            )
            delta["memory_control"] = {"pending_action": None}
            return delta
        delta = _base_delta(_pending_delete_reply(target), started_at=started_at)
        delta["memory_control"] = {
            "pending_action": {"type": "delete", "target": target}
        }
        return delta

    if action_type == "forget_by_query":
        query = str(action.get("query", "")).strip()
        targets = await find_memory_targets(store, owner_id=owner_id, query=query)
        if not targets:
            delta = _base_delta(
                "I couldn't find a saved memory matching that.",
                started_at=started_at,
            )
            delta["memory_control"] = {"pending_action": None}
            return delta
        if len(targets) > 1:
            delta = _base_delta(_multiple_matches_reply(targets), started_at=started_at)
            delta["memory_control"] = {"pending_action": None}
            return delta
        target = targets[0]
        delta = _base_delta(_pending_delete_reply(target), started_at=started_at)
        delta["memory_control"] = {
            "pending_action": {"type": "delete", "target": target}
        }
        return delta

    if action_type == "confirm_pending":
        pending = (state.get("memory_control", {}) or {}).get("pending_action") or {}
        target = pending.get("target")
        if not isinstance(target, dict):
            delta = _base_delta(
                "There isn't a pending memory change to confirm.",
                started_at=started_at,
            )
            delta["memory_control"] = {"pending_action": None}
            return delta
        deleted = await delete_memory_target(
            store,
            owner_id=owner_id,
            target=target,  # type: ignore[arg-type]
        )
        kind = target.get("kind", "memory")
        reply = (
            f"Deleted that saved {kind}."
            if deleted
            else "I couldn't delete that memory because it was already gone."
        )
        delta = _base_delta(reply, started_at=started_at)
        delta["memory_control"] = {"pending_action": None}
        return delta

    if action_type == "cancel_pending":
        delta = _base_delta(
            "Cancelled. I didn't change your memory.", started_at=started_at
        )
        delta["memory_control"] = {"pending_action": None}
        return delta

    delta = _base_delta(
        "I can show saved memory, turn proactive recall on or off, save a style "
        "preference, or help delete a specific saved memory.",
        started_at=started_at,
    )
    delta["memory_control"] = {"pending_action": None}
    return delta

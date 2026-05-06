"""Service logic for executing explicit memory-management actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.gates.memory_control.operations import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
    list_memory_for_owner,
    save_preference_rule,
    set_memory_recall,
)
from agent.memory.modes import MemoryMode
from agent.memory.procedural_profile import aget_procedural_profile
from agent.gates.memory_control.router import (
    CancelPendingAction,
    ConfirmPendingAction,
    ForgetByIndexAction,
    ForgetByQueryAction,
    ListAction,
    SavePreferenceAction,
    SetRecallAction,
    StatusAction,
    parse_memory_control_action,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id


@dataclass(frozen=True)
class MemoryControlServiceResult:
    """Result of executing one memory-management action."""

    response_text: str
    memory_control: dict[str, Any]
    procedural_profile: dict[str, Any] | None = None


def _empty_memory_reply() -> str:
    """Return a concise reply for an empty memory store.

    Returns:
        str: User-facing empty-memory reply.
    """

    return (
        "I don't have any saved facts, session summaries, or style rules for you "
        "right now."
    )


def _format_memory_overview(previews: dict[str, list[str]]) -> str:
    """Render memory previews into a short user-facing list.

    Args:
        previews (dict[str, list[str]]): Memory preview rows grouped by facts,
            sessions, and rules.

    Returns:
        str: User-facing memory overview.
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
        target (MemoryControlTarget): Memory target selected for deletion.

    Returns:
        str: User-facing deletion confirmation prompt.
    """

    return (
        f"I found this saved {target['kind']}:\n\n"
        f'"{target["preview"]}"\n\n'
        "Do you want me to delete it?"
    )


def _multiple_matches_reply(targets: list[MemoryControlTarget]) -> str:
    """Return disambiguation wording for multiple deletion matches.

    Args:
        targets (list[MemoryControlTarget]): Candidate memory targets matching
            the deletion query.

    Returns:
        str: User-facing disambiguation prompt.
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
        str: User-facing no-persistent-memory reply.
    """

    return (
        "You're in guest mode, so I don't have persistent memory to show or edit "
        "for this session."
    )


async def _owner_or_reply(state: AgentState) -> tuple[str | None, str | None]:
    """Resolve the memory owner or return a user-facing failure reply.

    Args:
        state (AgentState): Current graph state containing user/session identity.

    Returns:
        tuple[str | None, str | None]: ``(owner_id, None)`` on success,
            otherwise ``(None, reply)``.
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
    context: WorkflowContext,
) -> str:
    """Return memory status text for one owner.

    Args:
        owner_id (str): Owner whose memory status should be loaded.
        context (WorkflowContext): Workflow context carrying memory dependencies.

    Returns:
        str: User-facing memory status text.
    """

    store = context.memory_store
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


def _capability_reply() -> MemoryControlServiceResult:
    """Return the fallback reply when the action type is unknown or invalid."""

    return MemoryControlServiceResult(
        response_text=(
            "I can show saved memory, turn proactive recall on or off, save a style "
            "preference, or help delete a specific saved memory."
        ),
        memory_control={"pending_action": None},
    )


async def _handle_set_recall(
    *, action: SetRecallAction, store: Any, owner_id: str
) -> MemoryControlServiceResult:
    await set_memory_recall(store, owner_id=owner_id, enabled=action.enabled)
    state_text = "on" if action.enabled else "off"
    reply = (
        f"I turned proactive recall {state_text}. "
        "Style preferences can still shape how I respond, but I "
        f"{'may' if action.enabled else 'will not'} proactively bring up past sessions."
    )
    return MemoryControlServiceResult(
        response_text=reply,
        memory_control={"pending_action": None},
        procedural_profile={"proactive_recall_enabled": action.enabled},
    )


async def _handle_save_preference(
    *,
    action: SavePreferenceAction,
    store: Any,
    owner_id: str,
    state: AgentState,
) -> MemoryControlServiceResult:
    saved_rule = await save_preference_rule(
        store,
        owner_id=owner_id,
        rule_text=action.rule_text,
        evidence=state.get("message", ""),
    )
    return MemoryControlServiceResult(
        response_text=f"Saved: {saved_rule}",
        memory_control={"pending_action": None},
    )


async def _handle_forget_by_index(
    *, action: ForgetByIndexAction, store: Any, owner_id: str
) -> MemoryControlServiceResult:
    target = await find_memory_target_by_index(
        store,
        owner_id=owner_id,
        kind=action.target_kind,
        index_1based=action.target_index,
    )
    if target is None:
        return MemoryControlServiceResult(
            response_text=(
                f"I couldn't find saved {action.target_kind} #{action.target_index}."
            ),
            memory_control={"pending_action": None},
        )
    return MemoryControlServiceResult(
        response_text=_pending_delete_reply(target),
        memory_control={"pending_action": {"type": "delete", "target": target}},
    )


async def _handle_forget_by_query(
    *, action: ForgetByQueryAction, store: Any, owner_id: str
) -> MemoryControlServiceResult:
    targets = await find_memory_targets(store, owner_id=owner_id, query=action.query)
    if not targets:
        return MemoryControlServiceResult(
            response_text="I couldn't find a saved memory matching that.",
            memory_control={"pending_action": None},
        )
    if len(targets) > 1:
        return MemoryControlServiceResult(
            response_text=_multiple_matches_reply(targets),
            memory_control={"pending_action": None},
        )
    target = targets[0]
    return MemoryControlServiceResult(
        response_text=_pending_delete_reply(target),
        memory_control={"pending_action": {"type": "delete", "target": target}},
    )


async def _handle_confirm_pending(
    *, store: Any, owner_id: str, state: AgentState
) -> MemoryControlServiceResult:
    pending = (state.get("memory_control", {}) or {}).get("pending_action") or {}
    target = pending.get("target")
    if not isinstance(target, dict):
        return MemoryControlServiceResult(
            response_text="There isn't a pending memory change to confirm.",
            memory_control={"pending_action": None},
        )
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
    return MemoryControlServiceResult(
        response_text=reply,
        memory_control={"pending_action": None},
    )


async def execute_memory_control_action(
    state: AgentState,
    context: WorkflowContext,
) -> MemoryControlServiceResult:
    """Execute an explicit memory-management action.

    Args:
        state (AgentState): Current graph state with ``memory_control.action`` set
            by the gate.
        context (WorkflowContext): Workflow context carrying memory dependencies.

    Returns:
        MemoryControlServiceResult: User-facing reply plus memory-management state
            updates.
    """

    if context.memory_mode == MemoryMode.INCOGNITO:
        return MemoryControlServiceResult(
            response_text=_incognito_reply(),
            memory_control={"pending_action": None},
        )

    owner_id, failure_reply = await _owner_or_reply(state)
    if owner_id is None:
        return MemoryControlServiceResult(
            response_text=failure_reply or _empty_memory_reply(),
            memory_control={"pending_action": None},
        )

    raw_action = (state.get("memory_control", {}) or {}).get("action", {}) or {}
    action = parse_memory_control_action(raw_action)
    if action is None:
        return _capability_reply()

    store = context.memory_store

    match action:
        case ListAction():
            previews = await list_memory_for_owner(store, owner_id=owner_id)
            return MemoryControlServiceResult(
                response_text=_format_memory_overview(previews),
                memory_control={"pending_action": None},
            )
        case StatusAction():
            return MemoryControlServiceResult(
                response_text=await _handle_status(owner_id=owner_id, context=context),
                memory_control={"pending_action": None},
            )
        case SetRecallAction():
            return await _handle_set_recall(
                action=action, store=store, owner_id=owner_id
            )
        case SavePreferenceAction():
            return await _handle_save_preference(
                action=action, store=store, owner_id=owner_id, state=state
            )
        case ForgetByIndexAction():
            return await _handle_forget_by_index(
                action=action, store=store, owner_id=owner_id
            )
        case ForgetByQueryAction():
            return await _handle_forget_by_query(
                action=action, store=store, owner_id=owner_id
            )
        case ConfirmPendingAction():
            return await _handle_confirm_pending(
                store=store, owner_id=owner_id, state=state
            )
        case CancelPendingAction():
            return MemoryControlServiceResult(
                response_text="Cancelled. I didn't change your memory.",
                memory_control={"pending_action": None},
            )

    return _capability_reply()

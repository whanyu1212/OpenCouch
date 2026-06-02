"""Deletion and pending-deletion memory-control service helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, cast

from agent.memory.control.actions import ForgetByIndexAction, ForgetByQueryAction
from agent.memory.control.operations import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
)
from agent.memory.control.types import MemoryControlServiceResult


def _pending_delete_reply(target: MemoryControlTarget) -> str:
    """Return confirmation wording for a pending memory deletion."""

    return (
        f"I found this saved {target['kind']}:\n\n"
        f'"{target["preview"]}"\n\n'
        "Do you want me to delete it?"
    )


def _multiple_matches_reply(targets: list[MemoryControlTarget]) -> str:
    """Return disambiguation wording for multiple deletion matches."""

    lines = [
        "I found multiple saved memories that might match. Which one should I delete?"
    ]
    lines.extend(
        f"{index}. {target['kind']}: {target['preview']}"
        for index, target in enumerate(targets, start=1)
    )
    return "\n\n".join([lines[0], "\n".join(lines[1:])])


def _pending_delete_options(targets: list[MemoryControlTarget]) -> dict[str, Any]:
    """Return pending-state payload for a disambiguation list."""

    return {"type": "delete_options", "targets": targets}


def _pending_delete_options_from_action(
    pending_action: Mapping[str, Any] | None,
) -> list[MemoryControlTarget] | None:
    """Extract pending deletion candidates from serialized memory-control state."""

    if not pending_action or pending_action.get("type") != "delete_options":
        return None
    raw_targets = pending_action.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        return None
    targets: list[MemoryControlTarget] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            return None
        if raw_target.get("kind") not in {"fact", "session", "rule"}:
            return None
        if not isinstance(raw_target.get("key"), str):
            return None
        if not isinstance(raw_target.get("preview"), str):
            return None
        namespace = raw_target.get("namespace")
        if not isinstance(namespace, list) or not all(
            isinstance(part, str) for part in namespace
        ):
            return None
        rule_id = raw_target.get("rule_id")
        if rule_id is not None and not isinstance(rule_id, str):
            return None
        targets.append(
            {
                "kind": cast(Literal["fact", "session", "rule"], raw_target["kind"]),
                "namespace": namespace,
                "key": raw_target["key"],
                "rule_id": rule_id,
                "preview": raw_target["preview"],
            }
        )
    return targets


def _pending_delete_selection_index(query: str, option_count: int) -> int | None:
    """Return zero-based selected option index from a user disambiguation reply."""

    match = re.search(r"\b(\d+)\b", query)
    if match is None:
        return None
    index_1based = int(match.group(1))
    if index_1based < 1 or index_1based > option_count:
        return None
    return index_1based - 1


def _invalid_delete_selection_reply(targets: list[MemoryControlTarget]) -> str:
    """Return wording for an out-of-range deletion option reply."""

    return (
        "Please choose one of the listed memory options by number, or say cancel.\n\n"
        + _multiple_matches_reply(targets)
    )


def _is_session_wide_forget_query(query: str) -> bool:
    """Return whether a forget request should clear current-session candidates."""

    normalized = query.casefold()
    if any(
        marker in normalized
        for marker in (
            "this session",
            "current session",
            "today",
            "just said",
            "said earlier",
            "everything",
            "all of this",
            "this conversation",
            "our conversation",
        )
    ):
        return True
    return any(
        marker in normalized
        for marker in (
            "don't save",
            "do not save",
            "dont save",
            "don't remember",
            "do not remember",
            "dont remember",
            "forget this",
            "forget that",
        )
    )


async def handle_forget_by_index(
    *,
    action: ForgetByIndexAction,
    store: Any,
    owner_id: str,
) -> MemoryControlServiceResult:
    """Prepare deletion of a visible saved memory selected by index."""

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


async def handle_forget_by_query(
    *,
    action: ForgetByQueryAction,
    store: Any,
    owner_id: str,
    pending_action: Mapping[str, Any] | None,
) -> MemoryControlServiceResult:
    """Prepare deletion of a saved memory selected by query or option number."""

    clear_session_buffer = _is_session_wide_forget_query(action.query)
    pending_options = _pending_delete_options_from_action(pending_action)
    if pending_options is not None:
        selection_index = _pending_delete_selection_index(
            action.query,
            option_count=len(pending_options),
        )
        if selection_index is None:
            return MemoryControlServiceResult(
                response_text=_invalid_delete_selection_reply(pending_options),
                memory_control={
                    "pending_action": _pending_delete_options(pending_options)
                },
                clear_session_buffer=clear_session_buffer,
            )
        target = pending_options[selection_index]
        return MemoryControlServiceResult(
            response_text=_pending_delete_reply(target),
            memory_control={"pending_action": {"type": "delete", "target": target}},
            clear_session_buffer=clear_session_buffer,
        )

    targets = await find_memory_targets(store, owner_id=owner_id, query=action.query)
    if not targets:
        return MemoryControlServiceResult(
            response_text="I couldn't find a saved memory matching that.",
            memory_control={"pending_action": None},
            clear_session_buffer=clear_session_buffer,
        )
    if len(targets) > 1:
        return MemoryControlServiceResult(
            response_text=_multiple_matches_reply(targets),
            memory_control={"pending_action": _pending_delete_options(targets)},
            clear_session_buffer=clear_session_buffer,
        )
    target = targets[0]
    return MemoryControlServiceResult(
        response_text=_pending_delete_reply(target),
        memory_control={"pending_action": {"type": "delete", "target": target}},
        clear_session_buffer=clear_session_buffer,
    )


async def handle_confirm_pending(
    *,
    store: Any,
    owner_id: str,
    pending_action: Mapping[str, Any] | None,
) -> MemoryControlServiceResult:
    """Confirm and execute a pending memory deletion."""

    pending = pending_action or {}
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


def cancel_pending_result() -> MemoryControlServiceResult:
    """Return the result for canceling a pending memory-control action."""

    return MemoryControlServiceResult(
        response_text="Cancelled. I didn't change your memory.",
        memory_control={"pending_action": None},
    )

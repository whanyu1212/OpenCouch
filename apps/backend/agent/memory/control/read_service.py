"""Read-only memory-control service helpers."""

from __future__ import annotations

from typing import Any

from agent.memory.control.operations import list_memory_for_owner
from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.control.types import MemoryControlServiceResult
from agent.runtime_context import WorkflowContext


def empty_memory_reply() -> str:
    """Return a concise reply for an empty memory store."""

    return (
        "I don't have any saved facts, session summaries, or style rules for you "
        "right now."
    )


def format_memory_overview(previews: dict[str, list[str]]) -> str:
    """Render memory previews into a short user-facing list."""

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
        return empty_memory_reply()
    return "Here's what I currently have saved:\n\n" + "\n".join(lines)


def incognito_memory_control_result() -> MemoryControlServiceResult:
    """Return the no-op result for incognito memory mode."""

    return MemoryControlServiceResult(
        response_text=(
            "You're in guest mode, so I don't have persistent memory to show or edit "
            "for this session."
        ),
        memory_control={"pending_action": None},
    )


def owner_or_failure_result(
    owner_id: str | None,
) -> tuple[str | None, MemoryControlServiceResult | None]:
    """Resolve the memory owner or return a user-facing failure result."""

    if owner_id is None:
        return (
            None,
            MemoryControlServiceResult(
                response_text=(
                    "I don't have a stable memory owner for this conversation, so "
                    "I can't show or edit saved memory here."
                ),
                memory_control={"pending_action": None},
            ),
        )
    return owner_id, None


async def handle_memory_list(
    *,
    store: Any,
    owner_id: str,
) -> MemoryControlServiceResult:
    """Return the saved-memory overview for one owner."""

    previews = await list_memory_for_owner(store, owner_id=owner_id)
    return MemoryControlServiceResult(
        response_text=format_memory_overview(previews),
        memory_control={"pending_action": None},
    )


async def handle_memory_status(
    *,
    owner_id: str,
    context: WorkflowContext,
) -> MemoryControlServiceResult:
    """Return memory status text for one owner."""

    store = context.memory_store
    profile = await aget_procedural_profile(store, user_id=owner_id)
    fact_count = await store.arecord_count((owner_id, "semantic"))
    session_count = await store.arecord_count((owner_id, "episodic"))
    return MemoryControlServiceResult(
        response_text=(
            "Memory status:\n\n"
            f"Saved facts: {fact_count}\n"
            f"Session summaries: {session_count}\n"
            f"Style preferences: {len(profile.rules)}\n"
            f"Proactive recall: {'on' if profile.proactive_recall_enabled else 'off'}"
        ),
        memory_control={"pending_action": None},
    )

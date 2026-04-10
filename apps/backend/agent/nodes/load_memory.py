"""Load-memory node for the OpenCouch agent graph."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.models import MessageRole, ModeType, ResponseKind
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


def memory_bootstrap_reply(is_guest_mode: bool) -> str:
    """Return a deterministic scaffold reply for the bootstrap graph."""

    if is_guest_mode:
        return (
            "Guest mode is active. I loaded no long-term memory, and this session will "
            "not be remembered after you exit."
        )
    return (
        "Persistent mode is active. I loaded your local memory context and will be able "
        "to carry context across future sessions."
    )


async def _retrieve_semantic_working_memory(
    store: OpenCouchMemoryStore,
    *,
    owner_id: str,
    query: str,
) -> list[str]:
    """Fetch the top semantic facts for this user and format them as strings.

    v0.1 implementation: substring-match the current message against the
    semantic namespace and return the formatted evidence quotes. This is
    deliberately minimal — the richer hybrid retrieval (vector search +
    graph expansion) lands in v0.3 when real semantic extraction starts
    producing content worth searching.
    """

    namespace = (owner_id, "semantic")
    records = await store.asearch(namespace, query=query, limit=5)
    formatted: list[str] = []
    for record in records:
        quote = record.value.get("evidence_quote")
        if quote:
            formatted.append(f"Previously noted: {quote}")
    return formatted


async def run_load_memory_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Load memory snippets and return only the keys this node updated."""

    memory_store = runtime.context["memory_store"]
    memory_mode = runtime.context.get("memory_mode", MemoryMode.INCOGNITO)
    is_guest_mode = memory_mode == MemoryMode.INCOGNITO

    owner_id = state.get("user_id") or state.get("session_id") or "local-default"

    # ── Step 1: Resolve working memory (skip retrieval entirely in guest mode) ─
    if is_guest_mode:
        working_memory: list[str] = []
    else:
        working_memory = await _retrieve_semantic_working_memory(
            memory_store,
            owner_id=owner_id,
            query=state["message"],
        )

    # ── Step 2: Append the inbound user turn + the deterministic bootstrap reply ─
    response_text = memory_bootstrap_reply(is_guest_mode)
    new_transcript = [
        *state.get("transcript", []),
        {"role": MessageRole.USER.value, "content": state["message"]},
        {"role": MessageRole.ASSISTANT.value, "content": response_text},
    ]

    summary = (
        "Guest session without long-term memory."
        if is_guest_mode
        else f"Loaded {len(working_memory)} memory snippets from the unified store."
    )

    # ── Step 3: Return only the keys this node updated ────────────────────
    progress = state.get("progress", {})
    return {
        "history": list(new_transcript),
        "transcript": new_transcript,
        "working_memory": list(working_memory),
        "memory": {
            "summary": summary,
            "active_concerns": [],
            "open_loops": [],
            "current_goal": None,
        },
        "progress": {
            **progress,
            "stage": "opening",
            "stage_source": "deterministic",
            "stage_reason": "start_load_memory_end",
            "is_guest": is_guest_mode,
        },
        "routing": {
            "route": "load_memory_only",
            "mode": "memory_bootstrap",
            "mode_source": "graph_bootstrap",
            "mode_type": ModeType.OPERATIONAL,
            "active_modalities": [],
            "semantic_signals": {},
        },
        "response": {
            "guidance": "startup_memory_bootstrap",
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
            "should_persist_memory": bool(working_memory) and not is_guest_mode,
        },
    }

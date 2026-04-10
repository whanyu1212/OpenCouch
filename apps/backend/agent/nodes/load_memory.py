"""Load-memory node for the OpenCouch agent graph.

This node runs on every turn as the spine's first step, immediately after
``START``. Its only job is to retrieve relevant long-term memory for the
**current user message** and publish it into ``working_memory`` so the
downstream crisis gate and therapeutic nodes can read it.

History / migration note (v0.3.1 → post-v0.3.1 cleanup):

    Prior to the dogfood pass that caught the "Loaded 0 memory snippets"
    bug, this node also wrote a deterministic bootstrap reply into the
    transcript, clobbered the ``response`` slot, and overwrote
    ``routing`` with a ``memory_bootstrap`` placeholder. All of that
    behavior assumed the node ran once per session, which is wrong —
    LangGraph runs it on every invocation because it lives on the
    ``START → load_memory_node → crisis_gate_node`` spine. The result
    was phantom "Persistent mode is active" assistant turns polluting
    the transcript on every turn, and the dispatcher seeing stale
    context.

    The fix: strip this node down to pure retrieval. Transcript growth
    happens elsewhere (the response nodes own their own appends); the
    initial ``response`` / ``routing`` scaffolds live in
    ``agent.graph.build_initial_state`` where they belong as one-time
    defaults, not per-turn overwrites. The ``memory_bootstrap_reply``
    helper was deleted with this refactor — nothing should ever show
    that string to a user.

Scope today:
- Only the semantic namespace is queried. Episodic retrieval lands
  with v0.4's session summarizer; procedural with v0.7.
- Retrieval scoring is token-recall via ``store.asearch`` — see
  ``agent/memory/store.py`` SEARCH_MATCH_THRESHOLD and the v0.3.1
  status log entry in ROADMAP.md for the algorithm.
- Incognito / guest mode skips retrieval entirely and returns an
  empty working memory, matching the "we don't remember you" contract.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.text_tokens import tokenize_meaningful
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def _retrieve_semantic_working_memory(
    store: OpenCouchMemoryStore,
    *,
    owner_id: str,
    query: str,
) -> list[str]:
    """Fetch the top semantic facts for this user and format them as strings.

    Uses the v0.3.1 token-recall scorer in ``store.asearch``: the store
    scores each record by the fraction of the query's meaningful tokens
    (after stopword filtering) that appear in the record's haystack, and
    returns results in score-descending order. The caller here just asks
    for ``limit=5`` — any scoring changes happen at the store layer.

    v0.4 will replace this with a hybrid retrieval that also pulls
    episodic summaries; today it's semantic-only because episodic and
    procedural namespaces are empty.
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
    """Retrieve relevant long-term memory for the current user message.

    Returns a delta containing only ``working_memory`` and a new
    ``memory.summary`` that reflects how many snippets were retrieved.
    Does NOT touch ``transcript``, ``history``, ``response``, or
    ``routing`` — those are owned by other parts of the graph and
    writing them here causes the phantom-turn bug this refactor fixed.

    Guest mode (``MemoryMode.INCOGNITO``) skips retrieval and returns
    an empty working memory with a matching summary. This matches the
    incognito contract: no reads from persistent storage, no trace of
    prior sessions.

    Observability: the summary string is structured to show all three
    signals a dogfood operator needs to understand retrieval behavior:
    (1) how many snippets hit for this query, (2) how many records
    exist in the user's namespace at all, and (3) how many meaningful
    tokens the query contributed after stopword filtering. This lets
    you distinguish "empty store" from "below-threshold query" from
    "no meaningful query" at a glance in the CLI Session Context panel.
    An INFO log line captures the same signals for grep-based analysis.
    """

    memory_store = runtime.context["memory_store"]
    memory_mode = runtime.context.get("memory_mode", MemoryMode.INCOGNITO)
    is_guest_mode = memory_mode == MemoryMode.INCOGNITO

    if is_guest_mode:
        return {
            "working_memory": [],
            "memory": {
                **state.get("memory", {}),
                "summary": "Guest session without long-term memory.",
            },
        }

    owner_id = state.get("user_id") or state.get("session_id") or "local-default"
    namespace = (owner_id, "semantic")
    store_size = memory_store.record_count(namespace)
    query = state["message"]
    meaningful_query_tokens = tokenize_meaningful(query)

    working_memory = await _retrieve_semantic_working_memory(
        memory_store,
        owner_id=owner_id,
        query=query,
    )

    summary = (
        f"Retrieved {len(working_memory)} of {store_size} semantic record(s) "
        f"(query had {len(meaningful_query_tokens)} meaningful token(s))."
    )

    logger.info(
        "load_memory_node: hits=%d store_size=%d query_tokens=%d owner=%r",
        len(working_memory),
        store_size,
        len(meaningful_query_tokens),
        owner_id,
    )

    return {
        "working_memory": list(working_memory),
        "memory": {
            **state.get("memory", {}),
            "summary": summary,
        },
    }

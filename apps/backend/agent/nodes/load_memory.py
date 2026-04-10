"""Load-memory node for the OpenCouch agent graph.

This node runs on every turn as the spine's first step, immediately after
``START``. Its only job is to retrieve relevant long-term memory for the
**current user message** and publish it into ``working_memory`` so the
downstream crisis gate and therapeutic nodes can read it.

History / migration note (v0.3.1 → v0.4):

    Prior to the v0.3.1 dogfood pass that caught the "Loaded 0 memory
    snippets" bug, this node also wrote a deterministic bootstrap reply
    into the transcript, clobbered the ``response`` slot, and overwrote
    ``routing`` with a ``memory_bootstrap`` placeholder. All of that
    behavior assumed the node ran once per session, which is wrong —
    LangGraph runs it on every invocation because it lives on the
    ``START → load_memory_node → crisis_gate_node`` spine. v0.3.1
    stripped the node to pure retrieval; see the historical comment
    below for the full fix.

v0.4 added episodic retrieval alongside semantic retrieval. The node
now queries two namespaces — ``(owner, "semantic")`` and
``(owner, "episodic")`` — and merges the results into a single
``working_memory`` list with distinct formatting:

- ``"Previously noted: {quote}"`` for semantic facts
- ``"Last session ({themes}): {summary}"`` for episodic arcs

The two paths share the same token-recall scorer in ``store.asearch``,
so retrieval calibration stays consistent across record types. The
episodic path has one additional rule: on the **first turn of a new
session** (the transcript contains only the current user message), the
most recent episodic summary is pre-pended to ``working_memory`` as a
catch-up entry regardless of query match. This gives the user the
"last time we talked…" feel on session start without bloating every
turn's prompt with catch-up text.

Scope today:
- Semantic namespace (v0.3): real extraction with hot-path dedup
- Episodic namespace (v0.4): single session arc per completed session,
  written by the summarizer function at session end
- Procedural namespace (v0.7): not yet wired; reads return empty
- Incognito / guest mode skips both retrieval paths and returns an
  empty working memory, matching the "we don't remember you" contract.

Retrieval scoring is token-recall via ``store.asearch`` — see
``agent/memory/store.py`` SEARCH_MATCH_THRESHOLD and the v0.3.1 status
log entry in ROADMAP.md for the algorithm details.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from agent.memory.text_tokens import tokenize_meaningful
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def _retrieve_semantic_working_memory(
    store: MemoryStore,
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
    """

    namespace = (owner_id, "semantic")
    records = await store.asearch(namespace, query=query, limit=5)
    formatted: list[str] = []
    for record in records:
        quote = record.value.get("evidence_quote")
        if quote:
            formatted.append(f"Previously noted: {quote}")
    return formatted


def _format_episodic_entry(record_value: dict[str, Any]) -> str | None:
    """Render a stored session arc as a single working_memory line.

    The format is ``"Last session (<themes>): <summary>"`` where themes
    is a comma-joined list (or ``untagged`` when empty). This format
    mirrors ``"Previously noted: <quote>"`` for semantic entries, so
    the downstream response prompts can recognize both kinds by their
    prefix without needing a structured working_memory list.

    Returns None when the record is missing a summary (which shouldn't
    happen for well-formed records, but the guard keeps the function
    robust against schema drift).
    """

    summary = record_value.get("summary")
    if not summary:
        return None
    themes_list = record_value.get("primary_themes") or []
    themes_str = ", ".join(themes_list) if themes_list else "untagged"
    return f"Last session ({themes_str}): {summary}"


async def _retrieve_episodic_working_memory(
    store: MemoryStore,
    *,
    owner_id: str,
    query: str,
    is_first_turn: bool,
) -> list[str]:
    """Fetch relevant episodic session arcs and format them as strings.

    Two branches:

    1. **Catch-up** (``is_first_turn=True``): the caller has just
       started a new session (the transcript contains only the current
       user turn). We pre-pend the single most recent episodic summary
       for this user regardless of query match, so the session opens
       with a "last time we talked…" context entry.

    2. **Query-based** (``is_first_turn=False``): on later turns, we
       go through the same token-recall scorer as semantic retrieval,
       limited to 2 entries so the prompt doesn't bloat. An off-topic
       query still produces zero hits.

    The caller combines the returned list with semantic results. v0.4
    initial ship uses ``limit=2`` for the query-based branch; this
    number is tunable and will likely grow once dogfood data shows
    what a reasonable episodic context size looks like.

    When ``is_first_turn`` is True AND a catch-up entry is returned,
    that entry is ALSO checked against the query-based path and
    deduped — we don't want to show the same summary twice if the
    user's first message happens to overlap with the most recent arc.
    """

    namespace = (owner_id, "episodic")
    formatted: list[str] = []

    if is_first_turn:
        # Catch-up: fetch the single most recent arc regardless of query.
        # `query=None` returns records in insertion order; since episodic
        # records are written in chronological order (one per session),
        # the last entry in the list is the most recent. We fetch up to
        # a safe upper bound and then take the last one.
        recent_records = await store.asearch(namespace, query=None, limit=50)
        if recent_records:
            catch_up = _format_episodic_entry(recent_records[-1].value)
            if catch_up is not None:
                formatted.append(catch_up)

    # Query-based retrieval — always runs, but on the first turn the
    # catch-up entry is already in `formatted` so we dedupe by comparing
    # the rendered string.
    query_records = await store.asearch(namespace, query=query, limit=2)
    for record in query_records:
        entry = _format_episodic_entry(record.value)
        if entry is not None and entry not in formatted:
            formatted.append(entry)

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

    v0.4 added episodic retrieval. The working_memory list is now a
    merged result: episodic entries first (catch-up and/or query-
    matched summaries), then semantic entries. Downstream response
    nodes can read the whole list without caring about the split;
    the two string prefixes (``"Last session"`` and ``"Previously
    noted"``) provide the visible distinction.

    Observability: the summary string reports counts for each layer
    separately plus the meaningful query token count. This lets a
    dogfood operator distinguish "nothing stored yet" from "stored
    but below threshold" across both namespaces at a glance.
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
    semantic_ns = (owner_id, "semantic")
    episodic_ns = (owner_id, "episodic")
    query = state["message"]
    meaningful_query_tokens = tokenize_meaningful(query)

    # Determine whether this is the first turn of the current session.
    # ``build_initial_state`` appends the current user message to the
    # transcript, so a single-entry transcript means "no prior turns
    # have run in this session yet" — the trigger for catch-up injection.
    transcript = state.get("transcript", [])
    is_first_turn = len(transcript) == 1

    # v0.8.1 observability: time the retrieval work so we have a
    # per-turn latency signal to answer "is retrieval expensive
    # enough to gate yet?" without committing to any gating
    # strategy. The timer covers the store interactions
    # (arecord_count + asearch on both namespaces) — i.e., the work
    # a gating decision would actually avoid. Python-side token
    # splitting happens before the timer starts and is microseconds
    # anyway, so including it would just add noise. The current
    # guidance (see the retrieval-gating discussion in the v0.8
    # follow-up notes): revisit gating when dogfood p95 exceeds
    # ~20ms per turn AND store_size consistently exceeds a few
    # hundred records. Until then, "always on" is the right call.
    retrieval_start = time.monotonic()

    semantic_store_size = await memory_store.arecord_count(semantic_ns)
    episodic_store_size = await memory_store.arecord_count(episodic_ns)

    # Episodic retrieval: catch-up on first turn + query-based matches.
    episodic_entries = await _retrieve_episodic_working_memory(
        memory_store,
        owner_id=owner_id,
        query=query,
        is_first_turn=is_first_turn,
    )

    # Semantic retrieval: v0.3.1 token-recall path.
    semantic_entries = await _retrieve_semantic_working_memory(
        memory_store,
        owner_id=owner_id,
        query=query,
    )

    retrieval_duration_ms = (time.monotonic() - retrieval_start) * 1000

    # Merge: episodic entries first (they're the "context prefix" that
    # frames the session), then semantic entries. Downstream response
    # nodes see the whole list and can reference either kind.
    working_memory = [*episodic_entries, *semantic_entries]

    summary = (
        f"Retrieved {len(semantic_entries)} of {semantic_store_size} semantic + "
        f"{len(episodic_entries)} of {episodic_store_size} episodic record(s) "
        f"(query had {len(meaningful_query_tokens)} meaningful token(s))."
    )

    logger.info(
        "load_memory_node: semantic=%d/%d episodic=%d/%d first_turn=%s "
        "query_tokens=%d duration_ms=%.2f owner=%r",
        len(semantic_entries),
        semantic_store_size,
        len(episodic_entries),
        episodic_store_size,
        is_first_turn,
        len(meaningful_query_tokens),
        retrieval_duration_ms,
        owner_id,
    )

    return {
        "working_memory": list(working_memory),
        "memory": {
            **state.get("memory", {}),
            "summary": summary,
        },
    }

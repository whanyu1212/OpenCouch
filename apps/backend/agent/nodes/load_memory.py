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
``working_memory`` list of structured dicts. Prompt builders and CLI
surfaces format those dicts on demand.

The two paths share the same token-recall scorer in ``store.asearch``,
so retrieval calibration stays consistent across record types. The
episodic path has one additional rule: on the **first turn of a new
session** (the transcript contains only the current user message), the
most recent episodic summary is pre-pended to ``working_memory`` as a
catch-up entry regardless of query match. This gives the user the
"last time we talked…" feel on session start without bloating every
turn's prompt with catch-up text.

Scope today:
- Semantic namespace (v0.3): real extraction with hot-path dedup.
  Loaded via token-recall scoring into ``working_memory`` as raw
  semantic-entry dicts.
- Episodic namespace (v0.4): single session arc per completed session,
  written by the summarizer function at session end. Loaded via
  token-recall scoring (with first-turn catch-up) into
  ``working_memory`` as raw episodic-entry dicts.
- Procedural namespace (v0.7): style rules + recall toggle loaded
  from the user's :class:`ProceduralProfile`. Attached to
  ``state["memory"]["procedural_rules"]`` and
  ``state["memory"]["proactive_recall_enabled"]`` — NOT mixed into
  ``working_memory``. Rules are directives that shape response
  style, not content to be referenced, so they go into a different
  state field with a different prompt treatment in Stage D.
- Incognito / guest mode skips all three retrieval paths and returns
  empty results across the board, matching the "we don't remember
  you" contract.

Retrieval scoring is token-recall via ``store.asearch`` — see
``agent/memory/store.py`` SEARCH_MATCH_THRESHOLD and the v0.3.1 status
log entry in ROADMAP.md for the algorithm details.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from langgraph.runtime import Runtime

from agent.memory.modes import MemoryMode
from agent.memory.procedural import aget_procedural_profile
from agent.memory.reconciliation import is_active_semantic_record_value
from agent.memory.store import MemoryStore
from agent.memory.text_tokens import tokenize_meaningful
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id
from agent.working_memory import (
    WorkingMemoryEntry,
    make_episodic_working_memory_entry,
    make_semantic_working_memory_entry,
)

if TYPE_CHECKING:
    from agent.memory.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


async def _retrieve_semantic_working_memory(
    store: MemoryStore,
    *,
    owner_id: str,
    query: str,
    query_embedding: list[float] | None,
    embedding_model: str | None,
) -> list[WorkingMemoryEntry]:
    """Fetch the top semantic facts for this user as structured entries.

    v0.8.1: routes through the store's hybrid retrieval path
    (:meth:`MemoryStore.asearch_similar`). When ``query_embedding`` is
    provided, the store runs both the v0.3.1 token-recall scan AND a
    cosine-similarity scan on stored embeddings, then combines the
    two ranked lists via Reciprocal Rank Fusion. When
    ``query_embedding`` is ``None`` (no embedding provider, embedding
    computation failed, guest mode short-circuited above), the
    hybrid path degenerates cleanly to token-recall-only and the
    behavior matches the v0.3.1/v0.4/v0.8 retrieval contract
    exactly — no conditional code path is needed here.

    The caller is responsible for pre-computing the query embedding
    with ``task_type="RETRIEVAL_QUERY"`` so asymmetric retrieval
    models (like Gemini's text-embedding-004) produce query-tuned
    vectors rather than document-tuned ones.
    """

    namespace = (owner_id, "semantic")
    records = await store.asearch_similar(
        namespace,
        query_text=query,
        query_embedding=query_embedding,
        embedding_model=embedding_model,
        limit=20,
    )
    entries: list[WorkingMemoryEntry] = []
    for record in records:
        if not is_active_semantic_record_value(record.value):
            continue
        quote = record.value.get("evidence_quote")
        if quote:
            subject_ref = record.value.get("subject") or {}
            object_ref = record.value.get("object") or {}
            entries.append(
                make_semantic_working_memory_entry(
                    evidence_quote=quote,
                    category=record.value.get("category", ""),
                    subject=subject_ref.get("identifier", "")
                    if isinstance(subject_ref, dict)
                    else str(subject_ref),
                    predicate=record.value.get("predicate", ""),
                    object=object_ref.get("identifier", "")
                    if isinstance(object_ref, dict)
                    else str(object_ref),
                )
            )
        if len(entries) >= 5:
            break
    return entries


async def _active_semantic_record_count(
    store: MemoryStore,
    *,
    owner_id: str,
) -> int:
    """Return how many active semantic facts this user currently has."""

    records = await store.asearch((owner_id, "semantic"), query=None, limit=1000)
    return sum(1 for record in records if is_active_semantic_record_value(record.value))


def _episodic_entry_from_record(
    record_value: dict[str, Any],
    *,
    is_catch_up: bool,
) -> WorkingMemoryEntry | None:
    """Convert a stored session arc into a structured working-memory entry."""

    summary = record_value.get("summary")
    if not summary:
        return None
    return make_episodic_working_memory_entry(
        summary=summary,
        primary_themes=record_value.get("primary_themes") or [],
        is_catch_up=is_catch_up,
    )


def _episodic_entry_identity(entry: WorkingMemoryEntry) -> tuple[str, tuple[str, ...]]:
    """Return the dedup identity for an episodic working-memory entry.

    ``is_catch_up`` is deliberately excluded. The same stored arc can be
    surfaced by both the catch-up path and the query-match path on turn 1,
    and we want that to collapse to a single entry.
    """

    if entry.get("type") != "episodic":
        return "", ()
    return (
        entry.get("summary", ""),
        tuple(entry.get("primary_themes") or []),
    )


async def _retrieve_episodic_working_memory(
    store: MemoryStore,
    *,
    owner_id: str,
    query: str,
    query_embedding: list[float] | None,
    embedding_model: str | None,
    is_first_turn: bool,
) -> list[WorkingMemoryEntry]:
    """Fetch relevant episodic session arcs as structured entries.

    Two branches:

    1. **Catch-up** (``is_first_turn=True``): the caller has just
       started a new session (the transcript contains only the current
       user turn). We pre-pend the single most recent episodic summary
       for this user regardless of query match, so the session opens
       with a "last time we talked…" context entry.

       v0.9 fix: uses ``store.alatest()`` which does a single
       ``ORDER BY insertion_order DESC LIMIT 1`` fetch. The previous
       approach (``asearch(query=None, limit=50)`` + ``[-1]``) silently
       returned the 50th-oldest record once a user exceeded 50 sessions.

    2. **Query-based** (``is_first_turn=False`` or additional matches
       on first turn): goes through the v0.8.1 hybrid retrieval path
       (:meth:`MemoryStore.asearch_similar`). When ``query_embedding``
       is provided, RRF-fuses token-recall and cosine similarity on
       the stored arc embeddings. When ``query_embedding`` is None,
       degenerates to token-recall only — same as the v0.4/v0.8
       contract. Limited to 2 entries so the prompt doesn't bloat.

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
    entries: list[WorkingMemoryEntry] = []

    if is_first_turn:
        # Catch-up: fetch the single most recent arc via alatest().
        # v0.9 fix: the previous approach (asearch(query=None, limit=50)
        # + [-1]) silently returned the 50th-oldest record once a user
        # exceeded 50 sessions. alatest() uses DESC LIMIT 1 and is
        # correct at any scale. It's a required protocol method — custom
        # MemoryStore implementations must provide it.
        latest = await store.alatest(namespace)
        if latest is not None:
            catch_up = _episodic_entry_from_record(
                latest.value,
                is_catch_up=True,
            )
            if catch_up is not None:
                entries.append(catch_up)

    # Query-based retrieval — always runs, but on the first turn the
    # catch-up entry is already in `formatted` so we dedupe by comparing
    # the rendered string.
    query_records = await store.asearch_similar(
        namespace,
        query_text=query,
        query_embedding=query_embedding,
        embedding_model=embedding_model,
        limit=2,
        max_age_days=30,
    )
    for record in query_records:
        entry = _episodic_entry_from_record(
            record.value,
            is_catch_up=False,
        )
        if entry is not None and _episodic_entry_identity(entry) not in {
            _episodic_entry_identity(existing) for existing in entries
        }:
            entries.append(entry)

    return entries


async def _retrieve_procedural_state(
    store: MemoryStore,
    *,
    owner_id: str,
) -> tuple[list[str], bool]:
    """Load the user's procedural profile and return its surface fields.

    Returns ``(rules, proactive_recall_enabled)``.

    - ``rules`` is a list of raw rule texts, in the order they were
      written. Rules are returned verbatim (second-person,
      evidence-grounded) — no reformatting, no prefix. The Stage D
      prompt builders apply the rendering ("You have the following
      style rules from past conversations…") when they inject the
      list into the system prompt.
    - ``proactive_recall_enabled`` is the user's ``/memory recall``
      toggle. Defaults to False for users with no profile yet,
      matching the schema default.

    Unlike semantic and episodic retrieval, procedural retrieval is
    NOT query-based. The full rule set is always loaded — rules are
    directives, and the agent needs to see all of them on every turn
    to apply them consistently. See schema.yaml §6 retrieval for the
    rationale (``procedural.enabled: true`` unconditionally).

    The empty-default-on-miss behavior comes from ``aget_procedural_profile``
    in ``agent/memory/procedural.py``: a user with no record yet gets
    a fresh empty profile without a store write. The caller here
    doesn't need to handle the ``None`` case.
    """

    profile = await aget_procedural_profile(store, user_id=owner_id)
    rule_texts = [rule.rule for rule in profile.rules]
    return rule_texts, profile.proactive_recall_enabled


async def _compute_query_embedding(
    embedding_provider: "EmbeddingProvider | None",
    query: str,
) -> tuple[list[float] | None, str | None, str]:
    """Compute the query embedding, returning (embedding, model, path).

    Extracted from ``run_load_memory_node`` so the embedding API call
    (50-200ms network I/O) can run concurrently with store calls via
    ``asyncio.gather``.

    Returns:
        A 3-tuple of ``(query_embedding, embedding_model, retrieval_path)``
        where ``retrieval_path`` is one of ``"hybrid_rrf"``,
        ``"token_recall"``, or ``"token_recall_after_embed_error"``.
    """

    if embedding_provider is None:
        return None, None, "token_recall"
    try:
        embeddings_out = await embedding_provider.aembed(
            [query],
            task_type="RETRIEVAL_QUERY",
        )
        if embeddings_out and embeddings_out[0] is not None:
            return embeddings_out[0], embedding_provider.model_name, "hybrid_rrf"
        return None, None, "token_recall"
    except Exception:
        logger.warning(
            "load_memory_node: embedding call failed; falling back to "
            "token-recall only for this turn.",
            exc_info=True,
        )
        return None, None, "token_recall_after_embed_error"


async def run_load_memory_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Retrieve relevant long-term memory for the current user message.

    Returns a delta containing:
    - ``working_memory`` — the merged semantic + episodic content list
    - ``memory.summary`` — a human-readable retrieval summary
    - ``memory.procedural_rules`` (v0.7) — raw procedural rule texts
    - ``memory.proactive_recall_enabled`` (v0.7) — the recall toggle

    Does NOT touch ``transcript``, ``history``, ``response``, or
    ``routing`` — those are owned by other parts of the graph and
    writing them here causes the phantom-turn bug this refactor fixed.

    Guest mode (``MemoryMode.INCOGNITO``) skips retrieval and returns
    empty values across all layers with a matching summary. This
    matches the incognito contract: no reads from persistent storage,
    no trace of prior sessions.

    v0.4 added episodic retrieval. The ``working_memory`` list is a
    merged result: episodic entries first (catch-up and/or query-
    matched summaries), then semantic entries. The state keeps these
    entries RAW; prompt/CLI surfaces format them on demand.

    v0.7 Stage C added procedural retrieval as a SEPARATE state field.
    Rules are directives (silent style shaping) rather than content to
    reference, so they live on ``memory.procedural_rules`` and get
    injected into the system prompt suffix by Stage D prompt builders.
    The recall toggle lives alongside them on
    ``memory.proactive_recall_enabled`` and governs whether the Stage D
    prompt builders emit the "do not proactively reference past
    sessions" constraint for semantic/episodic content.

    Observability: the summary string reports counts for each layer
    separately plus the meaningful query token count and the recall
    toggle state. This lets a dogfood operator distinguish "nothing
    stored yet" from "stored but below threshold" across all layers
    at a glance.
    """

    memory_store = runtime.context.memory_store
    memory_mode = runtime.context.memory_mode
    embedding_provider: EmbeddingProvider | None = runtime.context.embedding_provider
    is_guest_mode = memory_mode == MemoryMode.INCOGNITO

    if is_guest_mode:
        # Incognito: skip all retrieval paths including procedural.
        # Rules and the recall toggle are both empty — matches the
        # "we don't remember you" contract.
        return {
            "working_memory": [],
            "memory": {
                **state.get("memory", {}),
                "summary": "Guest session without long-term memory.",
                "procedural_rules": [],
                "proactive_recall_enabled": False,
            },
        }

    owner_id = resolve_owner_id(state)
    episodic_ns = (owner_id, "episodic")
    query = state["message"]
    meaningful_query_tokens = tokenize_meaningful(query)

    # Determine whether this is the first turn of the current session.
    # ``build_initial_state`` appends the current user message to the
    # transcript, so a single-entry transcript means "no prior turns
    # have run in this session yet" — the trigger for catch-up injection.
    transcript = state.get("transcript", [])
    is_first_turn = len(transcript) == 1

    retrieval_start = time.monotonic()

    # ── Phase 1: all independent work runs concurrently ──────────────
    # The embedding API call (50-200ms network I/O to Gemini) overlaps
    # with the store calls (local SQLite, <5ms each). aiosqlite
    # serializes store calls via a single worker thread, but the real
    # win is the embedding/store overlap.
    (
        (query_embedding, query_embedding_model, retrieval_path),
        semantic_store_size,
        episodic_store_size,
        (procedural_rules, proactive_recall_enabled),
    ) = await asyncio.gather(
        _compute_query_embedding(embedding_provider, query),
        _active_semantic_record_count(memory_store, owner_id=owner_id),
        memory_store.arecord_count(episodic_ns),
        _retrieve_procedural_state(memory_store, owner_id=owner_id),
    )

    # ── Phase 2: retrieval that depends on the embedding ─────────────
    episodic_entries, semantic_entries = await asyncio.gather(
        _retrieve_episodic_working_memory(
            memory_store,
            owner_id=owner_id,
            query=query,
            query_embedding=query_embedding,
            embedding_model=query_embedding_model,
            is_first_turn=is_first_turn,
        ),
        _retrieve_semantic_working_memory(
            memory_store,
            owner_id=owner_id,
            query=query,
            query_embedding=query_embedding,
            embedding_model=query_embedding_model,
        ),
    )

    retrieval_duration_ms = (time.monotonic() - retrieval_start) * 1000

    # Merge: episodic entries first (they frame the session), then
    # semantic entries. Procedural rules are NOT in working_memory —
    # they live on memory.procedural_rules and get injected into the
    # system prompt suffix by Stage D prompt builders, not referenced
    # as content.
    working_memory = [*episodic_entries, *semantic_entries]

    summary = (
        f"Retrieved {len(semantic_entries)} of {semantic_store_size} semantic + "
        f"{len(episodic_entries)} of {episodic_store_size} episodic record(s), "
        f"{len(procedural_rules)} procedural rule(s), "
        f"recall={'on' if proactive_recall_enabled else 'off'} "
        f"path={retrieval_path} "
        f"(query had {len(meaningful_query_tokens)} meaningful token(s))."
    )

    logger.info(
        "load_memory_node: semantic=%d/%d episodic=%d/%d procedural=%d "
        "recall=%s path=%s first_turn=%s query_tokens=%d duration_ms=%.2f owner=%r",
        len(semantic_entries),
        semantic_store_size,
        len(episodic_entries),
        episodic_store_size,
        len(procedural_rules),
        "on" if proactive_recall_enabled else "off",
        retrieval_path,
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
            "procedural_rules": procedural_rules,
            "proactive_recall_enabled": proactive_recall_enabled,
        },
        # v0.8 observability: flow the retrieval timing + per-layer
        # counts into the diagnostics dict so the CLI can render them
        # in the post-turn panel. The ``diagnostics`` field uses a
        # merge reducer (``_merge_dicts`` in state.py), so each node
        # returns only its own keys and LangGraph merges them
        # automatically — no manual dict spreading needed.
        #
        # v0.8.1: ``retrieval_path`` reports which scorer actually
        # ran — one of ``"hybrid_rrf"`` (embedding + token-recall
        # fused via RRF), ``"token_recall"`` (no embedding provider
        # configured or returned None), or
        # ``"token_recall_after_embed_error"`` (embedding call raised
        # and we fell back). Dogfood can watch this to verify the
        # hybrid path is actually running under normal operation.
        "diagnostics": {
            "load_memory_ms": round(retrieval_duration_ms, 2),
            "semantic_hits": len(semantic_entries),
            "semantic_store_size": semantic_store_size,
            "episodic_hits": len(episodic_entries),
            "episodic_store_size": episodic_store_size,
            "procedural_count": len(procedural_rules),
            "proactive_recall": proactive_recall_enabled,
            "retrieval_path": retrieval_path,
        },
    }

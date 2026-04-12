"""Phase-1 memory node signature sketch.

This file is a **design artifact**, not production code. It exists to
document the signatures, return shapes, and graph-topology relationships
of the memory nodes that will be implemented in phase 1 (and beyond),
**before** any actual implementation lands.

Why a sketch instead of an implementation?
- Phase 1 memory nodes need somewhere to write *from*. Until there is at
  least one therapeutic response node producing real conversational
  content, there is nothing for the extraction node to extract from. The
  sketch lets us lock the signatures while waiting for response content
  to land.
- Sketching surfaces signature mismatches between the new memory nodes
  and the existing graph topology before any code commits to data shapes.
- It serves as the bridge between ``schema.yaml`` (the data spec) and the
  eventual implementation files in ``agent/nodes/``.

How to read this file:
- Each function has a **complete signature** (types, parameters, return)
  and a **detailed docstring**, but the body is just ``...``.
- Each node is tagged with the **phase** it belongs to (1 / 2 / 3 / 4).
- Each node references the **schema.yaml section** it implements.
- Each node has a **graph topology comment** showing where it slots in
  relative to the existing nodes.
- The "Proposed WorkflowContext changes" block at the bottom shows what
  ``runtime_context.py`` needs to gain to support the new nodes.

This file is **not** imported by anything in production. It is safe to
modify, delete, or migrate piece-by-piece into ``agent/nodes/`` files when
phase 1 implementation actually begins.

Status: design draft, locked against schema v1 (2026-04-10).
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState

# ─── Phase 1: hot-path memory nodes ──────────────────────────────────────────
#
# These four nodes are the minimum viable memory layer. Once they exist (and
# at least one therapeutic response node exists for them to anchor to), the
# graph has all three CoALA layers operational at a basic level:
#   - Semantic: extract_semantic_facts_node writes facts after each response
#   - Episodic: summarize_session_node writes session arcs at session end
#   - Procedural: explicit_procedural_writer_node writes rules on user request
#
# Plus the always-on safety log:
#   - Crisis log: crisis_log_node writes a record on every crisis event


async def load_long_term_memory_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Load long-term memory snippets at the start of each turn.

    Replaces the current ``run_load_memory_node`` (which today only loads
    the bootstrap-stub working memory). The new node uses the unified
    memory store via ``runtime.context["memory_store"]`` to fetch:

    1. Top semantic facts by similarity to the current message (limit 5).
    2. Top episodic session arcs by similarity (limit 2, recent only).
    3. The procedural rule list (always; no similarity filter).
    4. (Phase 3+) 1-hop graph expansion from the seed entities returned
       by the semantic search.

    All four are merged, deduped, and returned in the ``working_memory``
    field. The downstream prompt builders inject ``working_memory`` into
    the response generation prompt.

    Schema reference: §5 retrieval.per_turn

    Graph topology:
        START → load_long_term_memory_node → crisis_gate_node → ...
        (replaces the existing load_memory_node at the same position)

    Phase: 1

    Args:
        state: Current graph state. Reads ``message``, ``history``,
            ``user_id``, ``session_id``.
        runtime: LangGraph runtime. Reads ``memory_store``, ``memory_mode``.

    Returns:
        Delta dict containing only ``working_memory`` (list[str]),
        ``memory`` (the rolling SessionMemoryState), and ``progress``
        (with the bootstrap stage info). All other state fields are
        untouched.
    """
    ...


async def extract_semantic_facts_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Extract memory-worthy semantic facts from the current turn.

    Runs **after** the response generation node (currently only
    ``crisis_response_node``; eventually also a therapeutic response node).
    Looks at the user message + assistant reply and emits zero or more
    structured ``MemoryWrite`` items via the unified memory store.

    The extractor is deliberately conservative:
    - Most turns produce zero writes (small talk, transient feelings,
      ambiguous claims).
    - Only memory-worthy facts (named people, named events, expressed
      preferences, named coping strategies, declared goals) get written.
    - Anything the user explicitly asked to forget is skipped.

    Hot-path deduplication: before writing, the node checks vector
    similarity against existing facts for this user. If similarity ≥ 0.95
    against ``text-embedding-3-small``, the existing fact's
    ``last_referenced_at`` is bumped instead of writing a duplicate.

    Schema reference:
        §2 namespaces.semantic
        §6 write_policies.semantic
        §9 q5 (deduplication threshold rationale)

    Graph topology:
        ... → response_node → extract_semantic_facts_node → END
        (or → therapeutic_response_node → extract_semantic_facts_node → END)

    Phase: 1

    Side effects:
        Writes 0–N records to the semantic namespace via runtime memory
        store. Returns no state delta — the writes happen as side effects
        and are not reflected in graph state.

    Args:
        state: Current graph state. Reads ``message``, ``history``,
            ``user_id``, ``session_id``, ``response.text``.
        runtime: LangGraph runtime. Reads ``memory_store``, ``llm_client``,
            ``memory_mode``.

    Returns:
        Empty dict (no state changes). The writes are side effects.

    Notes:
        - In ``incognito`` mode, this node is a no-op — writes go to the
          in-memory store which is discarded at runtime exit.
        - In ``local`` and ``synced`` modes, writes persist to the
          configured backend.
        - Failures in this node should NEVER fail the parent turn —
          always catch exceptions and log to ``logger.warning(...,
          exc_info=True)``.
    """
    ...


async def summarize_session_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate an end-of-session summary and write it to episodic memory.

    Runs **once per session, at session end**. Reads the full transcript
    from the checkpointer and produces a structured ``SessionArc`` record:
    primary themes, summary, mood arc, open loops, resolved threads,
    and the peak crisis level for the session.

    Triggers:
    - Explicit ``/end`` command from the user.
    - 20 minutes of inactivity (hardcoded in v1; see §9 q2).
    - Runtime shutdown (CLI close, server graceful shutdown).

    The session boundary detection is **not** part of this node — it
    lives in the runtime layer (``persistence.py``) and invokes this
    node when the trigger fires. This node assumes "we have decided to
    end the session; now write the arc."

    Schema reference:
        §2 namespaces.episodic
        §6 write_policies.episodic

    Graph topology:
        Out-of-band — does NOT live in the per-turn workflow. Triggered
        from the runtime layer at session end. Has its own
        single-node mini-workflow when invoked.

    Phase: 1 (writer) / 2 (background trigger)

    Phase 1 vs phase 2 split:
        - Phase 1: ships the node itself plus the explicit ``/end``
          trigger. Inactivity timeout and runtime-shutdown triggers
          are wired in phase 2 alongside the catch-up-at-startup logic.
        - Phase 2: adds the inactivity timer and shutdown hooks.

    Side effects:
        Writes one record to the episodic namespace via runtime memory
        store. Optionally writes a Session entity to the graph store
        (phase 3+).

    Args:
        state: Final state of the session being summarized. Reads
            ``user_id``, ``session_id``, ``transcript``, ``crisis``,
            ``progress``.
        runtime: LangGraph runtime. Reads ``memory_store``,
            ``llm_client``, ``memory_mode``.

    Returns:
        Delta dict containing ``progress.stage = "closed"`` and a
        ``response.text`` farewell message. The actual SessionArc is
        written as a side effect.

    Notes:
        - In ``incognito`` mode, this node is a no-op.
        - The summarizer prompt is structured-output (Pydantic
          ``SessionArc`` schema), temperature 0.
        - The summarizer reads the FULL transcript, not the working
          memory window — episodic records benefit from seeing the
          complete arc.
    """
    ...


async def crisis_log_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Append a crisis event to the always-on safety log.

    Runs alongside ``crisis_response_node`` whenever a crisis event is
    detected. **Always writes regardless of memory mode** — this is the
    "we don't remember you, but we do log safety events" asymmetry that
    makes the privacy promise compatible with operator audit requirements.

    The log record contains classifier metadata (level, override_kind,
    classifier_path, reason) and outcome flags (response_node_completed,
    llm_failure_occurred). It does NOT contain the user's message text
    or any conversation history.

    In incognito mode, the record's ``user_id_or_null`` is null and the
    ``session_id_opaque`` is the SHA-256 hash of the session_id with no
    reverse mapping. In local/synced modes, both fields are populated.

    Retention: 90 days for per-user records; aggregate stats indefinite.
    See §9 q6 for the legal-review caveat.

    Schema reference:
        §2 namespaces.crisis_log
        §6 write_policies.crisis_log

    Graph topology:
        ... → crisis_gate_node → crisis_response_node → crisis_log_node → END
        (the log node runs after crisis_response so it can record
         response_node_completed)

    Phase: 1

    Side effects:
        Writes one record to the crisis_log namespace via the always-on
        crisis log backend (independent of the user-facing memory store).
        The backend is configured per mode but is always non-null.

    Args:
        state: Current graph state. Reads ``crisis``, ``session_id``,
            ``user_id`` (may be null in incognito), ``response``.
        runtime: LangGraph runtime. Reads ``crisis_log_backend``,
            ``memory_mode``.

    Returns:
        Empty dict (no state changes). The write is a side effect.

    Notes:
        - Failures in this node MUST be loud — log at ERROR level with
          ``exc_info=True`` because a silent crisis log failure means
          the operator has no audit trail when something goes wrong.
        - The node deliberately runs AFTER ``crisis_response_node`` so
          it can record whether the response actually completed.
        - The node does NOT delete or modify any existing crisis log
          records. Retention purging is handled by a separate daily
          cleanup job (phase 2).
    """
    ...


async def explicit_procedural_writer_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Write a procedural rule when the user explicitly requests one.

    Detects user statements like:
    - "Please don't suggest meditation again."
    - "Stop asking me clarifying questions."
    - "Keep your responses shorter."

    When detected, immediately writes a procedural rule with:
    - The rule itself in second-person, evidence-grounded form.
    - The user's explicit request as the evidence_quote.
    - ``confidence: high`` (explicit user request is the highest signal).
    - ``source: explicit_user``.

    The detection step is a small structured-output LLM call that runs
    on every turn but typically returns nothing — explicit rule requests
    are rare. When it returns something, the rule is written immediately
    (no batching, no consolidation pass needed).

    The phase-4 background consolidation pass also writes procedural
    rules (inferred from accumulated facts), but those use a different
    code path with a different evidence requirement (3+ facts).

    Schema reference:
        §2 namespaces.procedural
        §6 write_policies.procedural
        §9 q3 (rule visibility and phrasing constraint)

    Graph topology:
        ... → response_node → explicit_procedural_writer_node → END
        (runs in parallel with extract_semantic_facts_node; both are
         post-response side-effect nodes)

    Phase: 1

    Side effects:
        Writes 0–1 procedural rule(s) to the procedural namespace.
        Returns no state delta.

    Args:
        state: Current graph state. Reads ``message``, ``history``,
            ``user_id``.
        runtime: LangGraph runtime. Reads ``memory_store``,
            ``llm_client``, ``memory_mode``.

    Returns:
        Empty dict (no state changes). The write is a side effect.

    Notes:
        - The rule MUST be written in second-person, evidence-grounded
          form. The detection prompt explicitly instructs the LLM:
            "Return the rule as 'You've said X, so I should Y'.
             Never use 'User dislikes X' or 'User wants Y'."
        - In incognito mode, this node is a no-op.
        - This node also handles the ``/memory recall on/off`` toggle
          when the user runs those commands — the toggle is stored as
          a special record in the procedural namespace.
    """
    ...


# ─── Phase 2: search and bulk-operations nodes ───────────────────────────────


async def memory_search_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Find memory records matching a search term across all layers.

    Used by the ``/memory search <term>`` CLI command. NOT part of the
    per-turn workflow — invoked on demand by the CLI command handler.

    Searches across semantic, episodic, and procedural namespaces using
    keyword + embedding similarity. Returns structured results grouped
    by layer for the CLI to render.

    Schema reference: §8 cli_commands.phase_2

    Phase: 2

    Args:
        state: Minimal — uses ``user_id`` only. Most search context comes
            from the search term, not from graph state.
        runtime: LangGraph runtime. Reads ``memory_store``, ``llm_client``
            (for embedding), ``memory_mode``.

    Returns:
        Delta dict with ``response.text`` set to a structured search
        result payload that the CLI renders. Does not modify any
        long-term memory.

    Notes:
        - Search results are PREVIEWS, not commits. The user must
          explicitly confirm a delete via ``/memory forget search ...``
          before any records are removed.
        - This node is read-only. It never writes to memory.
    """
    ...


async def crisis_log_aggregate_rollup_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Daily background job: roll per-user crisis logs into aggregate stats.

    Reads the previous day's per-user crisis log records and writes one
    aggregate statistic record to the ``crisis_log_aggregate`` namespace.
    The aggregate record contains:
    - Daily event count
    - Counts by level (0-3)
    - Counts by classifier path (deterministic / llm_fallback / override)
    - Total LLM failures
    - Response node completion rate

    The aggregate record contains NO per-user identifiers and is retained
    indefinitely (vs. the 90-day retention on per-user records).

    Schema reference:
        §2 namespaces.crisis_log_aggregate
        §9 q6

    Graph topology:
        Out-of-band — runs as part of the daily cleanup job, not in the
        per-turn workflow.

    Phase: 2

    Side effects:
        Writes one record per day to the crisis_log_aggregate namespace.
        Does NOT delete the per-user records — that's handled separately
        by the retention purger (also phase 2).

    Args:
        state: Unused (out-of-band background job).
        runtime: LangGraph runtime. Reads ``crisis_log_backend``.

    Returns:
        Empty dict.
    """
    ...


# ─── Phase 3: graph reasoning nodes ──────────────────────────────────────────


async def graph_expansion_query_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """1-hop graph expansion from semantic search seeds.

    Phase-3 enhancement to the load-memory node. After
    ``load_long_term_memory_node`` retrieves the top semantic hits,
    this node takes the entities mentioned in those hits and expands
    via 1-hop graph traversal — for each seed entity, fetch its
    directly connected facts via the graph store.

    The point: vector search finds facts that are *lexically similar* to
    the current message; graph expansion finds facts that are *connected*
    to those facts via the user's relationship graph. The combination
    gives the agent both kinds of relevance.

    Schema reference: §5 retrieval.per_turn.graph_expansion

    Graph topology:
        Could either:
          (a) Be a separate node that runs after load_long_term_memory_node
              and writes its results back into working_memory, OR
          (b) Be folded into load_long_term_memory_node's implementation
              (called as a function, not a node).
        Decision deferred to phase 3 implementation. Option (b) is
        simpler; option (a) is more inspectable in LangSmith traces.

    Phase: 3

    Args:
        state: Reads ``working_memory`` (the seeds from semantic search),
            ``user_id``.
        runtime: LangGraph runtime. Reads ``memory_store`` (which
            internally fans out to the graph backend).

    Returns:
        Delta dict with ``working_memory`` updated to include the
        graph-expanded facts.

    Notes:
        - Hops: 1 (hot-path latency budget). Deeper traversals only via
          the on-demand query path.
        - Edge filter: includes [KNOWS, WORRIES_ABOUT, EXPERIENCED, USES,
          WANTS]; excludes [OVERLAPS_WITH, RELATES_TO]. The hot path
          avoids inter-entity edges to keep results focused on facts
          directly about the user.
    """
    ...


# ─── Phase 4: consolidation nodes ────────────────────────────────────────────


async def consolidation_pass_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Background consolidation pass that merges, contradicts, and promotes.

    The phase-4 background job that runs nightly (or on accumulated-write
    threshold). Reads recent semantic facts and episodic records for one
    user, runs an LLM consolidation pass, and applies high-confidence
    proposals.

    Proposal types:
    - merge_facts: collapse near-duplicates that escaped hot-path dedup
    - mark_contradiction: flag conflicting facts
    - promote_to_procedural: turn a recurring fact into a style rule
      (requires ≥3 evidence quotes; see §7 promote_to_procedural_constraint)
    - infer_graph_edge: add inter-entity edges (OVERLAPS_WITH, etc.)
    - mark_dormant: age out unreferenced facts

    Background merge threshold: 0.85 against ``text-embedding-3-small``.
    Hot-path threshold (0.95) is permanent and not affected by this node.

    Schema reference:
        §7 consolidation.phase_4
        §9 q5 (tiered threshold rationale)

    Graph topology:
        Out-of-band — runs as a background job, not in the per-turn
        workflow. Invoked by the in-process scheduler (Option B from the
        consolidation discussion) with catch-up-at-startup logic.

    Phase: 4

    Side effects:
        - Writes/modifies records in semantic, procedural, and graph
          namespaces according to applied proposals.
        - Writes one row to the ``consolidation_runs`` log table for
          observability.

    Args:
        state: Unused (out-of-band background job).
        runtime: LangGraph runtime. Reads ``memory_store``, ``llm_client``.
            Also requires the per-user advisory lock to prevent
            concurrent consolidation in synced mode.

    Returns:
        Empty dict.

    Notes:
        - Concurrency: in local mode, an asyncio.Lock prevents reentrancy.
          In synced mode, ``pg_advisory_lock(hashtext('consolidate-' ||
          user_id))`` provides distributed locking.
        - The LLM call is the expensive part (5-15 seconds, several
          thousand tokens). This is why consolidation is a background
          job, not a hot-path operation.
        - Each merge proposal must include ``evidence_fact_ids: list[str]``
          to prevent LLM hallucination of patterns that don't exist.
    """
    ...


# ─── Proposed WorkflowContext changes ────────────────────────────────────────
#
# The current WorkflowContext (in agent/runtime_context.py) has four fields:
#
#     class WorkflowContext(TypedDict):
#         llm_client: BaseLLMClient | None
#         profile_memory_store: SqliteProfileMemoryStore   # legacy name
#         graph_memory_store: GraphMemoryStore               # legacy name
#         is_guest_mode: bool
#
# Phase 1 collapses ``profile_memory_store`` and ``graph_memory_store`` into a
# single unified ``memory_store`` (the OpenCouchMemoryStore that fans out
# internally). The ``is_guest_mode`` boolean becomes a richer ``memory_mode``
# enum. A new ``crisis_log_backend`` field is added because the crisis log
# is always-on regardless of memory mode and lives in its own backend.
#
# Proposed phase-1 shape:
#
#     class MemoryMode(StrEnum):
#         INCOGNITO = "incognito"
#         LOCAL = "local"
#         SYNCED = "synced"
#
#     class WorkflowContext(TypedDict):
#         llm_client: BaseLLMClient | None
#         memory_store: OpenCouchMemoryStore         # unified, replaces both legacy stores
#         crisis_log_backend: CrisisLogBackend       # always-on, mode-independent
#         memory_mode: MemoryMode                    # replaces is_guest_mode boolean
#
# The migration is straightforward:
#   1. Rename is_guest_mode → memory_mode (boolean → enum, with "incognito"
#      mapping to old guest mode and "local" mapping to old non-guest).
#   2. Build OpenCouchMemoryStore as a thin wrapper around the existing
#      profile_memory_store + graph_memory_store stubs initially, and
#      replace the internals piece-by-piece as the real backends land.
#   3. Add crisis_log_backend as a new field with a NullCrisisLogBackend
#      stub for incognito mode and SqliteCrisisLogBackend for local.
#   4. Update agent/persistence.py to construct the right memory_store
#      and crisis_log_backend based on the mode.
#
# All four changes are scoped to ``runtime_context.py``, ``persistence.py``,
# and the node files. The graph topology in ``graph.py`` stays unchanged
# until the new nodes are wired in.


# ─── Proposed phase-1 graph topology ─────────────────────────────────────────
#
# Current topology (from agent/graph.py:113-139):
#
#     START
#       → load_memory_node       (stub bootstrap reply)
#       → crisis_gate_node       (Command-routes to crisis or END)
#       → [crisis_response_node | END]
#
# Phase-1 target topology after this sketch is implemented:
#
#     START
#       → load_long_term_memory_node    (replaces load_memory_node)
#       → crisis_gate_node              (unchanged)
#       → [crisis_response_node | <therapeutic_response_node TBD>]
#                                       (the therapeutic branch terminates
#                                        at END today; will get its own
#                                        response node when content lands)
#       → crisis_log_node               (new; runs after crisis_response)
#       → extract_semantic_facts_node   (new; runs after any response)
#       → explicit_procedural_writer_node (new; runs in parallel with
#                                          semantic extraction)
#       → END
#
# The summarize_session_node and consolidation_pass_node are NOT in the
# per-turn graph — they're out-of-band, triggered by the runtime layer
# (persistence.py) at session end and on the daily schedule respectively.
#
# Topology notes:
# - extract_semantic_facts_node and explicit_procedural_writer_node can
#   run in parallel (they don't depend on each other and they write to
#   different namespaces). LangGraph supports this naturally — both nodes
#   become outgoing edges from the response node, and both have edges to
#   END.
# - crisis_log_node runs only on the crisis branch. It has its own edge
#   from crisis_response_node, parallel to the semantic+procedural pair.
# - The "no therapeutic response node yet" gap is the reason this sketch
#   is the next deliverable instead of the implementation. Once a
#   therapeutic response node lands, the topology can be wired and the
#   memory nodes have content to process.


# ─── Open implementation questions ───────────────────────────────────────────
#
# These are NOT design questions (the schema v1 has those resolved). They
# are implementation questions that will be answered when the actual code
# is written, but worth flagging here so they're not forgotten:
#
# 1. Should extract_semantic_facts_node and explicit_procedural_writer_node
#    share an LLM call (one prompt produces both kinds of writes) or use
#    separate calls? Sharing reduces cost; separating is more focused and
#    easier to test. Lean toward separate calls in v1.
#
# 2. The summarize_session_node summarizer prompt — how much of the
#    transcript does it see? The full thing (up to ~24k tokens for a 1-hour
#    session) or a sliding window? Probably full thing since modern models
#    handle 100k+ context easily.
#
# 3. The hot-path dedup check needs to compute embeddings for the candidate
#    fact AND fetch nearby existing facts. Batching multiple new facts into
#    one embedding call reduces latency but adds complexity. Phase 1 can
#    do one-at-a-time; optimize later if needed.
#
# 4. Where does the "session boundary detection" actually live? The
#    summarize_session_node is the writer, but who calls it? Probably
#    a method on PersistentAgentRuntime that checks last-activity
#    timestamp on every turn and fires the summarizer if it crosses the
#    20-minute threshold. Catch-up-at-startup is a separate code path
#    in the runtime's __aenter__.
#
# 5. The /memory CLI commands need to talk directly to the memory store,
#    not through the LangGraph workflow. They're synchronous user
#    commands, not graph turns. The CLI command handler should construct
#    a memory store reference from the runtime and call store methods
#    directly. This means the OpenCouchMemoryStore needs both a "graph
#    runtime" interface (used by nodes via runtime.context) and a
#    "direct" interface (used by the CLI).

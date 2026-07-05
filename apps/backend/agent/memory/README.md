# Memory Module

`agent.memory` is the OpenCouch long-term prompt-memory subsystem. It owns the
records that can later influence therapeutic responses, plus the policy and
retrieval code that decides when those records are read or written.

Postgres-backed typed records are the source of truth for durable deployments.
In-memory stores are for incognito/test paths, and SQLite remains an explicit
legacy fallback while migration work continues.

## Boundary

Use `agent.memory` for:

- semantic facts and user-context records that may affect future prompts
- episodic session arcs and summaries
- procedural response preferences and recall controls
- retrieval, ranking, deduplication, reconciliation, and write policy
- user-facing memory control services such as recall, save, and forget
- per-session candidate buffers promoted at session end

Do **not** use `agent.memory` for:

- safety audit rows; those live in [`agent/audit`](../audit)
- explicit feedback ratings; those live in [`agent/feedback`](../feedback)
- raw OpenAI SDK session history; that lives under [`agent/runtime`](../runtime)
- tool schemas; those live in [`agent/tools`](../tools)

## Mental Model

There are three durable memory kinds:

1. **Semantic memory**
   - many records per user
   - factual/user-context style data
   - retrieved by lexical + optional embedding ranking

2. **Episodic memory**
   - many records per user
   - session summaries and arcs
   - used as cross-session context and support evidence for future writes

3. **Procedural memory**
   - one profile per user
   - response-style rules plus `proactive_recall_enabled`
   - updated by explicit user controls and policy-gated write paths

A fourth shape, `SessionMemoryBuffer`, is runtime-owned session state rather than
long-term memory. It holds candidates until the session-end commit pass decides
what should become durable semantic or procedural memory.

## Claude-Style UX, Postgres Source of Truth

The target user/developer experience is inspired by Claude Code's memory model:
a concise always-readable index plus topic-oriented details that can be inspected
on demand. In OpenCouch, that should be a **generated/read model over typed
Postgres records**, not raw markdown as the canonical store.

Practical direction:

- Keep Postgres typed records as the durable source of truth.
- Keep semantic / episodic / procedural schemas explicit and testable.
- Expose a notebook-like inspection layer in future UI/CLI work:
  - a concise memory index: "what do we remember?"
  - topic groupings: preferences, coping strategies, relationships, goals,
    sensitivities, session arcs
  - provenance: source session, timestamps, confidence/policy reason, and
    whether a record can influence prompts
  - controls: delete, toggle recall, and inspect why a memory was or was not
    written
- Treat procedural preferences as the closest analogue to auto-memory notes.
- Keep therapeutic semantic/episodic memory policy-gated because it is sensitive
  user data, not general coding-agent notes.

This gives us Claude-style inspectability without giving up consent, deletion,
retention, multi-user storage, incognito behavior, or structured retrieval.

## Current Data Flow

### Per-turn recall

1. `agent.runtime.memory_context` decides whether recall is allowed for the turn
   and calls `memory.retrieval.service`.
2. `memory.retrieval.service` loads semantic, episodic, and procedural records
   for the resolved owner.
3. `memory.retrieval.ranking` performs lexical / dense / RRF ranking where
   applicable.
4. Retrieved records are converted into structured working-memory entries via
   `memory.entries`.
5. Specialist prompt builders render those entries into response prompts.

### Hot-path candidate capture

1. Response/runtime paths identify possible semantic or procedural candidates.
2. `memory.policy.write` and policy clamps decide whether to write now, hold for
   session end, require repetition, or skip.
3. Held candidates go into `SessionMemoryBuffer` under `memory.policy.candidates`.
4. The buffer is attached to active-session state, not immediately persisted as
   long-term memory.

### Session-end commit

1. `agent.runtime.session.commit` orchestrates session finalization memory work.
2. `memory.commit.service` evaluates held candidates with clustering, scoring,
   prior-session support, and overlap checks.
3. Semantic writes go through `memory.operations.semantic_writes`.
4. Procedural writes go through `memory.operations.procedural_profile`.
5. Episodic session arcs are extracted/summarized and persisted for future recall
   and support evidence.

### User-facing memory controls

1. Text/voice tools and runtime dispatch route explicit memory commands to
   `memory.control.service`.
2. Read, mutation, deletion, and pending-action helpers live under
   `memory.control`.
3. Control paths respect memory mode and owner resolution; they should not bypass
   policy or incognito boundaries.

## Package Map

### Storage

- [`store/base.py`](./store/base.py): `MemoryStore` protocol, `StoreRecord`,
  namespace helpers, and shared record parsing.
- [`store/memory.py`](./store/memory.py): in-memory store for incognito/tests.
- [`store/postgres.py`](./store/postgres.py): primary durable Postgres store.
- [`store/sqlite.py`](./store/sqlite.py): legacy SQLite fallback store.
- [`modes.py`](./modes.py): `MemoryMode` enum used by runtime wiring.

### Retrieval

- [`retrieval/service.py`](./retrieval/service.py): per-turn recall entry point
  used by the runtime memory context.
- [`retrieval/ranking.py`](./retrieval/ranking.py): lexical ranking, dense
  ranking, cosine similarity, and Reciprocal Rank Fusion.
- [`providers/embeddings.py`](./providers/embeddings.py): embedding provider
  protocol and provider factory.
- [`entries.py`](./entries.py): working-memory entry models and rendering
  helpers.
- [`notebook.py`](./notebook.py): read-only Claude-style inspection view over
  existing typed memory records.
- [`text_tokens.py`](./text_tokens.py): shared tokenization helpers used by
  retrieval and dedup.

### Write Policy And Candidate Buffers

- [`policy/candidates.py`](./policy/candidates.py): semantic/procedural
  candidates plus `SessionMemoryBuffer`.
- [`policy/write.py`](./policy/write.py): LLM-primary write-timing policy.
- [`policy/clamps.py`](./policy/clamps.py): post-policy safety/storage clamps.
- [`policy/thresholds.py`](./policy/thresholds.py): policy thresholds.
- [`policy/markers.py`](./policy/markers.py): marker helpers for memory-control
  requests.

### Operations

- [`operations/semantic_writes.py`](./operations/semantic_writes.py): semantic
  batch write, bump, and skip handling.
- [`operations/procedural_profile.py`](./operations/procedural_profile.py):
  procedural profile reads, rule construction, upsert, and recall toggles.
- [`operations/reconciliation.py`](./operations/reconciliation.py): conservative
  merge / replace / skip planning.
- [`operations/dedup.py`](./operations/dedup.py): semantic near-duplicate checks.
- [`operations/episodic.py`](./operations/episodic.py): episodic session-arc
  helpers.

### Session-End Commit

- [`commit/service.py`](./commit/service.py): session-end promotion pass.
- [`commit/clustering.py`](./commit/clustering.py): semantic/procedural cluster
  text helpers.
- [`commit/scoring.py`](./commit/scoring.py): support and transcript scoring
  helpers.
- [`commit/selection.py`](./commit/selection.py): candidate selection and
  semantic/procedural overlap resolution.

### Control Plane

- [`control/service.py`](./control/service.py): user-facing memory-control
  service facade.
- [`control/read_service.py`](./control/read_service.py): memory inspection and
  recall reads.
- [`control/mutation_service.py`](./control/mutation_service.py): preference and
  saved-memory mutations.
- [`control/deletion_service.py`](./control/deletion_service.py): deletion and
  pending delete flows.
- [`control/actions.py`](./control/actions.py),
  [`control/operations.py`](./control/operations.py), and
  [`control/types.py`](./control/types.py): shared action helpers and schemas.

### Extraction And Prompts

- [`extraction/session_extractor.py`](./extraction/session_extractor.py):
  session-level memory extraction.
- [`prompts/session_extraction.py`](./prompts/session_extraction.py): extraction
  prompt builders.
- [`prompts/summarization.py`](./prompts/summarization.py): summarization prompt
  builders.

### Types And Utilities

- [`types/semantic.py`](./types/semantic.py): semantic memory models.
- [`types/episodic.py`](./types/episodic.py): episodic session-arc models.
- [`types/procedural.py`](./types/procedural.py): procedural profile and rule
  models.
- [`types/therapeutic.py`](./types/therapeutic.py): therapeutic context models.
- [`types/primitives.py`](./types/primitives.py): shared primitive types.
- [`hashing.py`](./hashing.py): owner/session hashing helpers.

## Common Entry Points

- "How is memory stored?" Start with [`store/base.py`](./store/base.py), then
  [`store/postgres.py`](./store/postgres.py).
- "How does recall work?" Start with
  [`runtime/memory_context.py`](../runtime/memory_context.py), then
  [`retrieval/service.py`](./retrieval/service.py) and
  [`retrieval/ranking.py`](./retrieval/ranking.py).
- "Why did a fact/rule get written or skipped?" Start with
  [`policy/write.py`](./policy/write.py), [`policy/clamps.py`](./policy/clamps.py),
  and [`operations/reconciliation.py`](./operations/reconciliation.py).
- "What happens at session end?" Start with
  [`runtime/session/commit.py`](../runtime/session/commit.py), then
  [`commit/service.py`](./commit/service.py).
- "How does the user inspect, save, or forget memory?" Start with
  [`control/service.py`](./control/service.py). For a grouped read-only notebook
  view, see [`notebook.py`](./notebook.py).
- "Where are prompt-facing memory entries formatted?" Start with
  [`entries.py`](./entries.py) and the specialist prompt builders under
  [`agent/specialists`](../specialists).

## Persistence Behavior

- Durable deployments use Postgres selected by runtime configuration.
- Incognito mode uses in-memory stores and must not write durable user memory.
- SQLite is legacy compatibility only and requires explicit opt-in.
- Crisis audit and session feedback persistence are separate from prompt memory
  even when they share the same deployment database.

## Extension Rules

When adding memory behavior:

1. Decide which memory kind it belongs to: semantic, episodic, procedural, or
   session-buffer-only.
2. Preserve owner resolution, memory mode, recall toggle, deletion, and incognito
   boundaries.
3. Store typed records through `MemoryStore`; do not introduce raw markdown or
   ad hoc files as durable source of truth.
4. Add tests at the policy/service boundary and, for storage changes, contract or
   backend round-trip coverage.
5. If the data is operational safety or feedback rather than prompt context, use
   `agent.audit` or `agent.feedback` instead.

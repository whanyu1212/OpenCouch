# Memory Module

This package is the OpenCouch prompt-memory layer.

It owns:
- long-term memory storage abstractions
- retrieval and ranking
- procedural-profile reads and writes
- extraction / summarization / control prompt builders
- write-policy, dedup, and reconciliation helpers
- per-turn write orchestration and session-end commit
- user-facing memory controls (recall toggle, save preferences, forget)
- memory-layer data models and small utility helpers

It does **not** own always-on audit persistence.

Those backends live in [agent/audit](../audit):
- crisis log backends
- session feedback backends

Important distinction:
- `agent.audit.models` owns audit-related record schemas
- Audit types live in `agent.audit.models` (no longer re-exported here)

## Mental Model

There are 3 main memory shapes:

1. Semantic memory
- many records per user
- factual / user-context style data
- retrieved by lexical + embedding ranking

2. Episodic memory
- many records per user
- session summaries / arcs
- retrieved similarly to semantic memory

3. Procedural memory
- one profile document per user
- response-style rules plus `proactive_recall_enabled`
- updated via load → mutate → put helpers

## File Map

### Storage

- [store.py](./store.py): `MemoryStore` protocol, `StoreRecord`, in-memory `OpenCouchMemoryStore`, namespace conventions, search thresholds.
- [postgres_store.py](./postgres_store.py): primary durable Postgres implementation (default backend).
- [sqlite_store.py](./sqlite_store.py): SQLite fallback backend, selectable via `OPENCOUCH_PERSISTENCE_BACKEND=sqlite`.
- [modes.py](./modes.py): `MemoryMode` enum used by the runtime to choose in-memory vs durable behavior.

### Retrieval

- [retrieval.py](./retrieval.py): lexical ranking, dense ranking, cosine similarity, Reciprocal Rank Fusion.
- [recall.py](./recall.py): per-turn retrieval entry point used by the load-memory node; assembles the semantic + episodic + procedural working-memory bundle.
- [embeddings.py](./embeddings.py): embedding provider protocol, OpenAI / Gemini / null providers, provider factory.
- [text_tokens.py](./text_tokens.py): shared tokenizer used by retrieval and dedup.

### Write Pipeline

- [turn_write_service.py](./turn_write_service.py): `TurnWriteService` — per-turn orchestration that maps extracted candidates through policy → dedup → store writes.
- [session_commit_service.py](./session_commit_service.py): session-end commit of buffered candidates.
- [semantic_writes.py](./semantic_writes.py): batch semantic-write helper (`apply_semantic_writes_batch`) shared by turn and session-end paths.
- [dedup.py](./dedup.py): hot-path semantic near-duplicate detection.
- [reconciliation.py](./reconciliation.py): conservative merge / replace / skip planning for semantic and procedural writes.

### Policy & Heuristics

The [policy/](./policy) subpackage owns the decision layer between extracted candidates and persisted writes:

- [policy/candidates.py](./policy/candidates.py): candidate objects for semantic / procedural writes plus `SessionMemoryBuffer`.
- [policy/write.py](./policy/write.py): LLM-primary write policy with hard local safety / storage guards.
- [policy/semantic.py](./policy/semantic.py): semantic heuristics such as durability markers and negative-self-belief detection.
- [policy/small_talk.py](./policy/small_talk.py): pre-extractor filter for turns that should not produce memory writes.
- [policy/turn_routing.py](./policy/turn_routing.py): `should_skip_memory_extraction`, `get_session_turn_index`.
- [policy/constants.py](./policy/constants.py): procedural request classification markers and helpers.

### Procedural Profile

- [procedural_profile.py](./procedural_profile.py): main helper surface for procedural memory; profile reads / writes, rule upserts, proactive-recall toggle.

### Episodic

- [episodic.py](./episodic.py): episodic session-arc helpers used at session-end summarization.

### User Controls

The [user_controls/](./user_controls) subpackage owns user-facing memory commands:

- [user_controls/router.py](./user_controls/router.py): typed action models (discriminated union) and CLI / API routing.
- [user_controls/service.py](./user_controls/service.py): `apply_memory_action` dispatch (`match` / `case` over `TypedMemoryAction`).
- [user_controls/operations.py](./user_controls/operations.py): individual operation handlers (list, status, set_recall, save_preference, forget, confirm, cancel).
- [user_controls/patterns.py](./user_controls/patterns.py): regex / phrase patterns for natural-language memory commands.

### Prompt Builders

The [prompts/](./prompts) subpackage groups all memory-layer prompts:

- [prompts/extraction.py](./prompts/extraction.py): semantic extraction prompts.
- [prompts/procedural.py](./prompts/procedural.py): procedural-rule extraction prompts.
- [prompts/summarization.py](./prompts/summarization.py): session summarization prompts.
- [prompts/control.py](./prompts/control.py): user-control intent classification prompts.

### Utilities

- [hashing.py](./hashing.py): `hash_session_id()` and `iso_now()`.

### Types

- [models.py](./models.py): compatibility export surface for memory-layer models.
- [types/](./types): pydantic model definitions and compatibility exports grouped by concern:
  - `semantic.py`
  - `episodic.py`
  - `procedural.py`
  - `audit.py` compatibility re-exports from `agent.audit.models`
  - `therapeutic.py`
  - `primitives.py`

## Common Entry Points

If you are trying to understand a specific behavior, start here:

- "How is memory stored?"
  Start with [store.py](./store.py), then [postgres_store.py](./postgres_store.py). For SQLite-fallback behavior, see [sqlite_store.py](./sqlite_store.py).

- "How does retrieval work?"
  Start with [retrieval.py](./retrieval.py), then [embeddings.py](./embeddings.py), then [text_tokens.py](./text_tokens.py).

- "How are procedural rules managed?"
  Start with [procedural_profile.py](./procedural_profile.py).

- "Why did a fact or rule get written or skipped?"
  Start with [policy/candidates.py](./policy/candidates.py), [policy/write.py](./policy/write.py), and [reconciliation.py](./reconciliation.py).

- "What runs on every turn vs at session end?"
  Start with [turn_write_service.py](./turn_write_service.py) and [session_commit_service.py](./session_commit_service.py).

- "What prompt is used for extraction, summarization, or user controls?"
  Start in [prompts/](./prompts).

- "How does the user toggle recall or save a preference?"
  Start with [user_controls/router.py](./user_controls/router.py).

- "Where did the crisis log / session feedback code go?"
  Go to [agent/audit](../audit).

## Runtime Wiring

This package is mostly infrastructure. The main runtime integration points are outside this directory:

- [agent/persistence.py](../persistence.py): chooses memory store implementation and owns lifecycle.
- [agent/runtime/backends.py](../runtime/backends.py): selects Postgres vs SQLite vs in-memory based on settings.
- [agent/nodes/](../nodes): extraction, commit, summarization, and memory-loading nodes call into this package.
- [agent/graph.py](../graph.py): wires the nodes into the workflow.

## Persistence Backend

Postgres is the default durable backend (see `core/config.py:DEFAULT_PERSISTENCE_BACKEND`). SQLite remains a supported fallback selectable via `OPENCOUCH_PERSISTENCE_BACKEND=sqlite` for local-only installs without Docker. The in-memory `OpenCouchMemoryStore` is used for `INCOGNITO` mode and tests.

## Practical Boundary

Use `agent.memory` for:
- memory that can later influence prompts or retrieval
- helper logic that decides what should become memory
- shared schemas for semantic / episodic / procedural records

Use `agent.audit` for:
- safety / audit records that must be persisted regardless of prompt-memory behavior
- session feedback persistence
- operator-facing audit trails

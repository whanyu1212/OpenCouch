# Memory Module

This package is the OpenCouch prompt-memory layer.

It owns:
- long-term memory storage abstractions
- retrieval and ranking
- procedural-profile reads and writes
- extraction and summarization prompt builders
- write-policy, dedup, and reconciliation helpers
- per-turn write orchestration and session-end commit
- memory helpers used by user-facing memory controls
- memory-layer data models and small utility helpers

It does **not** own always-on audit persistence.
It also does not own the memory-control gate that handles user-facing
commands such as recall toggles, saved preferences, and forget requests.

Those backends live outside `agent.memory`:
- crisis log backends live in [agent/audit](../audit)
- session feedback backends live in [agent/feedback](../feedback)

Important distinction:
- `agent.audit.models` owns audit-related record schemas
- `agent.feedback.models` owns explicit session-feedback schemas

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

- [store/](./store): `MemoryStore` protocol, `StoreRecord`, in-memory `OpenCouchMemoryStore`, namespace conventions, and search thresholds.
- [store/postgres.py](./store/postgres.py): primary durable Postgres implementation (default backend).
- [store/sqlite.py](./store/sqlite.py): SQLite fallback backend, selectable via `OPENCOUCH_PERSISTENCE_BACKEND=sqlite`.
- [modes.py](./modes.py): `MemoryMode` enum used by the runtime to choose in-memory vs durable behavior.

### Retrieval

- [retrieval.py](./retrieval.py): lexical ranking, dense ranking, cosine similarity, Reciprocal Rank Fusion.
- [recall.py](./recall.py): per-turn retrieval entry point used by the runtime turn memory context; assembles the semantic + episodic + procedural working-memory bundle.
- [embeddings.py](./embeddings.py): embedding provider protocol, OpenAI / null providers, provider factory.
- [text_tokens.py](./text_tokens.py): shared tokenizer used by retrieval and dedup.

### Write Pipeline

- [session_commit_service.py](./session_commit_service.py): session-end commit of buffered candidates.
- [semantic_writes.py](./semantic_writes.py): batch semantic-write helper (`apply_semantic_writes_batch`) used by session-end paths.
- [dedup.py](./dedup.py): hot-path semantic near-duplicate detection.
- [reconciliation.py](./reconciliation.py): conservative merge / replace / skip planning for semantic and procedural writes.

### Policy Layer

The [policy/](./policy) subpackage owns the decision layer between memory candidates and persisted writes:

- [policy/candidates.py](./policy/candidates.py): candidate objects for semantic / procedural writes plus `SessionMemoryBuffer`; held candidates carry the policy decision that held them.
- [policy/write.py](./policy/write.py): LLM-primary write-timing policy for semantic / procedural candidates, with narrow post-policy safety / storage clamps.
- [policy/semantic.py](./policy/semantic.py): semantic policy constants for session-only categories.

### Procedural Profile

- [procedural_profile.py](./procedural_profile.py): main helper surface for procedural memory; profile reads / writes, rule upserts, proactive-recall toggle.

### Episodic

- [episodic.py](./episodic.py): episodic session-arc helpers used at session-end summarization.

### Prompt Builders

The [prompts/](./prompts) subpackage groups memory-layer prompts:

- [prompts/summarization.py](./prompts/summarization.py): session summarization prompts.

### Working-Memory Rendering

- [entries.py](./entries.py): structured working-memory entries and rendering helpers for prompts / diagnostics.

### Utilities

- [hashing.py](./hashing.py): `hash_session_id()` and `iso_now()`.

### Types

- [models.py](./models.py): compatibility export surface for memory-layer models.
- [types/](./types): pydantic model definitions and compatibility exports grouped by concern:
  - `semantic.py`
  - `episodic.py`
  - `procedural.py`
  - `therapeutic.py`
  - `primitives.py`

## Common Entry Points

If you are trying to understand a specific behavior, start here:

- "How is memory stored?"
  Start with [store/](./store), then [store/postgres.py](./store/postgres.py). For SQLite-fallback behavior, see [store/sqlite.py](./store/sqlite.py).

- "How does retrieval work?"
  Start with [retrieval.py](./retrieval.py), then [embeddings.py](./embeddings.py), then [text_tokens.py](./text_tokens.py).

- "How are procedural rules managed?"
  Start with [procedural_profile.py](./procedural_profile.py).

- "Why did a fact or rule get written or skipped?"
  Start with [policy/candidates.py](./policy/candidates.py), [policy/write.py](./policy/write.py), and [reconciliation.py](./reconciliation.py).

- "What runs on every turn vs at session end?"
  Start with [recall.py](./recall.py), [session_commit_service.py](./session_commit_service.py), and [episodic.py](./episodic.py).

- "What prompt is used for summarization?"
  Start in [prompts/](./prompts).

- "How does the user toggle recall or save a preference?"
  Start with [control](./control).

- "Where did the crisis log / session feedback code go?"
  Crisis logs live in [agent/audit](../audit); session feedback lives in
  [agent/feedback](../feedback).

## Runtime Wiring

This package is mostly infrastructure. The main runtime integration points are outside this directory:

- [agent/runtime/runtime.py](../runtime/runtime.py): chooses memory store implementation and owns lifecycle.
- [agent/runtime/backends.py](../runtime/backends.py): selects Postgres vs SQLite vs in-memory based on settings.
- [control](./control): handles user-facing memory tool requests and
  memory-control service operations.
- [agent/runtime/memory_context.py](../runtime/memory_context.py): builds the
  runner-turn memory delta consumed by the text runtime.
- [agent/runtime/openai_text_runtime.py](../runtime/openai_text_runtime.py): wires
  memory loading into the OpenAI Agents SDK text runtime.

## Persistence Backend

Postgres is the default durable backend (see `config.py:DEFAULT_PERSISTENCE_BACKEND`). SQLite remains a supported fallback selectable via `OPENCOUCH_PERSISTENCE_BACKEND=sqlite` for local-only installs without Docker. The in-memory `OpenCouchMemoryStore` is used for `INCOGNITO` mode and tests.

## Practical Boundary

Use `agent.memory` for:
- memory that can later influence prompts or retrieval
- helper logic that decides what should become memory
- shared schemas for semantic / episodic / procedural records

Use `agent.audit` for:
- safety / audit records that must be persisted regardless of prompt-memory behavior
- session feedback persistence
- operator-facing audit trails

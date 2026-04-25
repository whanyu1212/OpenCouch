# Memory Module

This package is the OpenCouch prompt-memory layer.

It owns:
- long-term memory storage abstractions
- retrieval and ranking
- procedural-profile reads and writes
- extraction / summarization prompt builders
- write-policy, dedup, and reconciliation helpers
- memory-layer data models and small utility helpers

It does **not** own always-on audit persistence anymore.

Those backends now live in [agent/audit](../audit):
- crisis log backends
- session feedback backends

Important distinction:
- `agent.audit.models` owns audit-related record schemas
- `agent.memory.types.audit` only keeps compatibility re-exports

## Mental Model

There are 3 main memory shapes:

1. Semantic memory
- many records per user
- factual/user-context style data
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
- [sqlite_store.py](./sqlite_store.py): durable SQLite implementation of the same store interface.
- [modes.py](./modes.py): `MemoryMode` enum used by the runtime to choose in-memory vs durable behavior.

### Retrieval

- [retrieval.py](./retrieval.py): lexical ranking, dense ranking, cosine similarity, Reciprocal Rank Fusion.
- [embeddings.py](./embeddings.py): embedding provider protocol, OpenAI/Gemini/null providers, provider factory.
- [text_tokens.py](./text_tokens.py): shared tokenizer used by retrieval and dedup.

### Semantic Write Pipeline

- [candidates.py](./candidates.py): candidate objects for semantic/procedural writes plus `SessionMemoryBuffer`.
- [write_policy.py](./write_policy.py): deterministic promotion policy for whether a candidate should become durable memory.
- [semantic_policy.py](./semantic_policy.py): semantic heuristics such as durability markers and negative-self-belief detection.
- [dedup.py](./dedup.py): hot-path semantic near-duplicate detection.
- [reconciliation.py](./reconciliation.py): conservative merge/replace/skip planning for semantic and procedural writes.

### Procedural Profile

- [procedural.py](./procedural.py): main helper surface for procedural memory; profile reads/writes, rule upserts, proactive-recall toggle.
- [constants.py](./constants.py): procedural request classification markers and helpers.
- [procedural_prompts.py](./procedural_prompts.py): prompt builders for procedural-rule extraction/writing.

### Prompt Builders

- [extraction_prompts.py](./extraction_prompts.py): semantic extraction prompts.
- [summarization_prompts.py](./summarization_prompts.py): session summarization prompts.

### Guards and Utilities

- [small_talk_gate.py](./small_talk_gate.py): pre-extractor filter for turns that should not produce memory writes.
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

- “How is memory stored?”
  Start with [store.py](./store.py), then [sqlite_store.py](./sqlite_store.py).

- “How does retrieval work?”
  Start with [retrieval.py](./retrieval.py), then [embeddings.py](./embeddings.py), then [text_tokens.py](./text_tokens.py).

- “How are procedural rules managed?”
  Start with [procedural.py](./procedural.py).

- “Why did a fact or rule get written or skipped?”
  Start with [candidates.py](./candidates.py), [write_policy.py](./write_policy.py), and [reconciliation.py](./reconciliation.py).

- “What prompt is used for extraction or summarization?”
  Start with [extraction_prompts.py](./extraction_prompts.py), [procedural_prompts.py](./procedural_prompts.py), and [summarization_prompts.py](./summarization_prompts.py).

- “Where did the crisis log / session feedback code go?”
  Go to [agent/audit](../audit).

## Runtime Wiring

This package is mostly infrastructure. The main runtime integration points are outside this directory:

- [agent/persistence.py](../persistence.py): chooses memory store implementation and owns lifecycle.
- [agent/nodes/](../nodes): extraction, commit, summarization, and memory-loading nodes call into this package.
- [agent/graph.py](../graph.py): wires the nodes into the workflow.

## Practical Boundary

Use `agent.memory` for:
- memory that can later influence prompts or retrieval
- helper logic that decides what should become memory
- shared schemas for semantic/episodic/procedural records

Use `agent.audit` for:
- safety/audit records that must be persisted regardless of prompt-memory behavior
- session feedback persistence
- operator-facing audit trails

---
title: Architecture
sidebar_position: 1
---

# Architecture

Single Python backend. Explicit LangGraph agent graph. Three memory
layers backed by SQLite. CLI-first development.

---

## Turn pipeline

:::tip Safety-first ordering
The crisis gate is the **first node** in the graph. Memory only
loads on the therapeutic branch — if a message triggers a crisis
response, memory retrieval is skipped entirely.
:::

```text
START
  → crisis_gate_node              safety first — hybrid regex + LLM
  → [crisis branch]               crisis_response → crisis_log → finalize
     OR
     [therapeutic branch]          load_memory → therapeutic_subgraph → finalize
  → finalize_turn_node             append via operator.add reducer
  → extract_semantic_facts    ┐    parallel fan-out
  → extract_procedural_rules  ┘    diagnostics via _merge_dicts reducer
END
```

Every node with I/O has `RetryPolicy(max_attempts=2)` as
defense-in-depth. `finalize_turn_node` is the only node without
retry — it's pure state, no I/O.

---

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Execution | LangGraph `StateGraph` | Explicit branches, deterministic ordering, checkpoint persistence |
| LLM | `BaseLLMClient` protocol | Provider-agnostic — Gemini + OpenAI satisfy the same interface |
| Embedding | `EmbeddingProvider` protocol | Gemini `text-embedding-004` default; null provider for offline |
| Storage | SQLite via `aiosqlite` | Single-file, no server, survives restart |
| Memory | `MemoryStore` protocol | In-memory for incognito/tests; SQLite for persistent |
| Retrieval | Hybrid RRF | Embedding cosine + token-recall fused via Reciprocal Rank Fusion |
| Knowledge | `knowledge/*.md` files | Reviewed content outside code; composed at runtime |
| Context | `WorkflowContext` frozen dataclass | Attribute access, type-safe, immutable per turn |
| Reducers | `operator.add` + `_merge_dicts` | Transcript accumulation + parallel diagnostics |
| Crisis log | Always-on | Privacy asymmetry — incognito scrubs user_id but still records |

---

## Persistence

:::info Four SQLite databases
Each under `.store/`, each owning its schema independently. Paths
overridable via CLI flags. Incognito mode uses in-memory backends
for all four — nothing touches disk.
:::

| Database | Owner | What it persists | Retention |
|---|---|---|---|
| `threads.sqlite3` | LangGraph checkpointer | Conversation state snapshots | Indefinite |
| `memory.sqlite3` | `SqliteMemoryStore` | Semantic facts, episodic arcs, procedural profiles | User-controlled |
| `crisis.sqlite3` | `SqliteCrisisLogBackend` | Crisis event audit trail | 90 days |
| `session_feedback.sqlite3` | `SqliteSessionFeedbackBackend` | End-of-session thumbs ratings | 180 days |

---

## Prompt layers

Five layers composed per turn, innermost first:

```text
┌─────────────────────────────────────────────────────┐
│  5. Turn posture — response guidance + modality     │
│  ┌───────────────────────────────────────────────┐  │
│  │  4. State context — working memory, rules     │  │
│  │  ┌─────────────────────────────────────────┐  │  │
│  │  │  3. Instructions — mode-specific block  │  │  │
│  │  │  ┌───────────────────────────────────┐  │  │  │
│  │  │  │  2. Mode — response_modes/*.md   │  │  │  │
│  │  │  │  ┌─────────────────────────────┐  │  │  │  │
│  │  │  │  │  1. Core — soul.md + policy │  │  │  │  │
│  │  │  │  └─────────────────────────────┘  │  │  │  │
│  │  │  └───────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

See [Prompt Assembly](/docs/agent/prompt-assembly) for the full
composition logic.

---

## Package layout

| Package | Owns |
|---|---|
| `agent/` | Graph entrypoint, state schema, models, runtime context |
| `agent/nodes/` | One file per graph node (8 nodes) |
| `agent/memory/` | Store protocol, embeddings, retrieval, dedup, hashing |
| `agent/therapeutic/` | Subgraph: dispatcher, six mode nodes, prompts |
| `agent/tools/` | Web search for crisis resources |
| `services/llm/` | `BaseLLMClient` + Gemini + OpenAI clients |
| `opencouch_cli/` | Rich-based interactive CLI |
| `voice/` | OpenAI Realtime voice sessions |
| `api/` | FastAPI routes (chat, threads, memory) |
| `tests/` | 630+ pytest tests |
| `eval/` | 5 eval harnesses with curated datasets |

---

## Quick links

| Topic | Page |
|---|---|
| Agent graph | [Graph](/docs/agent/graph) |
| Node catalog | [Nodes](/docs/agent/nodes) |
| Tools | [Tools](/docs/agent/tools) |
| State schema | [State](/docs/agent/state) |
| Memory | [Memory](/docs/memory/overview) |
| Crisis gate | [Crisis Gate](/docs/philosophy/crisis-gate) |
| Runtime | [Runtime](/docs/backend/runtime) |
| Observability | [Observability](/docs/observability/overview) |
| Privacy | [Privacy](/docs/memory/privacy) |

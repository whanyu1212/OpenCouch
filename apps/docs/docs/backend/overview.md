---
title: Architecture & Backend
sidebar_position: 1
---

# Architecture & Backend

OpenCouch is a single Python backend with an explicit LangGraph agent
workflow, three memory layers backed by SQLite, and a CLI-first
interface for development and dogfood.

<div className="docs-link-grid">
  <a className="docs-link-card" href="/docs/agent/graph">
    <strong>Agent graph</strong>
    <span>Turn pipeline, safety sequencing, and therapeutic subgraph routing.</span>
  </a>
  <a className="docs-link-card" href="/docs/memory/overview">
    <strong>Memory layers</strong>
    <span>Semantic facts, episodic arcs, procedural rules, and hybrid retrieval.</span>
  </a>
  <a className="docs-link-card" href="/docs/backend/runtime">
    <strong>Runtime and persistence</strong>
    <span>LangGraph checkpoints, SQLite runtime, and memory integration model.</span>
  </a>
  <a className="docs-link-card" href="/docs/philosophy/crisis-gate">
    <strong>Crisis safety</strong>
    <span>Always-on crisis gate, audit logging, and retention policy.</span>
  </a>
  <a className="docs-link-card" href="/docs/observability/overview">
    <strong>Observability</strong>
    <span>Per-turn diagnostics, stage timings, and /debug state.</span>
  </a>
  <a className="docs-link-card" href="/docs/agent/prompt-assembly">
    <strong>Prompt assembly</strong>
    <span>Layered prompt construction from knowledge, mode, modality, and turn context.</span>
  </a>
</div>

## Turn pipeline

Every user message flows through the same graph spine. No node is
optional except by early-exit (incognito mode skips memory, no LLM
skips extraction). The ordering is load-bearing — safety runs before
response generation, memory writes run after.

```text
START
  -> load_memory_node         (retrieve semantic + episodic + procedural context)
  -> crisis_gate_node         (hybrid regex + LLM safety classification)
  -> [crisis branch]          crisis_response_node -> crisis_log_node
     OR
     [therapeutic branch]     therapeutic_subgraph (dispatcher -> mode node)
  -> extract_semantic_facts   (LLM structured output, gated by small-talk check)
  -> extract_procedural_rules (LLM structured output, gated by small-talk check)
  -> finalize_turn_node       (append assistant reply + mode to transcript)
END
```

The crisis gate uses `Command(goto=...)` to route between branches.
Both branches converge at the shared extraction + finalize spine
before END.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Execution model | LangGraph `StateGraph` | Explicit branches (crisis vs. therapeutic), deterministic node ordering, checkpoint persistence |
| LLM abstraction | `BaseLLMClient` protocol | Provider-agnostic; Gemini and OpenAI clients satisfy the same interface |
| Embedding abstraction | `EmbeddingProvider` protocol | Same pattern; Gemini `text-embedding-004` is the default, `NullEmbeddingProvider` for offline/tests |
| Storage | SQLite via `aiosqlite` | Single-file portability, no external server, survives CLI restart |
| Memory store | `MemoryStore` protocol | In-memory for incognito/tests, SQLite for persistent mode; callers don't know which |
| Retrieval | Hybrid RRF (embedding + token-recall) | Embeddings close the stemming/synonym gap; token-recall preserves proper-noun precision; RRF fuses both with zero tuning |
| Prompt content | `knowledge/` markdown files | Reviewed content lives outside code; prompt builders compose at runtime |
| Crisis log | Always-on regardless of memory mode | Privacy asymmetry: incognito means "no memory of you" but crisis events are still auditable |

## Persistence

In persistent mode, all durable state is stored in three SQLite
databases. SQLite was chosen for single-file portability — each
database is a regular file the user can copy, back up, inspect with
`sqlite3`, or delete. No external server process is required.

| Database | Owner | What it persists |
|---|---|---|
| Thread checkpoints | LangGraph checkpointer | Conversation state snapshots (transcript, routing, progress) |
| Memory store | `SqliteMemoryStore` | Semantic facts, episodic arcs, procedural profiles, embedding vectors |
| Crisis log | `SqliteCrisisLogBackend` | Crisis event audit trail with per-day indexing |

The default file paths are relative to the working directory, but
all three are overridable via CLI flags (`--sqlite-path`,
`--memory-sqlite-path`, `--crisis-log-sqlite-path`) so users can
store them wherever makes sense for their setup.

In incognito mode, all three layers use in-memory backends instead
— nothing touches disk.

## Feature summary

### Crisis safety

- Hybrid regex + LLM crisis classifier running on every turn
- Three deterministic fast paths: imminent-risk override, clear
  self-harm patterns, idiomatic-safe override
- LLM fallback for ambiguous cases with sharp level boundaries
- Always-on crisis audit log (writes regardless of memory mode)
- 90-day retention purge via `/memory purge-crisis`

### Therapeutic response

- Six response modes: supportive, reflective, clarifying,
  psychoeducation, guided_exercise, closing
- Hybrid mode dispatcher: high-precision regex fast paths for
  obvious cases, LLM classifier for the ambiguous middle,
  regex fallback when no LLM is available
- Multi-turn guided exercise state tracking
- Closing detection with false-positive sensitivity

### Memory

- Three CoALA-inspired layers: semantic facts, episodic session
  arcs, procedural style rules
- Hybrid RRF retrieval (token-recall + embedding cosine)
- Embeddings via Gemini `text-embedding-004` with graceful
  token-recall fallback
- Hot-path dedup at write time via token-set Jaccard similarity
- Pre-extractor small-talk gate
- First-turn episodic catch-up
- Proactive recall toggle

### Privacy controls

- Per-record deletion, namespace-wide wipe, crisis log retention purge
- All destructive ops scoped to `session.owner_id()`
- Incognito mode: zero writes to disk

### Observability

- Per-turn diagnostics with stage timings and write counts
- Stage Timings panel, `/debug state`, `/history` with mode column
- Per-node streaming via multi-mode `astream`

### Eval harnesses

- Five assertion-based eval runners (no LLM-as-judge):
  dispatcher, extractor, crisis classifier, summarizer, retrieval
- Hand-curated datasets with regression-pin notes per case

## Prompt architecture

The backend does not treat prompts as one giant string. It uses a
layered composition model:

1. **Core** -- `knowledge/soul.md` + `knowledge/policy/` define the
   agent's identity and safety stance
2. **Mode** -- `knowledge/response_modes/*.md` provide per-mode
   knowledge (one file per therapeutic mode)
3. **Instructions** -- mode-specific instruction blocks in
   `agent/therapeutic/prompts.py` shape the response register
4. **State context** -- the prompt builder injects working memory,
   procedural rules, semantic signals, and session progress
5. **Turn posture** -- response guidance and modality overlays
   fine-tune the specific turn

See the [Prompt Assembly](/docs/agent/prompt-assembly) page for details.

## Package layout

| Package | What it owns |
|---|---|
| `agent/` | Graph entrypoint, state schema, models, runtime context |
| `agent/nodes/` | One file per graph node |
| `agent/memory/` | Store protocol, embeddings, retrieval, dedup, prompts, small-talk gate |
| `agent/therapeutic/` | Therapeutic subgraph: dispatcher, six mode nodes, prompt builders |
| `agent/prompts/` | Crisis prompts |
| `services/llm/` | `BaseLLMClient` protocol + Gemini and OpenAI implementations |
| `opencouch_cli/` | Rich-based interactive CLI |
| `core/` | Runtime config and provider selection |
| `tests/` | 476+ pytest tests |

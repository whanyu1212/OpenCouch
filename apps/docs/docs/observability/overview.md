---
title: Tracing & Evals
sidebar_position: 1
---

# Eval-driven development

OpenCouch uses two complementary surfaces for eval-driven development:

1. **LangSmith** for external trace-level observability, run search, and evaluation review.
2. **CLI inspection commands + live status** for local visibility into execution, state, and memory.

The CLI intentionally keeps the normal chat loop lightweight now:
the reply renders as soon as the response is ready, a live spinner
shows node progress while post-response work finishes, and deeper
inspection is available on demand through commands like `/status`,
`/context`, `/memory status`, and `/debug state`.

---

## How diagnostics flow

Each node writes its own keys into `state["diagnostics"]` via the
`_merge_dicts` reducer. Nodes return only their own keys — the
reducer handles merging automatically. No manual dict spreading.

```text
crisis_gate                load_memory
  · crisis_gate_ms           · load_memory_ms
  · crisis_level             · semantic_hits / episodic_hits
  · classifier_path          · retrieval_path
         │                          │
         └──────────┬───────────────┘
                    ▼
             finalize_turn
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    extract_facts       extract_procedural     ← parallel fan-out
    · extract_facts_ms  · extract_procedural_ms
    · semantic_writes   · procedural_writes
    · extract_facts_    · extract_procedural_
      reason              reason
          └─────────┬─────────┘
                    ▼
            AgentOutput.diagnostics
              + turn_total_ms (stamped by runtime)
```

:::info Parallel extractors, merged diagnostics
Both extractors write simultaneously after finalize. Because
`diagnostics` uses a `_merge_dicts` reducer, their keys merge
without racing — no node needs to know what other nodes wrote.
:::

---

## Observability & evaluation

For text runs, LangSmith captures the LangGraph execution trace, including the top-level run plus child spans for graph nodes and subgraphs. In OpenCouch, LangSmith is the primary external surface for:

- inspecting graph execution paths
- filtering runs by thread and runtime metadata
- reviewing failures from local eval harnesses
- comparing behavior across prompt, model, or routing changes

OpenCouch also attaches runtime metadata such as `thread_id`, `channel`, `memory_mode`, `streaming`, and `user_scope` to text runs to make traces easier to search.

LangSmith complements the local CLI inspection surfaces below; it does not replace the project's deterministic eval runners or local debugging commands.

---

## CLI surfaces

### 1. Assistant Reply

```text
╭──────────── Support Reply ─────────────╮
│ It sounds like something's on your     │
│ mind. What's most present right now?   │
╰────────────────────────────────────────╯
```

Green border for therapeutic, red for crisis.

### 2. Live execution status

`run_turn_stream` emits one `StatusEvent` per node via LangGraph's
multi-mode streaming. The CLI renders a progress spinner while the
graph is still running:

```text
  ⠋ crisis_gate → load_memory → therapeutic → finalize
    → extract_facts + extract_procedural  ← parallel, order varies
```

The stream now also has a non-terminal `response_ready` event. That
means the CLI can render the finished reply as soon as
`finalize_turn_node` seals it, while the post-response memory tail
continues in the background. The next user turn still waits for that
tail before it is processed, so turn ordering and memory consistency
stay intact.

### 3. On-demand inspection commands

| Command | What it shows |
|---|---|
| `/status` | Thread id, mode, turn count, response tier, and active response LLM |
| `/history [n]` | Recent transcript with `mode` column per assistant turn |
| `/context` | Structured session context snapshot, including working memory and procedural rules |
| `/memory status` | Owner-scoped semantic / episodic / procedural counts, recall toggle, and store totals |
| `/debug state` | Raw graph state as pretty-printed JSON |

The old auto-rendered Turn Diagnostics, Stage Timings, and Session
Context panels are no longer part of the default chat loop. Their
underlying diagnostics still exist in graph state and traces; the CLI
just no longer prints those panels automatically after every turn.

---

## Live streaming

Stage labels are defined in `agent/models.py` as `STAGE_LABELS` and
shared between the CLI and WebSocket API so all clients display
consistent text:

| Internal stage | Friendly label |
|---|---|
| `crisis_gate` | safety check |
| `memory_control_gate` | checking memory command |
| `memory_control` | updating memory |
| `grounded_lookup_gate` | checking factual lookup |
| `grounded_lookup` | looking up factual answer |
| `crisis_resource_lookup` | looking up crisis resources |
| `crisis_response` | generating crisis reply |
| `crisis_log` | writing crisis log |
| `load_memory` | loading memory |
| `memory_profile_load` | loading profile memory |
| `memory_graph_load` | querying graph memory |
| `memory_profile_save` | saving profile memory |
| `memory_graph_save` | writing graph memory |
| `therapeutic` | generating therapeutic reply |
| `extract_facts` | extracting facts |
| `extract_procedural` | extracting style rules |
| `finalize` | finalizing turn |
| `session_stage` | reading context |
| `response_generation` | generating |

Unknown stages fall through to their raw name so future additions
render without a mapping update.

---

## Diagnostics keys reference

| Key | Node | Value |
|---|---|---|
| `crisis_gate_ms` | crisis_gate | Assessment wall-clock time |
| `crisis_classifier_path` | crisis_gate | `override` / `llm_primary` / `llm_fallback` / `deterministic` |
| `crisis_level` | crisis_gate | Normalized level (0–3) |
| `crisis_resource_lookup_ms` | crisis_resource_lookup | Resource lookup wall-clock time (crisis branch only) |
| `resource_lookup_status` | crisis_resource_lookup | `found` / `no_location` / `search_failed` / `no_verified_results` / `not_attempted` |
| `memory_control_gate_ms` | memory_control_gate | Gate evaluation wall-clock time |
| `memory_control_action` | memory_control_gate | Detected command kind (or empty when none) |
| `grounded_lookup_gate_ms` | grounded_lookup_gate | Gate evaluation wall-clock time |
| `grounded_lookup_status` | grounded_answer | `answered` / `no_verified_answer` / `search_failed` / `search_unavailable` / `not_attempted` |
| `load_memory_ms` | load_memory | Retrieval wall-clock time |
| `semantic_hits` | load_memory | Semantic entries retrieved |
| `semantic_store_size` | load_memory | Total semantic records in store |
| `episodic_hits` | load_memory | Episodic entries retrieved |
| `episodic_store_size` | load_memory | Total episodic records in store |
| `procedural_count` | load_memory | Rules loaded from profile |
| `proactive_recall` | load_memory | Recall toggle state |
| `retrieval_path` | load_memory | `hybrid_rrf` / `token_recall` / `token_recall_after_embed_error` |
| `extract_facts_ms` | extract_facts | Extraction wall-clock time |
| `semantic_writes` | extract_facts | Immediate semantic writes that actually committed on this turn |
| `semantic_candidates` | extract_facts | Total candidates returned by the LLM |
| `semantic_commit_now_candidates` | extract_facts | Candidates the policy classified as commit-now |
| `semantic_session_end_holds` | extract_facts | Semantic candidates held for session-end review |
| `semantic_repeat_required` | extract_facts | Semantic candidates blocked pending stronger repetition evidence |
| `semantic_policy_drops` | extract_facts | Semantic candidates dropped by deterministic write policy |
| `semantic_bumps` | extract_facts | Existing facts bumped (dedup match) |
| `extract_facts_reason` | extract_facts | Skip reason or extraction outcome |
| `extract_procedural_ms` | extract_procedural | Extraction wall-clock time |
| `procedural_writes` | extract_procedural | Immediate procedural rules written |
| `procedural_candidates` | extract_procedural | Total candidates returned by the LLM |
| `procedural_commit_now_candidates` | extract_procedural | Candidates the policy classified as commit-now |
| `procedural_session_end_holds` | extract_procedural | Procedural candidates buffered for session-end promotion |
| `procedural_policy_drops` | extract_procedural | Procedural candidates dropped by deterministic write policy |
| `extract_procedural_reason` | extract_procedural | Skip reason or extraction outcome |
| `turn_total_ms` | runtime | Total turn wall-clock (stamped outside the graph) |

---

## Adding diagnostics to a new node

```python
import time

start = time.monotonic()

# ... node logic ...

return {
    "diagnostics": {
        "my_node_ms": round((time.monotonic() - start) * 1000, 2),
        "my_writes": write_count,
    }
}
```

:::tip No spreading needed
The `diagnostics` field uses a `_merge_dicts` reducer — return only
your own keys and the reducer handles merging with other nodes'
diagnostics automatically. Never `**state.get("diagnostics", {})`.
:::

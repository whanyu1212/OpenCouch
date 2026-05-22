---
title: Tracing
sidebar_position: 1
---

# Trace-driven development

OpenCouch uses two complementary surfaces for trace-driven development:

1. **Opik** for primary external trace-level observability and run search.
2. **CLI inspection commands + live status** for local visibility into execution, state, and memory.

LangSmith remains supported as an optional secondary LangChain tracing
integration, but Opik is the default external surface to use when
reviewing graph behavior.

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
              response ready
                    │
            AgentOutput.diagnostics
              + turn_total_ms (stamped by runtime)
```

:::info Runtime diagnostics
Runtime stages and side-effect services use the same structured diagnostics channel, so
turn-level timings and retrieval counters land in one `AgentOutput`.
:::

---

## Observability

For text runs, Opik captures the runtime execution trace, including
the top-level run plus child spans for runtime stages and SDK calls. In
OpenCouch, Opik is the primary external surface for:

- inspecting runtime execution paths
- filtering runs by thread and runtime metadata
- reviewing failures from tests and manual trace runs
- comparing behavior across prompt, model, or routing changes

OpenCouch also attaches runtime metadata such as `thread_id`, `channel`, `memory_mode`, `streaming`, and `user_scope` to text runs to make traces easier to search.

Opik complements the local CLI inspection surfaces below; it does not replace
backend tests or local debugging commands.

Enable Opik by setting:

```env
OPIK_API_KEY=...
OPIK_WORKSPACE=...
OPIK_PROJECT_NAME=opencouch-dev
```

`OPIK_PROJECT_NAME` is optional; the graph wrapper defaults to
`opencouch-dev` when it is unset.

LangSmith can be enabled alongside Opik through the standard
`LANGSMITH_*` and `LANGCHAIN_*` environment variables when a secondary
trace backend is useful.

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
  ⠋ crisis_gate → turn_dispatch → load_memory → therapeutic → finalize
```

The stream now also has a non-terminal `response_ready` event. That
means the CLI can render the finished reply as soon as
turn finalization seals it.

### 3. On-demand inspection commands

| Command | What it shows |
|---|---|
| `/status` | Thread id, mode, turn count, response tier, and active response LLM |
| `/history [n]` | Recent transcript with `mode` column per assistant turn |
| `/context` | Structured session context snapshot, including working memory and procedural rules |
| `/memory status` | Owner-scoped semantic / episodic / procedural counts, recall toggle, and store totals |
| `/debug state` | Raw runtime state as pretty-printed JSON |

The old auto-rendered Turn Diagnostics, Stage Timings, and Session
Context panels are no longer part of the default chat loop. Their
underlying diagnostics still exist in runtime state and traces; the CLI
just no longer prints those panels automatically after every turn.

---

## Live streaming

Stage labels are defined in `agent/models.py` as `STAGE_LABELS` and
shared between the CLI and WebSocket API so all clients display
consistent text:

| Internal stage | Friendly label |
|---|---|
| `crisis_gate` | safety check |
| `turn_dispatch` | routing turn |
| `memory_control` | updating memory |
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
| `runtime_extraction` | extracting facts and style rules after response finalization |
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
| `crisis_classifier_path` | crisis_gate | `llm_primary` |
| `crisis_level` | crisis_gate | Normalized level (0–3) |
| `crisis_resource_lookup_ms` | crisis_resource_lookup | Resource lookup wall-clock time (crisis branch only) |
| `resource_lookup_status` | crisis_resource_lookup | `found` / `no_location` / `location_refused` / `no_verified_results` / `not_attempted` |
| `turn_dispatch_ms` | turn_dispatch | Safe-turn routing wall-clock time |
| `memory_control.action` | turn_dispatch | Detected command kind (or empty when none) |
| `grounded_lookup.status` | grounded_answer | `answered` / `no_verified_answer` / `not_attempted` |
| `load_memory_ms` | load_memory | Retrieval wall-clock time |
| `semantic_hits` | load_memory | Semantic entries retrieved |
| `semantic_store_size` | load_memory | Total semantic records in store |
| `episodic_hits` | load_memory | Episodic entries retrieved |
| `episodic_store_size` | load_memory | Total episodic records in store |
| `procedural_count` | load_memory | Rules loaded from profile |
| `proactive_recall` | load_memory | Recall toggle state |
| `retrieval_path` | load_memory | `hybrid_rrf` / `token_recall` / `token_recall_after_embed_error` |
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

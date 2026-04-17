---
title: Observability
sidebar_position: 1
---

# Observability

The CLI shows you **why** the agent did what it did, not just what
it said. Every turn produces diagnostics panels plus on-demand
debugging commands.

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

## CLI panels

### 1. Assistant Reply

```text
╭──────────── Support Reply ─────────────╮
│ It sounds like something's on your     │
│ mind. What's most present right now?   │
╰────────────────────────────────────────╯
```

Green border for therapeutic, red for crisis.

### 2. Turn Diagnostics

| Column | What it shows |
|---|---|
| mode | Which therapeutic mode shaped the reply |
| source | How it was selected (keyword, llm, default) |
| type | THERAPEUTIC, OPERATIONAL, or CRISIS |
| safety | normal, distress, check, or crisis |

### 3. Stage Timings

```text
  stage              time (ms)   writes   store Δ
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  crisis_gate        1500.62        -         -
  load_memory            1.75        -         -
  therapeutic            -           -         -
  finalize               0.08        -         -
  extract_facts      2525.98        1        +1
  extract_procedural 1066.69        0         0
  turn_total         5440.67        -         -
```

**writes** = what the extractor attempted.
**store Δ** = what actually landed after dedup.
When writes=1 but Δ=0, dedup caught a duplicate.

### 4. Session Context

```text
  turn_count        3
  working_memory    · Previously noted: my sister Sarah...
                    · Last session (anxiety): talked about...
  procedural_rules  · Don't suggest meditation
  proactive_recall  off
  exercise          grounding_5_4_3_2_1 (step 2)
```

Working memory renders structured `WorkingMemoryEntry` dicts
on demand via `format_working_memory_entries()`. Exercise row
only appears when a guided exercise is active.

---

## Debug commands

| Command | What it shows |
|---|---|
| `/debug state` | Raw graph state as pretty-printed JSON |
| `/context` | Session Context panel on demand |
| `/history [n]` | Recent transcript with `mode` column per assistant turn |
| `/memory status` | Per-namespace counts, recall toggle, feedback count, owner_id |
| `/status` | Thread id, mode, turn count, LLM client status |

---

## Live streaming

`run_turn_stream` emits one `StatusEvent` per node via LangGraph's
multi-mode streaming. The CLI renders a progress spinner:

```text
  ⠋ crisis_gate → load_memory → therapeutic → finalize
    → extract_facts + extract_procedural  ← parallel, order varies
```

Stage labels are mapped from internal node names:

| Node name | CLI label |
|---|---|
| `crisis_gate_node` | `crisis_gate` |
| `load_memory_node` | `load_memory` |
| `therapeutic_subgraph` | `therapeutic` |
| `finalize_turn_node` | `finalize` |
| `extract_semantic_facts_node` | `extract_facts` |
| `extract_procedural_rules_node` | `extract_procedural` |

Unknown nodes fall through to their raw name so future additions
render without a mapping update.

---

## Diagnostics keys reference

| Key | Node | Value |
|---|---|---|
| `crisis_gate_ms` | crisis_gate | Assessment wall-clock time |
| `crisis_classifier_path` | crisis_gate | `deterministic` / `llm_fallback` / `override` |
| `crisis_level` | crisis_gate | Normalized level (0–3) |
| `load_memory_ms` | load_memory | Retrieval wall-clock time |
| `semantic_hits` | load_memory | Semantic entries retrieved |
| `semantic_store_size` | load_memory | Total semantic records in store |
| `episodic_hits` | load_memory | Episodic entries retrieved |
| `episodic_store_size` | load_memory | Total episodic records in store |
| `procedural_count` | load_memory | Rules loaded from profile |
| `proactive_recall` | load_memory | Recall toggle state |
| `retrieval_path` | load_memory | `hybrid_rrf` / `token_recall` / `token_recall_after_embed_error` |
| `extract_facts_ms` | extract_facts | Extraction wall-clock time |
| `semantic_writes` | extract_facts | Facts the LLM produced |
| `semantic_bumps` | extract_facts | Existing facts bumped (dedup match) |
| `extract_facts_reason` | extract_facts | Skip reason or extraction outcome |
| `extract_procedural_ms` | extract_procedural | Extraction wall-clock time |
| `procedural_writes` | extract_procedural | Rules written |
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

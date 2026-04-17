---
title: Crisis Gate
sidebar_position: 2
---

# Crisis Gate

First node in the graph. Every message passes through it before
memory loads, before routing, before response generation. Cannot be
skipped.

---

## Classification flow

Four layers, each able to short-circuit:

| Layer | What it does | Short-circuits when |
|---|---|---|
| **1. Override** | Imminent-risk or idiomatic-safe patterns | Obvious boundary case detected |
| **2. Regex ladder** | Deterministic self-harm patterns | Clear match without ambiguity |
| **3. LLM classifier** | Structured output with sharp level boundaries | LLM client available, no fast-path match |
| **4. Normalization** | Final `CrisisAssessment` | Always runs — produces the route decision |

Most turns resolve at layer 1 or 2 without invoking the LLM.

---

## Fast paths

| Path | Condition | Result |
|---|---|---|
| **Imminent risk** | Plan/intent language for self-harm or suicide | Level 3 — crisis response |
| **Idiomatic safe** | Trigger words in safe phrases ("dying to try it") | Level 0 — pass through |
| **Clear self-harm** | Unambiguous self-harm language | Level 2 — crisis response |

---

## Route decision

| Route | Condition | Pipeline |
|---|---|---|
| `crisis` | `needs_crisis_response = true` | crisis_response → crisis_log → finalize |
| `therapeutic` | `needs_crisis_response = false` | load_memory → therapeutic_subgraph → finalize |

Expressed as `Command(goto=...)` — the only branching node in the
graph.

---

## Privacy asymmetry

Crisis log writes **regardless of memory mode**:

| In incognito | Behavior |
|---|---|
| `user_id_or_null` | `None` — no identity persisted |
| `session_id_opaque` | SHA-256 hash, no reverse mapping |
| Event recorded? | **Yes** — safety audit trail preserved |

Retention: 90 days. `/memory purge-crisis [days]` enforces the
window (exclusive boundary — cutoff date itself preserved).

---

## Diagnostics

| Key | Value |
|---|---|
| `crisis_gate_ms` | Wall-clock time for the full assessment |
| `crisis_classifier_path` | `deterministic` / `llm_fallback` / `override` |
| `crisis_level` | Normalized level (0–3) |

---

## Design rules

| Rule | Why |
|---|---|
| Response pipeline waits for the gate | Safety sequencing > latency |
| RetryPolicy(max_attempts=2) | Defense-in-depth for transient failures |
| 42-case eval dataset | Covers imminent risk, clear self-harm, idiomatic-safe, boundary cases |

## Key files

| File | Purpose |
|---|---|
| `agent/nodes/crisis_gate.py` | Override detection, deterministic ladder, LLM fallback, normalization |
| `agent/nodes/crisis_response.py` | PFA-overlay response + web-searched local resources |
| `agent/nodes/crisis_log.py` | Always-on audit record writer |
| `agent/memory/crisis_log.py` | Backend protocol + in-memory + null |
| `agent/memory/sqlite_crisis_log.py` | SQLite backend with 90-day retention |
| `eval/runners/crisis_gate_eval.py` | Deterministic eval runner |

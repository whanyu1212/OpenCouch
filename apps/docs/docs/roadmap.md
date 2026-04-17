---
title: Roadmap
sidebar_position: 99
---

# Roadmap

What's shipped, what's in progress, and what's planned.

---

## Shipped

| Feature | What landed |
|---|---|
| **Web Frontend** | Next.js chat UI with streaming, thread management, and memory inspection. Lives in `apps/web/`. |
| **API Layer** | FastAPI with REST (`POST /api/chat`) and WebSocket (`/api/chat/stream`) endpoints. Thread management, memory status, session end. Lives in `apps/backend/api/`. |
| **Voice Chat** | OpenAI Realtime API integration with crisis gate, memory extraction, and web-searched crisis resources. Lives in `apps/backend/voice/`. |
| **Session Feedback** | End-of-session thumbs rating captured at `/end`, `/exit`, and `POST /threads/{id}/end`. SQLite-backed, incognito-safe. *(Branch: `feat/session-feedback-phase-1`)* |

---

## In progress

| Feature | Status | What's left |
|---|---|---|
| **Session feedback — closing mode** | Designed, not wired | Add a regex fast path for obvious closings ("bye", "I'm done") so the feedback prompt can also fire during natural closing turns, not just CLI/API end commands |
| **Session feedback — voice** | Designed, not wired | Voice disconnect bypasses `end_session()` — needs to either route through the runtime or gain its own feedback hook |

---

## Planned

### Messaging Channels

Adapters for Telegram, WhatsApp, and Discord. The `Channel` enum
already has slots (`Channel.TELEGRAM`, `Channel.WHATSAPP`); the
agent graph is channel-agnostic. Each adapter maps platform message
formats to `AgentInput` / `AgentOutput`. Crisis responses would need
channel-specific formatting (inline buttons, embeds).

### Acoustic Crisis Detection

Voice mode currently uses transcript-only crisis detection. Real
gaps: voice cracking, sobbing, pressured speech, prosodic flatness.
A user saying "I'm fine" through tears scores level 0.

Requires either a curated distressed-voice dataset (ethically
fraught) or a validated off-the-shelf acoustic classifier (not a
solved problem). Calendar-gated on dataset and model maturity.

### Graph Memory

Graphiti + Neo4j for entity/relationship extraction from semantic
facts. Enables relational reasoning: "you mentioned your sister and
your work stress — they tend to co-occur." The wire frame exists
(`agent/memory/graph_store.py` with `NullGraphMemoryStore`); the
`graphiti-core` dependency is in `pyproject.toml` but the
integration is intentionally disabled pending design.

### Background Consolidation

Automatic fact merging, dormant marking, and a `consolidation_runs`
log. Schema is defined (`ConsolidationProposal`,
`ConsolidationRunRecord` in `agent/memory/models.py`); the
implementation node is sketched but not wired into the graph. Adds
`/memory restore` as an undo for destructive operations.

### Clinical Review

A trained clinician reviews the `knowledge/response_modes/*.md`
files, the prompt builders in `agent/therapeutic/prompts.py`, and
agent responses across dogfood sessions. This is the gate before
"a trusted friend could try it" becomes a defensible claim. Calendar
dependency, not engineering.

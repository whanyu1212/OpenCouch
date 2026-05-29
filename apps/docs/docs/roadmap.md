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
| **Web Frontend** | Next.js chat UI with streaming, persisted setup state, thread management, memory inspection, visible error fallbacks, and OpenAI Realtime voice entrypoint. Lives in `apps/web/`. |
| **API Layer** | FastAPI with REST (`POST /api/chat`) and WebSocket (`/api/chat/stream`) endpoints. Thread management, memory status, session end. Lives in `apps/backend/api/`. |
| **Voice Chat (OpenAI Realtime)** | Browser speech-to-speech over OpenAI Realtime WebRTC with app-owned tools, Realtime session policy, incognito/persistent modes, turn recording, and shared end-session finalization. Lives across `apps/web/src/components/realtime-voice-session-provider.tsx` and `apps/backend/agent/voice/`. |
| **Session Feedback** | End-of-session thumbs rating captured at `/end`, `/exit`, and `POST /api/threads/{id}/end`. Postgres-first durable backend with incognito-safe in-memory mode and legacy SQLite fallback. |
| **Crisis Gate — LLM-only** | Crisis classification is a structured LLM call with strict truth-table enforcement. Provider failures surface through retries/errors instead of silently degrading to regex rules. |
| **Routing — LLM-primary** | Crisis, therapeutic dispatch, grounded lookup, memory-control, guided-exercise selection, and memory write policy use LLM-owned classifiers with local validation and hard confirmation gates where needed. |
| **Knowledge Overhaul** | `core_identity.md` defines assistant role, product stance, voice, therapeutic grounding, cultural sensitivity, repair patterns, and boundary-setting voice. `boundaries.md` expands redirection patterns and dependency framing. |
| **OpenAI Embeddings** | `text-embedding-3-large` as the configured provider, with token-only retrieval when no API key is available. Hybrid RRF retrieval achieves 14/17 recall@5 vs 6/17 token-only. |

---

## In progress

| Feature | Status | What's left |
|---|---|---|
| **Response quality review** | Manual dogfood path | Needs broader review of ordinary support turns and longer dogfood transcripts. |
| **Memory integration regression coverage** | Pytest-first | Runtime and recall tests cover semantic, episodic, procedural, correction, deletion, and cross-feature behavior. Remaining work is wider live dogfood coverage and voice parity. |
| **Session feedback — closing mode** | Closing signal wired, feedback UX pending | Closing detection is LLM-primary and emits `session_action=suggest_end_session`; feedback prompt still needs to fire from natural closings, not just CLI/API end commands. |
| **Session feedback — voice** | Designed, not wired | Voice disconnect now routes through `end_session()` in persistent mode; the remaining work is collecting an explicit feedback value from the voice end-state UI. |

---

## Planned

### Messaging Channels

WhatsApp, Discord, and Telegram adapters. The agent graph is
channel-agnostic; each adapter would map platform message formats
to `AgentInput` / `AgentOutput`. Crisis responses would need
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
your work stress — they tend to co-occur." The `graphiti-core`
dependency is in `pyproject.toml`, but the integration is
intentionally disabled pending design.

### Background Consolidation

Automatic fact merging, dormant marking, and a `consolidation_runs`
log. Schema is defined (`ConsolidationProposal`,
`ConsolidationRunRecord` in `agent/memory/models.py`); the
implementation is planned but not wired into the graph. Adds
`/memory restore` as an undo for destructive operations.

### Session Intent, Stage, and Response Guidance

Three state fields (`progress.intent`, `progress.stage`,
`response.guidance`) are defined in the schema but not yet populated
by the runtime. When implemented, they enable session-level steering:
the agent knows whether to deepen, stabilize, or close based on
conversation arc rather than just the current message.

### Crisis Gate Production Telemetry

Model ID, prompt version, raw/normalized levels, confidence values,
timeout/parse failure counters, and degraded-mode alerts. The
production telemetry layer is not yet in place.

### Clinical Review

A trained clinician reviews the `agent/prompts/sources/response_styles/*.md`
files, the agent-owned prompt builders in `agent/specialists/`, and
agent responses across dogfood sessions. This is the gate before
"a trusted friend could try it" becomes a defensible claim. Calendar
dependency, not engineering.

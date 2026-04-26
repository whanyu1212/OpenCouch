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
| **Voice Chat (LiveKit)** | LiveKit-native worker with WebRTC room transport, `TherapeuticAgent` ↔ `CrisisAgent` handoffs, bounded `GroundingTask` (10 voice-allowlisted exercises), `@function_tool` declarations, and three-phase memory (startup load / mid-session retrieval / shutdown transcript replay). Lives in `apps/backend/voice/livekit/`. |
| **Session Feedback** | End-of-session thumbs rating captured at `/end`, `/exit`, and `POST /threads/{id}/end`. SQLite-backed, incognito-safe. |
| **Telegram DM Gateway** | Standalone local dogfood gateway for Telegram DMs. Uses `Channel.TELEGRAM`, the existing persistent text runtime, allowlisted numeric sender IDs, canonical owner ID memory, `/start`, `/help`, `/end`, plain-text replies, 20-minute runtime sessions, and an advisory lock for duplicate gateway prevention. |
| **Session Trajectory Eval** | Unified runner for short (inline) and long (checkpoint) trajectory datasets. 25 long-trajectory cases covering approach, boundary enforcement, crisis arcs, closing, venting, and response style transitions. Concurrent hybrid execution with `--concurrency`, `--case`, `--verbose`. |
| **Crisis Gate — LLM-primary** | LLM is the primary crisis classifier; regex is fallback only. Override precedence fix, prompt hardening (conversation fencing, anti-injection, adversarial examples), strict truth table enforcement. |
| **Dispatcher — LLM-primary** | LLM handles all response style + approach classification. Context-blind regex fast paths removed. LLM-based mid-exercise exit detection. Exercise approach persistence. |
| **Knowledge Overhaul** | `soul.md` expanded with therapeutic grounding, cultural sensitivity, repair patterns, boundary-setting voice. `identity.md` rewritten with product philosophy. `boundaries.md` expanded with redirection patterns and dependency framing. |
| **OpenAI Embeddings** | `text-embedding-3-large` as default provider, Gemini as fallback. Hybrid RRF retrieval achieves 14/17 recall@5 vs 6/17 token-only. |

---

## In progress

| Feature | Status | What's left |
|---|---|---|
| **Response quality rubric** | Designed, not implemented | LLM-as-judge eval runner to test empathy, tone, banned phrases, question stacking, conciseness. Needs rubric dataset + grading runner. |
| **Memory integration eval** | Designed, not implemented | Test whether retrieved memory shapes responses. Cross-session continuity, procedural rule enforcement, appropriate recall. |
| **Session feedback — closing mode** | Designed, not wired | Closing detection is now LLM-primary; feedback prompt needs to fire on natural closings, not just CLI/API end commands. |
| **Session feedback — voice** | Designed, not wired | Voice disconnect bypasses `end_session()` — needs to either route through the runtime or gain its own feedback hook. |

---

## Planned

### Messaging Channels

WhatsApp and Discord adapters. `Channel.WHATSAPP` already exists;
Discord would need an enum addition. The agent graph is channel-agnostic.
Each adapter maps platform message formats to `AgentInput` /
`AgentOutput`. Crisis responses would need channel-specific formatting
(inline buttons, embeds). Telegram groups, media, and richer formatting
remain future scope beyond the shipped DM text gateway.

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
by any node. When implemented, they enable session-level steering:
the agent knows whether to deepen, stabilize, or close based on
conversation arc rather than just the current message. The eval
runner already supports assertions for all three — just re-add the
dataset expectations.

### Crisis Gate Production Telemetry

Model ID, prompt version, raw/normalized levels, confidence values,
timeout/parse failure counters, and degraded-mode alerts. The
production telemetry layer is not yet in place.

### Clinical Review

A trained clinician reviews the `agent/prompts/sources/response_modes/*.md`
files, the prompt builders in `agent/therapeutic/prompts.py`, and
agent responses across dogfood sessions. This is the gate before
"a trusted friend could try it" becomes a defensible claim. Calendar
dependency, not engineering.

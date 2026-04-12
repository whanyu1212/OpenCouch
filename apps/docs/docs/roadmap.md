---
title: Roadmap
sidebar_position: 99
---

# Roadmap

What's planned but not yet shipped.

## Web Frontend

A proper web UI that replaces the CLI as the primary user-facing
surface. The backend already exposes the full agent pipeline through
`PersistentAgentRuntime` — the frontend would call it via a FastAPI
layer with WebSocket streaming for real-time response delivery.
Likely stack: React/Next.js with a clean chat interface, session
management, and memory inspection panels mirroring what the CLI's
`/memory list`, `/context`, and `/debug state` provide today.

## Voice Chat

LiveKit + OpenAI Realtime integration using **Pattern C**: Realtime
handles audio I/O and voice quality, the LangGraph agent runs
per-turn for safety and routing. The crisis gate runs on every
transcribed user turn before Realtime is allowed to respond —
transcript-only crisis detection ships first, acoustic
paralinguistic detection (voice cracking, sobbing, prosodic
flatness) follows in a later release.

See `notes/voice-chat-design.md` for the full Pattern A/B/C
analysis and the decision rationale.

## Messaging Channels

Adapters for messaging platforms so users can interact with the
agent where they already are:

- **Telegram** — bot API with webhook-based message handling
- **WhatsApp** — via the WhatsApp Business API or a provider like
  Twilio
- **Discord** — bot with slash commands and thread-based sessions

Each channel adapter would map platform-specific message formats to
the existing `AgentInput` / `AgentOutput` contract and the
`Channel` enum (`Channel.TELEGRAM`, `Channel.WHATSAPP`,
`Channel.DISCORD`). The `Channel` field already exists on
`AgentInput` — the agent graph is channel-agnostic by design, so
adding a new channel is a transport adapter, not a graph change.

Crisis responses would need channel-specific formatting (e.g.,
inline buttons for crisis hotline links on Telegram, embeds on
Discord).

## Graph Memory

Graphiti + Neo4j for entity/relationship extraction from semantic
facts. Enables relational reasoning: "you mentioned your sister and
your work stress — they tend to co-occur." Replaces flat semantic
fact retrieval with 1-hop graph expansion in `load_memory_node`.

## Clinical Review

A trained clinician reads the `knowledge/response_modes/*.md` files
and `agent/therapeutic/prompts.py`, reviews the agent's responses
across several dogfood sessions, and provides feedback on clinical
quality. This is the last gate before "a trusted friend could try
it" becomes a defensible claim. Requires a clinician's time, which
is a calendar dependency more than an engineering one.

## Background Consolidation

Automatic fact merging, dormant marking, and a
`consolidation_runs` log. Adds `/memory restore` as an undo for
destructive operations (`/memory forget`, `/memory clear`). The
consolidation pass runs on a looser dedup threshold (0.85 Jaccard)
than the hot-path dedup (0.95) because consolidation mistakes are
observable via the log and correctable, while hot-path mistakes are
not.

---
title: API Reference
sidebar_position: 2
---

# API Reference

The FastAPI app mounts all application routes under `/api`.

Run locally:

```bash
cd apps/backend
uv run uvicorn main:app --port 8000 --reload
```

## Text chat

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | `GET` | Health check |
| `/api/chat` | `POST` | Run one full text turn and return the completed response |
| `/api/chat/stream` | WebSocket | Run one text turn and stream status, chunks, and final response |

`POST /api/chat` and `/api/chat/stream` both accept a chat request
with `message`, `thread_id`, optional `user_id`, and optional
`response_model_tier`. Reuse the same `thread_id` to continue a
conversation. Reuse the same `user_id` across thread ids to share
long-term memory.

## Threads

| Route | Method | Purpose |
|---|---|---|
| `/api/threads` | `GET` | List known text threads |
| `/api/threads/{thread_id}/state` | `GET` | Return raw runtime state for a thread |
| `/api/threads/{thread_id}/history` | `GET` | Return user/assistant transcript turns |
| `/api/threads/{thread_id}/session-status` | `GET` | Return active-session tracking status |
| `/api/threads/{thread_id}/end` | `POST` | Finalize a text session and persist session-end memory |

## Memory

| Route | Method | Purpose |
|---|---|---|
| `/api/memory/status` | `GET` | Return memory counts, store totals, and recall state |
| `/api/memory/recall` | `PATCH` | Enable or disable proactive memory recall |
| `/api/memory/facts` | `GET` | List semantic facts |
| `/api/memory/sessions` | `GET` | List episodic session arcs |
| `/api/memory/rules` | `GET` | List procedural style rules |
| `/api/memory/facts/{index}` | `DELETE` | Delete one semantic fact by displayed index |
| `/api/memory/sessions/{index}` | `DELETE` | Delete one episodic arc by displayed index |
| `/api/memory/rules/{index}` | `DELETE` | Delete one procedural rule by displayed index |

## Voice

| Route | Method | Purpose |
|---|---|---|
| `/api/voice/livekit/token` | `POST` | Create a LiveKit participant token and optional agent dispatch config |
| `/api/voice/livekit/finalization-status/{thread_id}` | `GET` | Poll disconnect-time memory finalization status |
| `/api/voice/livekit/test` | `GET` | Serve the standalone LiveKit browser test page |

Voice is LiveKit-native. The worker lives under `agent/voice/`; the
FastAPI routes only mint room tokens, expose a standalone LiveKit test
page, and report transcript-finalization status after disconnect.

## Client contracts

The response schema exposes the user-visible text plus routing
metadata:

| Field | Meaning |
|---|---|
| `response_text` | Assistant message |
| `response_type` | Public category: `therapeutic` or `crisis` |
| `response_style` | More specific style or operational branch, such as supportive, memory_control, grounded_lookup, or crisis_response |
| `therapeutic_approach` | Therapeutic approach overlay when applicable |
| `crisis` | Normalized crisis assessment |
| `diagnostics` | Per-turn timings and routing metadata |

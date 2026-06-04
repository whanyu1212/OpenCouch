---
title: Web UI
sidebar_position: 3
---

import webChatScreenshot from '@site/static/img/chat.png';

# Web UI

The web app in `apps/web` is the main browser surface for OpenCouch.
It uses Next.js 16, React 19, Zustand for client session state, and
FastAPI for the backend contract.

<img className="docs-screenshot" src={webChatScreenshot} alt="OpenCouch web chat" />

## Local development

For day-to-day web work, run the Compose backend stack and the frontend dev server separately:

```bash
# Terminal 1: Postgres + FastAPI backend on http://localhost:8080/api
./scripts/dev_api_stack.sh

# Terminal 2: Next.js frontend with hot reload
cd apps/web
NEXT_PUBLIC_API_URL=http://localhost:8080/api pnpm dev
```

Open `http://localhost:3000`.

The web service is also available in Compose for production-built smoke
tests:

```bash
./scripts/dev_full_stack.sh
# or: docker compose --profile web up
```

If you run the backend manually on port `8000`, the frontend default
`NEXT_PUBLIC_API_URL=http://localhost:8000/api` works without extra
configuration.

## Runtime shape

| Area | Current implementation |
|---|---|
| Text chat | WebSocket streaming through `/api/chat/stream`, with REST `/api/chat` available for synchronous turns |
| Session state | Zustand store persisted to local storage for setup choices such as user id, thread id, memory mode, model tier, and assistant voice |
| Thread history | REST calls to `/api/threads`, `/api/threads/{thread_id}/history`, and `/api/threads/{thread_id}/session-status` |
| Memory controls | REST calls under `/api/memory/*` for status, recall toggle, list, and deletes |
| Voice | OpenAI Realtime WebRTC through `/api/voice/realtime/*`, with app-owned tool calls, turn recording, and session finalization |
| Debug state | `/state` calls `/api/threads/{thread_id}/state` for raw developer inspection only |
| Error handling | Route `error.tsx`, `global-error.tsx`, loading fallback, not-found fallback, structured API errors, and visible REST error notices |

## Streaming lifecycle

Each chat turn opens one WebSocket connection, sends a single
`ChatRequest`, renders status/chunk/done messages, and then closes
the socket. The client tracks the active socket in a ref, closes it
on unmount or thread change, and ignores stale stream events when a
newer turn has started.

The stream protocol is:

```json
{"type": "status", "stage": "loading memory", "detail": ""}
{"type": "chunk", "text": "That sounds heavy."}
{"type": "done", "response": {"response_text": "..."}}
{"type": "error", "code": "agent_turn_failed", "message": "The turn could not be completed."}
```

Malformed stream frames are handled as protocol errors so the UI can
surface a retryable failure instead of crashing the page. Backend
`error` events are terminal and are surfaced as chat notices.

## Debug boundary

The `/state` route is a developer State Inspector. It displays raw
runtime state from `/api/threads/{thread_id}/state`, including transcript,
safety, memory, routing, and diagnostics fields. This helps local
dogfooding and debugging, but it is not the product API surface. Normal
UI flows should consume typed chat, history, session-status, memory, and
voice endpoints.

## Voice boundary

The Realtime voice connection needs a client-side provider because the
browser owns WebRTC audio and data-channel events. The app lazy-loads
the provider and mounts it when the user is on `/voice`, while a voice
session is connected, or while voice finalization is still in progress.
The provider keeps OpenAI audio transport out of the normal text-chat
bundle while keeping transcripts, tool activity, and memory finalization
in the shared Zustand session store.

The production voice route is `/voice`. A lower-level dogfood route,
`/voice/realtime-dev`, uses the same `connectRealtimeVoiceSession(...)`
client helper but exposes raw and parsed Realtime events for debugging.

The provider is intentionally separate from the text streaming path:

| Text chat | Voice |
|---|---|
| Opens `/api/chat/stream` WebSocket for one user turn. | Opens WebRTC directly to OpenAI Realtime with an ephemeral client secret. |
| Backend runs `run_turn_stream(...)`. | Backend creates config, executes tools, records finalized turns, and finalizes sessions. |
| Streaming status comes from runtime stages. | UI status comes from Realtime connection state, transcript events, tool activity, and finalization status. |

## Verification

Run these checks from `apps/web`:

```bash
pnpm lint
pnpm build
```

The repository CI runs both commands for `apps/web` so frontend
regressions fail before merge.

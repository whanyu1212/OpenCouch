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

Run the backend and frontend in separate terminals:

```bash
# Terminal 1: FastAPI backend
cd apps/backend
.venv/bin/python -m uvicorn main:app --port 8000 --reload

# Terminal 2: Next.js frontend
cd apps/web
pnpm dev
```

Open `http://localhost:3000`. By default the frontend talks to
`http://localhost:8000`. Override this with `NEXT_PUBLIC_API_URL`
in `apps/web/.env.local` when the API lives elsewhere:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Runtime shape

| Area | Current implementation |
|---|---|
| Text chat | WebSocket streaming through `/api/chat/stream`, with REST `/api/chat` available for synchronous turns |
| Session state | Zustand store persisted to local storage for setup choices such as user id, thread id, memory mode, model tier, and assistant voice |
| Thread history | REST calls to `/api/threads`, `/api/threads/{thread_id}/history`, and `/api/threads/{thread_id}/state` |
| Memory controls | REST calls under `/api/memory/*` for status, recall toggle, list, and deletes |
| Voice | OpenAI Realtime WebRTC through `/api/voice/realtime/*`, with app-owned tool calls, turn recording, and session finalization |
| Error handling | Route `error.tsx`, `global-error.tsx`, loading fallback, not-found fallback, and visible REST error notices |

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
```

Malformed stream frames are handled as protocol errors so the UI can
surface a retryable failure instead of crashing the page.

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

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
uv run uvicorn main:app --port 8000 --reload

# Terminal 2: Next.js frontend, from the repo root
pnpm install
pnpm --dir apps/web dev
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
| Session state | Zustand store persisted to local storage for setup choices such as user id, thread id, memory mode, model tier, and voice language |
| Thread history | REST calls to `/api/threads`, `/api/threads/{thread_id}/history`, and `/api/threads/{thread_id}/state` |
| Memory controls | REST calls under `/api/memory/*` for status, recall toggle, list, and deletes |
| Voice | LiveKit browser room via `/api/voice/livekit/token`, with finalization polling after disconnect |
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

LiveKit React hooks need a client-side provider. The app lazy-loads
the voice provider and mounts it when the user is on `/voice`, while
a voice session is connected, or while voice finalization is still in
progress. That keeps LiveKit out of the normal text-chat bundle while
still supporting direct refreshes of the voice route.

## Verification

Run these checks from the repo root:

```bash
pnpm --dir apps/web lint
pnpm --dir apps/web build
```

The repository CI runs both commands for `apps/web` so frontend
regressions fail before merge.

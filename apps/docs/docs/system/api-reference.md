---
title: API Reference
sidebar_position: 2
---

# API Reference

The FastAPI app mounts all application routes under `/api`.

Run locally:

```bash
cd apps/backend
.venv/bin/python -m uvicorn main:app --port 8000 --reload
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
| `/api/threads/{thread_id}/state` | `GET` | **Debug/internal.** Return raw implementation state for a thread |
| `/api/threads/{thread_id}/history` | `GET` | Return user/assistant transcript turns |
| `/api/threads/{thread_id}/session-status` | `GET` | Return active-session tracking status |
| `/api/threads/{thread_id}/end` | `POST` | Finalize a text session and persist session-end memory |
| `/api/threads/{thread_id}/feedback` | `POST` | Record post-session feedback without re-finalizing the session (body: `feedback`, optional `memory_mode`, `modality: text\|voice`) |

:::caution Debug state endpoint
`/api/threads/{thread_id}/state` powers the local State Inspector and mirrors the TUI's `/debug state` command. It returns raw runtime implementation state, including transcript, memory, safety, routing, and diagnostics fields. It is useful for development and dogfooding, but product clients should use typed endpoints such as `/history`, `/session-status`, `/memory/*`, and `/chat/stream`.
:::

Text session finalization returns the same stable envelope shape as voice finalization:

```json
{
  "finalized": true,
  "summary": "Session summary text",
  "detail": "Session finalized.",
  "themes": ["stress", "sleep"],
  "mood_opened": "tense",
  "mood_closed": "calmer",
  "turn_count": 4,
  "open_loops": [],
  "resolved_threads": []
}
```

When no durable summary is produced, `finalized` is `false`, `summary` is `null`, list fields are empty, and `detail` explains why.

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

Memory endpoints are scoped by `thread_id`, optional `user_id`, and optional `memory_mode`. In incognito mode, user-memory reads return empty counts/lists for semantic facts, episodic sessions, and procedural rules. Saved-memory mutation endpoints reject with `409` and a structured detail payload:

```json
{
  "detail": {
    "code": "incognito_memory_mutation_unavailable",
    "message": "Saved-memory controls are unavailable in incognito mode."
  }
}
```

Audit-oriented counts such as crisis logs and session feedback may still be non-zero because those stores are always-on and privacy-scrubbed in incognito mode.

## Voice

| Route | Method | Purpose |
|---|---|---|
| `/api/voice/realtime/session` | `POST` | Create an OpenAI Realtime client secret for browser WebRTC voice |
| `/api/voice/realtime/tools` | `POST` | Execute one app-owned Realtime function tool call |
| `/api/voice/realtime/turn` | `POST` | Persist a finalized voice user/assistant turn in app-owned history |
| `/api/voice/realtime/end` | `POST` | Finalize a persistent voice session through the runtime session finalizer |

Voice is OpenAI Realtime-native. The browser owns WebRTC audio, while
the backend owns session configuration, memory bootstrap, function-tool
execution, route/style inference during turn persistence, and
end-session memory finalization.

Session creation request:

```json
{
  "thread_id": "web-voice-abc123",
  "user_id": "alice",
  "memory_mode": "persistent",
  "assistant_voice": "marin"
}
```

Tool execution request:

```json
{
  "thread_id": "web-voice-abc123",
  "user_id": "alice",
  "memory_mode": "persistent",
  "current_user_message": "Can you look up the official guidance?",
  "transcript": [
    {"role": "user", "content": "Can you look up the official guidance?"}
  ],
  "tool_name": "answer_grounded_lookup",
  "arguments": {"query": "official guidance"}
}
```

Turn recording request:

```json
{
  "thread_id": "web-voice-abc123",
  "user_id": "alice",
  "memory_mode": "persistent",
  "user_text": "Can you look up the official guidance?",
  "assistant_text": "I found the official source...",
  "tool_calls": [
    {
      "tool_name": "answer_grounded_lookup",
      "status": "completed",
      "output": {"response_text": "I found the official source..."}
    }
  ]
}
```

The backend infers the recorded `route` and `response_style` from the
tool calls that occurred during the Realtime turn.

End-session request:

```json
{
  "thread_id": "web-voice-abc123",
  "memory_mode": "persistent",
  "feedback": "positive"
}
```

Both `/api/threads/{thread_id}/end` and `/api/voice/realtime/end` accept an
optional `feedback` label (`positive` / `negative` / `skip`) that is written
to the session-feedback store before summarization. Omit it (or send `null`)
to skip the feedback step.

Voice end-session responses use the same `finalized`, `summary`, `detail`, and session-arc envelope documented for text sessions.

## Client contracts

The text chat response schema exposes the user-visible text plus routing
metadata:

| Field | Meaning |
|---|---|
| `response_text` | Assistant message |
| `response_type` | Public category: `therapeutic` or `crisis` |
| `response_style` | More specific style or operational branch, such as supportive, memory_control, grounded_lookup, or crisis_response |
| `therapeutic_approach` | Therapeutic approach overlay when applicable |
| `crisis` | Normalized crisis assessment |
| `session_action` | UI hint: `suggest_end_session` when the assistant produced a closing reply, otherwise `none` |
| `diagnostics` | Per-turn timings and routing metadata |

The WebSocket stream emits status, chunk, done, and terminal error events:

```json
{"type": "status", "stage": "loading memory", "detail": ""}
{"type": "chunk", "text": "That sounds heavy."}
{"type": "done", "response": {"response_text": "..."}}
{"type": "error", "code": "agent_turn_failed", "message": "The turn could not be completed."}
```

Frontend clients should treat `error` as terminal and display `message` to the user.

---
title: Session Feedback
sidebar_position: 2
---

# Session Feedback

An explicit end-of-session feedback collector that captures a thumbs
rating when the user finishes a session. The data is stored in a
dedicated persistence backend — Postgres for the recommended durable path,
in-memory for incognito, and SQLite only as legacy compatibility pending
removal — always-on, session-opaque, incognito-safe.

---

## What's collected

A single `SessionFeedbackRecord` per end-session event, containing:

| Field | Value | Source |
|---|---|---|
| `id` | Fresh uuid4 | Server-generated |
| `session_id_opaque` | SHA-256 of thread_id | Server-derived, never the raw id |
| `user_id_or_null` | `state.user_id` (LOCAL/SYNCED) or `None` (incognito) | Server-derived, scrubbed in incognito |
| `recorded_at` | ISO-8601 with `Z` suffix | `iso_now()` at write time |
| `label` | `"positive"` / `"negative"` / `"skip"` | User's explicit choice |
| `turn_count_at_end` | `session_progress.turn_count` from the latest runtime state snapshot | Read from runtime state |
| `source` | `"cli_end"` / `"cli_exit"` / `"api_end"` | Which end-session surface captured it |
| `schema_version` | `1` | Fixed for Phase 1 |

**No user text is stored.** The record captures only the
classification label and structural metadata. This is a deliberate
privacy boundary — mirroring the crisis log's
"metadata-only, no content" contract.

## Where it lives

| Memory mode | Backend | Persistence |
|---|---|---|
| **LOCAL / SYNCED** | `PostgresSessionFeedbackBackend` for the supported durable path; `SqliteSessionFeedbackBackend` exists only for legacy compatibility pending removal | Survives CLI / server restarts when backed by Postgres |
| **INCOGNITO** | `InMemorySessionFeedbackBackend` | Dies at process exit |

For the recommended local Docker setup and future managed deployment,
feedback lives in the shared Postgres persistence layer. The
`.store/session_feedback.sqlite3` file remains only for legacy SQLite
compatibility until the backend is removed.

Default retention: **180 days** (wider than crisis log's 90 because
feedback analytics benefit from a longer lookback). Enforced via
`apurge_before` — an operator / scheduled-job concern, not touched
by normal write paths.

## Collection surfaces

### CLI `/end`

The `/end` command triggers the full end-session flow:

1. **Feedback prompt** — `"Quick check — did today feel helpful? [y/n/s]
   (or Enter to skip without recording)"`
2. **Feedback write** — `runtime.record_session_feedback(thread_id,
   label=..., source="cli_end")`
3. **Summarization** — `runtime.end_session(thread_id, ...)`
4. **Farewell**

Pressing Enter, Ctrl-C, or EOF at step 1 produces no record — the
flow skips to step 3 without writing feedback. Explicit `y`/`n`/`s`
maps to `positive`/`negative`/`skip`.

### CLI `/exit` (save=y branch)

Same flow as `/end` but with `source="cli_exit"`. The `/exit`
command first prompts to save a summary (`[Y/n]`). If the user says
yes (or presses Enter — default Y), the feedback + summarization
flow runs.

**`/exit` with save=n** intentionally skips both feedback and
summary. The user said "don't save my conversation" — asking for a
rating on that path would be inconsistent.

### HTTP `POST /api/threads/{id}/end`

The endpoint accepts an optional body:

```json
{"feedback": "positive"}
```

Valid labels: `"positive"`, `"negative"`, `"skip"`. Invalid values
are rejected with HTTP 422 by Pydantic before any code runs.

When `feedback` is `null` or the body is absent, no feedback record
is created — summarization runs unchanged. The response shape is
unaffected by the feedback write (no status surfaced).

### Not covered (Phase 1)

- **Closing mode** — the therapeutic closing node does NOT emit a
  feedback prompt. Closing is a tonal farewell, not a session
  termination. If we later want in-conversation feedback hints, the
  same `record_session_feedback()` method works — the closing node
  would call the runtime directly.
- **Voice disconnect** — persistent OpenAI Realtime voice sessions
  now finalize through `PersistentAgentRuntime.end_session()`. Voice
  feedback still needs product UI to collect a rating before the
  `/api/voice/realtime/end` path runs.

## Inspection

### CLI

```
/memory status
```

The memory status panel now includes a `session feedback records`
row alongside the crisis log count.

### API

```
GET /api/memory/status?thread_id=...
```

Response includes `"session_feedback_count": <int>`.

### Programmatic

```python
count = await runtime.session_feedback_backend.arecord_count()
records = await runtime.session_feedback_backend.alist_by_session(session_id_opaque)
```

## Graceful degradation

| Failure | Behavior |
|---|---|
| Backend write error (database outage, disk full) | `record_session_feedback()` returns `None`, logs WARNING, caller continues to summarization |
| State lookup error (runtime state store failure) | Same — returns `None`, logs WARNING |
| Invalid label via HTTP | 422 before any code runs |
| CLI prompt interrupted (Ctrl-C, EOF) | No record, summarization proceeds |
| Incognito mode | Records written to in-memory backend, scrubbed of user_id, ephemeral |

Feedback writes **never block summarization or farewell**. A backend
outage means one session's feedback is lost — the end-session flow
itself is unaffected.

## Idempotency (Phase 1)

Phase 1 does not provide idempotency. Two calls to
`record_session_feedback()` with different `id` values (which is
what happens on every call — `id` is a fresh `uuid4()`) produce two
rows. The CLI prompt flow is single-shot and the HTTP endpoint is
single-request, so accidental duplicates require an explicit
double-click or double-POST.

If explicit idempotency becomes necessary, the fix is an
idempotency-key column on `SessionFeedbackRecord`, not a UNIQUE
constraint on the opaque `id`.

## Key files

| File | What it does |
|---|---|
| `agent/feedback/models.py` | `FeedbackLabel`, `FeedbackSource`, `SessionFeedbackRecord` |
| `agent/feedback/session_feedback.py` | `SessionFeedbackBackend` protocol + in-memory + null backends |
| `agent/feedback/postgres_session_feedback.py` | Primary durable Postgres feedback backend |
| `agent/feedback/sqlite_session_feedback.py` | Legacy SQLite backend with CHECK constraints and retention purge, pending removal |
| `agent/runtime/runtime.py` | `record_session_feedback()` method, backend selection, lifecycle |
| `api/models.py` | `EndSessionRequest.feedback`, `MemoryStatusResponse.session_feedback_count` |
| `api/routes/threads.py` | `POST /api/threads/{id}/end` body handling |
| `opencouch_tui/` | TUI session-end flow: prompts for a feedback label and renders the session summary on `/end` and exit |

# Session Feedback

This package owns explicit user quality signals, currently end-of-session
feedback collected by trusted CLI/API/UI surfaces. Feedback is a small
operator-facing rating signal, not therapeutic memory and not a safety audit log.

## Runtime Boundary

- It is not loaded into prompt memory.
- It is not written by an OpenAI agent tool.
- It is persisted through mode-aware feedback backends.
- Runtime lifecycle code owns when and how records are created.

## File Map

| File | Responsibility |
| --- | --- |
| `models.py` | `FeedbackLabel`, `FeedbackSource`, `FeedbackModality`, and `SessionFeedbackRecord`. |
| `session_feedback.py` | `SessionFeedbackBackend` plus in-memory and null implementations. |
| `postgres_session_feedback.py` | Direct Postgres implementation for durable feedback. |

## Collection Surfaces

Feedback is collected only from trusted app surfaces:

- CLI `/end` and `/exit` save flows call
  `PersistentAgentRuntime.record_session_feedback` directly.
- Text and voice `/end` routes still accept optional feedback for API
  compatibility.
- `POST /api/threads/{thread_id}/feedback` is the preferred UI path for
  post-session feedback. It lets text and voice submit the same rating shape
  after finalization without ending a session twice.

Explicit `skip` creates a record. Dismissing the UI or sending `null` creates no
record.

## Record Shape

`agent.runtime.session_feedback` owns the write policy:

- hash the thread id before persistence;
- read turn count and user id from runtime state;
- scrub user id in incognito mode;
- attach the trusted feedback source;
- attach the interaction modality;
- degrade gracefully when persistence fails.

The feedback backends here only store and retrieve records.

`source` answers which trusted surface captured the feedback:

- `cli_end`
- `cli_exit`
- `api_end`

`modality` answers which interaction channel the user is rating:

- `text`
- `voice`

Keeping these separate avoids overloading `source`; both text and voice web UI
feedback are API-originated, but they need distinct modality metadata for
operator review.

## Persistence Behavior

Incognito runtimes use `InMemorySessionFeedbackBackend`, so records exist only
for the runtime lifetime and always store `user_id_or_null=None`. Persistent
modes require Postgres for durable feedback.

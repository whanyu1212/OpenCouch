# Feedback Backends

This package contains explicit user quality signals, currently end-of-session
feedback collected by trusted CLI/API/UI surfaces.

Feedback is separate from both therapeutic memory and safety audit logs:

- It is not loaded into prompt memory.
- It is not written by an OpenAI agent tool.
- It is persisted through mode-aware feedback backends.
- Runtime lifecycle code owns when and how records are created.

## File Map

| File | Responsibility |
| --- | --- |
| `models.py` | `FeedbackLabel`, `FeedbackSource`, and `SessionFeedbackRecord`. |
| `session_feedback.py` | `SessionFeedbackBackend` plus in-memory and null implementations. |
| `sqlite_session_feedback.py` | SQLite implementation for local durable feedback. |
| `postgres_session_feedback.py` | Postgres implementation for production durable feedback. |

## Runtime Boundary

`agent.runtime.session_feedback` owns the write policy:

- hash the thread id before persistence;
- read turn count and user id from runtime state;
- scrub user id in incognito mode;
- attach the trusted feedback source;
- degrade gracefully when persistence fails.

The feedback backends here only store and retrieve records.

# Audit Backends

This package contains always-on operational records for safety, feedback, and
operator review. Audit data is intentionally separate from prompt memory:
audit records are not loaded into `working_memory`, are not used to generate
ordinary therapeutic responses, and are not controlled by conversational memory
recall toggles.

## File Map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package marker and short package-level contract. |
| `models.py` | Source-of-truth pydantic models for crisis logs, classifier audit metadata, and session feedback records. |
| `crisis_log.py` | Defines `CrisisLogBackend` plus in-memory and null crisis-log implementations. Used by tests, incognito mode, and explicit fixtures. |
| `sqlite_crisis_log.py` | Durable SQLite implementation of `CrisisLogBackend` for local/synced runtimes. Stores indexed query columns plus the full serialized `CrisisLogRecord`. |
| `session_feedback.py` | Defines `SessionFeedbackBackend` plus in-memory and null feedback implementations. Used for end-of-session thumbs feedback. |
| `sqlite_session_feedback.py` | Durable SQLite implementation of `SessionFeedbackBackend` for local/synced runtimes. Stores session-keyed feedback rows plus the full serialized `SessionFeedbackRecord`. |

## Graph Significance

`crisis_log.py` is directly significant to the LangGraph flow. The top-level
graph routes crisis turns through:

```text
crisis_resource_lookup_node
  -> crisis_response_node
  -> crisis_log_node
  -> finalize_turn_node
```

`crisis_log_node` reads `runtime.context.crisis_log_backend` and appends one
`CrisisLogRecord` for crisis turns. The node returns no meaningful state delta;
its purpose is the audit side effect. This keeps safety observability separate
from response generation and from normal memory extraction.

The crisis log is always-on across memory modes:

- Incognito mode uses `InMemoryCrisisLogBackend`, so records exist only for the
  runtime lifetime.
- Local/synced modes use `SqliteCrisisLogBackend`, so records survive restarts.
- `NullCrisisLogBackend` should stay limited to explicit tests.

## Runtime Significance

`session_feedback.py` is not a graph node. It is runtime-owned operational
persistence used by `PersistentAgentRuntime.record_session_feedback` when a
session is explicitly ended through CLI or API flows.

That split is intentional:

- Crisis logging belongs in the graph because it must happen immediately after
  the crisis response branch.
- Session feedback belongs in the runtime because it is collected at session
  close, outside ordinary turn routing.

`PersistentAgentRuntime` selects concrete audit backends from memory mode and
exposes them through runtime context or runtime accessors.

## Privacy And Persistence

Audit records use opaque session identifiers and mode-aware persistence. They
should not be treated as therapeutic memory:

- Do not load audit rows into prompt memory.
- Do not let user memory recall controls disable crisis audit writes.
- Do not write ordinary therapeutic content into audit stores unless it belongs
  to an explicit audit record type.
- Keep purge/retention paths operator- or maintenance-driven, not agent-driven.

## Extension Rules

Add a new audit backend here only when the record is operational rather than
therapeutic memory. A good audit feature usually has these properties:

- It must be reviewable by operators or maintainers.
- It should persist independently of conversational memory settings.
- It should not influence normal therapeutic response generation unless a
  dedicated node explicitly reads it.
- It has a clear runtime or graph owner.

If a feature is meant to help the assistant remember the user, place it under
`agent.memory` instead.

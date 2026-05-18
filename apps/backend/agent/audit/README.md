# Audit Backends

This package contains always-on operational records for safety and operator
review. Audit data is intentionally separate from prompt memory: audit records
are not loaded into `working_memory`, are not used to generate ordinary
therapeutic responses, and are not controlled by conversational memory recall
toggles.

## File Map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package marker and short package-level contract. |
| `models.py` | Source-of-truth pydantic models for crisis logs and classifier audit metadata. |
| `crisis_log.py` | Defines `CrisisLogBackend` plus in-memory and null crisis-log implementations. Used by tests, incognito mode, and explicit fixtures. |
| `postgres_crisis_log.py` | Primary Postgres implementation of `CrisisLogBackend` for durable local/runtime deployments. |
| `sqlite_crisis_log.py` | Legacy SQLite implementation of `CrisisLogBackend` for compatibility fallback and migration coverage. |

## Runtime Significance

The OpenAI text runtime writes one `CrisisLogRecord` for crisis-response turns
through `agent.audit.crisis_log.write_crisis_log`. Its purpose is the audit side
effect, keeping safety observability separate from response generation and
ordinary memory operations.

The crisis log is always-on across memory modes:

- Incognito mode uses `InMemoryCrisisLogBackend`, so records exist only for the
  runtime lifetime.
- Local/synced modes use the configured durable backend; Postgres is recommended and SQLite remains a legacy fallback.
- `NullCrisisLogBackend` should stay limited to explicit tests.

Session feedback lives in `agent.feedback`; runtime lifecycle policy for
writing feedback records lives in `agent.runtime.session_feedback`.

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
  dedicated runtime service explicitly reads it.
- It has a clear runtime owner.

If a feature is meant to help the assistant remember the user, place it under
`agent.memory` instead.

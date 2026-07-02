# Safety Audit Ledger

This package owns OpenCouch's deployment-facing safety event ledger. It is not
therapeutic memory, prompt context, or a general observability bucket. The
runtime captures only minimal structured safety events; operators review,
summarize, export, or purge those records later without loading audit data back
into the assistant's ordinary responses.

## Why It Exists

For a user-facing deployment, the safety ledger should answer operational
questions that app logs and prompt memory should not own:

- Did the crisis classifier fire, at what level, and through which classifier
  path?
- Did the crisis-response branch complete?
- Did crisis-resource lookup run, find resources, or fall back?
- Did the runtime use the SDK path, SDK tool fallback path, or direct response
  LLM override?
- Can maintainers build daily safety summaries without exposing raw user text?

If nobody can review or summarize a record, it does not belong here.

## Runtime Flow

The crisis path captures events in one direction only:

1. The crisis gate writes turn-scoped `crisis` and `crisis_audit` state.
2. The crisis-response branch resolves resources and produces the user-facing
   safety reply.
3. The text or voice runtime finalizes and persists the conversation state.
4. `agent.audit.capture.capture_crisis_outcome` gets a small best-effort timeout
   window to invoke `agent.audit.crisis_log.write_crisis_log`.
5. The configured `CrisisLogBackend` appends the record when it completes within
   that window; timeout/failure is logged and must not break the conversation.
6. Operator review, status, export, summary, and retention paths read or purge
   records later through scripts/jobs, not the live response flow.

Audit rows must never be loaded into `working_memory` or used by normal
therapeutic response generation.

## File Map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package marker and short package-level contract. |
| `models.py` | Pydantic record and aggregate models for safety audit data. |
| `capture.py` | Runtime-facing bounded/best-effort capture seam. |
| `summary.py` | Daily aggregate helper for operator-facing counts. |
| `crisis_log.py` | `CrisisLogBackend`, in-memory/null implementations, and lower-level crisis record writer. |
| `postgres_crisis_log.py` | Primary durable Postgres implementation of `CrisisLogBackend`. |
| `sqlite_crisis_log.py` | Legacy SQLite fallback and migration-compatible implementation. |

## What Records Store

`CrisisLogRecord` stores structured operational metadata:

- opaque record id and SHA-256 session id
- `user_id_or_null`, with incognito mode always writing `None`
- detection timestamp, crisis level, classifier path, and bounded classifier
  reason
- response completion and LLM failure flags
- response path, response style, resource lookup status, resource count, tool
  calls, and fallback reason
- optional retention extension fields

It should not store raw transcripts, raw user messages, assistant responses, or
ordinary therapeutic memory.

## Persistence Behavior

The crisis log is always available across memory modes, but persistence is
mode-aware:

- Incognito mode uses `InMemoryCrisisLogBackend`; records die with the runtime.
- Persistent modes use the configured durable backend. Postgres is preferred for
  deployed environments; SQLite remains only as explicit legacy fallback during
  migration.
- `NullCrisisLogBackend` is reserved for explicit tests and fixtures.

User memory recall controls must not disable crisis event capture. Retention and
purge flows are operator- or maintenance-driven, never agent-driven.

## Operator Scripts

Use `scripts/audit_crisis_ledger.py` from the backend virtualenv for ad hoc
review work, for example:

```bash
cd apps/backend
.venv/bin/python ../../scripts/audit_crisis_ledger.py summary --date 2026-06-30
.venv/bin/python ../../scripts/audit_crisis_ledger.py export --date 2026-06-30 --pretty
.venv/bin/python ../../scripts/audit_crisis_ledger.py purge --before 2026-01-01 --yes
```

The script supports SQLite by default and Postgres via `--backend postgres` plus
`--database-url` or `OPENCOUCH_CRISIS_LOG_DATABASE_URL`.

## Extension Rules

Add audit records here only for operational safety or review data. A valid
extension must have:

- a clear runtime owner and write point
- a review, export, summary, or retention use case
- tests for record construction and every backend's round-trip behavior
- privacy boundaries that keep raw therapeutic content out by default

If a feature helps the assistant remember or personalize future replies, put it
under `agent.memory` instead.

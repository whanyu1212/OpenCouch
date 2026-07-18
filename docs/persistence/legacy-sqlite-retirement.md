# Legacy durable SQLite retirement

## Policy

Postgres is the only supported durable backend for application-owned runtime state, active sessions, long-term memory, crisis audit, and session feedback. All legacy application-owned SQLite implementations, including `SqliteMemoryStore` and the built-in SQLite memory tool paths, have been removed. In-memory stores remain available for incognito and explicit tests. OpenAI SDK text sessions are a separate storage surface and are preserved by memory retirement.

Source of truth: `apps/backend/agent/runtime/configuration.py` (`_LEGACY_SQLITE_DURABLE_MESSAGE` and `_validate_legacy_sqlite_durable_allowed`).

## Audit methodology

The inventory was produced by tracing application runtime, audit, feedback, and storage selectors through their factories and tests. The removed audit and feedback stores now use direct Postgres implementations rather than a cross-dialect storage layer.

## Application-store removal matrix

| Subsystem | Legacy SQLite implementation | Supported durable implementation | Selection path | Current role | Target |
|---|---|---|---|---|---|
| Thread persistence: runtime state and active sessions | Removed in #247 | `PostgresRuntimeStateStore`; `PostgresActiveSessionStore` | Durable threads use `thread_persistence_backend="postgres"`; incognito and explicitly non-durable tests use `InMemoryRuntimeStateStore` with `NullActiveSessionStore` | Postgres-only durable persistence; legacy SQLite selection is rejected | Remove obsolete thread SQLite path/configuration compatibility during final cleanup |
| Crisis audit | Removed in #247 | `PostgresCrisisLogBackend` | Persistent mode requires Postgres; incognito and explicit tests use `InMemoryCrisisLogBackend` | Postgres-only durable persistence | Remove inert constructor/path compatibility through #266 |
| Session feedback | Removed in #247 | `PostgresSessionFeedbackBackend` | Persistent mode requires Postgres; incognito and explicit tests use `InMemorySessionFeedbackBackend` | Postgres-only durable persistence | Remove inert constructor/path compatibility through #266 |
| Long-term memory | Removed in PR3 of #233 | `PostgresMemoryStore` | Persistent memory uses Postgres; in-memory stores cover incognito/tests | Postgres-only supported durable persistence; no built-in SQLite access remains | Complete |

## Explicitly separate scope

| Subsystem | Disposition |
|---|---|
| `SqliteMemoryStore` and SQLite memory tool paths | Removed in PR3 of #233. Current OpenCouch releases provide no built-in path to inspect or clear old `memory.sqlite3` files |
| OpenAI SDK text sessions | Separate storage surface and migration decision. Existing `text_sessions.sqlite3` behavior is preserved; incognito is forced to a distinct `:memory:` SDK store by `create_text_session_store`. Do not remove it under #233 without a focused follow-up |
| `InMemoryCrisisLogBackend` | Retain for non-durable tests and incognito/local behavior |
| `InMemorySessionFeedbackBackend` | Retain for non-durable service tests |
| Null backends | Retain disabled/no-op behavior |
| Test fakes | Retain wherever SQL semantics are irrelevant |

## Existing behavioral contracts

No new generic persistence test framework is needed.

- Runtime state: `apps/backend/tests/integration/persistence/test_runtime_state_store_contract.py`
  - missing records, serialized state round trips, overwrite/recency, and idempotent deletion against Postgres.
- Active sessions: `apps/backend/tests/integration/persistence/test_active_session_store_contract.py`
  - payload round trips, listing, mutation tokens, recovery markers, rotation, and deletion against Postgres.
- Postgres audit: `apps/backend/tests/integration/persistence/test_crisis_log_store_contract.py`
  - complete records, date buckets, ordering, counts, concurrency, cross-connection visibility, durable reopen and purge behavior, malformed timestamps, duplicate ids, and close behavior.
- Postgres feedback: `apps/backend/tests/integration/persistence/test_session_feedback_store_contract.py`
  - complete records, session isolation, ordering, counts, concurrency, durable reopen and purge behavior, schema constraints, duplicate ids, malformed timestamps, and close behavior.
- Postgres memory: `apps/backend/tests/integration/persistence/test_memory_store_contract.py`
  - semantic reads/writes, episodic arcs, procedural profiles, reconciliation-related store behavior, deletion, owner scoping, vectors, close behavior, and failure semantics.

## Test migration classification

Move tests to Postgres integration fixtures when they assert durable reopen behavior, SQL constraints, transaction visibility, recovery across connections, or Postgres-specific JSON/vector behavior.

Use in-memory adapters or fakes when tests assert orchestration, route selection, diagnostics, API shaping, or protocol behavior independent of SQL.

The Postgres contracts are authoritative; SQL-insensitive service and orchestration tests use in-memory adapters.

## Migration guidance

Application callers must configure all durable runtime-owned stores against Postgres. Persistent API/TUI startup rejects the legacy SQLite selector. The API and TUI use these environment variables:

```bash
OPENCOUCH_PERSISTENCE_BACKEND=postgres
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch
```

Direct runtime callers should use the grouped configuration boundary:

```python
RuntimePersistenceConfig.for_shared_backend(
    memory_mode=MemoryMode.LOCAL,
    persistence_backend="postgres",
    database_url=database_url,
)
```

Existing `crisis.sqlite3` and `session_feedback.sqlite3` rows are not copied or deleted automatically. Retain archived files if historical records are required; inspect them with an older release or an external read-only SQLite tool. The crisis-ledger operator script is Postgres-only.

After cutover, exercise one audit write and one feedback write in the target environment, then verify both through their supported read/count paths. `OPENCOUCH_ALLOW_LEGACY_SQLITE` does not restore removed application stores.

For long-term memory, there is no SQLite-to-Postgres importer. OpenCouch does
not copy or delete an existing `memory.sqlite3`; archive it if records must be
retained, or discard it after confirming it is no longer needed. The built-in
SQLite memory inspection and clear paths were removed with `SqliteMemoryStore`
in #233 PR3. Inspect an archived file with an older OpenCouch release or an
external read-only SQLite tool. Current memory operator scripts are
Postgres-only and require `--database-url` or
`OPENCOUCH_MEMORY_DATABASE_URL`.

OpenAI SDK text-session SQLite is separate from long-term memory. Existing
`text_sessions.sqlite3` behavior is preserved and is not part of #233.

The flat `memory_sqlite_path` compatibility parameter is inert: it cannot
select, inspect, import, copy, or delete SQLite memory. It remains only until
#266 removes the obsolete constructor/path compatibility surface.

## Planned removal order

1. ~~Migrate runtime-state and active-session SQL-sensitive tests to Postgres contracts.~~ Completed.
2. ~~Remove durable SQLite runtime-state and active-session implementations together because both inherit `thread_persistence_backend`.~~ Completed; incognito now uses explicit non-durable adapters.
3. ~~Migrate audit and feedback durable tests; retain in-memory/null implementations.~~ Completed; Postgres contracts are authoritative and the in-memory/null test paths remain.
4. ~~Remove durable SQLite audit and feedback backends and selection paths.~~ Completed; persistent mode now requires Postgres.
5. Remove obsolete application-store SQLite configuration, factories, migrations, exports, and compatibility tests. Internal factories and cross-dialect code are removed; only the inert flat `memory_sqlite_path` constructor/path compatibility remains until #266.
6. Decide OpenAI SDK text-session SQLite separately.
7. ~~Publish memory operator scripts and migration guidance through #233 PR2.~~ Completed.
8. ~~Remove the legacy SQLite memory implementation, runtime selection, compatibility tests, and built-in SQLite memory tool paths through #233 PR3.~~ Completed.

## Phase 1 exit criteria

- Every application-owned SQLite persistence path is classified.
- Existing Postgres and parity contracts are identified.
- Current legacy opt-in and in-memory exceptions are executable policy contracts.
- No production persistence behavior or generic cross-dialect abstraction changes in the inventory PR.

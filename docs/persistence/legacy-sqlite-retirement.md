# Legacy durable SQLite retirement

## Policy

Postgres is the supported durable persistence backend. Application-owned durable SQLite is legacy compatibility and is rejected unless `allow_legacy_sqlite=True` is explicitly set for temporary migration use. In-memory execution remains supported for incognito and tests. Credential-free TUI behavior is independently protected by `apps/backend/tests/integration/tui/test_opencouch_tui.py::test_tui_parser_defaults_to_dogfood_guest_mode`.

Source of truth: `apps/backend/agent/runtime/configuration.py` (`_LEGACY_SQLITE_DURABLE_MESSAGE` and `_validate_legacy_sqlite_durable_allowed`).

## Audit methodology

The inventory was produced by searching application runtime, audit, feedback, and storage modules for SQLite classes, paths, backend selectors, and SQL dialect construction, then tracing each result through its factory and tests. The application-owned durable selection surfaces are the thread backend, crisis audit, and session feedback; active sessions inherit the thread backend and have no independent selector. `KvStore` is shared audit/feedback implementation infrastructure parameterized by a dialect, not an independently selected persistence backend.

## Application-store removal matrix

| Subsystem | Legacy SQLite implementation | Supported durable implementation | Selection path | Current role | Target |
|---|---|---|---|---|---|
| Thread persistence: runtime state and active sessions | `SqliteRuntimeStateStore`; `SqliteActiveSessionStore` | `PostgresRuntimeStateStore`; `PostgresActiveSessionStore` | `thread_persistence_backend` is consumed by `create_runtime_state_store` and `build_runtime_resources`; active sessions have no independent selector | Coupled legacy durable compatibility and tests | Migrate both contract surfaces and remove their SQLite implementations and selection together |
| Crisis audit | `SqliteCrisisLogBackend` | `PostgresCrisisLogBackend` | `create_crisis_log_backend` | Legacy durable compatibility and parity tests | Retain in-memory audit support; remove durable SQLite backend and selection |
| Session feedback | `SqliteSessionFeedbackBackend` | `PostgresSessionFeedbackBackend` | `create_session_feedback_backend` | Legacy durable compatibility and parity tests | Retain in-memory/null support; remove durable SQLite backend and selection |

## Explicitly separate scope

| Subsystem | Disposition |
|---|---|
| `SqliteMemoryStore` | Deferred to #233 because reconciliation, vectors, episodic arcs, and procedural profiles require a memory-specific migration |
| OpenAI SDK text sessions | Separate storage surface and migration decision. Disk-backed SQLite remains guarded by the current legacy policy; incognito is forced to a distinct `:memory:` SDK store by `create_text_session_store`. Do not remove it under #247 without a focused follow-up |
| `InMemoryCrisisLogBackend` | Retain for non-durable tests and incognito/local behavior |
| `InMemorySessionFeedbackBackend` | Retain for non-durable service tests |
| Null backends | Retain disabled/no-op behavior |
| Test fakes | Retain wherever SQL semantics are irrelevant |

## Existing behavioral contracts

No new generic persistence test framework is needed.

- Runtime state: `apps/backend/tests/integration/persistence/test_runtime_state_store_contract.py`
  - missing records, serialized state round trips, overwrite/recency, and idempotent deletion across SQLite and Postgres.
- Active sessions: `apps/backend/tests/integration/persistence/test_active_session_store_contract.py`
  - payload round trips, listing, mutation tokens, recovery markers, rotation, and deletion across SQLite and Postgres.
- Crisis audit and feedback: `apps/backend/tests/integration/storage/test_kv_store_parity.py`
  - JSON round trips, cross-connection visibility, schema idempotency, failed-connect recovery, ordering, counts, purge semantics, close behavior, duplicate keys, and feedback ordering.
- Postgres audit: `apps/backend/tests/integration/postgres/test_postgres_crisis_log.py`
  - durable reopen behavior and full crisis-record fields.
- Postgres feedback: `apps/backend/tests/integration/postgres/test_postgres_session_feedback.py`
  - durable reopen behavior, ordering, modality, turn counts, and purge boundaries.

## Test migration classification

Move tests to Postgres integration fixtures when they assert durable reopen behavior, SQL constraints, transaction visibility, recovery across connections, or Postgres-specific JSON/vector behavior.

Use in-memory adapters or fakes when tests assert orchestration, route selection, diagnostics, API shaping, or protocol behavior independent of SQL.

Cross-backend parity tests are temporary migration guards. Delete their SQLite cases when the corresponding SQLite implementation is removed; keep the supported Postgres contracts.

## Planned removal order

1. Migrate runtime-state and active-session SQL-sensitive tests to Postgres contracts.
2. Remove durable SQLite runtime-state and active-session implementations in one coupled PR because both inherit `thread_persistence_backend`; do not leave one SQLite consumer behind.
3. Migrate audit and feedback durable tests; retain in-memory/null implementations.
4. Remove durable SQLite audit and feedback backends and selection paths.
5. Remove obsolete application-store SQLite configuration, factories, migrations, exports, and compatibility tests.
6. Decide OpenAI SDK text-session SQLite separately.
7. Complete memory-specific retirement through #233.

## Phase 1 exit criteria

- Every application-owned SQLite persistence path is classified.
- Existing Postgres and parity contracts are identified.
- Current legacy opt-in and in-memory exceptions are executable policy contracts.
- No production persistence behavior or generic cross-dialect abstraction changes in the inventory PR.

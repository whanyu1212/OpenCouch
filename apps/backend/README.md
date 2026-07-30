# OpenCouch Backend

Backend service for the OpenCouch mental health support product.

Current status:
- minimal FastAPI app entrypoint
- OpenAI Agents SDK text runtime with crisis and therapeutic branches
- Postgres durable persistence with ephemeral guest/incognito modes
- provider-backed LLM adapters
- local interactive CLI entrypoint
- pytest backend tests and targeted live-provider checks

Planned implementation order:
1. real chat API endpoint
2. auth and request context
3. memory write and retrieval
4. background summarization and jobs
5. observability

## Deployment contract: single worker

OpenCouch serves from **one worker process**. Runtime mutual exclusion is
process-local: `ThreadLockManager` (`agent/runtime/session/lock.py`) holds
per-thread `asyncio.Lock` objects and binds itself to a single OS thread and
event loop, so session finalization, feedback atomicity, and active-session
mutation serialize within a worker and not across workers.

Startup refuses to boot when a worker count above one is configured via
`WEB_CONCURRENCY`, `UVICORN_WORKERS`, `GUNICORN_WORKERS`, or
`OPENCOUCH_WORKERS`. Most cross-process misuse would fail loudly — the lock
manager raises — but the durable active-session mutation marker would
interleave silently and lose session state, so the guard turns that into a
deploy-time error instead.

Deploy one worker per process, stop-then-start, without overlapping replicas.
Supporting multiple workers would mean replacing every process-local lock with
a durable equivalent; treat it as a deliberate architectural change rather than
a configuration flag.

Local CLI:

```bash
uv run python -m opencouch_tui.cli_app --mode auto
```

Resume a persisted local thread:

```bash
uv run python -m opencouch_tui.cli_app --mode auto --thread-id local-demo
```

Run the CLI against the Dockerized Postgres memory backend. Postgres is the
only supported durable backend for long-term memory:

```bash
OPENCOUCH_PERSISTENCE_BACKEND=postgres \
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch \
uv run python -m opencouch_tui.cli_app --mode auto --thread-id local-demo
```

Legacy `memory.sqlite3` files are not imported, copied, or deleted during the
Postgres cutover; no importer is provided. Archive or discard an old file as
appropriate. `SqliteMemoryStore` and the built-in SQLite memory inspection and
clear paths have been removed. Use an older OpenCouch release or an external
read-only SQLite tool to inspect an archived file. The OpenAI Agents SDK
`text_sessions.sqlite3` store is separate and preserved.

Run backend tests:

```bash
uv run pytest
```

Run backend tests with coverage:

```bash
uv run pytest --cov --cov-report=term-missing
```

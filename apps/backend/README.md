# OpenCouch Backend

Backend service for the OpenCouch mental health support product.

Current status:
- minimal FastAPI app entrypoint
- OpenAI Agents SDK text runtime with crisis and therapeutic branches
- configurable persistence (Postgres recommended, SQLite fallback)
- provider-backed LLM adapters
- local interactive CLI entrypoint
- pytest backend tests and targeted live-provider checks

Planned implementation order:
1. real chat API endpoint
2. auth and request context
3. memory write and retrieval
4. background summarization and jobs
5. observability

Local CLI:

```bash
uv run python -m opencouch_tui.cli_app --mode auto
```

Resume a persisted local thread:

```bash
uv run python -m opencouch_tui.cli_app --mode auto --thread-id local-demo
```

Run the CLI against the Dockerized Postgres memory backend:

```bash
OPENCOUCH_PERSISTENCE_BACKEND=postgres \
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch \
uv run python -m opencouch_tui.cli_app --mode auto --thread-id local-demo
```

Run backend tests:

```bash
uv run pytest
```

Run backend tests with coverage:

```bash
uv run pytest --cov --cov-report=term-missing
```

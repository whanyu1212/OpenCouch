# OpenCouch Backend

Backend service for the OpenCouch mental health support product.

Current status:
- minimal FastAPI app entrypoint
- agent kernel with crisis and therapeutic subgraphs
- LangGraph workflow with SQLite-backed thread persistence
- provider-backed LLM adapters
- local interactive CLI entrypoint
- pytest backend tests and runner-based crisis evals

Planned implementation order:
1. real chat API endpoint
2. auth and request context
3. memory write and retrieval
4. background summarization and jobs
5. observability

Local CLI:

```bash
uv run python -m opencouch_cli --mode auto
```

Resume a persisted local thread:

```bash
uv run python -m opencouch_cli --mode auto --thread-id local-demo
```

Run the CLI against the Dockerized Postgres memory backend:

```bash
OPENCOUCH_PERSISTENCE_BACKEND=postgres \
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch \
uv run python -m opencouch_cli --mode auto --thread-id local-demo
```

Run backend tests:

```bash
uv run pytest
```

Run the local Telegram dogfood gateway:

```bash
OPENCOUCH_TELEGRAM_BOT_TOKEN="123456:abc..." \
OPENCOUCH_TELEGRAM_ALLOW_FROM="123456789" \
OPENCOUCH_TELEGRAM_OWNER_ID="hanyu" \
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER="fast" \
uv run python -m channels.gateway telegram
```

The Telegram gateway is standalone and does not require the FastAPI server.

Run backend tests with coverage:

```bash
uv run pytest --cov --cov-report=term-missing
```

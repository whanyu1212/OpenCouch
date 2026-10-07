# OpenCouch

A mental health support companion. This branch is a from-scratch rewrite; see
[#356](https://github.com/whanyu1212/OpenCouch/issues/356) for the architecture
plan. The previous implementation lives on `main` for reference.

## Layout

| Path | What |
|---|---|
| `apps/agent` | Python agent service: FastAPI + [Pydantic AI](https://pydantic.dev/docs/ai/), served over [AG-UI](https://docs.ag-ui.com) |
| `apps/web` | Next.js frontend using `@ag-ui/client` |

## Local development

Prerequisites: [uv](https://docs.astral.sh/uv/), Node 22 with pnpm, Docker.

```bash
# Agent on http://localhost:8080 (needs OPENAI_API_KEY in apps/agent/.env)
cp apps/agent/.env.example apps/agent/.env
cd apps/agent && uv sync && uv run opencouch-agent

# Web on http://localhost:3000
pnpm install
cp apps/web/.env.example apps/web/.env.local
pnpm dev:web
```

Without an API key, run the agent against Pydantic AI's test model with
`OPENCOUCH_MODEL=test`.

Conversation history is stored in Postgres when `OPENCOUCH_DATABASE_URL` is set,
and kept in memory (lost on restart) otherwise. To run the Postgres-backed
tests locally, set `OPENCOUCH_TEST_POSTGRES_URL`.

To run Postgres and the agent in Docker instead:

```bash
docker compose up --build
```

## Checks

```bash
cd apps/agent && uv run ruff check . && uv run mypy src tests && uv run pytest
pnpm lint:web && pnpm build:web
```

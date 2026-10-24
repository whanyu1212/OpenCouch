<p align="center">
  <img src="assets/brand/banner.png" width="100%" alt="opencouch: You don't have to hold it alone. A teal sofa with a sleeping cat under a reading lamp, rain on the window.">
</p>

# OpenCouch

A guided self-help program for anxiety, with an AI coach that helps people
practise skills instead of reassuring them. This branch is a from-scratch
rewrite; see
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
# Agent on http://localhost:8080 (needs OPENAI_API_KEY and AI_GATEWAY_API_KEY
# in apps/agent/.env)
cp apps/agent/.env.example apps/agent/.env
# --env-file exports the key; the settings loader alone doesn't pass it to the
# OpenAI client.
cd apps/agent && uv sync && uv run --env-file .env opencouch-agent

# Web on http://localhost:3000
pnpm install
cp apps/web/.env.example apps/web/.env.local
pnpm dev:web
```

Every turn is screened by Jev through Vercel AI Gateway before the agent
replies, so the agent won't start without `AI_GATEWAY_API_KEY`. Without API
keys, run it against Pydantic AI's test model with a fake Jev:
`OPENCOUCH_MODEL=test OPENCOUCH_FAKE_JEV=true`. The fake reads every message as
no risk, so it is refused with any other model.

Conversation history is stored in Postgres when `OPENCOUCH_DATABASE_URL` (or
`DATABASE_URL`) is set, and kept in memory (lost on restart) otherwise. To run
the Postgres-backed tests locally, set `OPENCOUCH_TEST_POSTGRES_URL`.

To run Postgres and the agent in Docker instead:

```bash
docker compose up --build
```

## The private demo

The demo runs on Vercel as one project with two
[Services](https://vercel.com/docs/services), configured in `vercel.json`:
`apps/agent` serves `/api/*` and `apps/web` serves everything else, so the
browser calls the agent on the same site (`apps/web/.env.production`).

- **Access:** Deployment Protection with Vercel Authentication covers every
  deployment, the agent's routes included, so there is no public URL. Testers
  get the project's shareable link (Hobby allows one per account), which can be
  revoked from the deployment's Share menu. The agent has no sign-in of its own
  until HY-221, so the protection is what keeps conversations private.
- **Environment variables:** `OPENAI_API_KEY`, `AI_GATEWAY_API_KEY`, and
  `DATABASE_URL` from the Neon database added through the Vercel Marketplace.
  Never `OPENCOUCH_FAKE_JEV`.
- **Deploying:** `vercel deploy --target preview` for a preview, then
  `vercel deploy --prod` for production. A smoke test after deploying: send a
  calm message and a crisis message, and check the crisis one gets the resource
  card.

## Checks

```bash
cd apps/agent && uv run ruff check . && uv run mypy src tests evals scripts && uv run pytest
pnpm lint:web && pnpm build:web
```

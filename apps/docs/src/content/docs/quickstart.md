---
title: Quickstart
description: Run the rewrite locally and preview its documentation.
---

Use a checkout of `develop`. Install uv, Node 22.12+ and pnpm (the root
`package.json` pins its version). Docker is optional for local Postgres.

## Agent without external services

From the repository root:

```sh
cd apps/agent
uv sync --locked
OPENCOUCH_MODEL=test OPENCOUCH_FAKE_JEV=true OPENCOUCH_FAKE_AUTH=true OPENCOUCH_BETA_OPEN=true uv run opencouch-agent
```

The agent listens on port 8080. This mode uses Pydantic AI's test model, treats
every request as one local user, and screens every message as no risk. It is
for plumbing checks, not reply-quality or safety evaluation. Fake screening
and fake authentication are refused with a live model.

Without `OPENCOUCH_DATABASE_URL` or `DATABASE_URL`, stores are in memory and
lost on restart. A configured database enables Postgres persistence.

## Live agent and web client

Copy `apps/agent/.env.example` to `apps/agent/.env` and supply the listed settings,
including `OPENAI_API_KEY`, `AI_GATEWAY_API_KEY`, and `NEON_AUTH_BASE_URL`.
Start the service from `apps/agent`:

```sh
uv run --env-file .env opencouch-agent
```

In another terminal, from the repository root:

```sh
pnpm install --frozen-lockfile
cp apps/web/.env.example apps/web/.env.local
# Fill in the web environment file before starting.
pnpm dev:web
```

The web client runs on port 3000 and points to `http://localhost:8080/api/agent`.
It requires Neon Auth configuration and a cookie secret; the agent's fake-auth
flag does not bypass the web sign-in UI. Configure matching beta invite lists
for the two services. See the
[repository README](https://github.com/whanyu1212/OpenCouch/blob/develop/README.md)
for the private demo and environment details.

## Documentation preview

From the repository root:

```sh
pnpm dev:docs
pnpm check:docs
pnpm build:docs
pnpm preview:docs
```

Open `http://localhost:4321/OpenCouch/`. Run the development server or production
preview separately; search is indexed during the production build.

Relevant PRs and `develop` pushes check and build the docs without publishing.
Only relevant pushes to `main` deploy to the existing GitHub Pages URL.

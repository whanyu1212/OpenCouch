---
title: Quick Start
sidebar_position: 2
---

import TerminalWindow from '@site/src/components/TerminalWindow';
import cliScreenshot from '@site/static/img/cli-example.png';

# Quick Start

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for backend dependency management
- pnpm for the web and docs apps
- A Gemini or OpenAI API key for LLM-backed runs

Deterministic CLI mode works without external model keys.

## Install

<TerminalWindow title="bash — install">
{`git clone https://github.com/whanyu1212/OpenCouch.git
cd OpenCouch
pnpm install
cd apps/backend
uv sync`}
</TerminalWindow>

## Environment

OpenCouch loads local environment files from the repo root and
`apps/backend` (`.env`, then `.env.local`). Real model runs need at
least one provider:

<TerminalWindow title="env — text models">
{`# Defaults to openai when unset.
LLM_PROVIDER=openai
OPENAI_API_KEY=...

# Alternative provider.
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=...
# GOOGLE_API_KEY=...`}
</TerminalWindow>

Optional surfaces need additional variables:

<TerminalWindow title="env — optional surfaces">
{`# Web voice via LiveKit + OpenAI Realtime.
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...

# Telegram dogfood gateway.
OPENCOUCH_TELEGRAM_BOT_TOKEN=123456:abc...
OPENCOUCH_TELEGRAM_ALLOW_FROM=123456789
OPENCOUCH_TELEGRAM_OWNER_ID=alice
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER=fast`}
</TerminalWindow>

## Run the CLI

### Deterministic mode

No LLM calls, in-memory only. Good for verifying the graph and slash
commands.

<TerminalWindow title="bash — deterministic CLI">
{`cd apps/backend
uv run python -m opencouch_cli \\
    --mode deterministic \\
    --memory-mode guest \\
    --thread-id scratch`}
</TerminalWindow>

### Full mode with persistent memory

Real LLM, SQLite-backed storage. Facts, session arcs, and style rules
survive restart.

<TerminalWindow title="bash — persistent CLI">
{`cd apps/backend
uv run python -m opencouch_cli \\
    --mode auto \\
    --memory-mode persistent \\
    --user-id alice \\
    --thread-id alice-s1`}
</TerminalWindow>

Reuse the same `--user-id` and `--thread-id` to resume a conversation.
Use the same `--user-id` with a new `--thread-id` to start a fresh
session that still has access to the user's long-term memory.

<img className="docs-screenshot" src={cliScreenshot} alt="OpenCouch CLI session" />

## Run the Web App

Start the backend and frontend in separate terminals:

<TerminalWindow title="bash — API server">
{`cd apps/backend
uv run uvicorn main:app --port 8000 --reload`}
</TerminalWindow>

<TerminalWindow title="bash — web UI">
{`# From the repository root
pnpm --dir apps/web dev`}
</TerminalWindow>

Open `http://localhost:3000`. The web app talks to
`http://localhost:8000` by default. Set `NEXT_PUBLIC_API_URL` in
`apps/web/.env.local` if the API runs somewhere else.

## Voice Mode

The current browser voice path is LiveKit-first. Start the API server,
then run the LiveKit worker:

<TerminalWindow title="bash — LiveKit voice">
{`# Terminal 1: API server
cd apps/backend
uv run uvicorn main:app --port 8000 --reload

# Terminal 2: LiveKit worker
cd apps/backend
uv run python -m voice.livekit.agent dev`}
</TerminalWindow>

For prompt and tool smoke tests without a browser room:

<TerminalWindow title="bash — voice console">
{`cd apps/backend

# Spoken, uses your mic
uv run python -m voice.livekit.agent console

# Text-only
uv run python -m voice.livekit.agent console --text`}
</TerminalWindow>

The standalone LiveKit test page is still available from the backend
at `/api/voice/livekit/test`; the Next.js voice page is the preferred
dogfood surface. See [Voice (LiveKit)](/docs/voice) for the full
architecture.

## Telegram Gateway

The Telegram gateway is a standalone process for direct-message
dogfood. FastAPI does not need to be running.

Create a bot with `@BotFather`, DM it once, then get your numeric
Telegram sender id:

<TerminalWindow title="bash — Telegram getUpdates">
{`curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"`}
</TerminalWindow>

Use `message.from.id` as the allowlisted sender. If Telegram returns a
409 webhook conflict, clear the webhook and retry:

<TerminalWindow title="bash — Telegram deleteWebhook">
{`curl "https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook"`}
</TerminalWindow>

Run the gateway from `apps/backend`:

<TerminalWindow title="bash — Telegram gateway">
{`OPENCOUCH_TELEGRAM_BOT_TOKEN="123456:abc..." \\
OPENCOUCH_TELEGRAM_ALLOW_FROM="123456789" \\
OPENCOUCH_TELEGRAM_OWNER_ID="alice" \\
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER=fast \\
OPENCOUCH_MEMORY_MODE=persistent \\
OPENCOUCH_TELEGRAM_THREAD_ROTATION_ENABLED=true \\
uv run python -m channels.gateway telegram`}
</TerminalWindow>

Use `/start` or `/help` for the static intro, send normal messages to
talk, and use `/end` to close the active session manually. After
`/end`, the next normal message starts a fresh session; `/start` is
not required again. See [Telegram Gateway](/docs/system/telegram) for
thread rotation and rendering details.

## Slash Commands

Once inside the text CLI:

| Command | What it does |
|---|---|
| `/help` | List all commands |
| `/status` | Thread id, response tier, turn count, and active response LLM |
| `/history [n]` | Recent messages with response-style metadata |
| `/context` | Session context snapshot |
| `/memory status` | Per-namespace counts, recall toggle |
| `/memory list` | Semantic facts + episodic arcs |
| `/memory list rules` | Procedural style rules |
| `/memory recall on\|off` | Toggle proactive content recall |
| `/memory forget fact\|session\|rule <n>` | Delete one record |
| `/memory clear facts\|sessions\|rules\|all` | Wipe a namespace |
| `/memory purge-crisis [days]` | Retention-purge crisis log |
| `/debug state` | Raw graph state as JSON |
| `/end` | Summarize session and save session-end memory |
| `/exit` | End session with save prompt |

## Tests and Evals

:::info For developers and contributors only
End users do not need to run these checks.
:::

Run backend tests from `apps/backend`:

<TerminalWindow title="bash — backend tests">
{`cd apps/backend
uv run pytest tests/`}
</TerminalWindow>

Run frontend checks from the repo root:

<TerminalWindow title="bash — web checks">
{`pnpm --dir apps/web lint
pnpm --dir apps/web build`}
</TerminalWindow>

Run eval harnesses from `apps/backend`:

<TerminalWindow title="bash — eval harnesses">
{`cd apps/backend

# No API key needed
uv run python ../../eval/runners/retrieval_eval.py --mode token-only

# LLM-backed examples
uv run python ../../eval/runners/crisis_gate_eval.py --mode hybrid
uv run python ../../eval/runners/therapeutic_routing_eval.py --mode hybrid
uv run python ../../eval/runners/therapeutic_behavior_eval.py --mode hybrid
uv run python ../../eval/runners/exercise_selection_eval.py --mode hybrid
uv run python ../../eval/runners/grounded_lookup_routing_eval.py --mode hybrid
uv run python ../../eval/runners/memory_control_routing_eval.py --mode hybrid
uv run python ../../eval/runners/memory_write_policy_eval.py --mode hybrid`}
</TerminalWindow>

See [Routing & Classifiers](/docs/agent/routing-classifiers) for the
current eval map.

## Eval-driven Observability

To enable Opik tracing for local text runs, add this to `.env`
before starting the CLI or API:

<TerminalWindow title="env — Opik tracing">
{`# Primary external trace backend.
OPIK_API_KEY=...
OPIK_WORKSPACE=...
OPIK_PROJECT_NAME=opencouch-dev

# Optional secondary LangSmith / LangChain tracing.
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=opencouch-dev

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=opencouch-dev`}
</TerminalWindow>

The `eval/runners/*` scripts remain the source of truth for behavioral
regression checks. Opik is the primary trace surface for graph execution,
run filtering, and experiment review. LangSmith tracing remains supported
as an optional secondary LangChain integration.

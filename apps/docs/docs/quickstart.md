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
- An OpenAI API key for LLM-backed runs

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
`apps/backend` (`.env`, then `.env.local`). Real model runs need an
OpenAI API key:

<TerminalWindow title="env — text models">
{`# Defaults to openai when unset.
LLM_PROVIDER=openai
OPENAI_API_KEY=...`}
</TerminalWindow>

Optional channels need additional variables:

<TerminalWindow title="env — optional surfaces">
{`# Telegram dogfood gateway.
OPENCOUCH_TELEGRAM_BOT_TOKEN=123456:abc...
OPENCOUCH_TELEGRAM_ALLOW_FROM=123456789
OPENCOUCH_TELEGRAM_OWNER_ID=alice
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER=fast`}
</TerminalWindow>

## Run the CLI

### Deterministic mode

No LLM calls, in-memory only. Good for verifying CLI rendering, slash
commands, and local persistence plumbing. User turns return a labeled
deterministic smoke response; use `auto` or `hybrid` for real crisis
classification and therapeutic generation.

<TerminalWindow title="bash — deterministic CLI">
{`cd apps/backend
uv run python -m opencouch_cli \\
    --mode deterministic \\
    --memory-mode guest \\
    --thread-id scratch`}
</TerminalWindow>

### Experimental TUI

The Textual TUI is an experimental dogfood surface. It keeps the REPL as
the canonical local CLI, but adds switchable `Dogfood`, `Debug`, and
`Chat` workspaces for comparing richer terminal workflows. It starts in
light mode by default; use `--theme dark` to start dark, or press
`Ctrl+Y` inside the TUI to switch themes. `Tab` and `Shift+Tab` cycle
workspaces, while `Ctrl+1`, `Ctrl+2`, and `Ctrl+3` jump directly.

<TerminalWindow title="bash — deterministic TUI">
{`./scripts/text_tui.sh \\
    --mode deterministic \\
    --memory-mode guest \\
    --view dogfood \\
    --theme light \\
    --thread-id scratch-tui`}
</TerminalWindow>

### Full mode with persistent memory

Real LLM with durable configured storage. Facts, session arcs, and style
rules survive restart; Postgres is recommended for the local durable path.

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

:::warning Web UI is temporarily behind the backend
The backend text agent is the current dogfooding surface while the app
shell catches up with the agent refactor. Use the CLI when you want the
most reliable local path.
:::

## Voice Mode

Voice mode runs from the web app at `/voice` using OpenAI Realtime
WebRTC. Start the backend and web app, complete setup, then open the
Voice tab. Persistent sessions reuse the same memory owner as text;
incognito voice sessions do not write durable memory.

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

### Session & Display

| Command | What it does |
|---|---|
| `/help` | List all commands |
| `/status` | Thread id, response tier, turn count, and active response LLM |
| `/history [n]` | Recent messages with response-style metadata |
| `/context` | Session context snapshot |
| `/keys` | Show keyboard shortcuts and prompt tips |
| `/ui <compact\|full>` | Switch toolbar density |
| `/theme <mono\|contrast\|calm>` | Switch prompt color theme |
| `/clear` | Clear terminal and redraw header |
| `/reset` | Clear the conversation history |
| `/end` | Summarize session and save session-end memory |
| `/exit` | End session with save prompt |

### Memory

| Command | What it does |
|---|---|
| `/memory status` | Per-namespace counts, recall toggle |
| `/memory list [facts\|sessions\|rules]` | Semantic facts, episodic arcs, or procedural rules |
| `/memory recall on\|off` | Toggle proactive content recall |
| `/memory forget fact\|session\|rule <n>` | Delete one record by index |
| `/memory clear facts\|sessions\|rules\|all` | Wipe a namespace |
| `/memory purge-crisis [days]` | Retention-purge crisis log |

### Threads

| Command | What it does |
|---|---|
| `/threads [n]` | List persisted thread ids (default: 12) |
| `/resume <thread-id>` | Switch to an existing thread |
| `/new [thread-id]` | Start a fresh thread |

### Runtime

| Command | What it does |
|---|---|
| `/mode <deterministic\|hybrid\|auto>` | Switch LLM resolution mode |
| `/response-tier <fast\|quality>` | Switch response quality/latency tradeoff |
| `/trace on\|off\|once` | Show or hide routing trace overlay |
| `/debug state` | Raw runtime state as JSON |

### Aliases

| Alias | Expands to |
|---|---|
| `/h` | `/help` |
| `/s` | `/status` |
| `/m` | `/memory` |
| `/k` | `/keys` |
| `/t` | `/theme` |
| `/q` | `/quit` |
| `/c` | `/clear` |

:::tip
The completion menu shows recently used commands at the top (marked with ↻) and suggests corrections for typos.
:::

## Tests and Evals

:::info For developers and contributors only
End users do not need to run these checks.
:::

Run backend tests from `apps/backend`:

<TerminalWindow title="bash — backend tests">
{`cd apps/backend
uv run pytest tests/unit tests/integration`}
</TerminalWindow>

Run frontend checks from the repo root:

<TerminalWindow title="bash — web checks">
{`pnpm --dir apps/web lint
pnpm --dir apps/web build`}
</TerminalWindow>

Use backend tests and targeted live-provider tests as the regression checks.

## Trace Observability

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

Opik is the primary trace surface for graph execution, run filtering, and
experiment review. LangSmith tracing remains supported as an optional secondary
LangChain integration.

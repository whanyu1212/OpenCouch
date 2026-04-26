---
title: Quick Start
sidebar_position: 2
---

import TerminalWindow from '@site/src/components/TerminalWindow';

# Quick Start

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Gemini or OpenAI API key for LLM-backed modes (optional — the
  CLI runs in deterministic mode without one)

## Install

<TerminalWindow title="bash — install">
{`git clone https://github.com/whanyu1212/OpenCouch.git
cd OpenCouch/apps/backend
uv sync`}
</TerminalWindow>

## Eval-driven development

:::info For developers and contributors only
End users do not need to configure this section — it powers internal observability and regression tracking during development.
:::

To enable LangSmith tracing for local text runs, add the following to your `.env` before starting the CLI or API:

<TerminalWindow title="env — LangSmith tracing">
{`LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=opencouch-dev

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=opencouch-dev`}
</TerminalWindow>

With tracing enabled, OpenCouch emits LangGraph text runs to LangSmith for observability and evaluation workflows. The existing `eval/runners/*` scripts remain the source of truth for behavioral regression checks; LangSmith adds trace inspection, run filtering, and experiment review.

## Run the CLI

### Deterministic mode (no API key needed)

No LLM calls, in-memory only. Good for verifying the pipeline works
and testing the CLI flow and slash commands.

<TerminalWindow title="bash — deterministic mode">
{`uv run python -m opencouch_cli \\
    --mode deterministic \\
    --memory-mode guest \\
    --thread-id scratch`}
</TerminalWindow>

### Full mode with persistent memory

Real LLM, SQLite-backed storage. Facts, session arcs, and style
rules survive CLI restart.

<TerminalWindow title="bash — persistent memory mode">
{`uv run python -m opencouch_cli \\
    --mode auto \\
    --memory-mode persistent \\
    --user-id alice \\
    --thread-id alice-s1`}
</TerminalWindow>

### Resume a prior session

Use the same `--user-id` and `--thread-id` to pick up where you
left off. The LangGraph checkpointer restores the transcript and
the memory store has your prior facts and arcs.

<TerminalWindow title="bash — resume session">
{`uv run python -m opencouch_cli \\
    --mode auto \\
    --memory-mode persistent \\
    --user-id alice \\
    --thread-id alice-s1`}
</TerminalWindow>

### Start a new session with the same memory

Same user, different thread. The agent sees your prior memory
(semantic facts, episodic arcs, procedural rules) but starts a
fresh conversation. First-turn episodic catch-up fires
automatically.

<TerminalWindow title="bash — new thread, same user memory">
{`uv run python -m opencouch_cli \\
    --mode auto \\
    --memory-mode persistent \\
    --user-id alice \\
    --thread-id alice-s2`}
</TerminalWindow>

### Voice mode

LiveKit-native voice runtime. Requires LiveKit credentials
(`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`) plus
`OPENAI_API_KEY` for the realtime model.

<TerminalWindow title="bash — voice (browser room)">
{`# Terminal 1: backend
uv run python run_server.py

# Terminal 2: agent worker
uv run python -m voice.livekit.agent start

# Open the test page (token-dispatch flow)
open "http://127.0.0.1:8080/api/voice/livekit/test?room=my-room&dispatch=1"`}
</TerminalWindow>

For prompt / tool smoke tests without a LiveKit room:

<TerminalWindow title="bash — voice (console)">
{`# Spoken (uses your mic)
uv run python -m voice.livekit.agent console

# Text-only (fastest)
uv run python -m voice.livekit.agent console --text`}
</TerminalWindow>

The voice runtime uses agent handoffs (`TherapeuticAgent` ↔
`CrisisAgent`) and bounded `GroundingTask`s for exercises rather
than running the full LangGraph text agent on every spoken turn.
Memory loads at session start, refreshes selectively mid-session,
and persists via a transcript replay on disconnect. See the
[Voice (LiveKit)](/docs/voice) page for the full architecture.

### Telegram dogfood gateway

The Telegram gateway is a standalone local process for direct-message
text dogfood. It uses the same persistent text runtime as the CLI/API,
but FastAPI does not need to be running.

Create a bot with `@BotFather`, DM it once, then get your numeric
Telegram sender ID:

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
uv run python -m channels.gateway telegram`}
</TerminalWindow>

Use `/start` or `/help` for the static intro, send normal messages to
talk, and use `/end` to close the active session manually. Restarting
the gateway does not require sending `/start` again.

## Slash commands

Once inside the text CLI:

| Command | What it does |
|---|---|
| `/help` | List all commands |
| `/status` | Thread id, mode, turn count |
| `/history [n]` | Recent messages with mode column |
| `/context` | Session context snapshot |
| `/memory status` | Per-namespace counts, recall toggle |
| `/memory list` | Semantic facts + episodic arcs |
| `/memory list rules` | Procedural style rules |
| `/memory recall on\|off` | Toggle proactive content recall |
| `/memory forget fact\|session\|rule <n>` | Delete one record |
| `/memory clear facts\|sessions\|rules\|all` | Wipe a namespace |
| `/memory purge-crisis [days]` | Retention-purge crisis log |
| `/debug state` | Raw graph state as JSON |
| `/end` | Summarize session and save to episodic memory |
| `/exit` | End session with save prompt |

## Run the tests

:::info For developers and contributors only
The test suite and eval harnesses below are for verifying changes during development. End users can skip this section.
:::

<TerminalWindow title="bash — backend tests">
{`uv run pytest tests/`}
</TerminalWindow>

### Observability & evaluation

If LangSmith tracing is enabled, these eval runs also emit traces to your configured LangSmith project, which makes it easier to inspect failures and compare behavior across prompt or model changes.

<TerminalWindow title="bash — eval harnesses">
{`# Retrieval quality (token-recall baseline, no API key needed)
uv run python eval/runners/retrieval_eval.py --mode token-only

# All five harnesses (requires API key)
uv run python eval/runners/crisis_gate_eval.py --mode auto
uv run python eval/runners/therapeutic_routing_eval.py --mode auto
uv run python eval/runners/extraction_eval.py --mode auto
uv run python eval/runners/summarization_eval.py --mode auto
uv run python eval/runners/retrieval_eval.py --mode auto`}
</TerminalWindow>

See the module docstring in `opencouch_cli/app.py` for all seven
CLI invocation patterns and detailed flag descriptions.

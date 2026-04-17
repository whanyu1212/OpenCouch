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

## Run the CLI

### Deterministic mode (no API key needed)

No LLM calls, in-memory only. Good for verifying the pipeline works
and testing CLI panels.

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

Natural voice conversations via the OpenAI Realtime API. Requires
`OPENAI_API_KEY` in `.env.local`.

<TerminalWindow title="bash — voice mode">
{`uv run python -m opencouch_cli --voice`}
</TerminalWindow>

This starts the FastAPI server and opens the voice test page in your
browser. Speak into your microphone to have a voice conversation.
The crisis gate, memory extraction, and session summarization all
run automatically. See the [Voice Chat](/docs/voice/overview) page
for details.

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

<TerminalWindow title="bash — backend tests">
{`uv run pytest tests/`}
</TerminalWindow>

## Run the eval harnesses

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

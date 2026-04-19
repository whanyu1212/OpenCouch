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

Experimental speech preview via the OpenAI Realtime API. Requires
`OPENAI_API_KEY`.

<TerminalWindow title="bash — voice mode">
{`uv run python -m opencouch_cli --voice`}
</TerminalWindow>

This starts the FastAPI server and opens the voice test page in your
browser. Speak into your microphone to have a voice conversation.
This path currently supports low-latency speech, interruption,
truncation, and a memory-backed prompt preload, but it does **not**
yet expose the full agentic stack: no tool calling, no autonomous
actions, and no disconnect-time summarization/memory write-back.
See the [Voice (Experimental)](/docs/voice) page for
details.

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

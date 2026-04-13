<div align="center">

# OpenCouch

**Open-source mental health support agent with persistent memory, crisis safety, and natural voice conversations.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent_graph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![OpenAI Realtime](https://img.shields.io/badge/OpenAI-Realtime_Voice-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/realtime)
[![FastAPI](https://img.shields.io/badge/FastAPI-API_layer-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-persistence-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![Tests](https://img.shields.io/badge/tests-495+-green?logo=pytest&logoColor=white)](#)

</div>

---

OpenCouch is a mental health support agent built on [LangGraph](https://langchain-ai.github.io/langgraph/) for text and the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) for voice. It uses three [CoALA](https://arxiv.org/abs/2309.02427)-inspired memory layers (semantic facts, episodic session arcs, procedural style rules), an always-on crisis safety gate, and six therapeutic response modes.

**Not a therapist. Not a diagnostic tool. Not an emergency service.** A support assistant for talking through difficult moments, reflecting on patterns, and practicing grounding techniques — with real memory that carries across sessions.

## Features

- **Text + Voice** — interactive CLI for text, OpenAI Realtime for natural voice conversations with ~300ms response time
- **Three memory layers** — semantic facts ("my sister Sarah"), episodic session arcs ("last time we talked about..."), procedural style rules ("don't suggest meditation")
- **Always-on crisis gate** — hybrid regex + LLM safety classifier on every turn, persistent audit log, web search for region-specific crisis hotlines
- **Six therapeutic modes** — supportive, reflective, clarifying, psychoeducation, guided exercise, closing
- **Hybrid retrieval** — embedding similarity + token-recall fused via Reciprocal Rank Fusion
- **Privacy controls** — inspect, delete, or wipe anything the agent remembers
- **SQLite persistence** — portable single-file databases, no external server required
- **REST + WebSocket API** — for web frontends, voice, and messaging channel adapters

## Quick start

```bash
cd apps/backend
uv sync

# Text mode
uv run python -m opencouch_cli --mode auto --memory-mode persistent --user-id alice --thread-id s1

# Voice mode (requires OPENAI_API_KEY)
uv run python -m opencouch_cli --voice

# Run tests
uv run pytest tests/
```

### Web frontend

```bash
# Terminal 1 — backend API
cd apps/backend
uv run uvicorn main:app --port 8000 --reload

# Terminal 2 — Next.js dev server
cd apps/web
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend connects to the backend at `localhost:8000`. On first load you'll choose **Persistent** (loads memory and history) or **Incognito** (fresh thread, nothing saved).

Pages: **Chat** (text with diagnostics), **Voice** (OpenAI Realtime), **Memory** (facts/sessions/rules), **State** (full agent state dict inspector).

See the [docs site](apps/docs/) for architecture, memory layers, voice integration, and contributor guides.

## Architecture

```
User (text or voice)
  → Load memory (semantic + episodic + procedural via hybrid RRF)
  → Crisis gate (regex + LLM, always-on)
  → Therapeutic subgraph (6 modes, hybrid dispatch)   ← text mode (LangGraph)
    OR OpenAI Realtime (implicit mode from system prompt) ← voice mode
  → Memory extractors (async, background)
  → Persist (SQLite)
```

**Text mode** runs the full LangGraph pipeline with explicit 6-mode dispatch, per-turn diagnostics, and guided exercise step tracking. **Voice mode** uses the OpenAI Realtime API for natural speech with the crisis gate as a synchronous pre-check and memory extractors running asynchronously.

## Project structure

```
apps/
  backend/
    agent/           — LangGraph graph, nodes, state, models
      memory/        — store, embeddings, retrieval, dedup, extractors
      therapeutic/   — dispatcher, 6 mode nodes, prompt builders
      nodes/         — graph node implementations
    api/             — FastAPI REST + WebSocket endpoints
    voice/           — OpenAI Realtime voice integration
    opencouch_cli/   — Rich-based interactive CLI
    services/llm/    — Gemini + OpenAI provider adapters
    tests/           — 495+ pytest tests
  web/               — Next.js 16 frontend (React 19, Tailwind v4)
  docs/              — Docusaurus documentation site
eval/
  datasets/          — hand-curated eval cases per layer
  runners/           — 5 assertion-based eval harnesses
knowledge/
  policy/            — crisis and privacy policy
  response_modes/    — per-mode therapeutic knowledge
  modalities/        — MI, CBT, grief, etc. (MI active, others authored)
```

## Docs

Run the documentation site locally:

```bash
cd apps/docs
pnpm install
npx docusaurus start --port 3001
```

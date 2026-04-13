<div align="center">

<br/>

<img src="apps/docs/static/img/favicon.svg" width="64" alt="OpenCouch" />

# OpenCouch

**An open-source mental health support agent with persistent memory, crisis safety, and natural voice.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![OpenAI Realtime](https://img.shields.io/badge/OpenAI-Realtime-412991?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/realtime)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)

[Documentation](apps/docs/) · [Quick Start](#quick-start) · [Architecture](#architecture)

</div>

<br/>

> **Not a therapist. Not a diagnostic tool. Not an emergency service.**
> A support assistant for talking through difficult moments, reflecting on patterns, and practicing structured exercises — with real memory that carries across sessions.

---

## Overview

OpenCouch is built on [LangGraph](https://langchain-ai.github.io/langgraph/) for text and the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) for voice. It uses three [CoALA](https://arxiv.org/abs/2309.02427)-inspired memory layers (semantic facts, episodic arcs, procedural rules), an always-on crisis safety gate, six therapeutic response modes, and seven LLM-routed modalities.

### Capabilities

| Area | What it does |
|:---|:---|
| **Interaction** | Text CLI, WebSocket streaming, OpenAI Realtime voice (~300ms), Next.js web UI |
| **Memory** | Semantic facts · episodic session arcs · procedural style rules — inspectable, deletable, portable |
| **Safety** | Hybrid regex + LLM crisis gate on every turn · persistent audit log · web search for regional hotlines |
| **Modes** | Supportive · reflective · clarifying · psychoeducation · guided exercise · closing |
| **Modalities** | MI · CBT · ACT · DBT skills · grief · IPT · PFA — selected per turn by the LLM dispatcher |
| **Exercises** | 12 multi-turn structured exercises across 6 subtypes (grounding, thought work, activation, ACT, self-compassion, emotion regulation) |
| **Retrieval** | Embedding similarity + token-recall fused via Reciprocal Rank Fusion |
| **Privacy** | Persistent vs incognito sessions · per-fact deletion · full memory wipe |
| **Quality** | 545+ tests · 10 eval datasets · 9 assertion-based eval harnesses |

---

## Quick Start

### CLI

```bash
cd apps/backend && uv sync

# text
uv run python -m opencouch_cli --mode auto --memory-mode persistent --user-id alice --thread-id s1

# voice (requires OPENAI_API_KEY)
uv run python -m opencouch_cli --voice
```

### Web

```bash
# terminal 1 — API server
cd apps/backend && uv run uvicorn main:app --port 8000 --reload

# terminal 2 — frontend
cd apps/web && pnpm install && pnpm dev
```

Open [localhost:3000](http://localhost:3000) and choose **Persistent** (loads memory and history) or **Incognito** (fresh thread, nothing saved).

### Docs

```bash
cd apps/docs && pnpm install && npx docusaurus start --port 3001
```

---

## Architecture

```mermaid
flowchart TB
    User([" 🧑 User (text · voice · web)"]):::user

    subgraph pipeline["Agent Pipeline"]
        direction TB
        MEM["🔍 Load Memory<br/><sub>semantic + episodic + procedural<br/>hybrid RRF retrieval</sub>"]
        CRISIS{"🛡️ Crisis Gate<br/><sub>regex + LLM · always-on</sub>"}

        CRISIS_R["🚨 Crisis Response<br/><sub>PFA overlay · hotline search<br/>audit log</sub>"]

        subgraph therapeutic["Therapeutic Subgraph"]
            direction TB
            DISPATCH["⚡ Dispatcher<br/><sub>mode + modality selection<br/>regex → LLM → fallback</sub>"]
            MODES["💬 Response Mode<br/><sub>supportive · reflective · clarifying<br/>psychoeducation · guided exercise · closing</sub>"]
            MODALITY["📚 Modality Overlay<br/><sub>MI · CBT · ACT · DBT<br/>grief · IPT · PFA</sub>"]
        end

        EXTRACT["🧠 Memory Extractors<br/><sub>semantic facts · procedural rules<br/>gated by small-talk check</sub>"]
        PERSIST[("💾 SQLite .store/<br/><sub>threads · memory · crisis log</sub>")]
    end

    User --> MEM
    MEM --> CRISIS
    CRISIS -- safe --> DISPATCH
    CRISIS -- risk --> CRISIS_R
    DISPATCH --> MODES
    MODALITY -.-> MODES
    MODES --> EXTRACT
    CRISIS_R --> EXTRACT
    EXTRACT --> PERSIST

    classDef user fill:#215f5a,stroke:#143432,color:#fff,font-weight:bold
    classDef default fill:#f6f4f0,stroke:#d6d2cb,color:#1a1815
```

---

## Contributing

```bash
# setup
cd apps/backend && uv sync --group dev

# tests (must pass before PR)
uv run pytest tests/

# evals
python eval/runners/therapeutic_routing_eval.py --mode deterministic
python eval/runners/exercise_selection_eval.py

# lint (runs automatically via pre-commit)
uv run ruff check . && uv run ruff format --check .
```

**Branch conventions:** `feature/*` for new capabilities, `fix/*` for bugs, `chore/*` for docs/infra. PRs target `develop`, releases merge `develop` → `main`.

**What to include in a PR:** clear title, summary of what changed and why, test plan checklist. If you're adding a guided exercise, include eval cases. If you're touching the crisis gate, include safety eval results.

---

## Roadmap

| Status | Area | What |
|:---:|:---|:---|
| 🟢 | Web frontend | Next.js UI with chat, voice, memory, state inspector |
| 🟢 | Guided exercises | 12 exercises across 6 subtypes with multi-turn state tracking |
| 🟢 | Modality routing | All 7 modalities LLM-routed per turn |
| 🟡 | Messaging channels | Telegram, WhatsApp, Discord adapters via `Channel` enum |
| 🟡 | Graph memory | Graphiti + Neo4j for entity-relationship reasoning |
| 🟡 | Background consolidation | Automatic fact merging, dormant marking, undo support |
| 🟡 | Acoustic crisis detection | Paralinguistic signals (voice cracking, prosodic flatness) |
| 🔵 | Clinical review | Clinician audit of knowledge files, prompts, and response quality |
| 🔵 | Deployment | Docker Compose, one-command setup, hosted demo |

🟢 shipped · 🟡 planned · 🔵 blocked on external dependency

---

<div align="center">
<sub>Built with care. Use with care.</sub>
</div>

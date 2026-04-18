<div align="center">

<img src="apps/docs/static/img/favicon.svg" width="80" alt="OpenCouch Logo" />

# OpenCouch

**An open-source mental health support agent with persistent memory, crisis safety, and natural voice.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![OpenAI Realtime](https://img.shields.io/badge/OpenAI-Realtime-412991?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/realtime)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)

[Documentation](https://whanyu1212.github.io/OpenCouch/) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Roadmap](#roadmap)

<br/>

<!-- Add a product screenshot or demo GIF here -->
<!-- <img src="docs/static/img/demo.gif" width="100%" alt="OpenCouch Demo" /> -->

> [!IMPORTANT]
> **Not a therapist. Not a diagnostic tool. Not an emergency service.**
> OpenCouch is a support assistant for difficult moments, reflective dialogue, and structured exercises, with memory continuity across sessions.

</div>

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [Roadmap](#roadmap)

---

## Overview

OpenCouch is a conversational support agent built on [LangGraph](https://langchain-ai.github.io/langgraph/) for text orchestration and the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) for low-latency voice. It supports eval-driven development with [LangSmith](https://www.langchain.com/langsmith) for observability and evaluation across the text graph.

It uses three [CoALA](https://arxiv.org/abs/2309.02427)-inspired memory layers (semantic facts, episodic arcs, and procedural rules) to retain continuity across sessions. An **always-on crisis safety gate**, six therapeutic response modes, and seven LLM-routed modalities keep responses safe, grounded, and adaptive.

## Key Features

- **Persistent Memory:** Stores semantic facts, episodic session arcs, and procedural rules across sessions.
- **Safety by Default:** Evaluates every user input with a crisis gate before response generation, with a persistent audit trail.
- **Traceable Execution:** Uses LangSmith to inspect graph execution, route selection, and per-turn text traces.
- **Evaluation-Ready:** Combines local eval runners with LangSmith projects for regression tracking and failure analysis.
- **Multimodal Interaction:** Supports text, web, and natural voice conversations (~300ms via OpenAI Realtime).
- **Guided Exercises:** Provides multi-turn, state-tracked exercises (e.g., grounding and reflection) for structured practice.

---

## Quick Start

### 1. Command Line Interface (CLI)
The CLI provides the fastest way to interact with OpenCouch locally.

```bash
cd apps/backend && uv sync

# Deterministic text mode (No API key needed, useful for local validation)
uv run python -m opencouch_cli --mode deterministic --memory-mode guest --thread-id scratch

# Full text mode with persistent memory (Requires provider API keys)
uv run python -m opencouch_cli --mode auto --memory-mode persistent --user-id alice --thread-id s1

# Voice mode (Requires OPENAI_API_KEY in .env.local)
uv run python -m opencouch_cli --voice
```

### Eval-driven development

#### Observability & evaluation

To enable LangSmith-backed observability and evaluation for local text runs, add the following to `.env` before running the CLI or API:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=opencouch-dev

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=opencouch-dev
```

### 2. Web Interface
Run the full stack with the Next.js frontend and FastAPI backend.

**Terminal 1 — API Server:**
```bash
cd apps/backend
uv run uvicorn main:app --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd apps/web
pnpm install && pnpm dev
```
Open [localhost:3000](http://localhost:3000) in your browser.

### 3. Documentation Site
Run the Docusaurus-powered docs site locally.

```bash
cd apps/docs
pnpm install && npx docusaurus start --port 3001
```

---

## Architecture

OpenCouch uses a directed graph that enforces safety checks before therapeutic generation and runs memory extraction after each finalized turn.

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "primaryColor": "#1a3a38",
    "primaryTextColor": "#e8f5f3",
    "primaryBorderColor": "#3d8b84",
    "lineColor": "#64CCC5",
    "secondaryColor": "#0f2422",
    "tertiaryColor": "#0d1f1e",
    "background": "#0d1f1e",
    "mainBkg": "#1a3a38",
    "nodeBorder": "#3d8b84",
    "clusterBkg": "#0b1716",
    "clusterBorder": "#3d8b84",
    "titleColor": "#64CCC5",
    "edgeLabelBackground": "#0f2422",
    "fontFamily": "ui-monospace, monospace"
  }
}}%%
flowchart TD
    %% Define Node Styles
    classDef inputNode fill:#1E293B,stroke:#94A3B8,stroke-width:2px,color:#F8FAFC
    classDef gateNode fill:#450A0A,stroke:#F87171,stroke-width:2px,color:#FEF2F2
    classDef safeNode fill:#064E3B,stroke:#34D399,stroke-width:2px,color:#ECFDF5
    classDef riskNode fill:#312E81,stroke:#A78BFA,stroke-width:2px,color:#F5F3FF
    classDef sysNode fill:#0F172A,stroke:#38BDF8,stroke-width:2px,color:#F0F9FF
    classDef bgTask fill:#172554,stroke:#60A5FA,stroke-width:2px,color:#DBEAFE,stroke-dasharray: 5 5
    classDef dbNode fill:#14532D,stroke:#4ADE80,stroke-width:2px,color:#F0FDF4

    IN(["User message"]):::inputNode

    subgraph GATE ["Safety Gate"]
        CG{"crisis_gate\nregex + LLM"}:::gateNode
    end

    subgraph SAFE ["Therapeutic Branch"]
        direction TB
        LM["load_memory\nsemantic · episodic · procedural"]:::safeNode
        TS[["therapeutic_subgraph\n6 modes · 7 modalities"]]:::safeNode
        LM ==> TS
    end

    subgraph RISK ["Crisis Branch"]
        CR[["crisis_response\nPFA overlay · local hotlines"]]:::riskNode
    end

    FT{{"finalize_turn\noperator.add reducer"}}:::sysNode

    subgraph EXTRACT ["Parallel Extraction"]
        direction LR
        EF[/"extract_facts"/]:::bgTask
        EP[/"extract_rules"/]:::bgTask
    end

    DB[("SQLite .store/\nthreads · memory · crisis log · feedback")]:::dbNode

    %% Logic Flows
    IN ==> CG
    CG ==>|Safe| LM
    CG -.->|Risk| CR
    TS ==> FT
    CR -.-> FT
    FT -.-> EF & EP
    EF & EP -.-> DB
```

---

## Project Structure

This repository is a monorepo managed with `uv` and `pnpm`.

```text
OpenCouch/
├── apps/
│   ├── backend/                # Python backend (FastAPI, LangGraph)
│   │   ├── agent/              # Conversation graph, nodes, state, runtime context
│   │   │   ├── nodes/          # Individual graph nodes
│   │   │   ├── memory/         # Memory retrieval, deduplication, embeddings
│   │   │   └── therapeutic/    # Therapeutic subgraph, modes, prompt logic
│   │   ├── services/llm/       # LLM adapters (Gemini, OpenAI, etc.)
│   │   ├── opencouch_cli/      # Interactive terminal CLI
│   │   ├── voice/              # OpenAI Realtime voice handlers
│   │   ├── api/                # FastAPI REST + WebSocket routes
│   │   └── tests/              # 630+ pytest unit/integration tests
│   ├── web/                    # Next.js chat application
│   └── docs/                   # Docusaurus documentation site
├── eval/                       # Evaluation harnesses + curated datasets
└── knowledge/                  # Therapeutic prompts, policy, and source-of-truth content
```

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for the full history. Recent highlights:

- **Voice session persistence** — voice chat now survives in-app tab switches and browser tab backgrounding
- **Session trajectory evals** — unified eval runner with 25 long-trajectory cases covering safety, modality, and mode transitions
- **Crisis gate hardening** — LLM-primary architecture with regex fallback, shadow monitoring, and adversarial-resistant prompt
- **Therapeutic dispatcher rewrite** — LLM-primary routing for all 6 modes and 7 modalities, mid-exercise exit detection

---

## Contributing

We welcome contributions. Please review the contribution guidelines before submitting a Pull Request.

### Development Setup

```bash
cd apps/backend && uv sync --group dev

# Run the test suite (must pass before opening a PR)
uv run pytest tests/

# Run evaluation checks
uv run python eval/runners/crisis_gate_eval.py --mode deterministic
uv run python eval/runners/therapeutic_routing_eval.py --mode deterministic

# Run linters and formatters
pre-commit run --all-files
```

**Branch Conventions:**
- `feature/*` for new capabilities
- `fix/*` for bug fixes
- `refactor/*` for architectural changes
- `docs/*` for documentation updates

*(Note: All PRs should target the `develop` branch.)*

---

## Roadmap

| Status | Component | Initiative |
|:---|:---|:---|
| ✓ Shipped | **Web Frontend** | Next.js UI with chat, threading, and memory inspection |
| ✓ Shipped | **Voice Chat** | OpenAI Realtime integration with crisis gate and memory |
| ✓ Shipped | **Guided Exercises** | 12 interactive exercises with multi-turn state tracking |
| ✓ Shipped | **Session Feedback** | End-of-session rating system via UI and CLI |
| ✓ Shipped | **API Layer** | FastAPI REST + WebSocket streaming |
| ○ Planned | **Messaging Channels** | Adapters for Telegram, WhatsApp, and Discord |
| ○ Planned | **Graph Memory** | Graphiti + Neo4j for entity-relationship reasoning |
| ○ Planned | **Consolidation** | Background fact merging, dormant marking, and undo support |
| ○ Planned | **Acoustic Safety** | Paralinguistic crisis detection (prosodic flatness, etc.) |
| ⏸ Blocked | **Clinical Review** | Expert clinician audit of knowledge files and safety logic |

---

<div align="center">
<sub>Built responsibly. Use responsibly.</sub>
<br/>
<br/>
<a href="LICENSE">AGPL-3.0 License</a>
</div>

<div align="center">

<img src="apps/docs/static/img/opencouch-banner-1280x420.png" width="100%" alt="OpenCouch banner" />

**A chat mental health companion that supports your well-being through reflection, guided exercises, and a memory that grows with you.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)

</div>

> [!IMPORTANT]
> **Not a therapist. Not a diagnostic tool. Not an emergency service.**
> OpenCouch is a supportive companion for self-reflection and wellness exercises. It is not a substitute for professional mental health care or medical advice.

> [!WARNING]
> **Invasive Changes In Progress:** OpenCouch is currently going through significant architecture and product changes. The web UI is broken for now while the app shell catches up with the backend refactor. For local dogfooding, use [`scripts/cli_dogfood.sh`](scripts/cli_dogfood.sh) to start the text agent. Voice support is temporarily removed and will be rebuilt from scratch. Expect breaking changes, moving APIs, and documentation that may temporarily lag behind the code while the system is being simplified and stabilized.

---

## Table of Contents
- [📖 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [Screenshots](#screenshots)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Environment](#environment)
  - [Local run commands](#local-run-commands)
- [🧠 Architecture](#-architecture)
  - [Supported Interfaces](#supported-interfaces)
- [📁 Project Structure](#-project-structure)
- [🧪 Development \& Validation](#-development--validation)
  - [Observability](#observability)
- [📝 Changelog](#-changelog)
- [🤝 Contributing](#-contributing)
- [🗺️ Roadmap](#️-roadmap)

---

## 📖 Overview

OpenCouch is a chat companion for day-to-day emotional support, self-reflection, and practical coping. It combines modern conversational AI with structured therapeutic patterns, so users can move between open-ended conversation, guided exercises, and longer-term reflection without starting over each time.

Unlike chatting with ChatGPT, Gemini, or Claude on the web, OpenCouch is not a blank general-purpose assistant. It is built around mental-health-adjacent product needs: safety-aware routing, continuity across sessions, structured memory, and concrete coping workflows. General AI chat can be helpful in the moment, but OpenCouch is designed to support ongoing daily use—remembering what has mattered, guiding users through exercises like grounding or thought work, and keeping the experience focused on emotional support rather than generic task completion.

Under the hood, the text runtime uses the OpenAI Agents SDK behind a FastAPI server, with Postgres-first durable persistence and a legacy SQLite fallback. Memory is split into three [CoALA](https://arxiv.org/abs/2309.02427)-inspired layers: semantic facts, episodic arcs, and procedural rules. Before the assistant responds, each turn goes through safety routing, and backend tests plus Opik traces help catch regressions in core routing behavior.

Voice support is intentionally absent during the current runtime cleanup and will be rebuilt from scratch after the text agent architecture settles.

The project is still pre-beta; a closed beta is planned.

## ✨ Key Features
- **Persistent Memory:** Retains context across sessions using semantic facts, episodic arcs, and procedural rules.
- **Safety First:** Built-in safety routing evaluates every turn before responding, backed by a durable crisis-audit log.
- **Guided Exercises:** 13 multi-turn, state-tracked exercises including grounding, breathing, thought work, and values reflection.
- **Optional Telegram Gateway:** Direct message interface with allow-listing, markdown rendering, and session rotation.
- **Tracing & Regression Checks:** Backend tests, live-provider checks, and Opik traces for regression tracking.

## Screenshots

<table>
  <tr>
    <td colspan="2" width="33%" align="center" valign="top"><img src="apps/docs/static/img/readme/landing.png" width="100%" alt="OpenCouch landing page" /></td>
    <td colspan="2" width="33%" align="center" valign="top"><img src="apps/docs/static/img/readme/chat.png" width="100%" alt="OpenCouch web chat" /></td>
    <td colspan="2" width="33%" align="center" valign="top"><img src="apps/docs/static/img/readme/voice.png" width="100%" alt="OpenCouch voice mode" /></td>
  </tr>
  <tr>
    <td width="16%"></td>
    <td colspan="2" width="33%" align="center" valign="top"><img src="apps/docs/static/img/readme/cli-example.png" width="100%" alt="OpenCouch CLI session" /></td>
    <td colspan="2" width="33%" align="center" valign="top"><img src="apps/docs/static/img/readme/telegram-example.png" width="100%" alt="OpenCouch Telegram DM" /></td>
    <td width="16%"></td>
  </tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop running for the Compose stack.
- `uv` and `pnpm` for manual backend/web development.
- Provider keys only for real model runs. Deterministic CLI and many local checks can run without external API keys.

### Environment

OpenCouch loads local environment files from the repo root and `apps/backend` (`.env`, then `.env.local`). Deterministic mode does not need external API keys. Real model runs need at least one configured provider.

<details>
<summary><b>View Environment Setup Details</b></summary>

For local persistence, the recommended path is the Dockerized Postgres service from `compose.yml`. Backend services default to that stack configuration when `OPENCOUCH_PERSISTENCE_BACKEND` is unset inside Compose.

```env
# Text model provider. Defaults to openai when unset.
LLM_PROVIDER=openai
OPENAI_API_KEY=...

# Alternative text provider.
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=...
# GOOGLE_API_KEY=...

# Local persistence backend for memory, checkpoints, audit, feedback,
# and active-session state.
# The Docker Compose stack defaults to these values automatically.
OPENCOUCH_PERSISTENCE_BACKEND=postgres
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@postgres:5432/opencouch
```

<details>
<summary><b>Optional: Telegram Gateway Environment</b></summary>

```env
# Telegram gateway.
OPENCOUCH_TELEGRAM_BOT_TOKEN=123456:abc...
OPENCOUCH_TELEGRAM_ALLOW_FROM=123456789
OPENCOUCH_TELEGRAM_OWNER_ID=alice
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER=fast
```

</details>

</details>

Keep real `.env` files local and out of version control.

### Local run commands

The most reliable dogfood path is `scripts/cli_dogfood.sh` for text. Compose starts the browser stack: Postgres, backend API, and web.

<details>
<summary><b>View commands for Compose, CLI, web, Telegram, and docs</b></summary>

#### Compose stack

```bash
# Start the full stack with logs attached.
docker compose -f compose.yml up

# Start in the background.
docker compose -f compose.yml up -d

# Rebuild images, then start.
docker compose -f compose.yml up --build

# Rebuild only the web container after frontend edits.
docker compose -f compose.yml up --build web

# Follow API logs after background start.
docker compose -f compose.yml logs -f api

# Stop the stack.
docker compose -f compose.yml down
```

Local URLs: web at [localhost:3000](http://localhost:3000), API at [localhost:8080](http://localhost:8080), health at [localhost:8080/api/health](http://localhost:8080/api/health), and Postgres at `postgresql://opencouch:opencouch@localhost:5432/opencouch`.

Compose reads `.env`, `.env.local`, `apps/backend/.env`, and `apps/backend/.env.local`. Inside Compose, API uses the in-network Postgres URL automatically.

#### Text CLI

```bash
# Preferred persistent dogfood command.
./scripts/cli_dogfood.sh --mode auto --memory-mode persistent --user-id dogfood --response-model-tier quality

# Raw backend CLI commands.
cd apps/backend
uv run python -m opencouch_cli --mode deterministic --memory-mode guest --thread-id scratch
uv run python -m opencouch_cli --mode auto --memory-mode persistent --user-id alice --thread-id s1
```

`scripts/cli_dogfood.sh` starts Dockerized Postgres first and forwards flags to `opencouch_cli`.

#### Manual web stack

Use this when you want each process in its own terminal. The manual stack uses port `8000` for the API because the web client defaults to `http://localhost:8000/api`.

```bash
# Terminal 1: API server.
cd apps/backend && uv run uvicorn main:app --port 8000 --reload

# Terminal 2: frontend, from the repo root.
pnpm install && pnpm --dir apps/web dev
```

#### Optional Telegram gateway

```bash
cd apps/backend
OPENCOUCH_TELEGRAM_BOT_TOKEN="123456:abc..." \
OPENCOUCH_TELEGRAM_ALLOW_FROM="123456789" \
OPENCOUCH_TELEGRAM_OWNER_ID="alice" \
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER="fast" \
uv run python -m channels.gateway telegram
```

#### Documentation site

The official documentation is live at [whanyu1212.github.io/OpenCouch](https://whanyu1212.github.io/OpenCouch/). Run the docs site locally only when editing documentation:

```bash
cd apps/docs
pnpm install && npx docusaurus start --port 3001
```

</details>

See [`apps/backend/README.md`](apps/backend/README.md) for backend-specific commands.

---

## 🧠 Architecture

Before response generation, each turn runs through safety routing. Memory writes happen in two phases: per-turn extraction, then a runtime-coordinated session-end commit for episodic summaries and held candidates.

### Supported Interfaces

- **CLI:** Local text harness for development and testing.
- **Web chat:** Next.js text UI backed by FastAPI REST and WebSocket streaming routes.
- **Optional Telegram:** Direct-message gateway with allow-listing, markdown rendering, `/end`, and session rotation.
- **Backend API:** FastAPI route layer used by the web UI and other clients.

```mermaid
flowchart TD
    %% Define Node Styles (Tinted for Light/Dark Mode)
    classDef inputNode fill:#64748B1A,stroke:#64748B,stroke-width:2px
    classDef gateNode fill:#EF44441A,stroke:#EF4444,stroke-width:2px
    classDef safeNode fill:#10B9811A,stroke:#10B981,stroke-width:2px
    classDef riskNode fill:#F59E0B1A,stroke:#F59E0B,stroke-width:2px
    classDef sysNode fill:#3B82F61A,stroke:#3B82F6,stroke-width:2px
    classDef dbNode fill:#64748B1A,stroke:#64748B,stroke-width:2px

    subgraph SURF ["Runtime Surfaces"]
        CLI["CLI"]:::inputNode
        WEB["Next.js web chat"]:::inputNode
        TG["Optional Telegram DM gateway<br/>thread rotation"]:::inputNode
        API["FastAPI REST/WebSocket"]:::inputNode
    end

    IN(["Text user message"]):::inputNode
    subgraph GATE ["Safety Gate"]
        CG{"crisis_gate<br/>LLM-only classifier"}:::gateNode
    end

    subgraph SAFE ["Therapeutic Branch"]
        direction TB
        TDISP{"turn_dispatch<br/>LLM route + active-flow lifecycle"}:::safeNode
        MC[["memory_control<br/>natural-language memory ops"]]:::safeNode
        GA[["grounded_answer<br/>search-grounded answer"]]:::safeNode
        LM["load_memory<br/>awaits recall / prefetch"]:::safeNode
        TDISP ==>|memory control| MC
        TDISP ==>|lookup| GA
        TDISP ==>|support| LM
    end

    subgraph THERAPY ["Therapeutic Subgraph"]
        direction TB
        TD{"therapeutic_dispatch<br/>LLM route plan + continuity"}:::safeNode
        TR[["therapeutic_response<br/>shared response node"]]:::safeNode
        GE[["guided_exercise_response<br/>LangGraph adapter"]]:::safeNode
        ER[["ExerciseRunner service<br/>selection • step state • deltas"]]:::safeNode
        TD ==>|response style| TR
        TD ==>|guided exercise| GE
        GE -.-> ER
    end

    subgraph RISK ["Crisis Branch"]
        RL[["crisis_resource_lookup<br/>location-aware resources"]]:::riskNode
        CR[["crisis_response<br/>PFA overlay • local hotlines"]]:::riskNode
        CL[/"crisis_log"/]:::riskNode
        RL ==> CR
        CR ==> CL
    end

    FT{{"finalize_turn<br/>checkpoint reply • set route"}}:::sysNode

    subgraph POST ["Runtime Memory Work (outside LangGraph)"]
        direction LR
        MP["memory prefetch<br/>turn-start speculation"]:::sysNode
        MP -.-> LM
    end

    subgraph SESSION ["Session-End Commit (ActiveSessionManager, outside the LangGraph workflow)"]
        direction TB
        SE(["session_end trigger<br/>/end • timeout • shutdown • voice disconnect"]):::sysNode
        SS(["summarize_session<br/>episodic arc"]):::sysNode
        CM(["commit_session_memory<br/>promote held semantic • procedural"]):::sysNode
        SE ==> SS
        SS ==> CM
    end

    DB[("Postgres + pgvector<br/>threads • memory • active sessions • crisis log • feedback • voice status")]:::dbNode

    %% Logic Flows
    CLI ==> IN
    WEB ==> API
    API ==> IN
    TG ==> IN
    VOICE ==> VIN
    VIN -.->|startup/mid-session memory| DB
    VIN -.->|disconnect transcript finalization| SE
    IN ==> CG
    IN -.->|runtime context| MP
    CG ==>|Safe| TDISP
    CG -.->|Risk| RL
    MC ==> FT
    GA ==> FT
    LM ==> TD
    TR ==> FT
    GE ==> FT
    CL -.-> FT
    CM -.->|promoted / reconciled writes| DB
    SS -.->|episodic arc| DB

    %% Subgraph Styling (Removes default gray background)
    style SURF fill:none,stroke:#64748B,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
    style GATE fill:none,stroke:#EF4444,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
    style SAFE fill:none,stroke:#10B981,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
    style THERAPY fill:none,stroke:#10B981,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
    style RISK fill:none,stroke:#F59E0B,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
    style POST fill:none,stroke:#3B82F6,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
    style SESSION fill:none,stroke:#3B82F6,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5
```

---

## 📁 Project Structure

This repository is a monorepo managed with `uv` and `pnpm`.

<details>
<summary><b>View Repository Tree</b></summary>

```text
OpenCouch/
├── apps/
│   ├── backend/                # Python backend (FastAPI, LangGraph)
│   │   ├── agent/              # Conversation graph, nodes, state, runtime context
│   │   │   ├── nodes/          # Individual graph nodes
│   │   │   ├── memory/         # Memory retrieval, deduplication, embeddings
│   │   │   ├── therapeutic/    # Therapeutic subgraph, exercises, prompt logic
│   │   ├── llm/                # LLM adapters (Gemini, OpenAI, etc.)
│   │   ├── opencouch_cli/      # Interactive terminal CLI
│   │   ├── channels/           # Telegram gateway and channel adapters
│   │   ├── api/                # FastAPI REST + WebSocket routes
│   │   └── tests/              # 1100+ pytest unit/integration tests
│   ├── web/                    # Next.js chat application
│   └── docs/                   # Docusaurus documentation site
```
</details>

---

## 🧪 Development & Validation

Backend:

```bash
cd apps/backend && uv sync --group dev

# Run the test suite before opening a PR.
uv run pytest tests/unit tests/integration
```

Web:

```bash
pnpm install
pnpm --dir apps/web lint
pnpm --dir apps/web build
```

Repository hooks:

```bash
uv run pre-commit run --all-files
```

### Observability

For local development trace review, add Opik credentials to `.env` before running the CLI or API:

```env
OPIK_API_KEY=...
OPIK_WORKSPACE=...
OPIK_PROJECT_NAME=opencouch-dev
```

LangSmith / LangChain tracing can be enabled as a secondary backend:

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

---

## 📝 Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for project history and release notes.

---

## 🤝 Contributing

We welcome contributions. Run the relevant checks in [Development & Validation](#-development--validation) before submitting a Pull Request.

**Branch Conventions:**
- `feature/*` for new capabilities
- `fix/*` for bug fixes
- `refactor/*` for architectural changes
- `docs/*` for documentation updates

*(Note: All PRs should target the `develop` branch.)*

---

## 🗺️ Roadmap

OpenCouch is pre-beta and currently focused on stabilizing the core chat, memory, and safety experience before expanding to more platforms.

| Horizon | Area | Focus |
|:---|:---|:---|
| ✅ **Shipped** | **Core product** | Web chat, threading, persistent/incognito sessions, memory inspection, and session feedback |
| ✅ **Shipped** | **Guided support** | 13 state-tracked coping exercises for grounding, breathing, thought work, values reflection, and related flows |
| ✅ **Shipped** | **Runtime & API** | FastAPI REST/WebSocket backend, Postgres-backed persistence, LangGraph checkpoints, crisis audit, and feedback storage |
| 🧪 **Dogfood** | **Messaging** | Telegram direct-message gateway with allow-listing, Markdown rendering, `/end`, and session rotation |
| 🔜 **Next** | **Product stabilization** | Closed beta readiness, onboarding polish, reliability improvements, clearer session lifecycle, and feedback-driven UX fixes |
| 🔜 **Next** | **Voice rebuild** | Rebuild voice from scratch after the text runtime cleanup, without carrying the legacy LiveKit worker forward |
| 🔜 **Next** | **Memory quality** | Background fact consolidation, dormant/obsolete memory handling, better review controls, and undo support |
| 🔜 **Next** | **Safety & evaluation** | Broader eval coverage, clinician-informed review of safety behavior, and stronger regression monitoring |
| 🧭 **Later** | **Mobile** | Native iOS app once the web and backend voice paths are stable |
| 🧭 **Later** | **More channels** | WhatsApp and Discord adapters after the core messaging abstraction is stable |
| 🧭 **Later** | **Graph memory** | Graphiti + Neo4j exploration for entity-relationship reasoning |
| 🧭 **Later** | **Acoustic safety** | Paralinguistic crisis signals such as prosody, flatness, or distress markers |
| 🛑 **Blocked** | **Clinical review** | Expert clinician audit of knowledge files, prompts, guided exercises, and safety logic |

---

<div align="center">
<sub>AGPL-3.0. Not a substitute for professional care.</sub>
<br/>
<br/>
<a href="LICENSE">AGPL-3.0 License</a>
</div>

<div align="center">

<img src="apps/docs/static/img/opencouch-banner-1280x420.png" width="100%" alt="OpenCouch banner" />

**A chat and voice mental health companion that supports your well-being through reflection, guided exercises, and a memory that grows with you.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![LiveKit](https://img.shields.io/badge/LiveKit-voice-FF2E63?style=flat-square&logo=livekit&logoColor=white)](https://livekit.io/)
[![OpenAI Realtime](https://img.shields.io/badge/OpenAI-Realtime-412991?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/realtime)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)

</div>

> [!IMPORTANT]
> **Not a therapist. Not a diagnostic tool. Not an emergency service.**
> OpenCouch is a supportive companion for self-reflection and wellness exercises. It is not a substitute for professional mental health care or medical advice.

> [!WARNING]
> **Invasive Changes In Progress:** OpenCouch is currently going through significant architecture and product changes. Expect breaking changes, moving APIs, and documentation that may temporarily lag behind the code while the system is being simplified and stabilized.

---

## Table of Contents
- [📖 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [Screenshots](#screenshots)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Environment](#environment)
  - [One-command local stack](#one-command-local-stack)
  - [Manual web stack](#manual-web-stack)
  - [CLI](#cli)
  - [Telegram Gateway](#telegram-gateway)
  - [Documentation Site](#documentation-site)
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

OpenCouch is a chat and voice companion for day-to-day emotional support, self-reflection, and practical coping. It combines modern conversational AI with structured therapeutic patterns, so users can move between open-ended conversation, guided exercises, and longer-term reflection without starting over each time.

Unlike chatting with ChatGPT, Gemini, or Claude on the web, OpenCouch is not a blank general-purpose assistant. It is built around mental-health-adjacent product needs: safety-aware routing, continuity across sessions, structured memory, and concrete coping workflows. General AI chat can be helpful in the moment, but OpenCouch is designed to support ongoing daily use—remembering what has mattered, guiding users through exercises like grounding or thought work, and keeping the experience focused on emotional support rather than generic task completion.

Under the hood, the text runtime is a [LangGraph](https://langchain-ai.github.io/langgraph/) graph behind a FastAPI server, with Postgres-first durable persistence and a legacy SQLite fallback. Memory is split into three [CoALA](https://arxiv.org/abs/2309.02427)-inspired layers: semantic facts, episodic arcs, and procedural rules. Before the assistant responds, each turn goes through safety routing, and local evals plus Opik traces help catch regressions in core routing behavior.

Voice support is experimental and LiveKit-first in the web app. The browser joins a LiveKit room, a LiveKit Agents worker runs the speech loop, and OpenAI Realtime handles the speech-to-speech model interaction. The older direct Realtime harness remains in the backend for experiments.

The project is still pre-beta; a closed beta is planned.

## ✨ Key Features
- **Persistent Memory:** Retains context across sessions using semantic facts, episodic arcs, and procedural rules.
- **Safety First:** Built-in safety routing evaluates every turn before responding, backed by a durable crisis-audit log.
- **Guided Exercises:** 13 multi-turn, state-tracked exercises including grounding, breathing, thought work, and values reflection.
- **Voice Support:** Browser voice sessions via LiveKit and OpenAI Realtime, with configurable voices, transcription hints, and interruption handling.
- **Telegram Gateway:** Direct message interface with allow-listing, markdown rendering, and session rotation.
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
- Docker Desktop running for the one-command local stack.
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
# active-session state, and LiveKit voice finalization status.
# The Docker Compose stack defaults to these values automatically.
OPENCOUCH_PERSISTENCE_BACKEND=postgres
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@postgres:5432/opencouch
```

Voice and Telegram need extra configuration:

```env
# Web voice via LiveKit + OpenAI Realtime model.
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...

# Telegram gateway.
OPENCOUCH_TELEGRAM_BOT_TOKEN=123456:abc...
OPENCOUCH_TELEGRAM_ALLOW_FROM=123456789
OPENCOUCH_TELEGRAM_OWNER_ID=alice
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER=fast
```

</details>

Keep real `.env` files local and out of version control.

### One-command local stack
Use this for the full local web + voice stack with backend reload, production-mode Next.js, and Dockerized Postgres persistence:

```bash
docker compose up --build
```

This starts:
- PostgreSQL + pgvector for runtime persistence: `postgresql://opencouch:opencouch@localhost:5432/opencouch`
- backend API: [localhost:8080/api/health](http://localhost:8080/api/health)
- LiveKit voice worker: `python -m voice.livekit.agent start`
- Next.js web UI in production mode: [localhost:3000](http://localhost:3000)

The first run can take a while. Docker needs to pull base images, install backend dependencies, build the production web bundle, and warm the voice worker dependencies. Later runs should be much faster because Docker reuses image layers and dependency caches unless the lockfiles or Dockerfiles change.

The Compose stack reads `.env`, `.env.local`, `apps/backend/.env`, and `apps/backend/.env.local` when present. For browser voice, set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and `OPENAI_API_KEY` before starting the stack.

Inside Compose, the API and voice worker default to `OPENCOUCH_PERSISTENCE_BACKEND=postgres`. That routes memory, LangGraph checkpoints, active-session state, crisis audit, session feedback, and LiveKit voice finalization status through the shared Postgres service. The backend default is also Postgres outside Compose, so you must export `OPENCOUCH_MEMORY_DATABASE_URL` (and friends) — or set `OPENCOUCH_PERSISTENCE_BACKEND=sqlite` to opt into the SQLite fallback for local-only installs without Docker.

For text-only development without LiveKit credentials, start just the API, Postgres, and production-mode web UI:

```bash
docker compose up --build postgres api web
```

The Compose web service runs `next build` + `next start`, so frontend source edits require rebuilding the `web` service:

```bash
docker compose up --build web
```

Stop everything with:

```bash
docker compose down
```

### Manual web stack
Use this when you want each process in its own terminal. The manual stack uses port `8000` for the API because the web client defaults to `http://localhost:8000/api`; the Compose stack uses container-friendly port `8080` and sets `NEXT_PUBLIC_API_URL` for the web container.

Terminal 1 — API server:
```bash
cd apps/backend
uv run uvicorn main:app --port 8000 --reload
```

Terminal 2 — frontend:
```bash
pnpm install
pnpm --dir apps/web dev
```

Open [localhost:3000](http://localhost:3000) in your browser.

Optional terminal 3 — LiveKit voice worker:
```bash
cd apps/backend
uv run python -m voice.livekit.agent start
```

### CLI
The CLI is the fastest way to interact with the backend locally.

```bash
cd apps/backend && uv sync

# Deterministic text mode: no API key needed.
uv run python -m opencouch_cli --mode deterministic --memory-mode guest --thread-id scratch

# Full text mode with persistent memory.
uv run python -m opencouch_cli --mode auto --memory-mode persistent --user-id alice --thread-id s1

# Voice mode: requires LiveKit env vars plus OPENAI_API_KEY.
uv run python -m opencouch_cli --voice
```

For everyday persistent-mode dogfooding, [`scripts/cli_dogfood.sh`](scripts/cli_dogfood.sh) wraps the two prerequisites into one command: it ensures the Dockerized Postgres service is up (via `docker compose up -d postgres --wait`) and then launches the CLI from `apps/backend`. Any flags are forwarded to the CLI:

```bash
./scripts/cli_dogfood.sh
./scripts/cli_dogfood.sh --memory-mode persistent --user-id alice --thread-id s1
```

This assumes `OPENCOUCH_PERSISTENCE_BACKEND=postgres` and `OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch` are set in your `.env` (see [Environment](#environment)). Use the raw `uv run python -m opencouch_cli ...` invocations above when you want guest mode, deterministic mode, or the SQLite fallback without starting Postgres.

### Telegram Gateway
Run the standalone Telegram gateway. It does not require the FastAPI server.

```bash
cd apps/backend
OPENCOUCH_TELEGRAM_BOT_TOKEN="123456:abc..." \
OPENCOUCH_TELEGRAM_ALLOW_FROM="123456789" \
OPENCOUCH_TELEGRAM_OWNER_ID="alice" \
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER="fast" \
uv run python -m channels.gateway telegram
```

See [`apps/backend/README.md`](apps/backend/README.md) for backend-specific commands.

### Documentation Site

> **For developers and contributors only.** The hosted docs are available at the link below — running locally is only needed if you're editing documentation.

The official documentation is live at: **[https://whanyu1212.github.io/OpenCouch/](https://whanyu1212.github.io/OpenCouch/)**

To run the Docusaurus-powered docs site locally:

```bash
cd apps/docs
pnpm install && npx docusaurus start --port 3001
```

---

## 🧠 Architecture

Before response generation, each turn runs through safety routing. Memory writes happen in two phases: per-turn extraction, then a runtime-coordinated session-end commit for episodic summaries and held candidates.

### Supported Interfaces

- **CLI:** Local text and voice harness for development and testing.
- **Web chat:** Next.js text UI backed by FastAPI REST and WebSocket streaming routes.
- **Web voice:** LiveKit browser sessions with a LiveKit Agents worker and OpenAI Realtime model.
- **Telegram:** Direct-message gateway with allow-listing, markdown rendering, `/end`, and session rotation.
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
        VOICE["LiveKit voice"]:::inputNode
        TG["Telegram DM gateway<br/>thread rotation"]:::inputNode
        API["FastAPI REST/WebSocket"]:::inputNode
    end

    IN(["User message / transcript"]):::inputNode

    subgraph GATE ["Safety Gate"]
        CG{"crisis_gate<br/>rules + LLM classifier"}:::gateNode
    end

    subgraph SAFE ["Therapeutic Branch"]
        direction TB
        MCG{"memory_control_gate<br/>LLM + deterministic fallback"}:::safeNode
        MC[["memory_control<br/>slash + natural language"]]:::safeNode
        GLG{"grounded_lookup_gate<br/>LLM + hard-yes fallback"}:::safeNode
        GA[["grounded_answer<br/>search-grounded answer"]]:::safeNode
        LM["load_memory<br/>semantic • episodic • procedural"]:::safeNode
        MCG ==>|memory control| MC
        MCG ==>|ordinary turn| GLG
        GLG ==>|lookup| GA
        GLG ==>|support| LM
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

    subgraph POST ["Runtime Memory Side Effects (outside LangGraph)"]
        direction LR
        MX["TurnExtractionCoordinator<br/>background after graph END"]:::sysNode
        EF["extract_semantic_facts<br/>write policy: commit • hold • drop"]:::sysNode
        EP["extract_procedural_rules<br/>write policy: commit • hold • drop"]:::sysNode
        SB[("session buffer<br/>held semantic • procedural")]:::sysNode
        MX -.-> EF
        MX -.-> EP
        EF -.->|hold| SB
        EP -.->|hold| SB
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
    VOICE ==> API
    API ==> IN
    TG ==> IN
    IN ==> CG
    CG ==>|Safe| MCG
    CG -.->|Risk| RL
    MC ==> FT
    GA ==> FT
    LM ==> TD
    TR ==> FT
    GE ==> FT
    CL -.-> FT
    FT -.->|runtime schedules| MX
    EF -.->|immediate writes| DB
    EP -.->|immediate writes| DB
    SB -.->|held candidates| CM
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
│   │   │   └── therapeutic/    # Therapeutic subgraph, modes, prompt logic
│   │   ├── llm/                # LLM adapters (Gemini, OpenAI, etc.)
│   │   ├── opencouch_cli/      # Interactive terminal CLI
│   │   ├── voice/              # LiveKit voice worker + direct Realtime harness
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
uv run pytest tests/
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

See [`CHANGELOG.md`](CHANGELOG.md) for the full history. Recent highlights as of **May 2026**:

- **May 2026 — Therapeutic dispatch & state-surface cleanup** — the therapeutic router collapsed to an LLM-primary policy (~821 LOC removed across four deleted dispatch modules) with a small deterministic exercise-state bookkeeping layer. Three carrying-cost-only output channels (`response_style_type`, `response_style_source`, `response_kind`) were removed end-to-end across the agent, public API, frontend, and docs, with the public `AgentOutput.response_type` now derived once from the crisis assessment rather than written by five separate nodes. 1025/1025 backend tests pass.
- **May 2026 — Postgres-first runtime** — the Docker Compose stack now routes memory, LangGraph checkpoints, active-session state, crisis audit, feedback, and LiveKit voice finalization through Dockerized Postgres, with SQLite kept as a local compatibility fallback outside Compose. A `scripts/cli_dogfood.sh` helper ensures Postgres is up before launching the CLI, and `get_settings()` now fails fast with an actionable error when the Postgres backend is selected without `OPENCOUCH_MEMORY_DATABASE_URL`.
- **May 2026 — Agent module restructure & service extraction** — large structural cleanup of the agent package (net **−1043 lines across 172 files**): voice and active-session modules pulled into `agent/`, memory store promoted to a backend-aware package, risk-gating subsystems grouped under `agent/gates/`, facade and wrapper modules dissolved, and standalone services extracted for memory control, session finalization, runtime streaming, and turn-extraction coordination. Routing decisions are now typed across the agent router, grounded-lookup router, and memory-control router.
- **May 2026 — Off-turn memory extraction** — semantic and procedural memory extraction now runs after the user-visible reply has rendered, removing ~250–300ms median (and up to ~800ms p95) of post-turn latency from the perceived response time. Extractor edges and candidate-policy evaluation also run in parallel where safe.
- **May 2026 — Full local product stack** — the one-command Compose setup runs Postgres, the FastAPI backend, production-mode Next.js web UI, and the LiveKit voice worker together for a closer-to-real local environment.
- **Apr–May 2026 — Web and voice experience refresh** — chat, memory, state, and voice routes now have stronger session continuity, clearer setup/end-session flows, voice selection, mic warmup states, and route-persistent text streaming.
- **Apr–May 2026 — Safety, memory, and routing hardening** — crisis routing, grounded lookup, memory control, therapeutic dispatch, guided exercises, extraction policy, and session summarization are backed by deterministic tests and Opik tracing.
- **Recent — Guided support coverage** — OpenCouch includes 13 state-tracked coping exercises, including grounding, breathing, thought work, and values reflection, with tests covering selection, flow, and memory behavior.
- **Recent — Channel expansion** — Telegram direct-message support now includes allow-listing, `/end`, markdown rendering, session rotation, startup recovery, and non-blocking maintenance sweeps.

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

OpenCouch is pre-beta and currently focused on stabilizing the core chat, memory, safety, and voice experience before expanding to more platforms.

| Horizon | Area | Focus |
|:---|:---|:---|
| ✅ **Shipped** | **Core product** | Web chat, threading, persistent/incognito sessions, memory inspection, and session feedback |
| ✅ **Shipped** | **Voice** | LiveKit-backed browser voice sessions with OpenAI Realtime, safety routing, transcript handling, and memory integration |
| ✅ **Shipped** | **Guided support** | 13 state-tracked coping exercises for grounding, breathing, thought work, values reflection, and related flows |
| ✅ **Shipped** | **Runtime & API** | FastAPI REST/WebSocket backend, Postgres-backed persistence, LangGraph checkpoints, crisis audit, and feedback storage |
| 🧪 **Dogfood** | **Messaging** | Telegram direct-message gateway with allow-listing, Markdown rendering, `/end`, and session rotation |
| 🔜 **Next** | **Product stabilization** | Closed beta readiness, onboarding polish, reliability improvements, clearer session lifecycle, and feedback-driven UX fixes |
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

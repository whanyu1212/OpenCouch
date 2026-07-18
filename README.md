<div align="center">

<img src="apps/docs/static/img/opencouch-banner-1280x420.png" width="100%" alt="OpenCouch banner" />

**A chat mental health companion that supports your well-being through reflection, guided exercises, and a memory that grows with you.**

[![CI](https://img.shields.io/github/actions/workflow/status/whanyu1212/OpenCouch/ci.yml?branch=develop&label=CI&style=flat-square)](https://github.com/whanyu1212/OpenCouch/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-2563EB?style=flat-square)](https://whanyu1212.github.io/OpenCouch/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![OpenAI Agents SDK](https://img.shields.io/badge/OpenAI_Agents_SDK-runtime-10B981?style=flat-square)](apps/backend/agent/README.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-1100%2B-success?style=flat-square)](apps/backend/tests)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)

[**Documentation**](https://whanyu1212.github.io/OpenCouch/) · [**Changelog**](CHANGELOG.md) · [**Roadmap**](#roadmap) · [**Architecture**](#architecture)

</div>

> [!IMPORTANT]
> **Not a therapist. Not a diagnostic tool. Not an emergency service.**
> OpenCouch is a supportive companion for self-reflection and wellness exercises — not a substitute for professional mental health care or medical advice. If you are in crisis, contact local emergency services or a regional crisis line.

> [!NOTE]
> **Project status — pre-beta, active stabilization.** The backend runtime, memory, safety, voice, and observability paths are actively evolving. Web chat, text TUI, and OpenAI Realtime voice backend modules exist, but product UX and documentation may lag the code during stabilization. For local text dogfooding, run [`scripts/text_tui.sh`](scripts/text_tui.sh).

---

## Table of contents
- [Overview](#overview)
- [Key features](#key-features)
- [Screenshots](#screenshots)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Development & validation](#development--validation)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

OpenCouch is a chat companion for day-to-day emotional support, self-reflection, and practical coping. The text runtime uses the OpenAI Agents SDK behind a FastAPI server, with Postgres-first durable persistence, three-layer memory ([CoALA](https://arxiv.org/abs/2309.02427)-inspired), and safety routing on every turn.

**What sets it apart from a general AI assistant:**

- **Continuity by default** — memory and session state persist across days; users don't restart their context every conversation.
- **Safety-aware routing** — a classifier evaluates every turn before response generation and can hand off to a specialist crisis agent with durable audit.
- **Structured coping flows** — 13 multi-turn guided exercises (grounding, breathing, thought work, values) with consent and step-state tracking, instead of free-form advice.
- **Designed for ongoing use** — the product surface is intentionally narrow (emotional support and reflection), not a general task assistant.

Voice support is being rebuilt on the OpenAI Realtime path. The backend has active Realtime voice runtime modules and test coverage, while the product voice UX remains pre-beta and dogfood-oriented. The project is pre-beta; a closed beta is planned.

## Key features

- **Three-layer persistent memory.** Semantic facts, episodic arcs, and procedural rules persisted across sessions ([CoALA](https://arxiv.org/abs/2309.02427)-inspired), with per-turn extraction and session-end consolidation.
- **Safety routing on every turn.** Pre-response safety classification with a durable crisis-audit log; specialist crisis agent takes over when triggered.
- **13 state-tracked guided exercises.** Multi-turn flows for grounding, box breathing, thought work, values reflection, and related coping skills — with consent and step-state preserved across turns.
- **Postgres-first persistence.** Thread state, long-term memory, active sessions, crisis log, and feedback all live in Postgres; guest/incognito runs remain ephemeral, while legacy SQLite is limited to memory and SDK-session compatibility.
- **Multiple surfaces.** Web chat (Next.js) and text TUI (`scripts/text_tui.sh`), both backed by the same FastAPI runtime.
- **1100+ backend tests + tracing.** Unit, integration, voice, audit, observability, and live-provider suites; vendor-neutral tracing supports state diagnostics, structured logging, and OpenTelemetry/OTLP export.

## Screenshots

<table>
  <tr>
    <td width="50%" align="center" valign="top"><img src="apps/docs/static/img/readme/landing.png" width="100%" alt="OpenCouch landing page" /><br/><sub>Landing</sub></td>
    <td width="50%" align="center" valign="top"><img src="apps/docs/static/img/readme/chat.png" width="100%" alt="OpenCouch web chat" /><br/><sub>Web chat</sub></td>
  </tr>
  <tr>
    <td width="50%" align="center" valign="top" colspan="2"><img src="apps/docs/static/img/readme/cli-example.png" width="50%" alt="OpenCouch text TUI session" /><br/><sub>Text TUI</sub></td>
  </tr>
</table>

<sub>Voice UX screenshots are omitted while the OpenAI Realtime voice path remains dogfood/pre-beta.</sub>

---

## Quick start

The fastest persistent dogfood path — clone, set an API key, start local Postgres, then launch the text TUI:

```bash
git clone https://github.com/whanyu1212/OpenCouch.git
cd OpenCouch
cat > apps/backend/.env <<'EOF'
OPENAI_API_KEY=sk-...
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch
EOF
docker compose -f compose.yml up -d postgres --wait
./scripts/text_tui.sh --mode auto --memory-mode persistent --user-id dogfood
```

For a no-key deterministic smoke test, run `./scripts/text_tui.sh --mode deterministic --memory-mode guest`. For the full browser experience, see the [Compose section](#local-run-commands) below.

### Prerequisites
- Docker Desktop with Docker Compose v2 for `compose.yml` and local Postgres persistence.
- An OpenAI API key in `.env` or `apps/backend/.env` for real model runs and OpenAI Realtime voice. Deterministic TUI sessions and many local checks can run without external API keys.
- `uv` for manual backend/TUI development outside Docker.
- Node.js 22+ with Corepack/`pnpm` for manual web or docs development outside Docker.
- Optional: an OTLP collector endpoint if you want to export OpenTelemetry traces locally.

### Environment

OpenCouch loads local environment files from the repo root and `apps/backend` (`.env`, then `.env.local`). Deterministic mode does not need external API keys. Real model runs need an OpenAI API key.

<details>
<summary><b>View Environment Setup Details</b></summary>

For local persistence, the recommended path is the Dockerized Postgres service from `compose.yml`. The Compose API service sets `OPENCOUCH_PERSISTENCE_BACKEND=postgres` and shares one `OPENCOUCH_MEMORY_DATABASE_URL` for memory, thread state, active sessions, crisis audit, and feedback.

```env
# Text/voice model provider. Defaults to openai when unset.
LLM_PROVIDER=openai
OPENAI_API_KEY=...

# Local persistence backend for memory, checkpoints, audit, feedback,
# and active-session state.
# Compose sets these automatically for the API container.
OPENCOUCH_PERSISTENCE_BACKEND=postgres
OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@postgres:5432/opencouch

# For manual host-side backend/TUI runs against the Compose database, use:
# OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch
```

</details>

Keep real `.env` files local and out of version control.

### Local run commands

The most reliable dogfood path is `scripts/text_tui.sh`. Compose starts the browser stack: Postgres, backend API, and web; you can also start only Postgres when using the TUI with persistent memory.

<details>
<summary><b>View commands for Compose, TUI, web, and docs</b></summary>

#### Recommended local browser testing

For day-to-day browser testing, run the backend stack in Compose and run the Next.js app locally for faster hot reload.

```bash
# Terminal 1, from the repo root: Postgres + API on http://localhost:8080/api.
docker compose up api postgres
```

```bash
# Terminal 2, from the repo root: local frontend with hot reload.
cd apps/web
NEXT_PUBLIC_API_URL=http://localhost:8080/api pnpm dev
```

Compose service names are `api` and `postgres`; there is no `backend` service. The web client defaults to `http://localhost:8000/api`, so set `NEXT_PUBLIC_API_URL=http://localhost:8080/api` whenever the API is running through Compose.

Local URLs:

- API base: [localhost:8080/api](http://localhost:8080/api)
- API health: [localhost:8080/api/health](http://localhost:8080/api/health)
- Web UI: [localhost:3000](http://localhost:3000)
- Postgres: `postgresql://opencouch:opencouch@localhost:5432/opencouch`

If the web UI shows “Connection error. Check that the backend is running on the configured API URL,” stop `pnpm dev` and restart it with the Compose API URL:

```bash
cd apps/web
NEXT_PUBLIC_API_URL=http://localhost:8080/api pnpm dev
```

Optionally persist that setting for local Compose testing:

```bash
printf 'NEXT_PUBLIC_API_URL=http://localhost:8080/api\n' > apps/web/.env.local
```

#### Compose stack

`compose.yml` starts the local Postgres + FastAPI backend stack by default. The Next.js web container is still available for full-stack smoke testing, but it is profile-gated because day-to-day frontend work is faster with local `pnpm dev` hot reload.

```bash
# Start the daily backend stack: Postgres + API.
./scripts/dev_api_stack.sh
# Equivalent raw Compose command:
docker compose -f compose.yml up postgres api

# Start the daily backend stack in the background.
docker compose -f compose.yml up -d postgres api

# Start the full stack, including production-built web.
./scripts/dev_full_stack.sh
# Equivalent raw Compose command:
docker compose -f compose.yml --profile web up

# Rebuild images, then start the full stack.
docker compose -f compose.yml --profile web up --build

# Follow API logs after background start.
docker compose -f compose.yml logs -f api

# Stop the stack while preserving local Postgres data.
docker compose -f compose.yml down

# Stop the stack and permanently delete the local Postgres data volume.
docker compose -f compose.yml down -v
```

Compose reads `.env`, `.env.local`, `apps/backend/.env`, and `apps/backend/.env.local`. Inside Compose, the API uses the in-network Postgres URL automatically and reuses it for all Postgres-backed stores. Compose exposes the API on `8080`; manual backend development usually uses `8000`.

#### Text TUI

```bash
# Start Dockerized Postgres for persistent local dogfooding.
docker compose -f compose.yml up -d postgres --wait

# Preferred TUI command for persistent dogfooding.
./scripts/text_tui.sh --mode auto --memory-mode persistent --user-id dogfood --response-model-tier quality

# Deterministic guest-mode TUI command; no provider key required.
./scripts/text_tui.sh --mode deterministic --memory-mode guest --thread-id scratch

# Raw backend TUI command.
cd apps/backend
uv run python -m opencouch_tui --mode auto --memory-mode persistent --user-id alice --thread-id s1
```

The old text CLI/REPL entrypoints are deprecated; use the TUI commands above for local dogfooding.

#### Manual web stack

Use this when you want each process in its own terminal. The fully manual stack uses port `8000` for the API because the web client defaults to `http://localhost:8000/api`.

```bash
# Terminal 1, from the repo root: Postgres for durable persistence.
docker compose -f compose.yml up -d postgres --wait

# Terminal 2: API server using the host-exposed Compose Postgres port.
cd apps/backend
OPENCOUCH_PERSISTENCE_BACKEND=postgres OPENCOUCH_MEMORY_DATABASE_URL=postgresql://opencouch:opencouch@localhost:5432/opencouch uv run uvicorn main:app --port 8000 --reload

# Terminal 3: frontend, from the repo root.
pnpm install && pnpm --dir apps/web dev
```

When finished, stop the manual processes with `Ctrl+C`, then tear down Postgres:

```bash
# Preserve local Postgres data for the next run.
docker compose -f compose.yml down

# Or permanently delete the local Postgres data volume.
docker compose -f compose.yml down -v
```

For the recommended mixed workflow, run Postgres + API in Compose and run the frontend locally:

```bash
# Terminal 1: Postgres + API on http://localhost:8080/api.
./scripts/dev_api_stack.sh

# Terminal 2: local frontend with hot reload.
NEXT_PUBLIC_API_URL=http://localhost:8080/api pnpm --dir apps/web dev
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

## Architecture

OpenCouch owns the product state machine (thread locks, routing policy, memory lifecycle, audit, persistence, and observability). The OpenAI Agents SDK Runner owns the text model/tool execution loop and the model-visible conversation history. The OpenAI Realtime voice path shares app-owned memory, crisis resources, guided exercises, audit, and session finalization seams. Every turn runs safety routing before response generation; memory writes happen in two phases — per-turn extraction and a session-end commit.

```mermaid
flowchart LR
    classDef surf fill:#64748B1A,stroke:#64748B,stroke-width:2px
    classDef app fill:#2563EB1A,stroke:#2563EB,stroke-width:2px
    classDef sdk fill:#10B9811A,stroke:#10B981,stroke-width:2px
    classDef store fill:#7C3AED1A,stroke:#7C3AED,stroke-width:2px

    SURF["**Surfaces**<br/>Web · TUI · API"]:::surf
    APP["**OpenCouch runtime**<br/>safety routing · state · memory<br/>audit · session lifecycle"]:::app
    SDK["**Agents SDK Runner**<br/>therapeutic · crisis · exercise<br/>tools · model calls · tracing"]:::sdk
    STORE[("**Postgres**<br/>state · memory · sessions<br/>crisis log · feedback")]:::store

    SURF --> APP
    APP <--> SDK
    APP <--> STORE
```

**Interfaces:** Next.js web chat · text TUI · FastAPI REST + WebSocket · OpenAI Realtime voice backend.
**Agent roster:** therapeutic (default), crisis (safety hand-off), guided-exercise (multi-turn skills).
**Tools exposed to the SDK/runtime:** memory ops · grounded lookup · crisis resources · guided-exercise skills.
**Observability:** vendor-neutral spans/events across text, memory, voice, safety, audit, and tools, with state diagnostics, structured logging, and OpenTelemetry exporters.

For the full runtime diagram, tool dispatch flow, and SDK/app state boundary, see [`apps/docs` → Architecture](https://whanyu1212.github.io/OpenCouch/) and [`apps/backend/agent/README.md`](apps/backend/agent/README.md).

---

## Project structure

This repository is a monorepo managed with `uv` and `pnpm`.

<details>
<summary><b>View repository tree</b></summary>

```text
OpenCouch/
├── apps/
│   ├── backend/                # Python backend (FastAPI, OpenAI Agents SDK)
│   │   ├── agent/              # Runtime agents, tools, state, memory, and services
│   │   │   ├── runtime/        # OpenAI text runtime, SDK sessions, persistence lifecycle
│   │   │   ├── voice/          # OpenAI Realtime voice runtime, tools, and turn finalization
│   │   │   ├── observability/  # Vendor-neutral tracing, diagnostics, logging, and OTel export
│   │   │   ├── specialists/    # Therapeutic, crisis, and guided-exercise agent roles
│   │   │   ├── tools/          # Tool implementations exposed to the SDK/runtime
│   │   │   ├── memory/         # Memory retrieval, write policy, embeddings, stores
│   │   ├── llm/                # LLM adapter protocol and OpenAI client
│   │   ├── opencouch_tui/      # Preferred interactive terminal TUI
│   │   ├── api/                # FastAPI REST + WebSocket routes
│   │   └── tests/              # 1100+ pytest unit/integration tests
│   ├── web/                    # Next.js chat application
│   └── docs/                   # Docusaurus documentation site
├── eval/                       # Trajectory runners and evaluation datasets
├── scripts/                    # Local dogfooding and utility scripts (text_tui.sh, voice_realtime_dogfood.md)
└── compose.yml                 # Local Postgres + API + web stack
```
</details>

---

## Development & validation

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

OpenCouch uses a vendor-neutral tracing layer for privacy-safe spans and events across text, memory, voice, tools, safety routing, and crisis audit. Available exporters include bounded state diagnostics, structured logging, and OpenTelemetry/OTLP.

```env
# Standard OpenTelemetry configuration for OTLP export.
OTEL_SERVICE_NAME=opencouch-agent
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_RESOURCE_ATTRIBUTES=deployment.environment=local
```

Vendor-specific LLM observability adapters are intentionally deferred until a first target is selected. The current default is to keep instrumentation app-owned and export through the neutral recorder/exporter layer.

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for dated release notes and project history.

---

## Contributing

Contributions are welcome. Because OpenCouch sits in a sensitive product space, please open an issue to discuss anything beyond a small fix or doc tweak before opening a PR — this avoids work that conflicts with safety, memory, or routing changes already in flight.

**Before opening a PR**

1. Run the relevant checks from [Development & validation](#development--validation): backend tests, web lint/build, and `pre-commit`.
2. Add tests for new service or routing behavior. Live-provider tests live under `tests/live` and are opt-in.
3. Target the `develop` branch. PR templates live in [`.github/`](.github/).

**Branch conventions**
- `feature/*` for new capabilities
- `fix/*` for bug fixes
- `refactor/*` for architectural changes
- `docs/*` for documentation updates

**Filing issues, asking questions**

- **Bugs and feature requests:** [GitHub Issues](https://github.com/whanyu1212/OpenCouch/issues).
- **Safety concerns** (a routing miss, an unsafe response, a missed crisis signal): please file a private security-style report rather than a public issue.
- **Discussion and questions:** GitHub Issues with the `question` label until a Discussions board is set up.

By contributing, you agree your contributions are licensed under the project's [AGPL-3.0 license](LICENSE).

---

## Roadmap

OpenCouch is pre-beta and currently focused on stabilizing the core chat, memory, and safety experience before expanding to more platforms.

| Horizon | Area | Focus |
|:---|:---|:---|
| ✅ **Shipped** | **Core product** | Web chat, threading, persistent/incognito sessions, memory inspection, and session feedback |
| ✅ **Shipped** | **Guided support** | 13 state-tracked coping exercises for grounding, breathing, thought work, values reflection, and related flows |
| ✅ **Shipped** | **Runtime & API** | FastAPI REST/WebSocket backend, OpenAI Agents SDK text runtime, Postgres-backed state/session persistence, crisis audit, and feedback storage |
| ✅ **Shipped** | **Observability** | Vendor-neutral tracing across runtime, memory, voice, tools, safety/audit, with state diagnostics, structured logging, and OpenTelemetry export |
| 🔜 **Next** | **Product stabilization** | Closed beta readiness, onboarding polish, reliability improvements, clearer session lifecycle, and feedback-driven UX fixes |
| 🔜 **Next** | **Voice UX stabilization** | Continue hardening the OpenAI Realtime voice path for dogfood and beta readiness, without reintroducing the legacy LiveKit worker |
| 🔜 **Next** | **Memory quality** | Background fact consolidation, dormant/obsolete memory handling, better review controls, and undo support |
| 🔜 **Next** | **Safety & evaluation** | Broader eval coverage, clinician-informed review of safety behavior, and stronger regression monitoring |
| 🧭 **Later** | **Mobile** | Native iOS app once the web and backend voice paths are stable |
| 🧭 **Later** | **More channels** | WhatsApp and Discord adapters after the core messaging abstraction is stable |
| 🧭 **Later** | **Graph memory** | Graphiti + Neo4j exploration for entity-relationship reasoning |
| 🧭 **Later** | **Vendor observability adapters** | Optional Opik, Arize, LangSmith, or similar adapters after choosing a first integration target |
| 🧭 **Later** | **Acoustic safety** | Paralinguistic crisis signals such as prosody, flatness, or distress markers |
| ⏳ **Awaiting expert** | **Clinical review** | Expert clinician audit of knowledge files, prompts, guided exercises, and safety logic |

---

## License

OpenCouch is released under the [GNU Affero General Public License v3.0](LICENSE). If you run a modified version as a network service, the AGPL requires you to make the corresponding source available to its users. See the license file for full terms.

---

<div align="center">
<sub>OpenCouch is not a substitute for professional care. If you are in crisis, please contact local emergency services or a regional crisis line.</sub>
</div>

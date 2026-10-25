# OpenCouch

A guided self-help program for anxiety, with an AI coach that helps people
practise skills instead of reassuring them. It's being rewritten from scratch
on this branch; the clinical content of the program is still open. The
architecture plan is in issue #356 and the work is tracked in Linear. The
previous implementation (an open-ended companion) is on `main`; use it as a
reference for prompts and eval scenarios, not as a design to copy.

These instructions are for every coding agent. Each app has its own
`AGENTS.md` with the rules for that code; read it before changing files there.

## Where to look

- **The task:** the Linear issue named in the branch (e.g. `hy-347`). Read it
  before starting, and keep to its scope.
- **The architecture plan:** issue #356.
- **What a PR must say:** `.github/pull_request_template.md`, including
  **Safety impact**.
- **What CI runs:** `.github/workflows/ci.yml`.

## Layout

- `apps/agent`: Python service. FastAPI + Pydantic AI, served over AG-UI.
  Features: `safety/` (the Jev check, risk policy, conversation-level risk,
  the gate that screens every turn in `turn.py`, the crisis responder, the
  vetted crisis resource catalog and the crisis audit log), `sessions/`
  (thread persistence), `beta/` (invites and requests for early access) and
  `program/` (per-user records that span threads: program position and
  worries so far). Planned: `coach/` (the agent, replacing `agent/`, and the
  reassurance redirect), and the program's state machine and content in
  `program/`. Python standards are in `apps/agent/AGENTS.md`.
- `apps/web`: Next.js frontend on `@ag-ui/client`. See `apps/web/AGENTS.md`
  before writing Next.js code.

## Commands

```bash
# Agent (run from apps/agent)
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests evals scripts
uv run pytest
# Run without API keys or Neon Auth: the test model, a fake Jev that screens
# nothing, every request treated as one local user, and no beta invite needed
OPENCOUCH_MODEL=test OPENCOUCH_FAKE_JEV=true OPENCOUCH_FAKE_AUTH=true OPENCOUCH_BETA_OPEN=true uv run opencouch-agent

# Web (run from the repo root)
pnpm lint:web && pnpm test:web && pnpm build:web
```

## Architecture rules

- **Workflow outside, limited agent autonomy inside.** Code owns the structure
  of every turn: safety screening and risk, the crisis path, reassurance
  detection, session lifecycle, program state transitions and record writes.
  The model decides what to say and which of a few safe tools to call.
- Safety screening runs on every turn, in code, before the agent. Never skip
  it. If it fails, the level never drops: from ELEVATED up the crisis
  responder replies, below it the coach is told screening didn't run, and the
  missed message is screened again on the next turn.
- Tools *request* state changes; code validates and applies them.
- The server owns conversation history and state. Client-sent AG-UI state is
  untrusted user input: never let it lower a crisis level or skip a safety step.
  A crisis check-in tap only counts on a turn where the server offered it.
- Program content (lessons, skill steps, questionnaires) is fixed, reviewed
  data in the repo. The model explains and personalises it; it never writes it
  at runtime.
- Cap model calls per turn with `UsageLimits`.

## Workflow

- One PR per Linear issue, ideally under ~400 changed lines, targeting
  `develop`, with the Linear ID in the branch name and `Refs #356`. Fill in the
  PR template, including **Safety impact**.
- Conventional commit messages (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
- Review the full diff for bugs before opening a PR (in Claude Code,
  `/code-review`).

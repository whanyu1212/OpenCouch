# OpenCouch

A guided self-help program for anxiety, with an AI coach that helps people
practise skills instead of reassuring them. It's being rewritten from scratch
on this branch; the clinical content of the program is still open. The
architecture plan is in issue #356 and the work is tracked in Linear. The
previous implementation (an open-ended companion) is on `main`; use it as a
reference for prompts and eval scenarios, not as a design to copy.

## Layout

- `apps/agent`: Python service. FastAPI + Pydantic AI, served over AG-UI.
  Features: `safety/` (screening, conversation-level risk, crisis path) and
  `sessions/` (thread persistence). Planned: `coach/` (the agent, replacing
  `agent/`, and the reassurance redirect) and `program/` (program state,
  content, practice records).
- `apps/web`: Next.js frontend on `@ag-ui/client`. See `apps/web/AGENTS.md`
  before writing Next.js code.

## Commands

```bash
# Agent (run from apps/agent)
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests evals scripts
uv run pytest
OPENCOUCH_MODEL=test uv run opencouch-agent   # run without an API key

# Web (run from the repo root)
pnpm lint:web && pnpm build:web
```

## Architecture rules

- **Workflow outside, limited agent autonomy inside.** Code owns the structure
  of every turn: safety screening and risk, the crisis path, reassurance
  detection, session lifecycle, program state transitions and record writes.
  The model decides what to say and which of a few safe tools to call.
- Safety screening runs on every turn, in code, before the agent. If it fails,
  fail toward a crisis-aware reply. Never skip it.
- Tools *request* state changes; code validates and applies them.
- The server owns conversation history and state. Client-sent AG-UI state is
  untrusted user input: never let it lower a crisis level or skip a safety step.
  A crisis check-in tap only counts on a turn where the server offered it.
- Program content (lessons, skill steps, questionnaires) is fixed, reviewed
  data in the repo. The model explains and personalises it; it never writes it
  at runtime.
- Cap model calls per turn with `UsageLimits`.

## Python standards

- **Google-style docstrings** on public modules, classes and functions
  (enforced by ruff `D` rules). Don't repeat types in `Args:`; they are in the
  signature. Private helpers need a docstring only when the reason isn't obvious.
- **Comments explain why, not what.** Label safety decisions explicitly, e.g.
  `# Fail closed: classifier errors route to a crisis-aware reply.`
- **Typed everything** (mypy strict). Pydantic models at boundaries (API, LLM
  structured output, database); dataclasses internally. No `dict[str, Any]` for
  state.
- **Organize by feature** (`safety/`, `coach/`, `program/`), with each
  package owning its models, logic and tools. Not by layer.
- **Keep units small:** aim for files under ~300 lines and functions under ~40.
  Ruff enforces a complexity limit of 10.
- **Pass dependencies explicitly** through agent `deps` or function arguments.
  No module-level globals or singletons beyond cached settings.
- **Keep prompts separate from logic**, as named constants or files, so prompt
  changes show up clearly in diffs.
- **Spell names out in full** (`crisis_assessment`, not `ca`).
- **Tests read like specs:** `test_classifier_error_routes_to_crisis_aware_reply`.
  Use Pydantic AI's `TestModel` / `FunctionModel`; never call a real model in
  the default test run. `safety/` needs full branch coverage.

## TypeScript standards

- TypeScript strict; no `any`. ESLint must pass.
- Components render from typed AG-UI messages and state; keep protocol handling
  in `src/lib/`, not in components.

## Workflow

- One PR per Linear issue, ideally under ~400 changed lines, targeting
  `develop`, with the Linear ID in the branch name and `Refs #356`. Fill in the
  PR template, including **Safety impact**.
- Conventional commit messages (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
- Run `/code-review` before opening a PR.

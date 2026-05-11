# Changelog

## 2026-05-11 — Text and Voice Dogfooding Scripts

### Local dogfooding
- Added `scripts/voice_agent.sh`, a dedicated LiveKit voice-agent launcher that starts the Dockerized Postgres service by default and then runs `agent.voice.agent`; it defaults to `start` but forwards LiveKit agent commands such as `console`, `console --text`, and `connect --room <name>`
- Added voice-script flags for common dogfooding configuration without requiring manual environment exports: `--user-id`, `--thread-id`, `--memory-mode`, `--backend`, `--database-url`, and `--no-postgres`
- Kept `scripts/cli_dogfood.sh` as the text-agent wrapper, so local users can run the text and voice agents independently while the web UI is being reworked

### LiveKit text-mode verification
- Fixed the LiveKit local text console path so typed turns go through the same OpenCouch pre-turn policy hook as spoken turns, including crisis classification, turn policy, and exercise-consent state
- Verified live text-mode box-breathing entry against Postgres: direct typed consent now grants `grounding_box_breathing`, starts `VoiceExerciseTask`, and begins from the first catalog exercise step
- Removed the duplicate post-exercise completion check-in after handoff back to the therapeutic agent

### Validation
- Focused LiveKit voice tests passed (`67 passed`)
- `scripts/voice_agent.sh --help`, unknown-option handling, shell syntax, and pre-commit passed for the new script

## 2026-05-08 — Turn Dispatch + Grounded Tool Simplification

This entry continues the graph-slimming pass by collapsing safe-turn routing
and grounded lookup into smaller LLM-primary services, with the graph kept to
true lifecycle boundaries.

### Turn-level routing
- Replaced the separate `memory_control_gate` and `grounded_lookup_gate` graph
  nodes with one `turn_dispatch` node that returns a typed route plan for
  memory control, grounded lookup, or normal therapeutic support
- Deleted the old grounded-lookup gate package and memory-control regex router,
  moving memory-control action parsing into a plain service module instead of a
  graph gate
- Updated the agent graph, public state fields, voice tool wiring, docs, and
  routing tests so tool invocation is handled at the turn-dispatch level rather
  than through scattered regex/pattern gates

### Grounded tools
- Consolidated `grounded_lookup` and `web_search` into one
  `grounded_search` execution module backed by provider-native search tools
- Added structured factual lookup preflight and structured search-grounded
  results with explicit source lists, avoiding text-marker parsing for success
  or verification status
- Added structured crisis-location classification, including a
  `location_refused` status so crisis responses respect an explicit refusal to
  share location instead of guessing or asking again

### Evals and validation
- Added turn-dispatch tool-usage eval coverage and grounded-tool quality evals
  for factual lookup, non-crisis mental-health resources, crisis resources, and
  explicit location refusal
- Reusable LLM-as-judge helpers now support rubric-based quality checks for
  grounded tool outputs
- Focused backend checks passed (`52 passed`), grounded-tool quality passed in
  scripted and live modes (`11/11` each), and pre-commit passed for the touched
  files

## 2026-05-08 — Therapeutic Dispatch Simplification + Output-State Trim

This entry covers a focused architectural pass on top of the 2026-05-07 restructure: collapsing the therapeutic dispatcher to LLM-primary policy and deleting three carrying-cost-only fields from the agent's output state. Same `refactor/agent-restructure` branch.

### Therapeutic dispatch — LLM-primary policy
- Deleted four dispatch modules totaling ~821 LOC (`fallback.py`, `guards.py`, `patterns.py`, `regex_catalog.py`) that encoded a parallel regex-based routing system layered in front of the LLM classifier; replaced with a single 109 LOC `router.py` whose only non-LLM logic is exercise-state bookkeeping (clear `exercise_state` when the LLM routes away from an active exercise; pin the original therapeutic approach when the LLM stays inside one)
- Trimmed the `dispatch/__init__.py` public surface from 57 re-exported names to 22 by dropping helpers that no longer exist; the surviving underscore-prefixed exports are the small set of internal helpers genuinely shared between `router.py` and `routing.py`
- Removed the response-style postprocessor pipeline (`_ensure_reflective_question`, `_ensure_psychoeducation_question`, `_ensure_attuned_opening`, plus the threading through `TherapeuticResponseStyleConfig` and `run_streamed_response_style`) — these were regex-based hedges against LLM output that became dead weight once the dispatcher was trusted to pick the right style
- `exercises/selection.py` shrank from 412 to 326 LOC as a follow-on cleanup

### Output-state surface — three field deletions
The agent's `AgentGraphOutputState` previously declared 8 channels; three of them did no runtime work and have been removed end-to-end (writers, schemas, public API, tests, frontend, docs):
- **`response_style_type`** — a 3-value enum (`OPERATIONAL`/`THERAPEUTIC`/`CRISIS`) that was fully derivable from `response_kind` + `response_style`, never branched on, and explicitly labeled "Deprecated style-type label retained for call-site compatibility" in the CLI; the `ResponseStyleType` enum class itself is gone, along with its entry in the checkpoint deserializer's msgpack allowlist
- **`response_style_source`** — provenance metadata (`crisis_gate`, `grounded_lookup_gate`, etc.) that was display-only with no consumers branching on its value; deletion removed three CLI display sites, one web `<Pill>` element, and the `responseStyleSource` field from the Zustand session store
- **`response_kind`** — a 2-value enum (`THERAPEUTIC`/`CRISIS`) whose information was already encoded in `crisis.level`; replaced by a single derivation at the public-output boundary, so the public `AgentOutput.response_type` is now computed once in `state_to_output` from `crisis.level >= 2` rather than written by 5 different nodes and re-read

The public `AgentOutput.response_type` field is unchanged in shape and value — only the internal computation changed from "5 writers + 1 reader" to "0 writers + 1 derivation."

### Therapeutic subgraph contract
- Tightened `TherapeuticSubgraphOutput` to its minimal load-bearing shape (`response_text`, `response_style`, `therapeutic_approach`, `exercise_state`, `diagnostics`); the explicit output schema continues to prevent the LangGraph subgraph-completion footgun where the parent re-appends an already-merged `transcript` reducer-channel
- `TherapeuticSubgraphInput` similarly trimmed; subgraph internal state (`AgentState`) is unchanged

### Validation
- Backend test suite at **1025 passed, 0 failed, 15 skipped** throughout each of the three field deletions; assertions on the deleted fields were removed (~17 across `test_state_contracts.py`, `test_crisis_gate.py`, `test_therapeutic_routing.py`, `test_grounded_lookup.py`, `test_diagnostics_reducer.py`, `test_opencouch_cli.py`, `test_session_trajectory_eval_helpers.py`, `test_grounded_lookup.py`)
- Frontend type definitions, the assistant-message Pill rendering, the debug `state` page sections, and the CLI diagnostics line were updated in the same commit so the public API and UI stay coherent
- Docs (`AgentGraph.tsx`, `StateFields.tsx`, `NodeCatalog.tsx`, `agent/README.md`) updated to reflect the trimmed state surface

## 2026-05-07 — Agent Module Restructure + Service Extraction + Latency Wins

This entry covers ~50 commits on `refactor/agent-restructure` since the 2026-05-03 entry, with a net **−1043 lines across 172 files** — a structural simplification rather than a feature push, plus one user-perceptible latency improvement and end-of-week dogfooding ergonomics.

### Module structure cleanup
- Promoted the agent into its own coherent package by moving `voice/` into `agent/voice/`, `active_session_*` into `agent/runtime/`, and `working-memory entries` under `agent/memory/`, ending the long-standing split where related code lived in sibling top-level directories
- Promoted `memory/store/` to a package with backend-specific submodules (in-memory, SQLite, Postgres) and grouped risk-gating subsystems under `agent/gates/` so safety, grounded lookup, and memory control share a discoverable home
- Flattened `services/llm/` into `llm/` after dissolving the single-purpose `services/` parent directory, and flattened other 1-file directories that were adding navigation overhead without organisational value
- Co-located prompts with their consumers (rather than in a global `prompts/` tree), dissolved facade modules that only re-exported, deleted second-tier wrappers, and removed dead delegation wrappers in `persistence.py`
- Symmetrized audit backends so `crisis_log` and `session_feedback` follow the same SQLite/Postgres backend layout as the memory store

### Service and coordinator extraction
- Extracted the `TurnExtractionCoordinator` to own background-extraction lifecycle (start, await, cancel, finalize) instead of scattering coordination across multiple call sites
- Extracted standalone services for memory control, session finalization, runtime streaming, runtime session tracking, runtime backend factories, and runtime session-state calculations, each pulled out of large composite modules into focused units with clearer single responsibilities
- Consolidated the memory write loop, prompts, and `user_controls` API into a coherent shape; extracted shared semantic-memory write primitives and a shared write executor so policy decisions and execution share the same code path

### Typed router results
- Replaced loose dict-shaped routing decisions with typed result models for the agent router, grounded-lookup router, and memory-control router, eliminating a class of "did I spell the key right?" bugs and giving downstream consumers static type information

### Performance
- Moved memory extraction **off the user-visible turn** so the assistant reply renders without waiting for semantic + procedural extraction to complete (the largest single change in this window at +1355/−361 lines), addressing the ~250–300ms median and ~600–800ms p95 `post_finalize_ms` measured during the prior latency profiling round
- Added native parallel extractor edges and parallel candidate-policy evaluation so multi-extractor turns no longer serialise unnecessarily
- Added speculative pre-fetch of turn memory in `nodes/` so the response writer rarely waits on a cold retrieval

### Local dogfooding ergonomics
- Added an upfront `get_settings()` validation that raises a clear, actionable `ValueError` when `OPENCOUCH_PERSISTENCE_BACKEND=postgres` (the default) is selected without `OPENCOUCH_MEMORY_DATABASE_URL`, replacing the previous abstract `thread_database_url is required when thread_persistence_backend='postgres'` failure that surfaced deep inside the runtime — the new message names both escape hatches (set the URL with the docker compose default, or switch to `OPENCOUCH_PERSISTENCE_BACKEND=sqlite`)
- Added `scripts/cli_dogfood.sh`, a thin wrapper that ensures the Dockerized Postgres service is healthy (`docker compose up -d postgres --wait`) before launching the CLI from `apps/backend`, forwarding any flags through `"$@"`; raw `uv run python -m opencouch_cli` invocations remain canonical for guest, deterministic, and SQLite-fallback cases that do not need Postgres
- Updated the root README CLI section to introduce the wrapper alongside the existing raw invocations, with a backlink to the Environment section and an explicit "when not to use it" callout

### Validation
- Backend test suite (62 test files) green throughout the refactor; structural changes were rename/move-only with import updates, so behaviour is unchanged
- Branch net delta confirms the cleanup intent: −1043 lines across 172 changed files, with the largest deletions concentrated in dissolved facade and wrapper modules

## 2026-05-03 — Postgres Compose Runtime + Voice UX Polish

### Local persistence and Compose
- Moved the recommended local stack to Dockerized Postgres by default through `OPENCOUCH_PERSISTENCE_BACKEND=postgres`, covering memory, LangGraph checkpoints, active-session state, crisis audit, session feedback, and LiveKit voice finalization status for API/voice worker runs inside Compose
- Kept SQLite as the compatibility default outside Compose while documenting the Postgres path as the current durable local runtime
- Removed the unsupported pgvector HNSW index on the 3072-dimension embedding column, which Postgres rejects because HNSW vector indexes are limited below that dimension count; retained the typed vector column and added schema regression coverage
- Consolidated backend and web Dockerfiles, removed stale dev/prod Dockerfile splits, and switched the Compose web service to production-mode `next build` + `next start`
- Updated the root README to call out first-run Docker slowness from image pulls, dependency installation, production web build, and voice worker warmup

### Web session UX
- Added a clearer Home action above Chat, Voice, Memory, and State on desktop and mobile, replacing the less obvious `+ New` affordance
- Replaced the always-visible setup guidance with a Getting Started dialog that explains persistent vs incognito mode, Chat vs Voice, memory review, State diagnostics, and clean session endings
- Reworked chat opening prompts into more useful structured starters, removed the unused plus button from the chat composer, and refocused the text input after assistant replies and shortcut errors
- Changed text session ending to show an inline `Session ended` card with summary details and actions for starting a new session, continuing in the thread, or reviewing memory
- Removed the unused legacy sidebar component after the app moved fully to the compact conversation shell

### Voice UX and behavior
- Added selectable LiveKit/OpenAI Realtime assistant voices and plumbed the selected voice through token metadata into the LiveKit worker
- Kept the microphone closed until the voice session is ready, making warmup/wait states explicit instead of collecting speech while the agent is still starting
- Changed voice session ending to show an options dialog instead of automatically routing the user to Chat; users can continue in Chat, start a new session, review memory, or stay on Voice
- Tightened voice exercise behavior so once the user asks for or accepts a technique, the agent starts the first step instead of repeatedly asking for confirmation
- Updated guided exercise instructions to remind users to tell the agent when they have completed body, breathing, or imagery actions the agent cannot observe

### Validation
- Web lint and production build passed after the UI cleanup
- Focused LiveKit voice tests and Postgres memory schema coverage passed during the voice and persistence changes
- Pre-commit checks passed for the touched documentation/UI files

## 2026-05-01 — Thin Nodes, Fat Services + Eval/Docs Alignment

### Backend architecture
- Thinned the LangGraph backend by extracting node-heavy logic into focused services: `load_memory_service`, `episodic_service`, `session_commit_service`, `backstops`, and `agent/safety/service`
- Kept graph behavior stable while reducing orchestration complexity across `load_memory`, `crisis_gate`, `extract_facts`, `summarize_session`, and `commit_session_memory`
- Added shared memory orchestration helpers and completed the active-session/runtime cleanup needed to support the new boundaries

### Eval and test alignment
- Fixed stale routing eval harness assumptions after the state-shape and shared-node refactors: grounded lookup now reads `grounded_lookup.query`, memory control now reads `memory_control.action`, and therapeutic routing grades `response_style` directly instead of inferring style from shared node names
- Revalidated the affected routing suites in both deterministic and hybrid modes: grounded lookup (`14/14` hybrid), memory control (`11/11` hybrid), and therapeutic routing (`54/54` hybrid)
- Full backend test suite passed (`1156 passed, 13 skipped`) and targeted routing/state-contract regression coverage passed (`226 passed`)

### Documentation
- Updated Docusaurus state-contract and architecture docs to reflect nested routing scratch fields, thin-node/service-backed ownership, and session-end summarization/commit flow
- Refreshed observability and architecture components so the docs no longer describe stale flat fields or the pre-refactor node-heavy implementation model

## 2026-04-29 — Route-Persistent Text Streaming

### Web chat and voice navigation
- Moved active text chat streaming into shared session state so an in-progress reply continues while navigating between Chat, Voice, Memory, and State instead of being cancelled by route unmounts
- Kept chat status, notices, partial assistant text, and final response metadata synchronized through the shared store so returning to Chat shows the current or completed reply
- Blocked starting a LiveKit voice session while a text reply is still in progress, matching the existing busy-session guard used for thread switching and session reset

## 2026-04-28 — Local Dev Compose Stack

### Developer workflow
- Added a root `compose.yml` that starts the backend API, LiveKit voice worker, and Next.js web UI together with `docker compose up --build`
- Added a backend development Dockerfile that keeps the Python environment outside the bind-mounted source tree, so the API can run with `uvicorn --reload` inside Compose without hiding the container virtualenv
- Configured Compose to reuse backend and web dependency caches through named volumes while keeping source files bind-mounted for local iteration

### Documentation
- Documented the one-command Compose path, service URLs, required voice environment variables, and shutdown command in the root README
- Added the text-only Compose path for API + web development without LiveKit credentials, and clarified Docker Desktop and port expectations

## 2026-04-28 — OpenAI Hybrid Prompt Stabilization

### Text and voice prompt behavior
- Refined the shared therapeutic prompt sources for support, closing, guided exercise, CBT continuity, session staging, and voice-facing response behavior so text and LiveKit voice stay aligned on the same therapeutic boundaries
- Added explicit level-1 ambiguous safety-check guidance so concerning but unclear user language asks one direct safety question without prematurely escalating to hotline, 988, emergency-services, ER, or crisis-line guidance
- Kept safety-clarification turns on the normal LLM-primary therapeutic routing path instead of forcing a broad clarifying-mode override, preserving supportive responses for non-crisis ambiguity
- Fixed guided-exercise resume handling so requests like returning to the grounding step hold or resume the current exercise instead of being treated as an exit

### Eval and memory-policy stability
- Capped prefixed LLM write-policy reasons after adding the `llm_policy` marker, preventing valid model reasons from exceeding the policy schema length limit
- Updated text, voice, behavior, and trajectory eval expectations to accept semantically valid OpenAI wording and response-style choices while keeping safety and routing assertions strict
- Expanded regression coverage for ambiguous safety checks, level-1 crisis non-escalation, exercise resume classification, and write-policy reason length bounds

### Validation
- OpenAI hybrid evals passed for crisis gate (`46/46`), therapeutic routing (`54/54`), therapeutic behavior (`19/19`), short session trajectories (`8/8`), long session trajectories (`39/39`), extraction (`33/33`), summarization (`13/13`), procedural writer (`18/18`), exercise selection (`31/31`), memory-control routing (`11/11`), and grounded lookup routing (`14/14`)
- Voice evals passed for therapeutic process (`5/5`), memory control (`6/6`), and lookup tools (`5/5`)
- Deterministic support evals passed for memory write policy (`8/8`), exercise flow (`32/32`), and exercise memory (`10/10`), with retrieval hybrid completing successfully
- Focused backend tests passed (`234 passed`) and `pre-commit run --all-files` completed successfully

## 2026-04-28 — Session Experience Refresh + LiveKit Prewarm

### Web session experience
- Reworked the setup landing screen into a responsive desktop/mobile session-start experience with persistent/incognito mode cards, persisted identity prefill, and a lightweight memory-model diagram explaining user ID, thread ID, local memory, and incognito behavior
- Added a shared conversation shell for chat and voice with a slim desktop nav rail, mobile bottom tab bar, route-aware top bars, session controls, previous-session access, and explicit end-session actions
- Refreshed the chat, voice, memory, and state pages with the updated warm clinical visual system, including welcome prompts, composer styling, voice stage controls, memory/state surfaces, and mobile-friendly chrome
- Moved text response-tier and session controls into the session pill flow so the main conversation canvas has more room on desktop and mobile

### LiveKit voice startup
- Added LiveKit worker prewarm for blocking voice assets, including Silero VAD loading and OpenCouch runtime/LLM client initialization, so the first session on a worker avoids more cold-start cost
- Added a browser-to-worker output warmup stream path that requests one first-response warmup per LiveKit session without changing the user-facing conversation flow
- Kept LiveKit voice session finalization and memory write-back aligned with the existing crisis/memory runtime while tightening docs around voice identity and thread semantics

### Documentation and validation
- Updated README, quickstart, and backend runtime docs to describe the current web session experience, LiveKit worker usage, and memory ownership behavior
- Web lint passed for the frontend changes (`pnpm --filter web lint`)
- Focused pre-commit checks passed for the touched session setup and backend router files during the UI/revert work

## 2026-04-27 — Therapeutic Subgraph Refactor + Exercise Routing Cleanup

### Therapeutic subgraph structure
- Split the therapeutic dispatcher into focused modules for constants, LLM classification, regex catalog, state-aware guards, fallback routing, and prompt construction while keeping `agent.therapeutic.dispatcher` as the public compatibility surface
- Split guided-exercise internals into dedicated modules for exercise definitions, registry/indexes, selection, step classification, state deltas, response builders, memory side effects, and node orchestration while preserving the existing `guided_exercise.py` import surface
- Split therapeutic prompt assembly into `prompting/` modules for source selection, state-context formatting, mode instructions, and builders while keeping `prompts.py` as the compatibility export
- Added shared streaming and response-delta helpers for simple therapeutic modes so supportive, reflective, clarifying, psychoeducation, closing, and technique responses use the same fallback/streaming path

### Exercise routing and response behavior
- Kept guided-exercise selection LLM-primary and option-aware so ambiguous requests can offer a small choice set instead of silently defaulting to 5-4-3-2-1 grounding
- Centralized voice-eligible exercise IDs in the exercise registry and updated LiveKit voice tasks to consume that registry instead of maintaining a duplicate list
- Tightened psychoeducation response instructions for practical tips/options requests so severity-level coping guidance stays compact and within behavior-eval length expectations
- Cleaned up stale inline comments, replaced loose inline type comments with real type annotations in guided-exercise helpers, and kept load-bearing routing/regex comments where they document false-positive boundaries

### Validation
- Full backend test suite passed (`1130 passed, 13 skipped`)
- Therapeutic routing hybrid eval passed (`54/54`)
- Therapeutic behavior hybrid eval passed (`19/19`)
- Exercise flow deterministic eval passed (`32/32`)
- Exercise memory deterministic eval passed (`10/10`)
- Exercise selection hybrid eval passed (`31/31`)
- Focused pre-commit, py_compile, and diff whitespace checks passed for the changed backend files

## 2026-04-26 — Telegram Session Rotation Hardening + Markdown Rendering

### Telegram session rotation
- Hardened rotated Telegram sessions with startup recovery for orphaned registry rows, interrupted legacy-migration state recovery, non-blocking per-chat maintenance sweeps, and active-pointer guards before reclaiming closed thread checkpoints
- Added one-shot lease/liveness retry for rotated Telegram turns so transient races between active-thread resolution and runtime execution re-resolve the active pointer before showing the maintenance message
- Fixed reclaim bookkeeping so transient reset failures remain retryable and only become stuck after repeated failures or long-aged closed sessions
- Tightened runtime active-session mutation markers so failed turns remain observable as interrupted until finalization, instead of clearing the marker on graph exceptions

### Telegram rendering
- Added safe Telegram HTML rendering for common Markdown emitted by the response writer: bold, italic, inline code, fenced code blocks, headings, and HTTP(S) links
- Split rendered Telegram replies into API-safe chunks while preserving open HTML tags across chunk boundaries
- Kept unsafe raw HTML and non-HTTP(S) links escaped so model output cannot inject Telegram parse-mode markup

### Validation
- Added runtime and Telegram regression coverage for interrupted markers, startup orphan recovery, lease retry, non-blocking sweeps, reclaim retry semantics, registry migration, and rendered HTML replies
- Full backend test suite passed (`1104 passed, 13 skipped`)
- Targeted mypy, pre-commit, py_compile, and diff whitespace checks passed for the changed backend files

## 2026-04-25 — Behavior Eval Stabilization + Extraction Eval Fixes

### Eval sweep
- Ran the deterministic and hybrid behavior eval suite with LangSmith tracing disabled to avoid quota noise during validation
- Fixed deterministic crisis classification for negated self-harm planning language so protective statements like "not planning to hurt myself" no longer escalate to crisis response
- Tightened therapeutic fallback routing for narrow reflective and clarifying cases while preserving LLM-primary routing for broader ambiguous turns
- Fixed guided-exercise selection so "completely stuck" no longer false-matches the continuum exercise through the `complete` selector
- Clarified the dispatcher closing boundary so mid-conversation "thanks, that helps" acknowledgments stay supportive, while explicit wind-down language still routes to closing
- Fixed grounded factual lookup routing so session wrap-up takeaway requests containing "today" stay inside therapeutic closing/support instead of routing to web search

### Extraction eval
- Updated `extraction_eval.py` to grade both immediate semantic writes and session-held semantic candidates, matching the current write-policy architecture where sensitive categories like `trigger` and `loss` are intentionally held
- Strengthened semantic extraction prompt examples for stable life context, bereavement, relationship-in-acknowledgment, therapist mentions, and pure-small-talk boundaries
- Added narrow deterministic semantic backstops for high-precision facts the LLM may skip: helper relationships, therapist mentions, PhD context, and perfectionism triggers
- Reclassified bare name-only acknowledgments such as "Thanks Sarah" / "ok Sarah" as skip cases because they are too ambiguous to persist safely without role or support context

### Validation
- Crisis eval passed in deterministic and hybrid modes (`46/46`)
- Therapeutic routing hybrid passed (`53/53`)
- Therapeutic behavior hybrid passed (`18/18`)
- Long session trajectory hybrid passed (`39/39`)
- Memory trajectory hybrid passed (`15/15`)
- Extraction hybrid passed (`33/33`)
- Summarization hybrid passed (`13/13`) and procedural writer hybrid passed (`18/18`)
- Targeted extractor, diagnostics, and state-contract tests passed (`67 passed`)

## 2026-04-23 — Memory Internals Streamlining + Crisis Gate Cleanup

### Memory internals
- Removed stale design artifacts and dead stubs from `agent/memory/` (`schema.yaml`, `nodes_sketch.py`, `graph_store.py`, `profile_store.py`) so the runtime package reflects only live code
- Split `agent/memory/models.py` into domain modules under `agent/memory/types/` while keeping `agent.memory.models` as a compatibility re-export surface
- Centralized shared semantic and procedural write-policy heuristics to eliminate drift between candidate building, policy decisions, and extraction-time guards
- Extracted shared hybrid retrieval scoring into `agent/memory/retrieval.py` so the in-memory and SQLite stores now share the same lexical/dense fusion logic
- Standardized substantive memory-module function docstrings on the VS Code autodocstring `Args:` / `Returns:` format for more consistent maintenance

### Memory package boundary cleanup
- Moved the always-on audit backends (`crisis_log`, `session_feedback`, and their SQLite implementations) out of `agent/memory/` into `agent/audit/` so the memory package now stays focused on retrieval, write policy, and storage concerns
- Updated runtime, API, CLI, eval, and test imports to the new `agent.audit.*` paths, fixing stale references after the package move
- Added `apps/backend/agent/memory/README.md` documenting the package boundary, functionality map, and common entry points for the memory subsystem

### Load-memory retrieval quality
- Refactored `load_memory_node` into smaller retrieval, mapping, summary, and diagnostics helpers with named retrieval constants and typed `retrieval_path` values
- Replaced the old capped semantic-store scan in diagnostics with exact namespace counts via `arecord_count(...)`, so observability no longer silently undercounts past 1000 records
- Fixed semantic retrieval so inactive facts are filtered before hybrid ranking and truncation, preventing dormant or superseded records from crowding active memory out of the top retrieval window
- Added standalone `load_memory` node coverage plus backend-parity regression tests for hybrid retrieval filtering behavior

### Crisis gate cleanup
- Moved deterministic crisis regex policy into `agent/safety/crisis_rules.py` so `crisis_gate.py` now focuses on node orchestration, schema normalization, and routing
- Tightened `CrisisAssessmentSchema` with schema-native field descriptions and simplified the node flow by removing duplicate validation layers
- Removed unused crisis-gate shadow monitoring and disagreement logging after confirming it was adding complexity without an active operational consumer
- Added direct standalone `run_crisis_gate_node(...)` tests for override routing, deterministic fallback, LLM-primary success, LLM-exception fallback, and truth-table enforcement

### CLI therapeutic theme refinement
- Updated the OpenCouch CLI Rich theme from amber-dominant tones to a calmer **sage + muted blue** palette to better match the therapeutic product tone
- Applied clearer color-role separation across `primary`, `accent`, `brand`, `panel`, `warning`, and `danger` to improve emotional hierarchy and crisis-state contrast

### Other fixes
- Fixed mid-exercise text routing to read live exercise state from `progress` instead of stale routing state, preventing exercise continuity from drifting after side turns
- Renamed `ResponseKind` to `ResponseCategory` and `CrisisOverrideKind` to `CrisisOverrideOutcome` for clearer shared model terminology

## 2026-04-22 — Therapeutic Knowledge Enrichment + Architecture Overhaul

### Knowledge enrichment
- Added CBT conversation arc template (`cbt_arc.md`) with 5-phase session shape (Orient → Identify → Examine → Shift → Ground), transition signals, and arc-level failure modes
- Added long-session guidance: second-pass routing (underlying belief exploration, cross-situational pattern work, detailed experiment design, consolidation), regulation gates, rupture detection, and last-20% pacing rule
- Added cross-session continuity guidance (`cross_session_continuity.md`) covering session bridging, noticing change over time, handling regression/setbacks, and cross-session pattern naming
- Added memory hooks to all 7 modality files describing what retrieved memory matters per therapeutic approach and how to use it
- Added continuum technique to the guided exercise catalog (13th exercise) for all-or-nothing beliefs, with 5 steps and compound keyword triggers

### Modality-specific episodic memory (Option D)
- Added typed `ModalityContext` discriminated union (CBTContext, MIContext, ACTContext, GriefContext, IPTContext, DBTContext, PFAContext) to `SessionArc` for structured therapeutic artifacts
- Session summarizer now receives an `approach_hint` (dominant approach computed from per-turn accumulation in `SessionMemoryBuffer`) and extracts approach-specific fields
- Working memory formatting renders approach context as concise suffixes: `Last session (work stress, CBT): ... [Thought: I'm going to get fired; Action step: speak up in one meeting]`
- Load memory node passes approach fields through to episodic working memory entries

### Architecture: rename mode → response_style, modality → therapeutic_approach
- Renamed `TherapeuticMode` → `TherapeuticResponseStyle` and `TherapeuticModality` → `TherapeuticApproach` across models, state, dispatcher, prompts, API, CLI, tests, eval, and docs
- `RoutingState` fields: `mode` → `response_style`, `mode_source` → `response_style_source`, `mode_type` → `response_style_type`, `modality` → `therapeutic_approach`
- `AgentOutput`, `Message`, `ChatResponse`, `MessageResponse` fields updated accordingly
- `SessionArc`: `modality_used` → `approach_used`, `modality_context` → `approach_context`

### Architecture: technique response style
- Added `technique` as the 7th response style — when active, the therapeutic approach drives the response shape directly (e.g., CBT Socratic questioning rhythm overrides generic reflective instructions)
- Resolves the mode-vs-modality conflict discovered during dogfooding where `reflective` mode instructions overrode CBT arc guidance
- Dispatcher LLM prompt describes technique routing signals: active thought examination, evidence evaluation, values exploration, prediction testing
- Falls back to supportive when no approach is active

### Prompt system reorganization
- Moved `knowledge/` (repo root) → `agent/prompts/sources/` (co-located with prompt assembly code)
- Deduplicated shared helpers: `CORE_SOURCES`, `compose_sources()`, `format_recent_history()` extracted into `agent/prompts/__init__.py` — single source of truth for both `crisis.py` and `therapeutic/prompts.py`
- Dockerfile simplified (removed separate `COPY knowledge` line)
- CI workflow simplified (removed `knowledge/**` path trigger)

### Bug fixes
- Fixed mid-exercise psychoeducation clearing exercise state — asking "how does grounding work?" during a grounding exercise no longer kills the exercise and restarts from scratch
- Added empty-store short-circuit in `load_memory_node` — skips the embedding API call (~100-200ms) when semantic and episodic stores are both empty

### Tooling
- Added `scripts/inspect_memory.py` for inspecting the SQLite memory store during dogfooding (`--user`, `--namespace`, `--all-users`, `--raw`)
- Added `eval/dogfood/` directory for manual dogfood transcripts (gitignored)

## 2026-04-20 — Memory Robustness + CLI Visual Redesign

### Memory robustness fixes
- Replaced ASCII-only tokenizer (`\b[a-z0-9]+\b`) with Unicode-aware pattern (`\b\w+\b`) plus CJK character-splitting post-processor — non-English text now produces meaningful token sets for dedup and retrieval instead of empty sets
- Extended CJK regex to cover astral-plane Extension B through H ranges (U+20000–U+323AF) for rare characters in names and classical texts
- Added procedural rule cap (`MAX_ACTIVE_RULES = 20`) with oldest-first eviction — excess rules are archived with `superseded_by = "eviction"` to prevent unbounded system prompt inflation
- Consolidated duplicated safety-conflict markers (`_PROCEDURAL_SAFETY_CONFLICT_MARKERS`, `_PROCEDURAL_TURN_SCOPED_MARKERS`, `_PROCEDURAL_EXPLICIT_REQUEST_MARKERS`) and the shared `_contains_any` helper into a single `agent/memory/constants.py` module, eliminating copy-paste drift risk between `write_policy.py` and `candidates.py`
- Added `aput_batch` method to the `MemoryStore` protocol and both implementations (in-memory loop, SQLite single-transaction with `BEGIN`/`COMMIT`/`ROLLBACK`) for future atomic multi-record writes
- Implemented episodic retrieval date filter (`max_age_days=30`) on `asearch_similar` — old session arcs are excluded from query-based retrieval while `alatest()` catch-up remains unfiltered; fixed Z vs +00:00 timestamp format mismatch in both stores
- Enriched semantic working memory entries with full SPO triple (category, subject, predicate, object) so the response LLM sees structured context like `[relationship] User WORRIES_ABOUT work — 'my boss is terrible'` instead of just the raw evidence quote
- Replaced silent `"local-default"` / `"unknown"` owner_id fallback across 6 node files with a shared `resolve_owner_id` helper that raises `ValueError` if neither `user_id` nor `session_id` is set, preventing silent cross-user memory namespace contamination

### CLI visual redesign
- Redesigned the CLI from the ground up with a "Midnight Journal" aesthetic — warm amber/sage/cream color palette replacing the old cold teal/green theme
- Replaced the double-box header with a Unicode block-art wordmark for "OpenCouch" using half-block characters, with "CLI" as a muted accent tag below
- Replaced full Rich panel boxes on informational messages with a minimal left-bar (`│`) indicator for quieter visual weight
- Lowercased all panel titles (diagnostics, session context, memory status, threads, history, etc.) for a calmer typographic hierarchy
- Switched all data tables from `SIMPLE_HEAVY` to `SIMPLE` box style with `hint`-colored headers
- Replaced the noisy `❯ you :` prompt with a softer `· you` prompt with breathing whitespace
- Added `Rule` separators between visual sections instead of nested panels

### Next.js web UI
- Updated the memory sidebar `CATEGORY_COLORS` map to match the current backend `SemanticCategory` values (`loss`, `trigger`, `goal`, `context` replacing stale `identity`, `emotional_pattern`, `life_event`, `health`, `belief`)
- Added per-record memory deletion on the Memory page — "forget" button on each fact, session arc, and procedural rule card with two-click confirmation (mirrors the CLI's `/memory forget` command)

## 2026-04-20 — Memory Rewrite + Response Model Tiers

### CLI improvements
- Redesigned the Rich terminal theme from cool teal to a warm amber/cream palette for better readability and softer aesthetics
- Replaced the blocking status display with a live Rich spinner that shows human-readable stage labels during graph execution
- Added `--response-model-tier` flag and `/response-tier <fast|quality>` slash command for switching response quality in-session
- Exposed active session-end holds (semantic and procedural) in the `/memory` debug inspector
- Added response tier display to session state and `/debug` output

### Next.js web UI
- Added an explicit `End Session` button in the sidebar that triggers the session-end commit seam and shows a summary card with themes and mood arc
- Added response model tier selector (fast/quality) to the sidebar, disabled during voice sessions
- Memory page now uses typed interfaces (`MemoryFact`, `MemorySession`, `MemoryRule`) instead of untyped `Record<string, unknown>` for safer rendering
- Memory page auto-refreshes after writes via a reactive `memoryRefreshVersion` counter in the Zustand store
- Added session feedback count to the memory overview tab
- Memory panel fact categories updated to match the rewritten schema (`loss`, `trigger`, `goal`, `context` instead of older categories)
- Sidebar locks identity fields (user/thread) while a persistent session is active to prevent accidental mid-session switches
- Session status polling checks the backend every 30s to reflect session liveness
- WebSocket stream now passes `response_model_tier` to the backend for per-turn tier selection

### Shared status label standardization
- Moved the human-readable pipeline stage labels (`loading memory`, `safety check`, etc.) into a shared `STAGE_LABELS` map in `agent/models.py`
- Both the CLI and WebSocket API now emit identical user-friendly labels instead of raw node names
- Web UI no longer displays internal identifiers like `crisis_gate` or `load_memory`

### Memory system rewrite
- Reworked memory writing around a policy-first flow: turn extraction now produces candidates first, then deterministic policy decides whether to commit immediately, hold for session end, require repetition, or drop
- Narrowed immediate semantic writes to lower-risk factual memory and moved more sensitive or interpretive content toward session-end commit or repetition-gated promotion
- Kept explicit procedural instructions effective on the same turn, while buffering more implicit stylistic preferences until they have stronger evidence
- Made session end the shared durable seam for episodic memory plus held semantic and procedural promotion, instead of relying mostly on per-turn writes
- Added reconciliation and supersession so stale semantic facts can go dormant, weaker overlapping writes do not pile up, and procedural replacements keep an audit trail
- Unified session-end handling across text, web, API shutdown, timeout, CLI shutdown, and voice disconnect so the same commit path runs regardless of how a session ends
- Persisted active session buffers in runtime-owned state so held candidates survive restarts before the session is finalized

### Memory evals and observability
- Added write-policy coverage and trajectory eval coverage for immediate writes, delayed writes, repetition-gated promotion, and real store-state assertions instead of diagnostics-only counting
- Tightened active-memory readouts so web, API, and CLI surfaces show active semantic facts and active procedural rules rather than stale superseded rows
- Brought the web text UI onto the new explicit session-end flow with a real `End Session` action instead of relying only on timeout or backend shutdown

### Response model tiers
- Added a split between pinned control-plane LLM usage and selectable response-writer LLM usage
- Kept safety, routing, memory extraction, summarization, helper calls, and voice behavior pinned to the control model for eval stability
- Routed only final therapeutic prose generation through a user-facing `fast` versus `quality` response tier
- Added text-chat tier selection to the web UI with product labels instead of raw model ids
- Added the same tier support to the CLI via `--response-model-tier` and `/response-tier <fast|quality>`

### Documentation
- Updated the Docusaurus memory and architecture docs to reflect the rewritten memory model: candidate extraction, deterministic write policy, persisted session holds, session-end promotion, and the shared session-end seam
- Updated the voice docs and quickstart guide to match the current disconnect-time memory write-back behavior and experimental Realtime limitations
- Cleaned up observability docs to match the lighter current CLI flow: reply-first rendering, live status spinner, on-demand inspection commands, and the new memory-policy diagnostics counters
- Flattened the `Agent Graph` docs navigation so it behaves like a category with an `Overview` child page instead of a content-bearing expander
- Increased the Docusaurus docs typography slightly at the theme level for better baseline readability

## 2026-04-19 — Realtime Voice Rewrite + UX Stabilization

### Backend voice bridge
- Rewrote `apps/backend/voice/realtime.py` around the documented OpenAI GA Realtime session model instead of the older custom loop
- Standardized the live voice path on `gpt-realtime` for speech-to-speech and `gpt-4o-transcribe` for input transcription
- Kept the Realtime prompt bounded and memory-backed so the session stays within instruction limits
- Added explicit support for configurable Realtime voices, defaulted the assistant voice to `cedar`, and validated supported voices at the API layer
- Added optional transcription language selection plus a backend transcription prompt to improve recognition of names, numbers, acronyms, and filler words
- Tightened the websocket contract around `ready`, `audio`, `transcript`, `interrupted`, `truncated`, and `error`

### Voice UX
- Rebuilt the standalone `test_page.html` into a proper audio harness with server VAD, interruption handling, truncation sync, and configurable voice/language
- Tuned interruption feel with smaller mic chunks, lower VAD threshold, and client-side local ducking before server truncation lands
- Brought the Next.js voice tab onto the same audio pipeline as the working test harness
- Re-enabled transcript history keyed by Realtime item ID so updates replace in place instead of duplicating turns
- Removed live captions after testing and kept only transcript history, labeled as approximate
- Added voice and language selectors to the web UI, disabled while connected to match Realtime constraints
- Labeled the voice tab as `experimental` and explicitly documented that it is speech-only today and does not yet expose agentic or autonomous actions

### Docs
- Rewrote the voice docs to reflect the current Realtime implementation instead of the older agentic voice design
- Documented the current voice configuration, latency/VAD behavior, and transcript limitations in the Docusaurus site
- Flattened the docs route so voice now lives directly at `/docs/voice`

### Deployment
- Added an initial GitHub Actions deployment workflow for the backend API targeting Google Cloud Run on `develop` and `main`
- Added a backend `Dockerfile`, a Cloud Run-friendly server entrypoint, and env-driven CORS configuration for deployment-time origin control
- Standardized the deployment path around Workload Identity Federation, Artifact Registry, Cloud Run, and Secret Manager instead of a long-lived service account key
- Documented the required GCP/GitHub setup and the current deployment tradeoffs for learning and internal operations
- Marked the current Cloud Run backend path as suitable for close-beta preview and internal testing, but not yet fully production-ready because persistence, final frontend CORS allowlists, and broader rollout hardening still need to be completed

### LLM configuration
- Removed explicit temperature configuration from the backend LLM interface, provider adapters, and call sites across routing, extraction, summarization, crisis handling, and therapeutic response writing
- Updated test doubles and eval notes to match the simplified interface and remove stale assumptions about temperature-driven behavior
- The rationale is to rely on the model providers' default generation behavior rather than hard-coding older sampling heuristics; newer hosted models are increasingly tuned around their default harness settings, and carrying per-call temperature overrides was adding interface noise without a clear product benefit

## 2026-04-18 — Voice Session Tab Persistence

### Web UI — voice session
- Fixed voice chat sessions being lost on in-app tab switches (Voice → Chat → Memory → back)
- Lifted WebSocket, AudioContext, MediaStream, and ScriptProcessorNode refs into module-level singletons so they survive component unmounts
- Added reactive voice state (connected, speaking, transcripts, error) to Zustand store for cross-page UI updates
- Added generation counter to scope `onclose`/`onended` callbacks — prevents stale sessions from tearing down newer ones
- Fixed AudioContext leak when microphone permission is denied
- Moved `visibilitychange` listener to AppShell so AudioContext resume works even when the voice page is not mounted
- Added pulsing "live" indicator in sidebar when a voice session is active on another page

### Housekeeping
- Scoped root `.gitignore` `lib/` rule to `/lib/` — was unintentionally shadowing `apps/web/src/lib/`
- Started tracking `CHANGELOG.md` in version control

## 2026-04-18 — Session Trajectory Eval + Safety Hardening

### Eval infrastructure
- Added unified session trajectory eval runner (`eval/runners/session_trajectory_eval.py`) supporting both inline-expect and checkpoint dataset schemas
- Added 21 long-trajectory cases covering modality selection, boundary enforcement, closing, venting, safety pivots, and mode transitions
- Added `--concurrency`, `--case`, `--verbose` flags for faster hybrid runs and single-case debugging
- Added 4 boundary enforcement cases: medication advice, repeated diagnosis pushing, legal advice, dependency framing
- Updated eval README to document unified runner and all flags

### Crisis gate
- Refactored to LLM-primary architecture — LLM classifies all non-override messages, regex is fallback only
- Fixed override precedence: imminent risk now checked before idiomatic-safe
- Scoped idiomatic-safe matching to current message only (prior idiom cannot suppress current threat)
- Added guard: idiomatic-safe skipped when clear self-harm or ambiguous patterns also match
- Added shadow monitoring: deterministic result logged alongside LLM for drift detection
- Hardened classifier prompt: conversation fencing, anti-injection instructions, adversarial examples (negation, quoted speech, sarcasm), confidence constraints
- Sanitized classifier reason output (length cap, control char removal, injection fence)
- Enforced strict level-to-flag truth table in normalize_crisis_assessment
- Tightened denial regex: requires explicit self-harm object after "not thinking about"
- Fixed denial false-positive: "no...ending it" now requires negation-of-intent phrasing
- Fixed post-safety-check gap: DISTRESS_PATTERNS now checked alongside AMBIGUOUS_PATTERNS
- Narrowed safety-check detection phrases to reduce false-context risk

### Therapeutic dispatcher
- Refactored to LLM-primary architecture — LLM handles all mode + modality classification
- Removed context-blind regex fast paths (reflective, confusion, closing) from primary path
- Demoted broad patterns to regex fallback only (kept narrowest forms: "why do I keep", "huh?", "what?")
- Added LLM-based mid-exercise exit detection — natural exit without regex traps
- Mid-exercise clarification preserves exercise state (repair turns don't exit)
- Tightened exercise exit patterns to unambiguous opt-out only
- Sharpened ACT vs CBT distinction in classifier prompt with concrete user-language examples
- Added exercise context injection into LLM prompt for continuation/exit decisions

### Agent changes
- Added OpenAI embedding provider (`text-embedding-3-large`) as default, Gemini as fallback
- Added extraction precision guard for early-session emerging patterns and negative self-beliefs
- Added context-aware exercise selection using recent conversation history
- Fixed modality persistence across exercise turns via routing merge reducer

### Eval coverage expansion
- Added 4 multi-turn crisis trajectory cases: gradual escalation, de-escalation after safety check, oscillating signals, post-crisis recovery (21 → 25 long trajectory cases)
- Added 7 small-talk gate boundary cases to extraction dataset: name-in-thanks, therapist mention, bare acknowledgments, 40-char boundary (26 → 33 extraction cases)
- Added deterministic fallback quality eval case: checks banned soul.md phrases in hardcoded responses (8 → 9 short trajectory cases)
- Fixed dotenv loading in retrieval eval runner so API keys from .env files are picked up
- Verified OpenAI embedding provider end-to-end: hybrid RRF achieves 14/17 recall@5 vs 6/17 token-only

### Deterministic fallback quality
- Fixed supportive fallback: removed "Thank you for sharing" and "I want you to know" (banned by soul.md)
- Fixed closing fallback: replaced "Thanks for sharing" with voice-compliant alternative

### Knowledge
- `soul.md`: added therapeutic grounding, cultural sensitivity, repair after missteps, boundary-setting voice, expanded antipatterns
- `identity.md`: full rewrite with product philosophy, capabilities, limits
- `boundaries.md`: expanded with redirection patterns, dependency framing, escalation guidance

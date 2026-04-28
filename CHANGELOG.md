# Changelog

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

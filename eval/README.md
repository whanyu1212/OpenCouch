# OpenCouch Eval Harness

This directory is being rebuilt from scratch.

The shared runner code lives in `eval/runners/base.py`. It intentionally owns
only the generic mechanics:

- load a JSON dataset from a top-level list or `{ "cases": [...] }`
- run cases asynchronously
- time each case
- capture unexpected exceptions as case errors
- aggregate pass/fail/error counts
- print a compact summary
- optionally write the full JSON summary

Domain evaluators should own their own case schema, app/runtime setup, model
configuration, and grading rules. Keep the base class thin; add shared helpers
beside it only when multiple evaluators need the same behavior.

## LLM judges

Reusable LLM-as-judge code lives in `eval/judges/`.

- `BaseLLMJudge`: abstract base for concrete judges. It owns prompt execution
  and combining deterministic failures with structured judge verdicts.
- `JudgeVerdict`: common verdict schema with `passed`, `score`, `reasoning`,
  `strengths`, `failures`, and `safety_concerns`.
- `RubricLLMJudge`: generic rubric judge for artifacts that can be described as
  input, output, hard failures, and qualitative dimensions.

Evaluators should still run hard checks first for state machines, routing,
schema contracts, and invariants. Use LLM judges for qualitative questions such
as pacing, coherence, adaptation, and therapeutic usefulness.

## Evaluators

Evals are split by purpose:

- `therapeutic_contract_eval.py`: small CI-safe checks for graph boundary,
  routing contract, state transitions, and expected hard failures.
- `therapeutic_behavior_eval.py`: broader routing, exercise selection, and
  lifecycle behavior cases. Defaults to scripted mode; `--mode live` uses the
  configured LLM clients.
- `therapeutic_quality_eval.py`: response-quality checks with hard rubrics for
  concision, concrete exercise guidance, and avoiding menu-style fallbacks.
  Defaults to scripted mode; `--mode live` uses the configured LLM clients.
- `therapeutic_exercise_trajectory_eval.py`: multi-turn guided-exercise
  trajectory checks. It always applies hard state/progression checks and can
  add an LLM judge with `--judge-mode live`.
- `therapeutic_path_trajectory_eval.py`: parent-graph therapeutic trajectory
  checks for the normal non-crisis path through turn dispatch, memory loading,
  therapeutic subgraph execution, guided-exercise continuity, cross-branch
  interruptions, and finalization.
- `crisis_topology_eval.py`: CI-safe parent-graph checks for the LLM-only
  crisis gate, crisis/non-crisis branch isolation, and visible failure when the
  classifier LLM is unavailable or fails.
- `crisis_node_eval.py`: standalone crisis-node contract checks for crisis
  gate routing, resource lookup state, response quality, and full crisis-log
  payloads with mocked input state.
- `crisis_classifier_quality_eval.py`: direct `CrisisRiskService` classifier
  checks. Defaults to scripted mode; `--mode live` evaluates the configured
  control model on level 0/1/2/3 and edge-case safety cases.
- `crisis_branch_quality_eval.py`: full crisis-branch response-quality checks
  with controlled classifier verdicts and controlled resource lookup results.
  `--mode live` uses the configured model for crisis response text, and
  `--judge-mode live` adds an LLM judge.
- `turn_dispatch_eval.py`: standalone safe-turn dispatch checks for
  therapeutic, grounded lookup, and memory-control routing. Defaults to
  scripted mode; `--mode live` evaluates the configured control model.
- `memory_control_node_eval.py`: standalone memory-control node contracts with
  seeded semantic, episodic, and procedural memory fixtures. It verifies
  listing, status, recall toggles, preference saves, deletion confirmation, and
  durable store changes.
- `memory_control_trajectory_eval.py`: multi-turn memory-control trajectories
  that run through turn dispatch and memory-control/load-memory branches. It
  checks memory-admin boundaries, pending deletion lifecycle, and whether saved
  procedural settings are visible to later memory loads.
- `tool_usage_eval.py`: parent-graph checks for turn-level dispatch and
  grounded-search tool invocation. It verifies that factual lookup,
  memory-control, therapeutic, and crisis-resource routes call only the
  expected tools.
- `grounded_tool_quality_eval.py`: output-quality checks for grounded factual
  lookup and crisis-resource lookup. It applies hard source/actionability
  checks and can add an LLM judge with `--judge-mode live`.
- `agent_session_trajectory_eval.py`: full parent-graph multi-turn session
  trajectories across therapeutic, grounded lookup, memory-control, and crisis
  branches. Defaults to scripted mode and can add a live session judge.
- `text_agent_harness_trajectory_eval.py`: full text-agent harness trajectories
  over `PersistentAgentRuntime`, covering production runtime wiring, routing,
  tools, memory writes, streaming, crisis interruption/logging, failure
  surfacing, and lifecycle state. Defaults to Postgres; use `--backend sqlite`
  for fallback compatibility coverage.
- `runtime_persistence_trajectory_eval.py`: Postgres-first
  `PersistentAgentRuntime` trajectory checks for checkpoint resume, memory
  extraction durability, active-session liveness, streaming persistence, crisis
  logs, feedback, and incognito isolation. Defaults to Postgres; use
  `--backend sqlite` only for fallback compatibility coverage.
- `runtime_recovery_trajectory_eval.py`: Postgres-first runtime recovery checks
  for thread-lock serialization, cross-thread isolation, interrupted mutation
  recovery, rotation-required leases, foreign mutation markers, and
  auto-finalization exclusions.
- `text_surface_runtime_eval.py`: text API, WebSocket, and CLI surface checks
  against the real persistent runtime, covering `/api/chat`, `/api/chat/stream`,
  history, end-session feedback, memory status, CLI `/end` finalization, and
  explicit failure/recovery contracts.
- `memory_write_policy_eval.py`: direct semantic/procedural write-policy checks
  for LLM-primary decisions, hard safety guards, and visible failure when the
  policy LLM is unavailable.
- `memory_extraction_quality_eval.py`: direct semantic/procedural extractor
  quality checks. Defaults to scripted mode for evaluator mechanics; `--mode
  live` evaluates the configured control model on precision-first extraction
  cases.
- `runtime_stress_eval.py`: manual long-session stress checks over the
  persistent runtime. This reports turn timing and verifies transcript/session
  growth without making normal CI evals slow.

```bash
apps/backend/.venv/bin/python -m eval.runners.therapeutic_contract_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_behavior_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_quality_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_exercise_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_path_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.crisis_node_eval
apps/backend/.venv/bin/python -m eval.runners.crisis_topology_eval
apps/backend/.venv/bin/python -m eval.runners.crisis_classifier_quality_eval
apps/backend/.venv/bin/python -m eval.runners.crisis_classifier_quality_eval --dataset eval/datasets/crisis/classifier_ambiguity_v1.json
apps/backend/.venv/bin/python -m eval.runners.crisis_branch_quality_eval
apps/backend/.venv/bin/python -m eval.runners.turn_dispatch_eval
apps/backend/.venv/bin/python -m eval.runners.turn_dispatch_eval --dataset eval/datasets/turn_dispatch/routing_quality_v1.json
apps/backend/.venv/bin/python -m eval.runners.memory_control_node_eval
apps/backend/.venv/bin/python -m eval.runners.memory_control_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.tool_usage_eval
apps/backend/.venv/bin/python -m eval.runners.grounded_tool_quality_eval
apps/backend/.venv/bin/python -m eval.runners.agent_session_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.text_agent_harness_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.text_agent_harness_trajectory_eval --backend sqlite
apps/backend/.venv/bin/python -m eval.runners.text_agent_harness_trajectory_eval --mode live --judge-mode live
apps/backend/.venv/bin/python -m eval.runners.runtime_persistence_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.runtime_persistence_trajectory_eval --backend sqlite
apps/backend/.venv/bin/python -m eval.runners.runtime_recovery_trajectory_eval
apps/backend/.venv/bin/python -m eval.runners.text_surface_runtime_eval
apps/backend/.venv/bin/python -m eval.runners.memory_write_policy_eval
apps/backend/.venv/bin/python -m eval.runners.memory_extraction_quality_eval
apps/backend/.venv/bin/python -m eval.runners.runtime_stress_eval
apps/backend/.venv/bin/python -m eval.runners.runtime_persistence_trajectory_eval --dataset eval/datasets/runtime/live_session_trajectory_v1.json --mode live --judge-mode live
```

Useful flags:

- `--plain`: disable Rich output.
- `--json-output eval/reports/therapeutic_contract.json`: write the full result
  payload.
- `--dataset eval/datasets/therapeutic/contract_v1.json`: override the dataset.
- `--mode live`: run behavior or quality cases with configured provider-backed
  LLM clients.
- `--judge-mode live`: run exercise trajectory, crisis node, crisis branch, or
  grounded-tool LLM judges. Memory-control trajectory evals also support this
  for response-quality judging.
- `--backend postgres|sqlite`: select the persistence backend for runtime
  trajectory evals. Postgres is the primary application backend and the
  default.
- `--dataset eval/datasets/runtime/live_session_trajectory_v1.json --mode live
  --judge-mode live`: run the small live LLM session suite over the primary
  runtime persistence backend.

Current coverage:

- non-exercise response routing through the shared response node
- guided exercise start
- guided exercise step advance
- guided exercise hold and stuck behavior
- guided exercise exit and state clearing
- guided exercise completion and state clearing
- clarifying side turns that preserve active exercise continuity
- active exercise clearing when dispatch leaves the exercise flow
- expected hard failures when the control LLM is missing or returns an
  unavailable exercise
- LLM-only crisis-gate topology, including no deterministic fallback on missing
  or failing classifier LLM
- crisis classifier quality for ordinary distress, ambiguous distress, explicit
  ideation, imminent risk, third-party crisis text, post-safety denial,
  history-dependent references, and prompt-injection attempts
- crisis classifier ambiguity boundaries for false-positive idioms, passive
  death wishes, intrusive thoughts, sarcasm, safety denial, and context-dependent
  references
- full crisis-branch response quality across found resources, missing location,
  location refusal, no verified lookup results, imminent means, and ideation
  without a stated plan
- standalone crisis node contracts for crisis-gate routing, resource lookup
  state, response quality, response-LLM selection, and audit-log payloads
- standalone turn-dispatch contracts for therapeutic routing, grounded lookup
  query creation, memory-control actions, pending-action clearing, and visible
  invalid-output failures
- turn-dispatch routing quality for memory-adjacent support, source-backed
  lookup, non-crisis resources, mixed-intent turns, and pending-action ambiguity
- standalone memory-control node contracts using manually seeded realistic
  semantic facts, session summaries, procedural rules, and proactive-recall
  state
- multi-turn memory-control trajectories for inspect-vs-support boundaries,
  deletion confirmation, abandoned deletion, preference saves, recall toggles,
  and ambiguous "remember" wording
- broader scripted behavior cases for exercise selection and state changes
- hard response-quality checks for concise, concrete, non-menu output
- multi-turn exercise trajectories with optional LLM-as-judge scoring
- parent-graph therapeutic trajectories across memory-conditioned support,
  guided exercise start/continue, exercise exit, turn finalization, crisis
  interruption, grounded lookup side-trips, and memory-control interruptions
- narrowed subgraph output keys
- parent-graph turn dispatch to therapeutic, grounded lookup, memory control,
  and crisis branches
- grounded factual lookup and crisis resource tools invoked exactly on their
  intended routes
- grounded factual lookup answers stay source-backed, concise, and scoped to
  verifiable facts
- mental-health-adjacent lookup cases for reading resources, psychoeducation,
  and non-crisis support directories
- crisis-resource outputs contain location-appropriate, actionable contact
  details without guessed resources
- crisis-resource lookup respects explicit location refusal without guessing
  localized resources
- full parent-graph session trajectories across branch transitions, including
  exercise interruption, lookup side-trips, memory-control pending actions,
  crisis logging, and later safe follow-up turns
- cross-branch interruption matrix coverage for exercise ↔ memory-control,
  pending memory deletion → therapeutic/crisis, grounded lookup → crisis,
  crisis → grounded lookup, and interrupted lookup recovery
- full text-agent harness trajectories over `PersistentAgentRuntime`, including
  support-memory recall, exercise/lookup/streaming continuity, crisis
  interruption and per-session audit logging, preference saves, mental-health
  resource lookup, ambiguous non-crisis distress, explicit tool failure
  surfacing, and live smoke coverage with optional LLM-as-judge grading
- Postgres-first runtime persistence trajectories for checkpoint resume,
  background extraction drain, cross-thread/user memory scoping, streaming
  single-write behavior, session finalization cleanup, crisis-log persistence,
  feedback persistence, and incognito non-persistence
- runtime recovery trajectories for same-thread concurrency serialization,
  cross-thread isolation, failed-turn interruption, foreign mutation markers,
  rotation-required session leases, explicit recovery, and auto-finalization
  exclusions
- text API, WebSocket, and CLI surface trajectories that verify production
  wiring into the persistent runtime, including feedback writes, end-session
  cleanup, explicit runtime failure surfacing, interrupted-session blocking,
  and recovery after `/end`
- direct memory write-policy decisions for semantic and procedural candidates,
  including LLM-primary paths, local hard safety guards, and no silent fallback
  on policy-LLM failure
- manual runtime stress coverage for long scripted sessions over the persistent
  backend
- a small live LLM Postgres session dataset for support and guided-exercise
  trajectories with optional LLM-as-judge grading

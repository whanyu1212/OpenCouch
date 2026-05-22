# Eval README

This directory contains deterministic evaluation assets for OpenCouch routing, runtime behavior, and crisis-response flows.

## Purpose

The eval suite helps answer four questions:

1. **Did the runtime route to the right path?**
2. **Did it select the right specialist agent?**
3. **Did it call the right tool, preserve state correctly, and apply the expected side effects?**
4. **Did crisis-specific helpers produce safe and grounded outputs?**

Most routing and behavior evals are designed to run without a live provider by using scripted LLM and SDK responses.

## Datasets

### `datasets/routing_matrix.jsonl`
Baseline single-turn routing coverage.

Covers:
- safe therapeutic turns
- memory control turns
- grounded lookup turns
- guided exercise start / continue
- crisis clarification
- crisis response

Use this file to verify the main route/runtime-mode backbone.

### `datasets/routing_boundaries.jsonl`
Boundary and precedence coverage for ambiguous inputs.

Covers:
- metaphorical distress staying non-crisis
- explicit vs implicit memory-reference boundaries
- crisis overriding grounded lookup
- grounded lookup preserving guided-exercise state

Use this file to catch false positives and route-priority regressions.

### `datasets/multiturn_routing.jsonl`
Multiturn routing and state-preservation coverage.

Covers:
- guided exercise → grounded lookup → resume
- memory cancel → safe therapeutic follow-up
- crisis clarification → crisis escalation
- therapeutic → crisis switch
- grounded lookup → therapeutic follow-up
- guided exercise → memory control → resume
- therapeutic → grounded lookup switch
- therapeutic → guided exercise switch
- pending memory action preservation across safe / grounded / crisis side turns
- crisis de-escalation back to therapeutic flow
- repeated high-risk follow-up consistency
- guided exercise interrupted by crisis → explicit resume

Use this file to verify specialist switching and resume behavior across turns.

### `datasets/behavior_matrix.jsonl`
Behavioral and side-effect coverage for runtime contracts.

Covers:
- grounded lookup success
- grounded lookup missing-tool fallback
- guided exercise step advancement
- crisis no-verified-resources behavior
- memory deletion confirmation side effects
- crisis clarification without resource lookup
- crisis clarification with location still avoiding resource lookup
- guided exercise preserve-without-advance
- memory missing-tool safety
- proactive recall enable / disable side effects
- save-preference and forget-by-query memory-tool execution

Use this file when validating state transitions, diagnostics, and response constraints.

### `datasets/trajectory_memory_modes.jsonl`
Longer trajectory coverage for durable-memory mode behavior.

Covers:
- persistent mode with matching durable memory available on the first turn
- persistent mode retaining seeded durable memory across later turns in the same eval case
- persistent "memory-light" baselines with no seed or no useful matching seed
- incognito mode suppressing durable-memory recall even when matching memory exists
- incognito mode preventing seeded durable memory from leaking into later turns

Use this file to compare persistent vs incognito memory behavior over short trajectories rather than single-turn routing alone.

### `datasets/trajectory_interruptions.jsonl`
Trajectory coverage for interruption, recovery, and resume behavior across memory modes.

Covers:
- guided exercise interrupted by crisis, then explicit resume
- crisis de-escalation back to safe therapeutic flow
- guided exercise preserved across a grounded side turn, then resumed
- recovery / relapse sequences that return from safe therapeutic flow to crisis response
- persistent-vs-incognito recall contrast on the non-crisis turns inside those trajectories

Use this file to verify that active-flow continuity and crisis precedence remain correct while memory mode still controls durable recall.

### `datasets/trajectory_endurance.jsonl`
Longer 5-turn endurance trajectories across memory modes.

Covers:
- incognito no-leak endurance across repeated named-entity turns
- persistent long-session continuity with relevant recall
- persistent memory-light control sessions without relevant recall
- longer guided-exercise arcs with multiple continue / preserve / clear transitions
- ambiguous recovery / relapse arcs that move through safe therapeutic, crisis clarification, and crisis response

Use this file to catch drift that only appears after several turns rather than in shorter trajectory checks.

### `datasets/session_quality_trajectories.jsonl`
Optional judged full-session quality trajectories.

Covers:
- persistent vs incognito support-session quality on the same durable-memory scenario
- persistent vs incognito relationship-tension quality on the same scenario
- persistent vs incognito work-stress quality
- persistent vs incognito self-criticism quality
- exercise side-turn + resume qualitative smoothness
- exercise interruption/resume and exercise restart quality
- crisis clarification → de-escalation quality
- recovery / relapse and repeated high-risk follow-up qualitative safety handling

Use this file with `run_routing_eval.py --judge` to score whole-session coherence, memory appropriateness, workflow smoothness, and safety handling after deterministic checks pass.

### `datasets/crisis_response_events.jsonl`
End-to-end crisis-response event coverage.

Covers:
- imminent risk with verified resources
- imminent risk without location
- high risk with refused location
- imminent risk with search failure fallback

Use this file to verify crisis-response routing, resource handling, and safety language.

### `datasets/crisis_templates.jsonl`
Crisis response template coverage.

Covers:
- moderate / high / imminent template variants
- verified-resource vs no-resource branches
- safe wording constraints
- phone-number preservation constraints

Use this file to validate generated crisis copy independently from runtime routing.

### `datasets/crisis_resources.jsonl`
Crisis resource lookup coverage.

Covers:
- structured location extraction
- verified resource lookup
- no-location cases
- refused-location cases
- no-verified-result cases
- search-failure fallback
- ambiguous-location cases
- selected live checks

Use this file to validate the crisis resource lookup layer directly.

### `datasets/live_text_runtime_smoke.jsonl`
Opt-in live LLM smoke coverage for real runtime paths.

Covers:
- OpenAI Agents SDK therapeutic response generation
- OpenAI Agents SDK guided-exercise tool use
- OpenAI Agents SDK grounded-lookup tool use
- OpenAI Agents SDK crisis response with resource lookup and audit logging
- OpenAI response-LLM override for persistent memory
- OpenAI response-LLM override for incognito privacy behavior

Use this file with `run_live_text_runtime_eval.py --live` when credentials are
configured and you want live provider coverage beyond deterministic fakes.

### `datasets/live_text_runtime_trajectories.jsonl`
Opt-in live LLM trajectory coverage for real OpenAI runtime paths.

Covers:
- OpenAI Agents SDK guided-exercise start / resume continuity
- OpenAI Agents SDK grounded lookup followed by ordinary support
- OpenAI Agents SDK repeated crisis-resource handling
- OpenAI response-LLM persistent-memory trajectory quality
- OpenAI response-LLM incognito-memory privacy behavior

Run these cases with `run_live_text_runtime_eval.py --live --suite trajectories`.

## Runners

### `runners/run_routing_eval.py`
Primary deterministic runtime eval runner for routing, behavior, and multiturn datasets.

Example usage from repo root:

```bash
apps/backend/.venv/bin/python eval/runners/run_routing_eval.py
```

Run a different dataset:

```bash
apps/backend/.venv/bin/python eval/runners/run_routing_eval.py \
  --dataset eval/datasets/behavior_matrix.jsonl
```

Run a specific case:

```bash
apps/backend/.venv/bin/python eval/runners/run_routing_eval.py \
  --dataset eval/datasets/behavior_matrix.jsonl \
  --case-id grounded_lookup_missing_tool_falls_back
```

Optional judge mode for full-session qualitative scoring:

```bash
apps/backend/.venv/bin/python eval/runners/run_routing_eval.py \
  --dataset eval/datasets/session_quality_trajectories.jsonl \
  --judge --provider openai
```

### `runners/run_live_text_runtime_eval.py`
Opt-in live runtime runner for broader OpenAI-backed smoke coverage.

Run from `apps/backend` with OpenAI:

```bash
.venv/bin/python ../../eval/runners/run_live_text_runtime_eval.py \
  --live --provider openai
```

Run the trajectory suite:

```bash
.venv/bin/python ../../eval/runners/run_live_text_runtime_eval.py \
  --live --provider openai --suite trajectories
```

Run smoke and trajectory cases together:

```bash
.venv/bin/python ../../eval/runners/run_live_text_runtime_eval.py \
  --live --provider openai --suite all
```

Run with LLM-as-judge scoring for cases that define `session_expected`:

```bash
.venv/bin/python ../../eval/runners/run_live_text_runtime_eval.py \
  --live --provider openai --suite trajectories --judge \
  --judge-model gpt-5.4 --min-judge-score 4
```

Run repeated judged samples to inspect transcript-quality variance:

```bash
.venv/bin/python ../../eval/runners/run_live_text_runtime_eval.py \
  --live --provider openai --suite trajectories --judge \
  --judge-model gpt-5.4 --min-judge-score 4 --samples 3
```

With `--samples` greater than `1`, each result includes a per-sample payload
with its own checks, failures, output transcript, and judge result.

The pytest wrappers are additionally gated by explicit flags:
- `RUN_LIVE_OPENAI_RUNTIME_EVALS=1`
- `RUN_LIVE_OPENAI_TRAJECTORY_EVALS=1`
- `RUN_LIVE_OPENAI_TRAJECTORY_JUDGE_EVALS=1`

The trajectory judge pytest wrapper uses OpenAI as judge, defaults to the
configured OpenAI model, and requires every qualitative judge dimension to score
at least `4`. Override those defaults with:
- `OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MODEL`
- `OPENCOUCH_LIVE_TRAJECTORY_JUDGE_MIN_SCORE`
- `OPENCOUCH_LIVE_TRAJECTORY_JUDGE_SAMPLES`

This is separate from the older classifier/style live-test flags so enabling
basic live tests does not unexpectedly run tool-using runtime evals.

The runner also accepts `--dataset path/to/custom.jsonl`, which overrides
`--suite` for ad-hoc live eval files.

## Coverage matrix

| Area | Route / Mode focus | Main datasets |
| --- | --- | --- |
| Safe therapeutic | `therapeutic` / `safe_therapeutic` | `routing_matrix.jsonl`, `routing_boundaries.jsonl`, `multiturn_routing.jsonl` |
| Memory control | `memory_control` / `memory_control` | `routing_matrix.jsonl`, `multiturn_routing.jsonl`, `behavior_matrix.jsonl` |
| Grounded lookup | `grounded_lookup` / `grounded_lookup` | `routing_matrix.jsonl`, `routing_boundaries.jsonl`, `multiturn_routing.jsonl`, `behavior_matrix.jsonl` |
| Guided exercise | `therapeutic` / `guided_exercise` | `routing_matrix.jsonl`, `routing_boundaries.jsonl`, `multiturn_routing.jsonl`, `behavior_matrix.jsonl`, `trajectory_interruptions.jsonl` |
| Crisis clarification | `therapeutic` / `crisis_clarification` | `routing_matrix.jsonl`, `behavior_matrix.jsonl`, `multiturn_routing.jsonl` |
| Crisis response | `crisis` / `crisis_response` | `routing_matrix.jsonl`, `routing_boundaries.jsonl`, `multiturn_routing.jsonl`, `behavior_matrix.jsonl`, `crisis_response_events.jsonl`, `trajectory_interruptions.jsonl`, `trajectory_endurance.jsonl` |
| Trajectory memory modes | persistent vs incognito durable recall behavior | `trajectory_memory_modes.jsonl` |
| Trajectory interruptions | exercise/crisis/recovery trajectories across memory modes | `trajectory_interruptions.jsonl` |
| Trajectory endurance | longer multi-turn continuity / no-leak / relapse coverage | `trajectory_endurance.jsonl` |
| Session quality trajectories | judged full-session coherence / memory / safety quality | `session_quality_trajectories.jsonl` |
| Crisis templates | copy + safety constraints | `crisis_templates.jsonl` |
| Crisis resources | lookup + normalization | `crisis_resources.jsonl` |
| Live text runtime | OpenAI-backed Agents SDK / response-LLM smoke and trajectory paths | `live_text_runtime_smoke.jsonl`, `live_text_runtime_trajectories.jsonl` |

## Current known gaps

### 1. No dedicated eval index existed before this README
The JSONL files were the practical source of truth, but there was no single document explaining:
- what each dataset covers
- how to run it
- where coverage is intentionally incomplete

### 2. Memory-control breadth is stronger, but conflicting-intent depth is still limited
Current evals now cover:
- enable/disable proactive recall
- save-preference memory-tool execution
- forget-by-query memory-tool execution
- preservation of pending memory actions across safe, grounded, and crisis side turns

Still weak or missing:
- conflicting memory intents
- deeper multi-turn save/forget combinations
- more ambiguous memory-control phrasing

### 3. Guided-exercise lifecycle coverage is broad, but some edge cases remain
Covered:
- start
- continue
- preserve
- resume
- explicit exit
- restart with a different exercise mid-flow
- invalid continue when no exercise is active
- crisis interruption during an active exercise
- explicit post-crisis resume

Still light:
- additional abandon / restart wording variants
- more conflicting continue / clear cue combinations

### 4. Grounded missing-tool fallback behavior is documented but semantically odd
The current contract for `grounded_lookup_missing_tool_falls_back` is:

- route falls back to `therapeutic`
- runtime mode becomes `safe_therapeutic`
- response uses the scripted final output
- grounded lookup state remains `not_attempted`
- grounded-tool diagnostics keys are absent

This is now covered by evals, but may still deserve a future product/runtime decision.

### 5. Crisis progression coverage is much better, but ambiguous safety signals remain
Current evals now cover:
- de-escalation back to therapeutic flow
- crisis interruption while another workflow is active
- repeated high-risk follow-up turns
- clarification with location still avoiding lookup

Still light:
- more ambiguous or conflicting safety signals
- mixed recovery / relapse patterns across longer crisis sequences

### 6. Mixed-intent precedence is still the highest-risk gap
Examples worth adding:
- crisis + memory control in the same turn
- crisis + guided exercise in the same turn
- grounded lookup + memory action in the same turn
- preserve / continue conflicts with multiple active cues

### 7. Full session trajectories are still only partially covered
The trajectory memory-mode, interruption, endurance, and judged session-quality datasets
now improve persistent-vs-incognito coverage for durable recall behavior plus longer
exercise/crisis/recovery switching and whole-session quality scoring, but the eval suite
is still lighter on:
- even longer 7-10+ turn conversational endurance checks
- backend-specific persistence parity inside the eval harness itself
- richer retrieval-quality assertions beyond presence/absence of recalled state
- judged coverage for additional nuanced scenarios beyond the current curated fourteen full-session cases

### 8. Live LLM coverage is opt-in smoke coverage, not a full benchmark
The live runtime eval runner now covers real provider paths for SDK response
generation, tool use, crisis response, grounded lookup, and memory-mode behavior.
Remaining gaps:
- no automated live CI budget or scheduled cadence
- repeated live judged sampling is supported, but the curated case set is still small
- no Postgres-backed live persistence parity in the eval harness
- no live voice-runtime eval equivalent

## Conventions for adding cases

When adding a new eval case:

1. Put it in the dataset that matches its primary purpose:
   - route selection → `routing_matrix.jsonl`
   - boundary/precedence → `routing_boundaries.jsonl`
   - multiturn switching/resume → `multiturn_routing.jsonl`
   - side effects / diagnostics / state mutation → `behavior_matrix.jsonl`
   - crisis-specific subsystems → crisis datasets

2. Prefer small, surgical cases.
   - One case should usually validate one contract or one precedence rule.

3. Include only the assertions needed to lock the intended behavior.
   - route
   - runtime mode
   - selected agent
   - key state fields
   - key diagnostics
   - required / forbidden response text

4. If a case captures surprising behavior, document it here under **Current known gaps** or update the relevant section.

## Suggested next additions

Priority candidates:
- guided exercise interrupted by crisis
- memory control interrupted by crisis
- proactive recall enable/disable coverage in eval datasets
- mixed-intent precedence cases
- explicit guided-exercise exit / restart cases
- deeper memory save / retrieval behavior coverage
- scheduled or manually approved live runtime eval runs with saved artifacts

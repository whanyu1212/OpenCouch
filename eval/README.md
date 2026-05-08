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

## Therapeutic subgraph

Therapeutic evals are split by purpose:

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

```bash
apps/backend/.venv/bin/python -m eval.runners.therapeutic_contract_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_behavior_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_quality_eval
apps/backend/.venv/bin/python -m eval.runners.therapeutic_exercise_trajectory_eval
```

Useful flags:

- `--plain`: disable Rich output.
- `--json-output eval/reports/therapeutic_contract.json`: write the full result
  payload.
- `--dataset eval/datasets/therapeutic/contract_v1.json`: override the dataset.
- `--mode live`: run behavior or quality cases with configured provider-backed
  LLM clients.
- `--judge-mode live`: run the exercise trajectory LLM judge.

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
- broader scripted behavior cases for exercise selection and state changes
- hard response-quality checks for concise, concrete, non-menu output
- multi-turn exercise trajectories with optional LLM-as-judge scoring
- narrowed subgraph output keys

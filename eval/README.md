# Eval

Evaluation assets for OpenCouch.

Purpose:
- keep datasets outside application code
- make safety expectations reviewable in git
- allow CI to run deterministic evals on every change
- allow optional live model comparisons outside CI

Initial structure:
- `datasets/`: JSON datasets with expected outcomes
- `runners/`: ad hoc or scripted evaluation entrypoints

Current CI path:
- CI should run backend tests separately from eval runners.
- `eval/runners/crisis_gate_eval.py` reads `datasets/crisis_detection_v1.json`
- the runner calls the actual `run_crisis_gate(...)` entrypoint
- CI runs the runner in deterministic mode
- local runs can use `--mode hybrid` with a configured provider client

Long-session eval path:
- `eval/runners/session_trajectory_live_eval.py` reads `datasets/session_trajectory_long_v1.json`
- the runner drives the real persisted thread runtime turn by turn
- datasets contain longer user-only conversations plus checkpoint expectations
- local runs can use:
  - `--mode deterministic` for stable baseline behavior
  - `--mode hybrid` for actual model-backed session runs
  - `--case <id>` to debug one conversation at a time

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

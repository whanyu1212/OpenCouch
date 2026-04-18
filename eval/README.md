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

Session trajectory eval path:
- `eval/runners/session_trajectory_eval.py` is the unified runner for both
  short and long trajectory datasets
- short dataset (`datasets/session_trajectory_v1.json`): inline per-turn
  `expect` blocks, supports deterministic and hybrid modes
- long dataset (`datasets/session_trajectory_long_v1.json`): sparse
  checkpoint assertions, hybrid-only (requires LLM client)
- the runner drives the real persisted thread runtime turn by turn
- local runs can use:
  - `--mode deterministic` for stable baseline behavior (short dataset)
  - `--mode hybrid` for actual model-backed session runs
  - `--mode auto` (default) to use LLM when available, fall back otherwise
  - `--dataset <path>` to select a dataset
  - `--case <id>` to debug one conversation at a time
- examples:
  - `python eval/runners/session_trajectory_eval.py --mode auto`
  - `python eval/runners/session_trajectory_eval.py --mode hybrid --dataset eval/datasets/session_trajectory_long_v1.json`
  - `python eval/runners/session_trajectory_eval.py --mode hybrid --dataset eval/datasets/session_trajectory_long_v1.json --case out_of_scope_boundary_and_recovery_with_closing`

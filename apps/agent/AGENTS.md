# Agent service

The root `AGENTS.md` covers the layout, commands and architecture rules. These
are the standards for the Python code here.

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
  Ruff enforces a complexity limit of 10 and caps branches and statements
  per function.
- **Pass dependencies explicitly** through agent `deps` or function arguments.
  No module-level globals or singletons beyond cached settings.
- **Keep prompts separate from logic**, as named constants or files, so prompt
  changes show up clearly in diffs.
- **Spell names out in full** (`crisis_assessment`, not `ca`).
- **Tests read like specs:** `test_classifier_error_routes_to_crisis_aware_reply`.
  Use Pydantic AI's `TestModel` / `FunctionModel`; never call a real model in
  the default test run. `safety/` needs full branch coverage.

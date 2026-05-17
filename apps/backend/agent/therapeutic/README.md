# Therapeutic Services

This package owns non-crisis therapeutic response planning and guided-exercise
behavior for the OpenAI text runtime.

## Runtime Shape

- `dispatch/planner.py` classifies therapeutic response style and approach.
- `dispatch/node.py` keeps the old function name as a compatibility adapter
  while `build_therapeutic_dispatch_update` remains the shared update builder.
- `response.py` and `response_styles.py` generate non-exercise therapeutic
  responses for direct service tests and response-LLM override paths.
- `exercises/runner.py` owns guided-exercise lifecycle transitions.
- `exercises/skills.py` and `exercises/definitions/` provide the exercise
  skill blocks consumed by `GuidedExerciseAgent`.

## State Contract

Therapeutic services read the current message, channel, installed skills,
conversation context, crisis state, working memory, procedural profile,
diagnostics, and current exercise state.

They may write:

- `response_text`
- `response_style`
- `session_action`
- `therapeutic_approach`
- `exercise_state`
- `diagnostics`

The OpenAI adapter owns branch execution and final transcript persistence. The
runtime owns session finalization and memory extraction.

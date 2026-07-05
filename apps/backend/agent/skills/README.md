# Agent skills

This package contains app-owned runtime components that package a specialized capability for the model without letting the model own state transitions.

Today that means **guided exercises**. Therapeutic response style guidance lives under `agent/specialists/therapeutic_response/` because it is prompt composition, not a standalone skill runtime.

## What belongs here

Put code in `agent/skills/` when all of these are true:

- the capability has a reusable domain model or state machine;
- local code owns selection, validation, and state updates;
- model-facing tools or prompt context only expose app-approved facts/actions;
- the behavior is shared across text and voice, even if the transport bindings differ.

Do not put generic prompt builders, specialist agent instructions, transport tool schemas, or persistence adapters here. Those belong under `agent/specialists/`, `agent/tools/`, `agent/voice/`, or `agent/runtime/`.

## Guided exercises

`guided_exercises/` is split by responsibility:

```text
guided_exercises/
  catalog/     # what exercises exist and when they are available
  lifecycle/   # what happens next for an active exercise
  rendering/   # how runtime-owned decisions are rendered for the model
```

### `catalog/`

The runtime source of truth for exercise content and availability.

- `types.py` defines `ExerciseDefinition`, `ExerciseStep`, and classifier/selector result schemas.
- `definitions/` groups the exercise definitions by therapeutic family.
- `registry.py` combines definitions, validates catalog integrity at import time, and exposes lookup/filter helpers.

Use `catalog/` when adding or changing exercises, channel support, selection aliases, capability gates, or step metadata.

### `lifecycle/`

The app-owned guided-exercise state machine.

- `service.py` exposes `GuidedExerciseSkillService.run_turn()`.
- `selection.py` chooses an exercise when starting.
- `step_classifier.py` classifies user progress as `complete`, `hold`, `stuck`, or `exit`.
- `state.py` builds and clears `exercise_state` deltas.
- `responses.py` builds response/state deltas for start, advance, hold, stuck, exit, complete, and resume.
- `memory.py` writes completion facts when appropriate.

Text guided-exercise progress should stay here and in app-owned state deltas. The model should not directly mutate `exercise_state`.

### `rendering/`

Prompt-local rendering only.

- `skill_context.py` renders a selected `ExerciseDefinition` and runtime action into the `load_guided_exercise_skill` context block.
- `directives.py` wraps lifecycle decisions into response-generation directives.

Rendering code should not select exercises, classify user replies, or update durable state.

## Text and voice surfaces

The text runtime runs the lifecycle service before generating guided-exercise prose. Voice Realtime does not run that full text lifecycle before speaking, so it exposes a narrow `record_guided_exercise_progress` tool that reuses shared validation and persists only server-computed progress state.

Tool names still use “skill” (`load_guided_exercise_skill`) because the model receives a bounded prompt-local capability block. The source of truth remains the catalog and lifecycle service, not the tool call itself.

# Therapeutic Subgraph

The therapeutic subgraph owns one turn of therapeutic response planning and
generation. It decides whether the turn should be a normal therapeutic response
or part of a guided exercise, then delegates to the matching response node.

LangGraph is intentionally thin here. The graph handles lifecycle and state
transitions; product behavior lives in plain Python modules that are easier to
test directly.

## Runtime Shape

```text
parent graph
  -> therapeutic subgraph
       START
         -> therapeutic_dispatch_node
         -> Command(goto=therapeutic_response_node)
         -> Command(goto=guided_exercise_response_node)
       therapeutic_response_node -> END
       guided_exercise_response_node -> END
```

The graph is built in `graph.py`.

The dispatch node returns a LangGraph `Command`. That command chooses one of two
actual response nodes:

- `therapeutic_response_node`: shared node for non-exercise response styles.
- `guided_exercise_response_node`: stateful guided exercise node.

`response_style` and `therapeutic_approach` are separate concepts:

- `response_style` decides the response family, for example `supportive`,
  `reflective`, `clarifying`, `psychoeducation`, `closing`, `technique`, or
  `guided_exercise`.
- `therapeutic_approach` shapes the clinical frame used by prompts, for example
  CBT, DBT, ACT, mindfulness, behavioral activation, or self-compassion.

Only `response_style == "guided_exercise"` routes to the guided exercise node.
All other styles route to the shared therapeutic response node.

## Directory Map

```text
therapeutic/
  graph.py                    LangGraph subgraph wiring
  response.py                 Shared non-exercise response node
  response_styles.py          Streaming helpers and response deltas
  dispatch/
    node.py                   LangGraph dispatch node
    planner.py                LLM-primary dispatch planning service
    prompt.py                 Dispatch prompt construction
    constants.py              Graph node names and style-to-node mapping
  exercises/
    node.py                   Guided exercise runtime node
    runner.py                 Framework-agnostic guided exercise runner
    selection.py              LLM-only exercise selector
    step_classifier.py        Step completion, hold, stuck, and exit classifier
    responses.py              Exercise response generation and deltas
    state.py                  Exercise state mutation helpers
    skills.py                 Prompt-local exercise skill rendering
    registry.py               Exercise catalog and availability filtering
    types.py                  Exercise dataclasses and classifier schemas
    memory.py                 Exercise completion memory writes
    definitions/              Exercise definitions grouped by family
  prompts/
    builders.py               Response-style system prompt builders
    context.py                Runtime prompt context assembly
    instructions.py           Shared instruction fragments
    sources.py                Prompt source loading
```

## State Contract

The subgraph reads the current message, channel, installed skills, conversation
context, crisis state, working memory, optional procedural profile, optional
diagnostics, and current exercise state.

The subgraph writes:

- `response_text`: generated response for the current turn.
- `response_style`: selected response style.
- `session_action`: optional UI hint. `closing` sets
  `suggest_end_session`; all other turns use the parent graph default `none`.
- `therapeutic_approach`: selected or preserved therapeutic approach.
- `exercise_state`: active guided exercise type, step index, step id, version,
  and pinned approach.
- `diagnostics`: routing trace when diagnostics are enabled.

The graph uses a narrowed input and output schema so the parent graph does not
duplicate reducers such as transcript handling.

## Dispatch

Dispatch is split into a LangGraph adapter and a framework-independent planner:

- `dispatch/node.py` reads graph state, calls the planner, applies exercise
  clearing when needed, and returns `Command(goto=...)`.
- `dispatch/planner.py` performs route planning with the classifier LLM and
  returns a `DispatchPlan`.
- `dispatch/constants.py` maps response styles to graph node names.

The planner requires the classifier LLM. It asks for a structured dispatch
decision and lets missing clients or provider failures surface to the graph
retry/error path instead of silently substituting a local route.

For active exercises, dispatch keeps exercise state only when the LLM continues
the guided exercise or asks a clarifying side question. Both paths reuse the
pinned exercise approach for continuity. Other non-exercise response styles
clear the active exercise before the response node runs, which prevents
explanatory or reflective side turns from accidentally keeping an exercise
alive after the user switches away.

When the planner selects `closing`, the dispatch node adds
`session_action="suggest_end_session"`. This is only a client hint. The
therapeutic subgraph never ends the persistent session or writes a session arc;
explicit runtime finalization remains outside the graph node.

## Non-Exercise Responses

`response.py` is the shared response node for all non-exercise therapeutic
styles. It uses a small mapping from response style to system prompt builder.

The actual streaming call and common response delta construction live in
`response_styles.py`. This keeps the node small and prevents each response style
from becoming its own graph node.

Non-exercise responses require a response LLM. If no response LLM is configured,
or if the provider call fails after graph retries, the error is allowed to
surface instead of returning canned therapeutic text.

## Prompt Construction

Prompt construction is centralized under `prompts/`:

- `sources.py` loads source markdown and prompt fragments.
- `instructions.py` owns shared response instructions.
- `context.py` assembles runtime prompt context from state.
- `builders.py` combines sources, instructions, safety overrides, memory
  context, recall settings, and response-style guidance.

The guided exercise prompt uses the pinned
`exercise_state.exercise_therapeutic_approach` when an exercise is active. That
prevents the exercise from changing clinical frame mid-flow even if the top
level route planner changes its approach for another turn.

## Guided Exercises

The guided exercise implementation lives in `exercises/`.

An exercise turn has two modes:

1. Start a new exercise.
2. Continue the active exercise.

### Starting An Exercise

When no exercise is active, `exercises/node.py` delegates to
`runner.py`, which calls `selection.py` to choose an exercise.

Selection is LLM-only:

- the selector receives the current message, recent history, therapeutic
  approach, channel, installed skills, and available exercise definitions;
- the available catalog is filtered by approach, channel, required skill, and
  voice support;
- the selector must return one supported exercise id with sufficient
  confidence;
- there is no deterministic menu fallback and no regex-based exercise routing.

If the selector cannot choose a valid exercise, the error is surfaced instead of
silently swapping in a deterministic fallback. The system expectation is that
the classifier LLM is available for primary routing and exercise selection.

When an exercise starts, `state.py` writes:

- `exercise_type`: selected exercise id.
- `exercise_step`: `0`.
- `exercise_step_id`: stable id for step `0`.
- `exercise_version`: selected definition version.
- `exercise_therapeutic_approach`: the current top-level approach, pinned for
  the exercise flow.

The first step instruction is then generated as the response.

### Continuing An Exercise

When an exercise is active, the node loads the current exercise definition and
current step, then asks `step_classifier.py` for the step state:

- `complete`: the user has satisfied the current step.
- `hold`: the user is still working or needs more time.
- `stuck`: the user is confused, blocked, or needs help.
- `exit`: the user wants to stop the exercise.

The classifier LLM judges the step state using the latest user message, current
step instruction, completion mode, and completion criteria. There are no regex
or local heuristic overrides for exits, stuck states, confirmations, item counts,
or resume requests.

The response behavior is:

- `exit`: acknowledge and clear exercise state.
- `hold`: keep the current step and offer a light continuation.
- `stuck`: keep the current step and provide extra support.
- `complete`: advance to the next step, or finish the exercise if this was the
  last step.

On completion, `memory.py` may write a `coping_strategy` memory fact when memory
is enabled, then the exercise state is cleared.

### Exercise Definition Fields

Exercises are plain dataclasses registered in `registry.py`.
`skills.py` renders a single runtime-selected definition into a compact,
prompt-local skill block for response generation. This keeps the model focused
on the selected exercise without exposing an arbitrary user-selectable skill
catalog.

```python
ExerciseDefinition(
    id="five_four_three",
    display_name="5-4-3-2-1 grounding",
    selection_use_case="When the user feels anxious, panicky, dissociated, or needs sensory grounding.",
    version=1,
    category="grounding",
    tags=("sensory", "panic", "anxiety", "dissociation"),
    duration_seconds=300,
    intensity="low",
    selection_aliases=("grounding", "54321", "five senses"),
    approaches=("mindfulness",),
    channels=("text", "voice"),
    voice_supported=True,
    steps=(
        ExerciseStep(
            id="see",
            instruction="Name five things you can see around you.",
            completion_mode="items",
            target_items=5,
            min_items=3,
            completion_criteria="The user names at least three visible things.",
        ),
        ExerciseStep(
            id="feel",
            instruction="Name four things you can feel.",
            completion_mode="items",
            target_items=4,
            min_items=2,
        ),
    ),
)
```

Key fields:

- `id`: stable internal exercise id.
- `display_name`: user-readable name.
- `selection_use_case`: when the selector should choose this exercise.
- `version`: definition version stored in active exercise state.
- `category`: broad family used for filtering and reporting.
- `tags`: selection/filtering tags.
- `duration_seconds`: rough expected duration.
- `intensity`: rough effort or emotional load.
- `selection_aliases`: natural phrases that help the selector recognize intent.
- `approaches`: compatible therapeutic approaches. Empty means generally
  available.
- `channels`: supported runtime channels.
- `required_skill`: optional installed skill gate.
- `voice_supported`: whether voice flows may use the exercise.
- `steps`: ordered instructions and completion rules.

### Completion Modes

Each `ExerciseStep` declares how completion should be judged:

- `id`: stable step id used alongside the numeric step index for durable
  continuity across future definition edits.
- `items`: the user should provide a list of items. The LLM classifier uses
  `target_items` and `min_items` as judgment criteria.
- `confirmation`: the user does a private action and confirms when done.
- `response`: the user gives a substantive answer.
- `llm_judged`: the step has criteria that need semantic judgment. Use
  `completion_criteria` to tell the classifier what counts.

Use the least powerful completion mode that fits the step. Prefer `items` or
`confirmation` for simple procedural steps, and reserve `llm_judged` for steps
that genuinely need semantic interpretation.

## Diagnostics

When diagnostics are present in state, dispatch appends a routing trace that
records the chosen style, approach, source, reason, and confidence. Diagnostics
are optional; the runtime does not require them for normal operation.

## Adding A Response Style

Most new response styles should use the shared response node.

1. Add the style to the therapeutic response style type.
2. Teach the dispatch prompt when to choose it.
3. Add prompt source and instruction support if needed.
4. Add the system prompt builder mapping in `response.py`.
5. Add or update routing and prompt tests.

Create a new graph node only when the style needs its own state machine or
special side effects. If it only changes tone or prompt framing, keep it in the
shared response node.

## Adding An Exercise

1. Add an `ExerciseDefinition` in the closest file under `exercises/definitions/`
   or create a new family file.
2. Include precise selector metadata: use case, aliases, approaches, channel
   support, and skill requirements.
3. Write steps with explicit completion modes and criteria.
4. Add it to the family file's `DEFINITIONS` tuple. If a new family file is
   added, import that family tuple in `registry.py`.
5. Add tests covering selection and flow.

Avoid hidden deterministic routing rules in exercise definitions. Selection
should remain catalog-driven and LLM-primary.

## Validation

Useful checks for this area:

```bash
cd apps/backend
.venv/bin/python -m pytest tests/integration/therapeutic tests/integration/graph/test_state_contracts.py tests/unit/therapeutic
```

For broader behavior and quality checks, run the therapeutic eval harness from
the repository root:

```bash
apps/backend/.venv/bin/python -m eval.runners.therapeutic_contract_eval --plain
apps/backend/.venv/bin/python -m eval.runners.therapeutic_behavior_eval --plain
apps/backend/.venv/bin/python -m eval.runners.therapeutic_quality_eval --plain
apps/backend/.venv/bin/python -m eval.runners.therapeutic_exercise_trajectory_eval --plain
```

For docs-only changes, run pre-commit on the changed file:

```bash
uv run pre-commit run --files apps/backend/agent/therapeutic/README.md
```

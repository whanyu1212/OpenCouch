# OpenCouch Text Agent

The backend text agent now runs through the OpenAI Agents SDK. The runtime is
organized around plain product services and one serving SDK runtime:

- `agent.runtime.OpenAITextRuntime` owns text-agent
  execution, specialist agent selection, SDK tool calls, and stream events.
- `agent.runtime.PersistentAgentRuntime` owns thread locks, app-owned state
  snapshots, OpenAI SDK sessions, active-session lifecycle,
  crisis audit, and public API/CLI history surfaces.
- `agent.runtime.build_initial_state`, `agent.runtime.state_to_output`, and
  `agent.runtime.run_agent` provide runtime-native turn helpers without a
  legacy workflow compatibility layer.

## Runtime State

Conversation state that belongs to OpenCouch is stored in
`opencouch_thread_state` through `agent.runtime.state_store`.

The OpenAI SDK session store is separate and model-visible. It holds short-term
conversation history for the SDK runner. The runtime state snapshot holds
product state such as diagnostics, crisis metadata, exercise state, pending
memory actions, and transcript fallback for API/CLI history.

Durable active-session coordination is stored by
`PostgresActiveSessionStore` in `opencouch_active_sessions` and is independent
from both runtime state and SDK session history.

Postgres is the only supported durable backend for application-owned runtime
state, long-term memory, crisis audit, session feedback, and active-session
recovery. In-memory stores remain the incognito/test path. The old SQLite
runtime-state, active-session, crisis-log, and feedback implementations have
been removed. `SqliteMemoryStore` remains only for explicit migration
inspection of old long-term-memory files.

OpenAI Agents SDK text sessions are a separate, model-visible short-term
history surface. Their SDK `SQLiteSession` option, including
`text_sessions.sqlite3`, is not an OpenCouch long-term-memory backend.

## Agent Shape

The serving text runtime uses three specialist roles:

- `TherapeuticAgent`: ordinary safe conversation plus operational memory and
  grounded lookup tools.
- `CrisisAgent`: crisis clarification, crisis response wording, crisis resource
  lookup tool ownership, and crisis-response diagnostics.
- `GuidedExerciseAgent`: guided exercise wording and exercise-skill tool
  ownership.

The application runtime still owns classification, consent/selection policy,
state transitions, persistence, and audit logging. Agents generate the response
or call the tool required by the runtime-selected branch.

Source layout:

- `agent/runtime/openai_text_runtime.py`: OpenAI-backed text turn execution.
- `agent/specialists/`: triage, therapeutic, crisis, and guided-exercise
  specialist agent definitions split by owner.
- `agent/tools/`: memory, grounded lookup, crisis lookup, therapeutic skill,
  and guided-exercise skill tools exposed to the SDK.
- `agent/memory/control/`: saved-memory inspection, preference updates, and
  pending deletion state shared by text and voice tools.

## Memory And Session Boundaries

Turn memory loading is orchestrated by `agent.runtime.memory_context`, which
turns durable recall into runner-turn state before `Runner.run`. Long-term
memory retrieval and storage remain in `agent.memory.retrieval.service` and
`agent.memory.store`, while session-end summarization and candidate promotion
run through `agent.runtime.session`.

OpenAI SDK sessions do not replace OpenCouch session lifecycle. The SDK session
is model-facing conversation memory; OpenCouch active sessions are product
liveness windows used for timeout finalization and channel rotation.

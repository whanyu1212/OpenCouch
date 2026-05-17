# OpenCouch Text Agent

The backend text agent now runs through the OpenAI Agents SDK. The runtime is
organized around plain product services plus a small adapter boundary:

- `agent.text_runtime.openai_adapter.OpenAITextAgentAdapter` owns text-agent
  execution, specialist agent selection, SDK tool calls, and stream events.
- `agent.persistence.PersistentAgentRuntime` owns thread locks, app-owned state
  snapshots, OpenAI SDK sessions, active-session lifecycle, memory extraction,
  crisis audit, and public API/CLI history surfaces.
- `agent.graph` remains only as a compatibility module for building the
  internal state shape and for one-shot `run_agent` calls.

## Runtime State

Conversation state that belongs to OpenCouch is stored in
`opencouch_thread_state` through `agent.runtime.state_store`.

The OpenAI SDK session store is separate and model-visible. It holds short-term
conversation history for the SDK runner. The runtime state snapshot holds
product state such as diagnostics, crisis metadata, exercise state, pending
memory actions, and transcript fallback for API/CLI history.

Active-session coordination is stored in `opencouch_active_sessions` and is
independent from both runtime state and SDK session history.

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

## Memory And Session Boundaries

Turn memory retrieval is implemented in `agent.memory.load_turn`. Long-term
memory storage remains in `agent.memory.store`, while session-end summarization
and candidate promotion run through `agent.runtime.session`.

OpenAI SDK sessions do not replace OpenCouch session lifecycle. The SDK session
is model-facing conversation memory; OpenCouch active sessions are product
liveness windows used for timeout finalization, memory extraction, and channel
rotation.

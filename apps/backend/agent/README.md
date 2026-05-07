# OpenCouch LangGraph Agent

This package contains the LangGraph workflow for the OpenCouch agent. The graph
is organized around three concerns:

- Safety-first routing before any optional memory work.
- Therapeutic response generation through a nested subgraph.
- Post-response persistence side effects that do not block response selection.

The main entrypoints are:

- `agent.graph.run_agent`: one-shot execution with a freshly compiled workflow.
- `agent.persistence.PersistentAgentRuntime`: thread-backed execution with a
  LangGraph checkpointer, long-term memory store, crisis log, and session
  finalization.

## Top-Level Flow

The top-level graph is assembled in `agent.graph.build_agent_workflow`.

```text
START
  -> crisis_gate_node
       -> crisis_resource_lookup_node -> crisis_response_node
          -> crisis_log_node -> finalize_turn_node
       -> memory_control_gate_node
          -> memory_control_node -> finalize_turn_node
          -> grounded_lookup_gate_node
             -> grounded_answer_node -> finalize_turn_node
             -> load_memory_node -> therapeutic_subgraph -> finalize_turn_node

finalize_turn_node
  -> extract_semantic_facts_node -> END
  -> extract_procedural_rules_node -> END
```

`crisis_gate_node` returns a LangGraph `Command` with both a state update and
the next node. There is no conditional edge for the crisis branch. This keeps
the routing decision and the audit metadata in one place.

`therapeutic_subgraph` is itself a compiled `StateGraph`, registered as one
node in the parent graph. It handles therapeutic response-style routing and
response generation, then returns only the channels it owns to avoid
duplicating reducer-backed transcript state in the parent graph.

## Runtime Context

Nodes receive runtime-only dependencies through `WorkflowContext` in
`agent.runtime_context`.

| Field | Purpose |
| --- | --- |
| `llm_client` | Control-plane LLM for classifiers, extractors, and default response generation. |
| `response_llm` | Optional response-writer override. Falls back to `llm_client` in the persistent runtime. |
| `memory_store` | Unified semantic, episodic, and procedural memory backend. |
| `crisis_log_backend` | Always-on safety audit backend. |
| `memory_mode` | Controls incognito vs local persistence behavior. |
| `embedding_provider` | Optional embedding provider for hybrid retrieval and write indexing. |
| `session_memory_buffer` | Runtime-owned buffer for semantic/procedural candidates held until session end. |

Graph state should contain serializable conversation state. Runtime context
should contain services, clients, stores, and other non-stateful dependencies.

## State Shape

The state schema lives in `agent.state`.

| State group | Channels | Owner |
| --- | --- | --- |
| Identity | `message`, `channel`, `user_id`, `session_id`, `installed_skills` | Turn input. |
| Conversation | `history`, `transcript`, `working_memory` | `build_initial_state`, `load_memory_node`, `finalize_turn_node`. |
| Persistent continuity | `session_memory`, `procedural_profile`, `session_progress`, `exercise_state`, `memory_control` | Memory load, turn counting, guided exercise, memory-control confirmation/action state, runtime session management. |
| Crisis | `crisis` | `crisis_gate_node`. |
| Output | `therapeutic_approach`, `response_style`, `response_text`, `should_persist_memory`, `diagnostics` | Routing and response nodes. |
| Private | `route`, `crisis_audit`, `grounded_lookup_query`, `grounded_lookup_status`, `inferred_location`, `found_resources`, `resource_lookup_status` | Crisis gate, grounded factual lookup, crisis resource lookup, crisis response/logging, and internal observability. |

Reducer-backed channels:

- `history` and `transcript` use `operator.add`; nodes must return only newly
  appended turns, not the full accumulated list.
- `session_memory`, `procedural_profile`, `session_progress`,
  `exercise_state`, and `diagnostics` merge dict deltas with right-side
  precedence.

## Crisis Branch

`crisis_gate_node` is the first executable node in the graph. It runs before
memory retrieval so a safety-critical turn is not delayed by optional memory
work.

Classification order:

1. Deterministic hard overrides from `agent.gates.safety.crisis_rules`.
2. LLM structured classifier when `llm_client` is available.
3. Deterministic fallback when no LLM is available or the LLM call fails.

The crisis truth table is enforced after classification:

- Level `0`: no crisis response, route to therapeutic branch.
- Level `1`: clarification needed, but no crisis response branch.
- Level `2`: clear self-harm or suicidal ideation, route to crisis response.
- Level `3`: imminent risk, route to crisis response.

When the branch is crisis:

- `crisis_resource_lookup_node` writes optional location/resource metadata.
- `crisis_response_node` writes the user-facing safety response from the
  resolved crisis and resource state.
- `crisis_log_node` writes an always-on audit record and returns no meaningful
  state delta.
- Memory extractors skip crisis turns after finalization so the response is not
  delayed and crisis content is not written into normal memory.

## Memory Load

`load_memory_node` runs only on the non-crisis branch.

`memory_control_gate_node` runs before memory load on non-crisis turns. Explicit
memory-management requests route to `memory_control_node`; ordinary therapeutic
turns continue to `grounded_lookup_gate_node`, then `load_memory_node`.

In incognito mode it returns empty working memory, a guest-session summary, and
disabled procedural recall.

In persistent mode it:

- Resolves the memory owner from `user_id` or `session_id`.
- Retrieves episodic context, semantic facts, and procedural rules.
- Computes query embeddings when an embedding provider is available.
- Writes `working_memory`, `session_memory.summary`, `procedural_profile`, and
  retrieval diagnostics.

`working_memory` is prompt-facing context for the current turn. It is not the
durable store itself.

## Memory Usage

The memory layer has three user-facing memory kinds:

| Kind | Stored as | Typical source |
| --- | --- | --- |
| Semantic facts | Records under `(owner_id, "semantic")` | Stable, low-risk facts the user explicitly shares. |
| Episodic sessions | Records under `(owner_id, "episodic")` | Session-end summaries from `run_summarize_session`. |
| Procedural rules | One profile under `(owner_id, "procedural")` | Explicit response-style preferences and promoted interaction preferences. |

Memory is scoped by owner. `resolve_owner_id` uses `user_id` when present and
falls back to `session_id`; the CLI `--user-id` flag is the stable way to share
memory across threads. CLI `guest` mode maps to `MemoryMode.INCOGNITO`, and CLI
`persistent` mode maps to `MemoryMode.LOCAL`.

Recall behavior is intentionally selective:

- `load_memory_node` retrieves relevant working memory for the current turn; it
  does not dump the whole durable store into the prompt.
- Procedural rules shape how the agent responds.
- `proactive_recall_enabled` controls whether the agent may bring up prior
  sessions or saved memories without being asked. Turning it off does not delete
  memory and does not disable style preferences.

Users can manage memory conversationally. These explicit requests route through
`memory_control_gate_node` to `memory_control_node` before memory loading:

| User wording | Effect |
| --- | --- |
| `What do you remember about me?` | Lists saved facts, session summaries, and style preferences. |
| `Memory status` | Shows counts and proactive-recall state. |
| `Don't bring up past sessions unless I ask.` | Turns proactive recall off. |
| `You can bring up past sessions if relevant.` | Turns proactive recall on. |
| `Remember that I prefer shorter responses.` | Saves an explicit procedural preference. |
| `Forget what you remember about presentations.` | Finds a matching memory and asks for confirmation before deleting. |
| `Forget fact #2`, `forget session #1`, `forget rule #1` | Targets a displayed memory by kind and index, then asks for confirmation. |

Memory-control turns are operational replies: they skip normal therapeutic
routing and the post-response extractors, so a memory-management request does
not accidentally create another memory. Deletes require an explicit confirmation
such as `yes`, `confirm`, or `delete it`; `no` or `cancel` clears the pending
change.

The CLI also has slash commands for operator control:

```text
/memory status
/memory list
/memory list facts|sessions|rules
/memory recall on|off
/memory forget fact|session|rule <n>
/memory clear facts|sessions|rules|all
/memory purge-crisis [days]
```

Slash commands are CLI-only and bypass the conversational graph. Use them when
dogfooding or inspecting the local SQLite store directly; use conversational
memory control when testing how the agent behaves in normal chat.

Write timing:

- Hot-path semantic/procedural extractors run after `finalize_turn_node`, so the
  user-visible response is persisted before memory side effects start.
- Some semantic/procedural candidates are buffered and promoted only at session
  end by `run_commit_session_memory`.
- Episodic session summaries are written only at session end.
- Crisis turns skip normal memory extraction; crisis audit logging is handled by
  the crisis branch and remains separate from user memory.

## Grounded Factual Lookup

`grounded_lookup_gate_node` runs after memory-control routing and before memory
loading. It is a narrow guard for explicit factual/current-information requests,
not a general therapeutic tool.

It routes to `grounded_answer_node` only when the user clearly asks the agent to
look something up, verify current information, check official sources, or find
location-specific non-crisis resources. Examples:

- "Can you look up affordable counselling services in Singapore?"
- "Can you check if 988 works outside the US?"
- "What are the current official eligibility rules for medical leave?"

It should not trigger for ordinary emotional disclosure, subjective therapeutic
questions, crisis turns, or memory-control turns. Those paths are handled before
or after the gate by their own nodes.

`grounded_answer_node` calls the provider with search grounding enabled and
returns an operational `grounded_lookup` response. If search is unavailable or
no verified answer is found, it says so rather than guessing. Grounded lookup
turns skip the post-response memory extractors so factual lookup requests do
not accidentally become durable user memory.

## Therapeutic Subgraph

The therapeutic subgraph is assembled in `agent.therapeutic.graph`.

```text
START
  -> therapeutic_dispatch_node
       -> therapeutic_response_node
       -> guided_exercise_response_node
  -> END
```

`therapeutic_dispatch_node` returns `Command(goto=...)`. Non-exercise response styles share `therapeutic_response_node`; guided exercises remain separate because they own exercise-state updates. Response nodes do not route further; each terminates the subgraph.

The subgraph has explicit input and output schemas. This is intentional:

- The subgraph can read the state it needs.
- The parent only receives response and exercise-continuity fields.
- Reducer-backed `history` and `transcript` are not emitted back to the parent,
  so they are not appended twice.

## Therapeutic Routing

The dispatcher uses an LLM-primary strategy:

1. If an exercise is active and the user gives an explicit deterministic exit
   signal, clear `exercise_state` and route to supportive.
2. Otherwise, call the structured LLM dispatcher when available.
3. If the LLM is unavailable or fails, use narrow regex fallback heuristics.

Response styles:

| Response style | Use when |
| --- | --- |
| `supportive` | The user is sharing feelings, venting, greeting, or needs validation without structure. |
| `reflective` | The user names a recurring pattern or asks why a pattern keeps happening. |
| `clarifying` | The message is too ambiguous to answer meaningfully. |
| `psychoeducation` | The user describes a specific reaction and asks for a normalizing explanation. |
| `technique` | The user wants structured therapeutic thought work without starting a named exercise track. |
| `guided_exercise` | The user explicitly asks to start or continue a supported structured exercise. |
| `closing` | The user clearly signals they are winding down, leaving, or asking for a wrap-up takeaway. |

The dispatcher also selects `therapeutic_approach`, such as `cbt`, `act`,
`dbt_skills`, `motivational_interviewing`, `grief_support`,
`interpersonal_therapy`, `pfa`, or `none`. The response nodes use that approach
to shape prompts, but the approach is not itself a route.

Wrap-up takeaway requests stay inside the `closing` style. Examples include
"Before we wrap up, what's the main takeaway?", "What should I remember from
this?", or "Can you put the main thing in one sentence?" The closing node should
give one concise synthesis and avoid reopening exploration. This is distinct
from session-end summarization: it does not end the session, write episodic
memory, or require a separate recap node.

## Guided Exercises

`guided_exercise_response_node` is the only therapeutic response style that owns
multi-turn exercise continuity.

It reads and writes:

- `exercise_state.exercise_type`
- `exercise_state.exercise_step`
- `exercise_state.exercise_therapeutic_approach`

Exercise starts set all three fields. Exercise continuation advances, holds,
rephrases, exits, or completes based on the current step and user reply.

During an active exercise:

- Guided-exercise turns preserve the active exercise.
- Clarifying and psychoeducation side turns can preserve the exercise when the
  dispatcher decides the user is asking a side question.
- Non-exercise exits clear `exercise_state`.

On completion, the node can set `should_persist_memory=True` and write a
completion fact when persistence is enabled.

## Finalization And Post-Response Work

`finalize_turn_node` appends the assistant response to both `history` and
`transcript`. It returns only the new assistant turn because both channels are
reducer-backed.

After finalization, two side-effect nodes run:

- `extract_semantic_facts_node`
- `extract_procedural_rules_node`

These nodes write diagnostics into state, but their durable memory writes are
side effects on `memory_store` or `session_memory_buffer`.

They skip when:

- The turn took the crisis route.
- No LLM client is available.
- The runtime is incognito.
- Small-talk or write-policy gates reject the turn.

## Persistent Runtime And Session End

`PersistentAgentRuntime` owns thread persistence and session lifecycle.

Per turn, it:

- Restores checkpointed state by `thread_id`.
- Builds a fresh turn input through `build_initial_state`.
- Invokes the compiled graph with a `WorkflowContext`.
- Tracks the maximum crisis level for the active session.
- Tracks dominant therapeutic approach candidates in the session buffer.
- Persists runtime-owned active-session metadata.

At session end, the runtime calls helpers outside the compiled graph:

- `run_summarize_session` writes an episodic session arc.
- `run_commit_session_memory` promotes buffered semantic/procedural candidates
  that have enough support.

This split keeps immediate turn response generation separate from slower
session-consolidation work.

## Adding Or Changing Nodes

When adding a new node:

1. Decide whether it belongs in the top-level graph, the therapeutic subgraph,
   or the runtime-only session-end path.
2. Define which state channels it reads and writes before editing graph wiring.
3. Return only the smallest state delta the node owns.
4. Do not return full reducer-backed lists unless the node is intentionally
   appending exactly those new items.
5. Keep routing decisions inside `Command`-returning nodes when the route and
   state update must stay atomic.
6. Add or update eval cases for behavior changes, especially crisis routing,
   therapeutic dispatch, guided-exercise continuity, and memory writes.

Common verification commands:

```bash
uv run pytest apps/backend/tests -q
uv run python eval/runners/exercise_selection_eval.py --mode deterministic
uv run python eval/runners/exercise_flow_eval.py
uv run python eval/runners/session_trajectory_eval.py --mode deterministic
```

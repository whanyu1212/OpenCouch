"""Terminal node that finalizes the turn's transcript before END.

This node exists because transcript finalization is the one concern that
spans both branches and runs last: regardless of whether the turn took
the crisis path or the therapeutic path, the final assistant response
lives in ``state["response"]["text"]`` and needs to be appended to
``transcript`` + ``history`` so the next turn's ``get_history`` call in
:class:`agent.persistence.PersistentAgentRuntime` sees it.

Why this isn't done in the runner, in ``extract_semantic_facts_node``,
or in ``build_initial_state``:

- **Runner approach:** would require calling LangGraph's ``update_state``
  after ``ainvoke`` to push the transcript back into the checkpoint.
  Messier than having the node in the graph.
- **Folding into extract_semantic_facts_node:** that node is a
  side-effect node that returns ``{}``. Changing its delta shape
  breaks the "pure side effect" contract and couples two unrelated
  concerns (fact extraction and transcript finalization).
- **build_initial_state:** the user message can be appended there (and
  is), but the assistant response isn't known until the response nodes
  have run, so at least the assistant-side append has to happen inside
  the graph.

The cleanest shape is one tiny node, one responsibility. Its delta
contains exactly ``transcript`` and ``history`` — nothing else. It
runs on both branches via the existing converge-before-END edges:

    crisis_response_node → crisis_log_node → extract_semantic_facts_node
      → finalize_turn_node → END
    therapeutic_subgraph → extract_semantic_facts_node
      → finalize_turn_node → END

The guard against appending an empty response is important: if some
response node short-circuits without setting ``response.text``, we'd
otherwise pollute the transcript with a blank assistant turn.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.models import MessageRole
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_finalize_turn_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],  # noqa: ARG001 — unused, shape contract
) -> dict[str, Any]:
    """Append the final assistant response to the transcript.

    The user message was already appended to ``transcript`` and
    ``history`` in :func:`agent.graph.build_initial_state` at turn
    start. This node's only job is to complete the exchange by
    appending the assistant side once the response nodes have
    finished writing to ``state["response"]["text"]``.

    Returns a delta containing ``transcript`` and ``history`` with
    the assistant turn appended. If the response text is empty (which
    should only happen if a branch short-circuits without producing
    a reply), returns an empty delta so the transcript stays clean.
    """

    response_text = state.get("response", {}).get("text", "").strip()
    if not response_text:
        # Nothing to append. Better to leave the transcript alone than
        # to write a blank assistant turn that the CLI would render.
        return {}

    # v0.8 observability pass: stamp the routing mode onto the assistant
    # turn dict so it round-trips through the checkpoint and surfaces
    # in the CLI's /history panel. The mode string comes straight from
    # ``state["routing"]["mode"]`` — whichever response node composed
    # this reply set that field as part of its own delta. Falls back
    # to ``None`` when routing hasn't resolved (edge cases: crisis
    # short-circuit paths that never wrote the key, historical state
    # predating this field). Persistence._messages_from_transcript
    # reads the same key on load.
    routing_mode = state.get("routing", {}).get("mode") or None

    assistant_turn = {
        "role": MessageRole.ASSISTANT.value,
        "content": response_text,
        "mode": routing_mode,
    }

    current_transcript = list(state.get("transcript", []))
    current_history = list(state.get("history", []))
    return {
        "transcript": [*current_transcript, assistant_turn],
        "history": [*current_history, assistant_turn],
    }

"""Terminal node that finalizes the turn transcript before END.

This node exists because transcript finalization is the one concern that
spans both branches and runs last: regardless of whether the turn took
the crisis path or the therapeutic path, the final assistant response
lives in ``state["response_text"]`` and needs to be appended to
``transcript`` so the next turn's ``get_history`` call in
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
contains exactly ``transcript`` - nothing else. It runs on both branches
via the existing converge-before-END edges:

    crisis_resource_lookup_node → crisis_response_node → crisis_log_node
      → finalize_turn_node → {extract_semantic_facts_node,
                              extract_procedural_rules_node} → END
    memory_control_node → finalize_turn_node → {extract_semantic_facts_node,
                                                extract_procedural_rules_node} → END
    grounded_answer_node → finalize_turn_node → {extract_semantic_facts_node,
                                                 extract_procedural_rules_node} → END
    therapeutic_subgraph → finalize_turn_node → {extract_semantic_facts_node,
                                                 extract_procedural_rules_node} → END

The guard against appending an empty response is important: if some
response node short-circuits without setting ``response_text``, we'd
otherwise pollute the transcript with a blank assistant turn.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.models import MessageRole
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_finalize_turn_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],  # noqa: ARG001 - required node shape
) -> dict[str, Any]:
    """Append the final assistant response to the transcript.

    The user message was already emitted into ``transcript`` by
    :func:`agent.graph.build_initial_state` at turn start. The field uses
    an ``operator.add`` reducer, so this node returns a single-element list
    containing just the new assistant turn — the reducer appends it to the
    accumulated state from the checkpoint.

    Returns a delta containing ``transcript`` with the new assistant turn.
    If the response text is empty (which
    should only happen if a branch short-circuits without producing
    a reply), returns an empty delta so the transcript stays clean.

    Args:
        state: Current graph state after a response node has run.
        runtime: LangGraph runtime, unused but required by node shape.

    Returns:
        State delta appending the assistant turn, or an empty delta.
    """

    response_text = str(state.get("response_text", "") or "").strip()

    # Mark the moment the response is locked in so the runtime can later
    # compute ``post_finalize_ms`` — the wall-clock between this point
    # and graph termination. Used to measure the latency wedge that
    # background extraction (#5) would close.
    finalize_done_at_monotonic = time.monotonic()

    if not response_text:
        # Nothing to append. Better to leave the transcript alone than
        # to write a blank assistant turn that the CLI would render.
        return {
            "diagnostics": {
                "finalize_done_at_monotonic": finalize_done_at_monotonic,
            }
        }

    # Stamp the routing mode onto the assistant turn so it round-trips through
    # the checkpoint and surfaces in the CLI's /history panel.
    routing_mode = state.get("response_style") or None

    assistant_turn = {
        "role": MessageRole.ASSISTANT.value,
        "content": response_text,
        "response_style": routing_mode,
    }

    # Return only the new assistant turn. The ``operator.add`` reducer
    # on ``transcript`` appends it to the accumulated state from the
    # checkpoint automatically.
    return {
        "transcript": [assistant_turn],
        "diagnostics": {
            "finalize_done_at_monotonic": finalize_done_at_monotonic,
        },
    }

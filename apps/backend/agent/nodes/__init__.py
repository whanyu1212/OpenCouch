"""LangGraph node modules for the OpenCouch workflow.

The graph wires nodes via deep imports (``from agent.nodes.crisis_gate
import run_crisis_gate_node``) rather than through this re-export
surface, so these names are primarily a public-surface index for
readers — they document which functions in this package are actual
graph entry points.

Memory extraction (semantic + procedural) is *not* a graph node. It runs
as a runtime-managed background task or, for one-shot ``run_agent``
callers, synchronously after ``ainvoke`` — see
:mod:`agent.memory.extraction_service` and
:class:`agent.persistence.PersistentAgentRuntime`.
"""

from agent.nodes.crisis_gate import run_crisis_gate_node
from agent.nodes.crisis_log import run_crisis_log_node
from agent.nodes.crisis_resource_lookup import run_crisis_resource_lookup_node
from agent.nodes.crisis_response import run_crisis_response_node
from agent.nodes.finalize_turn import run_finalize_turn_node
from agent.nodes.grounded_answer import run_grounded_answer_node
from agent.nodes.load_memory import run_load_memory_node
from agent.nodes.memory_control import run_memory_control_node
from agent.nodes.turn_dispatch import run_turn_dispatch_node

__all__ = [
    "run_crisis_gate_node",
    "run_crisis_log_node",
    "run_crisis_resource_lookup_node",
    "run_crisis_response_node",
    "run_finalize_turn_node",
    "run_grounded_answer_node",
    "run_load_memory_node",
    "run_memory_control_node",
    "run_turn_dispatch_node",
]

"""Node names and routing constants for therapeutic dispatch."""

from __future__ import annotations

from typing import Literal, TypeAlias


TherapeuticNodeName: TypeAlias = Literal[
    "supportive_response_node",
    "reflective_response_node",
    "clarifying_response_node",
    "psychoeducation_response_node",
    "closing_response_node",
    "guided_exercise_response_node",
    "technique_response_node",
]

SUPPORTIVE_NODE: TherapeuticNodeName = "supportive_response_node"
REFLECTIVE_NODE: TherapeuticNodeName = "reflective_response_node"
CLARIFYING_NODE: TherapeuticNodeName = "clarifying_response_node"
PSYCHOEDUCATION_NODE: TherapeuticNodeName = "psychoeducation_response_node"
CLOSING_NODE: TherapeuticNodeName = "closing_response_node"
GUIDED_EXERCISE_NODE: TherapeuticNodeName = "guided_exercise_response_node"
TECHNIQUE_NODE: TherapeuticNodeName = "technique_response_node"

# Mapping from mode name → subgraph node name. Kept as a dict so the
# dispatcher's logic stays pure (pick_therapeutic_mode returns a name)
# and the routing layer does the name-to-node translation.
_MODE_NODE_MAP: dict[str, TherapeuticNodeName] = {
    "supportive": SUPPORTIVE_NODE,
    "reflective": REFLECTIVE_NODE,
    "clarifying": CLARIFYING_NODE,
    "psychoeducation": PSYCHOEDUCATION_NODE,
    "closing": CLOSING_NODE,
    "guided_exercise": GUIDED_EXERCISE_NODE,
    "technique": TECHNIQUE_NODE,
}

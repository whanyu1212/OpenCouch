"""Node names and routing constants for therapeutic dispatch."""

from __future__ import annotations

from typing import Literal, TypeAlias


TherapeuticNodeName: TypeAlias = Literal[
    "therapeutic_response_node",
    "guided_exercise_response_node",
]

THERAPEUTIC_RESPONSE_NODE: TherapeuticNodeName = "therapeutic_response_node"
SUPPORTIVE_NODE: TherapeuticNodeName = THERAPEUTIC_RESPONSE_NODE
REFLECTIVE_NODE: TherapeuticNodeName = THERAPEUTIC_RESPONSE_NODE
CLARIFYING_NODE: TherapeuticNodeName = THERAPEUTIC_RESPONSE_NODE
PSYCHOEDUCATION_NODE: TherapeuticNodeName = THERAPEUTIC_RESPONSE_NODE
CLOSING_NODE: TherapeuticNodeName = THERAPEUTIC_RESPONSE_NODE
GUIDED_EXERCISE_NODE: TherapeuticNodeName = "guided_exercise_response_node"
TECHNIQUE_NODE: TherapeuticNodeName = THERAPEUTIC_RESPONSE_NODE

# Mapping from response-style name to subgraph node name. Kept as a dict so the
# dispatcher's logic stays pure and the routing layer does the translation.
_RESPONSE_STYLE_NODE_MAP: dict[str, TherapeuticNodeName] = {
    "supportive": SUPPORTIVE_NODE,
    "reflective": REFLECTIVE_NODE,
    "clarifying": CLARIFYING_NODE,
    "psychoeducation": PSYCHOEDUCATION_NODE,
    "closing": CLOSING_NODE,
    "guided_exercise": GUIDED_EXERCISE_NODE,
    "technique": TECHNIQUE_NODE,
}

"""Node names for therapeutic dispatch."""

from __future__ import annotations

from typing import Literal, TypeAlias


TherapeuticNodeName: TypeAlias = Literal[
    "therapeutic_response_node",
    "guided_exercise_response_node",
]

THERAPEUTIC_RESPONSE_NODE: TherapeuticNodeName = "therapeutic_response_node"
GUIDED_EXERCISE_NODE: TherapeuticNodeName = "guided_exercise_response_node"


def node_for_response_style(response_style: str) -> TherapeuticNodeName:
    """Return the subgraph node that owns a response style.

    Args:
        response_style (str): Dispatcher-selected therapeutic response style.

    Returns:
        TherapeuticNodeName: Guided-exercise node for active exercise work,
            otherwise the shared therapeutic response node.
    """

    if response_style == "guided_exercise":
        return GUIDED_EXERCISE_NODE
    return THERAPEUTIC_RESPONSE_NODE

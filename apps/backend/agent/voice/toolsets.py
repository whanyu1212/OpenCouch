"""LiveKit Toolset groupings for OpenCouch voice capabilities."""

from __future__ import annotations

from livekit.agents.llm import Toolset

from agent.voice.tools import (
    answer_grounded_factual_lookup,
    cancel_memory_deletion,
    confirm_memory_deletion,
    prepare_indexed_memory_deletion,
    prepare_memory_deletion,
    provide_crisis_resources,
    select_memory_deletion_candidate,
    set_proactive_memory_recall,
    show_memory_status,
    show_saved_memory,
)


class MemoryControlToolset(Toolset):
    """Group explicit saved-memory tools for the therapeutic voice agent."""

    def __init__(self) -> None:
        super().__init__(
            id="memory_control",
            tools=[
                show_saved_memory,
                show_memory_status,
                set_proactive_memory_recall,
                prepare_memory_deletion,
                prepare_indexed_memory_deletion,
                select_memory_deletion_candidate,
                confirm_memory_deletion,
                cancel_memory_deletion,
            ],
        )


class GroundedLookupToolset(Toolset):
    """Group search-grounded factual lookup tools for safe voice turns."""

    def __init__(self) -> None:
        super().__init__(
            id="grounded_lookup",
            tools=[answer_grounded_factual_lookup],
        )


class CrisisResourceToolset(Toolset):
    """Group crisis-resource lookup tools for the crisis voice agent."""

    def __init__(self) -> None:
        super().__init__(
            id="crisis_resources",
            tools=[provide_crisis_resources],
        )

"""Runtime-only dependencies injected into the LangGraph workflow.

This module owns the :class:`WorkflowContext` TypedDict that the graph nodes
read via ``runtime.context``. It lives in its own module so both
:mod:`agent.graph` (which registers the context schema with the StateGraph)
and the node modules (which read from ``runtime.context``) can import it
without creating an import cycle.
"""

from __future__ import annotations

from typing import TypedDict

from agent.memory_graph import GraphMemoryStore
from agent.memory_profile import SqliteProfileMemoryStore
from services.llm.base import BaseLLMClient


class WorkflowContext(TypedDict):
    """Runtime-only dependencies injected into the LangGraph workflow."""

    llm_client: BaseLLMClient | None
    profile_memory_store: SqliteProfileMemoryStore
    graph_memory_store: GraphMemoryStore
    is_guest_mode: bool

"""Runtime-only dependencies injected into the LangGraph workflow.

This module owns the :class:`WorkflowContext` TypedDict that the graph nodes
read via ``runtime.context``. It lives in its own module so both
:mod:`agent.graph` (which registers the context schema with the StateGraph)
and the node modules (which read from ``runtime.context``) can import it
without creating an import cycle.
"""

from __future__ import annotations

from typing import TypedDict

from agent.memory.crisis_log import CrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from services.llm.base import BaseLLMClient


class WorkflowContext(TypedDict):
    """Runtime-only dependencies injected into the LangGraph workflow."""

    llm_client: BaseLLMClient | None
    # The unified memory store is the single entry point for long-term
    # memory reads and writes. It fans out internally to the right
    # namespace (semantic / episodic / procedural). Typed as the
    # protocol so either implementation can be injected —
    # :class:`OpenCouchMemoryStore` for in-memory (tests, incognito)
    # or :class:`SqliteMemoryStore` for persistent (v0.8+, local mode).
    # See ``agent/memory/store.py`` for the protocol definition.
    memory_store: MemoryStore
    # The always-on crisis safety log. Writes regardless of memory_mode
    # — see schema.yaml §2 namespaces.crisis_log for the privacy
    # asymmetry rationale.
    crisis_log_backend: CrisisLogBackend
    memory_mode: MemoryMode

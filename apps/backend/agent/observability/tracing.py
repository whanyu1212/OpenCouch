"""Tracing integration helpers for compiled LangGraph workflows."""

from __future__ import annotations

import os
from typing import TypeVar

CompiledWorkflow = TypeVar("CompiledWorkflow")


def apply_graph_tracing(compiled: CompiledWorkflow) -> CompiledWorkflow:
    """Wrap a compiled LangGraph workflow with tracing when configured.

    Args:
        compiled (CompiledWorkflow): Compiled graph workflow.

    Returns:
        CompiledWorkflow: The original workflow, or a tracing-wrapped workflow.
    """

    tracing_disabled = os.getenv("OPENCOUCH_DISABLE_TRACING", "").strip().lower()
    if (
        tracing_disabled not in {"1", "true", "yes", "on"}
        and os.getenv("OPIK_API_KEY")
        and os.getenv("OPIK_WORKSPACE")
    ):
        from opik.integrations.langchain import OpikTracer, track_langgraph

        project_name = os.getenv("OPIK_PROJECT_NAME") or "opencouch-dev"
        return track_langgraph(compiled, OpikTracer(project_name=project_name))

    return compiled

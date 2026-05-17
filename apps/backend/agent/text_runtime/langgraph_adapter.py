"""LangGraph-backed implementation of the text-agent adapter contract."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from agent.graph_constants import FINALIZE_TURN_NODE
from agent.runtime.streaming import (
    chunk_event_from_custom_payload,
    status_stage_for_node,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentGraphOutputState, AgentState
from agent.text_runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)

AgentWorkflow = CompiledStateGraph[
    AgentState,
    WorkflowContext,
    AgentGraphInputState,
    AgentGraphOutputState,
]

AgentWorkflowBuilder = Callable[..., AgentWorkflow]


class LangGraphTextAgentAdapter:
    """Text-agent adapter that delegates to the existing LangGraph workflow."""

    def __init__(self, workflow: AgentWorkflow) -> None:
        self.workflow = workflow

    async def get_state(self, config: RunnableConfig) -> AgentState | None:
        """Return the latest LangGraph checkpoint values for a thread."""

        snapshot = await self.workflow.aget_state(config)
        values = snapshot.values or None
        if values is None:
            return None
        return cast(AgentState, dict(values))

    async def run_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> Mapping[str, Any]:
        """Run one non-streaming turn through LangGraph."""

        del session
        return await self.workflow.ainvoke(
            initial_state,
            config=config,
            context=context,
        )

    async def run_turn_stream(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        """Run one streaming turn and normalize LangGraph stream chunks."""

        del session
        async for chunk in self.workflow.astream(
            initial_state,
            config=config,
            context=context,
            stream_mode=("custom", "updates", "values"),
            subgraphs=True,
            version="v2",
        ):
            if chunk["type"] == "custom":
                event = chunk_event_from_custom_payload(chunk["data"])
                if event is not None:
                    yield TextRuntimeChunkEvent(text=event.text)
            elif chunk["type"] == "updates" and chunk["ns"] == ():
                for node_name in chunk["data"]:
                    yield TextRuntimeStatusEvent(
                        stage=status_stage_for_node(node_name),
                        turn_finalized=node_name == FINALIZE_TURN_NODE,
                    )
            elif chunk["type"] == "values" and chunk["ns"] == ():
                yield TextRuntimeStateEvent(state=cast(AgentState, chunk["data"]))

    async def update_state(
        self,
        config: RunnableConfig,
        values: Mapping[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        """Persist a state update through the LangGraph checkpointer."""

        await self.workflow.aupdate_state(config, values, as_node=as_node)

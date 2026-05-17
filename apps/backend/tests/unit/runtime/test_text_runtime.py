"""Tests for the text-agent runtime adapter seam."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.graph_constants import FINALIZE_TURN_NODE
from agent.persistence import PersistentAgentRuntime
from agent.text_runtime import (
    DEFAULT_TEXT_AGENT_RUNTIME,
    LangGraphTextAgentAdapter,
    OpenAITextAgentAdapter,
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    create_text_agent_adapter,
    resolve_text_agent_runtime,
)
from agent.state import AgentGraphInputState


class _FakeWorkflow:
    """Small LangGraph-shaped workflow fake for adapter unit tests."""

    def __init__(self) -> None:
        self.ainvoke_calls: list[tuple[Any, dict[str, Any]]] = []

    async def aget_state(self, config: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(values={"response_text": "persisted"})

    async def ainvoke(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self.ainvoke_calls.append(
            (initial_state, {"config": config, "context": context})
        )
        return {"response_text": "done"}

    async def astream(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: dict[str, Any],
        context: Any,
        stream_mode: tuple[str, ...],
        subgraphs: bool,
        version: str,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "custom", "data": {"type": "chunk", "text": "hello"}}
        yield {
            "type": "updates",
            "ns": (),
            "data": {FINALIZE_TURN_NODE: {"transcript": []}},
        }
        yield {
            "type": "values",
            "ns": (),
            "data": {"response_text": "hello"},
        }

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        self.updated_state = {
            "config": config,
            "values": values,
            "as_node": as_node,
        }


def test_resolve_text_agent_runtime_defaults_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default text runtime should now be OpenAI."""

    monkeypatch.delenv("OPENCOUCH_TEXT_AGENT_RUNTIME", raising=False)

    assert DEFAULT_TEXT_AGENT_RUNTIME == "openai"
    assert resolve_text_agent_runtime() == "openai"


def test_resolve_text_agent_runtime_normalizes_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime selection should tolerate common env formatting noise."""

    monkeypatch.setenv("OPENCOUCH_TEXT_AGENT_RUNTIME", " LangGraph ")

    assert resolve_text_agent_runtime() == "langgraph"


def test_resolve_text_agent_runtime_accepts_openai_value() -> None:
    """The OpenAI runtime can be selected explicitly for hybrid testing."""

    assert resolve_text_agent_runtime("openai") == "openai"


def test_resolve_text_agent_runtime_rejects_unknown_value() -> None:
    """Unsupported runtimes should fail loudly before a turn starts."""

    with pytest.raises(ValueError, match="Supported values: langgraph, openai"):
        resolve_text_agent_runtime("unknown")


def test_persistent_runtime_accepts_openai_text_runtime() -> None:
    """PersistentAgentRuntime should validate the selector during construction."""

    runtime = PersistentAgentRuntime(text_agent_runtime="openai")

    assert runtime._text_agent_runtime == "openai"


def test_persistent_runtime_defaults_to_openai_text_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PersistentAgentRuntime should use OpenAI when no override is set."""

    monkeypatch.delenv("OPENCOUCH_TEXT_AGENT_RUNTIME", raising=False)

    runtime = PersistentAgentRuntime()

    assert runtime._text_agent_runtime == "openai"


def test_create_text_agent_adapter_builds_openai_adapter_by_default() -> None:
    """The factory should make OpenAI the default serving adapter."""

    workflow = _FakeWorkflow()

    adapter = create_text_agent_adapter(
        checkpointer=object(),
        graph_builder=lambda **_: workflow,  # type: ignore[arg-type]
    )

    assert isinstance(adapter, OpenAITextAgentAdapter)
    assert adapter.checkpoint_workflow is workflow


def test_create_text_agent_adapter_builds_langgraph_adapter() -> None:
    """The factory should keep LangGraph available as an explicit rollback."""

    workflow = _FakeWorkflow()

    adapter = create_text_agent_adapter(
        checkpointer=object(),
        graph_builder=lambda **_: workflow,  # type: ignore[arg-type]
        runtime_name="langgraph",
    )

    assert isinstance(adapter, LangGraphTextAgentAdapter)
    assert adapter.workflow is workflow


def test_create_text_agent_adapter_builds_openai_adapter() -> None:
    """The factory should wire OpenAI with a LangGraph checkpoint adapter."""

    workflow = _FakeWorkflow()

    adapter = create_text_agent_adapter(
        checkpointer=object(),
        graph_builder=lambda **_: workflow,  # type: ignore[arg-type]
        runtime_name="openai",
    )

    assert isinstance(adapter, OpenAITextAgentAdapter)
    assert adapter.checkpoint_workflow is workflow


@pytest.mark.asyncio
async def test_langgraph_adapter_normalizes_stream_events() -> None:
    """LangGraph stream chunks should map to provider-neutral runtime events."""

    adapter = LangGraphTextAgentAdapter(cast(Any, _FakeWorkflow()))

    events = [
        event
        async for event in adapter.run_turn_stream(
            cast(AgentGraphInputState, {}),
            config={},
            context=object(),  # type: ignore[arg-type]
        )
    ]

    assert events == [
        TextRuntimeChunkEvent(text="hello"),
        TextRuntimeStatusEvent(stage="finalize", turn_finalized=True),
        TextRuntimeStateEvent(state=cast(Any, {"response_text": "hello"})),
    ]

"""Unit tests for the voice recall_saved_memory tool helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent.voice.tools import _execute_recall_saved_memory


@dataclass
class _StubWorkflowContext:
    memory_store: object
    embedding_provider: object


@dataclass
class _StubToolContext:
    workflow_context: _StubWorkflowContext
    agent_state: dict[str, Any]


@dataclass
class _StubLoadResult:
    working_memory: list[dict[str, Any]]
    proactive_recall_enabled: bool
    summary: str = ""
    procedural_rules: list[Any] | None = None
    diagnostics: dict[str, Any] | None = None


@dataclass
class _StubProceduralProfile:
    proactive_recall_enabled: bool
    rules: list[Any] = field(default_factory=list)


def _build_context(*, owner_id: str = "alice") -> _StubToolContext:
    workflow_context = _StubWorkflowContext(
        memory_store=object(),
        embedding_provider=object(),
    )
    agent_state: dict[str, Any] = {
        "user_id": owner_id,
        "session_id": "voice-thread",
    }
    return _StubToolContext(
        workflow_context=workflow_context,
        agent_state=agent_state,
    )


def _install_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    proactive_recall_enabled: bool,
    entries: list[dict[str, Any]] | None = None,
    loader_must_not_run: bool = False,
) -> dict[str, int]:
    """Install stubbed procedural-profile + load_memory_for_turn fakes.

    Returns a counter dict so tests can assert which paths were exercised.
    """

    calls = {"profile": 0, "loader": 0}

    async def fake_profile(_store: object, *, user_id: str) -> _StubProceduralProfile:
        calls["profile"] += 1
        del user_id
        return _StubProceduralProfile(
            proactive_recall_enabled=proactive_recall_enabled,
        )

    async def fake_loader(**_: object) -> _StubLoadResult:
        calls["loader"] += 1
        if loader_must_not_run:
            raise AssertionError(
                "load_memory_for_turn must not be called in this scenario"
            )
        return _StubLoadResult(
            working_memory=list(entries or []),
            proactive_recall_enabled=proactive_recall_enabled,
        )

    monkeypatch.setattr(
        "agent.voice.tools.handlers.aget_procedural_profile", fake_profile
    )
    monkeypatch.setattr("agent.voice.tools.handlers.load_memory_for_turn", fake_loader)
    return calls


@pytest.mark.asyncio
async def test_recall_refuses_when_query_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context()
    calls = _install_stubs(
        monkeypatch,
        proactive_recall_enabled=True,
        loader_must_not_run=True,
    )

    result = await _execute_recall_saved_memory(context, {"query": ""})

    assert result["refused"] is True
    assert result["results"] == []
    assert "No recall query" in str(result["response_text"])
    # Empty query must short-circuit before touching the store at all.
    assert calls["profile"] == 0
    assert calls["loader"] == 0


@pytest.mark.parametrize(
    "hostile_query",
    [
        ["list", "of", "strings"],
        {"nested": "dict"},
        None,
        42,
        True,
    ],
)
@pytest.mark.asyncio
async def test_recall_refuses_when_query_is_not_a_string(
    monkeypatch: pytest.MonkeyPatch,
    hostile_query: object,
) -> None:
    """Defense-in-depth: non-string query types must refuse, not coerce.

    The Realtime function-call protocol declares query as a string, but
    the helper should not trust upstream validation. Coercing a list to
    its ``str()`` form would land "['list']" as a recall query, which is
    a silent correctness issue rather than a privacy one.
    """

    context = _build_context()
    calls = _install_stubs(
        monkeypatch,
        proactive_recall_enabled=True,
        loader_must_not_run=True,
    )

    result = await _execute_recall_saved_memory(context, {"query": hostile_query})

    assert result["refused"] is True
    assert result["results"] == []
    # Same response shape as the empty-query refusal — non-string is just
    # another "no usable query" case from the caller's perspective.
    assert "No recall query" in str(result["response_text"])
    assert calls["profile"] == 0
    assert calls["loader"] == 0


@pytest.mark.asyncio
async def test_recall_refuses_before_retrieval_when_recall_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-retrieval gating: profile fetch decides; loader never runs."""

    context = _build_context()
    calls = _install_stubs(
        monkeypatch,
        proactive_recall_enabled=False,
        entries=[{"evidence_quote": "should not surface"}],
        loader_must_not_run=True,
    )

    result = await _execute_recall_saved_memory(
        context,
        {"query": "work stress"},
    )

    assert result["refused"] is True
    assert result["reason"] == "proactive_recall_disabled"
    assert result["results"] == []
    assert "should not surface" not in str(result["response_text"])
    # The profile fetch ran, but load_memory_for_turn must not have.
    assert calls["profile"] == 1
    assert calls["loader"] == 0


@pytest.mark.asyncio
async def test_recall_returns_entries_when_proactive_recall_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context()
    _install_stubs(
        monkeypatch,
        proactive_recall_enabled=True,
        entries=[
            {"evidence_quote": "Berlin marathon training", "kind": "fact"},
            {"summary": "Session about sleep anxiety", "kind": "session"},
        ],
    )

    result = await _execute_recall_saved_memory(
        context,
        {"query": "running"},
    )

    assert result.get("refused") is None
    assert result["result_count"] == 2
    assert result["query"] == "running"
    snippets = {entry["snippet"] for entry in result["results"]}
    assert "Berlin marathon training" in snippets
    assert "Session about sleep anxiety" in snippets


@pytest.mark.asyncio
async def test_recall_clamps_limit_to_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context()
    many_entries = [{"evidence_quote": f"fact-{i}"} for i in range(20)]
    _install_stubs(
        monkeypatch,
        proactive_recall_enabled=True,
        entries=many_entries,
    )

    # User-supplied limit beyond the cap should clamp at _RECALL_MAX_LIMIT (10).
    capped = await _execute_recall_saved_memory(
        context,
        {"query": "many", "limit": 99},
    )
    # Negative / zero limits should clamp upward to 1.
    floored = await _execute_recall_saved_memory(
        context,
        {"query": "few", "limit": -1},
    )

    assert capped["result_count"] == 10
    assert floored["result_count"] == 1


@pytest.mark.asyncio
async def test_recall_defaults_limit_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _build_context()
    entries = [{"evidence_quote": f"fact-{i}"} for i in range(7)]
    _install_stubs(
        monkeypatch,
        proactive_recall_enabled=True,
        entries=entries,
    )

    result = await _execute_recall_saved_memory(context, {"query": "something"})

    # Default limit is 5; we returned 7 entries; result should be capped at 5.
    assert result["result_count"] == 5

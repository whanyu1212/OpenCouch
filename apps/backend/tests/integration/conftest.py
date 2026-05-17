"""Integration-test defaults for runtime migration coverage."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_legacy_text_runtime_for_integration_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep existing integration suites on LangGraph unless they opt into OpenAI."""

    monkeypatch.setenv("OPENCOUCH_TEXT_AGENT_RUNTIME", "langgraph")

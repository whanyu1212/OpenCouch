"""Tests for the OpenCouch Textual TUI shell."""

from __future__ import annotations

from typing import Any

import pytest

from agent.models import (
    AgentOutput,
    CrisisAssessment,
    DoneEvent,
    Message,
    MessageRole,
    ResponseCategory,
    ResponseReadyEvent,
    StatusEvent,
)
from opencouch_console.runtime import ConsoleSession


def test_tui_parser_defaults_to_dogfood_guest_mode() -> None:
    """The TUI should start in a credential-free dogfood smoke configuration."""

    from opencouch_tui.app import build_parser, config_from_args

    args = build_parser().parse_args([])
    config = config_from_args(args)

    assert args.view == "dogfood"
    assert args.theme == "light"
    assert config.requested_mode == "auto"
    assert config.memory_mode == "guest"
    assert config.thread_id.startswith("local-tui-")


def test_tui_parser_rejects_invalid_view() -> None:
    """Only the three designed workspaces should be accepted."""

    from opencouch_tui.app import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--view", "memory"])


def test_tui_parser_rejects_invalid_theme() -> None:
    """Only the supported visual modes should be accepted."""

    from opencouch_tui.app import build_parser

    assert build_parser().parse_args(["--theme", "dark"]).theme == "dark"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--theme", "sepia"])


@pytest.mark.asyncio
async def test_tui_switches_workspaces_with_keybindings() -> None:
    """The TUI should switch workspaces without relying on Mac F-keys."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.active_view == "dogfood"
        await pilot.press("tab")
        assert app.active_view == "debug"
        await pilot.press("tab")
        assert app.active_view == "chat"
        await pilot.press("shift+tab")
        assert app.active_view == "debug"
        await pilot.press("ctrl+1")
        assert app.active_view == "dogfood"
        await pilot.press("ctrl+3")
        assert app.active_view == "chat"


@pytest.mark.asyncio
async def test_tui_switches_between_light_and_dark_themes() -> None:
    """The TUI should expose a first-class light/dark visual mode toggle."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(
        initial_theme="light",
        runtime_factory=_fake_runtime_factory,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.active_theme == "light"
        assert app.has_class("theme-light")

        await pilot.press("f4")

        assert app.active_theme == "dark"
        assert app.has_class("theme-dark")
        assert not app.has_class("theme-light")


@pytest.mark.asyncio
async def test_tui_help_bar_uses_readable_shortcut_labels() -> None:
    """Keyboard help should advertise terminal-safe shortcuts."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        help_bar = str(app.query_one("#help-bar").renderable)

    assert "Tab Next" in help_bar
    assert "Shift+Tab Previous" in help_bar
    assert "Ctrl+1 Dogfood" in help_bar
    assert "Ctrl+2 Debug" in help_bar
    assert "Ctrl+3 Chat" in help_bar
    assert "Ctrl+Y Theme" in help_bar
    assert "F1" not in help_bar
    assert "F2" not in help_bar
    assert "F3" not in help_bar
    assert "^1" not in help_bar
    assert "^2" not in help_bar


@pytest.mark.asyncio
async def test_tui_shows_ascii_wordmark_as_empty_state_only() -> None:
    """Dogfood and Chat should get a restrained brand empty state."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        dogfood_widget = app.query_one("#dogfood-transcript")
        chat_widget = app.query_one("#chat-transcript")
        debug_widget = app.query_one("#debug-log")
        dogfood = str(dogfood_widget.renderable)
        chat = str(chat_widget.renderable)
        debug = str(debug_widget.renderable)

    assert " ___  _ __   ___ _ __   ___ ___" in dogfood
    assert " ___  _ __   ___ _ __   ___ ___" in chat
    assert ".----------------------------." not in dogfood
    assert ".----------------------------." not in chat
    assert "OPENCOUCH" not in debug
    assert dogfood_widget.has_class("empty-transcript")
    assert chat_widget.has_class("empty-transcript")
    assert not debug_widget.has_class("empty-transcript")


@pytest.mark.asyncio
async def test_tui_submits_turn_and_updates_workspaces() -> None:
    """A submitted deterministic turn should update chat, inspector, and debug panes."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.click("#message-input")
        app.query_one("#message-input").value = "hello from tui"
        await pilot.press("enter")
        await pilot.pause(0.1)

        dogfood = str(app.query_one("#dogfood-transcript").renderable)
        chat = str(app.query_one("#chat-transcript").renderable)
        inspector = str(app.query_one("#dogfood-inspector").renderable)
        debug = str(app.query_one("#debug-log").renderable)
        dogfood_empty = app.query_one("#dogfood-transcript").has_class(
            "empty-transcript"
        )
        chat_empty = app.query_one("#chat-transcript").has_class("empty-transcript")

    assert "🧑 You" in dogfood
    assert "OPENCOUCH" not in dogfood
    assert not dogfood_empty
    assert "hello from tui" in dogfood
    assert "🛋 OpenCouch" in dogfood
    assert "deterministic tui response" in dogfood
    assert "🧑 You" in chat
    assert "🛋 OpenCouch" in chat
    assert "OPENCOUCH" not in chat
    assert not chat_empty
    assert "deterministic tui response" in chat
    assert "route: therapeutic" in inspector
    assert "safety: normal" in inspector
    assert "status: deterministic" in debug
    assert "done" in debug


class _FakeConsoleRuntime:
    """Minimal async runtime used by TUI shell tests."""

    def __init__(self, config) -> None:
        self.config = config
        self.session = ConsoleSession(
            requested_mode=config.requested_mode,
            resolved_mode="deterministic",
            thread_id=config.thread_id,
            owner_id=config.thread_id,
            memory_mode=config.memory_mode,
            persistence_backend="sqlite",
            user_id=None,
            response_model_tier=config.response_model_tier,
            llm_client=None,
            response_llm_client=None,
            history=[],
            last_context=None,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def refresh(self) -> None:
        return None

    async def run_turn_stream(self, message: str):
        self.session.history.append(Message(role=MessageRole.USER, content=message))
        output = _make_agent_output("deterministic tui response")
        self.session.history.append(
            Message(role=MessageRole.ASSISTANT, content=output.response_text)
        )
        yield StatusEvent(stage="deterministic")
        yield ResponseReadyEvent(output=output)
        yield DoneEvent(output=output)


def _fake_runtime_factory(config: Any) -> _FakeConsoleRuntime:
    return _FakeConsoleRuntime(config)


def _make_agent_output(response_text: str) -> AgentOutput:
    return AgentOutput(
        response_text=response_text,
        response_type=ResponseCategory.THERAPEUTIC,
        crisis=CrisisAssessment(reason="normal"),
        response_style="deterministic_smoke",
        therapeutic_approach="none",
        diagnostics={
            "text_agent_runtime": "deterministic_smoke",
            "openai_text_runtime_mode": "deterministic_smoke",
        },
    )

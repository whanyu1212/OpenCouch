"""Tests for the OpenCouch Textual TUI shell."""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

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
from agent.runtime import ThreadSummary
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

    assert build_parser().parse_args(["--view", "memory"]).view == "memory"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--view", "operator"])


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
        await pilot.press("tab")
        assert app.active_view == "memory"
        await pilot.press("shift+tab")
        assert app.active_view == "chat"
        await pilot.press("shift+tab")
        assert app.active_view == "debug"
        await pilot.press("ctrl+1")
        assert app.active_view == "dogfood"
        await pilot.press("ctrl+4")
        assert app.active_view == "memory"


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
    assert "Ctrl+4 Memory" in help_bar
    assert "Ctrl+Y Theme" in help_bar
    assert "F1" not in help_bar
    assert "F2" not in help_bar
    assert "F3" not in help_bar
    assert "^1" not in help_bar
    assert "^2" not in help_bar
    assert "PageUp/PageDown Scroll" in help_bar
    assert "Home/End Jump" in help_bar


@pytest.mark.asyncio
async def test_tui_scrolls_active_transcript_from_input_focus(monkeypatch) -> None:
    """Transcript scroll keys should work even while the composer keeps focus."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#message-input")
        dogfood_widget = app.query_one("#dogfood-transcript-scroll")
        calls: list[str] = []

        monkeypatch.setattr(
            dogfood_widget,
            "scroll_page_up",
            lambda *, animate=True: calls.append(f"pageup:{animate}"),
        )
        monkeypatch.setattr(
            dogfood_widget,
            "scroll_page_down",
            lambda *, animate=True: calls.append(f"pagedown:{animate}"),
        )
        monkeypatch.setattr(
            dogfood_widget,
            "scroll_home",
            lambda *, animate=True: calls.append(f"home:{animate}"),
        )
        monkeypatch.setattr(
            dogfood_widget,
            "scroll_end",
            lambda *, animate=True: calls.append(f"end:{animate}"),
        )

        await pilot.press("pageup")
        await pilot.press("pagedown")
        await pilot.press("home")
        await pilot.press("end")

    assert calls == [
        "pageup:False",
        "pagedown:False",
        "home:False",
        "end:False",
    ]


@pytest.mark.asyncio
async def test_tui_memory_workspace_renders_snapshot() -> None:
    """The memory workspace should render semantic, episodic, and procedural memory."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+4")
        await pilot.pause()
        memory_text = str(app.query_one("#memory-transcript").renderable)

    assert "Owner: local-tui" in memory_text
    assert "Semantic" in memory_text
    assert "WORRIES_ABOUT: presentations" in memory_text
    assert "Episodic" in memory_text
    assert "session-1: User discussed presentation anxiety." in memory_text
    assert "Procedural" in memory_text
    assert "recall: on" in memory_text
    assert "You prefer short step-by-step plans." in memory_text


@pytest.mark.asyncio
async def test_tui_defers_transcript_autoscroll_until_after_refresh(
    monkeypatch,
) -> None:
    """Transcript updates should schedule auto-scroll after layout refresh."""

    from opencouch_tui.app import OpenCouchTuiApp, TranscriptEntry

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        dogfood_widget = app.query_one("#dogfood-transcript-scroll")
        calls: list[str] = []

        monkeypatch.setattr(
            dogfood_widget,
            "scroll_end",
            lambda *, animate=True: calls.append(f"scroll_end:{animate}"),
        )

        def fake_call_after_refresh(callback, *args, **kwargs):
            calls.append(f"deferred:{kwargs.get('animate', True)}")
            callback(*args, **kwargs)

        monkeypatch.setattr(app, "call_after_refresh", fake_call_after_refresh)

        app._update_conversation_transcript(
            "#dogfood-transcript",
            [TranscriptEntry(role="assistant", message="latest message")],
        )

    assert calls == ["deferred:False", "scroll_end:False"]


@pytest.mark.asyncio
async def test_tui_blank_chat_screens_have_no_wordmark() -> None:
    """Dogfood and Chat should stay visually blank until conversation starts."""

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

    assert " ___  _ __   ___ _ __   ___ ___" not in dogfood
    assert " ___  _ __   ___ _ __   ___ ___" not in chat
    assert ".----------------------------." not in dogfood
    assert ".----------------------------." not in chat
    assert "OPENCOUCH" not in debug
    assert dogfood == ""
    assert chat == ""
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

        dogfood = _renderable_text(app.query_one("#dogfood-transcript").renderable)
        chat = _renderable_text(app.query_one("#chat-transcript").renderable)
        inspector = str(app.query_one("#dogfood-inspector").renderable)
        debug = str(app.query_one("#debug-log").renderable)
        debug_state = str(app.query_one("#debug-state").renderable)
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
    assert "Routing trace" in debug_state
    assert "1. triage → safe" in debug_state
    assert "source: policy / 0.91" in debug_state
    assert "2. dispatch → therapeutic" in debug_state
    assert "reason: supportive counseling path" in debug_state
    assert "status: deterministic" in debug
    assert "done" in debug


@pytest.mark.asyncio
async def test_tui_resume_command_switches_active_thread() -> None:
    """The TUI should resume an existing thread from the composer."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.click("#message-input")
        app.query_one("#message-input").value = "/resume saved-thread"
        await pilot.press("enter")
        await pilot.pause(0.1)

        dogfood = _renderable_text(app.query_one("#dogfood-transcript").renderable)
        status = str(app.query_one("#status").renderable)
        debug = str(app.query_one("#debug-log").renderable)

    assert "saved-thread" in status
    assert "saved thread question" in dogfood
    assert "saved thread answer" in dogfood
    assert "resumed thread: saved-thread" in debug


@pytest.mark.asyncio
async def test_tui_new_command_creates_fresh_thread() -> None:
    """The TUI should create a new thread without keeping prior thread transcript."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.click("#message-input")
        app.query_one("#message-input").value = "/new fresh-thread"
        await pilot.press("enter")
        await pilot.pause(0.1)

        dogfood = _renderable_text(app.query_one("#dogfood-transcript").renderable)
        status = str(app.query_one("#status").renderable)
        debug = str(app.query_one("#debug-log").renderable)
        error = str(app.query_one("#error-banner").renderable)

    assert "fresh-thread" in status
    assert "saved thread question" not in dogfood
    assert "saved thread answer" not in dogfood
    assert "started new thread: fresh-thread" in debug
    assert error == ""


@pytest.mark.asyncio
async def test_tui_resume_command_shows_error_for_missing_thread() -> None:
    """The TUI should reject resuming a missing thread and keep the current one."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    initial_thread = app.config.thread_id
    async with app.run_test() as pilot:
        await pilot.click("#message-input")
        app.query_one("#message-input").value = "/resume missing-thread"
        await pilot.press("enter")
        await pilot.pause(0.1)

        status = str(app.query_one("#status").renderable)
        debug = str(app.query_one("#debug-log").renderable)
        error = str(app.query_one("#error-banner").renderable)

    assert initial_thread in status
    assert "missing-thread" in error
    assert "command error:" in debug


@pytest.mark.asyncio
async def test_tui_renders_assistant_markdown_without_raw_markers() -> None:
    """Assistant transcript entries should render Markdown instead of raw markers."""

    from opencouch_tui.app import OpenCouchTuiApp, TranscriptEntry

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        rendered = app._render_transcript(
            [
                TranscriptEntry(role="user", message="please format this"),
                TranscriptEntry(
                    role="assistant",
                    message="## Plan\n\n- **Breathe** slowly\n- Notice `one thing`",
                ),
            ]
        )

    output = _renderable_text(rendered)
    assert "Plan" in output
    assert "Breathe" in output
    assert "one thing" in output
    assert "## Plan" not in output
    assert "**Breathe**" not in output


@pytest.mark.asyncio
async def test_tui_safe_slash_commands_render_debug_output() -> None:
    """Safe TUI slash commands should render useful output in the debug log."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.click("#message-input")

        for command in ["/help", "/keys", "/status", "/context", "/debug state"]:
            app.query_one("#message-input").value = command
            await pilot.press("enter")
            await pilot.pause(0.05)

        dogfood = _renderable_text(app.query_one("#dogfood-transcript").renderable)
        chat = _renderable_text(app.query_one("#chat-transcript").renderable)
        debug = str(app.query_one("#debug-log").renderable)

    assert "⚙ Command" in dogfood
    assert "commands\n/help" in dogfood
    assert "/memory list [facts|sessions|rules]" in dogfood
    assert "status\nmode: deterministic" in chat
    assert "debug state\nthread_id:" in chat
    assert "commands\n/help" in debug
    assert "/memory list [facts|sessions|rules]" in debug
    assert "keys\nTab / Shift+Tab" in debug
    assert "status\nmode: deterministic" in debug
    assert "context\nNo persisted context yet." in debug
    assert "debug state\nthread_id:" in debug


@pytest.mark.asyncio
async def test_tui_thread_history_and_memory_commands_render_debug_output() -> None:
    """Thread, history, and memory list commands should be available in the TUI."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.click("#message-input")

        app.query_one("#message-input").value = "hello from tui"
        await pilot.press("enter")
        await pilot.pause(0.1)

        for command in [
            "/threads 2",
            "/history 2",
            "/memory status",
            "/memory list facts",
            "/memory list sessions",
            "/memory list rules",
        ]:
            app.query_one("#message-input").value = command
            await pilot.press("enter")
            await pilot.pause(0.05)

        dogfood = _renderable_text(app.query_one("#dogfood-transcript").renderable)
        chat = _renderable_text(app.query_one("#chat-transcript").renderable)
        debug = str(app.query_one("#debug-log").renderable)

    assert "threads\n* local-tui" in dogfood
    assert "memory list rules\nowner: local-tui" in chat
    assert "You prefer short step-by-step plans." in chat
    assert "threads\n* local-tui" in debug
    assert "saved-thread" in debug
    assert "history\nuser: hello from tui" in debug
    assert "assistant: deterministic tui response" in debug
    assert "memory status\nowner: local-tui" in debug
    assert "facts: 1" in debug
    assert "memory list facts\nowner: local-tui" in debug
    assert "WORRIES_ABOUT: presentations" in debug
    assert "memory list sessions\nowner: local-tui" in debug
    assert "session-1: User discussed presentation anxiety." in debug
    assert "memory list rules\nowner: local-tui" in debug
    assert "You prefer short step-by-step plans." in debug


@pytest.mark.asyncio
async def test_tui_input_suggests_slash_commands() -> None:
    """The composer should provide inline discovery for slash commands."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        message_input = app.query_one("#message-input")
        suggester = message_input.suggester

    assert suggester is not None
    assert await suggester.get_suggestion("/st") == "/status"
    assert await suggester.get_suggestion("/memory l") == "/memory list"
    assert await suggester.get_suggestion("/memory list s") == "/memory list sessions"


@pytest.mark.asyncio
async def test_tui_shows_lightweight_slash_command_popup() -> None:
    """Typing slash commands should show a filtered command discovery popup."""

    from opencouch_tui.app import OpenCouchTuiApp

    app = OpenCouchTuiApp(runtime_factory=_fake_runtime_factory)
    async with app.run_test() as pilot:
        await pilot.pause()
        palette = app.query_one("#command-palette")

        assert not palette.display

        app._render_command_palette("/")
        all_commands = str(palette.renderable)

        app._render_command_palette("/memory l")
        memory_commands = str(palette.renderable)

        app._render_command_palette("hello")
        hidden_after_plain_text = palette.display

    assert "Slash commands" in all_commands
    assert "/status" in all_commands
    assert "/memory list facts" in memory_commands
    assert "/memory list sessions" in memory_commands
    assert "/status" not in memory_commands
    assert not hidden_after_plain_text


class _FakeConsoleRuntime:
    """Minimal async runtime used by TUI shell tests."""

    def __init__(self, config) -> None:
        self.config = config
        owner_id = config.user_id or config.thread_id
        self.session = ConsoleSession(
            requested_mode=config.requested_mode,
            resolved_mode="deterministic",
            thread_id=config.thread_id,
            owner_id=owner_id,
            memory_mode=config.memory_mode,
            persistence_backend="sqlite",
            user_id=config.user_id,
            response_model_tier=config.response_model_tier,
            llm_client=None,
            response_llm_client=None,
            history=[],
            last_context=None,
        )
        self._histories: dict[str, list[Message]] = {
            config.thread_id: [],
            "saved-thread": [
                Message(role=MessageRole.USER, content="saved thread question"),
                Message(role=MessageRole.ASSISTANT, content="saved thread answer"),
            ],
        }

    async def __aenter__(self):
        await self.refresh()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def refresh(self) -> None:
        self.session.history = list(self._histories.get(self.session.thread_id, []))
        self.session.last_context = None

    async def thread_exists(self, thread_id: str) -> bool:
        return thread_id in self._histories

    async def switch_thread(
        self,
        thread_id: str,
        *,
        require_existing: bool = False,
    ) -> bool:
        if require_existing and thread_id not in self._histories:
            return False
        self._histories.setdefault(thread_id, [])
        self.session.thread_id = thread_id
        await self.refresh()
        return True

    async def list_threads(self, *, limit: int = 12) -> list[ThreadSummary]:
        summaries = [
            ThreadSummary(
                thread_id=thread_id,
                turn_count=len(history) // 2,
                message_count=len(history),
                has_context=False,
            )
            for thread_id, history in self._histories.items()
        ]
        return summaries[:limit]

    async def load_memory_snapshot(self) -> dict[str, Any]:
        return {
            "owner_id": self.session.owner_id,
            "semantic": [
                {
                    "predicate": "WORRIES_ABOUT",
                    "object": {"identifier": "presentations"},
                }
            ],
            "episodic": [
                {
                    "session_id": "session-1",
                    "summary": "User discussed presentation anxiety.",
                }
            ],
            "procedural": {
                "proactive_recall_enabled": True,
                "rules": [
                    {"rule": "You prefer short step-by-step plans."},
                ],
            },
        }

    async def run_turn_stream(self, message: str):
        history = self._histories.setdefault(self.session.thread_id, [])
        history.append(Message(role=MessageRole.USER, content=message))
        output = _make_agent_output("deterministic tui response")
        history.append(
            Message(role=MessageRole.ASSISTANT, content=output.response_text)
        )
        self.session.history = list(history)
        yield StatusEvent(stage="deterministic")
        yield ResponseReadyEvent(output=output)
        yield DoneEvent(output=output)


def _fake_runtime_factory(config: Any) -> _FakeConsoleRuntime:
    return _FakeConsoleRuntime(config)


def _renderable_text(renderable: Any) -> str:
    console = Console(width=100, record=True)
    console.print(renderable)
    return console.export_text()


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
            "routing_trace": [
                {
                    "stage": "triage",
                    "decision": "safe",
                    "source": "policy",
                    "reason": "No immediate crisis signals detected.",
                    "confidence": 0.91,
                },
                {
                    "stage": "dispatch",
                    "decision": "therapeutic",
                    "source": "router",
                    "reason": "supportive counseling path",
                },
            ],
        },
    )

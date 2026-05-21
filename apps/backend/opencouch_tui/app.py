"""Textual TUI entry point for OpenCouch local dogfooding."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.events import Key
from textual.widgets import Header, Input, Static

from agent.models import ChunkEvent, DoneEvent, ResponseReadyEvent, StatusEvent
from agent.runtime import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
)
from opencouch_console.runtime import (
    ConsoleConfig,
    ConsoleErrorEvent,
    ConsoleRuntime,
)

TuiView = Literal["dogfood", "debug", "chat"]
TuiTheme = Literal["light", "dark"]
TuiRole = Literal["user", "assistant"]
RuntimeFactory = Callable[[ConsoleConfig], ConsoleRuntime]
VIEW_ORDER: tuple[TuiView, ...] = ("dogfood", "debug", "chat")
OPENCOUCH_WORDMARK = (
    r"  ___  _ __   ___ _ __   ___ ___  _   _  ___| |__  ",
    r" / _ \| '_ \ / _ \ '_ \ / __/ _ \| | | |/ __| '_ \ ",
    r"| (_) | |_) |  __/ | | | (_| (_) | |_| | (__| | | |",
    r" \___/| .__/ \___|_| |_|\___\___/ \__,_|\___|_| |_|",
    r"      |_|                                           ",
)


@dataclass
class TranscriptEntry:
    """One visible chat entry in the TUI transcript."""

    role: TuiRole
    message: str


def build_parser() -> argparse.ArgumentParser:
    """Build the OpenCouch TUI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the experimental OpenCouch Textual TUI.",
    )
    parser.add_argument(
        "--mode",
        choices=["deterministic", "hybrid", "auto"],
        default="auto",
        help="How to resolve the LLM client for crisis classification.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Stable thread identifier to resume a prior local conversation.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Stable owner identifier for persistent long-term memory.",
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(DEFAULT_THREAD_DB_PATH),
        help="Legacy SQLite path for persisted session state.",
    )
    parser.add_argument(
        "--memory-sqlite-path",
        default=str(DEFAULT_MEMORY_DB_PATH),
        help="Legacy SQLite path for local memory storage.",
    )
    parser.add_argument(
        "--crisis-log-sqlite-path",
        default=str(DEFAULT_CRISIS_LOG_DB_PATH),
        help="Legacy SQLite path for crisis audit storage.",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["guest", "persistent"],
        default="guest",
        help="Local memory behavior for the TUI.",
    )
    parser.add_argument(
        "--response-model-tier",
        "--response-tier",
        dest="response_model_tier",
        choices=["fast", "quality"],
        default="fast",
        help="Text response tier for therapeutic prose generation.",
    )
    parser.add_argument(
        "--view",
        choices=["dogfood", "debug", "chat"],
        default="dogfood",
        help="Initial TUI workspace.",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default="light",
        help="Initial TUI color theme.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ConsoleConfig:
    """Convert parsed TUI args into a console runtime config."""

    thread_id = args.thread_id or f"local-tui-{uuid4().hex[:8]}"
    return ConsoleConfig(
        requested_mode=args.mode,
        thread_id=thread_id,
        user_id=args.user_id,
        response_model_tier=args.response_model_tier,
        sqlite_path=args.sqlite_path,
        memory_mode=args.memory_mode,
        memory_sqlite_path=args.memory_sqlite_path,
        crisis_log_sqlite_path=args.crisis_log_sqlite_path,
    )


class OpenCouchTuiApp(App[None]):
    """Experimental Textual app for OpenCouch local dogfooding."""

    CSS = """
    Screen {
        layout: vertical;
        background: #f5f3ee;
        color: #24302d;
    }

    .theme-light {
        background: #f5f3ee;
        color: #24302d;
    }

    .theme-dark {
        background: #101815;
        color: #e5efe9;
    }

    .theme-light Header {
        background: #e4f2ed;
        color: #183b34;
    }

    .theme-dark Header {
        background: #0b1210;
        color: #d9f2e6;
    }

    #tabs {
        height: 3;
        padding: 1 2 0 2;
        text-style: bold;
    }

    .theme-light #tabs {
        background: #f5f3ee;
        color: #5c6762;
    }

    .theme-dark #tabs {
        background: #101815;
        color: #9fb3aa;
    }

    #status {
        height: 1;
        padding: 0 2;
    }

    .theme-light #status {
        color: #6a746f;
    }

    .theme-dark #status {
        color: #9fb3aa;
    }

    #error-banner {
        height: auto;
        padding: 0 2;
        color: $error;
    }

    #body {
        height: 1fr;
        padding: 1 2;
    }

    #dogfood-view,
    #debug-view,
    #chat-view {
        height: 1fr;
    }

    #dogfood-transcript,
    #chat-transcript,
    #debug-log,
    #debug-state {
        height: 1fr;
        padding: 1 2;
        border: round #d9d6ce;
        overflow-y: auto;
    }

    .empty-transcript {
        content-align: center middle;
    }

    .theme-light #dogfood-transcript,
    .theme-light #chat-transcript,
    .theme-light #debug-log,
    .theme-light #debug-state {
        background: #fffdf8;
        color: #24302d;
        border: round #d9d6ce;
    }

    .theme-dark #dogfood-transcript,
    .theme-dark #chat-transcript,
    .theme-dark #debug-log,
    .theme-dark #debug-state {
        background: #16211e;
        color: #e5efe9;
        border: round #2d4039;
    }

    #dogfood-inspector {
        width: 36;
        padding: 1 2;
        border: round #d9d6ce;
        margin-left: 1;
    }

    .theme-light #dogfood-inspector {
        background: #eef5f1;
        color: #56635d;
        border: round #d1ded7;
    }

    .theme-dark #dogfood-inspector {
        background: #111d19;
        color: #a9bab1;
        border: round #2f443c;
    }

    #message-input {
        margin: 0 2;
        border: round #d9d6ce;
    }

    .theme-light #message-input {
        background: #fffdf8;
        color: #24302d;
        border: round #c8d8d0;
    }

    .theme-dark #message-input {
        background: #111d19;
        color: #e5efe9;
        border: round #2f443c;
    }

    #help-bar {
        height: 1;
        padding: 0 2;
    }

    .theme-light #help-bar {
        background: #e8eee9;
        color: #5f6b65;
    }

    .theme-dark #help-bar {
        background: #0e1714;
        color: #a4b5ad;
    }
    """

    BINDINGS = [
        ("tab", "next_view", "Next view"),
        ("shift+tab", "previous_view", "Previous view"),
        ("ctrl+1", "show_dogfood", "Dogfood"),
        ("ctrl+2", "show_debug", "Debug"),
        ("ctrl+3", "show_chat", "Chat"),
        ("f1", "show_dogfood", "Dogfood"),
        ("f2", "show_debug", "Debug"),
        ("f3", "show_chat", "Chat"),
        ("f4", "toggle_theme", "Theme"),
        ("ctrl+y", "toggle_theme", "Theme"),
        ("ctrl+l", "clear_view", "Clear"),
        ("ctrl+t", "toggle_diagnostics", "Diagnostics"),
        ("escape", "blur_focus", "Blur"),
    ]

    def __init__(
        self,
        *,
        config: ConsoleConfig | None = None,
        initial_view: TuiView = "dogfood",
        initial_theme: TuiTheme = "light",
        runtime_factory: RuntimeFactory = ConsoleRuntime,
    ) -> None:
        super().__init__()
        self.config = config or ConsoleConfig(
            thread_id=f"local-tui-{uuid4().hex[:8]}",
        )
        self.active_view: TuiView = initial_view
        self.active_theme: TuiTheme = initial_theme
        self._runtime_factory = runtime_factory
        self._runtime: ConsoleRuntime | None = None
        self._diagnostics_visible = True
        self._dogfood_entries: list[TranscriptEntry] = []
        self._chat_entries: list[TranscriptEntry] = []
        self._debug_lines: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""

        yield Header(show_clock=False)
        yield Static(id="tabs")
        yield Static(id="status")
        yield Static("", id="error-banner")
        with Container(id="body"):
            with Horizontal(id="dogfood-view"):
                yield Static("", id="dogfood-transcript")
                yield Static("", id="dogfood-inspector")
            with Vertical(id="debug-view"):
                yield Static("", id="debug-log")
                yield Static("", id="debug-state")
            with Vertical(id="chat-view"):
                yield Static("", id="chat-transcript")
        yield Input(placeholder="Message OpenCouch...", id="message-input")
        yield Static(id="help-bar")

    async def on_mount(self) -> None:
        """Initialize runtime and visible workspace state."""

        self._runtime = await self._runtime_factory(self.config).__aenter__()
        self._apply_theme()
        self._render_status()
        self._render_inspector(None)
        self._render_transcripts()
        self._show_view(self.active_view)
        self._render_help_bar()
        self.query_one("#message-input", Input).focus()

    async def on_unmount(self) -> None:
        """Close the runtime context when the TUI exits."""

        if self._runtime is not None:
            await self._runtime.__aexit__(None, None, None)
            self._runtime = None

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Submit a user turn from the bottom input bar."""

        message = event.value.strip()
        event.input.value = ""
        if not message:
            return
        self.query_one("#error-banner", Static).update("")
        self._append_user_message(message)
        self._render_transcripts()
        self.run_worker(self._run_turn(message), exclusive=False)

    def on_key(self, event: Key) -> None:
        """Keep workspace cycling available while the composer is focused."""

        if event.key == "tab":
            event.prevent_default()
            event.stop()
            self.action_next_view()
        elif event.key == "shift+tab":
            event.prevent_default()
            event.stop()
            self.action_previous_view()

    async def _run_turn(self, message: str) -> None:
        runtime = self._require_runtime()
        accumulated = ""
        response_ready = False
        async for event in runtime.run_turn_stream(message):
            if isinstance(event, StatusEvent):
                self._append_debug(f"status: {event.stage}")
            elif isinstance(event, ChunkEvent):
                accumulated += event.text
                if not response_ready:
                    self._render_pending_assistant(accumulated)
            elif isinstance(event, ResponseReadyEvent):
                response_ready = True
                self._replace_or_append_assistant(event.output.response_text)
                self._render_inspector(event.output)
                self._append_debug("response_ready")
            elif isinstance(event, DoneEvent):
                if not response_ready:
                    self._replace_or_append_assistant(event.output.response_text)
                self._render_inspector(event.output)
                self._append_debug("done")
            elif isinstance(event, ConsoleErrorEvent):
                self.query_one("#error-banner", Static).update(event.message)
                self._append_debug(f"error: {event.message}")
            self._render_transcripts()
        self._render_status()

    def action_show_dogfood(self) -> None:
        """Switch to Dogfood workspace."""

        self._show_view("dogfood")

    def action_show_debug(self) -> None:
        """Switch to Debug workspace."""

        self._show_view("debug")

    def action_show_chat(self) -> None:
        """Switch to Chat workspace."""

        self._show_view("chat")

    def action_next_view(self) -> None:
        """Switch to the next workspace."""

        self._show_relative_view(1)

    def action_previous_view(self) -> None:
        """Switch to the previous workspace."""

        self._show_relative_view(-1)

    def action_toggle_theme(self) -> None:
        """Switch between light and dark visual themes."""

        self.active_theme = "dark" if self.active_theme == "light" else "light"
        self._apply_theme()
        self._render_tabs()
        self._render_status()
        self._render_help_bar()
        self._render_transcripts()

    def action_clear_view(self) -> None:
        """Clear the currently visible local pane."""

        if self.active_view == "debug":
            self._debug_lines.clear()
        else:
            self._dogfood_entries.clear()
            self._chat_entries.clear()
        self._render_transcripts()

    def action_toggle_diagnostics(self) -> None:
        """Toggle diagnostic side panels where relevant."""

        self._diagnostics_visible = not self._diagnostics_visible
        self.query_one("#dogfood-inspector").display = (
            self._diagnostics_visible and self.active_view == "dogfood"
        )
        self.query_one("#debug-state").display = (
            self._diagnostics_visible and self.active_view == "debug"
        )

    def action_blur_focus(self) -> None:
        """Return focus to the app shell."""

        self.screen.focus(None)

    def _show_view(self, view: TuiView) -> None:
        self.active_view = view
        self.query_one("#dogfood-view").display = view == "dogfood"
        self.query_one("#debug-view").display = view == "debug"
        self.query_one("#chat-view").display = view == "chat"
        self.query_one("#dogfood-inspector").display = (
            self._diagnostics_visible and view == "dogfood"
        )
        self.query_one("#debug-state").display = (
            self._diagnostics_visible and view == "debug"
        )
        self._render_tabs()
        self._render_help_bar()

    def _show_relative_view(self, offset: int) -> None:
        index = VIEW_ORDER.index(self.active_view)
        self._show_view(VIEW_ORDER[(index + offset) % len(VIEW_ORDER)])

    def _append_user_message(self, message: str) -> None:
        entry = TranscriptEntry(role="user", message=message)
        self._dogfood_entries.append(entry)
        self._chat_entries.append(entry)
        self._append_debug(f"🧑 You: {message}")

    def _replace_or_append_assistant(self, message: str) -> None:
        entry = TranscriptEntry(role="assistant", message=message)
        if self._dogfood_entries and self._dogfood_entries[-1].role == "assistant":
            self._dogfood_entries[-1] = entry
        else:
            self._dogfood_entries.append(entry)
        if self._chat_entries and self._chat_entries[-1].role == "assistant":
            self._chat_entries[-1] = entry
        else:
            self._chat_entries.append(entry)

    def _render_pending_assistant(self, message: str) -> None:
        if message:
            self._replace_or_append_assistant(message)

    def _append_debug(self, line: str) -> None:
        self._debug_lines.append(line)

    def _render_tabs(self) -> None:
        labels = {
            "dogfood": "Dogfood",
            "debug": "Debug",
            "chat": "Chat",
        }
        tabs = Text("OpenCouch TUI  ", style=self._muted_style())
        for view in VIEW_ORDER:
            text = f" {labels[view]} "
            if view == self.active_view:
                tabs.append(text, style=self._active_tab_style())
            else:
                tabs.append(text, style=self._muted_style())
            tabs.append("  ")
        self.query_one("#tabs", Static).update(tabs)

    def _render_status(self) -> None:
        runtime = self._runtime
        session = runtime.session if runtime is not None else None
        if session is None:
            self.query_one("#status", Static).update("starting OpenCouch TUI")
            return
        self.query_one("#status", Static).update(
            "  ".join(
                [
                    f"mode {session.resolved_mode}",
                    f"memory {session.memory_mode}",
                    f"theme {self.active_theme}",
                    f"thread {session.thread_id}",
                    f"owner {session.owner_id}",
                    f"response {session.response_model_tier}",
                ]
            )
        )

    def _render_inspector(self, output) -> None:
        runtime = self._runtime
        session = runtime.session if runtime is not None else None
        lines = []
        if session is not None:
            lines.extend(
                [
                    "OpenCouch dogfood",
                    f"mode: {session.resolved_mode}",
                    f"memory: {session.memory_mode}",
                    f"theme: {self.active_theme}",
                    f"owner: {session.owner_id}",
                    f"response: {session.response_model_tier}",
                ]
            )
        if output is not None:
            route = (output.diagnostics or {}).get("openai_text_runtime_mode")
            lines.extend(
                [
                    "",
                    f"route: {output.response_type.value}",
                    f"runtime: {route or '-'}",
                    f"safety: {'crisis' if output.crisis.level >= 2 else 'normal'}",
                    f"style: {output.response_style or '-'}",
                ]
            )
        self.query_one("#dogfood-inspector", Static).update("\n".join(lines))
        self.query_one("#debug-state", Static).update("\n".join(lines))

    def _render_transcripts(self) -> None:
        self._update_conversation_transcript(
            "#dogfood-transcript",
            self._dogfood_entries,
        )
        self._update_conversation_transcript(
            "#chat-transcript",
            self._chat_entries,
        )
        self.query_one("#debug-log", Static).update(self._render_debug_log())

    def _update_conversation_transcript(
        self,
        selector: str,
        entries: list[TranscriptEntry],
    ) -> None:
        transcript = self.query_one(selector, Static)
        if entries:
            transcript.remove_class("empty-transcript")
        else:
            transcript.add_class("empty-transcript")
        transcript.update(self._render_transcript(entries))

    def _render_transcript(self, entries: list[TranscriptEntry]) -> Text:
        if not entries:
            return self._render_empty_transcript()

        transcript = Text()
        for index, entry in enumerate(entries):
            role_label = "🧑 You" if entry.role == "user" else "🛋 OpenCouch"
            transcript.append(role_label, style=self._role_style(entry.role))
            transcript.append("\n")
            transcript.append(entry.message, style=self._message_style(entry.role))
            if index != len(entries) - 1:
                transcript.append("\n\n")
        return transcript

    def _render_empty_transcript(self) -> Text:
        banner = Text()
        for index, line in enumerate(OPENCOUCH_WORDMARK):
            banner.append(line, style=self._empty_state_style())
            if index != len(OPENCOUCH_WORDMARK) - 1:
                banner.append("\n")
        return banner

    def _render_debug_log(self) -> Text:
        debug = Text()
        for index, line in enumerate(self._debug_lines):
            debug.append(line, style=self._debug_style())
            if index != len(self._debug_lines) - 1:
                debug.append("\n")
        return debug

    def _render_help_bar(self) -> None:
        help_text = Text()
        help_text.append("Tab Next", style=self._shortcut_style())
        help_text.append("   Shift+Tab Previous", style=self._shortcut_style())
        help_text.append("   Ctrl+1 Dogfood", style=self._shortcut_style())
        help_text.append("   Ctrl+2 Debug", style=self._shortcut_style())
        help_text.append("   Ctrl+3 Chat", style=self._shortcut_style())
        help_text.append(
            f"   Ctrl+Y Theme {self.active_theme}", style=self._shortcut_style()
        )
        help_text.append("   Ctrl+T Diagnostics", style=self._shortcut_style())
        help_text.append("   Ctrl+L Clear", style=self._shortcut_style())
        help_text.append("   Esc Blur", style=self._shortcut_style())
        self.query_one("#help-bar", Static).update(help_text)

    def _apply_theme(self) -> None:
        for target in (self, self.screen):
            target.remove_class("theme-light")
            target.remove_class("theme-dark")
            target.add_class(f"theme-{self.active_theme}")

    def _role_style(self, role: TuiRole) -> str:
        if self.active_theme == "dark":
            return "bold #8bd4ff" if role == "user" else "bold #9ee6b6"
        return "bold #2563eb" if role == "user" else "bold #047857"

    def _message_style(self, role: TuiRole) -> str:
        if self.active_theme == "dark":
            return "#eaf3ef on #182823" if role == "user" else "#eff7f2 on #14241c"
        return "#23302d on #edf7ff" if role == "user" else "#23302d on #edf8f1"

    def _debug_style(self) -> str:
        return "#9fb3aa" if self.active_theme == "dark" else "#626d68"

    def _empty_state_style(self) -> str:
        return "bold #7fa697" if self.active_theme == "dark" else "bold #7f998f"

    def _muted_style(self) -> str:
        return "#98aaa1" if self.active_theme == "dark" else "#65716b"

    def _active_tab_style(self) -> str:
        if self.active_theme == "dark":
            return "bold #dff8ec on #23443a"
        return "bold #173f35 on #d7ece5"

    def _shortcut_style(self) -> str:
        return "#a9bab1" if self.active_theme == "dark" else "#5f6b65"

    def _require_runtime(self) -> ConsoleRuntime:
        if self._runtime is None:
            raise RuntimeError("OpenCouch TUI runtime has not started.")
        return self._runtime


def main(argv: list[str] | None = None) -> int:
    """Run the OpenCouch Textual TUI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    app = OpenCouchTuiApp(
        config=config,
        initial_view=args.view,
        initial_theme=args.theme,
    )
    app.run()
    return 0

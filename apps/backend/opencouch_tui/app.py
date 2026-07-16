"""Textual TUI entry point for OpenCouch local dogfooding."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.suggester import SuggestFromList
from textual.widgets import Header, Input, Static

from agent.models import (
    AgentOutput,
    ChunkEvent,
    DoneEvent,
    MessageRole,
    ResponseReadyEvent,
    StatusEvent,
)
from agent.observability.routing_trace import routing_trace_from_diagnostics
from opencouch_tui.runtime import (
    ConsoleConfig,
    ConsoleErrorEvent,
    ConsoleRuntime,
)
from opencouch_tui.parser import add_common_args
from opencouch_tui.command_helpers import (
    format_history_plain,
    format_memory_snapshot_plain,
    format_memory_status_plain,
    format_thread_summaries_plain,
    search_history_messages,
    search_memory_snapshot,
)
from opencouch_tui.dispatch.shared import (
    get_history_command_limit,
    get_history_window,
    get_threads_command_summaries,
    parse_memory_overview_command,
    parse_search_command,
)
from opencouch_tui.commands import SLASH_COMMANDS, SlashCommand
from opencouch_tui.presenters import (
    session_status_bar_parts,
    session_status_command_lines,
)

TuiView = Literal["dogfood", "debug", "chat", "memory"]
TuiTheme = Literal["light", "dark"]
TuiRole = Literal["user", "assistant", "system"]
RuntimeFactory = Callable[[ConsoleConfig], ConsoleRuntime]
VIEW_ORDER: tuple[TuiView, ...] = ("dogfood", "debug", "chat", "memory")

_TUI_SUPPORTED_COMMAND_PATHS: tuple[tuple[str, ...], ...] = (
    ("/help",),
    ("/keys",),
    ("/status",),
    ("/history",),
    ("/context",),
    ("/search",),
    ("/search", "history"),
    ("/search", "memory"),
    ("/search", "all"),
    ("/clear",),
    ("/threads",),
    ("/new",),
    ("/resume",),
    ("/memory", "status"),
    ("/memory", "list"),
    ("/memory", "list", "facts"),
    ("/memory", "list", "sessions"),
    ("/memory", "list", "rules"),
    ("/debug", "state"),
)


def _command_suggestion_text(command: str) -> str:
    return (
        command.replace(" [n]", "")
        .replace(" [thread-id]", "")
        .replace(" <thread-id>", "")
        .replace(" [facts|sessions|rules]", "")
    )


TUI_SLASH_COMMANDS: tuple[SlashCommand, ...] = tuple(
    command
    for command in SLASH_COMMANDS
    if command.path in _TUI_SUPPORTED_COMMAND_PATHS
)
TUI_SLASH_SUGGESTIONS: tuple[str, ...] = tuple(
    _command_suggestion_text(command.display) for command in TUI_SLASH_COMMANDS
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
    add_common_args(parser)
    parser.add_argument(
        "--memory-mode",
        choices=["guest", "persistent"],
        default="guest",
        help="Local memory behavior for the TUI.",
    )
    parser.add_argument(
        "--view",
        choices=["dogfood", "debug", "chat", "memory"],
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
    #chat-view,
    #memory-view {
        height: 1fr;
    }

    #dogfood-transcript-scroll,
    #chat-transcript-scroll,
    #debug-log-scroll,
    #debug-state-scroll,
    #memory-transcript-scroll {
        height: 1fr;
        border: round #d9d6ce;
        overflow-y: auto;
    }

    #dogfood-transcript,
    #chat-transcript,
    #debug-log,
    #debug-state,
    #memory-transcript {
        height: auto;
        min-height: 100%;
        padding: 1 2;
    }

    .empty-transcript {
        content-align: center middle;
    }

    .theme-light #dogfood-transcript-scroll,
    .theme-light #chat-transcript-scroll,
    .theme-light #debug-log-scroll,
    .theme-light #debug-state-scroll,
    .theme-light #memory-transcript-scroll {
        background: #fffdf8;
        color: #24302d;
        border: round #d9d6ce;
    }

    .theme-dark #dogfood-transcript-scroll,
    .theme-dark #chat-transcript-scroll,
    .theme-dark #debug-log-scroll,
    .theme-dark #debug-state-scroll,
    .theme-dark #memory-transcript-scroll {
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

    #command-palette {
        height: auto;
        max-height: 12;
        margin: 0 2;
        padding: 1 2;
        border: round #d9d6ce;
    }

    .theme-light #command-palette {
        background: #fff7df;
        color: #3f3420;
        border: round #d9c99f;
    }

    .theme-dark #command-palette {
        background: #2a2214;
        color: #f5e7c8;
        border: round #705c2d;
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
        ("ctrl+4", "show_memory", "Memory"),
        ("f1", "show_dogfood", "Dogfood"),
        ("f2", "show_debug", "Debug"),
        ("f3", "show_chat", "Chat"),
        ("f4", "toggle_theme", "Theme"),
        ("ctrl+y", "toggle_theme", "Theme"),
        ("ctrl+l", "clear_view", "Clear"),
        ("ctrl+r", "refresh_memory", "Refresh memory"),
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
        self._memory_text = Text("Loading memory…", style=self._muted_style())

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""

        yield Header(show_clock=False)
        yield Static(id="tabs")
        yield Static(id="status")
        yield Static("", id="error-banner")
        with Container(id="body"):
            with Horizontal(id="dogfood-view"):
                with ScrollableContainer(id="dogfood-transcript-scroll"):
                    yield Static("", id="dogfood-transcript")
                yield Static("", id="dogfood-inspector")
            with Vertical(id="debug-view"):
                with ScrollableContainer(id="debug-log-scroll"):
                    yield Static("", id="debug-log")
                with ScrollableContainer(id="debug-state-scroll"):
                    yield Static("", id="debug-state")
            with Vertical(id="chat-view"):
                with ScrollableContainer(id="chat-transcript-scroll"):
                    yield Static("", id="chat-transcript")
            with Vertical(id="memory-view"):
                with ScrollableContainer(id="memory-transcript-scroll"):
                    yield Static("", id="memory-transcript")
        yield Static("", id="command-palette")
        yield Input(
            placeholder="Message OpenCouch...",
            id="message-input",
            suggester=SuggestFromList(TUI_SLASH_SUGGESTIONS, case_sensitive=False),
        )
        yield Static(id="help-bar")

    async def on_mount(self) -> None:
        """Initialize runtime and visible workspace state."""

        self._runtime = await self._runtime_factory(self.config).__aenter__()
        self._apply_theme()
        self._hydrate_transcripts_from_session()
        self._render_status()
        self._render_inspector(None)
        self._render_transcripts()
        await self._refresh_memory_view()
        self._show_view(self.active_view)
        self._hide_command_palette()
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
        self._hide_command_palette()
        if not message:
            return
        self.query_one("#error-banner", Static).update("")
        if message.startswith("/"):
            await self._handle_command(message)
            return
        self._append_user_message(message)
        self._render_transcripts()
        self.run_worker(self._run_turn(message), exclusive=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Refresh slash-command discovery as the composer changes."""

        self._render_command_palette(event.value)

    async def _handle_command(self, message: str) -> None:
        """Handle slash commands for thread/session control."""

        runtime = self._require_runtime()
        parts = message.split()
        command = parts[0]

        if command == "/help":
            self._show_command_output(
                "commands",
                "\n".join(
                    [
                        *[command.display for command in TUI_SLASH_COMMANDS],
                    ]
                ),
            )
            return

        if command == "/keys":
            self._show_command_output(
                "keys",
                "\n".join(
                    [
                        "Tab / Shift+Tab: switch workspace",
                        "Ctrl+1..4: Dogfood / Debug / Chat / Memory",
                        "Ctrl+Y: toggle theme",
                        "Ctrl+R: refresh memory",
                        "PageUp/PageDown/Home/End: scroll active pane",
                    ]
                ),
            )
            return

        if command == "/status":
            session = runtime.session
            if session is None:
                self._show_command_error("Runtime session is not ready.")
                return
            self._show_command_output(
                "status",
                "\n".join(session_status_command_lines(session)),
            )
            return

        if command == "/threads":
            session = runtime.session
            active_thread_id = session.thread_id if session is not None else None
            error, summaries = await get_threads_command_summaries(
                parts,
                runtime=runtime,
            )
            if error is not None:
                self._show_command_error(error)
                return
            if not summaries:
                self._show_command_output("threads", "No persisted threads found.")
                return
            self._show_command_output(
                "threads",
                format_thread_summaries_plain(
                    summaries,
                    active_thread_id=active_thread_id,
                ),
            )
            return

        if command == "/history":
            error, limit = get_history_command_limit(parts)
            if error is not None:
                self._show_command_error(error)
                return
            session = runtime.session
            error, recent = get_history_window(
                session.history if session is not None else None,
                limit=limit or 6,
            )
            if error is not None:
                self._show_command_error(error)
                return
            if not recent:
                self._show_command_output("history", "No messages in this thread yet.")
                return
            self._show_command_output(
                "history",
                format_history_plain(recent, limit=limit or 6),
            )
            return

        if command == "/context":
            session = runtime.session
            if session is None:
                self._show_command_error("Runtime session is not ready.")
                return
            context = session.last_context
            if context is None:
                self._show_command_output("context", "No persisted context yet.")
                return
            self._show_command_output("context", str(context))
            return

        if command == "/search":
            error, parsed = parse_search_command(parts)
            if error is not None:
                self._show_command_error(error)
                return
            if parsed is None:
                self._show_command_error("Usage: /search <history|memory|all> <query>")
                return
            mode, query = parsed
            session = runtime.session
            history_results: list[tuple[str, str]] = []
            memory_results: list[tuple[str, str]] = []

            if mode in {"history", "all"}:
                if session is None:
                    self._show_command_error("Runtime session is not ready.")
                    return
                history_results = search_history_messages(session.history, query=query)

            if mode in {"memory", "all"}:
                snapshot = await runtime.load_memory_snapshot(include_notebook=False)
                memory_results = search_memory_snapshot(snapshot, query=query)

            results = (
                history_results
                if mode == "history"
                else memory_results
                if mode == "memory"
                else [*history_results, *memory_results]
            )
            if not results:
                empty_message = (
                    f'No history matches for "{query}".'
                    if mode == "history"
                    else f'No memory matches for "{query}".'
                    if mode == "memory"
                    else f'No matches for "{query}".'
                )
                self._show_command_output(f"search {mode}", empty_message)
                return
            self._show_command_output(
                f"search {mode}",
                "\n".join(f"{source}: {snippet}" for source, snippet in results[:8]),
            )
            return

        if command == "/debug":
            if parts[1:] != ["state"]:
                self._show_command_error("Usage: /debug state")
                return
            session = runtime.session
            if session is None:
                self._show_command_error("Runtime session is not ready.")
                return
            self._show_command_output(
                "debug state",
                "\n".join(
                    [
                        f"thread_id: {session.thread_id}",
                        f"owner_id: {session.owner_id}",
                        f"history_messages: {len(session.history)}",
                        f"has_context: {session.last_context is not None}",
                    ]
                ),
            )
            return

        if command == "/memory":
            await self._handle_memory_command(parts)
            return

        if command == "/clear":
            if len(parts) != 1:
                self._show_command_error("Usage: /clear")
                return
            self.action_clear_view()
            self._show_command_output("clear", f"Cleared {self.active_view} pane.")
            return

        if command == "/resume":
            if len(parts) != 2:
                self._show_command_error("Usage: /resume <thread-id>")
                return
            thread_id = parts[1]
            if not await runtime.switch_thread(thread_id, require_existing=True):
                self._show_command_error(
                    f"Thread {thread_id} does not exist. Use /new {thread_id} to create it."
                )
                return
            self._append_debug(f"resumed thread: {thread_id}")
            await self._after_thread_switch()
            return

        if command == "/new":
            if len(parts) > 2:
                self._show_command_error("Usage: /new [thread-id]")
                return
            thread_id = parts[1] if len(parts) == 2 else f"local-tui-{uuid4().hex[:8]}"
            if await runtime.thread_exists(thread_id):
                self._show_command_error(
                    f"Thread {thread_id} already exists. Use /resume {thread_id} or choose another id."
                )
                return
            await runtime.switch_thread(thread_id)
            self._append_debug(f"started new thread: {thread_id}")
            await self._after_thread_switch()
            return

        self._show_command_error(f"Unknown command: {command}")

    async def _handle_memory_command(self, parts: list[str]) -> None:
        error, overview = parse_memory_overview_command(parts)
        if error is not None:
            self._show_command_error(error)
            return
        if overview is None:
            self._show_command_error(
                "Usage: /memory status  |  /memory list [facts|sessions|rules]"
            )
            return

        action, kind = overview
        snapshot = await self._require_runtime().load_memory_snapshot(
            include_notebook=False
        )
        if action == "status":
            self._show_command_output(
                "memory status",
                format_memory_status_plain(snapshot),
            )
            return

        self._show_command_output(
            f"memory list {kind}",
            format_memory_snapshot_plain(snapshot, kind=kind or "all"),
        )

    def _show_command_output(self, title: str, body: str) -> None:
        self.query_one("#error-banner", Static).update("")
        entry = TranscriptEntry(role="system", message=f"{title}\n{body}")
        self._dogfood_entries.append(entry)
        self._chat_entries.append(entry)
        self._append_debug(f"{title}\n{body}")
        self._render_transcripts()

    def _render_command_palette(self, value: str) -> None:
        if not value.startswith("/"):
            self._hide_command_palette()
            return

        palette = self.query_one("#command-palette", Static)
        query = value.lower()
        matches = [
            command
            for command in TUI_SLASH_COMMANDS
            if _command_suggestion_text(command.display).lower().startswith(query)
            or command.display.lower().startswith(query)
        ]

        content = Text("Slash commands\n", style=self._role_style("system"))
        if not matches:
            content.append("No commands match. Try /help.", style=self._debug_style())
        else:
            current_category = None
            for command in matches[:10]:
                category = command.category.title()
                if category != current_category:
                    if current_category is not None:
                        content.append("\n")
                    content.append(f"{category}\n", style=self._muted_style())
                    current_category = category
                content.append(f"  {command.display}", style=self._role_style("system"))
                content.append(f" — {command.description}\n", style=self._debug_style())

        palette.update(content)
        palette.display = True

    def _hide_command_palette(self) -> None:
        palette = self.query_one("#command-palette", Static)
        palette.update("")
        palette.display = False

    async def _after_thread_switch(self) -> None:
        """Refresh app state after switching the active thread."""

        self.query_one("#error-banner", Static).update("")
        self._hydrate_transcripts_from_session()
        self._render_inspector(None)
        self._render_status()
        self._render_transcripts()
        await self._refresh_memory_view()

    def _show_command_error(self, message: str) -> None:
        self.query_one("#error-banner", Static).update(message)
        self._append_debug(f"command error: {message}")
        self._render_transcripts()

    def on_key(self, event: Key) -> None:
        """Keep workspace cycling and transcript scrolling available from input."""

        if event.key == "tab":
            event.prevent_default()
            event.stop()
            self.action_next_view()
        elif event.key == "shift+tab":
            event.prevent_default()
            event.stop()
            self.action_previous_view()
        elif (
            event.key == "escape" and self.query_one("#command-palette", Static).display
        ):
            event.prevent_default()
            event.stop()
            self._hide_command_palette()
        elif event.key in {"pageup", "pagedown", "home", "end"}:
            event.prevent_default()
            event.stop()
            self._scroll_active_transcript(event.key)

    async def _run_turn(self, message: str) -> None:
        runtime = self._require_runtime()
        accumulated = ""
        response_ready = False
        saw_done = False
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
                saw_done = True
                if not response_ready:
                    self._replace_or_append_assistant(event.output.response_text)
                self._render_inspector(event.output)
                self._append_debug("done")
            elif isinstance(event, ConsoleErrorEvent):
                self.query_one("#error-banner", Static).update(event.message)
                self._append_debug(f"error: {event.message}")
            self._render_transcripts()
        if saw_done:
            self.run_worker(self._refresh_memory_view(), exclusive=False)
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

    def action_show_memory(self) -> None:
        """Switch to Memory workspace."""

        self._show_view("memory")

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

    def action_refresh_memory(self) -> None:
        """Refresh the in-app memory workspace."""

        self.run_worker(self._refresh_memory_view(), exclusive=False)

    def action_blur_focus(self) -> None:
        """Return focus to the app shell."""

        self.screen.focus(None)

    def _show_view(self, view: TuiView) -> None:
        self.active_view = view
        self.query_one("#dogfood-view").display = view == "dogfood"
        self.query_one("#debug-view").display = view == "debug"
        self.query_one("#chat-view").display = view == "chat"
        self.query_one("#memory-view").display = view == "memory"
        self.query_one("#dogfood-inspector").display = (
            self._diagnostics_visible and view == "dogfood"
        )
        self.query_one("#debug-state").display = (
            self._diagnostics_visible and view == "debug"
        )
        self._render_tabs()
        self._render_help_bar()
        if view == "memory":
            self.run_worker(self._refresh_memory_view(), exclusive=False)

    def _show_relative_view(self, offset: int) -> None:
        index = VIEW_ORDER.index(self.active_view)
        self._show_view(VIEW_ORDER[(index + offset) % len(VIEW_ORDER)])

    def _active_transcript_selector(self) -> str:
        if self.active_view == "debug":
            return "#debug-log-scroll"
        if self.active_view == "chat":
            return "#chat-transcript-scroll"
        if self.active_view == "memory":
            return "#memory-transcript-scroll"
        return "#dogfood-transcript-scroll"

    def _scroll_active_transcript(self, key: str) -> None:
        transcript = self.query_one(
            self._active_transcript_selector(), ScrollableContainer
        )
        match key:
            case "pageup":
                transcript.scroll_page_up(animate=False)
            case "pagedown":
                transcript.scroll_page_down(animate=False)
            case "home":
                transcript.scroll_home(animate=False)
            case "end":
                transcript.scroll_end(animate=False)
            case _:
                return

    def _hydrate_transcripts_from_session(self) -> None:
        runtime = self._require_runtime()
        session = runtime.session
        if session is None:
            raise RuntimeError("ConsoleRuntime has not been entered.")

        entries: list[TranscriptEntry] = []
        for message in session.history:
            if message.role == MessageRole.USER:
                entries.append(TranscriptEntry(role="user", message=message.content))
            elif message.role == MessageRole.ASSISTANT:
                entries.append(
                    TranscriptEntry(role="assistant", message=message.content)
                )

        self._dogfood_entries = list(entries)
        self._chat_entries = list(entries)

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
            "memory": "Memory",
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
            "  ".join(session_status_bar_parts(session, active_theme=self.active_theme))
        )

    def _render_inspector(self, output: AgentOutput | None) -> None:
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
        self.query_one("#debug-state", Static).update(
            self._render_debug_state(output, lines)
        )

    def _render_debug_state(
        self,
        output: AgentOutput | None,
        summary_lines: list[str],
    ) -> Text:
        debug_state = Text()
        for index, line in enumerate(summary_lines):
            style = self._muted_style() if index == 0 else self._debug_style()
            debug_state.append(line, style=style)
            debug_state.append("\n")

        debug_state.append("\nRouting trace\n", style=self._role_style("user"))
        if output is None:
            debug_state.append("- none yet", style=self._debug_style())
            return debug_state

        entries = [
            {key: str(value) for key, value in entry.items()}
            for entry in routing_trace_from_diagnostics(output.diagnostics)
        ]
        if not entries:
            entries = self._fallback_routing_trace_entries(output)

        for index, entry in enumerate(entries):
            debug_state.append(
                f"{index + 1}. {entry.get('stage', '-')} → {entry.get('decision', '-')}\n",
                style=self._message_style("assistant"),
            )
            source = entry.get("source") or "-"
            confidence = entry.get("confidence")
            if confidence:
                source = f"{source} / {confidence}"
            debug_state.append(f"   source: {source}\n", style=self._muted_style())
            reason = entry.get("reason") or "-"
            debug_state.append(f"   reason: {reason}", style=self._debug_style())
            if index != len(entries) - 1:
                debug_state.append("\n\n")
        return debug_state

    def _fallback_routing_trace_entries(
        self,
        output: AgentOutput,
    ) -> list[dict[str, str]]:
        crisis = output.crisis
        if crisis.needs_crisis_response:
            safety_decision = "crisis"
        elif crisis.needs_clarification:
            safety_decision = "check"
        elif crisis.level >= 1:
            safety_decision = "distress"
        else:
            safety_decision = "normal"

        route_decision = output.response_style or "unknown"
        if output.therapeutic_approach and output.therapeutic_approach != "none":
            route_decision = f"{route_decision}/{output.therapeutic_approach}"

        return [
            {
                "stage": "safety",
                "decision": safety_decision,
                "source": "output",
                "reason": crisis.reason or "No structured safety reason was emitted.",
                "confidence": str(crisis.confidence or "-"),
            },
            {
                "stage": "dispatch",
                "decision": route_decision,
                "source": "output",
                "reason": output.response_type.value,
                "confidence": "-",
            },
        ]

    async def _refresh_memory_view(self) -> None:
        runtime = self._require_runtime()
        try:
            snapshot = await runtime.load_memory_snapshot()
        except Exception as exc:
            self._memory_text = Text(
                f"Unable to load memory: {exc}",
                style=self._debug_style(),
            )
            self._append_debug(f"memory load error: {exc}")
        else:
            self._memory_text = self._render_memory_snapshot(snapshot)
        memory_view = self.query_one("#memory-transcript", Static)
        memory_view.remove_class("empty-transcript")
        memory_view.update(self._memory_text)
        memory_scroll = self.query_one("#memory-transcript-scroll", ScrollableContainer)
        self.call_after_refresh(memory_scroll.scroll_home, animate=False)

    def _render_transcripts(self) -> None:
        self._update_conversation_transcript(
            "#dogfood-transcript",
            self._dogfood_entries,
        )
        self._update_conversation_transcript(
            "#chat-transcript",
            self._chat_entries,
        )
        debug_log = self.query_one("#debug-log", Static)
        debug_log.update(self._render_debug_log())
        debug_scroll = self.query_one("#debug-log-scroll", ScrollableContainer)
        self.call_after_refresh(debug_scroll.scroll_end, animate=False)
        memory_view = self.query_one("#memory-transcript", Static)
        memory_view.update(self._memory_text)

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
        transcript_scroll = self.query_one(f"{selector}-scroll", ScrollableContainer)
        self.call_after_refresh(transcript_scroll.scroll_end, animate=False)

    def _render_transcript(self, entries: list[TranscriptEntry]) -> Text | Group:
        if not entries:
            return self._render_empty_transcript()

        renderables: list[Text | Markdown] = []
        for index, entry in enumerate(entries):
            if entry.role == "user":
                role_label = "🧑 You"
            elif entry.role == "assistant":
                role_label = "🛋 OpenCouch"
            else:
                role_label = "⚙ Command"

            renderables.append(Text(role_label, style=self._role_style(entry.role)))
            if entry.role == "assistant":
                renderables.append(Markdown(entry.message))
            else:
                renderables.append(
                    Text(entry.message, style=self._message_style(entry.role))
                )
            if index != len(entries) - 1:
                renderables.append(Text(""))
        renderables.append(Text(""))
        return Group(*renderables)

    def _render_empty_transcript(self) -> Text:
        return Text()

    def _render_debug_log(self) -> Text:
        debug = Text()
        for index, line in enumerate(self._debug_lines):
            debug.append(line, style=self._debug_style())
            if index != len(self._debug_lines) - 1:
                debug.append("\n")
        return debug

    def _render_memory_snapshot(self, snapshot: dict[str, Any]) -> Text:
        notebook = snapshot.get("notebook")
        if not isinstance(notebook, dict) or snapshot.get(
            "has_unprojected_legacy_memory", False
        ):
            return self._render_raw_memory_snapshot(snapshot)
        return self._render_memory_notebook(snapshot["owner_id"], notebook)

    @staticmethod
    def _visible_raw_memory_records(records: object) -> list[Any]:
        if not isinstance(records, list):
            return []
        return [
            record
            for record in records
            if not isinstance(record, dict)
            or (
                record.get("user_visible", True)
                and not record.get("dormant_at")
                and not record.get("superseded_by")
            )
        ]

    def _render_raw_memory_snapshot(self, snapshot: dict[str, Any]) -> Text:
        memory = Text()
        memory.append(
            f"Owner: {snapshot['owner_id']}\n\n",
            style=self._role_style("assistant"),
        )
        semantic = self._visible_raw_memory_records(snapshot.get("semantic"))
        episodic = self._visible_raw_memory_records(snapshot.get("episodic"))
        procedural = snapshot.get("procedural")

        memory.append("Semantic\n", style=self._role_style("user"))
        if semantic:
            for record in semantic:
                memory.append(
                    f"- {record.get('predicate', '?')}: ", style=self._muted_style()
                )
                target = record.get("object", {})
                if isinstance(target, dict):
                    memory.append(
                        f"{target.get('identifier', '?')}\n",
                        style=self._message_style("user"),
                    )
                else:
                    memory.append(f"{target}\n", style=self._message_style("user"))
        else:
            memory.append("- none\n", style=self._debug_style())

        memory.append("\nEpisodic\n", style=self._role_style("assistant"))
        if episodic:
            for record in episodic:
                memory.append(
                    f"- {record.get('session_id', '?')}: {record.get('summary', '?')}\n",
                    style=self._message_style("assistant"),
                )
        else:
            memory.append("- none\n", style=self._debug_style())

        memory.append("\nProcedural\n", style=self._role_style("assistant"))
        if isinstance(procedural, dict):
            memory.append(
                f"- recall: {'on' if procedural.get('proactive_recall_enabled') else 'off'}\n",
                style=self._message_style("assistant"),
            )
            rules = self._visible_raw_memory_records(procedural.get("rules"))
            if rules:
                for rule in rules:
                    rule_text = (
                        rule.get("rule", "?") if isinstance(rule, dict) else rule
                    )
                    memory.append(
                        f"  • {rule_text}\n",
                        style=self._message_style("assistant"),
                    )
            else:
                memory.append("  • none\n", style=self._debug_style())
        else:
            memory.append("- none\n", style=self._debug_style())

        return memory

    def _render_memory_notebook(
        self, owner_id: object, notebook: dict[str, Any]
    ) -> Text:
        memory = Text()
        memory.append(f"Owner: {owner_id}\n", style=self._role_style("assistant"))

        counts = notebook.get("counts", {})
        count_text = "0 entries"
        if isinstance(counts, dict):
            total = int(counts.get("total_entries") or 0)
            semantic = int(counts.get("semantic") or 0)
            episodic = int(counts.get("episodic") or 0)
            procedural = int(counts.get("procedural_rules") or 0)
            count_text = (
                f"{total} entries ({semantic} facts, "
                f"{episodic} sessions, {procedural} rules)"
            )
        recall = "on" if notebook.get("proactive_recall_enabled") else "off"
        memory.append(
            f"Notebook: {count_text} · recall: {recall}\n\n",
            style=self._muted_style(),
        )

        topics = notebook.get("topics", [])
        if not isinstance(topics, list) or not topics:
            memory.append("No visible memory records.\n", style=self._debug_style())
            return memory

        rendered_topics = 0
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            entries = topic.get("entries", [])
            if not isinstance(entries, list):
                entries = []
            if rendered_topics:
                memory.append("\n")
            label = str(topic.get("label") or topic.get("id") or "Memory")
            memory.append(f"{label}\n", style=self._role_style("user"))
            if not entries:
                memory.append("- none\n", style=self._debug_style())
                rendered_topics += 1
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                title = str(entry.get("title") or entry.get("category") or "Memory")
                summary = str(entry.get("summary") or "").strip()
                memory.append(f"- {title}\n", style=self._message_style("assistant"))
                if summary:
                    memory.append(f"  {summary}\n", style=self._muted_style())
                provenance = self._memory_notebook_provenance_text(
                    entry.get("provenance")
                )
                if provenance:
                    memory.append(f"  · {provenance}\n", style=self._debug_style())
            rendered_topics += 1

        if rendered_topics == 0:
            memory.append("No visible memory records.\n", style=self._debug_style())
        return memory

    @staticmethod
    def _memory_notebook_provenance_text(provenance: object) -> str:
        if not isinstance(provenance, dict):
            return ""
        parts: list[str] = []
        confidence = provenance.get("confidence")
        if confidence:
            parts.append(f"confidence: {confidence}")
        source_session = provenance.get("source_session_id")
        if source_session:
            source = f"source: {source_session}"
            source_turn = provenance.get("source_turn_index")
            if source_turn is not None:
                source = f"{source} turn {source_turn}"
            parts.append(source)
        write_reason = provenance.get("write_reason")
        if write_reason:
            parts.append(f"reason: {write_reason}")
        return " · ".join(parts)

    def _render_help_bar(self) -> None:
        help_text = Text()
        help_text.append("Tab Next", style=self._shortcut_style())
        help_text.append("   Shift+Tab Previous", style=self._shortcut_style())
        help_text.append("   Ctrl+1 Dogfood", style=self._shortcut_style())
        help_text.append("   Ctrl+2 Debug", style=self._shortcut_style())
        help_text.append("   Ctrl+3 Chat", style=self._shortcut_style())
        help_text.append("   Ctrl+4 Memory", style=self._shortcut_style())
        help_text.append(
            f"   Ctrl+Y Theme {self.active_theme}", style=self._shortcut_style()
        )
        help_text.append("   Ctrl+T Diagnostics", style=self._shortcut_style())
        help_text.append("   PageUp/PageDown Scroll", style=self._shortcut_style())
        help_text.append("   Home/End Jump", style=self._shortcut_style())
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
            styles = {
                "user": "bold #8bd4ff",
                "assistant": "bold #9ee6b6",
                "system": "bold #ffd166",
            }
        else:
            styles = {
                "user": "bold #2563eb",
                "assistant": "bold #047857",
                "system": "bold #b45309",
            }
        return styles[role]

    def _message_style(self, role: TuiRole) -> str:
        if self.active_theme == "dark":
            styles = {
                "user": "#eaf3ef on #182823",
                "assistant": "#eff7f2 on #14241c",
                "system": "#f5e7c8 on #2a2214",
            }
        else:
            styles = {
                "user": "#23302d on #edf7ff",
                "assistant": "#23302d on #edf8f1",
                "system": "#3f3420 on #fff7df",
            }
        return styles[role]

    def _debug_style(self) -> str:
        return "#9fb3aa" if self.active_theme == "dark" else "#626d68"

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

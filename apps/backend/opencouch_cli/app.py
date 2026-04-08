"""Interactive CLI for the OpenCouch agent runtime.

Example:
    uv run python -m opencouch_cli --mode auto --thread-id local-demo --sqlite-path .opencouch_threads.sqlite3
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from uuid import uuid4

from agent.persistence import DEFAULT_THREAD_DB_PATH, PersistentAgentRuntime
from agent.models import (
    Channel,
    ChunkEvent,
    DoneEvent,
    Message,
    MessageRole,
    StatusEvent,
)
from agent.state import AgentState
from core.config import create_configured_llm_client
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from services.llm.base import BaseLLMClient

console = Console()


@dataclass(slots=True)
class RunnerSession:
    """Mutable local session state for the interactive CLI."""

    requested_mode: str
    resolved_mode: str
    llm_client: BaseLLMClient | None
    thread_id: str
    sqlite_path: str
    history: list[Message] = field(default_factory=list)
    last_context: AgentState | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI.

    Returns:
        Configured argument parser for the CLI entrypoint.
    """

    parser = argparse.ArgumentParser(
        description="Run the interactive OpenCouch CLI.",
        epilog=(
            "Example: uv run python -m opencouch_cli --mode auto "
            "--thread-id local-demo --sqlite-path .opencouch_threads.sqlite3"
        ),
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
        "--sqlite-path",
        default=str(DEFAULT_THREAD_DB_PATH),
        help="SQLite path for LangGraph thread checkpoints.",
    )
    return parser


def resolve_llm_client(mode: str) -> tuple[BaseLLMClient | None, str]:
    """Resolve the runtime LLM client for the selected mode.

    Args:
        mode: Requested runtime mode from the CLI.

    Returns:
        The resolved client and the effective mode label.
    """

    if mode == "deterministic":
        return None, "deterministic"

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def render_header(mode: str, thread_id: str) -> None:
    """Render the CLI header.

    Args:
        mode: Effective runtime mode for the current session.
        thread_id: Active persisted thread identifier.

    Returns:
        None.
    """

    subtitle = Text()
    subtitle.append("mode: ", style="bold white")
    subtitle.append(mode, style="bold cyan")
    subtitle.append("  |  thread: ", style="dim")
    subtitle.append(thread_id, style="bold white")
    subtitle.append("  |  type ", style="dim")
    subtitle.append("exit", style="bold")
    subtitle.append(" or ", style="dim")
    subtitle.append("quit", style="bold")
    subtitle.append(" to stop", style="dim")

    console.print(
        Panel(
            subtitle,
            title="[bold blue]OpenCouch[/bold blue]",
            subtitle_align="left",
            border_style="blue",
            expand=False,
        )
    )
    console.print(
        "[dim]slash commands: /help, /status, /history, /context, /reset, /clear, /mode, /exit[/dim]\n"
    )


def render_response(response_text: str, *, is_crisis: bool) -> None:
    """Render the assistant reply inside a styled panel.

    Args:
        response_text: Generated assistant reply text.
        is_crisis: Whether the response is a crisis response.

    Returns:
        None.
    """

    title = (
        "[bold red]Crisis Reply[/bold red]"
        if is_crisis
        else "[bold green]Support Reply[/bold green]"
    )
    border_style = "red" if is_crisis else "green"
    console.print(Panel(response_text, title=title, border_style=border_style))


def render_meta(
    *,
    mode: str | None,
    mode_source: str | None,
    mode_type: str | None,
    response_type: str,
    level: int,
    needs_clarification: bool,
    needs_crisis_response: bool,
    reason: str,
) -> None:
    """Render classifier metadata as a compact table.

    Args:
        mode: Selected graph mode for the response.
        mode_source: How the mode was selected (keyword/session_intent/llm/default).
        mode_type: Higher-level category for the selected mode.
        response_type: High-level response type label.
        level: Crisis level selected by the gate.
        needs_clarification: Whether the gate requested a safety check.
        needs_crisis_response: Whether the gate requested the crisis path.
        reason: Crisis-classifier explanation.

    Returns:
        None.
    """

    if needs_crisis_response:
        safety_status = "crisis"
    elif needs_clarification:
        safety_status = "check"
    elif level >= 1:
        safety_status = "distress"
    else:
        safety_status = "normal"

    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("mode", style="green", no_wrap=True)
    table.add_column("source", style="blue", no_wrap=True)
    table.add_column("mode type", style="yellow", no_wrap=True)
    table.add_column("type", style="cyan", no_wrap=True)
    table.add_column("safety", justify="center", no_wrap=True)
    table.add_column("clarify", justify="center", no_wrap=True)
    table.add_column("crisis", justify="center", no_wrap=True)
    table.add_column("reason", style="white")
    table.add_row(
        mode or "-",
        mode_source or "-",
        mode_type or "-",
        response_type,
        safety_status,
        "yes" if needs_clarification else "no",
        "yes" if needs_crisis_response else "no",
        reason,
    )
    console.print(table)
    console.print()


def render_context(state: AgentState | None) -> None:
    """Render the current structured session context.

    Args:
        state: Most recent graph input state snapshot.

    Returns:
        None.
    """

    if state is None:
        console.print(
            Panel(
                "No session context yet. Send a message first.",
                title="[bold blue]Session Context[/bold blue]",
                border_style="blue",
            )
        )
        console.print()
        return

    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("turn_count", str(state["turn_count"]))
    table.add_row(
        "working_memory",
        " | ".join(state.get("working_memory", []))
        if state.get("working_memory")
        else "-",
    )
    table.add_row("current_goal", state["current_goal"] or "-")
    table.add_row("response_guidance", state.get("response_guidance") or "-")
    table.add_row(
        "active_concerns",
        ", ".join(state["active_concerns"]) if state["active_concerns"] else "-",
    )
    table.add_row(
        "open_loops",
        " | ".join(state["open_loops"]) if state["open_loops"] else "-",
    )
    table.add_row("session_summary", state["session_summary"])
    console.print(
        Panel(
            table,
            title="[bold blue]Session Context[/bold blue]",
            subtitle="[dim]what the graph is carrying forward[/dim]",
            border_style="blue",
        )
    )
    console.print()


def render_help() -> None:
    """Render the available slash commands.

    Returns:
        None.
    """

    table = Table(show_header=True, header_style="bold blue", box=None)
    table.add_column("command", style="cyan", no_wrap=True)
    table.add_column("description", style="white")
    table.add_row("/help", "Show available commands.")
    table.add_row("/status", "Show current mode and session stats.")
    table.add_row("/history [n]", "Show the last n transcript messages. Default: 6.")
    table.add_row("/context", "Show the latest derived session context snapshot.")
    table.add_row("/reset", "Clear the conversation history.")
    table.add_row("/clear", "Clear the terminal and redraw the header.")
    table.add_row(
        "/mode <deterministic|hybrid|auto>",
        "Switch LLM resolution mode for future turns.",
    )
    table.add_row("/exit", "End the session.")
    console.print(
        Panel(table, title="[bold blue]Commands[/bold blue]", border_style="blue")
    )
    console.print()


def render_status(session: RunnerSession) -> None:
    """Render current runner status.

    Args:
        session: Mutable CLI session state.

    Returns:
        None.
    """

    turn_count = sum(
        1 for message in session.history if message.role == MessageRole.USER
    )
    table = Table(show_header=False, box=None)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("thread id", session.thread_id)
    table.add_row("sqlite path", session.sqlite_path)
    table.add_row("requested mode", session.requested_mode)
    table.add_row("resolved mode", session.resolved_mode)
    table.add_row(
        "llm client", "enabled" if session.llm_client is not None else "disabled"
    )
    table.add_row("turns", str(turn_count))
    table.add_row("messages", str(len(session.history)))
    table.add_row(
        "context snapshot", "available" if session.last_context is not None else "none"
    )
    console.print(
        Panel(table, title="[bold blue]Session Status[/bold blue]", border_style="blue")
    )
    console.print()


def render_history(session: RunnerSession, limit: int = 6) -> None:
    """Render the most recent transcript entries.

    Args:
        session: Mutable CLI session state.
        limit: Maximum number of recent messages to display.

    Returns:
        None.
    """

    if not session.history:
        console.print(
            Panel(
                "No conversation history yet.",
                title="[bold blue]History[/bold blue]",
                border_style="blue",
            )
        )
        console.print()
        return

    recent = session.history[-max(1, limit) :]
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("role", style="cyan", no_wrap=True)
    table.add_column("content", style="white")
    for message in recent:
        role = message.role.value
        table.add_row(role, message.content)
    console.print(
        Panel(table, title="[bold blue]Recent History[/bold blue]", border_style="blue")
    )
    console.print()


def render_info(message: str, *, style: str = "blue") -> None:
    """Render a lightweight informational panel.

    Args:
        message: Informational message to display.
        style: Border color/style token for the panel.

    Returns:
        None.
    """

    console.print(Panel(message, border_style=style, expand=False))
    console.print()


def set_mode(session: RunnerSession, mode: str) -> None:
    """Update the active runner mode for subsequent turns.

    Args:
        session: Mutable CLI session state.
        mode: Requested runtime mode.

    Returns:
        None.
    """

    llm_client, resolved_mode = resolve_llm_client(mode)
    session.requested_mode = mode
    session.resolved_mode = resolved_mode
    session.llm_client = llm_client


async def handle_command(
    command_text: str,
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
) -> bool:
    """Handle slash commands.

    Args:
        command_text: Raw slash command text from the user.
        session: Mutable CLI session state.
        runtime: Persistent runtime backing the active thread.

    Returns:
        True if the session should continue, False if it should exit.
    """

    raw = command_text.strip()
    if not raw:
        return True

    parts = raw.split()
    command = parts[0].lower()
    args = parts[1:]

    if command in {"/exit", "/quit"}:
        return False

    if command == "/help":
        render_help()
        return True

    if command == "/status":
        render_status(session)
        return True

    if command == "/history":
        limit = 6
        if args:
            try:
                limit = max(1, int(args[0]))
            except ValueError:
                render_info("Usage: /history [count]", style="yellow")
                return True
        render_history(session, limit=limit)
        return True

    if command == "/context":
        render_context(session.last_context)
        return True

    if command == "/reset":
        await runtime.reset_thread(session.thread_id)
        session.history.clear()
        session.last_context = None
        render_info("Persisted thread state cleared.", style="yellow")
        return True

    if command == "/clear":
        console.clear()
        render_header(session.resolved_mode, session.thread_id)
        return True

    if command == "/mode":
        if len(args) != 1 or args[0] not in {"deterministic", "hybrid", "auto"}:
            render_info("Usage: /mode <deterministic|hybrid|auto>", style="yellow")
            return True
        set_mode(session, args[0])
        render_info(
            f"Mode updated. requested={session.requested_mode}, resolved={session.resolved_mode}",
            style="green" if session.llm_client is not None else "yellow",
        )
        return True

    render_info(f"Unknown command: {command}. Try /help.", style="yellow")
    return True


async def chat_loop(mode: str, *, thread_id: str, sqlite_path: str) -> None:
    """Run the interactive CLI loop.

    Args:
        mode: Requested runtime mode for model resolution.
        thread_id: Stable thread identifier for the local conversation.
        sqlite_path: SQLite file used for persisted thread checkpoints.

    Returns:
        None.
    """

    llm_client, resolved_mode = resolve_llm_client(mode)
    session = RunnerSession(
        requested_mode=mode,
        resolved_mode=resolved_mode,
        llm_client=llm_client,
        thread_id=thread_id,
        sqlite_path=sqlite_path,
    )

    async with PersistentAgentRuntime(sqlite_path) as runtime:
        session.history = await runtime.get_history(thread_id)
        session.last_context = await runtime.get_state(thread_id)

        render_header(session.resolved_mode, session.thread_id)
        if session.history:
            render_info(
                f"Resumed thread {session.thread_id} with {len(session.history)} stored messages.",
                style="green",
            )

        while True:
            try:
                user_text = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]session ended[/dim]")
                break

            if not user_text:
                continue
            if user_text.startswith("/"):
                if not await handle_command(user_text, session, runtime):
                    break
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            accumulated_text = ""
            final_output = None

            _STAGE_LABELS = {
                "crisis_gate": "safety check",
                "session_stage": "reading context",
                "response_generation": "generating",
            }

            with Live(console=console, refresh_per_second=15) as live:
                async for event in runtime.run_turn_stream(
                    thread_id=session.thread_id,
                    message=user_text,
                    channel=Channel.TEST,
                    llm_client=session.llm_client,
                ):
                    if isinstance(event, StatusEvent):
                        label = _STAGE_LABELS.get(event.stage, event.stage)
                        live.update(Text(f"  ● {label}", style="dim"))

                    elif isinstance(event, ChunkEvent):
                        accumulated_text += event.text
                        live.update(
                            Panel(
                                accumulated_text,
                                border_style="green",
                            )
                        )

                    elif isinstance(event, DoneEvent):
                        final_output = event.output
                        # Render the final panel as the last Live frame so it
                        # stays on screen when Live exits — no duplicate render.
                        is_crisis = final_output.response_type.value == "crisis"
                        title = (
                            "[bold red]Crisis Reply[/bold red]"
                            if is_crisis
                            else "[bold green]Support Reply[/bold green]"
                        )
                        border = "red" if is_crisis else "green"
                        live.update(
                            Panel(
                                final_output.response_text,
                                title=title,
                                border_style=border,
                            )
                        )

            if final_output is not None:
                # If no chunks were streamed (deterministic path), the final
                # panel was set in the DoneEvent handler above. For the
                # non-streamed case, render_response was already called via
                # live.update, so we skip the duplicate.
                render_meta(
                    mode=final_output.mode,
                    mode_source=final_output.mode_source,
                    mode_type=(
                        final_output.mode_type.value
                        if final_output.mode_type is not None
                        else None
                    ),
                    response_type=final_output.response_type.value,
                    level=final_output.crisis.level,
                    needs_clarification=final_output.crisis.needs_clarification,
                    needs_crisis_response=final_output.crisis.needs_crisis_response,
                    reason=final_output.crisis.reason,
                )

            # Refresh session state from the persisted checkpoint.
            session.last_context = await runtime.get_state(session.thread_id)
            session.history = await runtime.get_history(session.thread_id)
            if session.last_context:
                render_context(session.last_context)


def main() -> int:
    """Run the OpenCouch CLI.

    Returns:
        Process exit code for the CLI session.
    """

    args = build_parser().parse_args()
    thread_id = args.thread_id or f"local-{uuid4().hex[:12]}"
    sqlite_path = str(Path(args.sqlite_path).expanduser())
    asyncio.run(chat_loop(args.mode, thread_id=thread_id, sqlite_path=sqlite_path))
    return 0

"""Interactive CLI for the OpenCouch agent runtime.

Example:
    uv run python -m opencouch_cli --mode auto --thread-id local-demo --sqlite-path .opencouch_threads.sqlite3
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from agent.memory.modes import MemoryMode
from agent.persistence import (
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    ThreadSummary,
)
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
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from services.llm.base import BaseLLMClient

CLI_THEME = Theme(
    {
        "primary": "bold #3d9990",
        "accent": "bold #d78b5f",
        "muted": "#838881",
        "info": "#a8cdc9",
        "success": "bold #65b8af",
        "warning": "bold #f0ad7e",
        "danger": "bold #e46e62",
        "panel": "#2d7a74",
    }
)

console = Console(theme=CLI_THEME)


@dataclass(slots=True)
class RunnerSession:
    """Mutable local session state for the interactive CLI."""

    requested_mode: str
    resolved_mode: str
    llm_client: BaseLLMClient | None
    thread_id: str
    sqlite_path: str
    memory_mode: str
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
    parser.add_argument(
        "--memory-mode",
        choices=["guest", "persistent", "ask"],
        default="ask",
        help="Local memory behavior: guest (ephemeral), persistent (SQLite), or ask at startup.",
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


def resolve_memory_mode(memory_mode: str) -> str:
    """Resolve memory mode from CLI arg, prompting when needed."""

    if memory_mode in {"guest", "persistent"}:
        return memory_mode

    console.print(
        Panel(
            "[info][1][/info] Guest Mode (private, in-memory only)\n"
            "[info][2][/info] Persistent Mode (save local memory in SQLite)",
            title="[primary]Choose Memory Mode[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    while True:
        choice = Prompt.ask("[accent]select[/accent]", default="1").strip()
        if choice == "1":
            return "guest"
        if choice == "2":
            return "persistent"
        render_info("Please choose 1 (guest) or 2 (persistent).", style="warning")


def render_header(mode: str, thread_id: str, memory_mode: str) -> None:
    """Render the CLI header.

    Args:
        mode: Effective runtime mode for the current session.
        thread_id: Active persisted thread identifier.
        memory_mode: Local memory mode (guest or persistent).

    Returns:
        None.
    """

    console.print(Rule("[primary]OpenCouch CLI[/primary]", style="panel"))
    console.print(
        Text.from_markup(
            f"[muted]mode:[/muted] [primary]{mode}[/primary]  [muted]|[/muted]  "
            f"[muted]memory:[/muted] [accent]{memory_mode}[/accent]  [muted]|[/muted]  "
            f"[muted]thread:[/muted] [info]{thread_id}[/info]  [muted]|[/muted]  "
            "[muted]type[/muted] [accent]exit[/accent] [muted]or[/muted] "
            "[accent]quit[/accent] [muted]to stop[/muted]"
        )
    )
    console.print(
        "[muted]slash commands:[/muted] [info]/help, /status, /history, /context, "
        "/memory, /threads, /resume, /new, /reset, /clear, /mode, /exit[/info]\n"
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
        "[danger]Crisis Reply[/danger]"
        if is_crisis
        else "[success]Support Reply[/success]"
    )
    border_style = "danger" if is_crisis else "success"
    console.print(
        Panel(
            response_text,
            title=title,
            border_style=border_style,
            box=box.ROUNDED,
        )
    )


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

    table = Table(show_header=True, header_style="primary", box=box.SIMPLE_HEAVY)
    table.add_column("mode", style="info", no_wrap=True)
    table.add_column("source", style="muted", no_wrap=True)
    table.add_column("mode type", style="info", no_wrap=True)
    table.add_column("type", style="info", no_wrap=True)
    table.add_column("safety", style="warning", no_wrap=True)
    table.add_column("clarify", justify="center", no_wrap=True)
    table.add_column("crisis", justify="center", no_wrap=True)
    table.add_column("reason", style="muted")
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
    console.print(
        Panel(
            table,
            title="[primary]Turn Diagnostics[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
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
                "[muted]No session context yet. Send a message first.[/muted]",
                title="[primary]Session Context[/primary]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    progress = state.get("progress", {})
    memory = state.get("memory", {})
    response_state = state.get("response", {})

    table = Table(show_header=False, box=box.SIMPLE_HEAVY)
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="info")
    table.add_row("turn_count", str(progress.get("turn_count", 0)))
    table.add_row(
        "working_memory",
        " | ".join(state.get("working_memory", []))
        if state.get("working_memory")
        else "-",
    )
    table.add_row("current_goal", memory.get("current_goal") or "-")
    table.add_row("response_guidance", response_state.get("guidance") or "-")
    active_concerns = memory.get("active_concerns") or []
    table.add_row(
        "active_concerns",
        ", ".join(active_concerns) if active_concerns else "-",
    )
    open_loops = memory.get("open_loops") or []
    table.add_row(
        "open_loops",
        " | ".join(open_loops) if open_loops else "-",
    )
    table.add_row("session_summary", memory.get("summary", ""))
    console.print(
        Panel(
            table,
            title="[primary]Session Context[/primary]",
            subtitle="[muted]what the graph is carrying forward[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_memory_status(runtime: PersistentAgentRuntime) -> None:
    """Render the memory layer's current state.

    Shows the memory mode, per-namespace record counts from the unified
    memory store, and the crisis log record count. Placeholders are
    included for fields that land in later phases (last consolidation
    timestamp in phase 4, proactive recall toggle in phase 2+).

    Args:
        runtime: Active persistent runtime. Reads the memory_store and
            crisis_log_backend via their public properties.

    Returns:
        None.
    """

    store = runtime.memory_store
    crisis_log = runtime.crisis_log_backend

    # Aggregate store record counts by namespace kind. Namespaces are
    # (user_id, kind) tuples; group by the kind for a clean summary.
    counts_by_kind: dict[str, int] = {"semantic": 0, "episodic": 0, "procedural": 0}
    for namespace in store.namespaces():
        if len(namespace) >= 2 and namespace[1] in counts_by_kind:
            counts_by_kind[namespace[1]] += store.record_count(namespace)
    total_records = store.record_count()

    # Crisis log record count — always-on backend, independent of mode.
    crisis_log_count = 0
    record_count_fn = getattr(crisis_log, "record_count", None)
    if callable(record_count_fn):
        crisis_log_count = record_count_fn()

    table = Table(show_header=False, box=box.SIMPLE_HEAVY)
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="info")
    table.add_row("memory_mode", str(runtime.memory_mode))
    table.add_row("semantic facts", str(counts_by_kind["semantic"]))
    table.add_row("episodic arcs", str(counts_by_kind["episodic"]))
    table.add_row("procedural rules", str(counts_by_kind["procedural"]))
    table.add_row("total memory records", str(total_records))
    table.add_row("crisis log events", str(crisis_log_count))
    # Placeholders for fields that land in later phases — shown so the
    # command shape stays stable as features are added.
    table.add_row("last consolidation", "(phase 4)")
    table.add_row("proactive recall", "(phase 2+)")
    console.print(
        Panel(
            table,
            title="[primary]Memory Status[/primary]",
            subtitle="[muted]what the memory layer is holding[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )


def render_memory_list(runtime: PersistentAgentRuntime) -> None:
    """Render every semantic memory record in a browsable table.

    Shipped in v0.3.1 as a dogfood-observability tool: without this,
    answering "what did extraction actually write?" requires a probe
    script. With it, operators can type ``/memory list`` at any point
    in a session and see the evidence quotes, categories, and predicates
    for each fact the extractor has landed.

    Scope:
    - Read-only. Mutation commands (``/memory forget``, ``/memory clear``)
      are scoped to v0.9 alongside the full CLI memory suite.
    - Semantic namespace only. Episodic lands in v0.4, procedural in
      v0.7; when those namespaces start getting written, this function
      will grow additional tables.
    - Evidence quotes are truncated to 80 chars in the table and shown
      in full in a follow-up details block only for records whose quote
      is longer than 80 chars — keeps the happy path clean.
    - Sorted by insertion order (the order records were written). When
      v0.8's consolidation tier ships, this may change to sort by
      last_referenced_at descending.

    Args:
        runtime: Active persistent runtime. Reads the memory_store via
            its public property.
    """

    store = runtime.memory_store

    # Gather all semantic records across every namespace. The store is
    # namespaced by (user_id, kind), so we iterate every namespace that
    # has kind == "semantic" and collect its records in insertion order.
    semantic_records: list[tuple[str, dict[str, object]]] = []
    for namespace in store.namespaces():
        if len(namespace) < 2 or namespace[1] != "semantic":
            continue
        # bucket.records preserves insertion order (dict in Python 3.7+),
        # so iterating gives us the chronological list.
        bucket = store._buckets.get(namespace)  # noqa: SLF001 — debug tool
        if bucket is None:
            continue
        for key, record in bucket.records.items():
            semantic_records.append((key, record.value))

    if not semantic_records:
        console.print(
            Panel(
                "[muted]No semantic records in the store yet. The extractor "
                "writes facts from concrete user statements; transient feelings, "
                "questions, and small talk produce zero extractions by design. "
                "Try mentioning a named person, a coping strategy, or a "
                "recurring trigger to see the store populate.[/muted]",
                title="[primary]Memory List (semantic)[/primary]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    table = Table(
        show_header=True,
        header_style="primary",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("#", style="muted", no_wrap=True, width=3)
    table.add_column("category", style="accent", no_wrap=True)
    table.add_column("predicate", style="info", no_wrap=True)
    table.add_column("evidence quote", style="info")
    table.add_column("conf", style="muted", no_wrap=True, width=6)

    # Truncation threshold for the inline quote column. Longer quotes
    # get shown in full below the table so the happy-path rendering
    # stays compact and scannable.
    quote_inline_limit = 80
    long_quotes: list[tuple[int, str]] = []

    for idx, (_key, value) in enumerate(semantic_records, start=1):
        category = str(value.get("category", "?"))
        predicate = str(value.get("predicate", "?"))
        confidence = str(value.get("confidence", "?"))
        quote = str(value.get("evidence_quote", ""))
        if len(quote) > quote_inline_limit:
            quote_display = quote[:quote_inline_limit].rstrip() + "…"
            long_quotes.append((idx, quote))
        else:
            quote_display = quote
        table.add_row(str(idx), category, predicate, quote_display, confidence)

    console.print(
        Panel(
            table,
            title=f"[primary]Memory List (semantic)[/primary] "
            f"[muted]— {len(semantic_records)} record(s)[/muted]",
            subtitle="[muted]what the extractor has written so far[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )

    if long_quotes:
        # Render full-length quotes below the table for any record whose
        # evidence was truncated. This keeps the table scannable while
        # still giving operators access to the verbatim text when they
        # need it.
        console.print()
        console.print("[muted]Full quotes (truncated in the table above):[/muted]")
        for idx, quote in long_quotes:
            console.print(f"  [accent]#{idx}[/accent] [info]{quote}[/info]")
    console.print()
    console.print()


def render_help() -> None:
    """Render the available slash commands.

    Returns:
        None.
    """

    table = Table(show_header=True, header_style="primary", box=box.SIMPLE_HEAVY)
    table.add_column("command", style="accent", no_wrap=True)
    table.add_column("description", style="info")
    table.add_row("/help", "Show available commands.")
    table.add_row("/status", "Show current mode and session stats.")
    table.add_row("/history [n]", "Show the last n transcript messages. Default: 6.")
    table.add_row("/context", "Show the latest derived session context snapshot.")
    table.add_row(
        "/memory status", "Show memory layer state (counts, mode, crisis log)."
    )
    table.add_row(
        "/memory list",
        "List every semantic fact the extractor has written this session.",
    )
    table.add_row("/threads [n]", "List persisted thread ids. Default: 12.")
    table.add_row("/resume <thread-id>", "Switch to an existing persisted thread.")
    table.add_row(
        "/new [thread-id]", "Start a fresh thread without restarting the CLI."
    )
    table.add_row("/reset", "Clear the conversation history.")
    table.add_row("/clear", "Clear the terminal and redraw the header.")
    table.add_row(
        "/mode <deterministic|hybrid|auto>",
        "Switch LLM resolution mode for future turns.",
    )
    table.add_row("/end", "End the current session with a closing message.")
    table.add_row("/exit", "End the session immediately.")
    console.print(
        Panel(
            table,
            title="[primary]Commands[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
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
    table = Table(show_header=False, box=box.SIMPLE_HEAVY)
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="info")
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
        Panel(
            table,
            title="[primary]Session Status[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
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
                "[muted]No conversation history yet.[/muted]",
                title="[primary]History[/primary]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    recent = session.history[-max(1, limit) :]
    table = Table(show_header=True, header_style="primary", box=box.SIMPLE_HEAVY)
    table.add_column("role", style="accent", no_wrap=True)
    table.add_column("content", style="info")
    for message in recent:
        role = message.role.value
        table.add_row(role, message.content)
    console.print(
        Panel(
            table,
            title="[primary]Recent History[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_info(message: str, *, style: str = "panel") -> None:
    """Render a lightweight informational panel.

    Args:
        message: Informational message to display.
        style: Border color/style token for the panel.

    Returns:
        None.
    """

    console.print(Panel(message, border_style=style, expand=False, box=box.ROUNDED))
    console.print()


def render_threads(
    threads: list[ThreadSummary],
    *,
    active_thread_id: str,
) -> None:
    """Render a compact table of persisted thread summaries."""

    if not threads:
        render_info("No persisted threads found.", style="warning")
        return

    table = Table(show_header=True, header_style="primary", box=box.SIMPLE_HEAVY)
    table.add_column("thread id", style="info")
    table.add_column("turns", style="muted", justify="right", no_wrap=True)
    table.add_column("messages", style="muted", justify="right", no_wrap=True)
    table.add_column("active", style="accent", no_wrap=True)
    for thread in threads:
        table.add_row(
            thread.thread_id,
            str(thread.turn_count),
            str(thread.message_count),
            "yes" if thread.thread_id == active_thread_id else "",
        )
    console.print(
        Panel(
            table,
            title="[primary]Persisted Threads[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
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


def generate_thread_id() -> str:
    """Generate a new local thread id for ad hoc CLI sessions."""

    return f"local-{uuid4().hex[:12]}"


async def switch_thread(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    *,
    thread_id: str,
    require_existing: bool,
) -> bool:
    """Switch the CLI session to another thread id.

    Args:
        session: Mutable CLI session state.
        runtime: Persistent runtime backing the active SQLite store.
        thread_id: Target thread identifier.
        require_existing: Whether the target thread must already exist.

    Returns:
        `True` when the switch succeeded.
    """

    target_thread_id = thread_id.strip()
    if not target_thread_id:
        return False

    state = await runtime.get_state(target_thread_id)
    if require_existing and state is None:
        return False
    if not require_existing and state is not None:
        return False

    session.thread_id = target_thread_id
    session.last_context = state
    session.history = await runtime.get_history(target_thread_id)
    return True


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

    if command == "/end":
        # Soft session termination: print a farewell and exit the loop.
        # In v0.4 this will also trigger summarize_session_node to write
        # an episodic arc before exit. For now the farewell is the whole
        # behavior.
        render_info(
            "Take care. We can pick this back up whenever you want. "
            f"(Thread: {session.thread_id})",
            style="success",
        )
        return False

    if command == "/memory":
        # v0.3.1 supports `/memory status` and `/memory list`. The list
        # command is a read-only dogfood tool added when the v0.3.1
        # retrieval work shipped — without it, answering "what did
        # extraction actually write?" required a probe script. Mutation
        # commands (/memory forget, /memory clear) remain scoped to v0.9.
        if len(args) == 0 or args[0] == "status":
            render_memory_status(runtime)
            return True
        if args[0] == "list":
            render_memory_list(runtime)
            return True
        render_info(
            "Unknown /memory subcommand. Available in v0.3.1: status, list",
            style="warning",
        )
        return True

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
                render_info("Usage: /history [count]", style="warning")
                return True
        render_history(session, limit=limit)
        return True

    if command == "/context":
        render_context(session.last_context)
        return True

    if command == "/threads":
        limit = 12
        if args:
            try:
                limit = max(1, int(args[0]))
            except ValueError:
                render_info("Usage: /threads [count]", style="warning")
                return True
        render_threads(
            await runtime.list_threads(limit=limit),
            active_thread_id=session.thread_id,
        )
        return True

    if command == "/resume":
        if len(args) != 1:
            render_info("Usage: /resume <thread-id>", style="warning")
            return True
        thread_id = args[0]
        if thread_id == session.thread_id:
            render_info(f"Already on thread {thread_id}.", style="warning")
            return True
        if not await switch_thread(
            session,
            runtime,
            thread_id=thread_id,
            require_existing=True,
        ):
            render_info(
                f"No persisted thread found for {thread_id}. Use /threads to inspect known ids.",
                style="warning",
            )
            return True
        render_header(session.resolved_mode, session.thread_id, session.memory_mode)
        render_info(
            f"Resumed thread {session.thread_id} with {len(session.history)} stored messages.",
            style="success",
        )
        if session.history:
            render_history(session, limit=len(session.history))
        if session.last_context is not None:
            render_context(session.last_context)
        return True

    if command == "/new":
        if len(args) > 1:
            render_info("Usage: /new [thread-id]", style="warning")
            return True
        thread_id = args[0] if args else generate_thread_id()
        if thread_id == session.thread_id:
            render_info(
                "The requested thread id is already active. Choose a different id or use /reset.",
                style="warning",
            )
            return True
        if not await switch_thread(
            session,
            runtime,
            thread_id=thread_id,
            require_existing=False,
        ):
            render_info(
                f"Thread {thread_id} already exists. Use /resume {thread_id} or choose another id.",
                style="warning",
            )
            return True
        render_header(session.resolved_mode, session.thread_id, session.memory_mode)
        render_info(f"Started new thread {session.thread_id}.", style="success")
        return True

    if command == "/reset":
        await runtime.reset_thread(session.thread_id)
        session.history.clear()
        session.last_context = None
        render_info("Persisted thread state cleared.", style="warning")
        return True

    if command == "/clear":
        console.clear()
        render_header(session.resolved_mode, session.thread_id, session.memory_mode)
        return True

    if command == "/mode":
        if len(args) != 1 or args[0] not in {"deterministic", "hybrid", "auto"}:
            render_info("Usage: /mode <deterministic|hybrid|auto>", style="warning")
            return True
        set_mode(session, args[0])
        render_info(
            f"Mode updated. requested={session.requested_mode}, resolved={session.resolved_mode}",
            style="success" if session.llm_client is not None else "warning",
        )
        return True

    render_info(f"Unknown command: {command}. Try /help.", style="warning")
    return True


async def chat_loop(
    mode: str,
    *,
    thread_id: str,
    sqlite_path: str,
    memory_mode: str,
) -> None:
    """Run the interactive CLI loop.

    Args:
        mode: Requested runtime mode for model resolution.
        thread_id: Stable thread identifier for the local conversation.
        sqlite_path: SQLite file used for persisted thread checkpoints.
        memory_mode: Local memory mode ("guest" or "persistent").

    Returns:
        None.
    """

    llm_client, resolved_mode = resolve_llm_client(mode)
    # CLI uses the string labels "guest" and "persistent" for the user-facing
    # mode. Translate to the graph-internal MemoryMode enum for the runtime.
    runtime_memory_mode = (
        MemoryMode.INCOGNITO if memory_mode == "guest" else MemoryMode.LOCAL
    )
    is_guest_mode = runtime_memory_mode == MemoryMode.INCOGNITO
    session = RunnerSession(
        requested_mode=mode,
        resolved_mode=resolved_mode,
        llm_client=llm_client,
        thread_id=thread_id,
        sqlite_path=":memory:" if is_guest_mode else sqlite_path,
        memory_mode=memory_mode,
    )

    async with PersistentAgentRuntime(
        sqlite_path,
        memory_mode=runtime_memory_mode,
    ) as runtime:
        session.history = await runtime.get_history(thread_id)
        session.last_context = await runtime.get_state(thread_id)

        render_header(session.resolved_mode, session.thread_id, session.memory_mode)
        if session.history:
            render_info(
                f"Resumed thread {session.thread_id} with {len(session.history)} stored messages.",
                style="success",
            )

        while True:
            try:
                user_text = Prompt.ask("[accent]you[/accent]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[muted]session ended[/muted]")
                break

            if not user_text:
                continue
            if user_text.startswith("/"):
                if not await handle_command(user_text, session, runtime):
                    break
                continue
            if user_text.lower() in {"exit", "quit"}:
                break

            console.print(Rule(style="panel"))
            accumulated_text = ""
            final_output = None

            _STAGE_LABELS = {
                "load_memory": "loading memory",
                "memory_profile_load": "loading profile memory",
                "memory_graph_load": "querying graph memory",
                "memory_profile_save": "saving profile memory",
                "memory_graph_save": "writing graph memory",
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
                        detail = f" ({event.detail})" if event.detail else ""
                        live.update(Text(f"  ● {label}{detail}", style="muted"))

                    elif isinstance(event, ChunkEvent):
                        accumulated_text += event.text
                        live.update(
                            Panel(
                                accumulated_text,
                                title="[success]Assistant[/success]",
                                border_style="success",
                                box=box.ROUNDED,
                            )
                        )

                    elif isinstance(event, DoneEvent):
                        final_output = event.output
                        # Render the final panel as the last Live frame so it
                        # stays on screen when Live exits — no duplicate render.
                        is_crisis = final_output.response_type.value == "crisis"
                        title = (
                            "[danger]Crisis Reply[/danger]"
                            if is_crisis
                            else "[success]Support Reply[/success]"
                        )
                        border = "danger" if is_crisis else "success"
                        live.update(
                            Panel(
                                final_output.response_text,
                                title=title,
                                border_style=border,
                                box=box.ROUNDED,
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
    thread_id = args.thread_id or generate_thread_id()
    sqlite_path = str(Path(args.sqlite_path).expanduser())
    memory_mode = resolve_memory_mode(args.memory_mode)
    asyncio.run(
        chat_loop(
            args.mode,
            thread_id=thread_id,
            sqlite_path=sqlite_path,
            memory_mode=memory_mode,
        )
    )
    return 0

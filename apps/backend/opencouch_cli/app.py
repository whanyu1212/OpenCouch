"""Interactive CLI for the OpenCouch agent runtime.

Run from ``apps/backend/`` so default local paths resolve to the
backend working directory. The CLI supports deterministic smoke tests,
hybrid LLM runs, persistent local memory, thread switching, and optional
voice mode.

Common invocations:

``uv run python -m opencouch_cli --mode deterministic --memory-mode guest``
    Zero LLM calls, in-memory only. Useful for checking rendering and
    deterministic graph paths.

``uv run python -m opencouch_cli --mode auto --memory-mode persistent``
    Real model when configured, durable local persistence, and memory
    writes enabled. Postgres is recommended; SQLite remains a legacy fallback.

``uv run python -m opencouch_cli --mode auto --memory-mode persistent --user-id alice``
    Stable owner namespace for semantic, episodic, and procedural memory
    across multiple threads.

Important slash commands:

``/memory status``
    Show memory counts, owner id, crisis-log count, and recall toggle.

``/memory list [facts|sessions|rules]``
    Browse stored semantic facts, episodic arcs, or procedural rules.

``/memory forget fact|session|rule <n>``
    Delete one owner-scoped memory item by its displayed index.

``/memory clear facts|sessions|rules|all``
    Clear owner-scoped memory content after typed confirmation.

``/memory purge-crisis [days]``
    Retention-purge crisis audit records after typed confirmation.

``/debug state``
    Dump the raw persisted graph state when rendered panels are not
    enough for diagnosis.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

from agent.memory.models import FeedbackLabel, FeedbackSource, StoredSessionArc
from agent.memory.modes import MemoryMode
from agent.memory.reconciliation import filter_active_semantic_records
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    ThreadSummary,
)
from agent.memory.procedural_profile import (
    aclear_procedural_rules,
    adelete_procedural_rule,
    aget_procedural_profile,
    aset_proactive_recall,
)
from agent.models import (
    AgentOutput,
    Channel,
    ChunkEvent,
    CrisisAssessment,
    DoneEvent,
    Message,
    MessageRole,
    ResponseCategory,
    ResponseReadyEvent,
    StatusEvent,
    friendly_stage,
)
from agent.state import AgentState
from agent.memory.entries import format_working_memory_entries
from config import (
    ResponseModelTier,
    create_configured_control_llm_client,
    create_configured_response_llm_client,
    get_settings,
)
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from llm.base import BaseLLMClient

CLI_THEME = Theme(
    {
        "primary": "bold #8FAE9D",  # soft sage — main emphasis
        "accent": "bold #A7BFA3",  # pale moss — secondary highlights
        "muted": "#7B817C",  # soft olive gray — subdued labels
        "info": "#D8DDD8",  # warm mist — body text
        "success": "bold #7FA08A",  # grounded green — positive states
        "warning": "bold #C9A56D",  # muted honey — caution
        "danger": "bold #C97B6B",  # soft terracotta — crisis/error
        "panel": "#58615C",  # deep moss gray — borders, rules
        "brand": "bold #9BB8C9",  # misty blue — brand text
        "hint": "dim #7B817C",  # faded olive gray — secondary hints
    }
)

console = Console(theme=CLI_THEME)


def _response_panel(
    output: AgentOutput,
    *,
    thread_id: str | None = None,
    turn_count: int | None = None,
) -> Panel:
    """Build the terminal assistant-response panel for one turn.

    Args:
        output: Agent output to display.
        thread_id: Optional thread id shown in the subtitle.
        turn_count: Optional turn count shown in the subtitle.

    Returns:
        Rich panel for the assistant response.
    """

    is_crisis = output.response_type.value == "crisis"
    title = (
        "[danger]  crisis  [/danger]" if is_crisis else "[success]  reply  [/success]"
    )
    border = "danger" if is_crisis else "panel"
    subtitle_parts: list[str] = []
    if thread_id:
        subtitle_parts.append(f"[muted]thread[/muted] [info]{thread_id}[/info]")
    if turn_count is not None:
        subtitle_parts.append(f"[muted]turn[/muted] [accent]{turn_count}[/accent]")
    if output.response_style:
        style_label = output.response_style
        if output.therapeutic_approach and output.therapeutic_approach != "none":
            style_label += f" [muted]/[/muted] {output.therapeutic_approach}"
        subtitle_parts.append(f"[muted]style[/muted] [primary]{style_label}[/primary]")

    return Panel(
        output.response_text,
        title=title,
        subtitle=Text.from_markup("  [panel]·[/panel]  ".join(subtitle_parts))
        if subtitle_parts
        else None,
        border_style=border,
        box=box.HEAVY if is_crisis else box.ROUNDED,
        padding=(1, 2),
    )


async def _drain_turn_stream_tail(
    stream,
) -> AgentOutput:
    """Consume the rest of a partially-read turn stream to completion.

    Args:
        stream: Async turn event stream after a response-ready event.

    Returns:
        Final agent output from the terminal done event.

    Raises:
        RuntimeError: If the stream ends without a done event.
    """

    final_output: AgentOutput | None = None
    async for event in stream:
        if isinstance(event, DoneEvent):
            final_output = event.output

    if final_output is None:
        raise RuntimeError(
            "run_turn_stream ended without a DoneEvent after response_ready."
        )
    return final_output


@dataclass(slots=True)
class RunnerSession:
    """Mutable local session state for the interactive CLI."""

    requested_mode: str
    resolved_mode: str
    llm_client: BaseLLMClient | None
    thread_id: str
    sqlite_path: str
    memory_mode: str
    # Optional stable owner for long-term memory. Falls back to thread_id.
    user_id: str | None = None
    history: list[Message] = field(default_factory=list)
    last_context: AgentState | None = None
    response_model_tier: ResponseModelTier = "fast"
    response_llm_client: BaseLLMClient | None = None

    def owner_id(self) -> str:
        """Return the effective owner identifier for memory operations.

        Returns:
            Explicit ``user_id`` when set, otherwise the active thread id.
        """

        return self.user_id or self.thread_id


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
        "--user-id",
        default=None,
        help=(
            "Stable owner identifier for long-term memory (semantic facts, "
            "episodic arcs, procedural rules). When set, memory writes are "
            "namespaced by this user_id rather than the thread_id, so "
            "switching threads preserves memory across sessions. Only "
            "meaningful in persistent memory mode — guest mode has no "
            "long-term storage to namespace. When omitted, falls back to "
            "the thread_id for backward compatibility."
        ),
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(DEFAULT_THREAD_DB_PATH),
        help=(
            "Legacy SQLite path for LangGraph thread checkpoints. Deprecated "
            "for normal local development; prefer "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres."
        ),
    )
    parser.add_argument(
        "--memory-sqlite-path",
        default=str(DEFAULT_MEMORY_DB_PATH),
        help=(
            "Legacy SQLite path for the memory store (semantic facts + "
            "episodic arcs). Deprecated for normal local development; prefer "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres."
        ),
    )
    parser.add_argument(
        "--crisis-log-sqlite-path",
        default=str(DEFAULT_CRISIS_LOG_DB_PATH),
        help=(
            "Legacy SQLite path for the crisis log (safety audit trail). "
            "Deprecated for normal local development; prefer "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres."
        ),
    )
    parser.add_argument(
        "--memory-mode",
        choices=["guest", "persistent", "ask"],
        default="ask",
        help="Local memory behavior: guest (ephemeral), persistent (configured backend), or ask at startup.",
    )
    parser.add_argument(
        "--response-model-tier",
        choices=["fast", "quality"],
        default="fast",
        help=(
            "Text response tier for therapeutic prose generation. "
            "'fast' favors lower latency; 'quality' favors richer replies."
        ),
    )
    parser.add_argument(
        "--disable-tracing",
        action="store_true",
        default=False,
        help=(
            "Disable optional tracing integrations for this CLI run. "
            "Equivalent to setting OPENCOUCH_DISABLE_TRACING=1."
        ),
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        default=False,
        help=(
            "Start voice mode instead of the text CLI. Launches the "
            "FastAPI server plus the LiveKit voice worker, then opens "
            "the LiveKit browser test page. Requires LIVEKIT_URL, "
            "LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the voice mode server. Default: 8000.",
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
        return create_configured_control_llm_client(), "hybrid"

    try:
        return create_configured_control_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def resolve_response_llm_client(
    mode: str,
    tier: ResponseModelTier,
) -> BaseLLMClient | None:
    """Resolve the response-writer LLM client for the selected mode and tier.

    Args:
        mode: Requested runtime mode from the CLI.
        tier: Response model tier for user-facing prose.

    Returns:
        Configured response LLM client, or None in deterministic mode
        or when client setup fails.
    """

    if mode == "deterministic":
        return None

    try:
        return create_configured_response_llm_client(tier)
    except Exception:
        return None


def resolve_memory_mode(memory_mode: str) -> str:
    """Resolve memory mode from CLI arg, prompting when needed.

    Args:
        memory_mode: Raw CLI memory mode argument.

    Returns:
        Resolved memory mode, either ``"guest"`` or ``"persistent"``.
    """

    if memory_mode in {"guest", "persistent"}:
        return memory_mode

    console.print()
    console.print(Rule(style="panel", characters="─"))
    console.print("  [primary]Choose Memory Mode[/primary]", highlight=False)
    console.print()
    console.print(
        "  [accent]1[/accent]  [info]Guest Mode[/info]  [hint]— private, in-memory only[/hint]"
    )
    console.print(
        "  [accent]2[/accent]  [info]Persistent Mode[/info]  [hint]— save local memory in SQLite[/hint]"
    )
    console.print()
    console.print(Rule(style="panel", characters="─"))
    while True:
        choice = Prompt.ask("  [muted]select[/muted]", default="1").strip()
        if choice == "1":
            return "guest"
        if choice == "2":
            return "persistent"
        render_info("Please choose 1 (guest) or 2 (persistent).", style="warning")


def render_header(
    mode: str,
    thread_id: str,
    memory_mode: str,
    *,
    user_id: str | None = None,
    response_model_tier: ResponseModelTier | None = None,
) -> None:
    """Render the CLI header.

    Args:
        mode: Effective runtime mode for the current session.
        thread_id: Active persisted thread identifier.
        memory_mode: Local memory mode (guest or persistent).
        user_id: Optional explicit owner identifier (set via
            ``--user-id``). When set, the header shows a ``user:``
            field so the operator can see at a glance which memory
            namespace the session is writing to. When None, the
            field is omitted for the thread-scoped display.
        response_model_tier: Optional response tier label.

    Returns:
        None.
    """

    # Block-letter wordmark — hand-crafted using Unicode half-block
    # characters for a bold, modern terminal logo.
    _LOGO_LINES = [
        "█▀▀█ █▀▀█ █▀▀ █▀▀▄   █▀▀ █▀▀█ █  █ █▀▀ █  █",
        "█  █ █▄▄█ █▀▀ █  █   █   █  █ █  █ █   █▀▀█",
        "▀▀▀▀ ▀    ▀▀▀ ▀  ▀   ▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀ ▀  ▀",
    ]

    console.print()
    console.print(Rule(style="panel", characters="─"))
    console.print()
    for line in _LOGO_LINES:
        console.print(Text(line, style="primary", justify="center"))
    console.print(Text("CLI", style="accent", justify="center"))
    console.print()
    console.print(
        Text(
            "A calm workspace for supportive conversations",
            style="muted",
            justify="center",
        )
    )
    console.print(
        Text(
            "PRIVATE BY DEFAULT  •  MEMORY ON YOUR TERMS",
            style="hint",
            justify="center",
        )
    )
    console.print()
    console.print(Rule(style="panel", characters="─"))
    console.print()

    # Session metadata — compact single-line layout
    session_parts: list[str] = [
        f"[muted]session[/muted] [primary]{mode}[/primary]",
        f"[muted]memory[/muted] [accent]{memory_mode}[/accent]",
    ]
    if response_model_tier is not None:
        session_parts.append(
            f"[muted]response[/muted] [success]{response_model_tier}[/success]"
        )
    console.print(Text.from_markup("   " + "  [panel]·[/panel]  ".join(session_parts)))

    identity_parts: list[str] = [
        f"[muted]thread[/muted] [info]{thread_id}[/info]",
    ]
    if user_id:
        identity_parts.append(f"[muted]owner[/muted] [info]{user_id}[/info]")
    console.print(Text.from_markup("   " + "  [panel]·[/panel]  ".join(identity_parts)))

    exit_hint = Text()
    exit_hint.append("   type ", style="hint")
    exit_hint.append("exit", style="accent")
    exit_hint.append(" or ", style="hint")
    exit_hint.append("quit", style="accent")
    exit_hint.append(" to stop", style="hint")
    console.print(exit_hint)

    console.print()

    # Quick-action hints — grouped on two lines
    console.print(
        Text.from_markup(
            "   [muted]quick actions[/muted]  "
            "[info]/help[/info]  [info]/status[/info]  [info]/history[/info]  "
            "[info]/context[/info]  [info]/memory[/info]  [info]/threads[/info]"
        )
    )
    console.print(
        Text.from_markup(
            "                  "
            "[info]/resume[/info]  [info]/new[/info]  [info]/reset[/info]  "
            "[info]/clear[/info]  [info]/mode[/info]  [info]/response-tier[/info]  "
            "[info]/debug[/info]  [info]/exit[/info]"
        )
    )
    console.print()


def render_response(
    response_text: str,
    *,
    is_crisis: bool,
    thread_id: str | None = None,
    turn_count: int | None = None,
) -> None:
    """Render the assistant reply inside a styled panel.

    Args:
        response_text: Generated assistant reply text.
        is_crisis: Whether the response is a crisis response.
        thread_id: Optional thread label shown in the panel footer.
        turn_count: Optional turn counter shown in the panel footer.

    Returns:
        None.
    """

    output = AgentOutput(
        response_text=response_text,
        response_type=(
            ResponseCategory.CRISIS if is_crisis else ResponseCategory.THERAPEUTIC
        ),
        crisis=CrisisAssessment(),
        response_style="crisis" if is_crisis else "support",
        response_style_source="cli",
        diagnostics={},
    )
    console.print(_response_panel(output, thread_id=thread_id, turn_count=turn_count))


def render_meta(
    *,
    response_style: str | None,
    response_style_source: str | None,
    response_style_type: str | None,
    response_type: str,
    level: int,
    needs_clarification: bool,
    needs_crisis_response: bool,
    reason: str,
    diagnostics: dict | None = None,
    memory_deltas: dict | None = None,
    verbose: bool = False,
) -> None:
    """Render classifier metadata as a compact diagnostics line.

    Args:
        response_style: Therapeutic response style selected for the turn.
        response_style_source: Source of the response style decision.
        response_style_type: Deprecated style-type label retained for
            call-site compatibility.
        response_type: High-level response category.
        level: Crisis severity level.
        needs_clarification: Whether safety clarification is needed.
        needs_crisis_response: Whether a crisis response is required.
        reason: One-line classifier reason.
        diagnostics: Optional per-node timing and decision metadata.
        memory_deltas: Optional before/after store count deltas.
        verbose: Whether to render the detailed timings table.

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

    diag = diagnostics or {}
    deltas = memory_deltas or {}

    turn_total = diag.get("turn_total_ms")
    try:
        turn_total_label = (
            f"{float(turn_total):.0f}ms" if turn_total is not None else "—"
        )
    except (TypeError, ValueError):
        turn_total_label = "—"

    delta_parts: list[str] = []
    for key, short in (("semantic", "s"), ("episodic", "e"), ("procedural", "p")):
        val = deltas.get(key)
        if val is None:
            continue
        delta_parts.append(f"{short}{val:+d}")
    delta_label = " ".join(delta_parts) if delta_parts else "no writes"

    reason_label = reason if len(reason) <= 72 else f"{reason[:69]}..."
    summary = Text.from_markup(
        "  [panel]·[/panel]  ".join(
            [
                f"[primary]{response_style or '-'}[/primary][muted]/{response_style_source or '-'}[/muted]",
                f"[info]{response_type}[/info]",
                f"[warning]{safety_status}[/warning]",
                f"[accent]{turn_total_label}[/accent]",
                f"[success]{delta_label}[/success]",
                f"[hint]{reason_label or '-'}[/hint]",
            ]
        )
    )
    console.print(
        Panel(
            summary,
            title="[muted]diagnostics[/muted]",
            border_style="panel",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )

    if verbose:
        _render_stage_timings(diag, deltas)
        console.print()


def _render_stage_timings(diagnostics: dict, memory_deltas: dict) -> None:
    """Render the per-turn stage timings + memory-write table.

    Extracted from ``render_meta`` for testability. Skips entirely when
    both inputs are empty; that is the "tests constructed an AgentOutput
    manually and didn't populate diagnostics" case and there's nothing
    useful to show.

    Timings come from the diagnostics dict stamped by each node:

    - ``load_memory_ms`` from ``run_load_memory_node``
    - ``crisis_gate_ms`` from ``run_crisis_gate_node``
    - ``extract_facts_ms`` from ``run_extract_semantic_facts_node``
    - ``extract_procedural_ms`` from ``run_extract_procedural_rules_node``
    - ``turn_total_ms`` stamped by ``run_turn`` / ``run_turn_stream``
      after the graph invocation completes

    A missing key renders as ``-`` so the table shape stays stable even
    when a skip path (incognito, no LLM) short-circuits a node before
    it wrote its timing. The "total" column is the outer ``run_turn``
    clock. It is useful because the node-level sums don't include edge work,
    checkpoint I/O, or the Python-side overhead between nodes.

    Memory-write deltas come from the CLI's before/after store counts
    (see the chat_loop bookkeeping), not from the diagnostics dict. The
    two paths carry different semantics: the diagnostics dict reports
    what the extractor nodes **tried to write** (``semantic_writes``,
    ``procedural_writes``), whereas the delta reports what actually
    **landed in the store** after any silent skip / dedup interactions.
    We render both when available to make "writer produced a candidate
    but dedup rejected it" visible at a glance.

    Args:
        diagnostics: Per-node timing and write-attempt metadata.
        memory_deltas: Store count deltas observed by the CLI.

    Returns:
        None.
    """

    if not diagnostics and not memory_deltas:
        return

    timing_table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    timing_table.add_column("stage", style="hint", no_wrap=True)
    timing_table.add_column("time (ms)", style="info", justify="right", no_wrap=True)
    timing_table.add_column("writes", style="accent", justify="right", no_wrap=True)
    timing_table.add_column("store Δ", style="success", justify="right", no_wrap=True)

    def _fmt_ms(key: str) -> str:
        """Format a millisecond diagnostic value.

        Args:
            key: Diagnostic key to read.

        Returns:
            Formatted value, or ``"-"`` when unavailable.
        """

        val = diagnostics.get(key)
        if val is None:
            return "-"
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return "-"

    def _fmt_count(key: str) -> str:
        """Format an integer diagnostic count.

        Args:
            key: Diagnostic key to read.

        Returns:
            Formatted count, or ``"-"`` when unavailable.
        """

        val = diagnostics.get(key)
        if val is None:
            return "-"
        return str(int(val))

    def _fmt_write_summary(
        key: str,
        *,
        held_key: str | None = None,
        repeat_key: str | None = None,
        drop_key: str | None = None,
    ) -> str:
        """Format write attempts with held/repeat/drop suffixes.

        Args:
            key: Base write-count diagnostic key.
            held_key: Optional held-count diagnostic key.
            repeat_key: Optional repeat-count diagnostic key.
            drop_key: Optional drop-count diagnostic key.

        Returns:
            Formatted write summary.
        """

        base = _fmt_count(key)
        if base == "-":
            return base

        extras: list[str] = []
        for label, extra_key in (
            ("h", held_key),
            ("r", repeat_key),
            ("d", drop_key),
        ):
            if extra_key is None:
                continue
            val = diagnostics.get(extra_key)
            if val is None or val == 0:
                continue
            extras.append(f"{label}{int(val)}")

        if not extras:
            return base
        return f"{base} ({' '.join(extras)})"

    def _fmt_delta(key: str) -> str:
        """Format a store-count delta.

        Args:
            key: Memory-delta key to read.

        Returns:
            Formatted signed delta, or ``"-"`` when unavailable.
        """

        val = memory_deltas.get(key)
        if val is None:
            return "-"
        return f"+{val}" if val > 0 else str(val)

    timing_table.add_row("load_memory", _fmt_ms("load_memory_ms"), "-", "-")
    timing_table.add_row("crisis_gate", _fmt_ms("crisis_gate_ms"), "-", "-")
    timing_table.add_row(
        "extract_facts",
        _fmt_ms("extract_facts_ms"),
        _fmt_write_summary(
            "semantic_writes",
            held_key="semantic_session_end_holds",
            repeat_key="semantic_repeat_required",
            drop_key="semantic_policy_drops",
        ),
        _fmt_delta("semantic"),
    )
    timing_table.add_row(
        "extract_procedural",
        _fmt_ms("extract_procedural_ms"),
        _fmt_write_summary(
            "procedural_writes",
            held_key="procedural_session_end_holds",
            drop_key="procedural_policy_drops",
        ),
        _fmt_delta("procedural"),
    )
    # Episodic writes happen at session end via the summarizer, not
    # per-turn, so only the store Δ column carries information here.
    # We still include the row so the panel layout stays stable.
    timing_table.add_row("episodic", "-", "-", _fmt_delta("episodic"))
    timing_table.add_row("turn_total", _fmt_ms("turn_total_ms"), "-", "-")

    console.print(
        Panel(
            timing_table,
            title="[muted]stage timings[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )


def render_context(state: AgentState | None) -> None:
    """Render the current structured session context.

    The panel shows working memory, session continuity, procedural
    rules, the proactive-recall toggle, and active guided-exercise
    state.

    Args:
        state: Most recent graph input state snapshot.

    Returns:
        None.
    """

    if state is None:
        console.print(
            Panel(
                "[hint]No session context yet. Send a message first.[/hint]",
                title="[muted]session context[/muted]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    session_progress = state.get("session_progress", {})
    exercise_state = state.get("exercise_state", {})
    session_memory = state.get("session_memory", {})
    procedural_profile = state.get("procedural_profile", {})
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")
    table.add_row("turn_count", str(session_progress.get("turn_count", 0)))

    # Keep each memory entry on its own wrapped line for terminal readability.
    working_memory = format_working_memory_entries(state.get("working_memory") or [])
    if working_memory:
        table.add_row(
            "working_memory",
            "\n".join(f"• {entry}" for entry in working_memory),
        )
    else:
        table.add_row("working_memory", "-")

    table.add_row("current_goal", session_memory.get("current_goal") or "-")
    active_concerns = session_memory.get("active_concerns") or []
    table.add_row(
        "active_concerns",
        ", ".join(active_concerns) if active_concerns else "-",
    )
    open_loops = session_memory.get("open_loops") or []
    table.add_row(
        "open_loops",
        "\n".join(f"• {loop}" for loop in open_loops) if open_loops else "-",
    )

    # Show the same procedural rule text injected into response prompts.
    procedural_rules = procedural_profile.get("procedural_rules") or []
    if procedural_rules:
        table.add_row(
            "procedural_rules",
            "\n".join(f"• {rule}" for rule in procedural_rules),
        )
    else:
        table.add_row("procedural_rules", "-")

    # Procedural rules are always applied; this toggle only controls
    # whether semantic/episodic memory is proactively referenced aloud.
    recall_enabled = bool(procedural_profile.get("proactive_recall_enabled", False))
    table.add_row("proactive_recall", "on" if recall_enabled else "off")

    # Surface mid-exercise state without requiring a raw state dump.
    exercise_type = exercise_state.get("exercise_type")
    exercise_step = exercise_state.get("exercise_step")
    if exercise_type:
        step_display = f" (step {exercise_step})" if exercise_step is not None else ""
        table.add_row("exercise", f"{exercise_type}{step_display}")

    table.add_row("session_summary", session_memory.get("summary", ""))
    console.print(
        Panel(
            table,
            title="[muted]session context[/muted]",
            subtitle="[hint]what the graph is carrying forward[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


async def _render_debug_state(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Dump the raw graph state for the active thread as JSON.

    Backs ``/debug state``. This is the full-state view for when the
    rendered context and timing panels are not enough to diagnose a
    turn.

    Pydantic models and other non-JSON types round-trip through
    ``default=str`` in ``json.dumps`` rather than crashing. Most
    state fields are already plain dicts (we serialize to JSON at
    checkpoint write time via LangGraph's JsonPlusSerializer), so
    this fallback only kicks in for an odd CrisisAssessment instance
    that survived the round-trip as a typed model.

    Degrades gracefully when no state exists yet (fresh thread or
    ``/reset`` was just called). Prints a warning panel instead of
    crashing on a None state.

    Args:
        runtime: Persistent runtime used to fetch checkpointed state.
        session: Active CLI session.

    Returns:
        None.
    """

    import json

    state = await runtime.get_state(session.thread_id)
    if state is None:
        console.print(
            Panel(
                "[muted]No state for this thread yet. Send a message first, "
                "or use /threads to pick a thread with prior turns.[/muted]",
                title="[muted]debug state[/muted]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    try:
        rendered = json.dumps(state, indent=2, default=str)
    except (TypeError, ValueError) as exc:
        rendered = f"<failed to serialize state as JSON: {exc}>"

    console.print(
        Panel(
            Text(rendered, style="info"),
            title=f"[muted]debug state ({session.thread_id})[/muted]",
            subtitle="[hint]raw graph state dict[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


async def render_memory_status(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Render the memory layer's current state.

    Shows the memory mode, owner-scoped user-facing memory counts,
    crisis log record count, session feedback count, and proactive
    recall state for the active owner.

    Args:
        runtime: Active persistent runtime. Reads the memory_store and
            crisis_log_backend via their public properties.
        session: Active CLI session. Used to identify the current
            thread for per-user state lookups (recall toggle).

    Returns:
        None.
    """

    store = runtime.memory_store
    crisis_log = runtime.crisis_log_backend
    session_feedback = runtime.session_feedback_backend

    owner = session.owner_id()
    semantic_records = await _collect_records_by_kind(
        runtime, kind="semantic", owner_id=owner
    )
    episodic_records = await _collect_records_by_kind(
        runtime, kind="episodic", owner_id=owner
    )
    total_store_records = await store.arecord_count()

    # Defensive getattr keeps old test doubles from breaking the status panel.
    crisis_log_count = 0
    arecord_count_fn = getattr(crisis_log, "arecord_count", None)
    if callable(arecord_count_fn):
        crisis_log_count = await arecord_count_fn()

    # Feedback is best-effort, so missing test doubles report zero.
    session_feedback_count = 0
    feedback_arecord_count_fn = getattr(session_feedback, "arecord_count", None)
    if callable(feedback_arecord_count_fn):
        session_feedback_count = await feedback_arecord_count_fn()

    profile = await aget_procedural_profile(store, user_id=owner)

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")
    table.add_row("memory_mode", str(runtime.memory_mode))
    # Surface the effective owner namespace to debug cross-thread memory.
    if session.user_id:
        table.add_row("owner_id", f"{owner} (from --user-id)")
    else:
        table.add_row("owner_id", f"{owner} (from thread_id)")
    table.add_row("semantic facts", str(len(semantic_records)))
    table.add_row("episodic arcs", str(len(episodic_records)))
    table.add_row("procedural rules", str(len(profile.rules)))
    table.add_row("total store records", str(total_store_records))
    table.add_row("crisis log events", str(crisis_log_count))
    table.add_row("session feedback records", str(session_feedback_count))
    recall_state = "on" if profile.proactive_recall_enabled else "off"
    table.add_row("proactive recall", recall_state)
    console.print(
        Panel(
            table,
            title="[muted]memory status[/muted]",
            subtitle="[hint]what the memory layer is holding[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )


async def _collect_records_by_kind(
    runtime: PersistentAgentRuntime,
    *,
    kind: str,
    owner_id: str | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Collect records for one kind, optionally scoped to one owner.

    The store is namespaced by ``(user_id, kind)``. When ``owner_id``
    is set, this reads only that owner's namespace. Otherwise it
    iterates every namespace whose second tuple element matches
    ``kind``. Used by the CLI renderers for both owner-scoped
    readouts and the full-store fallback paths.

    Args:
        runtime: Active persistent runtime.
        kind: Memory namespace kind to collect.
        owner_id: Optional owner namespace. When omitted, all namespaces
            of the requested kind are scanned.

    Returns:
        ``(key, value)`` pairs in store insertion order.
    """

    store = runtime.memory_store
    namespaces: list[tuple[str, ...]]
    if owner_id is not None:
        namespaces = [(owner_id, kind)]
    else:
        namespaces = [
            namespace
            for namespace in await store.anamespaces()
            if len(namespace) >= 2 and namespace[1] == kind
        ]

    records: list[tuple[str, dict[str, object]]] = []
    for namespace in namespaces:
        # Defensive cap; larger stores need pagination before CLI listing.
        namespace_records = await store.asearch(namespace, query=None, limit=1000)
        if kind == "semantic":
            namespace_records = filter_active_semantic_records(namespace_records)
        for record in namespace_records:
            records.append((record.key, record.value))
    return records


async def _collect_records_with_namespace(
    runtime: PersistentAgentRuntime,
    *,
    kind: str,
    owner_id: str,
) -> list[tuple[tuple[str, ...], str, dict[str, object]]]:
    """Collect ``(namespace, key, value)`` tuples for a kind + owner.

    Unlike :func:`_collect_records_by_kind` which returns just
    ``(key, value)`` for rendering, these handlers need the full
    namespace tuple so they can call ``store.adelete(namespace, key)``.

    The records are returned in the same insertion order as
    ``/memory list`` displays them, so the 1-indexed position the
    user types matches the position they see in the table. **If you
    change the sort order here, also change the corresponding table
    renderer; the indexes must stay synchronized or users will
    delete the wrong record.**

    Owner scoping: unlike the read-only ``_collect_records_by_kind``
    helper which iterates every namespace in the store (useful when
    rendering the full list), this helper filters to a single
    ``owner_id``. Destructive commands are always scoped to the
    active session's owner, never cross-user, so restricting the
    fetch at the source prevents any accidental cross-user deletion
    path from even being reachable through the CLI.

    Args:
        runtime: Active persistent runtime. Reads the memory store.
        kind: Namespace kind: ``"semantic"`` or ``"episodic"``.
            ``"procedural"`` is NOT supported here because procedural
            memory is stored as a single profile document per user,
            not as individual records; see the procedural forget /
            clear handlers which go through ``aget/aput_procedural_profile``
            instead.
        owner_id: The owner to scope the fetch to, typically
            ``session.owner_id()``.

    Returns:
        A list of ``(namespace, key, value)`` tuples in insertion order,
        ready to be 1-indexed for display and passed to ``adelete``.
    """

    target_namespace = (owner_id, kind)
    store = runtime.memory_store
    namespaces = await store.anamespaces()
    if target_namespace not in namespaces:
        return []

    records = await store.asearch(target_namespace, query=None, limit=1000)
    if kind == "semantic":
        records = filter_active_semantic_records(records)
    return [(target_namespace, record.key, record.value) for record in records]


def _format_entity_identifier(entity: object) -> str:
    """Return the ``identifier`` field of a serialized :class:`EntityRef`.

    The stored record's ``subject`` and ``object`` fields are dicts
    with ``type`` and ``identifier`` keys (from
    ``EntityRef.model_dump()``). This helper extracts the identifier
    for table rendering, falling back to ``"?"`` when the shape is
    wrong or the identifier is missing. Defensive against schema
    drift and malformed records.

    Args:
        entity: Serialized entity-like value.

    Returns:
        Entity identifier, or ``"?"`` when unavailable.
    """

    if isinstance(entity, dict):
        identifier = entity.get("identifier")
        if identifier:
            return str(identifier)
    return "?"


def _render_semantic_records_table(
    records: list[tuple[str, dict[str, object]]],
) -> None:
    """Render the semantic records as a table with a long-quote footer.

    Extracted from ``render_memory_list`` so the episodic rendering can
    use the same compact table pattern.

    The object column shows the target of each predicate so two facts
    with the same evidence quote remain distinguishable.

    Args:
        records: Semantic memory records as ``(key, value)`` pairs.

    Returns:
        None.
    """

    table = Table(
        show_header=True,
        header_style="muted",
        box=box.SIMPLE,
        show_lines=False,
    )
    table.add_column("#", style="muted", no_wrap=True, width=3)
    table.add_column("category", style="accent", no_wrap=True)
    table.add_column("predicate", style="info", no_wrap=True)
    table.add_column("object", style="accent", no_wrap=False, max_width=22)
    table.add_column("evidence quote", style="info")
    table.add_column("conf", style="muted", no_wrap=True, width=6)

    quote_inline_limit = 80
    long_quotes: list[tuple[int, str]] = []

    for idx, (_key, value) in enumerate(records, start=1):
        category = str(value.get("category", "?"))
        predicate = str(value.get("predicate", "?"))
        object_identifier = _format_entity_identifier(value.get("object"))
        confidence = str(value.get("confidence", "?"))
        quote = str(value.get("evidence_quote", ""))
        if len(quote) > quote_inline_limit:
            quote_display = quote[:quote_inline_limit].rstrip() + "…"
            long_quotes.append((idx, quote))
        else:
            quote_display = quote
        table.add_row(
            str(idx),
            category,
            predicate,
            object_identifier,
            quote_display,
            confidence,
        )

    console.print(
        Panel(
            table,
            title=f"[muted]memory · semantic[/muted] "
            f"[hint]— {len(records)} record(s)[/hint]",
            subtitle="[hint]what the extractor has written so far[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )

    if long_quotes:
        console.print()
        console.print("[muted]Full quotes (truncated in the table above):[/muted]")
        for idx, quote in long_quotes:
            console.print(f"  [accent]#{idx}[/accent] [info]{quote}[/info]")
    console.print()


def _render_episodic_records_table(
    records: list[tuple[str, dict[str, object]]],
) -> None:
    """Render the episodic session arcs as a table with a long-summary footer.

    Each row shows the session date, turn count, themes, mood arc,
    crisis level if present, and a truncated summary. Full summaries
    for long arcs are rendered below the table.

    Args:
        records: Episodic memory records as ``(key, value)`` pairs.

    Returns:
        None.
    """

    table = Table(
        show_header=True,
        header_style="muted",
        box=box.SIMPLE,
        show_lines=False,
        expand=True,
    )
    table.add_column("#", style="muted", no_wrap=True, width=3)
    table.add_column("date", style="muted", no_wrap=True, width=10)
    table.add_column("turns", style="muted", no_wrap=True, width=5, justify="right")
    table.add_column("themes", style="accent", no_wrap=False, max_width=20)
    table.add_column("mood", style="info", no_wrap=False, max_width=22)
    table.add_column("summary", style="info", ratio=1)

    summary_inline_limit = 80
    long_summaries: list[tuple[int, str]] = []

    for idx, (_key, value) in enumerate(records, start=1):
        ended_at = str(value.get("ended_at", ""))
        date_display = ended_at[:10] if len(ended_at) >= 10 else "—"

        turn_count = str(value.get("turn_count", "?"))

        themes_value = value.get("primary_themes")
        themes_display = "—"
        if isinstance(themes_value, list) and themes_value:
            themes_display = ", ".join(str(t) for t in themes_value)

        mood_arc = value.get("mood_arc") or {}
        if isinstance(mood_arc, dict):
            opened = str(mood_arc.get("opened", "?"))
            closed = str(mood_arc.get("closed", "?"))
            mood_display = f"{opened} → {closed}"
        else:
            mood_display = "—"

        summary = str(value.get("summary", ""))
        if len(summary) > summary_inline_limit:
            summary_display = summary[:summary_inline_limit].rstrip() + "…"
            long_summaries.append((idx, summary))
        else:
            summary_display = summary

        # Crisis marker appended to the themes column when non-zero.
        crisis_level = value.get("crisis_level_max", 0)
        if isinstance(crisis_level, int) and crisis_level > 0:
            themes_display = f"{themes_display} [warning]⚠{crisis_level}[/warning]"

        table.add_row(
            str(idx),
            date_display,
            turn_count,
            themes_display,
            mood_display,
            summary_display,
        )

    console.print(
        Panel(
            table,
            title=f"[muted]memory · episodic[/muted] "
            f"[hint]— {len(records)} session arc(s)[/hint]",
            subtitle="[hint]what the summarizer has written per session[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )

    if long_summaries:
        console.print()
        console.print("[muted]Full summaries (truncated in the table above):[/muted]")
        for idx, summary in long_summaries:
            console.print(f"  [accent]#{idx}[/accent] [info]{summary}[/info]")
    console.print()


def _render_procedural_rules_table(
    rules: list[dict[str, object]],
) -> None:
    """Render the procedural rule list as a browsable table.

    Unlike semantic facts (one record per fact) and episodic arcs
    (one record per session), procedural
    memory is stored as a single profile document per user with a
    ``rules`` list. This renderer takes that list (serialized as
    dicts from the store's JSON value column) and shows each rule
    as a numbered row.

    Columns:
        #             — 1-indexed position (use with /memory forget rule <n>)
        rule          — the rule text (second-person, evidence-grounded)
        evidence      — the user quote(s) that triggered the rule,
                        truncated inline with a long-evidence footer
                        for entries that overflow
        added         — YYYY-MM-DD date parsed from the added_at field
        conf          — confidence level

    Rules are shown in insertion order (the order the writer node
    wrote them), matching how semantic facts and episodic arcs are
    rendered.

    Args:
        rules: Serialized procedural rule dictionaries.

    Returns:
        None.
    """

    table = Table(
        show_header=True,
        header_style="muted",
        box=box.SIMPLE,
        show_lines=False,
        expand=True,
    )
    table.add_column("#", style="muted", no_wrap=True, width=3)
    table.add_column("rule", style="info", ratio=2)
    table.add_column("evidence", style="accent", ratio=1)
    table.add_column("added", style="muted", no_wrap=True, width=10)
    table.add_column("conf", style="muted", no_wrap=True, width=6)

    evidence_inline_limit = 60
    long_evidence: list[tuple[int, list[str]]] = []

    for idx, rule_value in enumerate(rules, start=1):
        rule_text = str(rule_value.get("rule", ""))
        confidence = str(rule_value.get("confidence", "?"))

        # Parse the added_at ISO timestamp to a YYYY-MM-DD prefix,
        # same convention as the episodic table.
        added_at = str(rule_value.get("added_at", ""))
        added_display = added_at[:10] if len(added_at) >= 10 else "—"

        # Evidence is a list[str]; join with " | " for display and
        # truncate if it overflows. Keep the full value for the
        # long-evidence footer.
        evidence_list = rule_value.get("evidence") or []
        if isinstance(evidence_list, list):
            evidence_strs = [str(e) for e in evidence_list]
        else:
            evidence_strs = []
        evidence_joined = " | ".join(evidence_strs) if evidence_strs else "—"
        if len(evidence_joined) > evidence_inline_limit:
            evidence_display = evidence_joined[:evidence_inline_limit].rstrip() + "…"
            long_evidence.append((idx, evidence_strs))
        else:
            evidence_display = evidence_joined

        table.add_row(
            str(idx),
            rule_text,
            evidence_display,
            added_display,
            confidence,
        )

    console.print(
        Panel(
            table,
            title=(
                f"[primary]Memory List (procedural)[/primary] "
                f"[muted]— {len(rules)} rule(s)[/muted]"
            ),
            subtitle=(
                "[muted]style rules the writer has recorded from your "
                "explicit requests[/muted]"
            ),
            border_style="panel",
            box=box.ROUNDED,
        )
    )

    if long_evidence:
        console.print()
        console.print(
            "[muted]Full evidence quotes (truncated in the table above):[/muted]"
        )
        for idx, evidence_strs in long_evidence:
            console.print(f"  [accent]#{idx}[/accent]")
            for quote in evidence_strs:
                console.print(f"    [info]{quote}[/info]")
    console.print()


def _render_procedural_rules_empty_state() -> None:
    """Empty-state panel for ``/memory list rules``.

    Shown when the active thread has no procedural rules stored yet.
    Educational: explains what rules are and how to get one written.

    Returns:
        None.
    """

    console.print(
        Panel(
            "[muted]No procedural rules for this thread yet.\n\n"
            "[accent]Procedural rules[/accent] [muted]are style preferences the "
            "agent has learned from your explicit requests — things like "
            '[accent]"please keep responses shorter"[/accent][muted] or '
            '[accent]"don\'t suggest meditation again"[/accent][muted]. They '
            "shape how the agent responds across every turn, silently.\n\n"
            "To get a rule written, tell the agent directly what you want "
            "it to do differently. The writer is conservative — small talk "
            "and passing comments don't produce rules. Only explicit "
            "directives do.[/muted]",
            title="[muted]memory · procedural[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def _render_memory_list_empty_state() -> None:
    """Render the educational empty-state panel for `/memory list`.

    Shown when both semantic and episodic namespaces are empty. The
    message explains what each layer captures and when to expect
    writes, so operators know this is expected behavior rather than a
    bug.

    Returns:
        None.
    """

    console.print(
        Panel(
            "[muted]No memory records in the store yet.\n\n"
            "[accent]Semantic facts[/accent] [muted]are written when you state a "
            "concrete persistent fact (a named person, a coping strategy, a "
            "stated goal). Transient feelings, questions, and small talk are "
            "skipped by design.\n\n"
            "[accent]Episodic arcs[/accent] [muted]are written when you end a "
            "session with [accent]/end[/accent] or confirm at [accent]/exit[/accent]. "
            "One arc per completed session.\n\n"
            "Try mentioning a named person, a coping strategy, or a recurring "
            "trigger to see the semantic store populate. Type [accent]/end[/accent] "
            "after a substantive conversation to see the episodic store populate.[/muted]",
            title="[muted]memory list[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


async def render_memory_list(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Render the active owner's memory records in browsable tables.

    Scope:
    - Read-only. Mutation commands (``/memory forget``, ``/memory clear``)
      are handled by separate commands.
    - The listing is owner-scoped, matching the active session's
      ``owner_id()``. This keeps the tables aligned with ``/memory
      status``, the web/API views, and the delete-by-index commands.
    - Semantic + episodic namespaces. Procedural rules are excluded
      here (they have their own command: ``/memory list rules``)
      because rules have a different storage shape — a single profile
      document per user rather than one record per rule — and mixing
      them into this renderer would require a third table inside the
      same panel with awkward shape mismatches.
    - Tables are rendered separately, not interleaved. Each has its own
      panel and its own long-content footer. Empty namespaces are
      suppressed (no empty table) unless BOTH are empty, in which case
      a single educational empty-state panel is shown.
    - Sorted by insertion order (the order records were written).

    Args:
        runtime: Active persistent runtime. Reads the memory_store via
            its public property.
        session: Active CLI session, for the effective owner id.

    Returns:
        None.
    """

    owner_id = session.owner_id()
    semantic_records = await _collect_records_by_kind(
        runtime, kind="semantic", owner_id=owner_id
    )
    episodic_records = await _collect_records_by_kind(
        runtime, kind="episodic", owner_id=owner_id
    )

    if not semantic_records and not episodic_records:
        _render_memory_list_empty_state()
        return

    if semantic_records:
        _render_semantic_records_table(semantic_records)

    if episodic_records:
        _render_episodic_records_table(episodic_records)

    console.print()


async def render_memory_recall_toggle(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
    *,
    enable: bool,
) -> None:
    """Handle the ``/memory recall on|off`` command.

    Toggles the ``proactive_recall_enabled`` flag on the active
    owner's procedural profile and shows a confirmation message.
    When flipping from off to on, it also renders a short explanation
    so the user understands what changes.

    Behavior:

    - ``enable=True`` + current OFF → write + show explanation
    - ``enable=True`` + current ON  → show "already on" message,
      no write
    - ``enable=False`` + current ON → write + brief confirmation
    - ``enable=False`` + current OFF → show "already off" message,
      no write

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the owner id.
        enable: Target state (True = on, False = off).

    Returns:
        None.
    """

    store = runtime.memory_store
    # Read current state first so we can detect no-op calls and
    # pick the right confirmation message.
    current = await aget_procedural_profile(store, user_id=session.owner_id())
    current_enabled = current.proactive_recall_enabled

    if enable and current_enabled:
        render_info(
            "Proactive recall is already on for this thread.",
            style="warning",
        )
        return

    if not enable and not current_enabled:
        render_info(
            "Proactive recall is already off for this thread.",
            style="warning",
        )
        return

    # Actual state change. Write first, then confirm.
    await aset_proactive_recall(store, user_id=session.owner_id(), enabled=enable)

    if enable:
        # Show the explanation on every off-to-on transition as a refresher.
        console.print(
            Panel(
                "[success]Proactive recall is now ON for this thread.[/success]\n\n"
                "[info]I'll start bringing up things from our past conversations "
                "when they seem relevant. For example, if we talked about your "
                "work schedule yesterday, I might say "
                '[accent]"last time you mentioned the morning standups have '
                'been rough — how was today?"[/accent][info]\n\n'
                "Style rules you've asked for are always applied silently "
                "regardless of this toggle — that part doesn't change.\n\n"
                "Type [accent]/memory recall off[/accent][info] any time to "
                "switch this back.[/info]",
                title="[muted]proactive recall: on[/muted]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    # Off needs only a brief confirmation.
    console.print(
        Panel(
            "[success]Proactive recall is now OFF for this thread.[/success]\n\n"
            "[info]I still remember what you tell me and use it to inform how "
            "I respond, but I won't proactively reference past sessions or "
            "earlier statements unless you ask me about them. Style rules "
            "you've asked for are still applied — the toggle only affects "
            "whether I mention past memory out loud.[/info]",
            title="[muted]proactive recall: off[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


async def render_memory_forget_rule(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
    *,
    index_str: str,
) -> None:
    """Handle the ``/memory forget rule <n>`` command.

    Deletes one procedural rule from the active owner's profile by its
    1-indexed position. Prompts for confirmation before deleting. The
    store write is atomic at the profile-document level.

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the thread_id.
        index_str: The raw argument the user typed after
            ``/memory forget rule``. Parsed to an int here; invalid
            or out-of-range inputs produce a warning without
            touching the store.

    Returns:
        None.
    """

    # Parse the index. Accept 1-indexed input (matching the display
    # convention of the other list commands) and convert to
    # 0-indexed for the list slice below.
    try:
        index_1based = int(index_str)
    except ValueError:
        render_info(
            f"Usage: /memory forget rule <n>  (got: {index_str!r})",
            style="warning",
        )
        return

    if index_1based < 1:
        render_info(
            f"Rule index must be 1 or greater (got: {index_1based}).",
            style="warning",
        )
        return

    store = runtime.memory_store
    profile = await aget_procedural_profile(store, user_id=session.owner_id())

    if not profile.rules:
        render_info(
            "No procedural rules to forget for this thread.",
            style="warning",
        )
        return

    if index_1based > len(profile.rules):
        render_info(
            f"Rule #{index_1based} does not exist "
            f"(only {len(profile.rules)} rule(s) for this thread).",
            style="warning",
        )
        return

    # Rules are short enough to inline in the confirmation prompt.
    target_rule = profile.rules[index_1based - 1]
    console.print()
    console.print(
        Panel(
            f"[info]{target_rule.rule}[/info]",
            title=(f"[warning]Delete rule #{index_1based}?[/warning]"),
            border_style="warning",
            box=box.ROUNDED,
        )
    )
    answer = Prompt.ask(
        "[muted]Delete this rule?[/muted] [accent][y/N][/accent]",
        choices=["y", "Y", "n", "N", ""],
        default="n",
        show_choices=False,
        show_default=False,
    )
    if answer.strip().lower() != "y":
        render_info("Cancelled — no rules deleted.", style="info")
        return

    # Procedural memory is stored as one profile document, not one record per rule.
    deleted = await adelete_procedural_rule(
        store,
        user_id=session.owner_id(),
        rule_id=target_rule.id,
    )
    if deleted is None:
        render_info(
            "The selected rule changed before deletion could be applied.",
            style="warning",
        )
        return
    profile, _removed_rule = deleted
    render_info(
        f"Deleted rule #{index_1based}. "
        f"{len(profile.rules)} rule(s) remaining for this thread.",
        style="success",
    )


# Privacy controls: /memory forget fact|session and /memory clear


def _parse_one_based_index(
    index_str: str,
    *,
    kind_label: str,
) -> int | None:
    """Parse a 1-indexed CLI argument into an int, or render a warning.

    Shared across forget handlers (``fact``, ``session``) so
    they produce identical error messages for the same failure modes
    (non-integer argument, zero, negative). Returns ``None`` when the
    parse fails or the value is out of the 1-indexed range; the caller
    should abort without touching the store in that case. Returns the
    parsed integer on success.

    Why a separate helper and not a one-liner: the warning messages
    are slightly different per failure mode (wrong type vs. zero vs.
    negative), and keeping all three paths in one place makes it
    easy to keep the phrasing consistent across fact / session /
    rule handlers. The rule handler at
    :func:`render_memory_forget_rule` inlines the same logic to avoid
    changing its mature confirmation flow.

    Args:
        index_str: Raw index argument from the command.
        kind_label: User-facing memory kind label.

    Returns:
        Parsed one-based index, or None after rendering a warning.
    """

    try:
        index_1based = int(index_str)
    except ValueError:
        render_info(
            f"Usage: /memory forget {kind_label} <n>  (got: {index_str!r})",
            style="warning",
        )
        return None

    if index_1based < 1:
        render_info(
            f"{kind_label.capitalize()} index must be 1 or greater "
            f"(got: {index_1based}).",
            style="warning",
        )
        return None

    return index_1based


def _render_forget_confirmation(
    *,
    kind_label: str,
    index_1based: int,
    preview_lines: list[str],
) -> bool:
    """Render a y/N confirmation panel for a single-record forget command.

    Shared helper for forget handlers (fact, session) that
    need to show a preview of the target before the user confirms.
    The panel mirrors rule-forget confirmation so the UX stays
    consistent across kinds.

    Args:
        kind_label: The word shown in the panel title, e.g., ``"fact"``
            or ``"session"``. Used as ``f"Delete {kind_label} #N?"``.
        index_1based: The 1-indexed position, shown in the title.
        preview_lines: A list of ``[label] value`` lines describing the
            target record. Lines are joined with newlines inside the
            panel body. Keep each line short enough to fit on one
            terminal row without wrapping.

    Returns:
        ``True`` if the user confirmed (typed ``y``), ``False``
        otherwise (including Enter-for-default-N, ``n``, or any
        non-``y`` input). The caller should treat ``False`` as "abort
        without touching the store."
    """

    body = "\n".join(f"[info]{line}[/info]" for line in preview_lines)
    console.print()
    console.print(
        Panel(
            body,
            title=f"[warning]Delete {kind_label} #{index_1based}?[/warning]",
            border_style="warning",
            box=box.ROUNDED,
        )
    )
    answer = Prompt.ask(
        f"[muted]Delete this {kind_label}?[/muted] [accent][y/N][/accent]",
        choices=["y", "Y", "n", "N", ""],
        default="n",
        show_choices=False,
        show_default=False,
    )
    return answer.strip().lower() == "y"


async def render_memory_forget_fact(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
    *,
    index_str: str,
) -> None:
    """Handle the ``/memory forget fact <n>`` command.

    Deletes one semantic fact from the active owner's semantic namespace by its
    1-indexed position in
    ``/memory list`` (or ``/memory list facts``). The index must
    match the number shown in the ``#`` column of the semantic
    records table — if you change the sort order in
    ``_collect_records_with_namespace``, update
    ``_render_semantic_records_table`` to match or the indexes
    will drift.

    UX mirrors the rule-forget flow (:func:`render_memory_forget_rule`):

    1. Parse the index. Invalid / zero / negative values render a
       warning and abort without any store interaction.
    2. Fetch the target namespace via
       :func:`_collect_records_with_namespace` using the session's
       owner_id. Out-of-range indexes render a warning that includes
       the current fact count so the user knows what went wrong.
    3. Show a preview panel (category, predicate, object, evidence
       quote) + y/N confirmation. Default is N.
    4. On confirm, call ``store.adelete(namespace, key)``. A single
       record delete, not a profile round-trip, because semantic
       memory is stored one-record-per-fact (unlike procedural).
    5. Render a success message with the remaining fact count.

    Owner scope: operations are always scoped to ``session.owner_id()``,
    which is the ``--user-id`` flag if set or the thread_id fallback.
    Cross-user deletion is not reachable through this command.

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the owner id.
        index_str: The raw argument the user typed after
            ``/memory forget fact``. Parsed to an int here.

    Returns:
        None.
    """

    index_1based = _parse_one_based_index(index_str, kind_label="fact")
    if index_1based is None:
        return

    facts = await _collect_records_with_namespace(
        runtime, kind="semantic", owner_id=session.owner_id()
    )

    if not facts:
        render_info(
            "No semantic facts to forget for this thread.",
            style="warning",
        )
        return

    if index_1based > len(facts):
        render_info(
            f"Fact #{index_1based} does not exist "
            f"(only {len(facts)} fact(s) for this thread).",
            style="warning",
        )
        return

    namespace, key, value = facts[index_1based - 1]

    # Preview mirrors the semantic table and truncates evidence for compactness.
    category = str(value.get("category", "?"))
    predicate = str(value.get("predicate", "?"))
    object_id = _format_entity_identifier(value.get("object"))
    quote = str(value.get("evidence_quote", ""))
    if len(quote) > 120:
        quote = quote[:117].rstrip() + "…"
    preview = [
        f"category:  {category}",
        f"predicate: {predicate}",
        f"object:    {object_id}",
        f"evidence:  {quote}",
    ]

    if not _render_forget_confirmation(
        kind_label="fact",
        index_1based=index_1based,
        preview_lines=preview,
    ):
        render_info("Cancelled — no facts deleted.", style="info")
        return

    deleted = await runtime.memory_store.adelete(namespace, key)
    if not deleted:
        # Race condition: the record was deleted between the fetch
        # and the delete call. Unlikely in the single-user CLI but
        # possible in a future multi-process scenario. Report
        # honestly rather than pretending the delete happened.
        render_info(
            f"Fact #{index_1based} was already gone before the delete "
            "landed (possibly deleted in another session).",
            style="warning",
        )
        return

    render_info(
        f"Deleted fact #{index_1based}. "
        f"{len(facts) - 1} fact(s) remaining for this thread.",
        style="success",
    )


async def render_memory_forget_session(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
    *,
    index_str: str,
) -> None:
    """Handle the ``/memory forget session <n>`` command.

    Deletes one episodic session arc from the active owner's episodic namespace by its
    1-indexed position in
    ``/memory list`` (or ``/memory list sessions``). Parallels
    :func:`render_memory_forget_fact` — same index contract, same
    confirmation pattern, same single-record adelete path.

    Preview shows the arc's summary (truncated) and themes so the
    user knows which session they're about to delete. Date is not
    shown in the preview because the summary field already includes
    temporal context.

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the owner id.
        index_str: The raw argument the user typed after
            ``/memory forget session``. Parsed to an int here.

    Returns:
        None.
    """

    index_1based = _parse_one_based_index(index_str, kind_label="session")
    if index_1based is None:
        return

    sessions = await _collect_records_with_namespace(
        runtime, kind="episodic", owner_id=session.owner_id()
    )

    if not sessions:
        render_info(
            "No episodic sessions to forget for this thread.",
            style="warning",
        )
        return

    if index_1based > len(sessions):
        render_info(
            f"Session #{index_1based} does not exist "
            f"(only {len(sessions)} session arc(s) for this thread).",
            style="warning",
        )
        return

    namespace, key, value = sessions[index_1based - 1]

    summary = str(value.get("summary", ""))
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "…"
    themes_value = value.get("primary_themes")
    themes_display = "—"
    if isinstance(themes_value, list) and themes_value:
        themes_display = ", ".join(str(t) for t in themes_value)
    ended_at = str(value.get("ended_at", ""))
    date_display = ended_at[:10] if len(ended_at) >= 10 else "—"
    preview = [
        f"date:    {date_display}",
        f"themes:  {themes_display}",
        f"summary: {summary}",
    ]

    if not _render_forget_confirmation(
        kind_label="session",
        index_1based=index_1based,
        preview_lines=preview,
    ):
        render_info("Cancelled — no session arcs deleted.", style="info")
        return

    deleted = await runtime.memory_store.adelete(namespace, key)
    if not deleted:
        render_info(
            f"Session #{index_1based} was already gone before the delete "
            "landed (possibly deleted in another session).",
            style="warning",
        )
        return

    render_info(
        f"Deleted session #{index_1based}. "
        f"{len(sessions) - 1} session arc(s) remaining for this thread.",
        style="success",
    )


async def render_memory_clear(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
    *,
    kind: str,
) -> None:
    """Handle the ``/memory clear <kind>`` command.

    Deletes every record in a namespace, or all namespaces, for the
    active owner. Unrecoverable.

    Because this is destructive at scale, the confirmation is
    **stronger** than the single-record forget commands: instead of
    y/N, the user must type the literal word ``clear`` to proceed.
    This prevents muscle-memory confirmations from nuking a memory
    store the user did not mean to lose. Any other input, including
    ``y``, ``yes``, ``CLEAR``, or an empty line, cancels.

    Supported kinds:

    - ``facts`` clears the semantic namespace
    - ``sessions`` clears the episodic namespace
    - ``rules`` clears the procedural profile (rules only;
                     ``proactive_recall_enabled`` is preserved
                     because it's a user preference, not content)
    - ``all`` clears all three of the above in one operation

    Implementation notes:

    Semantic and episodic memory use per-record deletes because the
    store protocol does not expose a namespace-clear primitive.
    Procedural memory uses a profile round-trip because it is stored as
    one profile document per owner. The ``all`` path is not atomic
    across kinds.

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the owner id.
        kind: One of ``"facts"``, ``"sessions"``, ``"rules"``,
            ``"all"``. Unknown kinds render a usage warning.

    Returns:
        None.
    """

    valid_kinds = {"facts", "sessions", "rules", "all"}
    if kind not in valid_kinds:
        render_info(
            "Usage: /memory clear <facts|sessions|rules|all>",
            style="warning",
        )
        return

    owner_id = session.owner_id()
    store = runtime.memory_store

    # Show concrete counts before destructive confirmation.
    counts: dict[str, int] = {"facts": 0, "sessions": 0, "rules": 0}
    if kind in ("facts", "all"):
        counts["facts"] = await store.arecord_count((owner_id, "semantic"))
    if kind in ("sessions", "all"):
        counts["sessions"] = await store.arecord_count((owner_id, "episodic"))
    if kind in ("rules", "all"):
        profile = await aget_procedural_profile(store, user_id=owner_id)
        counts["rules"] = len(profile.rules)

    # Skip scary confirmation for no-op clears.
    if sum(counts.values()) == 0:
        render_info(
            f"Nothing to clear for {kind}. Store is already empty "
            f"for this {'user' if session.user_id else 'thread'}.",
            style="info",
        )
        return

    # Only show counts for kinds being touched.
    affected_lines = []
    if kind in ("facts", "all") and counts["facts"] > 0:
        affected_lines.append(f"semantic facts:    {counts['facts']}")
    if kind in ("sessions", "all") and counts["sessions"] > 0:
        affected_lines.append(f"episodic sessions: {counts['sessions']}")
    if kind in ("rules", "all") and counts["rules"] > 0:
        affected_lines.append(f"procedural rules:  {counts['rules']}")

    body = "\n".join(f"[warning]{line}[/warning]" for line in affected_lines)
    body += (
        "\n\n[danger]This cannot be undone.[/danger]\n"
        "[muted]Type [accent]clear[/accent] to proceed, "
        "or anything else to cancel.[/muted]"
    )

    console.print()
    console.print(
        Panel(
            body,
            title=(
                f"[danger]Clear memory ({kind})[/danger] "
                f"[muted]— owner: {owner_id}[/muted]"
            ),
            border_style="danger",
            box=box.ROUNDED,
        )
    )
    answer = Prompt.ask(
        "[muted]Type the word to confirm[/muted]",
        default="",
        show_default=False,
    )
    if answer.strip() != "clear":
        render_info(
            "Cancelled — no memory cleared.",
            style="info",
        )
        return

    # Confirmed. Sweep each kind in the requested scope.
    deleted_counts: dict[str, int] = {"facts": 0, "sessions": 0, "rules": 0}

    if kind in ("facts", "all"):
        records = await _collect_records_with_namespace(
            runtime, kind="semantic", owner_id=owner_id
        )
        for namespace, key, _value in records:
            if await store.adelete(namespace, key):
                deleted_counts["facts"] += 1

    if kind in ("sessions", "all"):
        records = await _collect_records_with_namespace(
            runtime, kind="episodic", owner_id=owner_id
        )
        for namespace, key, _value in records:
            if await store.adelete(namespace, key):
                deleted_counts["sessions"] += 1

    if kind in ("rules", "all"):
        # Procedural is a profile-document, not per-record. Reset
        # the rules list but preserve the recall toggle (it's a
        # user preference, not content) and re-put the profile.
        _profile, deleted_counts["rules"] = await aclear_procedural_rules(
            store,
            user_id=owner_id,
        )

    # Render a success summary listing every kind that was touched.
    # Zero-count lines are suppressed so the panel is compact when
    # a kind was already empty.
    summary_lines = [
        f"{label}: {deleted_counts[label]}"
        for label in ("facts", "sessions", "rules")
        if deleted_counts[label] > 0
    ]
    if not summary_lines:
        # This can happen on a clear-between-fetch-and-commit race;
        # the counts were non-zero when we showed the warning but
        # are zero now. Rare but worth reporting honestly.
        render_info(
            "Clear completed, but no records were found to delete "
            "(they may have been removed between the confirmation "
            "and the sweep).",
            style="warning",
        )
        return

    render_info(
        "Cleared: " + ", ".join(summary_lines),
        style="success",
    )


# Default cutoff window (in days) for ``/memory purge-crisis``. 90 days
# matches the documented default retention policy and legal-review caveat
# on the always-on crisis log. Operators can override per-invocation
# (e.g., ``/memory purge-crisis 30`` for a tighter sweep) but the
# default should match the documented policy.
DEFAULT_CRISIS_RETENTION_DAYS = 90


async def render_memory_purge_crisis(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,  # noqa: ARG001 - kept for handler symmetry
    *,
    days: int,
) -> None:
    """Handle the ``/memory purge-crisis [days]`` command.

    Deletes crisis log records older than ``days`` days from the active
    runtime's crisis log backend.
    Calls :meth:`CrisisLogBackend.apurge_before` with ``today - days``
    as the exclusive cutoff, so records on the cutoff date itself are
    preserved (the semantics match the backend's docstring).

    Unlike ``/memory forget`` or ``/memory clear``, this command
    operates on the **crisis log**, which is always-on regardless of
    memory mode; even incognito sessions have an in-memory crisis
    log that the gate writes to. The purge affects whichever backend
    is currently wired, so operators can run this against an
    incognito session's in-memory log too (though it's less useful
    because the in-memory log dies at CLI exit anyway).

    Confirmation pattern: same typed ``purge`` gate as ``/memory clear``
    because the user must type the literal word ``purge`` to proceed, not
    ``y`` or ``purge-crisis`` or ``PURGE``. This is consistent with
    the destructive-command pattern and prevents muscle-memory mistakes
    from wiping the audit trail.

    Args:
        runtime: Active persistent runtime. Reads the crisis log
            backend via ``runtime.crisis_log_backend``.
        session: Active CLI session. Not currently read; included
            for signature symmetry with other destructive handlers
            and because a future enhancement might scope the purge
            to the session's owner_id (currently the crisis log is
            not owner-scoped, matching the privacy design).
        days: Retention window in days. Records with detected_date
            older than ``today - days`` are deleted. Must be >= 1;
            zero or negative values produce a warning without
            touching the log.

    Returns:
        None.
    """

    if days < 1:
        render_info(
            f"Retention window must be at least 1 day (got {days}).",
            style="warning",
        )
        return

    crisis_log = runtime.crisis_log_backend
    total_before = await crisis_log.arecord_count()

    # Compute the cutoff as (today - days) in the runtime's timezone.
    # We use UTC to match the crisis log records' ``detected_at``
    # strings which are always stored with a Z suffix (UTC). Using
    # the local timezone would create subtle boundary bugs when the
    # operator is in a non-UTC zone and runs the purge near midnight.
    today_utc = datetime.now(UTC).date()
    cutoff = today_utc - timedelta(days=days)

    if total_before == 0:
        render_info(
            "Crisis log is empty — nothing to purge.",
            style="info",
        )
        return

    # Show scan size before confirmation; post-purge reports actual deletes.
    body = (
        f"[warning]Retention window:[/warning] {days} day(s)\n"
        f"[warning]Cutoff date:[/warning]       {cutoff.isoformat()} "
        f"(records BEFORE this date will be deleted)\n"
        f"[warning]Crisis log size:[/warning]   {total_before} record(s) total\n\n"
        f"[danger]This cannot be undone.[/danger]\n"
        f"[muted]Type [accent]purge[/accent] to proceed, or anything else "
        f"to cancel.[/muted]"
    )

    console.print()
    console.print(
        Panel(
            body,
            title=(
                f"[danger]Purge crisis log[/danger] [muted]— retention {days}d[/muted]"
            ),
            border_style="danger",
            box=box.ROUNDED,
        )
    )
    answer = Prompt.ask(
        "[muted]Type the word to confirm[/muted]",
        default="",
        show_default=False,
    )
    if answer.strip() != "purge":
        render_info(
            "Cancelled — no crisis records purged.",
            style="info",
        )
        return

    # Confirmed. Run the purge and report the result.
    try:
        deleted = await crisis_log.apurge_before(cutoff)
    except Exception as exc:
        render_info(
            f"Purge failed: {exc}",
            style="danger",
        )
        return

    remaining = total_before - deleted
    render_info(
        f"Purged {deleted} crisis record(s) older than {cutoff.isoformat()}. "
        f"{remaining} record(s) remaining in the log.",
        style="success",
    )


async def render_memory_list_rules(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Render the active thread's procedural rules in a browsable table.

    Unlike ``render_memory_list``, this reads procedural rules from the
    active owner's profile because procedural memory is stored as one
    profile document per owner.

    Shows each rule with its index (usable with
    ``/memory forget rule <n>``), the rule text, the evidence quote
    that triggered it, the date it was added, and the confidence
    level. Falls back to an educational empty-state panel when the
    profile has no rules yet.

    Args:
        runtime: Active persistent runtime. Reads the memory_store
            via its public property.
        session: Active CLI session. Used to identify the current
            thread for the per-user profile lookup.

    Returns:
        None.
    """

    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.owner_id()
    )

    if not profile.rules:
        _render_procedural_rules_empty_state()
        return

    # Serialize the pydantic ProceduralRule instances back to dicts
    # so the renderer can use the same dict-based access pattern as
    # the semantic/episodic renderers. The round-trip is cheap and
    # keeps the renderer decoupled from the pydantic schema.
    rule_dicts = [rule.model_dump(mode="json") for rule in profile.rules]
    _render_procedural_rules_table(rule_dicts)
    console.print()


def render_session_summary(stored_arc: StoredSessionArc) -> None:
    """Render a session summary panel after the summarizer writes an arc.

    Shows the user the exact summary that was saved so they know what
    will be remembered and can correct it later.

    The summary display uses the structured fields from StoredSessionArc
    rather than just the prose summary, because the structure IS the
    signal: mood arc, themes, open loops, and crisis level all say
    something about what the session was about beyond what the summary
    paragraph captures.

    Args:
        stored_arc: Persisted session arc produced by the summarizer.

    Returns:
        None.
    """

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")

    # Summary text is the primary confirmation signal for the user.
    table.add_row("summary", stored_arc.summary)
    table.add_row(
        "themes",
        ", ".join(stored_arc.primary_themes) if stored_arc.primary_themes else "—",
    )
    table.add_row(
        "mood arc",
        f"{stored_arc.mood_arc.opened} → {stored_arc.mood_arc.closed}",
    )
    table.add_row("turns", str(stored_arc.turn_count))
    if stored_arc.duration_seconds > 0:
        minutes = stored_arc.duration_seconds // 60
        table.add_row("duration", f"~{minutes} minute(s)")
    if stored_arc.crisis_level_max > 0:
        table.add_row(
            "crisis signal",
            f"[warning]level {stored_arc.crisis_level_max}[/warning]",
        )
    if stored_arc.open_loops:
        table.add_row(
            "open loops",
            "\n".join(f"• {loop}" for loop in stored_arc.open_loops),
        )
    if stored_arc.resolved_threads:
        table.add_row(
            "resolved",
            "\n".join(f"• {item}" for item in stored_arc.resolved_threads),
        )

    console.print(
        Panel(
            table,
            title="[muted]session summary · saved to memory[/muted]",
            subtitle="[hint]this is what I'll remember next time we talk[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_help() -> None:
    """Render the available slash commands.

    Returns:
        None.
    """

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("command", style="accent", no_wrap=True)
    table.add_column("description", style="info")
    table.add_row("/help", "Show available commands.")
    table.add_row("/status", "Show current mode and session stats.")
    table.add_row("/history [n]", "Show the last n transcript messages. Default: 6.")
    table.add_row("/context", "Show the latest derived session context snapshot.")
    table.add_row(
        "/memory status",
        "Show memory layer state (counts, mode, crisis log, recall toggle).",
    )
    table.add_row(
        "/memory list [facts|sessions|rules]",
        "List semantic facts, episodic arcs, and/or procedural rules. "
        "Without a subcommand, shows semantic + episodic together.",
    )
    table.add_row(
        "/memory recall on|off",
        "Toggle whether the agent proactively references past memory "
        "content in replies. Style rules are always applied regardless.",
    )
    table.add_row(
        "/memory forget <fact|session|rule> <n>",
        "Delete one record by its 1-indexed position from /memory list. "
        "Shows a preview panel and asks for y/N confirmation.",
    )
    table.add_row(
        "/memory clear <facts|sessions|rules|all>",
        "Wipe an entire namespace for the active user. Unrecoverable. "
        "Requires typing the word 'clear' to confirm (stronger than y/N).",
    )
    table.add_row(
        "/memory purge-crisis [days]",
        "Delete crisis log records older than the retention window. "
        "Default 90 days. Requires typing 'purge' to confirm.",
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
    table.add_row(
        "/response-tier <fast|quality>",
        "Switch the therapeutic response quality/latency tradeoff.",
    )
    table.add_row(
        "/debug state",
        "Dump the raw graph state for the active thread (verbose diagnostics).",
    )
    table.add_row(
        "/end",
        "End the session; summarize it and save the arc to episodic memory.",
    )
    table.add_row(
        "/exit",
        "End the session; prompt to save a summary before closing.",
    )
    console.print(
        Panel(
            table,
            title="[muted]commands[/muted]",
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
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")
    table.add_row("thread id", session.thread_id)
    table.add_row("sqlite path", session.sqlite_path)
    table.add_row("requested mode", session.requested_mode)
    table.add_row("resolved mode", session.resolved_mode)
    table.add_row(
        "llm client", "enabled" if session.llm_client is not None else "disabled"
    )
    table.add_row("response tier", session.response_model_tier)
    table.add_row(
        "response llm",
        "enabled" if session.response_llm_client is not None else "disabled",
    )
    table.add_row("turns", str(turn_count))
    table.add_row("messages", str(len(session.history)))
    table.add_row(
        "context snapshot", "available" if session.last_context is not None else "none"
    )
    console.print(
        Panel(
            table,
            title="[muted]session status[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_history(session: RunnerSession, limit: int = 6) -> None:
    """Render the most recent transcript entries.

    Assistant turns show the response style recorded by the finalize
    node. Missing style metadata falls back to ``-`` for older or
    minimal checkpoints.

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
                title="[muted]history[/muted]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    recent = session.history[-max(1, limit) :]
    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("role", style="accent", no_wrap=True)
    table.add_column("style", style="muted", no_wrap=True)
    table.add_column("content", style="info")
    for message in recent:
        role = message.role.value
        style_display = message.response_style if message.response_style else "-"
        table.add_row(role, style_display, message.content)
    console.print(
        Panel(
            table,
            title="[muted]recent history[/muted]",
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

    style_color = {
        "success": "success",
        "warning": "warning",
        "danger": "danger",
        "muted": "hint",
        "info": "info",
        "panel": "panel",
    }.get(style, "panel")
    console.print(f"  [{style_color}]│[/{style_color}] {message}")
    console.print()


def render_threads(
    threads: list[ThreadSummary],
    *,
    active_thread_id: str,
) -> None:
    """Render a compact table of persisted thread summaries.

    Args:
        threads: Thread summaries returned by the runtime.
        active_thread_id: Currently selected thread id.

    Returns:
        None.
    """

    if not threads:
        render_info("No persisted threads found.", style="warning")
        return

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("thread id", style="info")
    table.add_column("turns", style="hint", justify="right", no_wrap=True)
    table.add_column("messages", style="hint", justify="right", no_wrap=True)
    table.add_column("active", style="accent", no_wrap=True)
    for thread in threads:
        table.add_row(
            thread.thread_id,
            str(thread.turn_count),
            str(thread.message_count),
            "[primary]·[/primary]" if thread.thread_id == active_thread_id else "",
        )
    console.print(
        Panel(
            table,
            title="[muted]threads[/muted]",
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
    session.response_llm_client = (
        resolve_response_llm_client(mode, session.response_model_tier)
        if llm_client is not None
        else None
    )


def set_response_model_tier(
    session: RunnerSession,
    response_model_tier: ResponseModelTier,
) -> None:
    """Update the response-writer tier for subsequent turns.

    Args:
        session: Mutable CLI session state.
        response_model_tier: Requested response model tier.

    Returns:
        None.
    """

    session.response_model_tier = response_model_tier
    session.response_llm_client = (
        resolve_response_llm_client(session.requested_mode, response_model_tier)
        if session.llm_client is not None
        else None
    )


def generate_thread_id() -> str:
    """Generate a new local thread id for ad hoc CLI sessions.

    Returns:
        New local thread id.
    """

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


def _prompt_for_session_feedback() -> FeedbackLabel | None:
    """Ask the user once for a thumbs rating at end-of-session.

    The prompt is opt-in by design:

    - Explicit ``y``, ``n``, ``s`` maps to ``"positive"``, ``"negative"``,
      ``"skip"``. All three produce a labeled record so analytics can
      distinguish "user actively skipped" from "user said nothing".
    - Empty input (bare Enter) maps to ``None``. No record written.
    - ``Ctrl-C`` / ``EOF`` maps to ``None``.
      No record written.

    No default value. Defaulting to ``"skip"`` would turn accidental
    Enter keypresses into explicit-skip records and inflate the skip
    rate in analytics; defaulting to anything else would bias toward
    that label. The explicit-no-default design means Enter is a
    no-op both for the user and for the dataset.

    Returns:
        Feedback label when explicitly selected, otherwise None.
    """

    try:
        response = Prompt.ask(
            "[muted]Quick check — did today feel helpful?[/muted] "
            "[accent][y/n/s] (or Enter to skip without recording)[/accent]",
            choices=["y", "Y", "n", "N", "s", "S", ""],
            default="",
            show_choices=False,
            show_default=False,
        )
    except (KeyboardInterrupt, EOFError):
        return None

    response = response.strip().lower()
    if response == "y":
        return "positive"
    if response == "n":
        return "negative"
    if response == "s":
        return "skip"
    return None  # Empty input means no feedback record.


async def _summarize_and_render(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    *,
    source: FeedbackSource,
) -> None:
    """End-session orchestration: capture feedback, then summarize.

    Shared helper for the ``/end`` and ``/exit`` (save=y branch)
    commands. Both trigger the same three-step end flow:

    1. Best-effort feedback prompt. User can decline by hitting Enter
       or Ctrl-C; explicit ``y``/``n``/``s`` produces a labeled
       record. ``source`` distinguishes which command triggered the
       flow (``"cli_end"`` or ``"cli_exit"``).
    2. Feedback persistence via ``runtime.record_session_feedback``.
       The runtime never raises; a backend outage means no record
       is written and the flow continues.
    3. Summarization via ``runtime.end_session``. If it returns a
       stored arc we render it; if it returns ``None`` (incognito,
       no LLM, thin session) we render a plain farewell.

    ``/exit`` save=n does not route through here because that branch
    skips both the feedback prompt and session summary.

    Args:
        session: Active CLI session.
        runtime: Persistent runtime backing the active thread.
        source: Feedback source label for any captured feedback.

    Returns:
        None.
    """

    # Best-effort feedback capture. Silent input means no record.
    label = _prompt_for_session_feedback()
    if label is not None:
        await runtime.record_session_feedback(
            session.thread_id,
            label=label,
            source=source,
        )

    # Summarization degrades to a plain farewell when no arc is produced.
    try:
        stored_arc = await runtime.end_session(
            session.thread_id,
            llm_client=session.llm_client,
        )
    except Exception:
        # Keep the end flow usable even if an unexpected summary error escapes.
        render_info(
            "Something went wrong while summarizing the session. Your "
            f"conversation is still saved in thread {session.thread_id}.",
            style="warning",
        )
        stored_arc = None

    if stored_arc is not None:
        render_session_summary(stored_arc)

    render_info(
        "Take care. We can pick this back up whenever you want. "
        f"(Thread: {session.thread_id})",
        style="success",
    )


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
        # Default toward saving a summary; users can delete it later if needed.
        save = Prompt.ask(
            "[muted]Save a session summary before exiting?[/muted] [accent][Y/n][/accent]",
            choices=["y", "Y", "n", "N", ""],
            default="y",
            show_choices=False,
            show_default=False,
        )
        if save.strip().lower() != "n":
            await _summarize_and_render(session, runtime, source="cli_exit")
        # If the user declines saving, skip feedback capture as well.
        return False

    if command == "/end":
        # Explicit end command means full closing flow without save confirmation.
        await _summarize_and_render(session, runtime, source="cli_end")
        return False

    if command == "/memory":
        if len(args) == 0 or args[0] == "status":
            await render_memory_status(runtime, session)
            return True
        if args[0] == "list":
            if len(args) >= 2 and args[1] == "rules":
                await render_memory_list_rules(runtime, session)
                return True
            if len(args) >= 2 and args[1] == "facts":
                # Render just the semantic table. Reuses the existing
                # helper to avoid duplicating the table rendering
                # logic across the list / list facts paths.
                semantic_records = await _collect_records_by_kind(
                    runtime, kind="semantic", owner_id=session.owner_id()
                )
                if not semantic_records:
                    _render_memory_list_empty_state()
                    return True
                _render_semantic_records_table(semantic_records)
                console.print()
                return True
            if len(args) >= 2 and args[1] == "sessions":
                episodic_records = await _collect_records_by_kind(
                    runtime, kind="episodic", owner_id=session.owner_id()
                )
                if not episodic_records:
                    _render_memory_list_empty_state()
                    return True
                _render_episodic_records_table(episodic_records)
                console.print()
                return True
            await render_memory_list(runtime, session)
            return True
        if args[0] == "recall":
            if len(args) < 2 or args[1] not in ("on", "off"):
                render_info(
                    "Usage: /memory recall on  |  /memory recall off",
                    style="warning",
                )
                return True
            await render_memory_recall_toggle(
                runtime, session, enable=(args[1] == "on")
            )
            return True
        if args[0] == "forget":
            if len(args) >= 2 and args[1] == "rule":
                if len(args) < 3:
                    render_info(
                        "Usage: /memory forget rule <n>",
                        style="warning",
                    )
                    return True
                await render_memory_forget_rule(runtime, session, index_str=args[2])
                return True
            if len(args) >= 2 and args[1] == "fact":
                if len(args) < 3:
                    render_info(
                        "Usage: /memory forget fact <n>",
                        style="warning",
                    )
                    return True
                await render_memory_forget_fact(runtime, session, index_str=args[2])
                return True
            if len(args) >= 2 and args[1] == "session":
                if len(args) < 3:
                    render_info(
                        "Usage: /memory forget session <n>",
                        style="warning",
                    )
                    return True
                await render_memory_forget_session(runtime, session, index_str=args[2])
                return True
            render_info(
                "Usage: /memory forget <fact|session|rule> <n>",
                style="warning",
            )
            return True
        if args[0] == "clear":
            if len(args) < 2:
                render_info(
                    "Usage: /memory clear <facts|sessions|rules|all>",
                    style="warning",
                )
                return True
            await render_memory_clear(runtime, session, kind=args[1])
            return True
        if args[0] == "purge-crisis":
            days = DEFAULT_CRISIS_RETENTION_DAYS
            if len(args) >= 2:
                try:
                    days = int(args[1])
                except ValueError:
                    render_info(
                        f"Usage: /memory purge-crisis [days]  "
                        f"(got: {args[1]!r}, expected an integer)",
                        style="warning",
                    )
                    return True
            await render_memory_purge_crisis(runtime, session, days=days)
            return True
        render_info(
            "Unknown /memory subcommand. Available: "
            "status, list [facts|sessions|rules], recall on|off, "
            "forget <fact|session|rule> <n>, clear <facts|sessions|rules|all>, "
            "purge-crisis [days]",
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
        render_header(
            session.resolved_mode,
            session.thread_id,
            session.memory_mode,
            user_id=session.user_id,
        )
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
        render_header(
            session.resolved_mode,
            session.thread_id,
            session.memory_mode,
            user_id=session.user_id,
        )
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
        render_header(
            session.resolved_mode,
            session.thread_id,
            session.memory_mode,
            user_id=session.user_id,
        )
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

    if command == "/response-tier":
        if len(args) != 1 or args[0] not in {"fast", "quality"}:
            render_info("Usage: /response-tier <fast|quality>", style="warning")
            return True
        set_response_model_tier(session, args[0])  # type: ignore[arg-type]
        render_info(
            f"Response tier updated. tier={session.response_model_tier}",
            style="success" if session.response_llm_client is not None else "warning",
        )
        return True

    if command == "/debug":
        if len(args) == 0 or args[0] != "state":
            render_info(
                "Usage: /debug state  (dumps raw graph state for the active thread)",
                style="warning",
            )
            return True
        await _render_debug_state(runtime, session)
        return True

    render_info(f"Unknown command: {command}. Try /help.", style="warning")
    return True


async def chat_loop(
    mode: str,
    *,
    thread_id: str,
    user_id: str | None = None,
    response_model_tier: ResponseModelTier = "fast",
    sqlite_path: str,
    memory_mode: str,
    memory_sqlite_path: str = str(DEFAULT_MEMORY_DB_PATH),
    crisis_log_sqlite_path: str = str(DEFAULT_CRISIS_LOG_DB_PATH),
) -> None:
    """Run the interactive CLI loop.

    Args:
        mode: Requested runtime mode for model resolution.
        thread_id: Stable thread identifier for the local conversation.
        user_id: Optional stable owner identifier for long-term memory.
            When set, memory writes are namespaced by this user_id
            rather than the thread_id. See
            :meth:`RunnerSession.owner_id` for the full resolution
            rationale. When None, the CLI uses ``thread_id`` as the
            effective owner.
        response_model_tier: Response tier for user-facing prose.
        sqlite_path: Legacy SQLite file used for persisted thread checkpoints.
            Deprecated for normal local development in favor of Postgres.
        memory_mode: Local memory mode ("guest" or "persistent").
        memory_sqlite_path: Legacy SQLite path for the memory store. Only
            used in persistent mode when the SQLite backend is selected.
        crisis_log_sqlite_path: Legacy SQLite path for the crisis log.
            Same persistence semantics as the memory path.

    Returns:
        None.
    """

    settings = get_settings()
    llm_client, resolved_mode = resolve_llm_client(mode)
    response_llm_client = (
        resolve_response_llm_client(mode, response_model_tier)
        if llm_client is not None
        else None
    )
    # CLI uses the string labels "guest" and "persistent" for the user-facing
    # mode. Translate to the graph-internal MemoryMode enum for the runtime.
    runtime_memory_mode = (
        MemoryMode.INCOGNITO if memory_mode == "guest" else MemoryMode.LOCAL
    )
    is_guest_mode = runtime_memory_mode == MemoryMode.INCOGNITO
    # In guest mode, --user-id is meaningless (no long-term storage to
    # namespace). We don't reject it at the CLI layer because that would
    # require a pre-parse check, but we do drop it from the session so
    # owner_id() falls through to thread_id. This matches the "guest mode
    # is ephemeral" contract documented in the flag's help text.
    effective_user_id = None if is_guest_mode else user_id
    session = RunnerSession(
        requested_mode=mode,
        resolved_mode=resolved_mode,
        llm_client=llm_client,
        thread_id=thread_id,
        sqlite_path=":memory:" if is_guest_mode else sqlite_path,
        memory_mode=memory_mode,
        user_id=effective_user_id,
        response_model_tier=response_model_tier,
        response_llm_client=response_llm_client,
    )

    async with AsyncExitStack() as stack:
        with console.status(
            "[accent]preparing session — warming up models and memory...[/accent]",
            spinner="dots",
        ):
            runtime = await stack.enter_async_context(
                PersistentAgentRuntime(
                    sqlite_path,
                    memory_mode=runtime_memory_mode,
                    memory_backend=settings.persistence_backend,
                    memory_database_url=settings.memory_database_url,
                    thread_persistence_backend=settings.persistence_backend,
                    thread_database_url=settings.memory_database_url,
                    crisis_log_persistence_backend=settings.persistence_backend,
                    crisis_log_database_url=settings.memory_database_url,
                    session_feedback_persistence_backend=settings.persistence_backend,
                    session_feedback_database_url=settings.memory_database_url,
                    memory_sqlite_path=memory_sqlite_path,
                    crisis_log_sqlite_path=crisis_log_sqlite_path,
                    default_llm_client=session.llm_client,
                )
            )
            session.history = await runtime.get_history(thread_id)
            session.last_context = await runtime.get_state(thread_id)

        pending_tail_task: asyncio.Task[AgentOutput] | None = None

        async def _finalize_pending_turn() -> None:
            """Wait for any background tail work and refresh session state.

            Returns:
                None.
            """

            nonlocal pending_tail_task
            if pending_tail_task is None:
                return

            await pending_tail_task
            pending_tail_task = None

            session.last_context = await runtime.get_state(session.thread_id)
            session.history = await runtime.get_history(session.thread_id)

        try:
            render_header(
                session.resolved_mode,
                session.thread_id,
                session.memory_mode,
                user_id=session.user_id,
                response_model_tier=session.response_model_tier,
            )
            render_info(
                "Session ready. Models, graph, and memory are warm.", style="success"
            )
            if session.history:
                render_info(
                    f"Resumed thread {session.thread_id} with {len(session.history)} stored messages.",
                    style="success",
                )

            while True:
                if pending_tail_task is not None and pending_tail_task.done():
                    await _finalize_pending_turn()

                try:
                    console.print(
                        "\n  [primary]·[/primary] [accent]you[/accent] ", end=""
                    )
                    user_text = (await asyncio.to_thread(Prompt.ask, "")).strip()
                except (EOFError, KeyboardInterrupt):
                    await _finalize_pending_turn()
                    console.print("\n  [hint]session ended[/hint]")
                    break

                await _finalize_pending_turn()

                if not user_text:
                    continue
                if user_text.startswith("/"):
                    if not await handle_command(user_text, session, runtime):
                        break
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    break

                console.print()
                console.print(Rule(style="panel", characters="─"))
                accumulated_text = ""
                final_output: AgentOutput | None = None
                response_ready_output: AgentOutput | None = None

                stream = runtime.run_turn_stream(
                    thread_id=session.thread_id,
                    user_id=session.owner_id(),
                    message=user_text,
                    channel=Channel.TEST,
                    llm_client=session.llm_client,
                    response_llm_client=session.response_llm_client,
                )

                with Live(console=console, refresh_per_second=15) as live:
                    status_renderable = Spinner(
                        "dots",
                        text=Text.assemble(
                            ("thinking", "primary"),
                            (" — waiting for pipeline", "hint"),
                        ),
                        style="primary",
                    )

                    def _stream_group(body) -> Group:
                        """Combine the spinner and response body for live rendering.

                        Args:
                            body: Rich renderable shown under the spinner.

                        Returns:
                            Rich group for the live display.
                        """

                        return Group(status_renderable, body)

                    live.update(_stream_group(Text("", style="muted")))
                    async for event in stream:
                        if isinstance(event, StatusEvent):
                            label = friendly_stage(event.stage)
                            detail = f" ({event.detail})" if event.detail else ""
                            status_renderable = Spinner(
                                "dots",
                                text=Text.assemble(
                                    (label, "primary"),
                                    (detail, "hint"),
                                ),
                                style="primary",
                            )
                            body = (
                                Panel(
                                    accumulated_text,
                                    title="[success]  reply  [/success]",
                                    subtitle=Text.from_markup(
                                        f"[muted]thread[/muted] [info]{session.thread_id}[/info]"
                                    ),
                                    border_style="panel",
                                    box=box.ROUNDED,
                                    padding=(1, 2),
                                )
                                if accumulated_text
                                else Text("", style="muted")
                            )
                            live.update(_stream_group(body))

                        elif isinstance(event, ChunkEvent):
                            accumulated_text += event.text
                            live.update(
                                _stream_group(
                                    Panel(
                                        accumulated_text,
                                        title="[success]  reply  [/success]",
                                        subtitle=Text.from_markup(
                                            f"[muted]thread[/muted] [info]{session.thread_id}[/info]"
                                        ),
                                        border_style="panel",
                                        box=box.ROUNDED,
                                        padding=(1, 2),
                                    )
                                )
                            )
                        elif isinstance(event, ResponseReadyEvent):
                            response_ready_output = event.output
                            live.update(
                                _response_panel(
                                    response_ready_output,
                                    thread_id=session.thread_id,
                                    turn_count=len(
                                        [
                                            m
                                            for m in session.history
                                            if m.role == MessageRole.USER
                                        ]
                                    )
                                    + 1,
                                )
                            )
                            break
                        elif isinstance(event, DoneEvent):
                            final_output = event.output
                            live.update(
                                _response_panel(
                                    final_output,
                                    thread_id=session.thread_id,
                                    turn_count=len(
                                        [
                                            m
                                            for m in session.history
                                            if m.role == MessageRole.USER
                                        ]
                                    )
                                    + 1,
                                )
                            )

                if response_ready_output is not None:
                    pending_tail_task = asyncio.create_task(
                        _drain_turn_stream_tail(
                            stream,
                        )
                    )
                    render_info(
                        "Saving memory in the background. Press Enter when you're ready for the next turn.",
                        style="muted",
                    )
                    continue

                if final_output is not None:
                    session.last_context = await runtime.get_state(session.thread_id)
                    session.history = await runtime.get_history(session.thread_id)
        finally:
            await _finalize_pending_turn()
            await runtime.finalize_active_sessions(llm_client=session.llm_client)


def main() -> int:
    """Run the OpenCouch CLI.

    With ``--voice``, starts the FastAPI server plus the LiveKit
    voice worker and opens the browser test page.
    Without ``--voice``, runs the interactive text CLI as usual.

    Returns:
        Process exit code for the CLI session.
    """

    args = build_parser().parse_args()

    if args.disable_tracing:
        os.environ["OPENCOUCH_DISABLE_TRACING"] = "1"

    if args.voice:
        return _run_voice_mode(args)

    thread_id = args.thread_id or generate_thread_id()
    sqlite_path = str(Path(args.sqlite_path).expanduser())
    memory_sqlite_path = str(Path(args.memory_sqlite_path).expanduser())
    crisis_log_sqlite_path = str(Path(args.crisis_log_sqlite_path).expanduser())
    memory_mode = resolve_memory_mode(args.memory_mode)
    asyncio.run(
        chat_loop(
            args.mode,
            thread_id=thread_id,
            user_id=args.user_id,
            response_model_tier=args.response_model_tier,
            sqlite_path=sqlite_path,
            memory_mode=memory_mode,
            memory_sqlite_path=memory_sqlite_path,
            crisis_log_sqlite_path=crisis_log_sqlite_path,
        )
    )
    return 0


def _run_voice_mode(args) -> int:
    """Start the voice mode server and open the browser.

    Launches uvicorn serving the FastAPI app plus a background
    ``agent.voice.agent`` worker, then opens the LiveKit test page
    in the default browser.

    The server runs in the foreground; Ctrl+C stops it.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code for voice mode startup and shutdown.
    """

    import webbrowser

    import uvicorn

    required_env = (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "OPENAI_API_KEY",
    )
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        console.print(
            Panel(
                "[accent]Voice mode requires LiveKit configuration.[/accent]\n\n"
                f"Missing env vars: [info]{', '.join(missing)}[/info]",
                title="Voice Mode Unavailable",
                style="warning",
            )
        )
        return 1

    port = args.port
    query = {"dispatch": "1"}
    if args.user_id:
        query["user"] = args.user_id
    if args.thread_id:
        query["thread"] = args.thread_id
    url = f"http://localhost:{port}/api/voice/livekit/test?{urlencode(query)}"
    worker_env = os.environ.copy()
    worker_env["OPENCOUCH_MEMORY_MODE"] = (
        "guest" if args.memory_mode == "guest" else "persistent"
    )
    worker_cmd = [sys.executable, "-m", "agent.voice.agent", "start"]

    console.print(Rule("[primary]OpenCouch Voice Mode[/primary]", style="panel"))
    console.print(
        f"[muted]Starting voice server on port[/muted] [info]{port}[/info]\n"
        f"[muted]Starting LiveKit worker:[/muted] [info]{' '.join(worker_cmd)}[/info]\n"
        f"[muted]Opening[/muted] [info]{url}[/info] [muted]in your browser...[/muted]\n"
        f"[muted]Press[/muted] [accent]Ctrl+C[/accent] [muted]to stop.[/muted]\n"
    )

    worker = subprocess.Popen(worker_cmd, env=worker_env)
    try:
        worker.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    else:
        console.print(
            Panel(
                "[accent]The LiveKit worker exited before the browser test page opened.[/accent]\n"
                "Check the worker logs above for the failure reason.",
                title="Voice Worker Failed",
                style="warning",
            )
        )
        return worker.returncode or 1

    # Open the browser after a short delay so the server has time
    # to start. We use a thread because uvicorn.run blocks.
    import threading

    def open_browser() -> None:
        """Open the voice test page after a short startup delay.

        Returns:
            None.
        """

        import time

        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        uvicorn.run("main:app", host="127.0.0.1", port=port, log_level="info")
    except KeyboardInterrupt:
        console.print("\n[muted]Voice server stopped.[/muted]")
    finally:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)

    return 0

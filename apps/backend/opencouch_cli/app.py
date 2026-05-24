"""Interactive CLI for the OpenCouch agent runtime.

Run from ``apps/backend/`` so default local paths resolve to the
backend working directory. The CLI supports deterministic smoke tests,
hybrid LLM runs, persistent local memory, and thread switching.

Common invocations:

``uv run python -m opencouch_cli --mode deterministic --memory-mode guest``
    Zero LLM calls, in-memory only. Useful for checking rendering and
    deterministic runtime paths.

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
    Dump the raw persisted runtime state when rendered panels are not
    enough for diagnosis.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import warnings
from contextlib import AsyncExitStack

from datetime import UTC, datetime
from pathlib import Path

from uuid import uuid4

from psycopg import OperationalError as PostgresOperationalError
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

from agent.feedback.models import FeedbackLabel, FeedbackSource
from agent.memory.models import StoredSessionArc
from agent.memory.modes import MemoryMode
from agent.memory.reconciliation import filter_active_semantic_records
from agent.runtime import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    ThreadSummary,
)
from agent.memory.procedural_profile import (
    aget_procedural_profile,
    aset_proactive_recall,
)
from agent.models import (
    AgentOutput,
    Channel,
    ChunkEvent,
    DoneEvent,
    MessageRole,
    ResponseReadyEvent,
    StatusEvent,
    friendly_stage,
)
from agent.state import AgentState
from agent.memory.entries import format_working_memory_entries
from agent.runtime.session.history import session_conversation_from_transcript
from agent.runtime.session.state import (
    get_transcript,
    render_session_conversation_json,
    render_session_conversation_markdown,
    render_session_conversation_text,
)
from config import (
    PersistenceBackend,
    ResponseModelTier,
    create_configured_control_llm_client,
    create_configured_response_llm_client,
    get_settings,
)
from opencouch_tui.command_helpers import (
    first_matching_text,
    format_entity_identifier,
    format_transcript_entries_plain,
    search_history_messages,
    snippet_around_match,
)
from opencouch_tui.dispatch.shared import (
    build_fact_forget_preview,
    build_session_forget_preview,
    confirmation_prompt_accepts,
    execute_memory_clear,
    execute_memory_forget,
    get_crisis_purge_plan,
    get_exit_save_confirmation_prompt,
    get_history_command_limit,
    get_memory_clear_plan,
    get_memory_forget_index,
    get_memory_forget_target,
    get_threads_command_summaries,
    get_typed_confirmation_prompt,
    get_yes_no_confirmation_prompt,
    parse_memory_clear_command,
    parse_memory_forget_command,
    parse_memory_overview_command,
    parse_memory_recall_command,
    parse_search_command,
    should_save_summary_on_exit,
)
from opencouch_tui.presenters import (
    left_rail_text as shared_left_rail_text,
    normal_reply_renderable as shared_normal_reply_renderable,
    owner_scope_display,
    render_doctor as shared_render_doctor,
    render_header as shared_render_header,
    render_help as shared_render_help,
    render_history as shared_render_history,
    render_keys as shared_render_keys,
    render_meta as shared_render_meta,
    render_onboarding as shared_render_onboarding,
    render_response as shared_render_response,
    render_stage_timings as shared_render_stage_timings,
    render_status as shared_render_status,
    render_threads as shared_render_threads,
    render_turn_activity as shared_render_turn_activity,
    render_turn_route as shared_render_turn_route,
    render_turn_trace as shared_render_turn_trace,
    response_panel_renderable as shared_response_panel_renderable,
)
from opencouch_tui.commands import (
    ALIASES,
    all_command_names,
    resolve_alias,
)
from opencouch_tui.input import (
    PromptToolbarState,
    available_prompt_themes,
    read_user_input,
    record_recent_command,
    set_prompt_theme,
)
from opencouch_tui.models import (
    ObservabilityMode,
    RunnerSession,
    TraceMode,
    UIMode,
)
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

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

# Prompt-toolbar "last action" line (best-effort, display-only).
_LAST_INFO_MESSAGE: str | None = None


def _split_stream_preview_text(
    accumulated_text: str,
) -> tuple[str | None, str]:
    """Separate tool-loading chatter from user-facing streamed reply text.

    The OpenAI Agents SDK can occasionally leak the therapeutic skill-loading
    call and its JSON payload into the raw text delta stream before the actual
    reply text arrives. For the live CLI preview we keep that internal chatter
    visually separate and only show the user-facing reply text in the reply
    panel.

    Args:
        accumulated_text: Raw streamed text accumulated so far.

    Returns:
        Tuple of ``(status_label, visible_text)`` where ``status_label`` is a
        muted helper line for the live preview, and ``visible_text`` is the
        cleaned reply text that should appear in the panel body.
    """

    stripped = accumulated_text.lstrip()
    tool_prefix = "load_therapeutic_response_skill("
    if not stripped.startswith(tool_prefix):
        return None, accumulated_text

    tool_status = "loading response style privately"
    payload_marker = "to=load_therapeutic_response_skill"
    payload_start = stripped.find(payload_marker)
    if payload_start == -1:
        return tool_status, ""

    json_start = stripped.find("{", payload_start + len(payload_marker))
    if json_start == -1:
        return tool_status, ""

    try:
        _, json_end = json.JSONDecoder().raw_decode(stripped[json_start:])
    except json.JSONDecodeError:
        return tool_status, ""

    visible_text = stripped[json_start + json_end :].lstrip()
    return tool_status, visible_text


def _live_preview_renderable(
    accumulated_text: str,
    *,
    thread_id: str,
) -> Group | Panel | Text:
    """Build the live-stream preview renderable for the current turn."""

    preview_status, visible_text = _split_stream_preview_text(accumulated_text)
    renderables: list[Panel | Text] = []

    if preview_status is not None:
        renderables.append(Text.from_markup(f"[muted]{preview_status}[/muted]"))

    if visible_text:
        renderables.append(
            _normal_reply_renderable(
                visible_text,
                thread_id=thread_id,
                turn_count=None,
                response_style=None,
                therapeutic_approach=None,
            )
        )

    if not renderables:
        return Text("", style="muted")
    if len(renderables) == 1:
        return renderables[0]
    return Group(*renderables)


def _response_panel(
    output: AgentOutput,
    *,
    thread_id: str | None = None,
    turn_count: int | None = None,
) -> Panel | Group:
    """Build the terminal assistant-response panel for one turn."""

    return shared_response_panel_renderable(
        output,
        console=console,
        thread_id=thread_id,
        turn_count=turn_count,
    )


def _normal_reply_renderable(
    response_text: str,
    *,
    thread_id: str | None,
    turn_count: int | None,
    response_style: str | None,
    therapeutic_approach: str | None,
) -> Group:
    """Build lightweight chrome for a normal assistant reply."""

    return shared_normal_reply_renderable(
        response_text,
        thread_id=thread_id,
        turn_count=turn_count,
        response_style=response_style,
        therapeutic_approach=therapeutic_approach,
        console=console,
    )


def _left_rail_text(value: str, *, style: str) -> Text:
    """Render multiline text against a subtle terminal left rail."""

    return shared_left_rail_text(value, style=style, console=console)


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


def _recoverable_error_message(prefix: str, exc: Exception) -> str:
    """Return a compact user-facing error for recoverable CLI failures."""

    detail = str(exc).strip() or type(exc).__name__
    return (
        f"{prefix}: {detail}\n"
        "The CLI stayed open. Fix the runtime configuration or retry the turn."
    )


def _owner_scope_display(session: RunnerSession) -> str:
    """Return a compact owner-scope label for status surfaces."""

    return owner_scope_display(session)


def _thread_scoped_memory_hint(session: RunnerSession) -> str | None:
    """Return the cross-thread recall hint for thread-scoped memory.

    Args:
        session (RunnerSession): Current CLI session state.

    Returns:
        str | None: Hint text when useful, otherwise None.
    """

    if session.memory_mode != "persistent" or session.user_id:
        return None
    return (
        "This is a thread-scoped memory session. For cross-thread recall while "
        "dogfooding, restart with --user-id <name>."
    )


def _render_persistence_startup_error(
    *,
    backend: PersistenceBackend,
    exc: BaseException,
) -> None:
    """Render an actionable persistence startup failure.

    Args:
        backend (PersistenceBackend): Configured persistence backend.
        exc (BaseException): Startup exception raised by the runtime.

    Returns:
        None.
    """

    if backend == "postgres":
        render_info(
            "Postgres persistence is configured, but the database is not "
            "reachable. Start it from the repo root with:\n\n"
            "  docker compose -f compose.yml up -d postgres --wait\n\n"
            "For normal text-agent dogfooding, prefer:\n\n"
            "  ./scripts/text_repl.sh --memory-mode persistent "
            "--user-id dogfood --response-model-tier quality\n\n"
            "For a no-database smoke test, use --memory-mode guest or set "
            "OPENCOUCH_PERSISTENCE_BACKEND=sqlite.",
            style="danger",
        )
        return

    render_info(
        f"Could not open the {backend} persistence backend: {exc}",
        style="danger",
    )


def _prompt_toolbar_state(
    session: RunnerSession,
    *,
    pending_status: str | None,
) -> PromptToolbarState:
    """Build prompt metadata for the enhanced REPL toolbar.

    Args:
        session (RunnerSession): Current CLI session state.
        pending_status (str | None): Short background-work status label.

    Returns:
        PromptToolbarState: Input-toolbar metadata.
    """

    return PromptToolbarState(
        resolved_mode=session.resolved_mode,
        memory_mode=session.memory_mode,
        response_model_tier=session.response_model_tier,
        thread_id=session.thread_id,
        user_id=session.user_id,
        pending_status=pending_status,
        ui_mode=session.ui_mode,
        last_action=_LAST_INFO_MESSAGE,
    )


def _pending_tail_status(session: RunnerSession) -> str:
    """Return a mode-aware label for post-response background work.

    Args:
        session (RunnerSession): Current CLI session state.

    Returns:
        str: User-facing pending status label.
    """

    if session.memory_mode == "persistent":
        return "saving memory"
    return "finishing turn"


def _pending_tail_message(session: RunnerSession) -> str:
    """Return the mode-aware post-response background-work message.

    Args:
        session (RunnerSession): Current CLI session state.

    Returns:
        str: User-facing informational message.
    """

    if session.memory_mode == "persistent":
        return (
            "Saving memory in the background. "
            "Press Enter when you're ready for the next turn."
        )
    return (
        "Finishing turn in the background. "
        "Press Enter when you're ready for the next turn."
    )


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
            "Legacy SQLite path for persisted session state. Deprecated "
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


def _persistent_mode_hint(persistence_backend: PersistenceBackend) -> str:
    """Return backend-aware copy for the persistent memory choice.

    Args:
        persistence_backend (PersistenceBackend): Configured persistence backend.

    Returns:
        str: Short user-facing persistent-mode description.
    """

    if persistence_backend == "postgres":
        return "save memory using Postgres"
    return "save memory using SQLite"


def resolve_memory_mode(
    memory_mode: str,
    *,
    persistence_backend: PersistenceBackend | None = None,
) -> str:
    """Resolve memory mode from CLI arg, prompting when needed.

    Args:
        memory_mode: Raw CLI memory mode argument.
        persistence_backend: Optional configured persistent storage backend.

    Returns:
        Resolved memory mode, either ``"guest"`` or ``"persistent"``.
    """

    if memory_mode in {"guest", "persistent"}:
        return memory_mode

    backend = persistence_backend or get_settings().persistence_backend
    persistent_hint = _persistent_mode_hint(backend)
    console.print()
    console.print(Rule(style="panel", characters="─"))
    console.print("  [primary]Choose Memory Mode[/primary]", highlight=False)
    console.print()
    console.print(
        "  [accent]1[/accent]  [info]Guest Mode[/info]  [hint]— private, in-memory only[/hint]"
    )
    console.print(
        f"  [accent]2[/accent]  [info]Persistent Mode[/info]  [hint]— {persistent_hint}[/hint]"
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
    """Render the CLI header."""

    shared_render_header(
        mode,
        thread_id,
        memory_mode,
        console=console,
        user_id=user_id,
        response_model_tier=response_model_tier,
    )


def render_response(
    response_text: str,
    *,
    is_crisis: bool,
    thread_id: str | None = None,
    turn_count: int | None = None,
) -> None:
    """Render the assistant reply inside a styled panel."""

    shared_render_response(
        response_text,
        console=console,
        is_crisis=is_crisis,
        thread_id=thread_id,
        turn_count=turn_count,
    )


def render_meta(
    *,
    response_style: str | None,
    response_type: str,
    level: int,
    needs_clarification: bool,
    needs_crisis_response: bool,
    reason: str,
    diagnostics: dict | None = None,
    memory_deltas: dict | None = None,
    verbose: bool = False,
) -> None:
    """Render classifier metadata as a compact diagnostics line."""

    shared_render_meta(
        console=console,
        response_style=response_style,
        response_type=response_type,
        level=level,
        needs_clarification=needs_clarification,
        needs_crisis_response=needs_crisis_response,
        reason=reason,
        diagnostics=diagnostics,
        memory_deltas=memory_deltas,
        verbose=verbose,
    )


def render_turn_activity(
    output: AgentOutput,
    *,
    observability_mode: ObservabilityMode,
) -> None:
    """Render tool activity after the route line."""

    shared_render_turn_activity(
        output,
        console=console,
        observability_mode=observability_mode,
    )


def render_turn_route(
    output: AgentOutput,
    *,
    pending_status: str | None = None,
) -> None:
    """Render a compact routing summary for one completed response."""

    shared_render_turn_route(
        output,
        console=console,
        pending_status=pending_status,
    )


def render_turn_trace(
    output: AgentOutput,
    *,
    status_stages: list[str] | None = None,
    pending_status: str | None = None,
) -> None:
    """Render the optional routing trace diagram for one turn."""

    shared_render_turn_trace(
        output,
        console=console,
        status_stages=status_stages,
        pending_status=pending_status,
    )


def _render_stage_timings(diagnostics: dict, memory_deltas: dict) -> None:
    """Render the per-turn stage timings + memory-write table."""

    shared_render_stage_timings(
        diagnostics,
        memory_deltas,
        console=console,
    )


def render_context(state: AgentState | None) -> None:
    """Render the current structured session context.

    The panel shows working memory, session continuity, procedural
    rules, the proactive-recall toggle, and active guided-exercise
    state.

    Args:
        state: Most recent persisted runtime state snapshot.

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
    turn_lifecycle = state.get("turn_lifecycle", {})
    memory_reference = state.get("memory_reference", {})
    grounded_lookup = state.get("grounded_lookup", {})
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")
    table.add_row("turn_count", str(session_progress.get("turn_count", 0)))

    if isinstance(turn_lifecycle, dict):
        active_flow = turn_lifecycle.get("active_flow", "none")
        action = turn_lifecycle.get("action", "none")
        table.add_row("turn_lifecycle", f"{active_flow} / {action}")

    if isinstance(memory_reference, dict):
        reference_mode = str(memory_reference.get("mode") or "none")
        if reference_mode != "none":
            table.add_row("memory_reference", reference_mode)

    if isinstance(grounded_lookup, dict):
        lookup_status = str(grounded_lookup.get("status") or "not_attempted")
        lookup_query = str(grounded_lookup.get("query") or "")
        if lookup_status != "not_attempted" or lookup_query:
            lookup_display = lookup_status
            if lookup_query:
                lookup_display = f"{lookup_status} · {lookup_query}"
            table.add_row("grounded_lookup", lookup_display)

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
            subtitle="[hint]what the session is carrying forward[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


async def _render_debug_state(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Dump the raw persisted state for the active thread as JSON.

    Backs ``/debug state``. This is the full-state view for when the
    rendered context and timing panels are not enough to diagnose a
    turn.

    Pydantic models and other non-JSON types round-trip through
    ``default=str`` in ``json.dumps`` rather than crashing. Most
    state fields are already plain dicts (we serialize to JSON at
    state persistence time via the runtime serializer), so this fallback
    only kicks in for an odd CrisisAssessment instance
    that survived the round-trip as a typed model.

    Degrades gracefully when no state exists yet (fresh thread or
    ``/reset`` was just called). Prints a warning panel instead of
    crashing on a None state.

    Args:
        runtime: Persistent runtime used to fetch persisted state.
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
            subtitle="[hint]raw persisted state dict[/hint]",
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
    """Return the ``identifier`` field of a serialized :class:`EntityRef`."""

    return format_entity_identifier(entity)


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

    index_1based = _parse_one_based_index(index_str, kind_label="rule")
    if index_1based is None:
        return

    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.owner_id()
    )
    error, target_rule = get_memory_forget_target(
        profile.rules,
        index_1based=index_1based,
        kind_title="Rule",
        empty_message="No procedural rules to forget for this thread.",
        count_label="rule(s)",
    )
    if error is not None:
        render_info(error, style="warning")
        return

    # Rules are short enough to inline in the confirmation prompt.
    console.print()
    console.print(
        Panel(
            f"[info]{target_rule.rule}[/info]",
            title=(f"[warning]Delete rule #{index_1based}?[/warning]"),
            border_style="warning",
            box=box.ROUNDED,
        )
    )
    answer = Prompt.ask(**get_yes_no_confirmation_prompt(subject="rule"))
    if not confirmation_prompt_accepts(answer):
        render_info("Cancelled — no rules deleted.", style="info")
        return

    deleted = await execute_memory_forget(
        runtime,
        owner_id=session.owner_id(),
        kind="rule",
        target=target_rule,
    )
    if not deleted["deleted"]:
        render_info(
            "The selected rule changed before deletion could be applied.",
            style="warning",
        )
        return

    render_info(
        f"Deleted rule #{index_1based}. "
        f"{deleted['remaining']} rule(s) remaining for this thread.",
        style="success",
    )


# Privacy controls: /memory forget fact|session and /memory clear


def _parse_one_based_index(
    index_str: str,
    *,
    kind_label: str,
) -> int | None:
    """Parse a 1-indexed CLI argument into an int, or render a warning."""

    error, index_1based = get_memory_forget_index(index_str, kind_label=kind_label)
    if error is not None:
        render_info(error, style="warning")
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
    answer = Prompt.ask(**get_yes_no_confirmation_prompt(subject=kind_label))
    return confirmation_prompt_accepts(answer)


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
    4. On confirm, call the shared forget execution helper. A single
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
    error, target_fact = get_memory_forget_target(
        facts,
        index_1based=index_1based,
        kind_title="Fact",
        empty_message="No semantic facts to forget for this thread.",
        count_label="fact(s)",
    )
    if error is not None:
        render_info(error, style="warning")
        return

    _namespace, _key, value = target_fact
    preview = build_fact_forget_preview(
        value,
        format_entity_identifier=_format_entity_identifier,
    )

    if not _render_forget_confirmation(
        kind_label="fact",
        index_1based=index_1based,
        preview_lines=preview,
    ):
        render_info("Cancelled — no facts deleted.", style="info")
        return

    deleted = await execute_memory_forget(
        runtime,
        owner_id=session.owner_id(),
        kind="fact",
        target=target_fact,
    )
    if not deleted["deleted"]:
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
    confirmation pattern, same single-record delete path.

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
    error, target_session = get_memory_forget_target(
        sessions,
        index_1based=index_1based,
        kind_title="Session",
        empty_message="No episodic sessions to forget for this thread.",
        count_label="session arc(s)",
    )
    if error is not None:
        render_info(error, style="warning")
        return

    _namespace, _key, value = target_session
    preview = build_session_forget_preview(value)

    if not _render_forget_confirmation(
        kind_label="session",
        index_1based=index_1based,
        preview_lines=preview,
    ):
        render_info("Cancelled — no session arcs deleted.", style="info")
        return

    deleted = await execute_memory_forget(
        runtime,
        owner_id=session.owner_id(),
        kind="session",
        target=target_session,
    )
    if not deleted["deleted"]:
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

    The execution sweep now lives in the shared dispatch layer so the
    CLI remains responsible only for rendering, confirmation, and exact
    user-facing messaging. The ``all`` path is still not atomic across
    kinds.

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

    # Show concrete counts before destructive confirmation.
    counts = await get_memory_clear_plan(runtime, owner_id=owner_id, kind=kind)

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
    answer = Prompt.ask(**get_typed_confirmation_prompt())
    if not confirmation_prompt_accepts(answer, expected_word="clear"):
        render_info(
            "Cancelled — no memory cleared.",
            style="info",
        )
        return

    deleted_counts = await execute_memory_clear(
        runtime,
        owner_id=owner_id,
        kind=kind,
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
    total_before, cutoff = await get_crisis_purge_plan(runtime, days=days)

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
    """Render the available slash commands grouped by category."""

    shared_render_help(console=console)


def render_keys() -> None:
    """Render keyboard shortcuts and prompt usage hints."""

    shared_render_keys(console=console)


def render_onboarding() -> None:
    """Render a one-time quick-start guide for first prompt."""

    shared_render_onboarding(console=console)


def render_status(session: RunnerSession) -> None:
    """Render current runner status."""

    shared_render_status(session, console=console)


def render_doctor(session: RunnerSession, *, verbose: bool = False) -> None:
    """Render runtime readiness checks for the active CLI session."""

    shared_render_doctor(session, console=console, verbose=verbose)


def render_history(session: RunnerSession, limit: int = 6) -> None:
    """Render the most recent transcript entries."""

    shared_render_history(session, console=console, limit=limit)


def render_info(message: str, *, style: str = "panel") -> None:
    """Render a lightweight informational panel.

    Args:
        message: Informational message to display.
        style: Border color/style token for the panel.

    Returns:
        None.
    """

    global _LAST_INFO_MESSAGE
    _LAST_INFO_MESSAGE = message

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
    """Render a compact table of persisted thread summaries."""

    shared_render_threads(
        threads,
        active_thread_id=active_thread_id,
        console=console,
        info_renderer=render_info,
    )


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


def set_trace_mode(session: RunnerSession, trace_mode: TraceMode) -> None:
    """Update the optional routing trace display mode.

    Args:
        session (RunnerSession): Mutable CLI session state.
        trace_mode (TraceMode): New trace mode.

    Returns:
        None.
    """

    session.trace_mode = trace_mode


def set_ui_mode(session: RunnerSession, ui_mode: UIMode) -> None:
    """Update prompt-toolbar density mode for subsequent prompts.

    Args:
        session (RunnerSession): Mutable CLI session state.
        ui_mode (UIMode): New toolbar density mode.

    Returns:
        None.
    """

    session.ui_mode = ui_mode


def set_observability_mode(
    session: RunnerSession,
    observability_mode: ObservabilityMode,
) -> None:
    """Update turn observability detail for subsequent turns.

    Args:
        session (RunnerSession): Mutable CLI session state.
        observability_mode (ObservabilityMode): New observability mode.

    Returns:
        None.
    """

    session.observability_mode = observability_mode


def set_session_prompt_theme(
    session: RunnerSession,
    theme_name: str,
) -> bool:
    """Update prompt theme preset for subsequent prompts.

    Args:
        session (RunnerSession): Mutable CLI session state.
        theme_name (str): New prompt theme preset name.

    Returns:
        bool: True when the theme was applied.
    """

    if not set_prompt_theme(theme_name):
        return False
    session.prompt_theme = theme_name  # type: ignore[assignment]
    return True


def toggle_trace_mode(session: RunnerSession) -> str:
    """Toggle trace display for command handling.

    Args:
        session (RunnerSession): Mutable CLI session state.

    Returns:
        str: The new trace mode.
    """

    session.trace_mode = "off" if session.trace_mode != "off" else "on"
    return session.trace_mode


def _trace_enabled(session: RunnerSession) -> bool:
    """Return whether the trace overlay should render for the current turn.

    Args:
        session (RunnerSession): Mutable CLI session state.

    Returns:
        bool: True when trace mode is on or once.
    """

    return session.trace_mode in {"on", "once"}


def _consume_trace_once(session: RunnerSession) -> None:
    """Turn one-shot trace mode off after it renders.

    Args:
        session (RunnerSession): Mutable CLI session state.

    Returns:
        None.
    """

    if session.trace_mode == "once":
        session.trace_mode = "off"


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


def _default_export_filename(*, format_name: str) -> str:
    """Return a timestamped default filename for transcript exports."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"session-{timestamp}.{format_name}"


async def _active_session_conversation(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
):
    """Return the active thread's canonical public conversation."""

    state = await runtime.get_state(session.thread_id)
    if state is None:
        return None, "No transcript is available for this thread yet."
    transcript = get_transcript(state)
    conversation = session_conversation_from_transcript(transcript)
    if conversation.is_empty:
        return None, "No transcript is available for this thread yet."
    return conversation, None


def _render_search_results(
    *,
    query: str,
    mode: str,
    results: list[tuple[str, str]],
    empty_message: str,
) -> None:
    """Render compact labeled search results."""

    if not results:
        render_info(empty_message, style="warning")
        return

    display_limit = 8
    visible_results = results[:display_limit]
    truncated_count = max(0, len(results) - display_limit)

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE, expand=True)
    table.add_column("source", style="accent", no_wrap=True, width=14)
    table.add_column("match", style="info")

    for source, snippet in visible_results:
        table.add_row(source, snippet)

    subtitle = f'[hint]query: "{query}"[/hint]'
    if truncated_count:
        subtitle += f" [hint]· and {truncated_count} more match(es)[/hint]"

    console.print(
        Panel(
            table,
            title=f"[muted]search · {mode}[/muted]",
            subtitle=subtitle,
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def _search_history_conversation(
    conversation,
    *,
    query: str,
) -> list[tuple[str, str]]:
    """Search the active conversation transcript for visible message matches."""

    return search_history_messages(conversation.messages, query=query)


async def _search_memory_records(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    *,
    query: str,
) -> list[tuple[str, str]]:
    """Search semantic, episodic, and procedural memory for the active owner."""

    owner_id = session.owner_id()
    results: list[tuple[str, str]] = []

    semantic_records = await _collect_records_by_kind(
        runtime,
        kind="semantic",
        owner_id=owner_id,
    )
    for index, (_key, value) in enumerate(semantic_records, start=1):
        object_identifier = _format_entity_identifier(value.get("object"))
        predicate = str(value.get("predicate", "") or "").strip()
        evidence_quote = str(value.get("evidence_quote", "") or "").strip()
        semantic_summary = " ".join(
            part for part in (predicate, object_identifier) if part and part != "?"
        ).strip()
        matched_text = first_matching_text(query, evidence_quote, semantic_summary)
        if matched_text is None:
            continue
        results.append(
            (
                "memory/fact",
                f"#{index}: {snippet_around_match(matched_text, query)}",
            )
        )

    episodic_records = await _collect_records_by_kind(
        runtime,
        kind="episodic",
        owner_id=owner_id,
    )
    for index, (_key, value) in enumerate(episodic_records, start=1):
        summary = str(value.get("summary", "") or "").strip()
        themes_value = value.get("primary_themes")
        themes = (
            ", ".join(str(theme) for theme in themes_value)
            if isinstance(themes_value, list)
            else ""
        )
        matched_text = first_matching_text(query, summary, themes)
        if matched_text is None:
            continue
        results.append(
            (
                "memory/session",
                f"#{index}: {snippet_around_match(matched_text, query)}",
            )
        )

    try:
        profile = await aget_procedural_profile(runtime.memory_store, user_id=owner_id)
    except AttributeError:
        profile = None

    if profile is not None:
        for index, rule in enumerate(profile.rules, start=1):
            evidence_text = " ".join(
                str(evidence).strip() for evidence in rule.evidence
            )
            matched_text = first_matching_text(query, rule.rule, evidence_text)
            if matched_text is None:
                continue
            results.append(
                (
                    "memory/rule",
                    f"#{index}: {snippet_around_match(matched_text, query)}",
                )
            )

    return results


async def _handle_search_command(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    args: list[str],
) -> bool:
    """Search the active transcript, stored memory, or both."""

    error, parsed = parse_search_command(["/search", *args])
    if error is not None:
        render_info(error, style="warning")
        return True
    if parsed is None:
        render_info("Usage: /search <history|memory|all> <query>", style="warning")
        return True

    mode, query = parsed

    if mode == "history":
        conversation, error = await _active_session_conversation(session, runtime)
        if conversation is None:
            render_info(
                error or "No transcript is available for this thread yet.",
                style="warning",
            )
            return True
        results = _search_history_conversation(conversation, query=query)
        _render_search_results(
            query=query,
            mode="history",
            results=results,
            empty_message=f'No history matches for "{query}".',
        )
        return True

    memory_results = await _search_memory_records(session, runtime, query=query)
    if mode == "memory":
        _render_search_results(
            query=query,
            mode="memory",
            results=memory_results,
            empty_message=f'No memory matches for "{query}".',
        )
        return True

    conversation, error = await _active_session_conversation(session, runtime)
    history_results: list[tuple[str, str]] = []
    if conversation is not None:
        history_results = _search_history_conversation(conversation, query=query)
    elif error:
        render_info(error, style="warning")
    combined_results = [*history_results, *memory_results]
    _render_search_results(
        query=query,
        mode="all",
        results=combined_results,
        empty_message=f'No matches for "{query}".',
    )
    return True


async def _handle_export_command(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    args: list[str],
) -> bool:
    """Export the active thread transcript in a user-selected format."""

    if not args or args[0] not in {"md", "json", "txt"}:
        render_info("Usage: /export <md|json|txt> [filename]", style="warning")
        return True

    conversation, error = await _active_session_conversation(session, runtime)
    if conversation is None:
        render_info(error or "No transcript is available to export.", style="warning")
        return True

    format_name = args[0]
    filename = " ".join(args[1:]).strip() if len(args) > 1 else ""
    if not filename:
        filename = _default_export_filename(format_name=format_name)

    path = Path(filename).expanduser()
    if format_name == "md":
        rendered = render_session_conversation_markdown(
            conversation,
            thread_id=session.thread_id,
        )
    elif format_name == "json":
        rendered = render_session_conversation_json(
            conversation,
            thread_id=session.thread_id,
        )
    else:
        rendered = render_session_conversation_text(
            conversation,
            thread_id=session.thread_id,
        )

    try:
        path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        render_info(f"Could not write export file: {exc}", style="warning")
        return True

    render_info(f"Exported transcript to {path}.", style="success")
    return True


async def _summarize_conversation_text(
    session: RunnerSession,
    conversation,
    *,
    detail: str,
) -> str:
    """Return a non-persisting summary of the active conversation."""

    transcript_text = format_transcript_entries_plain(
        list(conversation.transcript_entries()),
        uppercase_roles=True,
    )

    if session.llm_client is None:
        if detail == "full":
            return (
                "Summary\n"
                f"- Transcript messages: {len(conversation.messages)}\n"
                f"- User turns: {conversation.user_turn_count}\n"
                "- No summary LLM is configured in this session.\n"
                "- Use /export md for the full transcript."
            )
        return (
            "Summary\n"
            f"- Transcript messages: {len(conversation.messages)}\n"
            f"- User turns: {conversation.user_turn_count}\n"
            "- No summary LLM is configured in this session."
        )

    prompt = (
        "Summarize the following OpenCouch conversation.\n"
        f"Detail level: {detail}.\n"
        "If detail level is short, return 4-6 concise bullets.\n"
        "If detail level is full, use these sections exactly:\n"
        "Summary\nKey decisions\nOpen questions\nNext steps\n"
        "Be faithful to the transcript. Do not invent facts.\n\n"
        f"Transcript:\n{transcript_text}"
    )
    return await session.llm_client.generate_text(
        prompt=prompt,
        system_instruction=(
            "You summarize OpenCouch chat transcripts for the local CLI. "
            "Be concise, accurate, and faithful to the visible transcript."
        ),
    )


async def _handle_summary_command(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    args: list[str],
) -> bool:
    """Render a non-persisting recap of the active thread transcript."""

    detail = "short"
    if args:
        if len(args) != 1 or args[0] not in {"short", "full"}:
            render_info("Usage: /summary [short|full]", style="warning")
            return True
        detail = args[0]

    conversation, error = await _active_session_conversation(session, runtime)
    if conversation is None:
        render_info(
            error or "No transcript is available to summarize.", style="warning"
        )
        return True

    try:
        summary = await _summarize_conversation_text(
            session,
            conversation,
            detail=detail,
        )
    except Exception as exc:
        render_info(
            _recoverable_error_message("Session summary failed", exc),
            style="danger",
        )
        return True

    console.print(
        Panel(
            Text(summary.strip(), style="info"),
            title=f"[muted]session summary ({detail})[/muted]",
            subtitle="[hint]need the full conversation record? use /export md[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()
    return True


async def _summarize_and_render(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
    *,
    source: FeedbackSource,
    final_message: str | None = None,
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
        final_message: Optional success message rendered after summarization.
            When None, uses the default closing copy.

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

    message = final_message
    if message is None:
        message = (
            "Take care. We can pick this back up whenever you want. "
            f"(Thread: {session.thread_id})"
        )
    if message:
        render_info(message, style="success")


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
    command = resolve_alias(parts[0].lower())
    args = parts[1:]

    # Track for the recent-command quick picker.
    record_recent_command(command)

    if command in {"/exit", "/quit"}:
        # Default toward saving a summary; users can delete it later if needed.
        save = Prompt.ask(**get_exit_save_confirmation_prompt())
        if should_save_summary_on_exit(save):
            await _summarize_and_render(session, runtime, source="cli_exit")
        # If the user declines saving, skip feedback capture as well.
        return False

    if command == "/end":
        if args:
            if args[0] != "new" or len(args) > 2:
                render_info("Usage: /end [new [thread-id]]", style="warning")
                return True

            next_thread_id = args[1] if len(args) == 2 else generate_thread_id()
            if next_thread_id == session.thread_id:
                render_info(
                    "The next thread id is already active. Choose a different id.",
                    style="warning",
                )
                return True
            if await runtime.get_state(next_thread_id) is not None:
                render_info(
                    f"Thread {next_thread_id} already exists. Use /resume "
                    f"{next_thread_id} or choose another id.",
                    style="warning",
                )
                return True

            previous_thread_id = session.thread_id
            await _summarize_and_render(
                session,
                runtime,
                source="cli_end",
                final_message=f"Saved session {previous_thread_id}.",
            )
            if not await switch_thread(
                session,
                runtime,
                thread_id=next_thread_id,
                require_existing=False,
            ):
                render_info(
                    f"Session saved, but could not start thread {next_thread_id}.",
                    style="warning",
                )
                return True
            render_header(
                session.resolved_mode,
                session.thread_id,
                session.memory_mode,
                user_id=session.user_id,
                response_model_tier=session.response_model_tier,
            )
            render_info(
                f"Started new thread {session.thread_id}.",
                style="success",
            )
            hint = _thread_scoped_memory_hint(session)
            if hint:
                render_info(hint, style="warning")
            return True

        # Explicit end command means full closing flow without save confirmation.
        await _summarize_and_render(session, runtime, source="cli_end")
        return False

    if command == "/memory":
        error, overview = parse_memory_overview_command([command, *args])
        if error is not None:
            render_info(error, style="warning")
            return True
        if overview is not None:
            action, kind = overview
            if action == "status":
                await render_memory_status(runtime, session)
                return True
            if kind == "rules":
                await render_memory_list_rules(runtime, session)
                return True
            if kind == "facts":
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
            if kind == "sessions":
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
            error, enable = parse_memory_recall_command([command, *args])
            if error is not None:
                render_info(error, style="warning")
                return True
            if enable is None:
                profile = await aget_procedural_profile(
                    runtime.memory_store,
                    user_id=session.owner_id(),
                )
                state = "on" if profile.proactive_recall_enabled else "off"
                render_info(
                    f"Proactive recall is {state}. Use /memory recall on or /memory recall off to change it.",
                    style="info",
                )
                return True
            await render_memory_recall_toggle(runtime, session, enable=enable)
            return True
        if args[0] == "forget":
            error, parsed_forget = parse_memory_forget_command([command, *args])
            if error is not None:
                render_info(error, style="warning")
                return True
            if parsed_forget is None:
                render_info(
                    "Usage: /memory forget <fact|session|rule> <n>",
                    style="warning",
                )
                return True
            kind, index_str = parsed_forget
            if kind == "rule":
                await render_memory_forget_rule(runtime, session, index_str=index_str)
                return True
            if kind == "fact":
                await render_memory_forget_fact(runtime, session, index_str=index_str)
                return True
            await render_memory_forget_session(runtime, session, index_str=index_str)
            return True
        if args[0] == "clear":
            error, clear_kind = parse_memory_clear_command([command, *args])
            if error is not None:
                render_info(error, style="warning")
                return True
            if clear_kind is None:
                render_info(
                    "Usage: /memory clear <facts|sessions|rules|all>",
                    style="warning",
                )
                return True
            await render_memory_clear(runtime, session, kind=clear_kind)
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

    if command == "/search":
        return await _handle_search_command(session, runtime, args)

    if command == "/summary":
        return await _handle_summary_command(session, runtime, args)

    if command == "/export":
        return await _handle_export_command(session, runtime, args)

    if command == "/help":
        render_help()
        return True

    if command == "/keys":
        render_keys()
        return True

    if command == "/ui":
        if len(args) == 0:
            next_mode: UIMode = "compact" if session.ui_mode == "full" else "full"
            set_ui_mode(session, next_mode)
            render_info(f"UI mode updated. ui={session.ui_mode}", style="success")
            return True
        if len(args) != 1 or args[0] not in {"compact", "full"}:
            render_info("Usage: /ui <compact|full>", style="warning")
            return True
        set_ui_mode(session, args[0])  # type: ignore[arg-type]
        render_info(f"UI mode updated. ui={session.ui_mode}", style="success")
        return True

    if command == "/theme":
        theme_options = "|".join(available_prompt_themes())
        if len(args) == 0:
            render_info(
                f"Current theme: {session.prompt_theme}. Available themes: {theme_options}.",
                style="info",
            )
            return True
        if len(args) != 1:
            render_info(f"Usage: /theme <{theme_options}>", style="warning")
            return True
        if not set_session_prompt_theme(session, args[0]):
            render_info(f"Usage: /theme <{theme_options}>", style="warning")
            return True
        render_info(f"Theme updated. theme={session.prompt_theme}", style="success")
        return True

    if command == "/verbosity":
        if len(args) != 1 or args[0] not in {"compact", "verbose"}:
            render_info("Usage: /verbosity <compact|verbose>", style="warning")
            return True
        set_observability_mode(session, args[0])  # type: ignore[arg-type]
        render_info(
            f"Verbosity updated. verbosity={session.observability_mode}",
            style="success",
        )
        return True

    if command == "/status":
        render_status(session)
        return True

    if command == "/doctor":
        if len(args) > 1 or (len(args) == 1 and args[0] != "verbose"):
            render_info("Usage: /doctor [verbose]", style="warning")
            return True
        render_doctor(session, verbose=(len(args) == 1 and args[0] == "verbose"))
        return True

    if command == "/history":
        error, limit = get_history_command_limit([command, *args])
        if error is not None:
            render_info(error, style="warning")
            return True
        render_history(session, limit=limit or 6)
        return True

    if command == "/context":
        render_context(session.last_context)
        return True

    if command == "/threads":
        error, summaries = await get_threads_command_summaries(
            [command, *args],
            runtime=runtime,
        )
        if error is not None:
            render_info(error, style="warning")
            return True
        render_threads(
            summaries or [],
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
        hint = _thread_scoped_memory_hint(session)
        if hint:
            render_info(hint, style="warning")
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

    if command == "/trace":
        if len(args) == 0:
            render_info(
                f"Routing trace is {session.trace_mode}. "
                "Use /trace on, /trace off, or /trace once.",
                style="info",
            )
            return True
        if len(args) != 1 or args[0] not in {"on", "off", "once", "toggle"}:
            render_info("Usage: /trace on|off|once", style="warning")
            return True
        if args[0] == "toggle":
            new_mode = toggle_trace_mode(session)
        else:
            set_trace_mode(session, args[0])  # type: ignore[arg-type]
            new_mode = session.trace_mode
        render_info(f"Routing trace {new_mode}.", style="success")
        return True

    if command == "/debug":
        if len(args) == 0 or args[0] != "state":
            render_info(
                "Usage: /debug state  (dumps raw persisted state for the active thread)",
                style="warning",
            )
            return True
        await _render_debug_state(runtime, session)
        return True

    # Fuzzy-match against known commands + aliases for helpful suggestions.
    candidates = all_command_names() + list(ALIASES.keys())
    close = difflib.get_close_matches(command, candidates, n=3, cutoff=0.5)
    if close:
        suggestions = ", ".join(close)
        render_info(
            f"Unknown command: {command}. Did you mean: {suggestions}?  Type /help for full list.",
            style="warning",
        )
    else:
        render_info(
            f"Unknown command: {command}. Type /help for full list.", style="warning"
        )
    return True


async def _run_interactive_session(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
) -> None:
    """Run the prompt/stream loop for an already-initialized CLI session."""

    pending_tail_task: asyncio.Task[AgentOutput] | None = None

    async def _finalize_pending_turn() -> None:
        """Wait for any background tail work and refresh session state."""

        nonlocal pending_tail_task
        if pending_tail_task is None:
            return

        try:
            await pending_tail_task
        except Exception as exc:
            logger.warning("CLI response tail failed", exc_info=True)
            render_info(
                _recoverable_error_message("Response tail failed", exc),
                style="danger",
            )
        finally:
            pending_tail_task = None

        session.last_context = await runtime.get_state(session.thread_id)
        session.history = await runtime.get_history(session.thread_id)

    try:
        set_prompt_theme(session.prompt_theme)
        render_header(
            session.resolved_mode,
            session.thread_id,
            session.memory_mode,
            user_id=session.user_id,
            response_model_tier=session.response_model_tier,
        )
        if session.show_onboarding:
            render_onboarding()
            session.show_onboarding = False
        render_info(
            "Session ready. Models, graph, and memory are warm.",
            style="success",
        )
        hint = _thread_scoped_memory_hint(session)
        if hint:
            render_info(hint, style="warning")
        if session.history:
            render_info(
                f"Resumed thread {session.thread_id} with {len(session.history)} stored messages.",
                style="success",
            )

        while True:
            if pending_tail_task is not None and pending_tail_task.done():
                await _finalize_pending_turn()

            try:
                user_text = await read_user_input(
                    _prompt_toolbar_state(
                        session,
                        pending_status=_pending_tail_status(session)
                        if pending_tail_task is not None
                        else None,
                    )
                )
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
            status_stages: list[str] = []

            stream = runtime.run_turn_stream(
                thread_id=session.thread_id,
                user_id=session.owner_id(),
                message=user_text,
                channel=Channel.TEST,
                llm_client=session.llm_client,
                response_llm_client=session.response_llm_client,
            )

            try:
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
                        """Combine the spinner and response body for live rendering."""

                        return Group(status_renderable, body)

                    live.update(_stream_group(Text("", style="muted")))
                    async for event in stream:
                        if isinstance(event, StatusEvent):
                            status_stages.append(event.stage)
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
                            body = _live_preview_renderable(
                                accumulated_text,
                                thread_id=session.thread_id,
                            )
                            live.update(_stream_group(body))

                        elif isinstance(event, ChunkEvent):
                            accumulated_text += event.text
                            live.update(
                                _stream_group(
                                    _live_preview_renderable(
                                        accumulated_text,
                                        thread_id=session.thread_id,
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
            except Exception as exc:
                logger.warning("CLI turn stream failed", exc_info=True)
                render_info(
                    _recoverable_error_message("Turn failed", exc),
                    style="danger",
                )
                continue

            if response_ready_output is not None:
                pending_tail_task = asyncio.create_task(
                    _drain_turn_stream_tail(
                        stream,
                    )
                )
                if _trace_enabled(session):
                    render_turn_trace(
                        response_ready_output,
                        status_stages=status_stages,
                        pending_status=_pending_tail_status(session),
                    )
                    _consume_trace_once(session)
                render_turn_route(
                    response_ready_output,
                    pending_status=_pending_tail_status(session),
                )
                render_turn_activity(
                    response_ready_output,
                    observability_mode=session.observability_mode,
                )
                render_info(
                    _pending_tail_message(session),
                    style="muted",
                )
                continue

            if final_output is not None:
                if _trace_enabled(session):
                    render_turn_trace(
                        final_output,
                        status_stages=status_stages,
                    )
                    _consume_trace_once(session)
                render_turn_route(final_output)
                render_turn_activity(
                    final_output,
                    observability_mode=session.observability_mode,
                )
                session.last_context = await runtime.get_state(session.thread_id)
                session.history = await runtime.get_history(session.thread_id)
    finally:
        await _finalize_pending_turn()


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
) -> int:
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
        sqlite_path: Legacy SQLite file used for persisted session state.
            Deprecated for normal local development in favor of Postgres.
        memory_mode: Local memory mode ("guest" or "persistent").
        memory_sqlite_path: Legacy SQLite path for the memory store. Only
            used in persistent mode when the SQLite backend is selected.
        crisis_log_sqlite_path: Legacy SQLite path for the crisis log.
            Same persistence semantics as the memory path.

    Returns:
        Process exit code.
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
    runtime_persistence_backend: PersistenceBackend = (
        "sqlite" if is_guest_mode else settings.persistence_backend
    )
    runtime_database_url = None if is_guest_mode else settings.memory_database_url
    runtime_text_session_database_url = (
        None
        if is_guest_mode
        else settings.text_session_database_url or settings.memory_database_url
    )
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
        persistence_backend=settings.persistence_backend,
        user_id=effective_user_id,
        response_model_tier=response_model_tier,
        response_llm_client=response_llm_client,
    )

    async with AsyncExitStack() as stack:
        try:
            with console.status(
                "[accent]preparing session — warming up models and memory...[/accent]",
                spinner="dots",
            ):
                runtime = await stack.enter_async_context(
                    PersistentAgentRuntime(
                        session.sqlite_path,
                        memory_mode=runtime_memory_mode,
                        memory_backend=runtime_persistence_backend,
                        memory_database_url=runtime_database_url,
                        text_session_backend=settings.text_session_backend,
                        text_session_database_url=runtime_text_session_database_url,
                        thread_persistence_backend=runtime_persistence_backend,
                        thread_database_url=runtime_database_url,
                        crisis_log_persistence_backend=runtime_persistence_backend,
                        crisis_log_database_url=runtime_database_url,
                        session_feedback_persistence_backend=runtime_persistence_backend,
                        session_feedback_database_url=runtime_database_url,
                        memory_sqlite_path=memory_sqlite_path,
                        crisis_log_sqlite_path=crisis_log_sqlite_path,
                        default_llm_client=session.llm_client,
                        finalize_active_sessions_on_close=False,
                    )
                )
                session.history = await runtime.get_history(thread_id)
                session.last_context = await runtime.get_state(thread_id)
        except PostgresOperationalError as exc:
            _render_persistence_startup_error(
                backend=settings.persistence_backend,
                exc=exc,
            )
            return 2

        await _run_interactive_session(session, runtime)
    return 0


def main() -> int:
    """Run the OpenCouch CLI.

    Runs the interactive text CLI.

    Returns:
        Process exit code for the CLI session.
    """

    warnings.warn(
        "opencouch_cli is deprecated; prefer the Textual TUI via `uv run python -m opencouch_tui`.",
        DeprecationWarning,
        stacklevel=2,
    )
    console.print(
        Panel(
            "Legacy CLI is deprecated. Prefer the Textual TUI via "
            "[bold]uv run python -m opencouch_tui[/bold].",
            title="Deprecation Notice",
            border_style="warning",
        )
    )

    args = build_parser().parse_args()

    if args.disable_tracing:
        os.environ["OPENCOUCH_DISABLE_TRACING"] = "1"

    thread_id = args.thread_id or generate_thread_id()
    sqlite_path = str(Path(args.sqlite_path).expanduser())
    memory_sqlite_path = str(Path(args.memory_sqlite_path).expanduser())
    crisis_log_sqlite_path = str(Path(args.crisis_log_sqlite_path).expanduser())
    memory_mode = resolve_memory_mode(args.memory_mode)
    return asyncio.run(
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

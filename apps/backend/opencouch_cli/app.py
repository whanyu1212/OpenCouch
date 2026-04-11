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

from agent.memory.models import StoredSessionArc
from agent.memory.modes import MemoryMode
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
    ThreadSummary,
)
from agent.memory.procedural import (
    aget_procedural_profile,
    aput_procedural_profile,
    aset_proactive_recall,
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
        "--memory-sqlite-path",
        default=str(DEFAULT_MEMORY_DB_PATH),
        help=(
            "SQLite path for the memory store (semantic facts + episodic "
            "arcs). Only used in persistent mode. v0.8+."
        ),
    )
    parser.add_argument(
        "--crisis-log-sqlite-path",
        default=str(DEFAULT_CRISIS_LOG_DB_PATH),
        help=(
            "SQLite path for the crisis log (safety audit trail). Only "
            "used in persistent mode. v0.8+."
        ),
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


async def render_memory_status(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Render the memory layer's current state.

    Shows the memory mode, per-namespace record counts from the unified
    memory store, the crisis log record count, and (v0.7) the
    proactive-recall toggle state for the active thread.

    Async as of v0.8 because the memory store's ``arecord_count`` and
    ``anamespaces`` methods are async to support the SQLite
    implementation's connection-sharing contract. The crisis log's
    ``record_count`` is still sync — that's a different backend that
    will get its own async refactor in v0.8 Stage C.

    v0.7 Stage E: the proactive-recall row now shows the actual toggle
    state for the active thread. Previously it was a ``(phase 2+)``
    placeholder. The thread_id is used as the owner key because the
    CLI does not expose a ``--user-id`` flag yet (tracked in v0.9).

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

    # Aggregate store record counts by namespace kind. Namespaces are
    # (user_id, kind) tuples; group by the kind for a clean summary.
    counts_by_kind: dict[str, int] = {"semantic": 0, "episodic": 0, "procedural": 0}
    for namespace in await store.anamespaces():
        if len(namespace) >= 2 and namespace[1] in counts_by_kind:
            counts_by_kind[namespace[1]] += await store.arecord_count(namespace)
    total_records = await store.arecord_count()

    # Crisis log record count — always-on backend, independent of mode.
    # v0.8: the crisis log backend's record_count went async alongside
    # the memory store's, so the defensive ``getattr`` pattern now
    # looks for ``arecord_count`` and awaits it. Any backend that
    # doesn't implement the protocol method (e.g. a mock or stub)
    # falls through to the 0 default.
    crisis_log_count = 0
    arecord_count_fn = getattr(crisis_log, "arecord_count", None)
    if callable(arecord_count_fn):
        crisis_log_count = await arecord_count_fn()

    # v0.7: read the procedural profile for the active thread so the
    # recall toggle row shows the real state. Also used to show the
    # per-thread rule count (which may differ from the store-wide
    # total when multiple threads share a store backend).
    profile = await aget_procedural_profile(store, user_id=session.thread_id)

    table = Table(show_header=False, box=box.SIMPLE_HEAVY)
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="info")
    table.add_row("memory_mode", str(runtime.memory_mode))
    table.add_row("semantic facts", str(counts_by_kind["semantic"]))
    table.add_row("episodic arcs", str(counts_by_kind["episodic"]))
    table.add_row("procedural rules", str(counts_by_kind["procedural"]))
    table.add_row("total memory records", str(total_records))
    table.add_row("crisis log events", str(crisis_log_count))
    # v0.7 Stage E: real proactive-recall state from the profile.
    recall_state = "on" if profile.proactive_recall_enabled else "off"
    table.add_row("proactive recall", recall_state)
    # Placeholders for fields that land in later phases — shown so the
    # command shape stays stable as features are added.
    table.add_row("last consolidation", "(phase 4)")
    console.print(
        Panel(
            table,
            title="[primary]Memory Status[/primary]",
            subtitle="[muted]what the memory layer is holding[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )


async def _collect_records_by_kind(
    runtime: PersistentAgentRuntime,
    *,
    kind: str,
) -> list[tuple[str, dict[str, object]]]:
    """Collect all records across every namespace with the given kind.

    The store is namespaced by ``(user_id, kind)``, so this iterates
    every namespace whose second tuple element matches ``kind`` and
    returns the records in insertion order. Used by ``render_memory_list``
    to gather semantic and episodic records separately.

    v0.8 rewrite: previously reached into ``store._buckets`` directly,
    which only worked for :class:`OpenCouchMemoryStore`. Now uses
    ``asearch(ns, query=None, limit=<large>)`` which is part of the
    :class:`MemoryStore` protocol and works for both the in-memory
    and SQLite implementations. The ``query=None`` branch returns all
    records in insertion order, which is exactly what the CLI wants
    for a chronological listing.
    """

    records: list[tuple[str, dict[str, object]]] = []
    store = runtime.memory_store
    for namespace in await store.anamespaces():
        if len(namespace) < 2 or namespace[1] != kind:
            continue
        # limit=1000 is a defensive cap; for v0.8 we don't expect
        # any single user to have more records than this, and if
        # they do the CLI would need pagination anyway (v0.9 work).
        namespace_records = await store.asearch(namespace, query=None, limit=1000)
        for record in namespace_records:
            records.append((record.key, record.value))
    return records


def _format_entity_identifier(entity: object) -> str:
    """Return the ``identifier`` field of a serialized :class:`EntityRef`.

    The stored record's ``subject`` and ``object`` fields are dicts
    with ``type`` and ``identifier`` keys (from
    ``EntityRef.model_dump()``). This helper extracts the identifier
    for table rendering, falling back to ``"?"`` when the shape is
    wrong or the identifier is missing. Defensive against schema
    drift and malformed records.
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

    v0.8 addition: the table now includes an ``object`` column
    showing the identifier field of each record's ``object``
    (the target of the predicate — e.g., the named person for a
    ``KNOWS`` fact, the coping strategy for a ``USES`` fact). This
    closes a rendering gap surfaced during v0.8 dogfood: when a
    single turn produces two facts with identical evidence quotes
    but different objects (e.g., "I take fluoxetine and vyvanse
    daily" → one USES fact per medication), the old table showed
    two rows that looked identical. Adding the object column makes
    the distinction visible. The subject column is intentionally
    omitted because in practice it's almost always the user; showing
    it would add visual noise without distinguishing signal.
    """

    table = Table(
        show_header=True,
        header_style="primary",
        box=box.SIMPLE_HEAVY,
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
            title=f"[primary]Memory List (semantic)[/primary] "
            f"[muted]— {len(records)} record(s)[/muted]",
            subtitle="[muted]what the extractor has written so far[/muted]",
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

    Shipped with v0.4 alongside the session summarizer. Each row shows
    the session date, turn count, themes, mood arc (opened → closed),
    crisis level (if any), and a truncated summary. Full summaries for
    long arcs are rendered below the table, same pattern as the
    semantic long-quote footer.
    """

    table = Table(
        show_header=True,
        header_style="primary",
        box=box.SIMPLE_HEAVY,
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
        # Date derived from ended_at — just the YYYY-MM-DD prefix.
        ended_at = str(value.get("ended_at", ""))
        date_display = ended_at[:10] if len(ended_at) >= 10 else "—"

        turn_count = str(value.get("turn_count", "?"))

        themes_list = value.get("primary_themes") or []
        themes_display = (
            ", ".join(str(t) for t in themes_list)  # type: ignore[union-attr]
            if themes_list
            else "—"
        )

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
            title=f"[primary]Memory List (episodic)[/primary] "
            f"[muted]— {len(records)} session arc(s)[/muted]",
            subtitle="[muted]what the summarizer has written per session[/muted]",
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

    v0.7 Stage E addition. Unlike semantic facts (one record per
    fact) and episodic arcs (one record per session), procedural
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
    """

    table = Table(
        show_header=True,
        header_style="primary",
        box=box.SIMPLE_HEAVY,
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

        # Evidence is a list[str] — join with " | " for display and
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
            title="[primary]Memory List (procedural)[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def _render_memory_list_empty_state() -> None:
    """Render the educational empty-state panel for `/memory list`.

    Shown when both semantic and episodic namespaces are empty. The
    message explains what each layer captures and when to expect
    writes, so operators know this is expected behavior rather than
    a bug — especially important when v0.3 conservative extraction
    is rejecting most turns.
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
            title="[primary]Memory List[/primary]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


async def render_memory_list(runtime: PersistentAgentRuntime) -> None:
    """Render every memory record (semantic + episodic) in browsable tables.

    Shipped in v0.3.1 as a semantic-only dogfood-observability tool.
    Extended in v0.4 to render episodic session arcs in a second table
    alongside the semantic facts. Async as of v0.8 because record
    collection now goes through the async ``MemoryStore`` protocol
    methods so it works with the SQLite-backed implementation.

    Scope:
    - Read-only. Mutation commands (``/memory forget``, ``/memory clear``)
      are scoped to v0.9 alongside the full CLI memory suite.
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
    """

    semantic_records = await _collect_records_by_kind(runtime, kind="semantic")
    episodic_records = await _collect_records_by_kind(runtime, kind="episodic")

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

    v0.7 Stage E. Toggles the ``proactive_recall_enabled`` flag on
    the active thread's procedural profile and shows a confirmation
    message. When flipping from OFF to ON, also renders the
    first-run explanation from ``schema.yaml §6 retrieval
    proactive_recall.opt_in_confirmation_example`` so the user
    understands what changes.

    Behavior:

    - ``enable=True`` + current OFF → write + show explanation
    - ``enable=True`` + current ON  → show "already on" message,
      no write
    - ``enable=False`` + current ON → write + brief confirmation
    - ``enable=False`` + current OFF → show "already off" message,
      no write

    The thread_id is used as the owner key because the CLI does not
    expose a ``--user-id`` flag yet (tracked in v0.9). Every thread
    has its own independent recall preference.

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the thread_id.
        enable: Target state (True = on, False = off).
    """

    store = runtime.memory_store
    # Read current state first so we can detect no-op calls and
    # pick the right confirmation message.
    current = await aget_procedural_profile(store, user_id=session.thread_id)
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
    await aset_proactive_recall(store, user_id=session.thread_id, enabled=enable)

    if enable:
        # Flipping OFF → ON. Show the first-run explanation per
        # schema.yaml §6 retrieval proactive_recall.opt_in_confirmation_example.
        # This fires every time the user goes off→on, not just the
        # very first time — simpler than tracking a "has seen
        # explanation" flag and arguably better UX because users who
        # rarely touch the feature get a refresher.
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
                title="[primary]Proactive recall: ON[/primary]",
                border_style="panel",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    # Flipping ON → OFF. Brief confirmation — no need for an
    # explanation because "off" is the schema default, and users who
    # flip off know why they're doing it.
    console.print(
        Panel(
            "[success]Proactive recall is now OFF for this thread.[/success]\n\n"
            "[info]I still remember what you tell me and use it to inform how "
            "I respond, but I won't proactively reference past sessions or "
            "earlier statements unless you ask me about them. Style rules "
            "you've asked for are still applied — the toggle only affects "
            "whether I mention past memory out loud.[/info]",
            title="[primary]Proactive recall: OFF[/primary]",
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

    v0.7 Stage E. Deletes one procedural rule from the active
    thread's profile by its 1-indexed position (the same index
    that ``/memory list rules`` displays). Prompts for y/n
    confirmation before deleting. The store write is atomic via the
    profile-as-document shape — the rule is removed from the list
    and the whole profile is written back.

    Args:
        runtime: Active persistent runtime, for the memory store.
        session: Active CLI session, for the thread_id.
        index_str: The raw argument the user typed after
            ``/memory forget rule``. Parsed to an int here; invalid
            or out-of-range inputs produce a warning without
            touching the store.
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
    profile = await aget_procedural_profile(store, user_id=session.thread_id)

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

    # Show the rule being deleted + y/n prompt. Rules are short
    # enough (≤280 chars) to inline in the prompt text.
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
        choices=["y", "n", ""],
        default="n",
        show_choices=False,
        show_default=False,
    )
    if answer.strip().lower() != "y":
        render_info("Cancelled — no rules deleted.", style="info")
        return

    # Confirmed: remove the rule and write the profile back. This is
    # a load → mutate → put round-trip against the profile, not an
    # individual record delete, because procedural memory is stored
    # as a single profile document.
    profile.rules.pop(index_1based - 1)
    await aput_procedural_profile(store, user_id=session.thread_id, profile=profile)
    render_info(
        f"Deleted rule #{index_1based}. "
        f"{len(profile.rules)} rule(s) remaining for this thread.",
        style="success",
    )


async def render_memory_list_rules(
    runtime: PersistentAgentRuntime,
    session: RunnerSession,
) -> None:
    """Render the active thread's procedural rules in a browsable table.

    v0.7 Stage E. Unlike ``render_memory_list`` which aggregates
    semantic and episodic records across all threads stored in the
    memory store, this function reads the procedural profile for the
    CURRENT thread only. That's because rules are namespaced per
    user, and in the CLI the thread_id is the effective user id.

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
    """

    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
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

    Shipped with v0.4 as the closing farewell when ``/end`` triggers the
    session summarizer. Shows the user the exact summary that was saved,
    so they know what will be remembered — and can correct it later
    (via /memory forget in v0.9 or by telling the agent directly).

    The summary display uses the structured fields from StoredSessionArc
    rather than just the prose summary, because the structure IS the
    signal: mood arc, themes, open loops, and crisis level all say
    something about what the session was about beyond what the summary
    paragraph captures.
    """

    table = Table(show_header=False, box=box.SIMPLE_HEAVY)
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="info")

    # Summary text goes at the top — it's the main thing the user
    # wants to see and confirm.
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
            title="[primary]Session Summary[/primary] [muted]— saved to memory[/muted]",
            subtitle="[muted]this is what I'll remember next time we talk[/muted]",
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

    table = Table(show_header=True, header_style="primary", box=box.SIMPLE_HEAVY)
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
        "/memory list",
        "List every semantic fact and episodic arc stored for this thread.",
    )
    table.add_row(
        "/memory list rules",
        "List the procedural style rules the writer has recorded for this thread.",
    )
    table.add_row(
        "/memory recall on|off",
        "Toggle whether the agent proactively references past memory "
        "content in replies. Style rules are always applied regardless.",
    )
    table.add_row(
        "/memory forget rule <n>",
        "Delete one procedural rule by its 1-indexed position from /memory list rules.",
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


async def _summarize_and_render(
    session: RunnerSession,
    runtime: PersistentAgentRuntime,
) -> None:
    """Call the runtime's session summarizer and render the result.

    Shared helper for the ``/end`` and ``/exit`` commands — both
    trigger a session-end summary, but they have slightly different
    UX paths (``/end`` is silent-then-farewell, ``/exit`` prompts
    for consent first). Both end up here once the decision to
    summarize is made.

    When the summarizer returns a stored arc, we render it via
    ``render_session_summary`` so the user sees exactly what got
    saved. When it returns None — incognito mode, no LLM, or the LLM
    judged the session too thin to summarize — we render a plain
    farewell instead, with a reason-hinting line so the operator
    knows why nothing was saved.

    Failures inside the summarizer degrade silently (see
    ``run_summarize_session`` for the contract); this wrapper does
    NOT try to catch additional errors. A silent None return is
    indistinguishable from "ran but nothing to save", which is the
    right user-visible behavior.
    """

    try:
        stored_arc = await runtime.end_session(
            session.thread_id,
            llm_client=session.llm_client,
        )
    except Exception:
        # Belt-and-suspenders: the summarizer itself catches its own
        # errors, but if something unexpected escapes we still want
        # the /end flow to exit cleanly. Log to stderr via render_info
        # rather than crashing the CLI.
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
        # v0.4: offer to summarize the session before exiting. The user
        # can decline (and then gets only the farewell); declining means
        # the session's content is NOT saved to episodic memory. The
        # prompt defaults to Y because saving a summary is the safer
        # path — it's easier to /memory forget later than to reconstruct
        # a lost summary.
        save = Prompt.ask(
            "[muted]Save a session summary before exiting?[/muted] [accent][Y/n][/accent]",
            choices=["y", "n", ""],
            default="y",
            show_choices=False,
            show_default=False,
        )
        if save.strip().lower() != "n":
            await _summarize_and_render(session, runtime)
        return False

    if command == "/end":
        # v0.4: trigger the session summarizer before exiting. Unlike
        # /exit this does NOT prompt — the user explicitly chose /end,
        # which means they want to wrap up the session properly. The
        # summarizer runs, the resulting arc is rendered as a farewell,
        # and the loop exits.
        await _summarize_and_render(session, runtime)
        return False

    if command == "/memory":
        # v0.3.1 added /memory status and /memory list.
        # v0.7 adds /memory list rules, /memory recall on|off, and
        # /memory forget rule <n>.
        # Full /memory forget/clear suite for semantic + episodic is
        # still scoped to v0.9.
        if len(args) == 0 or args[0] == "status":
            await render_memory_status(runtime, session)
            return True
        if args[0] == "list":
            # /memory list               — semantic + episodic
            # /memory list rules         — procedural rules for this thread
            if len(args) >= 2 and args[1] == "rules":
                await render_memory_list_rules(runtime, session)
                return True
            await render_memory_list(runtime)
            return True
        if args[0] == "recall":
            # /memory recall on|off — toggle proactive recall for the
            # active thread. When flipping off→on, the handler shows
            # the first-run explanation from schema.yaml.
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
            # /memory forget rule <n> — delete one procedural rule
            # by its 1-indexed position. Prompts for y/n confirmation.
            # /memory forget fact|session — v0.9 scope; show a
            # helpful "not yet" message rather than a generic error.
            if len(args) >= 2 and args[1] == "rule":
                if len(args) < 3:
                    render_info(
                        "Usage: /memory forget rule <n>",
                        style="warning",
                    )
                    return True
                await render_memory_forget_rule(runtime, session, index_str=args[2])
                return True
            if len(args) >= 2 and args[1] in ("fact", "session"):
                render_info(
                    f"/memory forget {args[1]} is not yet available "
                    "(scoped to v0.9). Rules can be deleted with "
                    "/memory forget rule <n>.",
                    style="warning",
                )
                return True
            render_info(
                "Usage: /memory forget rule <n>",
                style="warning",
            )
            return True
        render_info(
            "Unknown /memory subcommand. Available in v0.7: "
            "status, list, list rules, recall on|off, forget rule <n>",
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
    memory_sqlite_path: str = str(DEFAULT_MEMORY_DB_PATH),
    crisis_log_sqlite_path: str = str(DEFAULT_CRISIS_LOG_DB_PATH),
) -> None:
    """Run the interactive CLI loop.

    Args:
        mode: Requested runtime mode for model resolution.
        thread_id: Stable thread identifier for the local conversation.
        sqlite_path: SQLite file used for persisted thread checkpoints.
        memory_mode: Local memory mode ("guest" or "persistent").
        memory_sqlite_path: v0.8 SQLite path for the memory store
            (semantic + episodic records). Only used in persistent
            mode; incognito ignores it and uses in-memory backing.
        crisis_log_sqlite_path: v0.8 SQLite path for the crisis log.
            Same persistence semantics as the memory path.

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
        memory_sqlite_path=memory_sqlite_path,
        crisis_log_sqlite_path=crisis_log_sqlite_path,
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
    memory_sqlite_path = str(Path(args.memory_sqlite_path).expanduser())
    crisis_log_sqlite_path = str(Path(args.crisis_log_sqlite_path).expanduser())
    memory_mode = resolve_memory_mode(args.memory_mode)
    asyncio.run(
        chat_loop(
            args.mode,
            thread_id=thread_id,
            sqlite_path=sqlite_path,
            memory_mode=memory_mode,
            memory_sqlite_path=memory_sqlite_path,
            crisis_log_sqlite_path=crisis_log_sqlite_path,
        )
    )
    return 0

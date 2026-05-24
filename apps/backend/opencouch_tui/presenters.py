"""Shared presenter data for terminal UI status and diagnostics."""

from __future__ import annotations

import textwrap
from typing import Any, Callable

from agent.models import (
    AgentOutput,
    CrisisAssessment,
    MessageRole,
    ResponseCategory,
    friendly_stage,
)
from agent.observability.routing_trace import routing_trace_from_diagnostics
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from opencouch_tui.commands import help_commands


def _count_user_turns(session: Any) -> int:
    """Return the number of user-authored messages in session history."""

    return sum(1 for message in session.history if message.role == MessageRole.USER)


def owner_scope_display(session: Any) -> str:
    """Return a compact owner-scope label for status surfaces."""

    if session.memory_mode == "guest":
        return "none (guest mode)"
    if getattr(session, "user_id", None):
        return f"{session.user_id} (from --user-id)"
    return f"{session.thread_id} (thread-scoped)"


def session_status_rows(session: Any) -> list[tuple[str, str]]:
    """Return normalized status rows for rich/table rendering."""

    rows = [
        ("thread id", session.thread_id),
        ("owner id", owner_scope_display(session)),
        ("memory mode", str(session.memory_mode)),
    ]
    if session.memory_mode == "guest":
        rows.append(("persistence", "ephemeral"))
    else:
        rows.append(("persistence", str(session.persistence_backend)))
        if getattr(session, "persistence_backend", None) == "sqlite":
            rows.append(("sqlite path", str(session.sqlite_path)))
    rows.extend(
        [
            ("requested mode", session.requested_mode),
            ("resolved mode", session.resolved_mode),
            (
                "llm client",
                "enabled"
                if getattr(session, "llm_client", None) is not None
                else "disabled",
            ),
            ("response tier", session.response_model_tier),
            ("trace mode", session.trace_mode),
            ("ui mode", session.ui_mode),
            ("verbosity", session.observability_mode),
            ("prompt theme", session.prompt_theme),
            (
                "response llm",
                "enabled"
                if getattr(session, "response_llm_client", None) is not None
                else "disabled",
            ),
            ("turns", str(_count_user_turns(session))),
            ("messages", str(len(session.history))),
            (
                "context snapshot",
                "available" if session.last_context is not None else "none",
            ),
        ]
    )
    return rows


def session_status_command_lines(session: Any) -> list[str]:
    """Return the lightweight `/status` lines used by the TUI command output."""

    return [
        f"mode: {session.resolved_mode}",
        f"memory: {session.memory_mode}",
        f"thread: {session.thread_id}",
        f"owner: {session.owner_id}",
        f"response: {session.response_model_tier}",
        f"messages: {len(session.history)}",
    ]


def session_status_bar_parts(session: Any, *, active_theme: str) -> list[str]:
    """Return the compact status-bar segments for the TUI."""

    return [
        f"mode {session.resolved_mode}",
        f"memory {session.memory_mode}",
        f"theme {active_theme}",
        f"thread {session.thread_id}",
        f"owner {session.owner_id}",
        f"response {session.response_model_tier}",
    ]


def runtime_doctor_rows(
    session: Any,
    *,
    verbose: bool = False,
) -> list[tuple[str, str, str]]:
    """Return normalized runtime-doctor rows for rendering."""

    rows: list[tuple[str, str, str]] = []

    if getattr(session, "llm_client", None) is None:
        rows.append(
            (
                "llm",
                "smoke",
                "No control LLM is configured; turns use deterministic smoke mode.",
            )
        )
    else:
        rows.append(
            (
                "llm",
                "ok",
                f"Control LLM enabled through {session.resolved_mode} mode.",
            )
        )

    if getattr(session, "response_llm_client", None) is not None:
        rows.append(
            (
                "response model",
                "ok",
                f"Dedicated {session.response_model_tier} response tier is enabled.",
            )
        )
    elif getattr(session, "llm_client", None) is not None:
        rows.append(
            (
                "response model",
                "shared",
                "Responses fall back to the control LLM client.",
            )
        )
    else:
        rows.append(
            (
                "response model",
                "skipped",
                "No response LLM is used in deterministic smoke mode.",
            )
        )

    if session.memory_mode == "guest":
        rows.append(
            (
                "persistence",
                "ephemeral",
                "Guest mode keeps thread and memory state in process-local storage.",
            )
        )
    else:
        rows.append(
            (
                "persistence",
                "ok",
                f"Persistent mode is using the {session.persistence_backend} backend.",
            )
        )

    if (
        session.memory_mode == "persistent"
        and getattr(session, "user_id", None) is None
    ):
        rows.append(
            (
                "owner scope",
                "warn",
                "Memory is thread-scoped; pass --user-id for cross-thread dogfooding.",
            )
        )
    else:
        rows.append(("owner scope", "ok", owner_scope_display(session)))

    rows.append(
        (
            "turn recovery",
            "ok",
            "Turn and response-tail failures are reported without closing the CLI.",
        )
    )

    if verbose:
        rows.extend(
            [
                ("requested mode", "info", session.requested_mode),
                ("resolved mode", "info", session.resolved_mode),
                ("response tier", "info", session.response_model_tier),
                ("trace mode", "info", session.trace_mode),
                ("ui mode", "info", session.ui_mode),
                ("prompt theme", "info", session.prompt_theme),
                ("verbosity", "info", session.observability_mode),
                (
                    "context snapshot",
                    "info",
                    "available" if session.last_context is not None else "none",
                ),
            ]
        )

    return rows


def render_help(*, console: Console) -> None:
    """Render the available slash commands grouped by category."""

    category_titles = {
        "session": "session",
        "display": "display",
        "memory": "memory",
        "threads": "threads",
        "runtime": "runtime",
        "debug": "debug",
    }
    category_order = ("session", "display", "memory", "threads", "runtime", "debug")
    grouped: dict[str, list[tuple[str, str]]] = {key: [] for key in category_order}
    for command in help_commands():
        grouped.setdefault(command.category, []).append(
            (command.display, command.description)
        )

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("category", style="hint", no_wrap=True)
    table.add_column("command", style="accent", no_wrap=True)
    table.add_column("description", style="info")
    for category in category_order:
        rows = grouped.get(category, [])
        if not rows:
            continue
        for display, description in rows:
            table.add_row(category_titles.get(category, category), display, description)
    console.print(
        Panel(
            table,
            title="[muted]command reference[/muted]",
            subtitle="[hint]type / to search commands from the prompt[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_keys(*, console: Console) -> None:
    """Render keyboard shortcuts and prompt usage hints."""

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("shortcut", style="accent", no_wrap=True)
    table.add_column("action", style="info")
    table.add_row("/", "Open slash-command completions at prompt start.")
    table.add_row("↑ / ↓", "Navigate input history.")
    table.add_row("Tab", "Accept highlighted completion.")
    table.add_row("Ctrl+L", "Clear prompt surface.")
    table.add_row("Enter", "Submit current input.")

    console.print(
        Panel(
            table,
            title="[muted]keyboard shortcuts[/muted]",
            subtitle="[hint]type /help for full command reference[/hint]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_onboarding(*, console: Console) -> None:
    """Render a one-time quick-start guide for first prompt."""

    hint = Text()
    hint.append("  Type ", style="hint")
    hint.append("/", style="accent")
    hint.append(" for commands", style="hint")
    hint.append("  ·  ", style="panel")
    hint.append("start typing to talk", style="hint")
    console.print(hint)
    console.print()


def render_header(
    mode: str,
    thread_id: str,
    memory_mode: str,
    *,
    console: Console,
    user_id: str | None = None,
    response_model_tier: str | None = None,
) -> None:
    """Render the startup/session header."""

    console.print()

    title = Text()
    title.append("  OpenCouch", style="brand")
    title.append("  ·  ", style="panel")
    title.append("text agent", style="info")
    console.print(title)

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
    elif memory_mode == "persistent":
        identity_parts.append("[muted]owner[/muted] [warning]thread-scoped[/warning]")
    console.print(Text.from_markup("   " + "  [panel]·[/panel]  ".join(identity_parts)))

    hint = Text()
    hint.append("/", style="accent")
    hint.append(" commands", style="hint")
    hint.append("  ·  ", style="panel")
    hint.append("/status", style="accent")
    hint.append(" session", style="hint")
    hint.append("  ·  ", style="panel")
    hint.append("exit", style="accent")
    hint.append(" to stop", style="hint")
    hint.pad_left(3)
    console.print(hint)
    console.print()


def left_rail_text(value: str, *, style: str, console: Console) -> Text:
    """Render multiline text against a subtle terminal left rail."""

    body = Text()
    wrote_line = False
    max_width = max(24, min(console.width - 6, 88))
    paragraphs = value.splitlines() or [""]
    for paragraph_index, paragraph in enumerate(paragraphs):
        wrapped = textwrap.wrap(paragraph, width=max_width) if paragraph else [""]
        for line in wrapped:
            if wrote_line:
                body.append("\n")
            body.append("  │ ", style="panel")
            body.append(line, style=style)
            wrote_line = True
        if paragraph_index < len(paragraphs) - 1:
            body.append("\n")
            body.append("  │", style="panel")
            wrote_line = True
    return body


def normal_reply_renderable(
    response_text: str,
    *,
    thread_id: str | None,
    turn_count: int | None,
    response_style: str | None,
    therapeutic_approach: str | None,
    console: Console,
) -> Group:
    """Build lightweight chrome for a normal assistant reply."""

    meta = Text("  assistant", style="success")
    meta_parts: list[tuple[str, str]] = []
    if thread_id:
        meta_parts.append(("thread", thread_id))
    if turn_count is not None:
        meta_parts.append(("turn", str(turn_count)))
    if response_style:
        style_label = response_style
        if therapeutic_approach and therapeutic_approach != "none":
            style_label += f" / {therapeutic_approach}"
        meta_parts.append(("style", style_label))

    for label, value in meta_parts:
        meta.append("  ·  ", style="panel")
        meta.append(label, style="muted")
        meta.append(" ", style="panel")
        meta.append(value, style="info")

    body = left_rail_text(response_text, style="info", console=console)
    return Group(meta, body)


def response_panel_renderable(
    output: AgentOutput,
    *,
    console: Console,
    thread_id: str | None = None,
    turn_count: int | None = None,
) -> Panel | Group:
    """Build the terminal assistant-response renderable for one turn."""

    is_crisis = output.response_type.value == "crisis"
    if not is_crisis:
        return normal_reply_renderable(
            output.response_text,
            thread_id=thread_id,
            turn_count=turn_count,
            response_style=output.response_style,
            therapeutic_approach=output.therapeutic_approach,
            console=console,
        )

    title = "[danger]  crisis  [/danger]"
    border = "danger"
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


def render_response(
    response_text: str,
    *,
    console: Console,
    is_crisis: bool,
    thread_id: str | None = None,
    turn_count: int | None = None,
) -> None:
    """Render the assistant reply inside a styled panel."""

    output = AgentOutput(
        response_text=response_text,
        response_type=(
            ResponseCategory.CRISIS if is_crisis else ResponseCategory.THERAPEUTIC
        ),
        crisis=CrisisAssessment(),
        response_style="crisis" if is_crisis else "support",
        diagnostics={},
    )
    console.print(
        response_panel_renderable(
            output,
            console=console,
            thread_id=thread_id,
            turn_count=turn_count,
        )
    )


def render_stage_timings(
    diagnostics: dict,
    memory_deltas: dict,
    *,
    console: Console,
) -> None:
    """Render the per-turn stage timings + memory-write table."""

    if not diagnostics and not memory_deltas:
        return

    timing_table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    timing_table.add_column("stage", style="hint", no_wrap=True)
    timing_table.add_column("time (ms)", style="info", justify="right", no_wrap=True)
    timing_table.add_column("writes", style="accent", justify="right", no_wrap=True)
    timing_table.add_column("store Δ", style="success", justify="right", no_wrap=True)

    def _fmt_ms(key: str) -> str:
        val = diagnostics.get(key)
        if val is None:
            return "-"
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return "-"

    def _fmt_delta(key: str) -> str:
        val = memory_deltas.get(key)
        if val is None:
            return "-"
        return f"+{val}" if val > 0 else str(val)

    timing_table.add_row("load_memory", _fmt_ms("load_memory_ms"), "-", "-")
    timing_table.add_row("crisis_gate", _fmt_ms("crisis_gate_ms"), "-", "-")
    timing_table.add_row("semantic_memory", "-", "-", _fmt_delta("semantic"))
    timing_table.add_row("procedural_memory", "-", "-", _fmt_delta("procedural"))
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


def render_meta(
    *,
    console: Console,
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
                f"[primary]{response_style or '-'}[/primary]",
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
        render_stage_timings(diag, deltas, console=console)
        console.print()


def route_label(output: AgentOutput) -> str:
    """Return the user-facing route label for one output."""

    route = output.response_style or "unknown"
    if output.therapeutic_approach and output.therapeutic_approach != "none":
        route = f"{route} / {output.therapeutic_approach}"
    return route


def tool_badges(output: AgentOutput) -> list[str]:
    """Return compact tool badges derived from turn diagnostics."""

    diagnostics = output.diagnostics or {}
    badges: list[str] = []

    if diagnostics.get("openai_therapeutic_skill_tool_calls"):
        badges.append("response-style")
    if diagnostics.get("openai_memory_tool_calls"):
        badges.append("memory")
    if diagnostics.get("openai_grounded_tool_calls"):
        badges.append("grounded lookup")

    return badges


def tool_activity_lines(output: AgentOutput) -> list[str]:
    """Return verbose tool-activity lines derived from diagnostics."""

    diagnostics = output.diagnostics or {}
    lines: list[str] = []

    therapeutic_calls = diagnostics.get("openai_therapeutic_skill_tool_calls") or []
    if therapeutic_calls:
        style = str(
            diagnostics.get("openai_therapeutic_skill_response_style") or ""
        ).strip()
        detail = f" → {style}" if style else ""
        lines.append(f"{therapeutic_calls[-1]}{detail}")

    memory_calls = diagnostics.get("openai_memory_tool_calls") or []
    lines.extend(str(call) for call in memory_calls if call)

    grounded_calls = diagnostics.get("openai_grounded_tool_calls") or []
    lines.extend(str(call) for call in grounded_calls if call)

    return lines


def render_turn_activity(
    output: AgentOutput,
    *,
    console: Console,
    observability_mode: str,
) -> None:
    """Render tool activity after the route line."""

    badges = tool_badges(output)
    if observability_mode == "compact":
        if not badges:
            return
        console.print(
            Text.from_markup(
                "   [muted]tools[/muted] "
                + "  [panel]·[/panel]  ".join(
                    f"[accent]{badge}[/accent]" for badge in badges
                )
            )
        )
        console.print()
        return

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")

    table.add_row("route", route_label(output))
    if badges:
        table.add_row("badges", ", ".join(badges))

    lines = tool_activity_lines(output)
    if lines:
        table.add_row("activity", "\n".join(f"• {line}" for line in lines))

    diagnostics = output.diagnostics or {}
    triage_confidence = str(diagnostics.get("openai_triage_confidence") or "").strip()
    tentative_route = str(
        diagnostics.get("openai_triage_tentative_route") or ""
    ).strip()
    if triage_confidence:
        triage_detail = triage_confidence
        if tentative_route:
            triage_detail = f"{triage_detail}; tentative {tentative_route}"
        table.add_row("triage", triage_detail)

    if output.crisis.needs_clarification:
        table.add_row("state", "awaiting safety clarification")
    elif output.crisis.needs_crisis_response:
        table.add_row("state", "crisis response active")

    console.print(
        Panel(
            table,
            title="[muted]turn activity[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_turn_route(
    output: AgentOutput,
    *,
    console: Console,
    pending_status: str | None = None,
) -> None:
    """Render a compact routing summary for one completed response."""

    crisis = output.crisis
    if crisis.needs_crisis_response:
        safety_label = "crisis"
        safety_style = "danger"
    elif crisis.needs_clarification:
        safety_label = "check"
        safety_style = "warning"
    elif crisis.level >= 1:
        safety_label = "distress"
        safety_style = "warning"
    else:
        safety_label = "normal"
        safety_style = "success"

    diagnostics = output.diagnostics or {}
    latency = diagnostics.get("turn_total_ms")
    try:
        latency_label = f"{float(latency):.0f}ms" if latency is not None else None
    except (TypeError, ValueError):
        latency_label = None

    parts: list[tuple[str, str]] = [
        ("   route ", "muted"),
        (route_label(output), "primary"),
        ("  ·  safety ", "panel"),
        (safety_label, safety_style),
    ]
    if latency_label:
        parts.extend([("  ·  ", "panel"), (latency_label, "accent")])
    if pending_status:
        parts.extend([("  ·  ", "panel"), (pending_status, "warning")])

    console.print()
    console.print(Text.assemble(*parts))
    console.print()


def fallback_routing_trace_entries(output: AgentOutput) -> list[dict[str, str]]:
    """Build trace entries when diagnostics have no structured trace yet."""

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
            "confidence": crisis.confidence,
        },
        {
            "stage": "dispatch",
            "decision": route_decision,
            "reason": "Final response route from agent output.",
        },
    ]


def clip_trace_text(value: str, max_length: int) -> str:
    """Clip a trace label for fixed-width diagram columns."""

    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 1]}..."


def render_turn_trace(
    output: AgentOutput,
    *,
    console: Console,
    status_stages: list[str] | None = None,
    pending_status: str | None = None,
) -> None:
    """Render the optional routing trace diagram for one turn."""

    entries: list[dict[str, str]] = [
        {key: str(value) for key, value in entry.items()}
        for entry in routing_trace_from_diagnostics(output.diagnostics)
    ]
    if not entries:
        entries = fallback_routing_trace_entries(output)

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("stage", style="hint", no_wrap=True)
    table.add_column("decision", style="primary")
    table.add_column("source", style="accent")
    table.add_column("reason", style="info")

    for entry in entries:
        stage = clip_trace_text(entry.get("stage", "-"), 11)
        decision = clip_trace_text(entry.get("decision", "-"), 22)
        reason = entry.get("reason") or "-"
        source = entry.get("source") or "-"
        confidence = entry.get("confidence")
        if confidence:
            source = f"{source} / {confidence}"
        table.add_row(stage, decision, source, reason)

    note_parts: list[str] = []
    if status_stages:
        unique_stages = list(dict.fromkeys(status_stages))
        stage_labels = " -> ".join(friendly_stage(stage) for stage in unique_stages)
        note_parts.append(f"stages {stage_labels}")
    if pending_status:
        note_parts.append(f"tail {pending_status}")

    renderables: list[Table | Text] = [table]
    if note_parts:
        renderables.append(Text("  " + "  ·  ".join(note_parts), style="hint"))

    console.print()
    console.print(
        Panel(
            Group(*renderables),
            title="[muted]routing trace[/muted]",
            subtitle="[hint]/trace off hides this[/hint]",
            border_style="panel",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()


def render_status(session: Any, *, console: Console) -> None:
    """Render current runner status with a provided Rich console."""

    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column(style="hint", no_wrap=True)
    table.add_column(style="info")
    for label, value in session_status_rows(session):
        table.add_row(label, value)
    console.print(
        Panel(
            table,
            title="[muted]session status[/muted]",
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_doctor(session: Any, *, console: Console, verbose: bool = False) -> None:
    """Render runtime readiness checks for the active session."""

    table = Table(show_header=True, header_style="muted", box=box.SIMPLE)
    table.add_column("check", style="hint", no_wrap=True)
    table.add_column("status", style="accent", no_wrap=True)
    table.add_column("detail", style="info")

    for check, status, detail in runtime_doctor_rows(session, verbose=verbose):
        table.add_row(check, status, detail)

    console.print(
        Panel(
            table,
            title="[muted]runtime doctor[/muted]",
            subtitle="[hint]verbose diagnostics[/hint]" if verbose else None,
            border_style="panel",
            box=box.ROUNDED,
        )
    )
    console.print()


def render_history(session: Any, *, console: Console, limit: int = 6) -> None:
    """Render the most recent transcript entries."""

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


def render_threads(
    threads: list[Any],
    *,
    active_thread_id: str,
    console: Console,
    info_renderer: Callable[..., None] | None = None,
) -> None:
    """Render a compact table of persisted thread summaries."""

    if not threads:
        if info_renderer is not None:
            info_renderer("No persisted threads found.", style="warning")
            return
        console.print("  [warning]│[/warning] No persisted threads found.")
        console.print()
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

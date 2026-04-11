"""Tests for interactive CLI thread-management commands."""

import pytest

from agent.models import Message, MessageRole
from opencouch_cli.app import RunnerSession, handle_command


class FakeRuntime:
    """Minimal runtime stub for CLI command tests."""

    def __init__(self) -> None:
        self.states = {
            "thread-a": {"progress": {"turn_count": 2}, "transcript": []},
            "thread-b": {"progress": {"turn_count": 1}, "transcript": []},
        }
        self.histories = {
            "thread-a": [
                Message(role=MessageRole.USER, content="first"),
                Message(role=MessageRole.ASSISTANT, content="reply"),
            ],
            "thread-b": [
                Message(role=MessageRole.USER, content="other"),
                Message(role=MessageRole.ASSISTANT, content="reply"),
            ],
        }
        self.thread_summaries = []
        # v0.4: end_session tracking. Tests can set
        # ``end_session_returns`` to control what the fake returns, and
        # ``end_session_calls`` records invocations for assertions.
        self.end_session_returns: object | None = None
        self.end_session_calls: list[str] = []

    async def get_state(self, thread_id: str):
        return self.states.get(thread_id)

    async def get_history(self, thread_id: str):
        return list(self.histories.get(thread_id, []))

    async def list_threads(self, *, limit: int = 20):
        return self.thread_summaries[:limit]

    async def reset_thread(self, thread_id: str) -> None:
        self.states.pop(thread_id, None)
        self.histories.pop(thread_id, None)

    async def end_session(
        self,
        thread_id: str,
        *,
        llm_client=None,
    ):
        """v0.4 stub: record the call and return the canned result."""

        self.end_session_calls.append(thread_id)
        return self.end_session_returns


def _session() -> RunnerSession:
    """Return a baseline CLI session for command tests."""

    return RunnerSession(
        requested_mode="deterministic",
        resolved_mode="deterministic",
        llm_client=None,
        thread_id="thread-a",
        sqlite_path="/tmp/test.sqlite3",
        memory_mode="persistent",
        history=[
            Message(role=MessageRole.USER, content="first"),
            Message(role=MessageRole.ASSISTANT, content="reply"),
        ],
        last_context={"progress": {"turn_count": 2}, "transcript": []},
    )


@pytest.mark.asyncio
async def test_resume_command_switches_active_thread(monkeypatch) -> None:
    """The resume command should load history and context for another thread."""

    events: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "opencouch_cli.app.render_header",
        lambda mode, thread_id, memory_mode, **kwargs: events.append(
            ("header", f"{mode}:{thread_id}")
        ),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": events.append(("info", f"{style}:{message}")),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_context",
        lambda state: events.append(
            ("context", str(state.get("progress", {}).get("turn_count", 0)))
        ),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_history",
        lambda session, limit=6: events.append(
            ("history", f"{session.thread_id}:{limit}:{len(session.history)}")
        ),
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/resume thread-b", session, runtime)

    assert should_continue is True
    assert session.thread_id == "thread-b"
    assert len(session.history) == 2
    assert session.last_context == {"progress": {"turn_count": 1}, "transcript": []}
    assert ("header", "deterministic:thread-b") in events
    assert ("history", "thread-b:2:2") in events


@pytest.mark.asyncio
async def test_new_command_generates_fresh_thread_state(monkeypatch) -> None:
    """The new command should switch to an empty thread without restarting."""

    events: list[tuple[str, str]] = []

    monkeypatch.setattr("opencouch_cli.app.generate_thread_id", lambda: "thread-c")
    monkeypatch.setattr(
        "opencouch_cli.app.render_header",
        lambda mode, thread_id, memory_mode, **kwargs: events.append(
            ("header", f"{mode}:{thread_id}")
        ),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": events.append(("info", f"{style}:{message}")),
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/new", session, runtime)

    assert should_continue is True
    assert session.thread_id == "thread-c"
    assert session.history == []
    assert session.last_context is None
    assert ("header", "deterministic:thread-c") in events


@pytest.mark.asyncio
async def test_threads_command_uses_runtime_listing(monkeypatch) -> None:
    """The threads command should render the runtime thread summaries."""

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "opencouch_cli.app.render_threads",
        lambda threads, active_thread_id: captured.update(
            {"threads": threads, "active_thread_id": active_thread_id}
        ),
    )

    session = _session()
    runtime = FakeRuntime()
    runtime.thread_summaries = ["thread-a", "thread-b", "thread-c"]

    should_continue = await handle_command("/threads 2", session, runtime)

    assert should_continue is True
    assert captured == {
        "threads": ["thread-a", "thread-b"],
        "active_thread_id": "thread-a",
    }


# ─── /memory list command ──────────────────────────────────────────────
#
# The /memory list subcommand was added alongside the v0.3.1 retrieval
# work as a dogfood-observability tool. It dumps the semantic memory
# store's contents so operators can answer "what did the extractor
# actually write?" without running a probe script.
#
# These tests exercise the CLI dispatch layer (handle_command routing
# + the runtime contract) rather than the Rich rendering itself — Rich
# output is hard to assert on deterministically, and the existing CLI
# tests follow the same monkey-patch-the-render-function pattern.


@pytest.mark.asyncio
async def test_memory_list_command_dispatches_render(monkeypatch) -> None:
    """/memory list should route to render_memory_list with the runtime.

    v0.8: ``render_memory_list`` became async because the store's
    ``arecord_count`` / ``anamespaces`` helpers went async to support
    the SQLite-backed implementation. The test's monkey-patch
    substitute has to be an awaitable callable — sync lambdas don't
    satisfy that contract anymore."""

    captured: dict[str, object] = {}

    async def _fake_render_memory_list(runtime_arg):
        captured["runtime"] = runtime_arg

    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_list",
        _fake_render_memory_list,
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/memory list", session, runtime)

    assert should_continue is True
    assert captured == {"runtime": runtime}


@pytest.mark.asyncio
async def test_memory_status_still_dispatches_render_status(monkeypatch) -> None:
    """Bare /memory and /memory status should still route to render_memory_status.
    This guards against regressions in the subcommand dispatch — /memory
    used to accept only status, and we added list alongside it. Bare
    /memory with no args must still default to status."""

    render_list_calls: list[object] = []
    render_status_calls: list[object] = []

    async def _fake_render_memory_list(runtime_arg):
        render_list_calls.append(runtime_arg)

    async def _fake_render_memory_status(runtime_arg, session_arg):
        # v0.7 Stage E: render_memory_status gained a session parameter
        # so the recall toggle row can be populated from the profile.
        render_status_calls.append((runtime_arg, session_arg))

    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_list", _fake_render_memory_list
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_status", _fake_render_memory_status
    )

    session = _session()
    runtime = FakeRuntime()

    # Bare /memory → status
    await handle_command("/memory", session, runtime)
    # Explicit /memory status → status
    await handle_command("/memory status", session, runtime)

    assert len(render_status_calls) == 2
    assert len(render_list_calls) == 0


@pytest.mark.asyncio
async def test_memory_unknown_subcommand_warns(monkeypatch) -> None:
    """Unknown /memory subcommands should produce a warning, not crash or
    silently route to one of the known handlers."""

    info_messages: list[tuple[str, str]] = []

    async def _noop_async(runtime_arg):
        return None

    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": info_messages.append((style, message)),
    )
    # Stub the known async handlers so a misdispatch is visible as an
    # unexpected call rather than blowing up on Rich output.
    monkeypatch.setattr("opencouch_cli.app.render_memory_list", _noop_async)
    monkeypatch.setattr("opencouch_cli.app.render_memory_status", _noop_async)

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/memory foo", session, runtime)

    assert should_continue is True
    assert any(
        style == "warning" and "Unknown /memory subcommand" in msg
        for style, msg in info_messages
    )


# ─── v0.4 /end command and session summary ──────────────────────────────
#
# The /end command was rewired in v0.4 to trigger the session summarizer
# via runtime.end_session() and render the resulting arc as a farewell
# panel. These tests cover:
#
# 1. /end calls runtime.end_session with the active thread_id
# 2. /end renders the summary when one is returned
# 3. /end renders a plain farewell when the summarizer returns None
# 4. /end returns False from handle_command (exits the loop)
#
# Rendering details aren't asserted on — the render functions are
# monkey-patched, same pattern as the existing CLI tests.


@pytest.mark.asyncio
async def test_end_command_calls_end_session_on_runtime(monkeypatch) -> None:
    """/end should invoke runtime.end_session(thread_id) and then
    terminate the session by returning False from handle_command."""

    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_info", lambda message, style="panel": None
    )

    session = _session()
    runtime = FakeRuntime()
    runtime.end_session_returns = None  # LLM judged too thin to summarize

    should_continue = await handle_command("/end", session, runtime)

    assert should_continue is False
    assert runtime.end_session_calls == [session.thread_id]


@pytest.mark.asyncio
async def test_end_command_renders_summary_when_arc_returned(
    monkeypatch,
) -> None:
    """When the summarizer returns a StoredSessionArc, the CLI should
    render it via render_session_summary before the farewell."""

    rendered_arcs: list[object] = []
    farewell_calls: list[str] = []

    monkeypatch.setattr(
        "opencouch_cli.app.render_session_summary",
        lambda arc: rendered_arcs.append(arc),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": farewell_calls.append(message),
    )

    # Build a minimal "arc-like" sentinel object — the render function
    # is patched out, so the actual shape doesn't matter for this test.
    canned_arc = object()

    session = _session()
    runtime = FakeRuntime()
    runtime.end_session_returns = canned_arc

    should_continue = await handle_command("/end", session, runtime)

    assert should_continue is False
    assert rendered_arcs == [canned_arc]
    # The farewell still fires after the summary panel
    assert len(farewell_calls) == 1


@pytest.mark.asyncio
async def test_end_command_skips_summary_render_when_none_returned(
    monkeypatch,
) -> None:
    """When end_session returns None (incognito, no LLM, thin session,
    or silent failure), render_session_summary should NOT be called.
    Only the farewell is rendered."""

    rendered_arcs: list[object] = []
    farewell_calls: list[str] = []

    monkeypatch.setattr(
        "opencouch_cli.app.render_session_summary",
        lambda arc: rendered_arcs.append(arc),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": farewell_calls.append(message),
    )

    session = _session()
    runtime = FakeRuntime()
    runtime.end_session_returns = None

    should_continue = await handle_command("/end", session, runtime)

    assert should_continue is False
    assert rendered_arcs == []  # no summary panel rendered
    assert len(farewell_calls) == 1  # but farewell still fires


@pytest.mark.asyncio
async def test_end_command_degrades_on_runtime_exception(monkeypatch) -> None:
    """If runtime.end_session raises an unexpected exception, the CLI
    should catch it, render an info message, and still exit cleanly
    rather than crashing the loop."""

    async def _raising_end_session(thread_id, *, llm_client=None):
        raise RuntimeError("simulated runtime crash")

    info_messages: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": info_messages.append((style, message)),
    )
    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)

    session = _session()
    runtime = FakeRuntime()
    # Replace the method on this instance
    runtime.end_session = _raising_end_session  # type: ignore[method-assign]

    should_continue = await handle_command("/end", session, runtime)

    assert should_continue is False
    # The error message should appear before the farewell
    assert any(
        style == "warning" and "Something went wrong" in msg
        for style, msg in info_messages
    )


# ─── v0.8 /memory list rendering — object column ──────────────────────
#
# The v0.8 dogfood surfaced a rendering gap: when a single turn
# produced two semantic facts with identical evidence quotes but
# different objects (e.g., "I take fluoxetine and vyvanse daily"
# → one USES fact per medication), the table showed two rows that
# looked identical because the object.identifier wasn't displayed.
# These tests pin the fix: the semantic records table now includes
# an ``object`` column surfacing ``object.identifier``, so duplicate-
# looking quotes can be told apart.


def test_format_entity_identifier_extracts_identifier_field() -> None:
    """The helper should return the ``identifier`` field from a
    serialized EntityRef dict."""

    from opencouch_cli.app import _format_entity_identifier

    entity = {"type": "Person", "identifier": "Sarah"}
    assert _format_entity_identifier(entity) == "Sarah"


def test_format_entity_identifier_returns_placeholder_for_missing_field() -> None:
    """When the identifier is missing, the helper should return ``'?'``
    rather than raising. Defensive against schema drift."""

    from opencouch_cli.app import _format_entity_identifier

    assert _format_entity_identifier({"type": "Person"}) == "?"
    assert _format_entity_identifier({"identifier": ""}) == "?"
    assert _format_entity_identifier({}) == "?"


def test_format_entity_identifier_returns_placeholder_for_wrong_shape() -> None:
    """When the value isn't a dict at all, the helper should return
    ``'?'`` rather than crashing. Also defensive."""

    from opencouch_cli.app import _format_entity_identifier

    assert _format_entity_identifier(None) == "?"
    assert _format_entity_identifier("not-a-dict") == "?"
    assert _format_entity_identifier(42) == "?"


def test_render_semantic_records_table_shows_object_column(capsys) -> None:
    """The rendered table must include an ``object`` column header
    AND the object identifier for each record. This is the
    regression guard for the v0.8 dogfood rendering fix — if a
    future refactor drops the column or the per-row lookup, this
    test catches it."""

    from opencouch_cli.app import _render_semantic_records_table

    records = [
        (
            "fact-fluoxetine",
            {
                "category": "context",
                "subject": {"type": "User", "identifier": "user"},
                "predicate": "USES",
                "object": {"type": "CopingStrategy", "identifier": "fluoxetine"},
                "evidence_quote": "I take fluoxetine and vyvanse daily",
                "confidence": "high",
            },
        ),
        (
            "fact-vyvanse",
            {
                "category": "context",
                "subject": {"type": "User", "identifier": "user"},
                "predicate": "USES",
                "object": {"type": "CopingStrategy", "identifier": "vyvanse"},
                "evidence_quote": "I take fluoxetine and vyvanse daily",
                "confidence": "high",
            },
        ),
    ]

    _render_semantic_records_table(records)
    captured = capsys.readouterr()

    # The column header must appear
    assert "object" in captured.out
    # Both object identifiers must appear as row values — this is
    # the regression guard for the v0.8 dogfood fix
    assert "fluoxetine" in captured.out
    assert "vyvanse" in captured.out
    # And the evidence quote still appears. Rich wraps long quotes
    # across multiple lines inside the column cell, so we can't
    # assert the full phrase as a single substring. Assert that
    # all the distinctive tokens appear somewhere in the output,
    # which is enough to confirm the quote didn't get dropped.
    assert "daily" in captured.out  # the last token of the quote


def test_render_semantic_records_table_shows_placeholder_for_missing_object(
    capsys,
) -> None:
    """When a record has a missing or malformed object field,
    the table should render ``'?'`` rather than crashing or
    leaving a blank column."""

    from opencouch_cli.app import _render_semantic_records_table

    records = [
        (
            "fact-malformed",
            {
                "category": "context",
                "predicate": "USES",
                # object field missing entirely
                "evidence_quote": "malformed record",
                "confidence": "low",
            },
        ),
    ]

    _render_semantic_records_table(records)
    captured = capsys.readouterr()

    # The row should render with a '?' in the object column
    assert "malformed record" in captured.out
    assert "?" in captured.out


# ─── v0.7 Stage E: procedural CLI commands ─────────────────────────────────
#
# These tests cover the /memory list rules, /memory recall on|off, and
# /memory forget rule <n> commands. Unlike the earlier CLI tests that
# monkeypatch render functions, these tests use a real
# OpenCouchMemoryStore so the round-trip through the procedural store
# helpers is exercised end-to-end. The assertions check captured
# console output rather than mocked call records — this catches bugs
# in the renderers themselves, not just the dispatch layer.


class FakeProceduralRuntime:
    """Runtime stub with a real memory_store for procedural CLI tests.

    Unlike the thread-management ``FakeRuntime`` above, this runtime
    exposes a real ``OpenCouchMemoryStore`` so the /memory commands
    that talk to the store (list rules, recall on/off, forget rule)
    actually round-trip through the Stage A helper functions. The
    store starts empty and each test writes to it via the helpers
    before running the CLI command.

    Also exposes ``crisis_log_backend`` and ``memory_mode`` because
    ``render_memory_status`` reads those; they can be any non-None
    value since the status command just renders their string form.
    """

    def __init__(self) -> None:
        from agent.memory.store import OpenCouchMemoryStore

        self.memory_store = OpenCouchMemoryStore()
        self.crisis_log_backend = None
        self.memory_mode = "persistent"


@pytest.mark.asyncio
async def test_memory_list_rules_empty_state(capsys) -> None:
    """With no rules written yet, the empty-state panel renders."""

    from agent.memory.modes import MemoryMode
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    runtime.memory_mode = MemoryMode.LOCAL
    session = _session()

    await handle_command("/memory list rules", session, runtime)
    captured = capsys.readouterr()

    assert "No procedural rules for this thread yet" in captured.out
    assert "Memory List (procedural)" in captured.out


@pytest.mark.asyncio
async def test_memory_list_rules_renders_populated_rules(capsys) -> None:
    """A populated profile renders each rule in the table with its
    index, rule text, evidence, date, and confidence."""

    from agent.memory.modes import MemoryMode
    from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    runtime.memory_mode = MemoryMode.LOCAL
    session = _session()  # thread_id="thread-a"

    # Write two rules via the Stage A helper path
    await aadd_procedural_rule(
        runtime.memory_store,
        user_id="thread-a",
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        ),
    )
    await aadd_procedural_rule(
        runtime.memory_store,
        user_id="thread-a",
        rule=build_procedural_rule(
            rule_text="You've said meditation makes you more anxious.",
            evidence=["Please don't suggest meditation again"],
        ),
    )

    await handle_command("/memory list rules", session, runtime)
    captured = capsys.readouterr()

    # Both rules are rendered. Use short substrings that don't cross
    # Rich's column-wrap line boundaries — the table wraps long text
    # inside cells, so asserting on full rule phrases fails on
    # multi-word rules.
    assert "shorter" in captured.out
    assert "meditation" in captured.out
    # Evidence column content — "Please" and "keep" both land on
    # the first line of the evidence cell even when the full quote
    # wraps, so they're safe anchors.
    assert "Please" in captured.out
    # The panel title reports the count
    assert "2 rule(s)" in captured.out
    # The 1-indexed position column is visible
    assert " 1 " in captured.out
    assert " 2 " in captured.out


@pytest.mark.asyncio
async def test_memory_list_rules_isolates_threads(capsys) -> None:
    """Rules written for thread-a should not appear in thread-b's list.

    Regression guard for namespace isolation at the CLI layer. The
    profile is namespaced by thread_id, so a rule for thread-a must
    not leak into a separate thread's view.
    """

    from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()

    # Write a rule for thread-a only
    await aadd_procedural_rule(
        runtime.memory_store,
        user_id="thread-a",
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        ),
    )

    # List rules from thread-b's session — should be empty
    session_b = RunnerSession(
        requested_mode="deterministic",
        resolved_mode="deterministic",
        llm_client=None,
        thread_id="thread-b",
        sqlite_path="/tmp/test.sqlite3",
        memory_mode="persistent",
        history=[],
    )
    await handle_command("/memory list rules", session_b, runtime)
    captured = capsys.readouterr()

    # Empty-state panel, not the populated one
    assert "No procedural rules" in captured.out
    assert "You prefer shorter responses" not in captured.out


@pytest.mark.asyncio
async def test_memory_recall_on_from_off_writes_and_explains(capsys) -> None:
    """Flipping OFF → ON writes the toggle and shows the first-run explanation."""

    from agent.memory.procedural import aget_procedural_profile
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    # Sanity: starts OFF
    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
    )
    assert profile.proactive_recall_enabled is False

    await handle_command("/memory recall on", session, runtime)
    captured = capsys.readouterr()

    # Profile was updated
    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
    )
    assert profile.proactive_recall_enabled is True

    # First-run explanation content is present
    assert "Proactive recall is now ON" in captured.out
    # The example from schema.yaml opt_in_confirmation_example
    assert "past conversations" in captured.out
    # The reassurance that style rules are independent of the toggle
    assert "Style rules" in captured.out


@pytest.mark.asyncio
async def test_memory_recall_off_from_on_writes_confirmation(capsys) -> None:
    """Flipping ON → OFF writes the toggle and shows the brief confirmation."""

    from agent.memory.procedural import aget_procedural_profile, aset_proactive_recall
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    # Pre-set recall to ON so we can flip it off
    await aset_proactive_recall(
        runtime.memory_store, user_id=session.thread_id, enabled=True
    )

    await handle_command("/memory recall off", session, runtime)
    captured = capsys.readouterr()

    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
    )
    assert profile.proactive_recall_enabled is False

    assert "Proactive recall is now OFF" in captured.out
    # Reassurance that memory still shapes responses silently
    assert "I still remember" in captured.out


@pytest.mark.asyncio
async def test_memory_recall_already_on_is_noop(capsys) -> None:
    """Setting recall ON when already ON should produce a warning, no write."""

    from agent.memory.procedural import aget_procedural_profile, aset_proactive_recall
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    # Pre-set recall to ON
    await aset_proactive_recall(
        runtime.memory_store, user_id=session.thread_id, enabled=True
    )

    await handle_command("/memory recall on", session, runtime)
    captured = capsys.readouterr()

    assert "already on" in captured.out
    # Profile still True — no write happened, but the state is unchanged
    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
    )
    assert profile.proactive_recall_enabled is True


@pytest.mark.asyncio
async def test_memory_recall_invalid_arg_shows_usage(capsys) -> None:
    """``/memory recall`` with no arg or a bad arg should show usage."""

    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    await handle_command("/memory recall", session, runtime)
    captured = capsys.readouterr()
    assert "Usage: /memory recall on" in captured.out

    await handle_command("/memory recall maybe", session, runtime)
    captured = capsys.readouterr()
    assert "Usage: /memory recall on" in captured.out


@pytest.mark.asyncio
async def test_memory_forget_rule_y_confirms_and_deletes(capsys, monkeypatch) -> None:
    """A y/Y confirmation removes the rule from the profile."""

    from agent.memory.procedural import (
        aadd_procedural_rule,
        aget_procedural_profile,
        build_procedural_rule,
    )
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    # Write two rules
    await aadd_procedural_rule(
        runtime.memory_store,
        user_id=session.thread_id,
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        ),
    )
    await aadd_procedural_rule(
        runtime.memory_store,
        user_id=session.thread_id,
        rule=build_procedural_rule(
            rule_text="You've said meditation makes you more anxious.",
            evidence=["Please don't suggest meditation again"],
        ),
    )

    # Monkeypatch Prompt.ask to return 'y' without interactive input
    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "y")

    await handle_command("/memory forget rule 1", session, runtime)
    captured = capsys.readouterr()

    assert "Deleted rule #1" in captured.out
    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
    )
    # Rule 1 (shorter responses) was removed, rule 2 (meditation)
    # shifted into position 0.
    assert len(profile.rules) == 1
    assert "meditation" in profile.rules[0].rule


@pytest.mark.asyncio
async def test_memory_forget_rule_n_cancels(capsys, monkeypatch) -> None:
    """A 'n' or empty confirmation must NOT touch the profile."""

    from agent.memory.procedural import (
        aadd_procedural_rule,
        aget_procedural_profile,
        build_procedural_rule,
    )
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    await aadd_procedural_rule(
        runtime.memory_store,
        user_id=session.thread_id,
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        ),
    )

    # Monkeypatch Prompt.ask to return '' (the default 'n')
    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "")

    await handle_command("/memory forget rule 1", session, runtime)
    captured = capsys.readouterr()

    assert "Cancelled" in captured.out
    profile = await aget_procedural_profile(
        runtime.memory_store, user_id=session.thread_id
    )
    # Rule is still there
    assert len(profile.rules) == 1


@pytest.mark.asyncio
async def test_memory_forget_rule_out_of_range_warns(capsys) -> None:
    """Deleting a rule that doesn't exist should produce a warning."""

    from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    await aadd_procedural_rule(
        runtime.memory_store,
        user_id=session.thread_id,
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep it short"],
        ),
    )

    # Index 5 when only 1 rule exists
    await handle_command("/memory forget rule 5", session, runtime)
    captured = capsys.readouterr()

    assert "does not exist" in captured.out


@pytest.mark.asyncio
async def test_memory_forget_rule_no_rules_warns(capsys) -> None:
    """Forgetting a rule when the profile has none should produce a warning."""

    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    await handle_command("/memory forget rule 1", session, runtime)
    captured = capsys.readouterr()

    assert "No procedural rules to forget" in captured.out


@pytest.mark.asyncio
async def test_memory_forget_rule_bad_index_warns(capsys) -> None:
    """A non-integer index should produce a usage warning, not crash."""

    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    await handle_command("/memory forget rule xyz", session, runtime)
    captured = capsys.readouterr()

    assert "Usage: /memory forget rule <n>" in captured.out


@pytest.mark.asyncio
async def test_memory_forget_fact_shows_not_yet_message(capsys) -> None:
    """/memory forget fact is v0.9 scope; must show a clear 'not yet' message."""

    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    await handle_command("/memory forget fact 1", session, runtime)
    captured = capsys.readouterr()

    assert "not yet available" in captured.out
    assert "v0.9" in captured.out


@pytest.mark.asyncio
async def test_memory_status_shows_recall_toggle_state(capsys) -> None:
    """/memory status should render the actual recall toggle state
    (on or off) rather than a phase-2+ placeholder."""

    from agent.memory.procedural import aset_proactive_recall
    from opencouch_cli.app import handle_command

    runtime = FakeProceduralRuntime()
    session = _session()

    # Default state: OFF
    await handle_command("/memory status", session, runtime)
    captured = capsys.readouterr()
    assert "proactive recall" in captured.out
    assert "off" in captured.out
    assert "(phase 2+)" not in captured.out

    # Flip to ON and re-check
    await aset_proactive_recall(
        runtime.memory_store, user_id=session.thread_id, enabled=True
    )
    await handle_command("/memory status", session, runtime)
    captured = capsys.readouterr()
    assert "proactive recall" in captured.out
    assert " on" in captured.out  # leading space disambiguates from "onset"


# ─── v0.8: --user-id flag (pulled forward from v0.9) ──────────────────────
#
# These tests cover the RunnerSession.owner_id() helper and the end-to-end
# "rules written under --user-id are visible across thread switches" flow.
# The core invariant: before v0.8, every CLI memory call used
# session.thread_id directly, so each thread was its own effective user
# namespace. With the --user-id flag, users can switch threads without
# losing their rules / facts / episodic arcs. The tests below verify
# that decoupling actually works at the CLI layer.


class TestSessionOwnerId:
    """Unit tests for RunnerSession.owner_id() fallback precedence."""

    def test_owner_id_falls_back_to_thread_id_when_user_id_none(self) -> None:
        """Backward-compatible default: no --user-id → use thread_id."""

        session = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-alpha",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
        )
        assert session.user_id is None
        assert session.owner_id() == "thread-alpha"

    def test_owner_id_returns_user_id_when_set(self) -> None:
        """Explicit --user-id overrides the thread_id fallback."""

        session = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-alpha",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
            user_id="alice",
        )
        assert session.user_id == "alice"
        assert session.owner_id() == "alice"

    def test_owner_id_survives_thread_switch(self) -> None:
        """Changing thread_id while user_id stays set continues to
        return the user_id — the exact dogfood use case for the flag.

        Regression guard: session.thread_id gets mutated by /resume
        and /new commands. The owner_id() method must keep returning
        the explicit user_id across those mutations, otherwise a
        /resume would cause memory writes to jump namespaces
        mid-session."""

        session = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-alpha",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
            user_id="alice",
        )
        assert session.owner_id() == "alice"

        # Simulate /resume switching to a different thread
        session.thread_id = "thread-beta"
        assert session.owner_id() == "alice"

        # Simulate /new generating a fresh thread
        session.thread_id = "thread-gamma"
        assert session.owner_id() == "alice"


class TestUserIdCommandIntegration:
    """Integration tests: verify the /memory commands see rules scoped
    to --user-id, not thread_id, when the flag is set.

    These are the tests that prove cross-thread memory persistence
    actually works. Write a rule under user_id=alice from thread-a,
    switch to thread-b (same user), and /memory list rules should
    still show the rule.
    """

    @pytest.mark.asyncio
    async def test_rules_written_with_user_id_survive_thread_switch(
        self, capsys
    ) -> None:
        """Write a rule under --user-id=alice from one thread, switch
        to a different thread (same user), and the rule is still
        visible via /memory list rules.

        This is the canonical 'multi-session dogfood unblock' test —
        the thing that was impossible before the flag landed."""

        from agent.memory.procedural import (
            aadd_procedural_rule,
            build_procedural_rule,
        )
        from opencouch_cli.app import handle_command

        runtime = FakeProceduralRuntime()

        # Write a rule under user_id=alice (NOT thread-a)
        await aadd_procedural_rule(
            runtime.memory_store,
            user_id="alice",
            rule=build_procedural_rule(
                rule_text="You prefer shorter responses.",
                evidence=["Please keep it short"],
            ),
        )

        # Session 1: thread-a, user_id=alice
        session_a = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-a",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
            user_id="alice",
        )
        await handle_command("/memory list rules", session_a, runtime)
        captured = capsys.readouterr()
        assert "shorter" in captured.out
        assert "1 rule(s)" in captured.out

        # Session 2: thread-b, user_id=alice (same user, different thread)
        session_b = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-b",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
            user_id="alice",
        )
        await handle_command("/memory list rules", session_b, runtime)
        captured = capsys.readouterr()
        # Same rule visible from the new thread — this is the unlock
        assert "shorter" in captured.out
        assert "1 rule(s)" in captured.out

    @pytest.mark.asyncio
    async def test_rules_are_isolated_between_different_user_ids(self, capsys) -> None:
        """User alice's rules must not appear in user bob's list even
        if bob happens to reuse alice's thread_id. This is the
        companion invariant to the cross-thread test above.

        Regression guard for namespace isolation at the CLI layer.
        If owner_id() accidentally fell back to thread_id when
        user_id was set, bob would see alice's rules."""

        from agent.memory.procedural import (
            aadd_procedural_rule,
            build_procedural_rule,
        )
        from opencouch_cli.app import handle_command

        runtime = FakeProceduralRuntime()

        await aadd_procedural_rule(
            runtime.memory_store,
            user_id="alice",
            rule=build_procedural_rule(
                rule_text="You prefer shorter responses.",
                evidence=["Please keep it short"],
            ),
        )

        # Bob reuses thread-a but has a different user_id
        session_bob = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-a",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
            user_id="bob",
        )
        await handle_command("/memory list rules", session_bob, runtime)
        captured = capsys.readouterr()

        # Empty-state panel — bob has no rules. Note: the empty-state
        # help text itself contains the example phrase "please keep
        # responses shorter", so we can't use "shorter" as a negative
        # assertion. Check for a substring that's unique to the real
        # rule ("You prefer") and absent from the help text.
        assert "No procedural rules" in captured.out
        assert "You prefer" not in captured.out

    @pytest.mark.asyncio
    async def test_memory_status_shows_owner_id_source_when_flag_set(
        self, capsys
    ) -> None:
        """The /memory status panel now shows an owner_id row
        labeled with whether the value came from --user-id or fell
        back to the thread_id. This is the dogfood observability
        hook for confirming the flag took effect."""

        from opencouch_cli.app import handle_command

        runtime = FakeProceduralRuntime()

        session = RunnerSession(
            requested_mode="deterministic",
            resolved_mode="deterministic",
            llm_client=None,
            thread_id="thread-a",
            sqlite_path="/tmp/test.db",
            memory_mode="persistent",
            user_id="alice",
        )
        await handle_command("/memory status", session, runtime)
        captured = capsys.readouterr()

        assert "owner_id" in captured.out
        assert "alice" in captured.out
        assert "from --user-id" in captured.out

    @pytest.mark.asyncio
    async def test_memory_status_shows_owner_id_from_thread_when_no_flag(
        self, capsys
    ) -> None:
        """When --user-id is not set, /memory status labels the
        owner_id row as coming from the thread_id fallback."""

        from opencouch_cli.app import handle_command

        runtime = FakeProceduralRuntime()

        session = _session()  # no user_id
        await handle_command("/memory status", session, runtime)
        captured = capsys.readouterr()

        assert "owner_id" in captured.out
        assert "from thread_id" in captured.out


class TestParserUserIdFlag:
    """Unit tests for the --user-id argparse flag."""

    def test_user_id_flag_defaults_to_none(self) -> None:
        """Without --user-id, the parsed value is None."""

        from opencouch_cli.app import build_parser

        args = build_parser().parse_args([])
        assert args.user_id is None

    def test_user_id_flag_captures_explicit_value(self) -> None:
        """With --user-id, the parsed value is the provided string."""

        from opencouch_cli.app import build_parser

        args = build_parser().parse_args(["--user-id", "alice"])
        assert args.user_id == "alice"

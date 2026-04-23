"""Tests for interactive CLI thread-management commands."""

import asyncio
import re
import time

import pytest
from rich.console import Group
from rich.spinner import Spinner

from agent.memory.models import FeedbackLabel, FeedbackSource, SessionFeedbackRecord
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
from opencouch_cli.app import RunnerSession, chat_loop, handle_command


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

        # v0.10: session-feedback tracking. Same split pattern as
        # end_session. ``record_feedback_returns`` controls what the
        # stub returns; ``record_feedback_calls`` captures the args.
        self.record_feedback_returns: SessionFeedbackRecord | None = None
        self.record_feedback_calls: list[tuple[str, FeedbackLabel, FeedbackSource]] = []

        # v0.10: unified cross-method call log so tests can assert
        # cross-method ordering (e.g., "feedback must be recorded
        # before end_session"). Every stubbed method appends to this
        # shared log. Per-method lists are kept for backward compat
        # with existing assertions.
        self.call_log: list[tuple[str, ...]] = []

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
        self.call_log.append(("end_session", thread_id))
        return self.end_session_returns

    async def record_session_feedback(
        self,
        thread_id: str,
        *,
        label: FeedbackLabel,
        source: FeedbackSource,
    ) -> SessionFeedbackRecord | None:
        """v0.10 stub: record the call and return the canned result."""

        self.record_feedback_calls.append((thread_id, label, source))
        self.call_log.append(("record_feedback", thread_id, label, source))
        return self.record_feedback_returns


class _FakeChatLoopRuntime:
    """Minimal async context-manager runtime for chat_loop lifecycle tests."""

    def __init__(self) -> None:
        self.finalize_calls: list[object | None] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get_history(self, thread_id: str):
        return []

    async def get_state(self, thread_id: str):
        return None

    async def finalize_active_sessions(self, *, llm_client=None) -> None:
        self.finalize_calls.append(llm_client)


class _FakeResponseReadyRuntime(_FakeChatLoopRuntime):
    """Runtime stub that emits response_ready before delayed done."""

    def __init__(self, order: list[str]) -> None:
        super().__init__()
        self.order = order
        self.turn_calls = 0

    async def run_turn_stream(self, **kwargs):
        self.turn_calls += 1
        assert kwargs["thread_id"] == "thread-a"
        assert kwargs["response_llm_client"] is None
        yield StatusEvent(stage="finalize")
        yield ResponseReadyEvent(output=_make_agent_output("ready"))
        await asyncio.sleep(0.02)
        self.order.append("done_yielded")
        yield DoneEvent(output=_make_agent_output("done", turn_total_ms=25.0))


class _FakeTieredChatLoopRuntime(_FakeChatLoopRuntime):
    """Runtime stub that captures the separate response client argument."""

    def __init__(self, *, expected_response_llm) -> None:
        super().__init__()
        self.expected_response_llm = expected_response_llm

    async def run_turn_stream(self, **kwargs):
        assert kwargs["response_llm_client"] is self.expected_response_llm
        yield DoneEvent(output=_make_agent_output("tiered"))


def _make_agent_output(
    response_text: str,
    *,
    turn_total_ms: float | None = None,
) -> AgentOutput:
    diagnostics = {}
    if turn_total_ms is not None:
        diagnostics["turn_total_ms"] = turn_total_ms
    return AgentOutput(
        response_text=response_text,
        response_type=ResponseCategory.THERAPEUTIC,
        crisis=CrisisAssessment(),
        response_style="support",
        mode_source="test",
        diagnostics=diagnostics,
    )


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


def test_render_header_uses_prominent_title_panel_and_session_metadata(capsys) -> None:
    """The refreshed header should keep metadata while giving the CLI a stronger title block."""

    from opencouch_cli.app import render_header

    render_header(
        "deterministic",
        "thread-a",
        "persistent",
        user_id="alice",
        response_model_tier="quality",
    )
    out = capsys.readouterr().out

    # The brand name is rendered as block-art characters, so check for
    # the block elements that form the logo plus the "CLI" tag below it.
    assert "█▀▀█" in out  # block-art logo is present
    assert "CLI" in out
    assert "PRIVATE BY DEFAULT" in out
    assert "MEMORY ON YOUR TERMS" in out
    assert "A calm workspace for supportive conversations" in out
    assert "session" in out
    assert "memory" in out
    assert "thread" in out
    assert "owner" in out
    assert "response" in out
    assert "quick actions" in out
    assert "[/bold primary]" not in out


@pytest.mark.asyncio
async def test_chat_loop_shows_spinner_loading_state_before_stage_updates(
    monkeypatch,
) -> None:
    """The live loop should start with grouped spinner/status content before stage-specific updates."""

    captured_updates: list[object] = []

    class _FakeSpinnerRuntime(_FakeChatLoopRuntime):
        async def run_turn_stream(self, **kwargs):
            await asyncio.sleep(0)
            yield StatusEvent(stage="finalize")
            yield DoneEvent(output=_make_agent_output("done"))

    class _FakeLive:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def update(self, renderable) -> None:
            captured_updates.append(renderable)

    runtime = _FakeSpinnerRuntime()
    prompts = iter(["hi", EOFError()])

    def _prompt(*args, **kwargs):
        value = next(prompts)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(
        "opencouch_cli.app.resolve_llm_client",
        lambda mode: (None, "deterministic"),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.PersistentAgentRuntime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_prompt))
    monkeypatch.setattr("opencouch_cli.app.render_header", lambda *args, **kwargs: None)
    monkeypatch.setattr("opencouch_cli.app.render_info", lambda *args, **kwargs: None)
    monkeypatch.setattr("opencouch_cli.app.Live", _FakeLive)

    await chat_loop(
        "deterministic",
        thread_id="thread-a",
        user_id=None,
        sqlite_path=":memory:",
        memory_mode="persistent",
    )

    assert isinstance(captured_updates[0], Group)
    first_spinner = captured_updates[0].renderables[0]
    assert isinstance(first_spinner, Spinner)
    assert "thinking" in str(first_spinner.text)
    assert "waiting for pipeline" in str(first_spinner.text)

    assert isinstance(captured_updates[1], Group)
    second_spinner = captured_updates[1].renderables[0]
    assert isinstance(second_spinner, Spinner)
    assert "finalizing turn" in str(second_spinner.text)


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

    async def _fake_render_memory_list(runtime_arg, session_arg):
        captured["runtime"] = runtime_arg
        captured["session"] = session_arg

    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_list",
        _fake_render_memory_list,
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/memory list", session, runtime)

    assert should_continue is True
    assert captured == {"runtime": runtime, "session": session}


@pytest.mark.asyncio
async def test_memory_status_still_dispatches_render_status(monkeypatch) -> None:
    """Bare /memory and /memory status should still route to render_memory_status.
    This guards against regressions in the subcommand dispatch — /memory
    used to accept only status, and we added list alongside it. Bare
    /memory with no args must still default to status."""

    render_list_calls: list[object] = []
    render_status_calls: list[object] = []

    async def _fake_render_memory_list(runtime_arg, session_arg):
        render_list_calls.append((runtime_arg, session_arg))

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


def _patch_feedback_prompt_skip(monkeypatch) -> None:
    """Stub the v0.10 feedback prompt to return ``None`` (user skipped).

    Prevents the real ``Prompt.ask`` from blocking on stdin during
    tests. Every test that exercises ``/end`` or ``/exit`` save=y
    needs this.
    """
    monkeypatch.setattr("opencouch_cli.app._prompt_for_session_feedback", lambda: None)


# ─── v0.10 feedback-prompt helper (direct unit tests) ────────────────
#
# These tests exercise the label-mapping and decline-path contracts
# of ``_prompt_for_session_feedback`` directly. The end-to-end CLI
# tests above stub the helper out to avoid stdin blocking; here we
# patch ``Prompt.ask`` itself so the helper's own branching is
# actually covered.


@pytest.mark.parametrize(
    "response, expected",
    [
        ("y", "positive"),
        ("n", "negative"),
        ("s", "skip"),
        ("Y", "positive"),  # case-insensitive
        ("N", "negative"),
        ("S", "skip"),
        ("  y  ", "positive"),  # whitespace-tolerant
        ("", None),  # bare Enter
        ("garbage", None),  # out-of-set → declined
    ],
)
def test_prompt_for_session_feedback_maps_responses(
    monkeypatch, response, expected
) -> None:
    """The helper maps explicit ``y``/``n``/``s`` (case- and whitespace-
    tolerant) to the correct label, and anything else — including an
    empty string — returns ``None``.

    This is the label-mapping half of the contract; the exception
    half is covered by the dedicated KeyboardInterrupt / EOFError
    tests below.
    """

    from opencouch_cli.app import _prompt_for_session_feedback

    monkeypatch.setattr(
        "opencouch_cli.app.Prompt.ask",
        staticmethod(lambda *args, **kwargs: response),
    )
    assert _prompt_for_session_feedback() == expected


def test_prompt_for_session_feedback_handles_keyboard_interrupt(
    monkeypatch,
) -> None:
    """Ctrl-C during the prompt must be swallowed by the helper and
    surface as ``None`` — the caller writes no record, the farewell
    still fires. Prevents a user's accidental Ctrl-C from crashing
    the session-end flow."""

    from opencouch_cli.app import _prompt_for_session_feedback

    def _raise_kbi(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_raise_kbi))
    # Must return None, NOT propagate.
    assert _prompt_for_session_feedback() is None


def test_prompt_for_session_feedback_handles_eof(monkeypatch) -> None:
    """EOFError (piped stdin, subprocess invocation without a TTY)
    must be swallowed the same way as KeyboardInterrupt.

    Without this guard, running ``echo 'hi' | opencouch_cli ... /end``
    would crash the CLI at the prompt step even though the user
    never saw the prompt.
    """

    from opencouch_cli.app import _prompt_for_session_feedback

    def _raise_eof(*args, **kwargs):
        raise EOFError

    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_raise_eof))
    assert _prompt_for_session_feedback() is None


def test_prompt_for_session_feedback_accepts_uppercase_shortcuts(monkeypatch) -> None:
    """The feedback prompt should allow uppercase single-letter inputs."""

    from opencouch_cli.app import _prompt_for_session_feedback

    captured_kwargs: dict[str, object] = {}

    def _ask(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "S"

    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_ask))

    assert _prompt_for_session_feedback() == "skip"
    assert captured_kwargs["choices"] == ["y", "Y", "n", "N", "s", "S", ""]


@pytest.mark.asyncio
async def test_chat_loop_waits_for_runtime_entry_before_first_prompt(
    monkeypatch,
) -> None:
    """The CLI should show warmup status and not prompt until runtime entry/prewarm has completed."""

    order: list[str] = []

    class _SlowEnterRuntime(_FakeChatLoopRuntime):
        async def __aenter__(self):
            order.append("enter_start")
            await asyncio.sleep(0.01)
            order.append("enter_done")
            return self

    class _FakeStatus:
        def __enter__(self):
            order.append("status_start")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            order.append("status_end")
            return None

    runtime = _SlowEnterRuntime()

    def _prompt(*args, **kwargs):
        order.append("prompt")
        raise EOFError

    monkeypatch.setattr(
        "opencouch_cli.app.resolve_llm_client",
        lambda mode: (None, "deterministic"),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.PersistentAgentRuntime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_prompt))
    monkeypatch.setattr(
        "opencouch_cli.app.render_header",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "opencouch_cli.app.console.status",
        lambda *args, **kwargs: _FakeStatus(),
    )

    await chat_loop(
        "deterministic",
        thread_id="thread-a",
        user_id=None,
        sqlite_path=":memory:",
        memory_mode="persistent",
    )

    assert order == [
        "status_start",
        "enter_start",
        "enter_done",
        "status_end",
        "prompt",
    ]


@pytest.mark.asyncio
async def test_chat_loop_finalizes_active_sessions_on_eof(monkeypatch) -> None:
    """Raw CLI shutdown should still flush active sessions before runtime close."""

    runtime = _FakeChatLoopRuntime()

    monkeypatch.setattr(
        "opencouch_cli.app.resolve_llm_client",
        lambda mode: (None, "deterministic"),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.PersistentAgentRuntime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        "opencouch_cli.app.Prompt.ask",
        staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(EOFError)),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_header",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda *args, **kwargs: None,
    )

    await chat_loop(
        "deterministic",
        thread_id="thread-a",
        user_id=None,
        sqlite_path=":memory:",
        memory_mode="persistent",
    )

    assert runtime.finalize_calls == [None]


@pytest.mark.asyncio
async def test_chat_loop_prompts_again_before_turn_tail_finishes(monkeypatch) -> None:
    """The next prompt should start before delayed post-response work ends."""

    order: list[str] = []
    runtime = _FakeResponseReadyRuntime(order)
    prompt_calls = 0

    def _prompt(*args, **kwargs):
        nonlocal prompt_calls
        prompt_calls += 1
        if prompt_calls == 1:
            return "hi"
        order.append("prompt2_started")
        time.sleep(0.05)
        raise EOFError

    monkeypatch.setattr(
        "opencouch_cli.app.resolve_llm_client",
        lambda mode: (None, "deterministic"),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.PersistentAgentRuntime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_prompt))
    monkeypatch.setattr("opencouch_cli.app.render_header", lambda *args, **kwargs: None)
    monkeypatch.setattr("opencouch_cli.app.render_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_context", lambda *args, **kwargs: None
    )

    await chat_loop(
        "deterministic",
        thread_id="thread-a",
        user_id=None,
        sqlite_path=":memory:",
        memory_mode="persistent",
    )

    assert order.index("prompt2_started") < order.index("done_yielded")
    assert runtime.finalize_calls == [None]


@pytest.mark.asyncio
async def test_chat_loop_passes_response_tier_client_to_runtime(monkeypatch) -> None:
    """The CLI should thread a separate response client into runtime turns."""

    control_client = object()
    response_client = object()
    runtime = _FakeTieredChatLoopRuntime(expected_response_llm=response_client)
    prompts = iter(["hi", EOFError()])

    def _prompt(*args, **kwargs):
        value = next(prompts)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(
        "opencouch_cli.app.resolve_llm_client",
        lambda mode: (control_client, "hybrid"),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.resolve_response_llm_client",
        lambda mode, tier: response_client,
    )
    monkeypatch.setattr(
        "opencouch_cli.app.PersistentAgentRuntime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_prompt))
    monkeypatch.setattr("opencouch_cli.app.render_header", lambda *args, **kwargs: None)
    monkeypatch.setattr("opencouch_cli.app.render_info", lambda *args, **kwargs: None)

    await chat_loop(
        "hybrid",
        thread_id="thread-a",
        user_id=None,
        response_model_tier="quality",
        sqlite_path=":memory:",
        memory_mode="persistent",
    )

    assert runtime.finalize_calls == [control_client]


@pytest.mark.asyncio
async def test_end_command_calls_end_session_on_runtime(monkeypatch) -> None:
    """/end should invoke runtime.end_session(thread_id) and then
    terminate the session by returning False from handle_command."""

    _patch_feedback_prompt_skip(monkeypatch)
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
    # No feedback prompted (stubbed to None) → no record written
    assert runtime.record_feedback_calls == []


@pytest.mark.asyncio
async def test_response_tier_command_updates_session(monkeypatch) -> None:
    """The CLI should let the operator switch response tiers mid-session."""

    messages: list[tuple[str, str]] = []
    response_client = object()

    monkeypatch.setattr(
        "opencouch_cli.app.resolve_response_llm_client",
        lambda mode, tier: response_client,
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": messages.append((style, message)),
    )

    session = _session()
    session.llm_client = object()
    runtime = FakeRuntime()

    should_continue = await handle_command("/response-tier quality", session, runtime)

    assert should_continue is True
    assert session.response_model_tier == "quality"
    assert session.response_llm_client is response_client
    assert messages == [("success", "Response tier updated. tier=quality")]


@pytest.mark.asyncio
async def test_end_command_renders_summary_when_arc_returned(
    monkeypatch,
) -> None:
    """When the summarizer returns a StoredSessionArc, the CLI should
    render it via render_session_summary before the farewell."""

    _patch_feedback_prompt_skip(monkeypatch)
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

    _patch_feedback_prompt_skip(monkeypatch)
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

    _patch_feedback_prompt_skip(monkeypatch)

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


# ─── v0.10 session-feedback capture ──────────────────────────────────
#
# The v0.10 session-feedback collector captures an optional thumbs
# rating before summarization. Tests exercise both end-session
# surfaces that route through ``_summarize_and_render``: ``/end`` and
# ``/exit`` (save=y branch). ``/exit`` save=n is covered separately
# to pin the "no feedback prompt, no summary" contract.


@pytest.mark.asyncio
async def test_end_command_records_feedback_before_summary(monkeypatch) -> None:
    """When the user provides a feedback label at ``/end``, the CLI
    must call ``record_session_feedback`` with the correct source
    BEFORE ``end_session``. The ordering invariant is the whole
    point of the feedback-first contract — a backend outage during
    feedback must not block summarization, but feedback must be
    attempted first so the label still lands when summary fails."""

    # User says "positive" — feedback prompt returns "positive".
    monkeypatch.setattr(
        "opencouch_cli.app._prompt_for_session_feedback", lambda: "positive"
    )
    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_info", lambda message, style="panel": None
    )

    session = _session()
    runtime = FakeRuntime()
    runtime.end_session_returns = None

    should_continue = await handle_command("/end", session, runtime)

    assert should_continue is False
    # Feedback recorded with source="cli_end".
    assert runtime.record_feedback_calls == [(session.thread_id, "positive", "cli_end")]
    # end_session also fired.
    assert runtime.end_session_calls == [session.thread_id]
    # Ordering invariant: feedback before summary.
    # Filter the unified call_log down to the two methods we care about
    # and check the first two entries' method names.
    ordered = [entry[0] for entry in runtime.call_log]
    assert ordered == ["record_feedback", "end_session"], (
        f"Expected feedback before summary, got {ordered}"
    )


@pytest.mark.asyncio
async def test_end_command_skips_feedback_when_prompt_returns_none(
    monkeypatch,
) -> None:
    """When the user declines to rate (Enter / Ctrl-C / EOF),
    ``_prompt_for_session_feedback`` returns None and no feedback
    record is written. Summarization still runs."""

    monkeypatch.setattr("opencouch_cli.app._prompt_for_session_feedback", lambda: None)
    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_info", lambda message, style="panel": None
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/end", session, runtime)

    assert should_continue is False
    # No feedback recorded.
    assert runtime.record_feedback_calls == []
    # But summary still ran.
    assert runtime.end_session_calls == [session.thread_id]


@pytest.mark.asyncio
async def test_exit_save_yes_records_feedback_with_cli_exit_source(
    monkeypatch,
) -> None:
    """``/exit`` with save=y should trigger the same feedback capture
    flow as ``/end``, but with ``source="cli_exit"`` to distinguish
    the two surfaces in analytics."""

    # Prompt.ask answers the save confirmation with "y". The feedback
    # helper is patched directly (not through Prompt.ask) so this
    # test doesn't depend on the prompt-to-label mapping.
    monkeypatch.setattr(
        "opencouch_cli.app.Prompt.ask",
        staticmethod(lambda *args, **kwargs: "y"),
    )
    monkeypatch.setattr(
        "opencouch_cli.app._prompt_for_session_feedback", lambda: "negative"
    )
    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_info", lambda message, style="panel": None
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/exit", session, runtime)

    assert should_continue is False
    # Feedback recorded with source="cli_exit" (not cli_end).
    assert runtime.record_feedback_calls == [
        (session.thread_id, "negative", "cli_exit")
    ]
    assert runtime.end_session_calls == [session.thread_id]
    # Ordering invariant holds here too.
    ordered = [entry[0] for entry in runtime.call_log]
    assert ordered == ["record_feedback", "end_session"]


@pytest.mark.asyncio
async def test_exit_save_uppercase_yes_is_accepted(monkeypatch) -> None:
    """Uppercase Y should be accepted by the save-before-exit prompt."""

    captured_kwargs: dict[str, object] = {}

    def _ask(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "Y"

    monkeypatch.setattr("opencouch_cli.app.Prompt.ask", staticmethod(_ask))
    monkeypatch.setattr(
        "opencouch_cli.app._prompt_for_session_feedback", lambda: "positive"
    )
    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_info", lambda message, style="panel": None
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/exit", session, runtime)

    assert should_continue is False
    assert captured_kwargs["choices"] == ["y", "Y", "n", "N", ""]
    assert runtime.record_feedback_calls == [
        (session.thread_id, "positive", "cli_exit")
    ]
    assert runtime.end_session_calls == [session.thread_id]


@pytest.mark.asyncio
async def test_exit_save_no_skips_both_feedback_and_summary(monkeypatch) -> None:
    """``/exit`` with save=n must skip BOTH the feedback prompt AND
    the summary. The user said "don't save my conversation"; asking
    for a rating on that branch would be inconsistent."""

    # Prompt.ask returns "n" for the save confirmation.
    monkeypatch.setattr(
        "opencouch_cli.app.Prompt.ask",
        staticmethod(lambda *args, **kwargs: "n"),
    )

    # Feedback prompt should never fire — stub to a raising callable
    # so the test fails loudly if it does.
    def _fail_if_called() -> None:
        raise AssertionError("feedback prompt should not fire on /exit save=n")

    monkeypatch.setattr(
        "opencouch_cli.app._prompt_for_session_feedback", _fail_if_called
    )
    monkeypatch.setattr("opencouch_cli.app.render_session_summary", lambda arc: None)
    monkeypatch.setattr(
        "opencouch_cli.app.render_info", lambda message, style="panel": None
    )

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/exit", session, runtime)

    assert should_continue is False
    # Neither feedback nor summary should have been invoked.
    assert runtime.record_feedback_calls == []
    assert runtime.end_session_calls == []


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

    Also exposes ``crisis_log_backend``, ``session_feedback_backend``,
    and ``memory_mode`` because ``render_memory_status`` reads all
    three. They can be any value including ``None``; the panel uses
    defensive ``getattr(..., "arecord_count", None)`` so a missing
    backend falls through to the 0 default.
    """

    def __init__(self) -> None:
        from agent.memory.store import OpenCouchMemoryStore

        self.memory_store = OpenCouchMemoryStore()
        self.crisis_log_backend = None
        self.session_feedback_backend = None
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
    assert "memory" in captured.out and "procedural" in captured.out


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
    # The first-run explanation should mention past conversations.
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


# Note: the pre-v0.9 placeholder test ``test_memory_forget_fact_shows_not_yet_message``
# was removed in v0.9 when ``/memory forget fact`` shipped for real.
# See ``TestMemoryForgetFact`` at the bottom of this file for the
# replacement coverage of the real handler.


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

    @pytest.mark.asyncio
    async def test_memory_status_counts_active_owner_records_only(self, capsys) -> None:
        """The status panel should mirror the active owner-facing counts."""

        from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule
        from opencouch_cli.app import handle_command

        runtime = FakeProceduralRuntime()

        await _seed_semantic_fact(
            runtime,
            owner_id="alice",
            key="fact-active",
            evidence_quote="my sister Sarah visited",
            object_identifier="Sarah",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id="alice",
            key="fact-inactive",
            evidence_quote="old relationship note",
            object_identifier="Old note",
            dormant_at="2026-04-13T00:00:00Z",
            superseded_by="fact-active",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id="bob",
            key="fact-bob",
            evidence_quote="bob fact",
            object_identifier="Bob fact",
        )
        await _seed_episodic_arc(
            runtime,
            owner_id="alice",
            key="arc-alice",
            summary="alice session summary",
            themes=["family"],
        )
        await _seed_episodic_arc(
            runtime,
            owner_id="bob",
            key="arc-bob",
            summary="bob session summary",
            themes=["work"],
        )
        await aadd_procedural_rule(
            runtime.memory_store,
            user_id="alice",
            rule=build_procedural_rule(
                rule_text="Keep replies short.",
                evidence=["Please keep replies short"],
            ),
        )
        await aadd_procedural_rule(
            runtime.memory_store,
            user_id="bob",
            rule=build_procedural_rule(
                rule_text="Ask fewer questions.",
                evidence=["Please ask fewer questions"],
            ),
        )

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
        captured = capsys.readouterr().out

        assert re.search(r"semantic facts\s+1\b", captured)
        assert re.search(r"episodic arcs\s+1\b", captured)
        assert re.search(r"procedural rules\s+1\b", captured)
        assert re.search(r"total store records\s+7\b", captured)


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


class TestRenderContext:
    """Tests for the v0.8 additions to ``render_context``.

    The panel now shows procedural rules, the proactive-recall toggle,
    and guided-exercise tracking fields alongside the pre-existing
    memory/progress surface. We assert these by capturing the Rich
    output stream and checking for substring markers.
    """

    def test_shows_procedural_rules_when_present(self, capsys) -> None:
        """Procedural rules from memory.procedural_rules render as bullets."""

        from opencouch_cli.app import render_context

        state = {
            "progress": {"turn_count": 3},
            "memory": {
                "summary": "s",
                "current_goal": None,
                "procedural_rules": [
                    "You prefer short replies.",
                    "Don't suggest meditation.",
                ],
                "proactive_recall_enabled": False,
            },
            "response": {"guidance": "-"},
            "working_memory": [],
        }
        render_context(state)  # type: ignore[arg-type]
        out = capsys.readouterr().out

        assert "procedural_rules" in out
        assert "You prefer short replies" in out
        assert "Don't suggest meditation" in out

    def test_shows_proactive_recall_toggle(self, capsys) -> None:
        """The recall row shows on/off based on the memory field."""

        from opencouch_cli.app import render_context

        state = {
            "progress": {"turn_count": 1},
            "memory": {
                "summary": "",
                "current_goal": None,
                "procedural_rules": [],
                "proactive_recall_enabled": True,
            },
            "response": {"guidance": "-"},
            "working_memory": [],
        }
        render_context(state)  # type: ignore[arg-type]
        out = capsys.readouterr().out

        assert "proactive_recall" in out
        assert "on" in out

    def test_shows_exercise_state_when_active(self, capsys) -> None:
        """Guided exercise type + step render when progress carries them."""

        from opencouch_cli.app import render_context

        state = {
            "progress": {
                "turn_count": 1,
                "exercise_type": "box_breathing",
                "exercise_step": 3,
            },
            "memory": {"summary": "", "current_goal": None},
            "response": {"guidance": "-"},
            "working_memory": [],
            "routing": {},
        }
        render_context(state)  # type: ignore[arg-type]
        out = capsys.readouterr().out

        assert "exercise" in out
        assert "box_breathing" in out
        assert "step 3" in out

    def test_omits_exercise_row_when_inactive(self, capsys) -> None:
        """No exercise_type → exercise row is absent (avoid clutter).

        Rationale: exercises are episodic, most turns don't have one,
        and showing an empty row on every turn would crowd the panel.
        """

        from opencouch_cli.app import render_context

        state = {
            "progress": {"turn_count": 1},
            "memory": {"summary": "", "current_goal": None},
            "response": {"guidance": "-"},
            "working_memory": [],
            "routing": {},
        }
        render_context(state)  # type: ignore[arg-type]
        out = capsys.readouterr().out

        # "exercise" should not appear when no exercise is active.
        # We check with boundary to avoid accidental matches in help text
        # or other labels that might contain the substring.
        assert "exercise_type" not in out
        # The routing-driven row title "exercise" followed by a mode
        # string is absent — we settle for checking the type name
        # doesn't leak through.
        assert "box_breathing" not in out

    def test_working_memory_renders_as_bullets_not_pipes(self, capsys) -> None:
        """Multiple working_memory entries render as newline bullets.

        Pre-v0.8 rendered them as ``a | b | c``; v0.8 shows ``• a``,
        ``• b``, ``• c`` on separate lines so long entries wrap.
        """

        from opencouch_cli.app import render_context

        state = {
            "progress": {"turn_count": 1},
            "memory": {"summary": "", "current_goal": None},
            "response": {"guidance": "-"},
            "working_memory": [
                {
                    "type": "semantic",
                    "evidence_quote": "I have a sister named Sarah.",
                },
                {
                    "type": "episodic",
                    "summary": "talked about my dog passing.",
                    "primary_themes": ["grief"],
                    "is_catch_up": False,
                },
            ],
        }
        render_context(state)  # type: ignore[arg-type]
        out = capsys.readouterr().out

        # Bullet markers for each entry — the Rich table wraps the cell
        # so we check for both entries independently.
        assert "• Previously noted: I have a sister named Sarah" in out
        assert "• Last session (grief): talked about my dog passing" in out


def test_render_response_shows_footer_metadata(capsys) -> None:
    """Reply panels should surface thread and turn metadata in the footer."""

    from opencouch_cli.app import render_response

    render_response(
        "hello there",
        is_crisis=False,
        thread_id="thread-a",
        turn_count=3,
    )
    out = capsys.readouterr().out

    assert "reply" in out
    assert "thread" in out
    assert "thread-a" in out
    assert "turn" in out
    assert "3" in out
    assert "style" in out
    assert "support" in out


def test_render_meta_defaults_to_compact_summary(capsys) -> None:
    """Default diagnostics rendering should prefer the compact summary panel."""

    from opencouch_cli.app import render_meta

    render_meta(
        response_style="support",
        response_style_source="keyword",
        response_style_type="therapeutic",
        response_type="support",
        level=0,
        needs_clarification=False,
        needs_crisis_response=False,
        reason="steady and calm",
        diagnostics={"turn_total_ms": 142.0},
        memory_deltas={"semantic": 1, "procedural": 0},
    )
    out = capsys.readouterr().out

    assert "diagnostics" in out
    assert "support" in out
    assert "keyword" in out
    assert "142ms" in out
    assert "s+1" in out
    assert "p+0" in out
    assert "stage timings" not in out


class TestRenderStageTimings:
    """Tests for the v0.8 ``_render_stage_timings`` helper."""

    def test_empty_inputs_skip_panel_entirely(self, capsys) -> None:
        """With no diagnostics and no deltas, the helper prints nothing."""

        from opencouch_cli.app import _render_stage_timings

        _render_stage_timings({}, {})
        out = capsys.readouterr().out
        assert out == ""

    def test_stage_timings_table_shows_all_stages(self, capsys) -> None:
        """A populated diagnostics dict renders a full timings row per stage."""

        from opencouch_cli.app import _render_stage_timings

        _render_stage_timings(
            diagnostics={
                "load_memory_ms": 1.23,
                "crisis_gate_ms": 2.34,
                "extract_facts_ms": 3.45,
                "extract_procedural_ms": 4.56,
                "turn_total_ms": 99.99,
                "semantic_writes": 1,
                "procedural_writes": 0,
            },
            memory_deltas={"semantic": 1, "episodic": 0, "procedural": 0},
        )
        out = capsys.readouterr().out

        assert "load_memory" in out
        assert "crisis_gate" in out
        assert "extract_facts" in out
        assert "extract_procedural" in out
        assert "turn_total" in out
        assert "1.23" in out
        assert "2.34" in out
        assert "99.99" in out

    def test_missing_keys_render_as_dashes(self, capsys) -> None:
        """A diagnostics dict missing a key shows ``-`` instead of crashing."""

        from opencouch_cli.app import _render_stage_timings

        _render_stage_timings(
            diagnostics={"load_memory_ms": 5.0},
            memory_deltas={},
        )
        out = capsys.readouterr().out

        assert "5.00" in out
        # Rows for missing stages still appear with "-" in the time column
        assert "crisis_gate" in out
        assert "extract_facts" in out

    def test_policy_hold_counts_render_in_writes_column(self, capsys) -> None:
        """Phase-1 policy counters should surface in the writes column."""

        from opencouch_cli.app import _render_stage_timings

        _render_stage_timings(
            diagnostics={
                "extract_facts_ms": 3.45,
                "extract_procedural_ms": 4.56,
                "semantic_writes": 1,
                "semantic_session_end_holds": 2,
                "semantic_repeat_required": 1,
                "semantic_policy_drops": 1,
                "procedural_writes": 0,
                "procedural_session_end_holds": 1,
                "procedural_policy_drops": 1,
            },
            memory_deltas={"semantic": 1, "procedural": 0},
        )
        out = capsys.readouterr().out

        assert "1 (h2 r1 d1)" in out
        assert "0 (h1 d1)" in out


class TestDebugStateCommand:
    """Tests for the v0.8 ``/debug state`` command."""

    @pytest.mark.asyncio
    async def test_debug_state_prints_json(self, capsys) -> None:
        """/debug state dumps the thread's state dict as JSON."""

        runtime = FakeRuntime()
        session = _session()

        result = await handle_command("/debug state", session, runtime)

        assert result is True
        captured = capsys.readouterr().out
        # The panel title should appear
        assert "debug state" in captured
        # Expect the turn_count from our fake state to show up in JSON
        assert '"turn_count": 2' in captured

    @pytest.mark.asyncio
    async def test_debug_state_handles_missing_state(self, capsys) -> None:
        """When the thread has no state, the command prints a warning panel."""

        runtime = FakeRuntime()
        runtime.states.clear()  # Simulate fresh thread with no persisted state
        session = _session()

        result = await handle_command("/debug state", session, runtime)

        assert result is True
        captured = capsys.readouterr().out
        assert "debug state" in captured
        assert "No state for this thread yet" in captured

    @pytest.mark.asyncio
    async def test_debug_requires_subcommand(self, capsys) -> None:
        """`/debug` without `state` shows usage help."""

        runtime = FakeRuntime()
        session = _session()

        result = await handle_command("/debug", session, runtime)

        assert result is True
        captured = capsys.readouterr().out
        assert "Usage: /debug state" in captured

    @pytest.mark.asyncio
    async def test_debug_state_rejects_unknown_subcommand(self, capsys) -> None:
        """`/debug unknown` shows the usage help, not a crash."""

        runtime = FakeRuntime()
        session = _session()

        result = await handle_command("/debug somethingelse", session, runtime)

        assert result is True
        captured = capsys.readouterr().out
        assert "Usage: /debug state" in captured


# ─────────────────────────────────────────────────────────────────────────
# v0.9 privacy controls — /memory forget fact|session + /memory clear
# ─────────────────────────────────────────────────────────────────────────
#
# The v0.9 privacy controls extend the existing rule-forget pattern to
# semantic facts and episodic arcs, plus add a namespace-wipe command.
# These tests cover:
#
# - Per-record forget: valid / empty / out-of-range / bad-int / decline /
#   confirm paths for each of fact and session
# - `/memory clear`: typed-confirmation contract, per-kind scoping,
#   empty-store short-circuit, and cross-kind sweep for `clear all`
# - `/memory list` subcommand filtering
#
# All tests use the FakeProceduralRuntime above (it already has a real
# OpenCouchMemoryStore) + monkeypatched Prompt.ask to avoid interactive
# stdin reads. The store is seeded directly via store.aput so the tests
# don't depend on the extractor LLM pipeline.


async def _seed_semantic_fact(
    runtime: "FakeProceduralRuntime",
    *,
    owner_id: str,
    key: str,
    evidence_quote: str,
    predicate: str = "KNOWS",
    object_identifier: str = "Sarah",
    dormant_at: str | None = None,
    superseded_by: str | None = None,
    user_visible: bool = True,
) -> None:
    """Helper: write one semantic fact directly into the fake store.

    Bypasses the extractor LLM pipeline so the forget tests can focus
    on the delete path without needing a working extractor. The record
    shape mimics what the real :func:`_memory_write_to_semantic_fact`
    produces in ``agent/nodes/extract_facts.py``.
    """

    namespace = (owner_id, "semantic")
    await runtime.memory_store.aput(
        namespace,
        key=key,
        value={
            "id": key,
            "category": "relationship",
            "subject": {"type": "user", "identifier": "me"},
            "predicate": predicate,
            "object": {"type": "person", "identifier": object_identifier},
            "evidence_quote": evidence_quote,
            "confidence": "high",
            "source_session_id": "test",
            "source_turn_index": 0,
            "created_at": "2026-04-12T00:00:00Z",
            "last_referenced_at": "2026-04-12T00:00:00Z",
            "dormant_at": dormant_at,
            "superseded_by": superseded_by,
            "user_visible": user_visible,
        },
    )


async def _seed_episodic_arc(
    runtime: "FakeProceduralRuntime",
    *,
    owner_id: str,
    key: str,
    summary: str,
    themes: list[str] | None = None,
) -> None:
    """Helper: write one episodic session arc directly into the fake store."""

    namespace = (owner_id, "episodic")
    await runtime.memory_store.aput(
        namespace,
        key=key,
        value={
            "session_id": key,
            "started_at": "2026-04-11T00:00:00Z",
            "ended_at": "2026-04-11T00:15:00Z",
            "duration_seconds": 900,
            "turn_count": 5,
            "summary": summary,
            "primary_themes": themes or [],
            "mood_arc": {"opened": "tense", "closed": "calmer"},
            "open_loops": [],
            "resolved_threads": [],
            "crisis_level_max": 0,
        },
    )


class TestMemoryForgetFact:
    """Tests for the v0.9 ``/memory forget fact <n>`` command."""

    @pytest.mark.asyncio
    async def test_forget_fact_y_confirms_and_deletes(
        self, capsys, monkeypatch
    ) -> None:
        """A y confirmation removes the fact from the semantic namespace."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="my sister Sarah visited this weekend",
            object_identifier="Sarah",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-2",
            evidence_quote="I take fluoxetine daily",
            predicate="USES",
            object_identifier="fluoxetine",
        )

        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "y")

        await handle_command("/memory forget fact 1", session, runtime)
        captured = capsys.readouterr().out

        assert "Deleted fact #1" in captured
        # fact-1 is gone, fact-2 remains
        remaining = await runtime.memory_store.arecord_count(
            (session.owner_id(), "semantic")
        )
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_forget_fact_n_cancels(self, capsys, monkeypatch) -> None:
        """A cancelled confirmation must NOT touch the store."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="my sister Sarah visited this weekend",
        )

        # Empty string is the default (declined) — matches the rule-forget pattern
        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "")

        await handle_command("/memory forget fact 1", session, runtime)
        captured = capsys.readouterr().out

        assert "Cancelled" in captured
        remaining = await runtime.memory_store.arecord_count(
            (session.owner_id(), "semantic")
        )
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_forget_fact_empty_store_warns(self, capsys) -> None:
        """Forgetting a fact when none exist produces a warning, not a crash."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory forget fact 1", session, runtime)
        captured = capsys.readouterr().out

        assert "No semantic facts to forget" in captured

    @pytest.mark.asyncio
    async def test_forget_fact_out_of_range_warns(self, capsys) -> None:
        """An index beyond the fact count produces a warning."""

        runtime = FakeProceduralRuntime()
        session = _session()
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="seed",
        )

        await handle_command("/memory forget fact 5", session, runtime)
        captured = capsys.readouterr().out

        assert "does not exist" in captured
        assert "only 1 fact(s)" in captured

    @pytest.mark.asyncio
    async def test_forget_fact_indexes_only_active_facts(
        self, capsys, monkeypatch
    ) -> None:
        """Hidden inactive facts must not shift the visible delete index."""

        from agent.memory.reconciliation import filter_active_semantic_records

        runtime = FakeProceduralRuntime()
        session = _session()
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-hidden",
            evidence_quote="old hidden fact",
            object_identifier="Hidden",
            dormant_at="2026-04-13T00:00:00Z",
            superseded_by="fact-visible",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-visible",
            evidence_quote="visible fact",
            object_identifier="Visible",
        )

        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "y")

        await handle_command("/memory forget fact 1", session, runtime)
        captured = capsys.readouterr().out

        assert "Deleted fact #1" in captured
        remaining = await runtime.memory_store.asearch(
            (session.owner_id(), "semantic"), query=None, limit=1000
        )
        active_remaining = filter_active_semantic_records(remaining)
        assert len(remaining) == 1
        assert len(active_remaining) == 0

    @pytest.mark.asyncio
    async def test_forget_fact_bad_index_warns(self, capsys) -> None:
        """Non-integer argument produces a usage warning, not a crash."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory forget fact xyz", session, runtime)
        captured = capsys.readouterr().out

        assert "Usage: /memory forget fact" in captured

    @pytest.mark.asyncio
    async def test_forget_fact_zero_index_warns(self, capsys) -> None:
        """Zero is a common off-by-one mistake — warn explicitly."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory forget fact 0", session, runtime)
        captured = capsys.readouterr().out

        assert "1 or greater" in captured


class TestMemoryForgetSession:
    """Tests for the v0.9 ``/memory forget session <n>`` command."""

    @pytest.mark.asyncio
    async def test_forget_session_y_confirms_and_deletes(
        self, capsys, monkeypatch
    ) -> None:
        """A y confirmation removes the arc from the episodic namespace."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await _seed_episodic_arc(
            runtime,
            owner_id=session.owner_id(),
            key="arc-1",
            summary="first session about work anxiety",
            themes=["work stress"],
        )
        await _seed_episodic_arc(
            runtime,
            owner_id=session.owner_id(),
            key="arc-2",
            summary="second session about sleep",
            themes=["sleep"],
        )

        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "y")

        await handle_command("/memory forget session 1", session, runtime)
        captured = capsys.readouterr().out

        assert "Deleted session #1" in captured
        remaining = await runtime.memory_store.arecord_count(
            (session.owner_id(), "episodic")
        )
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_forget_session_empty_store_warns(self, capsys) -> None:
        """Empty episodic namespace produces a warning."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory forget session 1", session, runtime)
        captured = capsys.readouterr().out

        assert "No episodic sessions to forget" in captured

    @pytest.mark.asyncio
    async def test_forget_session_preview_shows_summary(
        self, capsys, monkeypatch
    ) -> None:
        """The confirmation panel previews the summary so the user
        knows which arc they're about to delete."""

        runtime = FakeProceduralRuntime()
        session = _session()
        await _seed_episodic_arc(
            runtime,
            owner_id=session.owner_id(),
            key="arc-1",
            summary="Talked about sister Sarah's visit this weekend",
            themes=["family"],
        )
        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "n")

        await handle_command("/memory forget session 1", session, runtime)
        captured = capsys.readouterr().out

        # Preview body includes the summary and themes
        assert "Sarah" in captured
        assert "family" in captured


class TestMemoryClear:
    """Tests for the v0.9 ``/memory clear <kind>`` command with typed
    confirmation."""

    @pytest.mark.asyncio
    async def test_clear_facts_requires_typed_confirmation(
        self, capsys, monkeypatch
    ) -> None:
        """'y' is NOT enough for clear — user must type the word 'clear'."""

        runtime = FakeProceduralRuntime()
        session = _session()
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="seed",
        )

        # User types 'y' — should NOT delete anything
        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "y")

        await handle_command("/memory clear facts", session, runtime)
        captured = capsys.readouterr().out

        assert "Cancelled" in captured
        remaining = await runtime.memory_store.arecord_count(
            (session.owner_id(), "semantic")
        )
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_clear_facts_with_typed_clear_proceeds(
        self, capsys, monkeypatch
    ) -> None:
        """Typing the literal word 'clear' proceeds with deletion."""

        runtime = FakeProceduralRuntime()
        session = _session()
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="seed",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-2",
            evidence_quote="seed2",
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "clear"
        )

        await handle_command("/memory clear facts", session, runtime)
        captured = capsys.readouterr().out

        assert "Cleared" in captured
        assert "facts: 2" in captured
        remaining = await runtime.memory_store.arecord_count(
            (session.owner_id(), "semantic")
        )
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_clear_rejects_uppercase_clear(self, capsys, monkeypatch) -> None:
        """The typed confirmation is case-sensitive — 'CLEAR' is not 'clear'.

        Intentional strictness: the whole point of the typed confirmation
        is to prevent muscle-memory mistakes, and case-insensitive matching
        would make it easier to accidentally confirm.
        """

        runtime = FakeProceduralRuntime()
        session = _session()
        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="seed",
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "CLEAR"
        )

        await handle_command("/memory clear facts", session, runtime)
        captured = capsys.readouterr().out

        assert "Cancelled" in captured
        remaining = await runtime.memory_store.arecord_count(
            (session.owner_id(), "semantic")
        )
        assert remaining == 1

    @pytest.mark.asyncio
    async def test_clear_all_wipes_three_namespaces(self, capsys, monkeypatch) -> None:
        """`/memory clear all` sweeps facts, sessions, and rules in one op."""

        from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule

        runtime = FakeProceduralRuntime()
        session = _session()

        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="fact seed",
        )
        await _seed_episodic_arc(
            runtime,
            owner_id=session.owner_id(),
            key="arc-1",
            summary="arc seed",
        )
        await aadd_procedural_rule(
            runtime.memory_store,
            user_id=session.owner_id(),
            rule=build_procedural_rule(
                rule_text="You prefer shorter responses.",
                evidence=["Please keep it short"],
            ),
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "clear"
        )

        await handle_command("/memory clear all", session, runtime)
        captured = capsys.readouterr().out

        assert "Cleared" in captured
        # All three counters should show up
        assert "facts: 1" in captured
        assert "sessions: 1" in captured
        assert "rules: 1" in captured

        # And the store is actually empty
        semantic_count = await runtime.memory_store.arecord_count(
            (session.owner_id(), "semantic")
        )
        episodic_count = await runtime.memory_store.arecord_count(
            (session.owner_id(), "episodic")
        )
        from agent.memory.procedural import aget_procedural_profile

        profile = await aget_procedural_profile(
            runtime.memory_store, user_id=session.owner_id()
        )
        assert semantic_count == 0
        assert episodic_count == 0
        assert profile.rules == []

    @pytest.mark.asyncio
    async def test_clear_preserves_recall_toggle_when_clearing_rules(
        self, capsys, monkeypatch
    ) -> None:
        """`/memory clear rules` wipes rules but leaves ``proactive_recall_enabled``
        alone because it's a user preference, not content."""

        from agent.memory.procedural import (
            aadd_procedural_rule,
            aget_procedural_profile,
            aset_proactive_recall,
            build_procedural_rule,
        )

        runtime = FakeProceduralRuntime()
        session = _session()

        await aadd_procedural_rule(
            runtime.memory_store,
            user_id=session.owner_id(),
            rule=build_procedural_rule(
                rule_text="You prefer shorter responses.",
                evidence=["Please keep it short"],
            ),
        )
        # Turn recall ON — this is the user preference we want to preserve
        await aset_proactive_recall(
            runtime.memory_store, user_id=session.owner_id(), enabled=True
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "clear"
        )

        await handle_command("/memory clear rules", session, runtime)

        profile = await aget_procedural_profile(
            runtime.memory_store, user_id=session.owner_id()
        )
        assert profile.rules == []
        # Preference survived
        assert profile.proactive_recall_enabled is True

    @pytest.mark.asyncio
    async def test_clear_empty_store_short_circuits_without_confirmation(
        self, capsys, monkeypatch
    ) -> None:
        """No confirmation prompt when there's nothing to delete anyway.

        This matters because an empty-store confirmation panel would be
        a false-alarm — the user would think they're about to lose data
        when nothing is actually at risk. The handler skips the prompt
        and renders an info message instead.
        """

        runtime = FakeProceduralRuntime()
        session = _session()

        # Track whether Prompt.ask was called
        prompt_calls = []

        def _track_prompt(*args, **kwargs):
            prompt_calls.append((args, kwargs))
            return ""

        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", _track_prompt)

        await handle_command("/memory clear facts", session, runtime)
        captured = capsys.readouterr().out

        assert "Nothing to clear" in captured
        assert len(prompt_calls) == 0, "Prompt.ask should not be called for empty store"

    @pytest.mark.asyncio
    async def test_clear_rejects_unknown_kind(self, capsys) -> None:
        """`/memory clear whatever` shows the usage warning."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory clear whatever", session, runtime)
        captured = capsys.readouterr().out

        assert "Usage: /memory clear" in captured

    @pytest.mark.asyncio
    async def test_clear_without_kind_shows_usage(self, capsys) -> None:
        """`/memory clear` alone shows usage, not a crash."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory clear", session, runtime)
        captured = capsys.readouterr().out

        assert "Usage: /memory clear" in captured


class TestMemoryListSubcommands:
    """Tests for the v0.9 ``/memory list facts|sessions`` subcommands."""

    @pytest.mark.asyncio
    async def test_list_facts_renders_semantic_only(self, capsys) -> None:
        """`/memory list facts` shows semantic records, not episodic."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="my sister Sarah visited",
        )
        await _seed_episodic_arc(
            runtime,
            owner_id=session.owner_id(),
            key="arc-1",
            summary="unrelated session about sleep",
            themes=["sleep"],
        )

        await handle_command("/memory list facts", session, runtime)
        captured = capsys.readouterr().out

        # Semantic table title should appear
        assert "memory" in captured and "semantic" in captured
        # The semantic fact's evidence should appear
        assert "Sarah" in captured
        # The episodic arc should NOT appear in this filtered view
        assert "episodic" not in captured

    @pytest.mark.asyncio
    async def test_list_sessions_renders_episodic_only(self, capsys) -> None:
        """`/memory list sessions` shows episodic records, not semantic."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await _seed_semantic_fact(
            runtime,
            owner_id=session.owner_id(),
            key="fact-1",
            evidence_quote="unrelated semantic fact",
        )
        await _seed_episodic_arc(
            runtime,
            owner_id=session.owner_id(),
            key="arc-1",
            summary="work anxiety session",
            themes=["work stress"],
        )

        await handle_command("/memory list sessions", session, runtime)
        captured = capsys.readouterr().out

        assert "memory" in captured and "episodic" in captured
        # Substring rather than full phrase because Rich's table wraps
        # long cells across multiple lines, breaking contiguous-match
        # assertions. "work anxiety" appears on one line, "session"
        # on the next.
        assert "work anxiety" in captured
        assert "semantic" not in captured

    @pytest.mark.asyncio
    async def test_list_facts_empty_renders_empty_state(self, capsys) -> None:
        """Empty semantic store renders the educational empty-state panel."""

        runtime = FakeProceduralRuntime()
        session = _session()

        await handle_command("/memory list facts", session, runtime)
        captured = capsys.readouterr().out

        assert "No memory records" in captured

    @pytest.mark.asyncio
    async def test_list_facts_hides_inactive_and_other_owner_records(
        self, capsys
    ) -> None:
        """The fact list should show only active semantic facts for the owner."""

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

        await _seed_semantic_fact(
            runtime,
            owner_id="alice",
            key="fact-active",
            evidence_quote="my sister Sarah visited",
            object_identifier="Sarah",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id="alice",
            key="fact-inactive",
            evidence_quote="retired hidden fact",
            object_identifier="Hidden fact",
            dormant_at="2026-04-13T00:00:00Z",
            superseded_by="fact-active",
        )
        await _seed_semantic_fact(
            runtime,
            owner_id="bob",
            key="fact-bob",
            evidence_quote="bob-only fact",
            object_identifier="BobOnly",
        )

        await handle_command("/memory list facts", session, runtime)
        captured = capsys.readouterr().out

        assert "Sarah" in captured
        assert "retired hidden fact" not in captured
        assert "BobOnly" not in captured


# ─────────────────────────────────────────────────────────────────────────
# v0.8.1 crisis log retention purge — /memory purge-crisis [days]
# ─────────────────────────────────────────────────────────────────────────
#
# The CLI command wraps CrisisLogBackend.apurge_before with a typed
# confirmation gate (same UX pattern as /memory clear). Backend-level
# tests for apurge_before itself live in test_crisis_log.py and
# test_sqlite_crisis_log.py — these tests cover the CLI path only:
# argument parsing, cutoff computation, typed confirmation, and
# result rendering.


def _build_crisis_record_for_cli(
    *,
    record_id: str,
    detected_at: str,
):  # -> CrisisLogRecord (imported inside body to keep test module imports light)
    from agent.memory.models import CrisisLogRecord

    return CrisisLogRecord(
        id=record_id,
        session_id_opaque="sess-hashed",
        user_id_or_null=None,
        detected_at=detected_at,
        level=1,
        classifier_path="deterministic",
        confidence="medium",
        reason="cli purge test",
        override_kind="none",
        response_node_completed=True,
        llm_failure_occurred=False,
    )


class FakeRuntimeWithCrisisLog:
    """Runtime stub with a real in-memory crisis log backend.

    ``FakeProceduralRuntime`` sets ``crisis_log_backend = None`` because
    the procedural tests don't need it. The purge-crisis CLI tests do
    need a real backend to assert delete behavior, so we use this
    specialized stub instead. The ``memory_store`` is also a real
    in-memory store in case a future test needs to assert that the
    semantic/episodic data survives a crisis log purge (they're
    independent backends so it should, but the stub is ready for that
    pin).
    """

    def __init__(self) -> None:
        from agent.audit.crisis_log import InMemoryCrisisLogBackend
        from agent.memory.store import OpenCouchMemoryStore

        self.memory_store = OpenCouchMemoryStore()
        self.crisis_log_backend = InMemoryCrisisLogBackend()
        self.memory_mode = "persistent"


class TestMemoryPurgeCrisis:
    """Tests for the v0.8.1 ``/memory purge-crisis [days]`` command."""

    @pytest.mark.asyncio
    async def test_purge_empty_log_short_circuits(self, capsys) -> None:
        """An empty crisis log should render a friendly info message
        without prompting for confirmation. No false-alarm panel for a
        no-op purge."""

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()

        await handle_command("/memory purge-crisis", session, runtime)
        captured = capsys.readouterr().out

        assert "Crisis log is empty" in captured
        # The scary panel should NOT have been rendered.
        assert "This cannot be undone" not in captured

    @pytest.mark.asyncio
    async def test_purge_requires_typed_purge_confirmation(
        self, capsys, monkeypatch
    ) -> None:
        """'y' is NOT enough — the user must type the word 'purge'."""

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()
        # Seed one old record that would be purged if confirmed.
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(
                record_id="old", detected_at="2020-01-01T10:00:00Z"
            )
        )

        monkeypatch.setattr("opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "y")

        await handle_command("/memory purge-crisis 30", session, runtime)
        captured = capsys.readouterr().out

        assert "Cancelled" in captured
        assert await runtime.crisis_log_backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_purge_rejects_uppercase_purge(self, capsys, monkeypatch) -> None:
        """Case-sensitive confirmation — 'PURGE' is not 'purge'.

        Matches the typed-confirmation strictness of ``/memory clear``.
        The whole point is to prevent muscle-memory mistakes; case-
        insensitive matching would make muscle-memory easier.
        """

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(
                record_id="old", detected_at="2020-01-01T10:00:00Z"
            )
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "PURGE"
        )

        await handle_command("/memory purge-crisis 30", session, runtime)
        captured = capsys.readouterr().out

        assert "Cancelled" in captured
        assert await runtime.crisis_log_backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_purge_typed_purge_deletes_old_records(
        self, capsys, monkeypatch
    ) -> None:
        """Typing the literal 'purge' proceeds with deletion."""

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()
        # Old record that should be purged.
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(
                record_id="old1", detected_at="2020-01-01T10:00:00Z"
            )
        )
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(
                record_id="old2", detected_at="2020-01-02T10:00:00Z"
            )
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "purge"
        )

        await handle_command("/memory purge-crisis 30", session, runtime)
        captured = capsys.readouterr().out

        assert "Purged 2 crisis record(s)" in captured
        assert await runtime.crisis_log_backend.arecord_count() == 0

    @pytest.mark.asyncio
    async def test_purge_preserves_recent_records(self, capsys, monkeypatch) -> None:
        """A retention window of 10000 days should delete nothing —
        ancient cutoffs preserve all realistic records. This pins the
        cutoff computation (today - days) against an operator mistake
        where an off-by-sign bug would accidentally delete everything."""

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()
        # A record from yesterday. Whatever today is, yesterday is not
        # 10000 days old, so the purge should preserve it.
        from datetime import UTC, datetime, timedelta

        yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(record_id="yesterday", detected_at=yesterday)
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "purge"
        )

        await handle_command("/memory purge-crisis 10000", session, runtime)

        assert await runtime.crisis_log_backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_purge_rejects_invalid_day_count(self, capsys) -> None:
        """A non-integer days argument should produce a usage warning."""

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()

        await handle_command("/memory purge-crisis xyz", session, runtime)
        captured = capsys.readouterr().out

        assert "expected an integer" in captured

    @pytest.mark.asyncio
    async def test_purge_rejects_zero_days(self, capsys) -> None:
        """Zero days is meaningless — warn instead of treating it as
        'purge everything.'"""

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()
        # Seed a record so the command doesn't short-circuit on empty.
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(
                record_id="r1", detected_at="2026-04-10T10:00:00Z"
            )
        )

        await handle_command("/memory purge-crisis 0", session, runtime)
        captured = capsys.readouterr().out

        assert "at least 1 day" in captured
        assert await runtime.crisis_log_backend.arecord_count() == 1

    @pytest.mark.asyncio
    async def test_purge_default_window_is_90_days(self, capsys, monkeypatch) -> None:
        """Without an explicit days argument, the default is 90."""

        from opencouch_cli.app import DEFAULT_CRISIS_RETENTION_DAYS

        assert DEFAULT_CRISIS_RETENTION_DAYS == 90

        runtime = FakeRuntimeWithCrisisLog()
        session = _session()
        await runtime.crisis_log_backend.aappend(
            _build_crisis_record_for_cli(
                record_id="r1", detected_at="2020-01-01T10:00:00Z"
            )
        )

        monkeypatch.setattr(
            "opencouch_cli.app.Prompt.ask", lambda *args, **kwargs: "purge"
        )

        await handle_command("/memory purge-crisis", session, runtime)
        captured = capsys.readouterr().out

        # The panel body should echo the 90-day window.
        assert "90 day" in captured
        # And the old record should be gone (2020-01-01 is well outside
        # a 90-day window from any realistic test run date).
        assert await runtime.crisis_log_backend.arecord_count() == 0

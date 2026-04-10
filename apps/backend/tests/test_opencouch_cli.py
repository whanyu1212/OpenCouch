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

    async def get_state(self, thread_id: str):
        return self.states.get(thread_id)

    async def get_history(self, thread_id: str):
        return list(self.histories.get(thread_id, []))

    async def list_threads(self, *, limit: int = 20):
        return self.thread_summaries[:limit]

    async def reset_thread(self, thread_id: str) -> None:
        self.states.pop(thread_id, None)
        self.histories.pop(thread_id, None)


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
        lambda mode, thread_id, memory_mode: events.append(
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
        lambda mode, thread_id, memory_mode: events.append(
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
    """/memory list should route to render_memory_list with the runtime."""

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_list",
        lambda runtime: captured.update({"runtime": runtime}),
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

    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_list",
        lambda runtime: render_list_calls.append(runtime),
    )
    monkeypatch.setattr(
        "opencouch_cli.app.render_memory_status",
        lambda runtime: render_status_calls.append(runtime),
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

    monkeypatch.setattr(
        "opencouch_cli.app.render_info",
        lambda message, style="panel": info_messages.append((style, message)),
    )
    # Also stub the known handlers so a misdispatch would be visible
    # as an unexpected call instead of a test failure on Rich output.
    monkeypatch.setattr("opencouch_cli.app.render_memory_list", lambda runtime: None)
    monkeypatch.setattr("opencouch_cli.app.render_memory_status", lambda runtime: None)

    session = _session()
    runtime = FakeRuntime()

    should_continue = await handle_command("/memory foo", session, runtime)

    assert should_continue is True
    assert any(
        style == "warning" and "Unknown /memory subcommand" in msg
        for style, msg in info_messages
    )

"""Shared runtime and LLM test doubles for integration suites."""

from __future__ import annotations

from agent.feedback.models import FeedbackLabel, FeedbackSource, SessionFeedbackRecord
from agent.models import Message, MessageRole


class FakeRuntime:
    """Minimal runtime stub for CLI command tests."""

    def __init__(self) -> None:
        self.states = {
            "thread-a": {"session_progress": {"turn_count": 2}, "transcript": []},
            "thread-b": {"session_progress": {"turn_count": 1}, "transcript": []},
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


class FakeSummaryLLM:
    """Minimal text-only LLM stub for /summary command tests."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, str | None]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        assert use_search is False
        self.calls.append((prompt, system_instruction))
        return self.response_text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ):
        _ = (prompt, system_instruction)
        if False:
            yield ""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema,
        system_instruction: str | None = None,
        use_search: bool = False,
    ):
        raise NotImplementedError

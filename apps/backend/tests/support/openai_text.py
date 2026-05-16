"""Shared OpenAI text-runtime test fakes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any


class FakeOpenAISDKRunner:
    """Deterministic Agents SDK runner fake for text-runtime tests."""

    def __init__(self, final_output: str = "openai reply") -> None:
        self.final_output = final_output
        self.run_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
    ) -> SimpleNamespace:
        self.run_calls.append(
            {"agent": agent, "input_text": input_text, "context": context}
        )
        return SimpleNamespace(final_output=self.final_output)

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
    ) -> "FakeOpenAIStream":
        self.stream_calls.append(
            {"agent": agent, "input_text": input_text, "context": context}
        )
        return FakeOpenAIStream(self.final_output)


class FakeOpenAIStream:
    """Deterministic streaming result fake for the Agents SDK."""

    def __init__(self, final_output: str) -> None:
        self.final_output = final_output

    async def stream_events(self) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(
                type="response.output_text.delta",
                delta=self.final_output,
            ),
        )

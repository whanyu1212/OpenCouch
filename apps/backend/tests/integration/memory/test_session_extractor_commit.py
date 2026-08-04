"""End-to-end: session-end extractor -> buffer -> commit writes durable memory.

This is the test that proves the previously-DEAD path is revived. Before the
rebuild, the extraction front-half was deleted (LangGraph teardown), so the
buffer was always empty and ``commit_session_memory`` short-circuited to None
(commit/service.py:257-262), writing nothing. Here the extractor populates the
buffer from a transcript, and the existing commit pipeline then writes the fact
to the store — the full revived chain.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import pytest

from agent.memory.extraction.session_extractor import extract_session_candidates
from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.store import OpenCouchMemoryStore
from agent.memory.types.semantic import ExtractionResult, MemoryWrite
from agent.models import Message, MessageRole
from agent.runtime.session.history import SessionConversation
from agent.runtime.session import run_commit_session_memory
from agent.state import AgentState
from llm.base import BaseLLMClient, StructuredResponseT


_QUOTE = "Family conflict is a big trigger for panic."


def _conversation() -> SessionConversation:
    # The fact is corroborated by two distinct user turns so commit-time support
    # scoring promotes it.
    return SessionConversation(
        messages=(
            Message(role=MessageRole.USER, content="hi"),
            Message(role=MessageRole.ASSISTANT, content="hello"),
            Message(role=MessageRole.USER, content=_QUOTE),
            Message(role=MessageRole.ASSISTANT, content="that sounds hard"),
            Message(
                role=MessageRole.USER,
                content="Yeah, family conflict always sets off my panic.",
            ),
            Message(role=MessageRole.ASSISTANT, content="let's work on it"),
        )
    )


def _state() -> AgentState:
    return cast(
        AgentState,
        {
            "user_id": "user-1",
            "session_id": "thread-test",
            "transcript": [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": _QUOTE},
                {
                    "role": "user",
                    "content": "Yeah, family conflict always sets off my panic.",
                },
            ],
        },
    )


class _ExtractorLLM(BaseLLMClient):
    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("structured only")

    async def generate_text_stream(
        self, *, prompt: str, system_instruction: str | None = None
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        if response_schema.__name__ == "ExtractionResult":
            return ExtractionResult(  # type: ignore[return-value]
                facts=[
                    MemoryWrite(
                        category="trigger",
                        subject={"type": "User", "identifier": "user-1"},
                        predicate="WORRIES_ABOUT",
                        object={
                            "type": "Concern",
                            "identifier": "family conflict panic",
                        },
                        evidence_quote=_QUOTE,
                        confidence="high",
                        source_session_id="thread-test",
                        source_turn_index=0,
                    )
                ],
                reason="durable trigger confirmed across turns",
            )
        # No procedural rules in this scenario.
        from agent.memory.types.procedural import ProceduralExtractionResult

        return ProceduralExtractionResult(rules=[], reason="none")  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_extractor_then_commit_writes_durable_fact() -> None:
    store = OpenCouchMemoryStore()
    buffer = SessionMemoryBuffer(session_id="thread-test")

    # 1. Extract from the whole transcript -> populates the buffer.
    await extract_session_candidates(
        user_turn_texts=_conversation().user_texts(),
        session_id="thread-test",
        session_buffer=buffer,
        llm_client=_ExtractorLLM(),
        memory_mode=MemoryMode.LOCAL,
    )
    assert len(buffer.held_semantic_candidates) == 1  # front-half works

    # 2. Commit the buffer -> the previously-dead short-circuit is now passed.
    result = await run_commit_session_memory(
        _state(),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,  # no cross-session arc; in-session corroboration suffices
    )

    assert result is not None  # not the empty-buffer None short-circuit
    assert result.semantic_writes == 1
    assert await store.arecord_count(("user-1", "semantic")) == 1
    records = await store.asearch(("user-1", "semantic"), query=None)
    assert records[0].value["write_timing"] == "session_end"

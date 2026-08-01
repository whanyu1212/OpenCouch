"""Unit tests for the session-end memory extractor (#162 follow-on rebuild).

Covers the LLM-free logic deterministically (provenance attribution, skip gates)
and the extract -> buffer wiring with a fake structured-output LLM. The
extract -> commit end-to-end (proving the previously-dead commit short-circuit is
revived) lives in tests/integration/memory/test_session_extractor_commit.py.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent.memory.extraction.session_extractor import extract_session_candidates
from agent.memory.modes import MemoryMode
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.types.procedural import (
    ProceduralExtractionResult,
    ProceduralRuleDraft,
)
from agent.memory.types.semantic import ExtractionResult, MemoryWrite
from llm.base import BaseLLMClient, StructuredResponseT


def _user_turn_texts(*texts: str) -> list[str]:
    return list(texts)


def _semantic_write(quote: str, *, turn_index: int = 0) -> MemoryWrite:
    return MemoryWrite(
        category="preference",
        subject={"type": "User", "identifier": "user"},
        predicate="WANTS",
        object={"type": "Goal", "identifier": "thing"},
        evidence_quote=quote,
        confidence="high",
        source_session_id="seed",  # deliberately wrong; extractor must overwrite
        source_turn_index=turn_index,
    )


class _FakeExtractLLM(BaseLLMClient):
    """Returns canned extraction results, keyed by response schema name."""

    def __init__(
        self,
        *,
        facts: list[MemoryWrite] | None = None,
        rules: list[ProceduralRuleDraft] | None = None,
        raise_on: str | None = None,
    ) -> None:
        self._facts = facts or []
        self._rules = rules or []
        self._raise_on = raise_on

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("extractor must use structured output")

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
        name = response_schema.__name__
        if self._raise_on == name:
            raise RuntimeError(f"forced failure in {name}")
        if name == "ExtractionResult":
            return ExtractionResult(facts=list(self._facts), reason="ok")  # type: ignore[return-value]
        if name == "ProceduralExtractionResult":
            return ProceduralExtractionResult(rules=list(self._rules), reason="ok")  # type: ignore[return-value]
        raise AssertionError(f"unexpected schema {name}")


@pytest.mark.asyncio
async def test_skips_incognito() -> None:
    buffer = SessionMemoryBuffer(session_id="t")
    await extract_session_candidates(
        user_turn_texts=_user_turn_texts("I always spiral before talks"),
        session_id="t",
        session_buffer=buffer,
        llm_client=_FakeExtractLLM(facts=[_semantic_write("x")]),
        memory_mode=MemoryMode.INCOGNITO,
    )
    assert buffer.held_semantic_candidates == []


@pytest.mark.asyncio
async def test_skips_without_llm() -> None:
    buffer = SessionMemoryBuffer(session_id="t")
    await extract_session_candidates(
        user_turn_texts=_user_turn_texts("I always spiral"),
        session_id="t",
        session_buffer=buffer,
        llm_client=None,
        memory_mode=MemoryMode.LOCAL,
    )
    assert buffer.held_semantic_candidates == []


@pytest.mark.asyncio
async def test_skips_empty_session() -> None:
    buffer = SessionMemoryBuffer(session_id="t")
    await extract_session_candidates(
        user_turn_texts=[],
        session_id="t",
        session_buffer=buffer,
        llm_client=_FakeExtractLLM(facts=[_semantic_write("x")]),
        memory_mode=MemoryMode.LOCAL,
    )
    assert buffer.held_semantic_candidates == []


@pytest.mark.asyncio
async def test_holds_semantic_and_corrects_provenance() -> None:
    # The quote appears in the SECOND user turn; the extractor must attribute
    # source_turn_index=1 (overriding the LLM's wrong self-report) and set the
    # real session id.
    buffer = SessionMemoryBuffer(session_id="thread-1")
    await extract_session_candidates(
        user_turn_texts=_user_turn_texts(
            "hello there",
            "I always want my answers kept short",
        ),
        session_id="thread-1",
        session_buffer=buffer,
        llm_client=_FakeExtractLLM(
            facts=[
                _semantic_write("I always want my answers kept short", turn_index=0)
            ],
        ),
        memory_mode=MemoryMode.LOCAL,
    )
    assert len(buffer.held_semantic_candidates) == 1
    held = buffer.held_semantic_candidates[0]
    assert held.candidate.payload.source_session_id == "thread-1"
    assert held.candidate.payload.source_turn_index == 1  # corrected to the real turn
    assert held.hold_action == "commit_at_session_end"


@pytest.mark.asyncio
async def test_holds_procedural_rule() -> None:
    buffer = SessionMemoryBuffer(session_id="thread-1")
    await extract_session_candidates(
        user_turn_texts=_user_turn_texts(
            "Please just give me the steps, skip the validation"
        ),
        session_id="thread-1",
        session_buffer=buffer,
        llm_client=_FakeExtractLLM(
            rules=[
                ProceduralRuleDraft(
                    rule="You prefer direct, step-by-step answers.",
                    evidence=["Please just give me the steps, skip the validation"],
                    confidence="high",
                )
            ],
        ),
        memory_mode=MemoryMode.LOCAL,
    )
    assert len(buffer.held_procedural_candidates) == 1
    assert buffer.held_procedural_candidates[0].hold_action == "commit_at_session_end"


@pytest.mark.asyncio
async def test_one_extractor_failure_does_not_block_the_other() -> None:
    # Semantic pass raises; procedural still buffers. Extraction is a side-effect
    # path and must degrade, never propagate.
    buffer = SessionMemoryBuffer(session_id="thread-1")
    await extract_session_candidates(
        user_turn_texts=_user_turn_texts("give me steps only"),
        session_id="thread-1",
        session_buffer=buffer,
        llm_client=_FakeExtractLLM(
            rules=[
                ProceduralRuleDraft(
                    rule="Be direct.",
                    evidence=["give me steps only"],
                    confidence="high",
                )
            ],
            raise_on="ExtractionResult",
        ),
        memory_mode=MemoryMode.LOCAL,
    )
    assert buffer.held_semantic_candidates == []  # failed pass degraded to empty
    assert len(buffer.held_procedural_candidates) == 1  # other pass unaffected

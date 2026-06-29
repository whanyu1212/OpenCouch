"""Cross-dialect regression tests for semantic reconciliation behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest

from agent.memory.operations.semantic_writes import (
    BatchWriteItem,
    apply_semantic_writes_batch,
    memory_write_to_semantic_fact,
    write_new_semantic_fact,
)
from agent.memory.policy.candidates import build_semantic_candidate
from agent.memory.store import MemoryStore
from agent.memory.store.postgres import PostgresMemoryStore
from agent.memory.store.sqlite import SqliteMemoryStore
from agent.memory.types import EntityRef, MemoryWrite
from llm.base import BaseLLMClient, StructuredResponseT
from tests.support.persistence import postgres_database_url


class _CoexistReconciliationLLM(BaseLLMClient):
    """Fake classifier that would allow duplicates if exact dedup missed."""

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        raise AssertionError("Text generation is not used by reconciliation.")

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
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
        response: dict[str, Any] = {
            "action": "coexist",
            "record_indexes": [],
            "reason": "fake classifier would keep both records",
            "confidence": "high",
        }
        return response_schema(**response)


def _sarah_write(
    *,
    subject_identifier: str,
    session_id: str,
) -> MemoryWrite:
    return MemoryWrite(
        category="relationship",
        subject=EntityRef(type="User", identifier=subject_identifier),
        predicate="KNOWS",
        object=EntityRef(type="Person", identifier="Sarah"),
        evidence_quote="I have a sister named Sarah",
        confidence="high",
        source_session_id=session_id,
        source_turn_index=0,
    )


async def _delete_postgres_memory_records_for_owner(dsn: str, owner_id: str) -> None:
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM memory_records WHERE owner_id = %s",
                (owner_id,),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_session_end_semantic_reconciliation_dedups_user_subject_aliases(
    tmp_path: Path,
    backend: str,
) -> None:
    """Seeded Sarah fact plus session-end Sarah extraction collapses on both stores."""

    owner_id = f"semantic-reconciliation-{uuid4()}"
    session_id = f"thread-{uuid4()}"
    dsn: str | None = None
    if backend == "sqlite":
        store: MemoryStore = SqliteMemoryStore(tmp_path / f"memory-{uuid4()}.sqlite3")
    else:
        dsn = postgres_database_url()
        if not dsn:
            pytest.skip("Postgres integration tests disabled")
        store = PostgresMemoryStore(dsn)

    try:
        seeded_fact = memory_write_to_semantic_fact(
            _sarah_write(subject_identifier=owner_id, session_id=session_id),
            write_timing="immediate",
            write_reason="seeded test fact",
            policy_version="test_v1",
        )
        await write_new_semantic_fact(store, owner_id=owner_id, fact=seeded_fact)

        extracted_alias_write = _sarah_write(
            subject_identifier="test-user",
            session_id=session_id,
        )
        outcome = await apply_semantic_writes_batch(
            store,
            owner_id=owner_id,
            items=[
                BatchWriteItem(
                    candidate=build_semantic_candidate(
                        extracted_alias_write,
                        message="I have a sister named Sarah",
                    ),
                    write_timing="session_end",
                    write_reason="session-end extraction",
                    policy_version="test_v1",
                )
            ],
            llm_client=_CoexistReconciliationLLM(),
            log_context="semantic_reconciliation_parity_test",
        )

        assert outcome.written == 0
        assert outcome.bumped == 1
        assert outcome.skipped == 0
        assert await store.arecord_count((owner_id, "semantic")) == 1

        [record] = await store.asearch((owner_id, "semantic"), query=None, limit=10)
        assert record.value["subject"]["identifier"] == owner_id
        assert record.value["object"]["identifier"] == "Sarah"
    finally:
        await store.aclose()
        if dsn is not None:
            await _delete_postgres_memory_records_for_owner(dsn, owner_id)

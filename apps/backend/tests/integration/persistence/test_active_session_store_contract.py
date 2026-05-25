"""Backend-parity contract tests for active-session stores."""

from __future__ import annotations

from uuid import uuid4

import pytest

from tests.support.persistence_contracts import open_active_session_store

pytestmark = pytest.mark.asyncio


def _thread_id(prefix: str) -> str:
    """Return a unique thread id scoped to one contract test."""

    return f"{prefix}-{uuid4().hex}"


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_active_session_store_round_trip_list_and_delete(
    backend: str,
    tmp_path,
) -> None:
    """Payload save/load/list/delete behavior should match across backends."""

    thread_a = f"a-{_thread_id(backend)}"
    thread_b = f"z-{_thread_id(backend)}"
    payload_a = (
        f'{{"thread_id":"{thread_a}","session_buffer":{{"session_id":"{thread_a}"}}}}'
    )
    payload_b = (
        f'{{"thread_id":"{thread_b}","session_buffer":{{"session_id":"{thread_b}"}}}}'
    )

    async with open_active_session_store(backend, tmp_path=tmp_path) as store:
        try:
            assert await store.load_row(thread_a) is None

            await store.save_payload(thread_b, payload_b)
            await store.save_payload(thread_a, payload_a)

            assert await store.load_row(thread_a) == (
                payload_a,
                None,
                None,
                False,
                None,
            )
            assert await store.load_row(thread_b) == (
                payload_b,
                None,
                None,
                False,
                None,
            )

            listed_ids = await store.list_ids()
            owned_ids = [
                thread_id
                for thread_id in listed_ids
                if thread_id in {thread_a, thread_b}
            ]
            assert owned_ids == [thread_a, thread_b]
        finally:
            await store.delete_session(thread_a)
            await store.delete_session(thread_b)

    async with open_active_session_store(backend, tmp_path=tmp_path) as store:
        assert await store.load_row(thread_a) is None
        assert await store.load_row(thread_b) is None


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_active_session_store_mutation_rotation_and_delete(
    backend: str,
    tmp_path,
) -> None:
    """Mutation and rotation metadata should have the same persistence semantics."""

    thread_id = _thread_id(f"{backend}-mutation")
    payload_json = (
        f'{{"thread_id":"{thread_id}","session_buffer":{{"session_id":"{thread_id}"}}}}'
    )

    async with open_active_session_store(backend, tmp_path=tmp_path) as store:
        await store.save_payload(thread_id, payload_json)

        await store.set_mutation(
            thread_id,
            mutation_token="token-1",
            mutation_kind="turn",
            finalize_required_reason="interrupted",
        )
        assert await store.load_row(thread_id) == (
            payload_json,
            "token-1",
            "turn",
            False,
            "interrupted",
        )

        await store.clear_mutation(thread_id, "wrong-token")
        assert await store.load_row(thread_id) == (
            payload_json,
            "token-1",
            "turn",
            False,
            "interrupted",
        )

        await store.clear_mutation(thread_id, "token-1")
        assert await store.load_row(thread_id) == (
            payload_json,
            None,
            None,
            False,
            "interrupted",
        )

        await store.set_rotation_required(thread_id)
        assert await store.load_row(thread_id) == (
            payload_json,
            None,
            None,
            True,
            "interrupted",
        )

        await store.delete_session(thread_id)
        assert await store.load_row(thread_id) is None

        await store.delete_session(thread_id)
        assert await store.load_row(thread_id) is None

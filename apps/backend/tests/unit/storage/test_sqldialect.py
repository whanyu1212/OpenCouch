"""Unit tests for the SQL-dialect shim (STEP 1).

These cover the dialect components in isolation: placeholder rendering, the
JSON encode/decode inverse property on adversarial payloads, and (SQLite only,
since it needs no external service) connect/schema/commit/visibility.
"""

from __future__ import annotations

import json

import pytest
from psycopg.types.json import Jsonb

from agent.storage.sqldialect import POSTGRES_DIALECT, SQLITE_DIALECT

# Adversarial payloads that stress JSONB normalization vs verbatim TEXT.
_ADVERSARIAL_PAYLOADS = [
    {"b": 1, "a": 2, "c": 3},  # key ordering JSONB may reorder
    {"f": 0.30000000000000004},  # non-canonical float
    {"big": 9007199254740993},  # int beyond f64 exact range
    {"u": "café — ☕ — 日本語"},  # non-ASCII unicode
    {"n": None, "present": "x"},  # null/optional field
    {"nested": {"z": [1, 2, {"deep": True}], "y": None}},
]


def test_placeholders_render_per_count() -> None:
    assert SQLITE_DIALECT.placeholders(1) == "?"
    assert SQLITE_DIALECT.placeholders(3) == "?, ?, ?"
    assert POSTGRES_DIALECT.placeholders(7) == ", ".join(["%s"] * 7)
    assert POSTGRES_DIALECT.placeholders(10) == ", ".join(["%s"] * 10)


def test_placeholders_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        SQLITE_DIALECT.placeholders(0)


@pytest.mark.parametrize("payload", _ADVERSARIAL_PAYLOADS)
def test_sqlite_encode_decode_inverse(payload: dict) -> None:
    encoded = SQLITE_DIALECT.encode_value(payload)
    assert isinstance(encoded, str)  # TEXT column gets a string
    decoded = SQLITE_DIALECT.decode_value(encoded)
    assert decoded == payload


@pytest.mark.parametrize("payload", _ADVERSARIAL_PAYLOADS)
def test_postgres_encode_decode_inverse(payload: dict) -> None:
    encoded = POSTGRES_DIALECT.encode_value(payload)
    assert isinstance(encoded, Jsonb)  # psycopg adapter wraps the dict
    # The Jsonb wrapper carries the original object; decode is identity since
    # psycopg returns parsed dicts on read.
    assert encoded.obj == payload
    assert POSTGRES_DIALECT.decode_value(payload) == payload


def test_sqlite_encode_default_str_guard() -> None:
    # default=str must remain so a stray non-JSON type degrades gracefully
    # rather than raising at write time.
    class _Weird:
        def __str__(self) -> str:
            return "weird"

    encoded = SQLITE_DIALECT.encode_value({"x": _Weird()})
    assert json.loads(encoded) == {"x": "weird"}


@pytest.mark.asyncio
async def test_sqlite_connect_schema_commit_visibility(tmp_path) -> None:
    db = str(tmp_path / "shim.db")
    ddls = ("CREATE TABLE IF NOT EXISTS t (k TEXT PRIMARY KEY, value TEXT NOT NULL)",)

    conn = await SQLITE_DIALECT.connect(db)
    await SQLITE_DIALECT.apply_schema(conn, ddls)
    # idempotent re-apply must not raise
    await SQLITE_DIALECT.apply_schema(conn, ddls)

    await SQLITE_DIALECT.write(
        conn,
        f"INSERT INTO t (k, value) VALUES ({SQLITE_DIALECT.placeholder}, "
        f"{SQLITE_DIALECT.placeholder})",
        ("a", SQLITE_DIALECT.encode_value({"v": 1})),
    )
    await SQLITE_DIALECT.commit(conn)
    await conn.close()

    # Re-open a FRESH connection: the write must be durable across connections.
    conn2 = await SQLITE_DIALECT.connect(db)
    rows = await SQLITE_DIALECT.read(conn2, "SELECT value FROM t WHERE k = ?", ("a",))
    assert len(rows) == 1
    assert SQLITE_DIALECT.decode_value(rows[0]["value"]) == {"v": 1}
    await conn2.close()


@pytest.mark.asyncio
async def test_sqlite_count_uses_aliased_key_access(tmp_path) -> None:
    db = str(tmp_path / "count.db")
    conn = await SQLITE_DIALECT.connect(db)
    await SQLITE_DIALECT.apply_schema(
        conn, ("CREATE TABLE IF NOT EXISTS t (k TEXT, value TEXT)",)
    )
    rows = await SQLITE_DIALECT.read(conn, "SELECT COUNT(*) AS count FROM t", ())
    # key access must work via the alias (the cross-dialect convention)
    assert rows[0]["count"] == 0
    await conn.close()

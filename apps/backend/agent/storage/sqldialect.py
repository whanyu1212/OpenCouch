"""SQL-dialect shim for SQLite/PostgreSQL key-value audit backends.

The crisis-log and session-feedback backends are the same append-only
key/value store written twice — once for ``aiosqlite`` and once for
``psycopg``. The two implementations differ only in a handful of
backend-specific seams, which this module isolates behind a single
:class:`SqlDialect` value object:

- **placeholder token** — ``?`` (SQLite) vs ``%s`` (psycopg).
- **connection factory** — path-normalize + ``aiosqlite.connect`` vs
  ``psycopg.AsyncConnection.connect`` with ``dict_row`` + autocommit.
- **schema-DDL atomicity** — bare-connection DDL + commit (SQLite) vs a
  ``conn.transaction()`` + cursor block (psycopg).
- **JSON value adapter pair** — ``json.dumps(..., default=str)`` into a TEXT
  column then ``json.loads`` back (SQLite) vs ``Jsonb(...)`` wrapping then an
  identity read (psycopg). These MUST stay exact inverses per dialect.
- **cursor ceremony** — ``conn.execute`` directly (SQLite) vs always opening
  ``conn.cursor()`` (psycopg), hidden behind read/write helpers.
- **commit policy** — explicit ``conn.commit()`` (SQLite) vs no-op under
  psycopg autocommit.

Aggregate access is standardized on ``SELECT COUNT(*) AS count`` + ``row["count"]``
for both dialects: ``aiosqlite.Row`` supports key access only when the column is
aliased, and ``dict_row`` is key-only, so the alias is mandatory.

Two ready instances are exported: :data:`SQLITE_DIALECT` and
:data:`POSTGRES_DIALECT`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SqlDialect:
    """One backend's SQL dialect: the seams that differ between drivers.

    Instances are stateless value objects shared by every backend of a given
    driver. All per-connection state lives on the caller; the dialect only
    knows *how* to do each driver-specific operation.
    """

    name: str
    placeholder: str
    _connect: Callable[[str], Awaitable[Any]]
    _apply_schema: Callable[[Any, Sequence[str]], Awaitable[None]]
    _encode_value: Callable[[Mapping[str, Any]], Any]
    _decode_value: Callable[[Any], Any]
    _read: Callable[[Any, str, Sequence[Any]], Awaitable[list[Any]]]
    _write: Callable[[Any, str, Sequence[Any]], Awaitable[int]]
    _commit: Callable[[Any], Awaitable[None]]

    def placeholders(self, count: int) -> str:
        """Render ``count`` comma-separated parameter placeholders.

        Parameterized per-statement so a wider INSERT (session feedback binds 10
        columns, crisis log 7) is data, not a hardcoded ``VALUES`` string.

        Args:
            count (int): Number of bound parameters in the statement.

        Returns:
            str: e.g. ``"?, ?, ?"`` (SQLite) or ``"%s, %s, %s"`` (psycopg).
        """

        if count < 1:
            raise ValueError("placeholders(count) requires count >= 1")
        return ", ".join([self.placeholder] * count)

    async def connect(self, target: str) -> Any:
        """Open a live, row-factory-configured async connection.

        Args:
            target (str): SQLite path / ``":memory:"`` or a PostgreSQL DSN.

        Returns:
            Any: The driver's async connection handle.
        """

        return await self._connect(target)

    async def apply_schema(self, conn: Any, ddls: Sequence[str]) -> None:
        """Apply schema DDL inside the dialect's atomicity ceremony.

        Args:
            conn (Any): Open async connection.
            ddls (Sequence[str]): Idempotent ``CREATE ... IF NOT EXISTS`` DDL.

        Returns:
            None: Schema is created if absent.
        """

        await self._apply_schema(conn, ddls)

    def encode_value(self, serialized: Mapping[str, Any]) -> Any:
        """Encode a JSON-able dict for this dialect's value column.

        Must be the exact inverse of :meth:`decode_value` for the same dialect.

        Args:
            serialized (Mapping[str, Any]): JSON-mode model dump.

        Returns:
            Any: A bound parameter (TEXT string for SQLite, ``Jsonb`` for psycopg).
        """

        return self._encode_value(serialized)

    def decode_value(self, raw: Any) -> Any:
        """Decode a stored value column back into a dict.

        Args:
            raw (Any): The row's ``value`` cell (TEXT for SQLite, dict for psycopg).

        Returns:
            Any: A plain ``dict`` ready for Pydantic validation.
        """

        return self._decode_value(raw)

    async def read(self, conn: Any, sql: str, params: Sequence[Any]) -> list[Any]:
        """Execute a SELECT and return all fetched rows.

        Args:
            conn (Any): Open async connection.
            sql (str): Query text with this dialect's placeholders.
            params (Sequence[Any]): Bound parameters.

        Returns:
            list[Any]: Fetched rows (key-accessible for both dialects).
        """

        return await self._read(conn, sql, params)

    async def write(self, conn: Any, sql: str, params: Sequence[Any]) -> int:
        """Execute an INSERT/DELETE and return the affected row count.

        Does NOT commit; callers invoke :meth:`commit` to hit the durability
        boundary (a no-op under psycopg autocommit).

        Args:
            conn (Any): Open async connection.
            sql (str): Statement text with this dialect's placeholders.
            params (Sequence[Any]): Bound parameters.

        Returns:
            int: ``cursor.rowcount`` (0 if the driver leaves it unset).
        """

        return await self._write(conn, sql, params)

    async def commit(self, conn: Any) -> None:
        """Commit pending writes (no-op under psycopg autocommit).

        Args:
            conn (Any): Open async connection.

        Returns:
            None: Writes are durable after this returns.
        """

        await self._commit(conn)


# --------------------------------------------------------------------------- #
# SQLite (aiosqlite) implementation                                           #
# --------------------------------------------------------------------------- #


async def _sqlite_connect(target: str) -> Any:
    import aiosqlite

    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(target)
    conn.row_factory = aiosqlite.Row
    return conn


async def _sqlite_apply_schema(conn: Any, ddls: Sequence[str]) -> None:
    for ddl in ddls:
        await conn.execute(ddl)
    await conn.commit()


def _sqlite_encode_value(serialized: Mapping[str, Any]) -> Any:
    # ``default=str`` is the belt-and-suspenders guard for any non-JSON type
    # that slips past the model dump; it is a no-op for already-JSON dicts.
    return json.dumps(serialized, default=str)


def _sqlite_decode_value(raw: Any) -> Any:
    return json.loads(raw)


async def _sqlite_read(conn: Any, sql: str, params: Sequence[Any]) -> list[Any]:
    async with conn.execute(sql, params) as cursor:
        return list(await cursor.fetchall())


async def _sqlite_write(conn: Any, sql: str, params: Sequence[Any]) -> int:
    cursor = await conn.execute(sql, params)
    return int(cursor.rowcount or 0)


async def _sqlite_commit(conn: Any) -> None:
    await conn.commit()


SQLITE_DIALECT = SqlDialect(
    name="sqlite",
    placeholder="?",
    _connect=_sqlite_connect,
    _apply_schema=_sqlite_apply_schema,
    _encode_value=_sqlite_encode_value,
    _decode_value=_sqlite_decode_value,
    _read=_sqlite_read,
    _write=_sqlite_write,
    _commit=_sqlite_commit,
)


# --------------------------------------------------------------------------- #
# PostgreSQL (psycopg) implementation                                         #
# --------------------------------------------------------------------------- #


async def _postgres_connect(target: str) -> Any:
    import psycopg
    from psycopg.rows import dict_row

    return await psycopg.AsyncConnection.connect(
        target,
        row_factory=dict_row,
        autocommit=True,
    )


async def _postgres_apply_schema(conn: Any, ddls: Sequence[str]) -> None:
    async with conn.transaction():
        async with conn.cursor() as cursor:
            for ddl in ddls:
                await cursor.execute(ddl)


def _postgres_encode_value(serialized: Mapping[str, Any]) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(serialized)


def _postgres_decode_value(raw: Any) -> Any:
    # psycopg returns JSONB as an already-parsed dict.
    return raw


async def _postgres_read(conn: Any, sql: str, params: Sequence[Any]) -> list[Any]:
    async with conn.cursor() as cursor:
        await cursor.execute(sql, params)
        return list(await cursor.fetchall())


async def _postgres_write(conn: Any, sql: str, params: Sequence[Any]) -> int:
    async with conn.cursor() as cursor:
        await cursor.execute(sql, params)
        return int(cursor.rowcount or 0)


async def _postgres_commit(conn: Any) -> None:
    # No-op: the connection is opened with autocommit=True.
    return None


POSTGRES_DIALECT = SqlDialect(
    name="postgres",
    placeholder="%s",
    _connect=_postgres_connect,
    _apply_schema=_postgres_apply_schema,
    _encode_value=_postgres_encode_value,
    _decode_value=_postgres_decode_value,
    _read=_postgres_read,
    _write=_postgres_write,
    _commit=_postgres_commit,
)


__all__ = [
    "SqlDialect",
    "SQLITE_DIALECT",
    "POSTGRES_DIALECT",
]

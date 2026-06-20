"""Shared append-only key/value store body for audit backends.

Both the crisis-log and session-feedback backends are the same store: an
append-only table with a ``value`` JSON column, a ``BIGSERIAL``/``AUTOINCREMENT``
``insertion_order`` surrogate that records true append order, a date column for
retention purges, and a key column for list-by-key reads. :class:`KvStore`
holds that one logic body; the per-table differences (table name, key column,
date column, INSERT column list, serializer/deserializer) live in
:class:`KvTableConfig`, and the per-driver differences live in
:class:`~agent.storage.sqldialect.SqlDialect`.

The connection opens lazily on first async use and stays attached until
``aclose``. Each runtime instance owns its own store; the class is not
thread-safe. On a failed first-time schema apply, the half-open connection is
closed and not retained, so the store stays re-attemptable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Generic, TypeVar

from agent.memory.hashing import extract_iso_date
from agent.storage.sqldialect import SqlDialect

logger = logging.getLogger(__name__)

R = TypeVar("R")


@dataclass(frozen=True)
class KvTableConfig(Generic[R]):
    """Per-table configuration for a :class:`KvStore`.

    Args:
        table: Physical table name.
        key_column: Column the list-by-key read filters on (``detected_date``
            for crisis log, ``session_id_opaque`` for feedback). Note crisis log
            lists by date and feedback by session; both are a single equality
            filter, so one ``alist_by_key`` covers both.
        date_column: Column the retention purge compares against the cutoff.
        insert_columns: Ordered INSERT column list, ``value`` last.
        ddls: Idempotent schema DDL tuple (resolved per dialect by the caller).
        to_row: Maps a record to the non-``value`` bound parameters, in
            ``insert_columns`` order (excluding ``value``).
        date_of: Extracts the record's ISO date string for the date column.
        serialize: Dumps a record to a JSON-able dict for the value column.
        deserialize: Validates a decoded dict back into a record.
    """

    table: str
    key_column: str
    date_column: str
    insert_columns: tuple[str, ...]
    ddls: tuple[str, ...]
    to_row: Callable[[R], Sequence[Any]]
    date_of: Callable[[R], str]
    serialize: Callable[[R], Mapping[str, Any]]
    deserialize: Callable[[Mapping[str, Any]], R]


class KvStore(Generic[R]):
    """Append-only key/value store shared across SQLite and PostgreSQL.

    Instances are constructed with a connection ``target`` (SQLite path or
    PostgreSQL DSN), a :class:`SqlDialect`, and a :class:`KvTableConfig`.
    """

    def __init__(
        self,
        *,
        target: str,
        dialect: SqlDialect,
        config: KvTableConfig[R],
        backend_label: str,
    ) -> None:
        """Initialize the store with its connection target and table config.

        Args:
            target (str): SQLite path / ``":memory:"`` or PostgreSQL DSN.
            dialect (SqlDialect): Driver-specific SQL seams.
            config (KvTableConfig[R]): Per-table layout and (de)serialization.
            backend_label (str): Class-style label for error/log messages.

        Returns:
            None: Stores configuration for lazy connection.
        """

        self._target = target
        self._dialect = dialect
        self._config = config
        self._backend_label = backend_label
        self._connection: Any | None = None
        self._closed = False

    async def _ensure_connection(self) -> Any:
        """Open the connection on first use, applying schema once.

        On a first-time schema-apply failure the half-open connection is closed
        and left unassigned, so the store remains re-attemptable rather than
        wedged with a connection whose schema never ran.

        Returns:
            Any: Shared async connection for the store instance.
        """

        if self._closed:
            raise RuntimeError(f"{self._backend_label} is closed.")
        if self._connection is not None:
            return self._connection

        conn = await self._dialect.connect(self._target)
        try:
            await self._dialect.apply_schema(conn, self._config.ddls)
        except BaseException:
            try:
                await conn.close()
            except Exception:
                logger.warning(
                    "%s: connection close during failed schema apply raised; ignoring",
                    self._backend_label,
                    exc_info=True,
                )
            raise
        self._connection = conn
        return self._connection

    async def aappend(self, record: R) -> None:
        """Append one record.

        Args:
            record (R): Record to persist.

        Returns:
            None: Writes the record and commits per dialect policy.
        """

        conn = await self._ensure_connection()
        cfg = self._config
        columns = ", ".join(cfg.insert_columns)
        placeholders = self._dialect.placeholders(len(cfg.insert_columns))
        params = [
            *cfg.to_row(record),
            self._dialect.encode_value(cfg.serialize(record)),
        ]
        await self._dialect.write(
            conn,
            f"INSERT INTO {cfg.table} ({columns}) VALUES ({placeholders})",
            params,
        )
        await self._dialect.commit(conn)

    async def alist_by_key(self, key: str) -> list[R]:
        """List records matching the table's key column, in append order.

        Args:
            key (str): Key value (a date prefix for crisis log, a session id
                for feedback).

        Returns:
            list[R]: Matching records ordered by ``insertion_order`` ascending.
        """

        conn = await self._ensure_connection()
        cfg = self._config
        rows = await self._dialect.read(
            conn,
            f"SELECT value FROM {cfg.table} "
            f"WHERE {cfg.key_column} = {self._dialect.placeholder} "
            "ORDER BY insertion_order ASC",
            (key,),
        )
        return [
            cfg.deserialize(self._dialect.decode_value(row["value"])) for row in rows
        ]

    async def arecord_count(self) -> int:
        """Count records in the table.

        Returns:
            int: Total record count, or 0 when the store is closed.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        rows = await self._dialect.read(
            conn,
            f"SELECT COUNT(*) AS count FROM {self._config.table}",
            (),
        )
        row = rows[0] if rows else None
        return int(row["count"]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Delete records older than an exclusive cutoff date.

        Args:
            cutoff (date): Exclusive cutoff; rows with ``date_column < cutoff``
                are deleted.

        Returns:
            int: Number of records deleted, or 0 when the store is closed.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        cfg = self._config
        deleted = await self._dialect.write(
            conn,
            f"DELETE FROM {cfg.table} "
            f"WHERE {cfg.date_column} < {self._dialect.placeholder}",
            (cutoff.isoformat(),),
        )
        await self._dialect.commit(conn)
        return deleted

    async def aclose(self) -> None:
        """Close the store, releasing the connection.

        Returns:
            None: Marks the store closed and drops the connection.
        """

        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                logger.warning(
                    "%s: connection close raised; ignoring",
                    self._backend_label,
                    exc_info=True,
                )
            finally:
                self._connection = None


def iso_date_param(value: str) -> str:
    """Extract the ISO date string from an ISO timestamp, for the date column.

    Args:
        value (str): ISO-8601 timestamp.

    Returns:
        str: ``YYYY-MM-DD`` date prefix.
    """

    return extract_iso_date(value)


__all__ = [
    "KvStore",
    "KvTableConfig",
    "iso_date_param",
]

"""Postgres-backed requests for early access."""

from datetime import datetime

from psycopg_pool import AsyncConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS access_requests (
    email        TEXT PRIMARY KEY,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgresAccessRequests:
    """Requests in their own table, one row per address."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create_schema(self) -> None:
        """Create the requests table if it doesn't exist."""
        async with self._pool.connection() as connection:
            await connection.execute(SCHEMA)

    async def add(self, email: str) -> None:
        """Record a request; asking again keeps the first time it was asked."""
        async with self._pool.connection() as connection:
            await connection.execute(
                "INSERT INTO access_requests (email) VALUES (%s)"
                " ON CONFLICT (email) DO NOTHING",
                (email,),
            )

    async def all(self) -> list[tuple[str, datetime]]:
        """Every request as (email, requested at), oldest first."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT email, requested_at FROM access_requests ORDER BY requested_at"
            )
            return await cursor.fetchall()

    async def remove(self, email: str) -> bool:
        """Forget a request. Returns whether there was one."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM access_requests WHERE email = %s", (email,)
            )
            return cursor.rowcount > 0

"""Postgres-backed crisis audit log."""

from psycopg_pool import AsyncConnectionPool

from opencouch_agent.safety.audit import AuditEntry

# No foreign key to `conversations`: entries must outlive a deleted thread,
# and incognito threads have no row there.
SCHEMA = """
CREATE TABLE IF NOT EXISTS crisis_audit_log (
    id                BIGSERIAL PRIMARY KEY,
    thread_id         TEXT NOT NULL,
    turn_at           TIMESTAMPTZ NOT NULL,
    level_before      TEXT NOT NULL,
    level_after       TEXT NOT NULL,
    reason            TEXT NOT NULL,
    screening_failure TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS crisis_audit_log_thread_id_idx
    ON crisis_audit_log (thread_id, id);
"""

_APPEND = """
INSERT INTO crisis_audit_log
    (thread_id, turn_at, level_before, level_after, reason, screening_failure)
VALUES (%s, %s, %s, %s, %s, %s)
"""


class PostgresCrisisAuditLog:
    """Audit log stored in its own Postgres table, one row per entry."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create_schema(self) -> None:
        """Create the audit table if it doesn't exist."""
        async with self._pool.connection() as connection:
            await connection.execute(SCHEMA)

    async def append(self, entry: AuditEntry) -> None:
        """Add one entry to the log."""
        # `.value` explicitly: psycopg can dump an enum by its name instead.
        failure = entry.screening_failure
        async with self._pool.connection() as connection:
            await connection.execute(
                _APPEND,
                (
                    entry.thread_id,
                    entry.at,
                    entry.level_before.value,
                    entry.level_after.value,
                    entry.reason,
                    None if failure is None else failure.value,
                ),
            )

"""Typed profile-memory storage and deterministic memory extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from agent.state import AgentState

PROFILE_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id TEXT NOT NULL,
    category TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(owner_id, category, value)
);
"""

PROFILE_MEMORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_profile_memory_owner_updated
ON profile_memory(owner_id, updated_at DESC);
"""

CANONICAL_CONCERNS = frozenset(
    {
        "overwhelm or stress",
        "anxiety or rumination",
        "grief or loss",
        "self-worth or shame",
        "relationship strain",
        "work or school pressure",
        "sleep or exhaustion",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileMemoryWrite:
    """One typed memory item ready to be persisted."""

    category: str
    value: str
    source: str
    confidence: str
    evidence: str


@dataclass(frozen=True, slots=True)
class ProfileMemoryRecord:
    """One typed memory item retrieved from persistent storage."""

    category: str
    value: str
    source: str
    confidence: str
    evidence: str
    updated_at: str


class SqliteProfileMemoryStore:
    """SQLite-backed store for stable, reviewable profile memory."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the SQLite connection and create required tables."""

        if self._db is not None:
            return
        if self.sqlite_path != Path(":memory:"):
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self.sqlite_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(PROFILE_MEMORY_SCHEMA)
        await self._db.execute(PROFILE_MEMORY_INDEX)
        await self._db.commit()

    async def close(self) -> None:
        """Close the SQLite connection when the runtime shuts down."""

        if self._db is not None:
            await self._db.close()
            self._db = None

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SqliteProfileMemoryStore must be initialized first.")
        return self._db

    async def list_memories(
        self,
        owner_id: str,
        *,
        limit: int = 8,
    ) -> list[ProfileMemoryRecord]:
        """Return the most relevant stored profile memories for one owner."""

        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT category, value, source, confidence, evidence, updated_at
            FROM profile_memory
            WHERE owner_id = ?
            ORDER BY
                CASE source WHEN 'explicit' THEN 0 ELSE 1 END,
                updated_at DESC
            LIMIT ?
            """,
            (owner_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            ProfileMemoryRecord(
                category=row["category"],
                value=row["value"],
                source=row["source"],
                confidence=row["confidence"],
                evidence=row["evidence"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def upsert_memories(
        self,
        owner_id: str,
        writes: list[ProfileMemoryWrite],
    ) -> None:
        """Insert or refresh typed memory items for one owner."""

        if not writes:
            return

        db = self._require_db()
        now = datetime.now(UTC).isoformat()
        for write in writes:
            await db.execute(
                """
                INSERT INTO profile_memory (
                    owner_id, category, value, source, confidence, evidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, category, value) DO UPDATE SET
                    source = excluded.source,
                    confidence = excluded.confidence,
                    evidence = excluded.evidence,
                    updated_at = excluded.updated_at
                """,
                (
                    owner_id,
                    write.category,
                    write.value,
                    write.source,
                    write.confidence,
                    write.evidence,
                    now,
                    now,
                ),
            )
        await db.commit()


def extract_profile_memory_writes(state: AgentState) -> list[ProfileMemoryWrite]:
    """Extract narrow, typed long-term memory writes from the completed turn."""

    writes: list[ProfileMemoryWrite] = []
    evidence = state["message"][:240].strip()
    session_intent = state.get("session_intent")
    session_intent_source = state.get("session_intent_source")

    if session_intent_source == "explicit":
        if session_intent == "just_need_to_vent":
            writes.append(
                ProfileMemoryWrite(
                    category="support_preference",
                    value="Sometimes wants space before advice.",
                    source="explicit",
                    confidence="high",
                    evidence=evidence,
                )
            )
        elif session_intent == "grounding_or_calm_down":
            writes.append(
                ProfileMemoryWrite(
                    category="support_preference",
                    value="Prefers grounding or calming support when activated.",
                    source="explicit",
                    confidence="high",
                    evidence=evidence,
                )
            )
        elif session_intent == "guided_cbt_work":
            writes.append(
                ProfileMemoryWrite(
                    category="support_preference",
                    value="Open to structured CBT-style exercises.",
                    source="explicit",
                    confidence="high",
                    evidence=evidence,
                )
            )
        elif session_intent == "psychoeducation":
            writes.append(
                ProfileMemoryWrite(
                    category="support_preference",
                    value="Sometimes wants plain-language mind-body explanations.",
                    source="explicit",
                    confidence="high",
                    evidence=evidence,
                )
            )
        elif session_intent == "reflection_and_pattern_finding":
            writes.append(
                ProfileMemoryWrite(
                    category="support_preference",
                    value="Sometimes wants help reflecting on recurring patterns.",
                    source="explicit",
                    confidence="high",
                    evidence=evidence,
                )
            )

    if state["turn_count"] >= 3:
        for concern in state.get("active_concerns", []):
            if concern not in CANONICAL_CONCERNS:
                continue
            writes.append(
                ProfileMemoryWrite(
                    category="recurring_concern",
                    value=concern,
                    source="inferred",
                    confidence="medium",
                    evidence=evidence,
                )
            )

    deduped: list[ProfileMemoryWrite] = []
    seen: set[tuple[str, str]] = set()
    for write in writes:
        key = (write.category, write.value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(write)
    return deduped


def compile_working_memory(
    profile_memories: list[ProfileMemoryRecord],
    graph_memories: list[str],
    *,
    limit: int = 8,
) -> list[str]:
    """Compile retrieved profile and graph memories into prompt-ready strings."""

    entries: list[str] = []
    for record in profile_memories:
        if record.category == "support_preference":
            entries.append(f"Support preference: {record.value}")
        elif record.category == "recurring_concern":
            entries.append(f"Recurring concern: {record.value}")
        else:
            entries.append(
                f"{record.category.replace('_', ' ').title()}: {record.value}"
            )

    entries.extend(f"Related history: {memory}" for memory in graph_memories)

    unique_entries: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        unique_entries.append(entry)
        if len(unique_entries) >= limit:
            break
    return unique_entries

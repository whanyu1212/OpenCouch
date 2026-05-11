"""Tests for the shared memory-subsystem helpers.

These helpers were extracted from ``agent/nodes/crisis_log.py`` and
``agent/persistence.py`` so future memory subsystems (e.g. the
session-feedback collector) can reuse them. This test module locks
in the exact semantics of the extracted helpers so the refactor is
verifiably behaviour-preserving.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from agent.memory.hashing import hash_session_id, iso_now


# ── hash_session_id ────────────────────────────────────────────────


class TestHashSessionId:
    """Behaviour mirrors the private ``_hash_session_id`` helper that
    previously lived in ``agent/nodes/crisis_log.py``."""

    def test_none_uses_placeholder(self) -> None:
        """Passing ``None`` hashes the literal ``"__no_session_id__"``."""
        expected = hashlib.sha256(b"__no_session_id__").hexdigest()
        assert hash_session_id(None) == expected

    def test_empty_string_uses_placeholder(self) -> None:
        """Empty string is treated identically to ``None`` — same
        placeholder, same hash."""
        assert hash_session_id("") == hash_session_id(None)

    def test_regular_string_hashes_raw_bytes(self) -> None:
        """Non-empty input is SHA-256 of its UTF-8 bytes directly."""
        expected = hashlib.sha256(b"abc").hexdigest()
        assert hash_session_id("abc") == expected

    def test_is_deterministic(self) -> None:
        """Same input always produces the same hash."""
        first = hash_session_id("thread-42")
        second = hash_session_id("thread-42")
        assert first == second

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Collision resistance sanity check — any two distinct inputs
        land on distinct hashes in practice."""
        assert hash_session_id("a") != hash_session_id("b")

    def test_unicode_input_encodes_as_utf8(self) -> None:
        """Non-ASCII input is handled via UTF-8, matching the previous
        implementation."""
        session = "测试-π"
        expected = hashlib.sha256(session.encode("utf-8")).hexdigest()
        assert hash_session_id(session) == expected

    def test_returns_64_char_hex_string(self) -> None:
        """SHA-256 output is always 64 lowercase hex characters."""
        result = hash_session_id("anything")
        assert re.fullmatch(r"[0-9a-f]{64}", result) is not None


# ── iso_now ────────────────────────────────────────────────────────


class TestIsoNow:
    """Behaviour mirrors the private ``_iso_now`` helper that previously
    lived in ``agent/persistence.py``. The ``Z`` suffix is a deliberate
    choice callers depend on."""

    def test_ends_with_z_suffix(self) -> None:
        """Callers and stored records rely on the ``Z`` suffix rather
        than the raw ``+00:00`` offset."""
        assert iso_now().endswith("Z")

    def test_does_not_contain_plus_offset(self) -> None:
        """If Python's default ``+00:00`` ever leaks through, the stored
        timestamp format would silently break for downstream readers."""
        assert "+00:00" not in iso_now()

    def test_round_trips_via_fromisoformat(self) -> None:
        """The string the helper emits must be parseable by the stdlib
        ``datetime.fromisoformat`` after swapping ``Z`` back for
        ``+00:00`` — the canonical ISO-8601 deserialization path."""
        raw = iso_now()
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_is_utc(self) -> None:
        """The helper always emits UTC — parsed back out, the UTC offset
        must be exactly zero."""
        parsed = datetime.fromisoformat(iso_now().replace("Z", "+00:00"))
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


# ── Backward-compat aliases ───────────────────────────────────────


class TestBackwardCompatAliases:
    """The refactor preserves the old private names at their original
    import sites so unmigrated callers continue to work."""

    def test_crisis_log_reexports_hash_session_id(self) -> None:
        """``from agent.nodes.crisis_log import _hash_session_id`` still
        works and resolves to the same callable."""
        from agent.nodes.crisis_log import _hash_session_id

        assert _hash_session_id is hash_session_id

    def test_persistence_reexports_iso_now(self) -> None:
        """``from agent.persistence import _iso_now`` still works and
        resolves to the shared helper."""
        from agent.persistence import _iso_now

        assert _iso_now is iso_now

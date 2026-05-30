"""Tests for the shared memory-subsystem helpers.

These helpers are shared by audit, feedback, and persistence code. This test
module locks in the exact semantics of the extracted helpers so refactors are
verifiably behaviour-preserving.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

import pytest

from agent.memory.hashing import extract_iso_date, hash_session_id, iso_now


# ── hash_session_id ────────────────────────────────────────────────


class TestHashSessionId:
    """Behaviour mirrors the private ``_hash_session_id`` helper that
    previously lived in crisis-log code."""

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
    """Behaviour mirrors the private runtime ``_iso_now`` helper.

    The ``Z`` suffix is a deliberate choice callers depend on.
    """

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


# ── extract_iso_date ───────────────────────────────────────────────


class TestExtractIsoDate:
    """Behaviour mirrors the four private ``_extract_date_prefix`` static
    methods previously duplicated across the crisis and feedback drivers.

    The ``date.fromisoformat`` validation is the load-bearing part: a bad
    timestamp must raise rather than silently land in a wrong date bucket.
    """

    def test_extracts_date_portion_from_full_timestamp(self) -> None:
        """A full ISO-8601 timestamp yields just the ``YYYY-MM-DD`` prefix."""
        assert extract_iso_date("2026-04-16T10:00:00Z") == "2026-04-16"

    def test_bare_date_is_returned_unchanged(self) -> None:
        """An already-bare date has no ``"T"`` to split on and round-trips
        unchanged — callers may pass either a timestamp or a date."""
        assert extract_iso_date("2026-04-16") == "2026-04-16"

    def test_malformed_prefix_raises_valueerror(self) -> None:
        """The ``date.fromisoformat`` side effect rejects a non-date prefix,
        preserving the fail-loud contract the driver copies provided."""
        with pytest.raises(ValueError):
            extract_iso_date("not-a-date")

    def test_invalid_calendar_date_raises_valueerror(self) -> None:
        """A syntactically date-shaped but impossible value (month 13) is
        still rejected — ``fromisoformat`` validates the calendar, not just
        the shape."""
        with pytest.raises(ValueError):
            extract_iso_date("2026-13-99T00:00:00Z")


# ── Public aliases ─────────────────────────────────────────────────


class TestPublicAliases:
    """Shared timestamp/hash helpers stay importable from runtime surfaces."""

    def test_crisis_log_uses_shared_hash_session_id(self) -> None:
        """Crisis logging uses the shared hash helper directly."""
        from agent.memory.hashing import hash_session_id as crisis_hash_session_id

        assert crisis_hash_session_id is hash_session_id

    def test_persistence_reexports_iso_now(self) -> None:
        """``from agent.runtime import _iso_now`` still works and
        resolves to the shared helper."""
        from agent.runtime import _iso_now

        assert _iso_now is iso_now

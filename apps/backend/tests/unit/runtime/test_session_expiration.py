"""Tests for pure active-session expiration policy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.runtime.session.state import parse_iso_timestamp, session_has_expired


@pytest.mark.parametrize("value", [None, "", "not-a-timestamp"])
def test_parse_iso_timestamp_returns_none_for_invalid_values(
    value: str | None,
) -> None:
    assert parse_iso_timestamp(value) is None


def test_parse_iso_timestamp_accepts_utc_z_suffix() -> None:
    assert parse_iso_timestamp("2026-07-17T12:30:00Z") == datetime(
        2026,
        7,
        17,
        12,
        30,
        tzinfo=timezone.utc,
    )


@pytest.mark.parametrize("last_active_at", [None, "", "not-a-timestamp"])
def test_session_has_expired_treats_invalid_timestamp_as_expired(
    last_active_at: str | None,
) -> None:
    assert session_has_expired(
        last_active_at,
        session_timeout=timedelta(minutes=30),
        now=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
    )


def test_session_has_expired_at_exact_timeout_boundary() -> None:
    assert session_has_expired(
        "2026-07-17T12:30:00Z",
        session_timeout=timedelta(minutes=30),
        now=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
    )


def test_session_has_expired_returns_false_before_timeout() -> None:
    assert not session_has_expired(
        "2026-07-17T12:30:01Z",
        session_timeout=timedelta(minutes=30),
        now=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
    )


def test_session_has_expired_respects_timezone_offsets() -> None:
    assert session_has_expired(
        "2026-07-17T14:30:00+02:00",
        session_timeout=timedelta(minutes=30),
        now=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    ("last_active_at", "now"),
    [
        ("2026-07-17T12:30:00Z", datetime(2026, 7, 17, 13, 0)),
        (
            "2026-07-17T12:30:00",
            datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
        ),
    ],
)
def test_session_has_expired_rejects_mismatched_timezone_awareness(
    last_active_at: str,
    now: datetime,
) -> None:
    with pytest.raises(
        ValueError,
        match="now timezone awareness must match last_active_at",
    ):
        session_has_expired(
            last_active_at,
            session_timeout=timedelta(minutes=30),
            now=now,
        )

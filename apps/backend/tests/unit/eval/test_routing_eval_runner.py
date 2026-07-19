"""Unit tests for the deterministic routing eval runner."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners import run_routing_eval as routing_eval  # noqa: E402


def test_score_expected_checks_working_memory_records_without_order_dependency() -> (
    None
):
    result: dict[str, Any] = {
        "working_memory": [
            {
                "evidence_quote": "Presentations make me anxious.",
                "category": "work",
            },
            {
                "evidence_quote": "I have a sister named Maya",
                "category": "relationship",
            },
        ]
    }
    checks: list[str] = []
    failures: list[str] = []

    routing_eval._score_expected(
        {
            "working_memory": {
                "min_count": 2,
                "max_count": 2,
                "must_include": [
                    {
                        "evidence_quote": "I have a sister named Maya",
                        "category": "relationship",
                    }
                ],
                "must_not_include": [{"evidence_quote": "I adopted a dog named Pixel"}],
            }
        },
        result=result,
        output={"response_text": ""},
        checks=checks,
        failures=failures,
        label_prefix="turn 1",
    )

    assert failures == []
    assert any("working_memory included" in check for check in checks)
    assert any("working_memory did not include forbidden" in check for check in checks)


def test_score_expected_fails_when_forbidden_working_memory_record_matches() -> None:
    result: dict[str, Any] = {
        "working_memory": [
            {
                "evidence_quote": "I adopted a dog named Pixel",
                "category": "pet",
            }
        ]
    }
    checks: list[str] = []
    failures: list[str] = []

    routing_eval._score_expected(
        {
            "working_memory": {
                "must_not_include": [{"evidence_quote": "I adopted a dog named Pixel"}]
            }
        },
        result=result,
        output={"response_text": ""},
        checks=checks,
        failures=failures,
        label_prefix="turn 1",
    )

    assert failures == [
        "turn 1 working_memory contained forbidden record "
        "{'evidence_quote': 'I adopted a dog named Pixel'}"
    ]

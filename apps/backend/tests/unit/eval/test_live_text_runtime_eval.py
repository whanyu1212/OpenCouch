"""Unit tests for the opt-in live text-runtime eval runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners import run_live_text_runtime_eval as live_eval  # noqa: E402


def test_load_cases_preserves_runtime_provider_and_turn_shape(tmp_path: Path) -> None:
    dataset = tmp_path / "live_cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "response_memory",
                "runtime": "response_llm",
                "providers": ["openai"],
                "memory_mode": "incognito",
                "user_id": "eval-user",
                "turns": [
                    {
                        "message": "I am anxious about presentations again.",
                        "memory_seed": [
                            {
                                "namespace": ["eval-user", "semantic"],
                                "key": "fact-presentations",
                                "value": {
                                    "evidence_quote": "Presentations make me anxious.",
                                    "category": "work",
                                },
                            }
                        ],
                        "expected": {"runtime_mode": "safe_therapeutic"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = live_eval._load_cases(dataset)

    assert len(cases) == 1
    assert cases[0].id == "response_memory"
    assert cases[0].runtime == "response_llm"
    assert cases[0].providers == ("openai",)
    assert cases[0].memory_mode.value == "incognito"
    assert cases[0].turns[0].message == "I am anxious about presentations again."
    assert cases[0].turns[0].memory_seed[0]["key"] == "fact-presentations"


def test_select_cases_keeps_openai_runtime_cases() -> None:
    cases = [
        live_eval.EvalCase(
            id="openai_sdk",
            runtime="agents_sdk",
            providers=("openai",),
            turns=[],
            memory_mode=live_eval.MemoryMode.LOCAL,
            user_id="eval-user",
            session_expected=None,
        ),
        live_eval.EvalCase(
            id="openai_response",
            runtime="response_llm",
            providers=("openai",),
            turns=[],
            memory_mode=live_eval.MemoryMode.LOCAL,
            user_id="eval-user",
            session_expected=None,
        ),
    ]

    selected = live_eval._select_cases(
        cases,
        case_ids=None,
        provider="openai",
    )

    assert [case.id for case in selected] == ["openai_sdk", "openai_response"]


def test_parse_providers_rejects_non_openai_provider() -> None:
    try:
        live_eval._parse_providers(["legacy"])
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:  # pragma: no cover - defensive assertion clarity
        raise AssertionError("expected ValueError for unsupported provider")


def test_score_expected_supports_live_runtime_quality_guards() -> None:
    output: dict[str, Any] = {
        "selected_agent": "OpenCouch guided exercise text agent",
        "route": "therapeutic",
        "runtime_mode": "guided_exercise",
        "response_style": "guided_exercise",
        "response_text": "Let us start with a slow inhale and keep this gentle.",
        "working_memory_count": 0,
        "crisis_log_count": 0,
        "diagnostics": {
            "openai_guided_exercise_tool_calls": ["load_guided_exercise_skill"],
            "openai_guided_exercise_tool_fallback": False,
        },
    }
    result: dict[str, Any] = {
        "exercise_state": {
            "exercise_type": "grounding_box_breathing",
            "exercise_step": 0,
        }
    }
    checks: list[str] = []
    failures: list[str] = []

    live_eval._score_expected(
        {
            "selected_agent": "OpenCouch guided exercise text agent",
            "route": "therapeutic",
            "runtime_mode": "guided_exercise",
            "response_text_min_chars": 20,
            "diagnostics_contains": {
                "openai_guided_exercise_tool_calls": "load_guided_exercise_skill"
            },
            "diagnostics": {"openai_guided_exercise_tool_fallback": False},
            "state": {
                "exercise_state.exercise_type": "grounding_box_breathing",
                "exercise_state.exercise_step": 0,
            },
            "must_not_include": ["Deterministic smoke mode"],
        },
        result=result,
        output=output,
        checks=checks,
        failures=failures,
        label_prefix="turn 1",
    )

    assert failures == []
    assert any("response_text length" in check for check in checks)
    assert any(
        "diagnostics.openai_guided_exercise_tool_calls contained" in check
        for check in checks
    )

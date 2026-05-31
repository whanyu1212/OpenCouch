"""Unit tests for the opt-in live text-runtime eval runner."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agent.memory.types import EntityRef, MemoryWrite

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.runners import run_live_text_runtime_eval as live_eval  # noqa: E402

TRAJECTORY_DATASET = (
    REPO_ROOT / "eval" / "datasets" / "live_text_runtime_trajectories.jsonl"
)


def test_dataset_paths_for_first_class_suites(tmp_path: Path) -> None:
    custom_dataset = tmp_path / "custom.jsonl"

    assert live_eval._dataset_paths_for_suite("smoke") == (live_eval.SMOKE_DATASET,)
    assert live_eval._dataset_paths_for_suite("trajectories") == (
        live_eval.TRAJECTORY_DATASET,
    )
    assert live_eval._dataset_paths_for_suite("memory_writes") == (
        live_eval.MEMORY_WRITE_DATASET,
    )
    assert live_eval._dataset_paths_for_suite("all") == (
        live_eval.SMOKE_DATASET,
        live_eval.TRAJECTORY_DATASET,
        live_eval.MEMORY_WRITE_DATASET,
    )
    assert live_eval._resolve_dataset_paths(
        dataset=custom_dataset,
        suite="all",
    ) == (custom_dataset,)


def test_load_cases_from_all_suite_combines_smoke_and_trajectories() -> None:
    cases = live_eval._load_cases_from_paths(live_eval._dataset_paths_for_suite("all"))

    assert len(cases) == 22
    assert cases[0].id == "openai_agents_sdk_therapeutic_smoke"
    assert (
        cases[-1].id == "memory_write_privacy_override_after_memory_like_statement_live"
    )


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


def test_live_trajectory_dataset_defines_openai_multiturn_cases() -> None:
    cases = live_eval._load_cases(TRAJECTORY_DATASET)

    assert [case.id for case in cases] == [
        "openai_agents_sdk_guided_exercise_resume_trajectory_live",
        "openai_agents_sdk_grounded_then_support_trajectory_live",
        "openai_agents_sdk_crisis_resource_trajectory_live",
        "openai_response_llm_persistent_memory_trajectory_live",
        "openai_response_llm_incognito_memory_trajectory_live",
    ]
    assert all(case.providers == ("openai",) for case in cases)
    assert all(len(case.turns) >= 2 for case in cases)
    assert all(case.session_expected for case in cases)

    cases_by_id = {case.id: case for case in cases}
    assert (
        cases_by_id["openai_response_llm_persistent_memory_trajectory_live"].runtime
        == "response_llm"
    )
    assert (
        cases_by_id[
            "openai_response_llm_incognito_memory_trajectory_live"
        ].memory_mode.value
        == "incognito"
    )
    assert all(
        case.runtime == "agents_sdk"
        for case_id, case in cases_by_id.items()
        if case_id.startswith("openai_agents_sdk")
    )
    guided_case = cases_by_id[
        "openai_agents_sdk_guided_exercise_resume_trajectory_live"
    ]
    assert guided_case.turns[1].expected["state"]["exercise_state.exercise_step"] == 0
    assert guided_case.turns[2].expected["state"]["exercise_state.exercise_step"] == 1
    for case_id in (
        "openai_response_llm_persistent_memory_trajectory_live",
        "openai_response_llm_incognito_memory_trajectory_live",
    ):
        for turn in cases_by_id[case_id].turns:
            forbidden = set(turn.expected["must_not_include"])
            assert "load_therapeutic_response_skill" in forbidden
            assert "<tool_call>" in forbidden


def test_live_memory_write_dataset_defines_saved_memory_quality_cases() -> None:
    cases = live_eval._load_cases(live_eval.MEMORY_WRITE_DATASET)

    assert [case.id for case in cases] == [
        "memory_write_presentation_anxiety_semantic_live",
        "memory_write_repeated_short_plan_preference_live",
        "memory_write_transient_mood_not_saved_live",
        "memory_write_single_self_belief_not_promoted_live",
        "memory_write_repeated_self_belief_promotes_live",
        "memory_write_incognito_no_durable_write_live",
        "memory_write_paraphrased_presentation_panic_semantic_live",
        "memory_write_paraphrased_procedural_preference_live",
        "memory_write_overlap_prefers_procedural_live",
        "memory_write_repeated_transient_exhaustion_not_saved_live",
        "memory_write_privacy_override_after_memory_like_statement_live",
    ]
    assert all(case.runtime == "response_llm" for case in cases)
    assert all(case.memory_write_expected for case in cases)
    incognito_case = {case.id: case for case in cases}[
        "memory_write_incognito_no_durable_write_live"
    ]
    assert (
        incognito_case.memory_mode.value == "incognito"
        and incognito_case.memory_write_expected["saved_memory_count"] == 0
    )


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


def test_score_memory_write_expected_checks_saved_record_quality() -> None:
    output: dict[str, Any] = {
        "saved_semantic_records": [
            {
                "category": "trigger",
                "predicate": "WORRIES_ABOUT",
                "object": {"type": "Concern", "identifier": "presentations"},
                "evidence_quote": "Presentations make me anxious.",
                "write_timing": "session_end",
            }
        ],
        "saved_procedural_records": [
            {
                "rule": "Use short step-by-step plans when the user feels anxious.",
                "evidence": ["Short step-by-step plans help me."],
                "write_timing": "promotion",
            }
        ],
        "saved_memory_count": 2,
        "held_memory_count": 0,
        "memory_commit_result": {
            "semantic_writes": 1,
            "procedural_writes": 1,
        },
        "postgres_reopen": {
            "saved_memory_count": 2,
        },
    }
    checks: list[str] = []
    failures: list[str] = []

    live_eval._score_memory_write_expected(
        {
            "saved_memory_count": 2,
            "held_memory_count": 0,
            "semantic_records": [
                {
                    "object_identifier_contains": "presentation",
                    "evidence_contains": "Presentations make me anxious",
                    "write_timing": "session_end",
                }
            ],
            "procedural_records": [
                {
                    "rule_contains": "short step-by-step plans",
                    "write_timing": "promotion",
                }
            ],
            "memory_commit_result": {
                "semantic_writes": 1,
                "procedural_writes": 1,
            },
            "postgres_reopen_saved_memory_count": 2,
        },
        output=output,
        checks=checks,
        failures=failures,
    )

    assert failures == []
    assert "memory_write saved_memory_count matched 2" in checks
    assert any("semantic record matched" in check for check in checks)
    assert any("procedural record matched" in check for check in checks)


def test_score_memory_write_expected_flags_forbidden_saved_memory() -> None:
    checks: list[str] = []
    failures: list[str] = []

    live_eval._score_memory_write_expected(
        {
            "saved_memory_count": 0,
            "must_not_save_semantic_object_contains": ["right now"],
        },
        output={
            "saved_semantic_records": [
                {
                    "object": {"identifier": "right now anxiety"},
                    "evidence_quote": "I feel anxious right now.",
                }
            ],
            "saved_procedural_records": [],
            "saved_memory_count": 1,
            "held_memory_count": 0,
        },
        checks=checks,
        failures=failures,
    )

    assert "memory_write saved_memory_count expected 0, got 1" in failures
    assert any("forbidden semantic object" in failure for failure in failures)


def test_memory_write_evidence_filter_keeps_only_user_grounded_quotes() -> None:
    user_turns = [
        "Can we slow that thought down for now?",
        "Please keep plans short for me when I am anxious.",
    ]

    assert live_eval._evidence_quote_is_user_grounded(
        user_turns,
        '"Can we slow that thought down for now?"',
    )
    assert not live_eval._evidence_quote_is_user_grounded(
        user_turns,
        "Let's take it one piece at a time.",
    )
    assert live_eval._filter_user_grounded_evidence(
        user_turns,
        [
            '"Can we slow that thought down for now?"',
            "Let's take it one piece at a time.",
            "Please keep plans short for me when I am anxious.",
        ],
    ) == [
        '"Can we slow that thought down for now?"',
        "Please keep plans short for me when I am anxious.",
    ]


def test_memory_write_semantic_normalization_uses_eval_owner_id() -> None:
    fact = MemoryWrite(
        category="trigger",
        subject=EntityRef(type="User", identifier="model-picked-wrong-id"),
        predicate="EXPERIENCED",
        object=EntityRef(type="Concern", identifier="presentations"),
        evidence_quote="Presentations make me anxious.",
        confidence="high",
        source_session_id="model-session",
        source_turn_index=3,
    )

    normalized = live_eval._normalize_semantic_fact_for_memory_write_eval(
        fact,
        owner_id="eval-owner",
        session_id="eval-session",
        user_turns=["Presentations make me anxious."],
    )

    assert normalized.subject.identifier == "eval-owner"
    assert normalized.source_session_id == "eval-session"
    assert normalized.source_turn_index == 0


def test_make_memory_store_defaults_to_in_memory_store() -> None:
    store = live_eval._make_memory_store(
        persistence_backend="memory",
        memory_database_url=None,
    )

    try:
        assert isinstance(store, live_eval.OpenCouchMemoryStore)
    finally:
        asyncio.run(store.aclose())


def test_make_memory_store_builds_postgres_store_when_configured() -> None:
    store = live_eval._make_memory_store(
        persistence_backend="postgres",
        memory_database_url="postgresql://opencouch:opencouch@localhost/opencouch",
    )

    try:
        assert isinstance(store, live_eval.PostgresMemoryStore)
        assert store.dsn == "postgresql://opencouch:opencouch@localhost/opencouch"
    finally:
        asyncio.run(store.aclose())


def test_make_memory_store_requires_postgres_dsn() -> None:
    with pytest.raises(ValueError, match="--memory-database-url"):
        live_eval._make_memory_store(
            persistence_backend="postgres",
            memory_database_url=None,
        )


def test_run_case_samples_aggregates_per_sample_judge_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = live_eval.EvalCase(
        id="sampled-case",
        runtime="agents_sdk",
        providers=("openai",),
        turns=[],
        memory_mode=live_eval.MemoryMode.LOCAL,
        user_id="eval-user",
        session_expected={"quality_focus": "stay coherent"},
    )
    calls: list[dict[str, Any]] = []

    async def fake_run_case(case_arg, **kwargs):
        sample_index = len(calls) + 1
        calls.append({"case": case_arg, **kwargs})
        return live_eval.EvalResult(
            id="sampled-case",
            runtime="agents_sdk",
            passed=sample_index == 1,
            checks=[f"sample {sample_index} deterministic checks passed"],
            failures=[]
            if sample_index == 1
            else ["judge continuity expected >= 4, got 3"],
            output={"turns": [{"turn": 1, "response_text": f"sample {sample_index}"}]},
            judge={"continuity": 5 if sample_index == 1 else 3},
        )

    monkeypatch.setattr(live_eval, "_run_case", fake_run_case)

    result = asyncio.run(
        live_eval._run_case_samples(
            case,
            live_client=object(),
            judge_client=object(),
            min_judge_score=4,
            openai_agent_model="gpt-test",
            samples=2,
            persistence_backend="postgres",
            memory_database_url="postgresql://example",
        )
    )

    assert len(calls) == 2
    assert all(call["persistence_backend"] == "postgres" for call in calls)
    assert all(call["memory_database_url"] == "postgresql://example" for call in calls)
    assert result.passed is False
    assert result.sample_count == 2
    assert result.failures == ["sample 2: judge continuity expected >= 4, got 3"]
    assert result.output == {"sample_count": 2}
    assert result.samples is not None
    assert result.samples[0]["sample"] == 1
    assert result.samples[0]["judge"] == {"continuity": 5}
    assert result.samples[1]["passed"] is False
    assert result.samples[1]["judge"] == {"continuity": 3}


def test_run_case_attaches_and_scores_memory_write_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = live_eval.EvalCase(
        id="memory-write-case",
        runtime="response_llm",
        providers=("openai",),
        turns=[
            live_eval.EvalTurn(
                message="Presentations make me anxious.",
                expected={"route": "therapeutic"},
                memory_seed=None,
                memory_mode=live_eval.MemoryMode.LOCAL,
                user_id="eval-user",
                prior_state=None,
            )
        ],
        memory_mode=live_eval.MemoryMode.LOCAL,
        user_id="eval-user",
        session_expected=None,
        memory_write_expected={"saved_memory_count": 1},
    )

    class FakeRuntime:
        def __init__(self, *, model: str) -> None:
            self.model = model

        async def run_turn(self, initial_state, **kwargs):
            return {
                **dict(initial_state),
                "route": "therapeutic",
                "response_text": "That sounds like a real presentation trigger.",
                "response_style": "supportive",
                "diagnostics": {
                    "openai_text_runtime_mode": "safe_therapeutic",
                    "openai_selected_agent": "OpenCouch therapeutic response agent",
                },
                "transcript": [
                    {"role": "user", "content": "Presentations make me anxious."},
                    {
                        "role": "assistant",
                        "content": "That sounds like a real presentation trigger.",
                    },
                ],
            }

    async def fake_run_memory_write_quality(**kwargs):
        return {
            "saved_semantic_records": [{"object": {"identifier": "presentations"}}],
            "saved_procedural_records": [],
            "saved_memory_count": 1,
            "saved_semantic_count": 1,
            "saved_procedural_count": 0,
            "held_memory_count": 0,
        }

    monkeypatch.setattr(live_eval, "OpenAITextRuntime", FakeRuntime)
    monkeypatch.setattr(
        live_eval,
        "_run_memory_write_quality",
        fake_run_memory_write_quality,
        raising=False,
    )

    result = asyncio.run(
        live_eval._run_case(
            case,
            live_client=object(),
            judge_client=None,
            min_judge_score=4,
            openai_agent_model="gpt-test",
        )
    )

    assert result.passed is True
    assert result.output["memory_write"]["saved_memory_count"] == 1
    assert "memory_write saved_memory_count matched 1" in result.checks


def test_run_case_attaches_memory_write_judge_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = live_eval.EvalCase(
        id="memory-write-judge-case",
        runtime="response_llm",
        providers=("openai",),
        turns=[
            live_eval.EvalTurn(
                message="Short plans help me.",
                expected={"route": "therapeutic"},
                memory_seed=None,
                memory_mode=live_eval.MemoryMode.LOCAL,
                user_id="eval-user",
                prior_state=None,
            )
        ],
        memory_mode=live_eval.MemoryMode.LOCAL,
        user_id="eval-user",
        session_expected=None,
        memory_write_expected={"saved_memory_count": 1},
    )

    class FakeRuntime:
        def __init__(self, *, model: str) -> None:
            self.model = model

        async def run_turn(self, initial_state, **kwargs):
            return {
                **dict(initial_state),
                "route": "therapeutic",
                "response_text": "A short plan makes sense.",
                "response_style": "supportive",
                "diagnostics": {
                    "openai_text_runtime_mode": "safe_therapeutic",
                    "openai_selected_agent": "OpenCouch therapeutic response agent",
                },
                "transcript": [
                    {"role": "user", "content": "Short plans help me."},
                    {"role": "assistant", "content": "A short plan makes sense."},
                ],
            }

    async def fake_run_memory_write_quality(**kwargs):
        return {
            "saved_semantic_records": [],
            "saved_procedural_records": [
                {"rule": "Use short plans.", "evidence": ["Short plans help me."]}
            ],
            "saved_memory_count": 1,
            "saved_semantic_count": 0,
            "saved_procedural_count": 1,
            "held_memory_count": 0,
        }

    async def fake_judge_memory_write_quality(**kwargs):
        return live_eval.MemoryWriteQualityJudgeResult(
            passes_quality_bar=True,
            memory_mode_respected=True,
            saved_memory_grounded=5,
            saved_memory_usefulness=5,
            saved_memory_specificity=4,
            saved_memory_sensitivity=5,
            no_transient_or_creepy_memory=True,
            rationale="saved rule is grounded and useful",
        )

    monkeypatch.setattr(live_eval, "OpenAITextRuntime", FakeRuntime)
    monkeypatch.setattr(
        live_eval,
        "_run_memory_write_quality",
        fake_run_memory_write_quality,
        raising=False,
    )
    monkeypatch.setattr(
        live_eval,
        "_judge_memory_write_quality",
        fake_judge_memory_write_quality,
        raising=False,
    )

    result = asyncio.run(
        live_eval._run_case(
            case,
            live_client=object(),
            judge_client=object(),
            min_judge_score=4,
            openai_agent_model="gpt-test",
        )
    )

    assert result.passed is True
    assert result.judge["memory_write"]["saved_memory_grounded"] == 5
    assert "memory_write judge quality bar passed" in result.checks


def test_serialize_result_includes_sample_payloads() -> None:
    result = live_eval.EvalResult(
        id="sampled-case",
        runtime="response_llm",
        passed=True,
        checks=["sample 1: ok", "sample 2: ok"],
        failures=[],
        output={"samples": []},
        judge=None,
        sample_count=2,
        samples=[
            {
                "sample": 1,
                "passed": True,
                "checks": ["ok"],
                "failures": [],
                "output": {"turns": []},
                "judge": {"continuity": 5},
            }
        ],
    )

    serialized = live_eval._serialize_result(result)

    assert serialized["sample_count"] == 2
    assert serialized["samples"][0]["judge"] == {"continuity": 5}

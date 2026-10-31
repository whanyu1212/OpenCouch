import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import httpx2
import pytest
from typesafe_sdk import Noul

from evals.crisis import record
from evals.crisis.cases import Case
from evals.crisis.record import build_meta, prepare_run, record_cases
from evals.crisis.runs import (
    CasesChangedError,
    ErrorRow,
    InputVersionMismatchError,
    RunCheckError,
    load_meta,
    read_errors,
    read_recorded,
    write_meta,
)
from opencouch_agent.safety import USER_TURN_QUESTIONS, TypeSafeJevClient
from opencouch_agent.safety.screening import user_check_state
from tests.safety.test_jev import answered


class FakeJev:
    """Answers every question with 0.1, except for texts it is told to fail."""

    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.states: list[str] = []

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        self.states.append(state)
        if any(text in state for text in self.failing):
            raise TimeoutError
        return dict.fromkeys(questions, 0.1)


CASES = [Case("a", "first message", "no_crisis"), Case("b", "second", "self_harm")]


def write_cases(path: Path, cases: list[Case]) -> Path:
    path.write_text("".join(json.dumps(asdict(case)) + "\n" for case in cases))
    return path


async def test_cases_are_sent_exactly_as_a_live_first_message(tmp_path: Path) -> None:
    jev = FakeJev()

    await record_cases(jev, CASES[:1], tmp_path, concurrency=1)

    assert jev.states == [user_check_state([], "first message")]
    [row] = read_recorded(tmp_path)
    assert row.probabilities == dict.fromkeys(USER_TURN_QUESTIONS, 0.1)
    assert row.label == "no_crisis"


async def test_failed_cases_go_to_errors_not_the_recording(tmp_path: Path) -> None:
    recorded, failed = await record_cases(
        FakeJev(failing={"second"}), CASES, tmp_path, concurrency=2
    )

    assert (recorded, failed) == (1, 1)
    assert [row.id for row in read_recorded(tmp_path)] == ["a"]
    assert read_errors(tmp_path) == [ErrorRow(id="b", error="TimeoutError")]


async def test_resuming_retries_only_what_is_missing(tmp_path: Path) -> None:
    await record_cases(FakeJev(failing={"second"}), CASES, tmp_path, concurrency=2)
    jev = FakeJev()

    recorded, failed = await record_cases(jev, CASES, tmp_path, concurrency=2)

    assert (recorded, failed) == (1, 0)
    assert jev.states == [user_check_state([], "second")]
    assert read_errors(tmp_path) == []
    assert sorted(row.id for row in read_recorded(tmp_path)) == ["a", "b"]


async def test_resume_refuses_to_mix_sdk_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_file = write_cases(tmp_path / "cases.jsonl", CASES)
    meta = build_meta("resume", cases_file, CASES, is_bhh=False)
    write_meta(tmp_path, meta.model_copy(update={"sdk_version": "old-sdk"}))
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(RunCheckError, match="different SDK version"):
        await record.record_jev(CASES, tmp_path, concurrency=1)
    assert load_meta(tmp_path).sdk_version == "old-sdk"


async def test_resume_merges_served_models_from_retried_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_file = write_cases(tmp_path / "cases.jsonl", CASES)
    meta = build_meta("resume", cases_file, CASES, is_bhh=False)
    write_meta(tmp_path, meta.model_copy(update={"served_models": ["jev-old"]}))
    await record_cases(FakeJev(failing={"second"}), CASES, tmp_path, concurrency=1)
    body = answered(dict.fromkeys(USER_TURN_QUESTIONS, 0.1)).json()
    body["model"] = "jev-new"
    client = TypeSafeJevClient(
        api_key="test-key",
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=body)),
    )
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr(record, "TypeSafeJevClient", lambda **_: client)

    assert await record.record_jev(CASES, tmp_path, concurrency=1) == (1, 0)

    resumed = load_meta(tmp_path)
    assert resumed.served_models == ["jev-new", "jev-old"]
    assert resumed.requested_model == "typesafe-ai/jev"
    assert resumed.sdk_version
    assert read_errors(tmp_path) == []


async def test_noop_resume_does_not_invent_provenance_for_old_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_file = write_cases(tmp_path / "cases.jsonl", CASES)
    meta = build_meta("complete", cases_file, CASES, is_bhh=False)
    write_meta(tmp_path, meta)
    await record_cases(FakeJev(), CASES, tmp_path, concurrency=1)
    client = TypeSafeJevClient(
        api_key="test-key",
        transport=httpx2.MockTransport(lambda _: pytest.fail("Unexpected request")),
    )
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    monkeypatch.setattr(record, "TypeSafeJevClient", lambda **_: client)

    assert await record.record_jev(CASES, tmp_path, concurrency=1) == (0, 0)
    assert load_meta(tmp_path) == meta


def test_a_new_run_stores_an_absolute_cases_path_and_no_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_file = write_cases(tmp_path / "bhh_extra.jsonl", CASES)
    monkeypatch.setattr(record, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.chdir(tmp_path)

    run_dir, cases = prepare_run(
        split=None, cases_path=Path("bhh_extra.jsonl"), resume=None
    )

    meta = load_meta(run_dir)
    assert meta.cases_path == str(cases_file.resolve())
    assert meta.dataset is None  # a name starting with "bhh" isn't BHH
    assert cases == CASES


def test_resuming_with_a_larger_limit_extends_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_file = write_cases(tmp_path / "cases.jsonl", CASES)
    monkeypatch.setattr(record, "RESULTS_DIR", tmp_path / "results")
    run_dir, first = prepare_run(
        split=None, cases_path=cases_file, resume=None, limit=1
    )

    _, extended = prepare_run(split=None, cases_path=None, resume=run_dir)

    assert (first, extended) == (CASES[:1], CASES)
    assert load_meta(run_dir).limit is None


def test_resuming_a_run_recorded_with_other_questions_is_refused(
    tmp_path: Path,
) -> None:
    cases_file = write_cases(tmp_path / "cases.jsonl", CASES)
    meta = build_meta("old", cases_file, CASES, is_bhh=False)
    write_meta(tmp_path, meta.model_copy(update={"jev_input_version": "stale"}))

    with pytest.raises(InputVersionMismatchError):
        prepare_run(split=None, cases_path=None, resume=tmp_path)


def test_resuming_a_run_whose_cases_changed_is_refused(tmp_path: Path) -> None:
    cases_file = write_cases(tmp_path / "cases.jsonl", CASES)
    write_meta(tmp_path, build_meta("old", cases_file, CASES, is_bhh=False))
    write_cases(cases_file, [Case("a", "first message, edited", "no_crisis")])

    with pytest.raises(CasesChangedError):
        prepare_run(split=None, cases_path=None, resume=tmp_path)

from collections.abc import Mapping
from pathlib import Path

import pytest
from typesafe_sdk import Noul

from evals.crisis import record
from evals.crisis.cases import Case
from evals.crisis.record import build_meta, prepare_run, record_cases
from evals.crisis.runs import (
    ErrorRow,
    InputVersionMismatchError,
    load_meta,
    read_errors,
    read_recorded,
    write_meta,
)
from opencouch_agent.safety import USER_TURN_QUESTIONS
from opencouch_agent.safety.screening import user_check_state


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


def test_a_new_run_stores_an_absolute_cases_path_and_no_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_file = tmp_path / "bhh_extra.jsonl"
    cases_file.write_text('{"id": "a", "text": "hi", "label": "none"}\n')
    monkeypatch.setattr(record, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.chdir(tmp_path)

    run_dir, cases = prepare_run(
        split=None, cases_path=Path("bhh_extra.jsonl"), resume=None
    )

    meta = load_meta(run_dir)
    assert meta.cases_path == str(cases_file.resolve())
    assert meta.dataset is None  # a name starting with "bhh" isn't BHH
    assert cases == [Case("a", "hi", "none")]


def test_resuming_a_run_recorded_with_other_questions_is_refused(
    tmp_path: Path,
) -> None:
    meta = build_meta("old", tmp_path / "cases.jsonl", is_bhh=False)
    write_meta(tmp_path, meta.model_copy(update={"jev_input_version": "stale"}))

    with pytest.raises(InputVersionMismatchError):
        prepare_run(split=None, cases_path=None, resume=tmp_path)

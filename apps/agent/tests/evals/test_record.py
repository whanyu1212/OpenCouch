from collections.abc import Mapping
from pathlib import Path

from typesafe_sdk import Noul

from evals.crisis.cases import Case
from evals.crisis.record import read_jsonl, record_cases
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
    [row] = read_jsonl(tmp_path / "recorded.jsonl")
    assert row["probabilities"] == dict.fromkeys(USER_TURN_QUESTIONS, 0.1)
    assert row["label"] == "no_crisis"


async def test_failed_cases_go_to_errors_not_the_recording(tmp_path: Path) -> None:
    recorded, failed = await record_cases(
        FakeJev(failing={"second"}), CASES, tmp_path, concurrency=2
    )

    assert (recorded, failed) == (1, 1)
    assert [r["id"] for r in read_jsonl(tmp_path / "recorded.jsonl")] == ["a"]
    assert read_jsonl(tmp_path / "errors.jsonl") == [
        {"id": "b", "error": "TimeoutError"}
    ]


async def test_resuming_retries_only_what_is_missing(tmp_path: Path) -> None:
    await record_cases(FakeJev(failing={"second"}), CASES, tmp_path, concurrency=2)
    jev = FakeJev()

    recorded, failed = await record_cases(jev, CASES, tmp_path, concurrency=2)

    assert (recorded, failed) == (1, 0)
    assert jev.states == [user_check_state([], "second")]
    assert read_jsonl(tmp_path / "errors.jsonl") == []
    assert sorted(r["id"] for r in read_jsonl(tmp_path / "recorded.jsonl")) == [
        "a",
        "b",
    ]

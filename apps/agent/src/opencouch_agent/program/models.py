"""The program's per-user records, and the rules every store keeps (HY-334)."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

WorrySource = Literal["quick_capture", "chat"]
WorryStatus = Literal["parked", "reviewed"]
WorrySort = Literal["solvable", "hypothetical"]
Questionnaire = Literal["gad7"]
FormulationKind = Literal["worry_theme", "safety_behavior"]

# Each questionnaire's number of items and highest answer per item. The item
# wording and what totals mean belong to the check-in itself (HY-338).
QUESTIONNAIRE_SHAPES: Final[dict[Questionnaire, tuple[int, int]]] = {"gad7": (7, 3)}


@dataclass(frozen=True)
class ProgramPosition:
    """Where a user is in the program.

    Only the program state machine (HY-226) should change it.

    Attributes:
        enrolled_at: When they started the program.
        week: The program week, from 1.
        step: The step within the week; names come with the program content.
    """

    enrolled_at: datetime
    week: int
    step: str

    def __post_init__(self) -> None:
        require_timezone(self.enrolled_at)
        if self.week < 1:
            raise ValueError(f"week must be 1 or more, not {self.week}")


@dataclass(frozen=True)
class Worry:
    """A worry the user caught, parked until worry time.

    Attributes:
        id: The worry's ID.
        text: What the user wrote.
        captured_at: When they caught it.
        source: Quick capture or a conversation with the coach.
        status: Parked until worry time, then reviewed.
        sort: How a reviewed worry was sorted. A reviewed worry with no sort
            no longer mattered by worry time.
        next_step: The one step agreed for a solvable worry.
    """

    id: str
    text: str
    captured_at: datetime
    source: WorrySource
    status: WorryStatus = "parked"
    sort: WorrySort | None = None
    next_step: str | None = None

    def __post_init__(self) -> None:
        require_timezone(self.captured_at)
        if self.status == "parked" and self.sort is not None:
            raise ValueError("a parked worry isn't sorted yet")
        check_review(self.sort, self.next_step)


@dataclass(frozen=True)
class CheckIn:
    """A completed questionnaire, such as the weekly GAD-7.

    Attributes:
        id: The check-in's ID.
        questionnaire: Which questionnaire.
        answers: The answer to each item, in item order.
        total: The sum of the answers.
        taken_at: When the user completed it.
    """

    id: str
    questionnaire: Questionnaire
    answers: tuple[int, ...]
    total: int
    taken_at: datetime

    def __post_init__(self) -> None:
        require_timezone(self.taken_at)
        check_answers(self.questionnaire, self.answers)
        if self.total != sum(self.answers):
            raise ValueError("a check-in's total is the sum of its answers")


@dataclass(frozen=True)
class FormulationEntry:
    """One entry in the user's worry map, which they confirmed (HY-339).

    Attributes:
        id: The entry's ID.
        kind: A worry theme or a safety behavior.
        text: The entry, in the user's confirmed wording.
        confirmed_at: When the user confirmed it.
    """

    id: str
    kind: FormulationKind
    text: str
    confirmed_at: datetime

    def __post_init__(self) -> None:
        require_timezone(self.confirmed_at)
        check_entry_text(self.text)


def new_record_id() -> str:
    """A fresh, unguessable record ID."""
    return str(uuid.uuid4())


def parked_worry(text: str, *, source: WorrySource, captured_at: datetime) -> Worry:
    """A new worry, parked until worry time."""
    return Worry(new_record_id(), text, captured_at, source)


def scored_check_in(
    questionnaire: Questionnaire, answers: tuple[int, ...], *, taken_at: datetime
) -> CheckIn:
    """A new check-in, with its total worked out from the answers."""
    # A tuple even if a list is passed, so the frozen record can't change.
    answers = tuple(answers)
    return CheckIn(new_record_id(), questionnaire, answers, sum(answers), taken_at)


def confirmed_entry(
    kind: FormulationKind, text: str, *, confirmed_at: datetime
) -> FormulationEntry:
    """A new worry-map entry the user has confirmed."""
    return FormulationEntry(new_record_id(), kind, text, confirmed_at)


def check_review(sort: WorrySort | None, next_step: str | None) -> None:
    """Check a worry-time outcome before applying it.

    Raises:
        ValueError: If a next step is given for a worry that isn't solvable.
    """
    if next_step is not None and sort != "solvable":
        raise ValueError("only a solvable worry has a next step")


def check_entry_text(text: str) -> None:
    """Check a formulation entry's wording.

    Raises:
        ValueError: If the text is blank.
    """
    if not text.strip():
        raise ValueError("a formulation entry needs some text")


def check_answers(questionnaire: Questionnaire, answers: tuple[int, ...]) -> None:
    """Check a questionnaire's answers have its shape.

    Raises:
        ValueError: If there are too many or too few answers, or one is out
            of range.
    """
    items, highest = QUESTIONNAIRE_SHAPES[questionnaire]
    if len(answers) != items:
        raise ValueError(f"{questionnaire} has {items} items, not {len(answers)}")
    # `bool` is an `int` in Python, but True isn't an answer.
    if any(type(answer) is not int for answer in answers):
        raise ValueError(f"{questionnaire} answers are whole numbers")
    if any(not 0 <= answer <= highest for answer in answers):
        raise ValueError(f"{questionnaire} answers run from 0 to {highest}")


def require_timezone(moment: datetime) -> None:
    """Refuse a time with no timezone, which would compare wrongly.

    Raises:
        ValueError: If `moment` has no timezone.
    """
    if moment.tzinfo is None:
        raise ValueError("record times must be timezone-aware")

"""The Postgres tables for program records, and the checks on rows read back."""

from pydantic import AwareDatetime, BaseModel

from opencouch_agent.program.models import (
    CheckIn,
    FormulationEntry,
    FormulationKind,
    ProgramPosition,
    Questionnaire,
    Worry,
    WorrySort,
    WorrySource,
    WorryStatus,
)

# No foreign keys to conversations: records belong to the user, not a thread.
# The CHECKs repeat the dataclasses' rules, so a bad row can't be written even
# by another client and then make a user's records unreadable. Each `seq`
# keeps insertion order, to break ties between records from the same moment.
# A check-in's total isn't stored: it's worked out from the answers on read,
# so the two can't disagree.
SCHEMA = """
CREATE TABLE IF NOT EXISTS program_positions (
    user_id     TEXT PRIMARY KEY,
    enrolled_at TIMESTAMPTZ NOT NULL,
    week        INTEGER NOT NULL CHECK (week >= 1),
    step        TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS worries (
    seq         BIGSERIAL,
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('quick_capture', 'chat')),
    status      TEXT NOT NULL CHECK (status IN ('parked', 'reviewed')),
    sort        TEXT CHECK (sort IN ('solvable', 'hypothetical')),
    next_step   TEXT,
    CHECK (status = 'reviewed' OR sort IS NULL),
    -- `sort IS NOT NULL` too: a NULL sort would make the comparison unknown,
    -- which a CHECK lets through.
    CHECK (next_step IS NULL OR (sort IS NOT NULL AND sort = 'solvable'))
);

CREATE INDEX IF NOT EXISTS worries_user_id_captured_at_idx
    ON worries (user_id, captured_at);

CREATE TABLE IF NOT EXISTS check_ins (
    seq           BIGSERIAL,
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    questionnaire TEXT NOT NULL CHECK (questionnaire IN ('gad7')),
    answers       INTEGER[] NOT NULL,
    taken_at      TIMESTAMPTZ NOT NULL,
    -- The GAD-7's shape (QUESTIONNAIRE_SHAPES): seven answers from 0 to 3.
    CHECK (questionnaire <> 'gad7' OR (
        cardinality(answers) = 7
        AND array_position(answers, NULL) IS NULL
        AND 0 <= ALL (answers) AND 3 >= ALL (answers)
    ))
);

CREATE INDEX IF NOT EXISTS check_ins_user_id_taken_at_idx
    ON check_ins (user_id, taken_at);

CREATE TABLE IF NOT EXISTS formulation_entries (
    seq          BIGSERIAL,
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('worry_theme', 'safety_behavior')),
    -- Some non-whitespace, as `check_entry_text` requires.
    text         TEXT NOT NULL CHECK (text ~ '\\S'),
    confirmed_at TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS formulation_entries_user_id_confirmed_at_idx
    ON formulation_entries (user_id, confirmed_at);
"""

WORRY_COLUMNS = "id, text, captured_at, source, status, sort, next_step"
CHECK_IN_COLUMNS = "id, questionnaire, answers, taken_at"
ENTRY_COLUMNS = "id, kind, text, confirmed_at"


class _PositionRow(BaseModel):
    enrolled_at: AwareDatetime
    week: int
    step: str


class _WorryRow(BaseModel):
    id: str
    text: str
    captured_at: AwareDatetime
    source: WorrySource
    status: WorryStatus
    sort: WorrySort | None
    next_step: str | None


class _CheckInRow(BaseModel):
    id: str
    questionnaire: Questionnaire
    answers: tuple[int, ...]
    taken_at: AwareDatetime


class _EntryRow(BaseModel):
    id: str
    kind: FormulationKind
    text: str
    confirmed_at: AwareDatetime


# Each row is checked by its model before it becomes a record: the database
# boundary rule in apps/agent/AGENTS.md.


def position_from(row: object) -> ProgramPosition:
    """A `program_positions` row as a `ProgramPosition`."""
    return ProgramPosition(**_PositionRow.model_validate(row).model_dump())


def worry_from(row: object) -> Worry:
    """A `worries` row as a `Worry`."""
    return Worry(**_WorryRow.model_validate(row).model_dump())


def check_in_from(row: object) -> CheckIn:
    """A `check_ins` row as a `CheckIn`, its total worked out from the answers."""
    checked = _CheckInRow.model_validate(row)
    return CheckIn(
        id=checked.id,
        questionnaire=checked.questionnaire,
        answers=checked.answers,
        total=sum(checked.answers),
        taken_at=checked.taken_at,
    )


def entry_from(row: object) -> FormulationEntry:
    """A `formulation_entries` row as a `FormulationEntry`."""
    return FormulationEntry(**_EntryRow.model_validate(row).model_dump())

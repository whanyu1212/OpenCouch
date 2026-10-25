"""Per-user program records that span conversations (HY-334).

Conversations are stored per thread; these belong to the user. A worry caught
at 2pm, worry time at 6pm and Sunday's weekly session are separate threads
reading the same records. Every call takes the user ID and only ever reads
or changes that user's records.
"""

from dataclasses import replace
from datetime import datetime
from typing import Protocol

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
    check_entry_text,
    check_review,
    confirmed_entry,
    parked_worry,
    require_timezone,
    scored_check_in,
)


class RecordStore(Protocol):
    """Persistence for per-user program records."""

    async def position(self, user_id: str) -> ProgramPosition | None:
        """The user's place in the program, or None if they haven't enrolled."""
        ...

    async def save_position(self, user_id: str, position: ProgramPosition) -> None:
        """Replace the user's place in the program."""
        ...

    async def add_worry(
        self, user_id: str, text: str, *, source: WorrySource, captured_at: datetime
    ) -> Worry:
        """Park a new worry for the user's next worry time."""
        ...

    async def worries(
        self,
        user_id: str,
        *,
        status: WorryStatus | None = None,
        since: datetime | None = None,
    ) -> list[Worry]:
        """The user's worries, oldest first.

        Args:
            user_id: Whose worries.
            status: Only worries with this status, if given.
            since: Only worries caught at or after this time, if given.
        """
        ...

    async def review_worry(
        self,
        user_id: str,
        worry_id: str,
        *,
        sort: WorrySort | None,
        next_step: str | None = None,
    ) -> Worry | None:
        """Mark one of the user's parked worries reviewed at worry time.

        A worry is reviewed once: a reviewed one is never overwritten.

        Args:
            user_id: Whose worry.
            worry_id: Which worry.
            sort: How it was sorted, or None if it no longer mattered.
            next_step: The agreed step, for a solvable worry only.

        Returns:
            The reviewed worry, or None if the user has no parked worry with
            that ID.

        Raises:
            ValueError: If a next step is given for a worry that isn't solvable.
        """
        ...

    async def delete_worry(self, user_id: str, worry_id: str) -> bool:
        """Delete one of the user's worries. Returns whether there was one."""
        ...

    async def add_check_in(
        self,
        user_id: str,
        questionnaire: Questionnaire,
        answers: tuple[int, ...],
        *,
        taken_at: datetime,
    ) -> CheckIn:
        """Save a completed questionnaire, with its total.

        Raises:
            ValueError: If the answers don't fit the questionnaire.
        """
        ...

    async def check_ins(
        self, user_id: str, *, questionnaire: Questionnaire | None = None
    ) -> list[CheckIn]:
        """The user's check-ins, oldest first; only one questionnaire's if given."""
        ...

    async def delete_check_in(self, user_id: str, check_in_id: str) -> bool:
        """Delete one of the user's check-ins. Returns whether there was one."""
        ...

    async def add_formulation_entry(
        self, user_id: str, kind: FormulationKind, text: str, *, confirmed_at: datetime
    ) -> FormulationEntry:
        """Save an entry the user confirmed for their worry map.

        Raises:
            ValueError: If the text is blank.
        """
        ...

    async def formulation(self, user_id: str) -> list[FormulationEntry]:
        """The user's worry map, oldest entry first."""
        ...

    async def edit_formulation_entry(
        self, user_id: str, entry_id: str, text: str
    ) -> FormulationEntry | None:
        """Change an entry's wording; None if the user has no entry with that ID.

        Raises:
            ValueError: If the text is blank.
        """
        ...

    async def delete_formulation_entry(self, user_id: str, entry_id: str) -> bool:
        """Delete one of the user's entries. Returns whether there was one."""
        ...

    async def delete_user(self, user_id: str) -> None:
        """Delete every program record the user has."""
        ...


class InMemoryRecordStore:
    """Record store backed by dicts. For tests and local runs only.

    Each user's records keep insertion order, which breaks ties the way the
    Postgres store's sequences do.
    """

    def __init__(self) -> None:
        self._positions: dict[str, ProgramPosition] = {}
        self._worries: dict[str, dict[str, Worry]] = {}
        self._check_ins: dict[str, dict[str, CheckIn]] = {}
        self._formulation: dict[str, dict[str, FormulationEntry]] = {}

    async def position(self, user_id: str) -> ProgramPosition | None:
        """The user's place in the program; see `RecordStore.position`."""
        return self._positions.get(user_id)

    async def save_position(self, user_id: str, position: ProgramPosition) -> None:
        """Replace the user's place; see `RecordStore.save_position`."""
        self._positions[user_id] = position

    async def add_worry(
        self, user_id: str, text: str, *, source: WorrySource, captured_at: datetime
    ) -> Worry:
        """Park a new worry; see `RecordStore.add_worry`."""
        worry = parked_worry(text, source=source, captured_at=captured_at)
        self._worries.setdefault(user_id, {})[worry.id] = worry
        return worry

    async def worries(
        self,
        user_id: str,
        *,
        status: WorryStatus | None = None,
        since: datetime | None = None,
    ) -> list[Worry]:
        """The user's worries; see `RecordStore.worries`."""
        if since is not None:
            require_timezone(since)
        found = [
            worry
            for worry in self._worries.get(user_id, {}).values()
            if (status is None or worry.status == status)
            and (since is None or worry.captured_at >= since)
        ]
        return sorted(found, key=lambda worry: worry.captured_at)

    async def review_worry(
        self,
        user_id: str,
        worry_id: str,
        *,
        sort: WorrySort | None,
        next_step: str | None = None,
    ) -> Worry | None:
        """Mark a worry reviewed; see `RecordStore.review_worry`."""
        check_review(sort, next_step)
        mine = self._worries.get(user_id, {})
        if worry_id not in mine or mine[worry_id].status != "parked":
            return None
        reviewed = replace(
            mine[worry_id], status="reviewed", sort=sort, next_step=next_step
        )
        mine[worry_id] = reviewed
        return reviewed

    async def delete_worry(self, user_id: str, worry_id: str) -> bool:
        """Delete a worry; see `RecordStore.delete_worry`."""
        return self._worries.get(user_id, {}).pop(worry_id, None) is not None

    async def add_check_in(
        self,
        user_id: str,
        questionnaire: Questionnaire,
        answers: tuple[int, ...],
        *,
        taken_at: datetime,
    ) -> CheckIn:
        """Save a check-in; see `RecordStore.add_check_in`."""
        check_in = scored_check_in(questionnaire, answers, taken_at=taken_at)
        self._check_ins.setdefault(user_id, {})[check_in.id] = check_in
        return check_in

    async def check_ins(
        self, user_id: str, *, questionnaire: Questionnaire | None = None
    ) -> list[CheckIn]:
        """The user's check-ins; see `RecordStore.check_ins`."""
        found = [
            check_in
            for check_in in self._check_ins.get(user_id, {}).values()
            if questionnaire is None or check_in.questionnaire == questionnaire
        ]
        return sorted(found, key=lambda check_in: check_in.taken_at)

    async def delete_check_in(self, user_id: str, check_in_id: str) -> bool:
        """Delete a check-in; see `RecordStore.delete_check_in`."""
        return self._check_ins.get(user_id, {}).pop(check_in_id, None) is not None

    async def add_formulation_entry(
        self, user_id: str, kind: FormulationKind, text: str, *, confirmed_at: datetime
    ) -> FormulationEntry:
        """Save an entry; see `RecordStore.add_formulation_entry`."""
        entry = confirmed_entry(kind, text, confirmed_at=confirmed_at)
        self._formulation.setdefault(user_id, {})[entry.id] = entry
        return entry

    async def formulation(self, user_id: str) -> list[FormulationEntry]:
        """The user's worry map; see `RecordStore.formulation`."""
        entries = self._formulation.get(user_id, {}).values()
        return sorted(entries, key=lambda entry: entry.confirmed_at)

    async def edit_formulation_entry(
        self, user_id: str, entry_id: str, text: str
    ) -> FormulationEntry | None:
        """Reword an entry; see `RecordStore.edit_formulation_entry`."""
        check_entry_text(text)
        mine = self._formulation.get(user_id, {})
        if entry_id not in mine:
            return None
        mine[entry_id] = replace(mine[entry_id], text=text)
        return mine[entry_id]

    async def delete_formulation_entry(self, user_id: str, entry_id: str) -> bool:
        """Delete an entry; see `RecordStore.delete_formulation_entry`."""
        return self._formulation.get(user_id, {}).pop(entry_id, None) is not None

    async def delete_user(self, user_id: str) -> None:
        """Delete everything; see `RecordStore.delete_user`."""
        for records in (
            self._positions,
            self._worries,
            self._check_ins,
            self._formulation,
        ):
            records.pop(user_id, None)

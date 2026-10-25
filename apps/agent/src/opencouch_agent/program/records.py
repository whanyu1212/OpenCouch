"""Per-user program records that span conversations (HY-334).

Conversations are stored per thread; these belong to the user. A worry caught
at 2pm, worry time at 6pm and Sunday's weekly session are separate threads
reading the same records. Every call takes the user ID and only ever reads
or changes that user's records.
"""

import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal, Protocol

WorrySource = Literal["quick_capture", "chat"]
WorryStatus = Literal["parked", "reviewed"]
WorrySort = Literal["solvable", "hypothetical"]


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

    async def delete_user(self, user_id: str) -> None:
        """Delete every program record the user has."""
        ...


class InMemoryRecordStore:
    """Record store backed by dicts. For tests and local runs only."""

    def __init__(self) -> None:
        self._positions: dict[str, ProgramPosition] = {}
        self._worries: dict[str, dict[str, Worry]] = {}

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
        # Insertion order breaks ties, as Postgres's sequence does.
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

    async def delete_user(self, user_id: str) -> None:
        """Delete everything; see `RecordStore.delete_user`."""
        self._positions.pop(user_id, None)
        self._worries.pop(user_id, None)


def parked_worry(text: str, *, source: WorrySource, captured_at: datetime) -> Worry:
    """A new worry, parked until worry time, with a fresh unguessable ID."""
    return Worry(str(uuid.uuid4()), text, captured_at, source)


def check_review(sort: WorrySort | None, next_step: str | None) -> None:
    """Check a worry-time outcome before applying it.

    Raises:
        ValueError: If a next step is given for a worry that isn't solvable.
    """
    if next_step is not None and sort != "solvable":
        raise ValueError("only a solvable worry has a next step")


def require_timezone(moment: datetime) -> None:
    """Refuse a time with no timezone, which would compare wrongly.

    Raises:
        ValueError: If `moment` has no timezone.
    """
    if moment.tzinfo is None:
        raise ValueError("record times must be timezone-aware")

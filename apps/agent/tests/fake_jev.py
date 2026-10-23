"""A scripted stand-in for Jev, keyed by the latest user message."""

from collections.abc import Mapping

from typesafe_sdk import Noul

from opencouch_agent.safety.screening import user_check_state

# "LATEST USER MESSAGE:\n", taken from the real framing so the two can't drift.
_LATEST_MARKER = user_check_state([], "").rsplit("\n\n", 1)[1]


class ScriptedJev:
    """Answers from a script of readings per message; 0 for anything else.

    Attributes:
        screened: Every latest message Jev was asked about, in order.
    """

    def __init__(
        self,
        readings: Mapping[str, Mapping[str, float]] | None = None,
        *,
        failing: frozenset[str] = frozenset(),
    ) -> None:
        self._readings = readings or {}
        self._failing = failing
        self.screened: list[str] = []

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Answer for the message after the transcript's latest-message marker."""
        latest = state.rsplit(_LATEST_MARKER, 1)[1]
        self.screened.append(latest)
        if latest in self._failing:
            raise RuntimeError("Jev is unavailable")
        reading = self._readings.get(latest, {})
        return {name: reading.get(name, 0.0) for name in questions}

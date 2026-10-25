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
        failing: Messages Jev errors on. Change it to bring Jev back.
        timing_out: Messages Jev times out on.
    """

    def __init__(
        self,
        readings: Mapping[str, Mapping[str, float]] | None = None,
        *,
        failing: frozenset[str] = frozenset(),
        timing_out: frozenset[str] = frozenset(),
    ) -> None:
        self._readings = readings or {}
        self.failing = set(failing)
        self.timing_out = set(timing_out)
        self.screened: list[str] = []

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Answer for the message after the transcript's latest-message marker."""
        latest = state.rsplit(_LATEST_MARKER, 1)[1]
        self.screened.append(latest)
        if latest in self.failing:
            raise RuntimeError("Jev is unavailable")
        if latest in self.timing_out:
            raise TimeoutError
        reading = self._readings.get(latest, {})
        return {name: reading.get(name, 0.0) for name in questions}

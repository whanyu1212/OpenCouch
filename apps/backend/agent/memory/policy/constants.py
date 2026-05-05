"""Shared constants for the memory write pipeline.

These marker tuples and helpers are used by both
:mod:`agent.memory.candidates` (candidate promotion) and
:mod:`agent.memory.write_policy` (deterministic commit/hold/drop).
Keeping them in a single dependency-free module avoids duplication
and prevents circular imports between those two modules.
"""

from __future__ import annotations

from dataclasses import dataclass


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    """Return whether text contains any of the given marker substrings.

    Args:
        text (str): Text to scan.
        markers (tuple[str, ...]): Marker substrings to check for.

    Returns:
        bool: ``True`` when any marker appears in ``text``.
    """
    return any(marker in text for marker in markers)


@dataclass(frozen=True, slots=True)
class ProceduralRequestClassification:
    """Deterministic marker-based classification of a procedural request."""

    explicit: bool
    turn_scoped: bool
    safety_conflict: bool


PROCEDURAL_EXPLICIT_REQUEST_MARKERS: tuple[str, ...] = (
    "please ",
    "please,",
    "can you",
    "could you",
    "would you",
    "i want you to",
    "i need you to",
    "don't ",
    "do not ",
    "stop ",
    "keep ",
    "it helps when you",
    "i prefer you",
)

PROCEDURAL_TURN_SCOPED_MARKERS: tuple[str, ...] = (
    "for this reply",
    "for this response",
    "for this one",
    "just this once",
    "just for now",
    "for now",
    "this time",
    "next reply",
    "next response",
)

PROCEDURAL_SAFETY_CONFLICT_MARKERS: tuple[str, ...] = (
    "don't ask if i'm safe",
    "dont ask if i'm safe",
    "don't ask if im safe",
    "dont ask if im safe",
    "stop asking if i'm safe",
    "stop asking if im safe",
    "skip the safety check",
    "ignore safety",
    "don't give me crisis resources",
    "dont give me crisis resources",
    "don't mention 988",
    "dont mention 988",
)


def classify_procedural_request(text: str) -> ProceduralRequestClassification:
    """Classify procedural-request text using the shared marker sets.

    Args:
        text (str): Combined procedural-request text to classify.

    Returns:
        ProceduralRequestClassification: Marker-based procedural request signals.
    """

    lowered = text.lower()
    return ProceduralRequestClassification(
        explicit=contains_any(lowered, PROCEDURAL_EXPLICIT_REQUEST_MARKERS),
        turn_scoped=contains_any(lowered, PROCEDURAL_TURN_SCOPED_MARKERS),
        safety_conflict=contains_any(lowered, PROCEDURAL_SAFETY_CONFLICT_MARKERS),
    )

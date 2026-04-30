"""Deterministic semantic backstops for memory extraction."""

from __future__ import annotations

import re

from agent.memory.models import EntityRef, MemoryWrite

_HELPFUL_PERSON_RE = re.compile(r"\b(?P<name>[A-Z][a-zA-Z'-]{1,60})\s+helped me\b")
_PHD_CONTEXT_RE = re.compile(
    r"\b(?:i'?m|i am)\s+a\s+phd student\b(?P<context>[^.!?]*)",
    re.IGNORECASE,
)
_PERFECTIONISM_TRIGGER_RE = re.compile(
    r"\bcan'?t stand\b.{0,80}\b(?:isn'?t|is not|not)\s+perfect\b",
    re.IGNORECASE,
)


def get_deterministic_semantic_backstops(
    *,
    message: str,
    session_id: str | None,
    turn_index: int,
) -> list[MemoryWrite]:
    """Return high-precision semantic backstop writes for one message."""
    message = message.strip()
    lowered = message.lower()
    session_id = session_id or "__no_session__"
    subject = EntityRef(type="User", identifier="self")
    backstops: list[MemoryWrite] = []

    if "my therapist" in lowered:
        backstops.append(
            MemoryWrite(
                category="relationship",
                subject=subject,
                predicate="KNOWS",
                object=EntityRef(type="Person", identifier="therapist"),
                evidence_quote=message[:280],
                confidence="medium",
                source_session_id=session_id,
                source_turn_index=turn_index,
            )
        )

    helpful_person = _HELPFUL_PERSON_RE.search(message)
    if helpful_person is not None:
        backstops.append(
            MemoryWrite(
                category="relationship",
                subject=subject,
                predicate="KNOWS",
                object=EntityRef(
                    type="Person", identifier=helpful_person.group("name")
                ),
                evidence_quote=message[:280],
                confidence="medium",
                source_session_id=session_id,
                source_turn_index=turn_index,
            )
        )

    phd_context = _PHD_CONTEXT_RE.search(message)
    if phd_context is not None:
        backstops.append(
            MemoryWrite(
                category="context",
                subject=subject,
                predicate="EXPERIENCED",
                object=EntityRef(
                    type="Event",
                    identifier=f"PhD student{phd_context.group('context')}".strip(),
                ),
                evidence_quote=message[:280],
                confidence="medium",
                source_session_id=session_id,
                source_turn_index=turn_index,
            )
        )

    if _PERFECTIONISM_TRIGGER_RE.search(message):
        backstops.append(
            MemoryWrite(
                category="trigger",
                subject=subject,
                predicate="WORRIES_ABOUT",
                object=EntityRef(type="Concern", identifier="work not being perfect"),
                evidence_quote=message[:280],
                confidence="medium",
                source_session_id=session_id,
                source_turn_index=turn_index,
            )
        )

    return backstops

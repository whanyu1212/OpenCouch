"""Tests for the v0.8.2 pre-extractor small-talk gate.

The gate is a pure function with two inputs (message string → bool).
Tests are organized into two groups:

1. **True cases (small talk):** messages the gate should classify as
   small talk, causing the extractor to skip the LLM call. These are
   the "obvious wins" that save ~3-5s per turn.

2. **False cases (pass through):** messages the gate should NOT
   classify as small talk, letting the extractor run its full LLM
   pipeline. This group is the load-bearing regression guard — a
   false positive here means silent memory loss.

Design principle: when in doubt, add it to the False group. A missed
optimization (false negative) wastes one LLM call. A missed memory
write (false positive) is unrecoverable.
"""

import pytest

from agent.memory.policy.small_talk import (
    MAX_SMALL_TALK_LENGTH,
    SMALL_TALK_VOCABULARY,
    is_small_talk,
)


class TestSmallTalkTruePositives:
    """Messages that ARE unambiguously small talk.

    Every case here should return True, meaning the extractor skips
    the LLM call. If any of these start returning False after a
    vocabulary change, the gate is being too restrictive (losing its
    optimization benefit).
    """

    @pytest.mark.parametrize(
        "message",
        [
            "hi",
            "hello",
            "hey",
            "Hi!",
            "Hello there",
            "thanks",
            "Thank you",
            "thanks so much",
            "thx",
            "ok",
            "okay",
            "OK",
            "yeah",
            "yep",
            "nope",
            "sure",
            "got it",
            "makes sense",
            "sounds good",
            "cool",
            "nice",
            "great",
            "go ahead",
            "ready",
            "alright",
        ],
    )
    def test_small_talk_detected(self, message: str) -> None:
        assert is_small_talk(message) is True, (
            f"Expected True for {message!r} — this is unambiguous small "
            f"talk and the gate should skip the LLM call."
        )


class TestSmallTalkFalseNegatives:
    """Messages that are NOT small talk — the extractor MUST run.

    Every case here should return False, meaning the extractor
    proceeds with the full LLM call. **This is the critical group.**
    A regression here (returning True) means silent memory loss.
    When adding cases, prefer messages that are "close to the
    boundary" — short messages with one topical word that the gate
    might incorrectly classify.
    """

    @pytest.mark.parametrize(
        "message,reason",
        [
            # Named entities — these carry identity signal the extractor needs
            ("hello Sarah", "proper noun 'Sarah' is outside small-talk vocab"),
            ("thanks for the meditation tip", "topical word 'meditation'"),
            ("hi, my sister is visiting", "relationship content 'sister'"),
            # Emotional content — the extractor might capture mood triggers
            ("I feel stuck", "emotional content 'stuck'"),
            ("I'm anxious", "emotional content 'anxious'"),
            ("I feel better today", "emotional content 'better', 'today'"),
            ("things are rough", "emotional content 'rough'"),
            # Memory-worthy short statements
            ("my dog Max died", "named entity 'Max' + event 'died'"),
            ("I take fluoxetine", "medication name 'fluoxetine'"),
            ("work is hard", "topical word 'work'"),
            # Procedural requests — the procedural writer must see these
            (
                "please be more direct",
                "explicit style request 'direct'",
            ),
            (
                "stop suggesting meditation",
                "explicit style request 'meditation'",
            ),
            # Long messages — always pass through regardless of vocabulary
            (
                "thank you so much for everything that really helped me today",
                "over MAX_SMALL_TALK_LENGTH even though it's grateful",
            ),
            (
                "ok sounds good I appreciate that a lot honestly",
                "over MAX_SMALL_TALK_LENGTH",
            ),
            # Edge: message with ONLY stopwords (no meaningful tokens at all)
            # — we let these through rather than making assumptions
            ("I am so", "pure stopwords — degenerate, let the LLM decide"),
            ("", "empty message — handled upstream"),
            ("   ", "whitespace-only — treated as empty"),
        ],
    )
    def test_not_small_talk(self, message: str, reason: str) -> None:
        assert is_small_talk(message) is False, (
            f"Expected False for {message!r} — {reason}. "
            f"Returning True here would silently skip extraction."
        )


class TestGateConstants:
    """Pin the gate's configuration constants so changes are intentional."""

    def test_max_length_is_40(self) -> None:
        """The 40-char threshold was calibrated against dogfood messages.
        Changing it requires updating the module docstring and re-
        verifying the boundary cases in this test file."""

        assert MAX_SMALL_TALK_LENGTH == 40

    def test_vocabulary_is_frozen(self) -> None:
        """The vocabulary must be a frozenset for immutability. A
        regular set would be a mutation bug waiting to happen."""

        assert isinstance(SMALL_TALK_VOCABULARY, frozenset)

    def test_vocabulary_has_no_emotional_words(self) -> None:
        """Emotional words like 'sad', 'angry', 'anxious', 'happy' must
        NOT be in the small-talk vocabulary because they carry extractable
        signal. This test pins the most common emotional words as
        explicitly absent."""

        emotional_words = {
            "sad",
            "happy",
            "angry",
            "anxious",
            "worried",
            "scared",
            "frustrated",
            "overwhelmed",
            "depressed",
            "stressed",
            "upset",
            "hurt",
            "lonely",
            "afraid",
            "nervous",
            "hopeless",
            "stuck",
            "fine",
            "bad",
        }
        overlap = emotional_words & SMALL_TALK_VOCABULARY
        assert overlap == set(), (
            f"Emotional words must NOT be in SMALL_TALK_VOCABULARY: {overlap}. "
            f"These carry extractable signal and would cause silent memory loss."
        )

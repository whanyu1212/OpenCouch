from __future__ import annotations

import pytest

from agent.voice.safety_proof import (
    InvalidVoiceSafetyInterruptionProof,
    VoiceSafetyInterruptionProofService,
)


def test_interruption_proof_binds_trigger_identity_and_memory_mode() -> None:
    service = VoiceSafetyInterruptionProofService(
        secret=b"test-secret",
        clock=lambda: 100.0,
    )
    token = service.issue(
        thread_id="thread-1",
        client_turn_id="trigger-turn",
        user_text="I might hurt myself.",
        user_id="user-1",
        memory_mode="persistent",
        risk_level=3,
    )

    proof = service.verify(
        token,
        thread_id="thread-1",
        client_turn_id="trigger-turn",
        user_text="I might hurt myself.",
        user_id="user-1",
        memory_mode="persistent",
    )

    assert proof.risk_level == 3


def test_interruption_proof_allows_expiry_only_for_authorized_retry() -> None:
    now = 100.0
    service = VoiceSafetyInterruptionProofService(
        secret=b"test-secret",
        ttl_seconds=1,
        clock=lambda: now,
    )
    token = service.issue(
        thread_id="thread-1",
        client_turn_id="trigger-turn",
        user_text="I might hurt myself.",
        user_id="user-1",
        memory_mode="persistent",
        risk_level=3,
    )
    now = 102.0

    with pytest.raises(InvalidVoiceSafetyInterruptionProof, match="expired proof"):
        service.verify(
            token,
            thread_id="thread-1",
            client_turn_id="trigger-turn",
            user_text="I might hurt myself.",
            user_id="user-1",
            memory_mode="persistent",
        )

    proof = service.verify(
        token,
        thread_id="thread-1",
        client_turn_id="trigger-turn",
        user_text="I might hurt myself.",
        user_id="user-1",
        memory_mode="persistent",
        allow_expired=True,
    )

    assert proof.risk_level == 3
    with pytest.raises(InvalidVoiceSafetyInterruptionProof, match="mismatched proof"):
        service.verify(
            token,
            thread_id="thread-1",
            client_turn_id="trigger-turn",
            user_text="Different text.",
            user_id="user-1",
            memory_mode="persistent",
            allow_expired=True,
        )


@pytest.mark.parametrize(
    ("thread_id", "client_turn_id", "user_text", "user_id", "memory_mode"),
    [
        (
            "other-thread",
            "trigger-turn",
            "I might hurt myself.",
            "user-1",
            "persistent",
        ),
        ("thread-1", "later-turn", "I might hurt myself.", "user-1", "persistent"),
        ("thread-1", "trigger-turn", "different text", "user-1", "persistent"),
        ("thread-1", "trigger-turn", "I might hurt myself.", "user-2", "persistent"),
        ("thread-1", "trigger-turn", "I might hurt myself.", "user-1", "incognito"),
    ],
)
def test_interruption_proof_rejects_mismatched_claims(
    thread_id: str,
    client_turn_id: str,
    user_text: str,
    user_id: str,
    memory_mode: str,
) -> None:
    service = VoiceSafetyInterruptionProofService(
        secret=b"test-secret",
        clock=lambda: 100.0,
    )
    token = service.issue(
        thread_id="thread-1",
        client_turn_id="trigger-turn",
        user_text="I might hurt myself.",
        user_id="user-1",
        memory_mode="persistent",
        risk_level=2,
    )

    with pytest.raises(InvalidVoiceSafetyInterruptionProof):
        service.verify(
            token,
            thread_id=thread_id,
            client_turn_id=client_turn_id,
            user_text=user_text,
            user_id=user_id,
            memory_mode=memory_mode,
        )

"""Authenticated proof that the server authorized a voice interruption."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass


_DEFAULT_TTL_SECONDS = 15 * 60
_PROCESS_SECRET = secrets.token_bytes(32)


class InvalidVoiceSafetyInterruptionProof(ValueError):
    """Raised when an interruption proof is invalid, expired, or mismatched."""


@dataclass(frozen=True, slots=True)
class VoiceSafetyInterruptionProof:
    """Trusted facts recovered from a valid interruption proof."""

    risk_level: int


class VoiceSafetyInterruptionProofService:
    """Issue and verify short-lived, payload-bound interruption proofs."""

    def __init__(
        self,
        *,
        secret: bytes | None = None,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        configured_secret = (
            os.getenv("VOICE_SAFETY_SIGNING_SECRET")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).encode()
        self._secret = secret or configured_secret or _PROCESS_SECRET
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    def issue(
        self,
        *,
        thread_id: str,
        client_turn_id: str,
        user_text: str,
        user_id: str | None,
        memory_mode: str,
        risk_level: int,
    ) -> str:
        """Return a signed proof without exposing transcript or thread content."""

        payload = {
            "v": 1,
            "exp": int(self._clock()) + self._ttl_seconds,
            "thread": _digest(thread_id),
            "trigger": _digest(f"{thread_id}\0{client_turn_id}"),
            "user": _digest(user_text.strip()),
            "owner": _digest(user_id or ""),
            "memory": memory_mode,
            "risk": risk_level,
        }
        encoded = _encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
        signature = _encode(hmac.digest(self._secret, encoded.encode(), "sha256"))
        return f"{encoded}.{signature}"

    def verify(
        self,
        token: str,
        *,
        thread_id: str,
        client_turn_id: str,
        user_text: str,
        user_id: str | None,
        memory_mode: str,
        allow_expired: bool = False,
    ) -> VoiceSafetyInterruptionProof:
        """Validate a proof for its thread and identify the triggering turn."""

        try:
            encoded, supplied_signature = token.split(".", 1)
            expected_signature = _encode(
                hmac.digest(self._secret, encoded.encode(), "sha256")
            )
            if not hmac.compare_digest(supplied_signature, expected_signature):
                raise InvalidVoiceSafetyInterruptionProof("invalid signature")
            payload = json.loads(_decode(encoded))
            expires_at = int(payload["exp"])
            risk_level = int(payload["risk"])
        except InvalidVoiceSafetyInterruptionProof:
            raise
        except Exception as exc:
            raise InvalidVoiceSafetyInterruptionProof("malformed proof") from exc

        if payload.get("v") != 1 or (
            expires_at < int(self._clock()) and not allow_expired
        ):
            raise InvalidVoiceSafetyInterruptionProof("expired proof")
        if (
            payload.get("thread") != _digest(thread_id)
            or payload.get("trigger") != _digest(f"{thread_id}\0{client_turn_id}")
            or payload.get("user") != _digest(user_text.strip())
            or payload.get("owner") != _digest(user_id or "")
            or payload.get("memory") != memory_mode
            or risk_level not in {2, 3}
        ):
            raise InvalidVoiceSafetyInterruptionProof("mismatched proof")
        return VoiceSafetyInterruptionProof(risk_level=risk_level)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


__all__ = [
    "InvalidVoiceSafetyInterruptionProof",
    "VoiceSafetyInterruptionProof",
    "VoiceSafetyInterruptionProofService",
]

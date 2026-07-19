"""Runtime configuration for agent tracing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Session-level tracing controls.

    Attributes:
        enabled: Whether tracing should emit spans/events.
        capture_debug_state: Whether compact diagnostics may be written to state.
        capture_model_io: Whether raw model inputs/outputs may be captured. This is
            intentionally disabled by default for privacy.
        exporters: Named exporter backends selected for the session.
        sample_rate: Fraction of eligible sessions to trace, from 0.0 to 1.0.
    """

    enabled: bool = False
    capture_debug_state: bool = False
    capture_model_io: bool = False
    exporters: tuple[str, ...] = ()
    sample_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate sampling configuration."""

        if not 0.0 <= self.sample_rate <= 1.0:
            msg = "TraceConfig.sample_rate must be between 0.0 and 1.0"
            raise ValueError(msg)

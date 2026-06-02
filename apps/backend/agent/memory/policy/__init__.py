"""Memory write policy helpers.

This package houses the decision layer that sits between memory candidates and
persisted writes:

- ``candidates`` — promote extractor outputs into ``MemoryCandidate`` instances.
- ``semantic`` — semantic policy constants for session-only categories.
- ``markers`` — text-cue helpers for memory-control, scoping, and repetition guards.
- ``thresholds`` — promotion thresholds for held session candidates.
- ``prompts`` — LLM classifier schemas and prompt builders.
- ``clamps`` — deterministic safety/product overrides for LLM decisions.
- ``write`` — the LLM-primary write-policy orchestration facade.
"""

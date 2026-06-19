"""Memory write policy helpers.

This package houses the decision layer that sits between memory candidates and
persisted writes:

- ``candidates`` — promote extractor outputs into ``MemoryCandidate`` instances.
- ``markers`` — text-cue helpers for memory-control, scoping, and repetition guards.
- ``thresholds`` — promotion thresholds for held session candidates.
- ``clamps`` — the non-LLM hard guard for semantic write decisions.
- ``write`` — re-exports the surviving guard/marker/threshold helpers.
"""

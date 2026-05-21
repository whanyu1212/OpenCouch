"""Memory write policy helpers.

This package houses the decision layer that sits between memory candidates and
persisted writes:

- ``candidates`` — promote extractor outputs into ``MemoryCandidate`` instances.
- ``semantic`` — semantic policy constants for session-only categories.
- ``write`` — the LLM-primary write-policy gate.
"""

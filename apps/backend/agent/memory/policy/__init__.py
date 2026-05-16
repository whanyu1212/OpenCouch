"""Memory write policy and turn-routing helpers.

This package houses the decision layer that sits between extracted memory
candidates and persisted writes:

- ``candidates`` — promote extractor outputs into ``MemoryCandidate`` instances.
- ``semantic`` — semantic policy constants for session-only categories.
- ``small_talk`` — lightweight discourse filter that suppresses extraction.
- ``write`` — the LLM-primary write-policy gate.
- ``turn_routing`` — turn-level skip/index helpers for memory side-effect nodes.
"""

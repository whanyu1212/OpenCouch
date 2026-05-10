"""Memory write policy and turn-routing heuristics.

This package houses the decision layer that sits between extracted memory
candidates and persisted writes:

- ``constants`` — small lookup tables and the procedural-request classifier.
- ``semantic`` — category-based heuristics for semantic facts.
- ``small_talk`` — lightweight discourse filter that suppresses extraction.
- ``candidates`` — promote extractor outputs into ``MemoryCandidate`` instances.
- ``write`` — the LLM-primary write-policy gate.
- ``turn_routing`` — turn-level skip/index helpers for memory side-effect nodes.
"""

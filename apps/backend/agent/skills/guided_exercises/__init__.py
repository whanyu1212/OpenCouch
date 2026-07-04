"""Guided exercise internals split by responsibility.

The Python registry and lifecycle service remain the runtime source of truth.
Skill docs provide standards-aligned packaging without changing exercise
selection or step progression semantics.

Key modules:
- ``registry`` and ``definitions`` hold the app-owned exercise catalog.
- ``rendering`` contains prompt-local skill context and ``SKILL.md`` helpers.
- ``engine.lifecycle`` runs the app-owned exercise state machine.
"""

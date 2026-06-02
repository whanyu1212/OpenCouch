"""Guided exercise internals split by responsibility.

The Python registry and lifecycle service remain the runtime source of truth.
Skill docs and loadouts provide standards-aligned packaging and observability
without changing exercise selection or step progression semantics.

Key modules:
- ``registry`` and ``definitions`` hold the app-owned exercise catalog.
- ``loadout`` projects available exercises from runtime state.
- ``rendering`` contains prompt-local skill context and ``SKILL.md`` helpers.
- ``lifecycle`` runs the app-owned exercise state machine.
"""

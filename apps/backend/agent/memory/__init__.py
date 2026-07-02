"""Prompt-memory layer for the OpenCouch agent.

Owns the runtime models, storage backends, retrieval helpers, write-policy
utilities, per-turn write orchestration, session-end commit, and user-facing
memory controls used by the agent's long-term memory features.

The default durable backend is Postgres; SQLite remains only as an explicit
legacy fallback via :mod:`agent.memory.store.sqlite`. Always-on audit
persistence lives under :mod:`agent.audit` instead of here.

See ``README.md`` in this directory for the file map and entry points.
"""

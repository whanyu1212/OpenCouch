"""SQLite audit-backend fallbacks for crisis log and session feedback.

Postgres is the default persistent backend; these SQLite implementations
remain as the supported fallback selectable via
``OPENCOUCH_PERSISTENCE_BACKEND=sqlite``.
"""

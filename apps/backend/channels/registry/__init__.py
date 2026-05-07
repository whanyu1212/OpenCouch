"""Telegram session registry — maps Telegram users to OpenCouch user IDs.

Primary: PostgresTelegramSessionRegistry
Fallback: SqliteTelegramSessionRegistry (legacy, used when Postgres is unavailable)
"""

from channels.registry.postgres import PostgresTelegramSessionRegistry
from channels.registry.sqlite_fallback import SqliteTelegramSessionRegistry

__all__ = [
    "PostgresTelegramSessionRegistry",
    "SqliteTelegramSessionRegistry",
]

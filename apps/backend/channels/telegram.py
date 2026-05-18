"""Telegram channel adapter for OpenCouch text runtime."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from secrets import token_bytes
from typing import Protocol
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.models import Channel, DoneEvent, ResponseReadyEvent
from agent.runtime import (
    ActiveSessionExists,
    ExpectedSessionLiveness,
    PersistentAgentRuntime,
    SessionInterrupted,
    SessionLeaseExpired,
    SessionStatus,
)
from config import PersistenceBackend, ResponseModelTier
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TELEGRAM_SESSION_DB_PATH = BACKEND_ROOT / ".store" / "telegram_sessions.sqlite3"
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_REPLY_CHUNK_SIZE = 4000
TELEGRAM_DEFAULT_SESSION_TRANSCRIPT_SOFT_LIMIT = 240
TELEGRAM_DEFAULT_ROTATION_SWEEP_INTERVAL_SECONDS = 30.0
TELEGRAM_DEFAULT_RECLAIM_INTERVAL_SECONDS = 300.0
TELEGRAM_DEFAULT_RECLAIM_GRACE_SECONDS = 3600.0
TELEGRAM_RECLAIM_STUCK_ATTEMPTS = 24
TELEGRAM_RECLAIM_STUCK_AGE = timedelta(days=14)
TELEGRAM_START_MESSAGE = (
    "OpenCouch is connected. Send a message when you want to talk.\n"
    "/end closes the current session. You can also just stop replying;\n"
    "OpenCouch will close the session after inactivity."
)
TELEGRAM_SESSION_CLOSED_MESSAGE = "Session closed."
TELEGRAM_UNAUTHORIZED_MESSAGE = (
    "This Telegram account is not allowed to use this OpenCouch bot."
)
TELEGRAM_DM_ONLY_MESSAGE = "OpenCouch Telegram MVP only supports direct messages."
TELEGRAM_TEXT_ONLY_MESSAGE = "OpenCouch Telegram MVP supports text messages only."
TELEGRAM_EMPTY_RESPONSE_MESSAGE = "I could not produce a reply. Please try again."
TELEGRAM_ERROR_MESSAGE = "I hit an error while processing that message."
TELEGRAM_MAINTENANCE_MESSAGE = (
    "OpenCouch needs to finish closing the previous session before I can continue. "
    "Please try again in a moment."
)
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_LEGACY_MIGRATION_STATES = {"pending", "finalizing", "finalized", "failed"}
_HTML_TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9]+)(?:\s[^>]*)?>")
_HTML_TOKEN_RE = re.compile(r"<[^>]*>|&[A-Za-z0-9#]+;|[\s\S]")


class TelegramConfigurationError(RuntimeError):
    """Raised when Telegram gateway configuration is invalid."""


@dataclass(frozen=True, slots=True)
class TelegramGatewayConfig:
    """Validated Telegram gateway settings."""

    bot_token: str
    allowed_user_ids: frozenset[int]
    owner_id: str
    proxy_url: str | None = None
    drop_pending_updates: bool = True
    response_model_tier: ResponseModelTier = "fast"
    thread_rotation_enabled: bool = False
    session_registry_sqlite_path: Path = DEFAULT_TELEGRAM_SESSION_DB_PATH
    session_transcript_soft_limit: int = TELEGRAM_DEFAULT_SESSION_TRANSCRIPT_SOFT_LIMIT
    rotation_sweep_interval_seconds: float = (
        TELEGRAM_DEFAULT_ROTATION_SWEEP_INTERVAL_SECONDS
    )
    reclaim_interval_seconds: float = TELEGRAM_DEFAULT_RECLAIM_INTERVAL_SECONDS
    reclaim_grace_seconds: float = TELEGRAM_DEFAULT_RECLAIM_GRACE_SECONDS

    @classmethod
    def from_env(cls) -> TelegramGatewayConfig:
        """Build Telegram settings from environment variables.

        Returns:
            Validated Telegram gateway configuration.

        Raises:
            TelegramConfigurationError: If required settings are missing or
                malformed.
        """

        token = _required_env("OPENCOUCH_TELEGRAM_BOT_TOKEN")
        owner_id = _required_env("OPENCOUCH_TELEGRAM_OWNER_ID")
        allowed_raw = os.getenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "")
        allowed_user_ids = _parse_allowed_user_ids(allowed_raw)
        if not allowed_user_ids:
            raise TelegramConfigurationError(
                "OPENCOUCH_TELEGRAM_ALLOW_FROM must contain at least one numeric "
                "Telegram user ID. Use getUpdates to discover your ID before "
                "starting the gateway."
            )

        response_tier = _parse_response_model_tier(
            os.getenv("OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER", "fast")
        )

        return cls(
            bot_token=token,
            allowed_user_ids=frozenset(allowed_user_ids),
            owner_id=owner_id,
            proxy_url=_optional_env("OPENCOUCH_TELEGRAM_PROXY"),
            drop_pending_updates=_parse_bool_env(
                "OPENCOUCH_TELEGRAM_DROP_PENDING_UPDATES",
                default=True,
            ),
            response_model_tier=response_tier,
            thread_rotation_enabled=_parse_bool_env(
                "OPENCOUCH_TELEGRAM_THREAD_ROTATION_ENABLED",
                default=False,
            ),
            session_registry_sqlite_path=_parse_path_env(
                "OPENCOUCH_TELEGRAM_SESSION_DB_PATH",
                default=DEFAULT_TELEGRAM_SESSION_DB_PATH,
            ),
            session_transcript_soft_limit=_parse_positive_int_env(
                "OPENCOUCH_TELEGRAM_SESSION_TRANSCRIPT_SOFT_LIMIT",
                default=TELEGRAM_DEFAULT_SESSION_TRANSCRIPT_SOFT_LIMIT,
            ),
            rotation_sweep_interval_seconds=_parse_positive_float_env(
                "OPENCOUCH_TELEGRAM_ROTATION_SWEEP_INTERVAL_SECONDS",
                default=TELEGRAM_DEFAULT_ROTATION_SWEEP_INTERVAL_SECONDS,
            ),
            reclaim_interval_seconds=_parse_positive_float_env(
                "OPENCOUCH_TELEGRAM_RECLAIM_INTERVAL_SECONDS",
                default=TELEGRAM_DEFAULT_RECLAIM_INTERVAL_SECONDS,
            ),
            reclaim_grace_seconds=_parse_positive_float_env(
                "OPENCOUCH_TELEGRAM_RECLAIM_GRACE_SECONDS",
                default=TELEGRAM_DEFAULT_RECLAIM_GRACE_SECONDS,
            ),
        )


def _required_env(name: str) -> str:
    """Return a required non-empty environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Trimmed environment value.

    Raises:
        TelegramConfigurationError: If the value is missing.
    """

    value = os.getenv(name, "").strip()
    if not value:
        raise TelegramConfigurationError(f"{name} is required.")
    return value


def _optional_env(name: str) -> str | None:
    """Return a trimmed optional environment variable.

    Args:
        name: Environment variable name.

    Returns:
        Trimmed value, or ``None`` when unset.
    """

    value = os.getenv(name, "").strip()
    return value or None


def _parse_path_env(name: str, *, default: Path) -> Path:
    """Parse a filesystem path environment variable.

    Args:
        name: Environment variable name.
        default: Path used when the variable is unset.

    Returns:
        Parsed path.
    """

    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser() if raw else default


def _parse_positive_int_env(name: str, *, default: int) -> int:
    """Parse a positive integer environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        Parsed positive integer.

    Raises:
        TelegramConfigurationError: If the value is not positive.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise TelegramConfigurationError(
            f"{name} must be a positive integer; got {raw!r}."
        ) from exc
    if value <= 0:
        raise TelegramConfigurationError(
            f"{name} must be a positive integer; got {raw!r}."
        )
    return value


def _parse_positive_float_env(name: str, *, default: float) -> float:
    """Parse a positive float environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        Parsed positive float.

    Raises:
        TelegramConfigurationError: If the value is not positive.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise TelegramConfigurationError(
            f"{name} must be a positive number; got {raw!r}."
        ) from exc
    if value <= 0:
        raise TelegramConfigurationError(
            f"{name} must be a positive number; got {raw!r}."
        )
    return value


def _parse_allowed_user_ids(raw: str) -> set[int]:
    """Parse a comma-separated Telegram user ID allowlist.

    Args:
        raw: Comma-separated numeric Telegram IDs.

    Returns:
        Parsed numeric IDs.

    Raises:
        TelegramConfigurationError: If any entry is not numeric.
    """

    ids: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if not item.isdecimal():
            raise TelegramConfigurationError(
                "OPENCOUCH_TELEGRAM_ALLOW_FROM must be a comma-separated list "
                f"of numeric Telegram user IDs; got {item!r}."
            )
        ids.add(int(item))
    return ids


def _parse_bool_env(name: str, *, default: bool) -> bool:
    """Parse a boolean environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        Parsed boolean value.

    Raises:
        TelegramConfigurationError: If the value is not boolean-like.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TelegramConfigurationError(
        f"{name} must be a boolean value such as true or false; got {raw!r}."
    )


def _parse_response_model_tier(raw: str) -> ResponseModelTier:
    """Parse the Telegram response model tier.

    Args:
        raw: Raw tier string.

    Returns:
        Valid response model tier.

    Raises:
        TelegramConfigurationError: If the tier is unsupported.
    """

    normalized = raw.strip().lower()
    if normalized == "fast":
        return "fast"
    if normalized == "quality":
        return "quality"
    raise TelegramConfigurationError(
        "OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER must be 'fast' or 'quality'."
    )


def telegram_thread_id(chat_id: int | str) -> str:
    """Return the OpenCouch thread ID for a Telegram DM chat.

    Args:
        chat_id: Telegram chat identifier.

    Returns:
        Stable OpenCouch thread ID.
    """

    return f"telegram:dm:{chat_id}"


def telegram_session_thread_id(chat_id: int | str, session_id: str) -> str:
    """Return a rotated OpenCouch thread ID for one Telegram session.

    Args:
        chat_id: Telegram chat identifier.
        session_id: Durable session identifier.

    Returns:
        Rotated OpenCouch thread ID.
    """

    return f"telegram:dm:{chat_id}:session:{session_id}"


def is_rotated_telegram_thread_id(thread_id: str) -> bool:
    """Return whether a thread id belongs to Telegram rotation.

    Args:
        thread_id: OpenCouch thread identifier.

    Returns:
        True for rotated Telegram session thread ids.
    """

    return thread_id.startswith("telegram:dm:") and ":session:" in thread_id


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for registry rows.

    Returns:
        ISO-8601 UTC timestamp.
    """

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_registry_timestamp(value: str | None) -> datetime | None:
    """Parse a registry timestamp.

    Args:
        value: Stored ISO timestamp.

    Returns:
        Parsed datetime, or ``None`` if unavailable.
    """

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _legacy_migration_state(value: object) -> str:
    """Normalize stored legacy migration state values.

    Args:
        value: Stored SQLite value.

    Returns:
        One of ``pending``, ``finalizing``, ``finalized``, or ``failed``.
    """

    if value is None:
        return "pending"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return "finalized"
    if normalized in {"0", "false", "no", ""}:
        return "pending"
    if normalized in _LEGACY_MIGRATION_STATES:
        return normalized
    return "failed"


def _generate_ulid() -> str:
    """Generate a sortable ULID string without an extra dependency.

    Returns:
        A 26-character Crockford base32 ULID.
    """

    value = (int(datetime.now(UTC).timestamp() * 1000) << 80) | int.from_bytes(
        token_bytes(10),
        "big",
    )
    chars: list[str] = []
    for _ in range(26):
        chars.append(_ULID_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


@dataclass(slots=True)
class TelegramActiveSession:
    """Registry view of one Telegram chat's active pointer."""

    chat_id: str
    active_thread_id: str | None
    active_started_at: str | None
    legacy_migration_state: str
    migration_last_error: str | None
    close_requested_reason: str | None
    close_requested_at: str | None


@dataclass(slots=True)
class TelegramClosedSession:
    """Registry view of one closed Telegram session."""

    chat_id: str
    thread_id: str
    closed_at: str


class TelegramSessionRegistry(Protocol):
    """Storage boundary for Telegram chat-to-session thread pointers."""

    async def ensure_started(self) -> None:
        """Open the registry connection and create tables.

        Returns:
            None.
        """

    async def aclose(self) -> None:
        """Close the registry connection.

        Returns:
            None.
        """

    async def ensure_chat(self, chat_id: int | str) -> TelegramActiveSession:
        """Ensure and return the active pointer row for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Active pointer row.
        """

    async def get_active(self, chat_id: int | str) -> TelegramActiveSession | None:
        """Return the active pointer row for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Active pointer row, or ``None``.
        """

    async def create_session(self, chat_id: int | str) -> str:
        """Create and activate a new rotated session thread for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Newly active OpenCouch thread id.
        """

    async def set_legacy_migration_state(
        self,
        chat_id: int | str,
        *,
        state: str,
        error: str | None = None,
    ) -> None:
        """Record legacy thread migration state.

        Args:
            chat_id: Telegram chat identifier.
            state: Legacy migration state.
            error: Optional migration error.

        Returns:
            None.
        """

    async def reset_finalizing_legacy_migrations(self) -> None:
        """Mark interrupted legacy migrations as failed on startup.

        Returns:
            None.
        """

    async def set_pending_close(self, chat_id: int | str, reason: str) -> None:
        """Persist a pending close request for the chat's active pointer.

        Args:
            chat_id: Telegram chat identifier.
            reason: Close reason.

        Returns:
            None.
        """

    async def clear_pending_close(self, chat_id: int | str) -> None:
        """Clear a pending close request for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            None.
        """

    async def close_thread(
        self,
        chat_id: int | str,
        thread_id: str,
        reason: str,
    ) -> None:
        """Mark a session closed and clear the active pointer if it matches.

        Args:
            chat_id: Telegram chat identifier.
            thread_id: Session thread id.
            reason: Close reason.

        Returns:
            None.
        """

    async def list_active(self) -> list[TelegramActiveSession]:
        """List all active pointer rows.

        Returns:
            Active pointer rows.
        """

    async def list_unclosed_sessions(self) -> list[TelegramClosedSession]:
        """List session rows that have not been closed in the registry.

        Returns:
            Open session rows.
        """

    async def list_reclaimable(self, grace: timedelta) -> list[TelegramClosedSession]:
        """List closed sessions old enough for checkpoint reclaim.

        Args:
            grace: Minimum age since close before reclaim.

        Returns:
            Closed session rows.
        """

    async def mark_reclaim_result(
        self,
        thread_id: str,
        *,
        error: str | None = None,
    ) -> None:
        """Record checkpoint reclaim outcome.

        Args:
            thread_id: Session thread id.
            error: Optional reclaim error.

        Returns:
            None.
        """


def build_telegram_session_registry(
    *,
    backend: PersistenceBackend,
    sqlite_path: str | Path,
    database_url: str | None,
) -> TelegramSessionRegistry:
    """Create the Telegram session registry for the available persistence backend.

    Args:
        backend (PersistenceBackend): Selected persistence backend.
        sqlite_path (str | Path): SQLite registry path for legacy fallback mode.
        database_url (str | None): PostgreSQL database URL for Postgres mode.

    Returns:
        TelegramSessionRegistry: Backend-specific registry implementation.
    """

    if database_url:
        from channels.registry.postgres import PostgresTelegramSessionRegistry

        return PostgresTelegramSessionRegistry(database_url)

    logger.warning(
        "Telegram registry falling back to legacy SQLite because no "
        "Postgres database URL is configured."
    )

    from channels.registry.sqlite_fallback import SqliteTelegramSessionRegistry

    return SqliteTelegramSessionRegistry(sqlite_path)


def split_telegram_text(
    text: str,
    *,
    limit: int = TELEGRAM_REPLY_CHUNK_SIZE,
) -> list[str]:
    """Split a reply into Telegram-safe plain-text chunks.

    Args:
        text: Reply body to send.
        limit: Maximum chunk length.

    Returns:
        Non-empty chunks that fit within ``limit``.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    remaining = text.strip() or TELEGRAM_EMPTY_RESPONSE_MESSAGE
    chunks: list[str] = []
    while len(remaining) > limit:
        boundary = remaining.rfind("\n", 0, limit + 1)
        if boundary <= 0:
            boundary = remaining.rfind(" ", 0, limit + 1)
        if boundary <= 0:
            boundary = limit

        chunk = remaining[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[boundary:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def render_telegram_markdown(text: str) -> str:
    """Render a safe subset of markdown to Telegram HTML.

    Args:
        text: Markdown-ish reply text.

    Returns:
        Telegram HTML parse-mode text.
    """

    try:
        parts = re.split(r"(```.*?```)", text.strip(), flags=re.DOTALL)
        rendered: list[str] = []
        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                code = part[3:-3]
                if code.startswith("\n"):
                    code = code[1:]
                first_line, _, rest = code.partition("\n")
                if rest and first_line.strip().replace("-", "").isalnum():
                    code = rest
                rendered.append(f"<pre><code>{html.escape(code)}</code></pre>")
                continue

            rendered_lines: list[str] = []
            for line in part.splitlines():
                heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
                if heading:
                    rendered_lines.append(
                        f"<b>{_render_inline_markdown(heading.group(1))}</b>"
                    )
                else:
                    rendered_lines.append(_render_inline_markdown(line))
            rendered.append("\n".join(rendered_lines))
        return "".join(rendered).strip() or html.escape(TELEGRAM_EMPTY_RESPONSE_MESSAGE)
    except Exception:
        logger.warning("telegram markdown rendering failed", exc_info=True)
        return html.escape(text.strip() or TELEGRAM_EMPTY_RESPONSE_MESSAGE)


def split_telegram_html(
    html_text: str,
    *,
    limit: int = TELEGRAM_REPLY_CHUNK_SIZE,
) -> list[str]:
    """Split Telegram HTML into chunks while preserving open tags.

    Args:
        html_text: Rendered Telegram HTML.
        limit: Maximum chunk length.

    Returns:
        Valid HTML chunks under the limit where possible.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")

    open_tags: list[tuple[str, str, str]] = []
    current: list[str] = []
    chunks: list[str] = []

    def current_text() -> str:
        return "".join(current)

    def closing_tags() -> str:
        return "".join(close for _, _, close in reversed(open_tags))

    def opening_tags() -> list[str]:
        return [open_tag for _, open_tag, _ in open_tags]

    for match in _HTML_TOKEN_RE.finditer(html_text):
        token = match.group(0)
        projected_len = len(current_text()) + len(token) + len(closing_tags())
        if current and projected_len > limit:
            chunk = f"{current_text()}{closing_tags()}".strip()
            if chunk:
                chunks.append(chunk)
            current = opening_tags()

        current.append(token)
        _update_html_tag_stack(token, open_tags)

    final = f"{current_text()}{closing_tags()}".strip()
    if final:
        chunks.append(final)
    return chunks or [html.escape(TELEGRAM_EMPTY_RESPONSE_MESSAGE)]


def split_telegram_markdown_html(
    text: str,
    *,
    limit: int = TELEGRAM_REPLY_CHUNK_SIZE,
) -> list[str]:
    """Render markdown to Telegram HTML and split it into safe chunks.

    Args:
        text: Markdown-ish reply text.
        limit: Maximum chunk length.

    Returns:
        Telegram HTML chunks.
    """

    rendered = render_telegram_markdown(text)
    chunks = split_telegram_html(rendered, limit=limit)
    safe_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= limit:
            safe_chunks.append(chunk)
            continue
        for plain_chunk in split_telegram_text(_strip_html_tags(chunk), limit=limit):
            safe_chunks.append(html.escape(plain_chunk))
    return safe_chunks


def _render_inline_markdown(text: str, *, depth: int = 0) -> str:
    """Render inline markdown to escaped Telegram HTML.

    Args:
        text: Inline markdown text.
        depth: Recursion depth for nested spans.

    Returns:
        Escaped Telegram HTML.
    """

    if depth > 4:
        return html.escape(text)

    output: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "`":
            end = text.find("`", index + 1)
            if end != -1:
                output.append(f"<code>{html.escape(text[index + 1 : end])}</code>")
                index = end + 1
                continue

        if text.startswith("**", index):
            end = text.find("**", index + 2)
            if end != -1:
                output.append(
                    f"<b>{_render_inline_markdown(text[index + 2 : end], depth=depth + 1)}</b>"
                )
                index = end + 2
                continue

        if text[index] == "*":
            end = text.find("*", index + 1)
            if end != -1 and text[index + 1 : end].strip():
                output.append(
                    f"<i>{_render_inline_markdown(text[index + 1 : end], depth=depth + 1)}</i>"
                )
                index = end + 1
                continue

        if text[index] == "[":
            rendered_link = _try_render_markdown_link(text, index)
            if rendered_link is not None:
                link_html, next_index = rendered_link
                output.append(link_html)
                index = next_index
                continue

        output.append(html.escape(text[index]))
        index += 1
    return "".join(output)


def _try_render_markdown_link(text: str, start: int) -> tuple[str, int] | None:
    """Render one markdown link if it starts at ``start``.

    Args:
        text: Inline markdown text.
        start: Candidate link start offset.

    Returns:
        Rendered link HTML and next offset, or ``None``.
    """

    label_end = text.find("]", start + 1)
    if label_end == -1 or not text.startswith("(", label_end + 1):
        return None
    url_end = text.find(")", label_end + 2)
    if url_end == -1:
        return None

    label = text[start + 1 : label_end]
    url = text[label_end + 2 : url_end].strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    link_html = (
        f'<a href="{html.escape(url, quote=True)}">'
        f"{html.escape(label.strip() or url)}</a>"
    )
    return link_html, url_end + 1


def _update_html_tag_stack(
    token: str,
    open_tags: list[tuple[str, str, str]],
) -> None:
    """Update an HTML tag stack for a rendered token.

    Args:
        token: Rendered HTML token.
        open_tags: Mutable open-tag stack.

    Returns:
        None.
    """

    match = _HTML_TAG_RE.fullmatch(token)
    if match is None:
        return
    is_close = bool(match.group(1))
    tag_name = match.group(2).lower()
    if tag_name not in {"a", "b", "i", "code", "pre"}:
        return
    if is_close:
        for index in range(len(open_tags) - 1, -1, -1):
            if open_tags[index][0] == tag_name:
                del open_tags[index:]
                return
    else:
        open_tags.append((tag_name, token, f"</{tag_name}>"))


def _strip_html_tags(text: str) -> str:
    """Remove rendered HTML tags for plain fallback splitting.

    Args:
        text: HTML text.

    Returns:
        Plain text.
    """

    return html.unescape(re.sub(r"<[^>]+>", "", text))


class TelegramChannel:
    """Translate Telegram message updates into OpenCouch runtime turns."""

    def __init__(
        self,
        *,
        config: TelegramGatewayConfig,
        runtime: PersistentAgentRuntime,
        llm_client: BaseLLMClient,
        response_llm_client: BaseLLMClient,
        session_registry: TelegramSessionRegistry | None = None,
    ) -> None:
        """Initialize the Telegram adapter.

        Args:
            config: Validated Telegram gateway settings.
            runtime: OpenCouch persistent runtime.
            llm_client: Control-plane LLM client.
            response_llm_client: Response-writing LLM client.
            session_registry: Optional storage backend for rotated Telegram sessions.
        """

        self._config = config
        self._runtime = runtime
        self._llm_client = llm_client
        self._response_llm_client = response_llm_client
        if not config.thread_rotation_enabled:
            self._registry = None
        elif session_registry is not None:
            self._registry = session_registry
        else:
            from channels.registry.sqlite_fallback import (
                SqliteTelegramSessionRegistry,
            )

            self._registry = SqliteTelegramSessionRegistry(
                config.session_registry_sqlite_path
            )
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._background_tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start registry-backed maintenance tasks when rotation is enabled.

        Returns:
            None.
        """

        if self._registry is None:
            return
        await self._registry.ensure_started()
        await self._startup_recovery_once()
        await self._maintenance_sweep_once()
        self._background_tasks = [
            asyncio.create_task(self._maintenance_sweep_loop()),
            asyncio.create_task(self._reclaim_sweep_loop()),
        ]

    async def stop(self) -> None:
        """Stop background tasks and close the session registry.

        Returns:
            None.
        """

        for task in self._background_tasks:
            task.cancel()
        for task in self._background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks = []
        if self._registry is not None:
            await self._registry.aclose()

    def _chat_lock(self, chat_id: int | str) -> asyncio.Lock:
        """Return the per-chat in-process lock.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Per-chat lock.
        """

        chat_key = str(chat_id)
        lock = self._chat_locks.get(chat_key)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[chat_key] = lock
        return lock

    async def _try_acquire_chat_lock(self, chat_id: int | str) -> asyncio.Lock | None:
        """Acquire a per-chat lock only if it is immediately available.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            The acquired lock, or ``None`` when foreground work owns it.
        """

        lock = self._chat_lock(chat_id)
        if lock.locked():
            return None
        await lock.acquire()
        return lock

    def _require_registry(self) -> TelegramSessionRegistry:
        """Return the active session registry.

        Returns:
            Telegram session registry.

        Raises:
            RuntimeError: If rotation is disabled.
        """

        if self._registry is None:
            raise RuntimeError("Telegram thread rotation is disabled.")
        return self._registry

    async def handle_start(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,  # noqa: ARG002
    ) -> None:
        """Handle Telegram `/start` without touching runtime memory.

        Args:
            update: Telegram update containing the command.
            context: Telegram callback context.

        Returns:
            None.
        """

        if not await self._ensure_allowed_private(update):
            return
        await self._reply(update, TELEGRAM_START_MESSAGE)

    async def handle_help(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,  # noqa: ARG002
    ) -> None:
        """Handle Telegram `/help` without touching runtime memory.

        Args:
            update: Telegram update containing the command.
            context: Telegram callback context.

        Returns:
            None.
        """

        if not await self._ensure_allowed_private(update):
            return
        await self._reply(update, TELEGRAM_START_MESSAGE)

    async def handle_end(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,  # noqa: ARG002
    ) -> None:
        """Handle Telegram `/end` by closing the active OpenCouch session.

        Args:
            update: Telegram update containing the command.
            context: Telegram callback context.

        Returns:
            None.
        """

        if not await self._ensure_allowed_private(update):
            return
        chat = update.effective_chat
        if chat is None:
            return

        if self._config.thread_rotation_enabled:
            await self._handle_end_rotated(update, chat.id)
            return

        await self._runtime.end_session(
            telegram_thread_id(chat.id),
            llm_client=self._llm_client,
        )
        await self._reply(update, TELEGRAM_SESSION_CLOSED_MESSAGE)

    async def handle_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,  # noqa: ARG002
    ) -> None:
        """Handle one allowed Telegram text message as an OpenCouch turn.

        Args:
            update: Telegram update containing a text message.
            context: Telegram callback context.

        Returns:
            None.
        """

        if not await self._ensure_allowed_private(update):
            return
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return

        text = (message.text or "").strip()
        if not text:
            return

        if self._config.thread_rotation_enabled:
            await self._handle_text_rotated(update, chat.id, text)
            return

        sent = False
        done_text: str | None = None
        try:
            stream = self._runtime.run_turn_stream(
                thread_id=telegram_thread_id(chat.id),
                message=text,
                channel=Channel.TELEGRAM,
                user_id=self._config.owner_id,
                installed_skills=[],
                llm_client=self._llm_client,
                response_llm_client=self._response_llm_client,
            )
            async for event in stream:
                if isinstance(event, ResponseReadyEvent) and not sent:
                    await self._reply(update, event.output.response_text)
                    sent = True
                elif isinstance(event, DoneEvent):
                    done_text = event.output.response_text
        except Exception:
            logger.exception("telegram runtime turn failed")
            await self._reply(update, TELEGRAM_ERROR_MESSAGE)
            return

        if not sent:
            await self._reply(update, done_text or TELEGRAM_EMPTY_RESPONSE_MESSAGE)

    async def handle_unsupported(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,  # noqa: ARG002
    ) -> None:
        """Reject non-text Telegram messages for MVP.

        Args:
            update: Telegram update containing unsupported content.
            context: Telegram callback context.

        Returns:
            None.
        """

        if not await self._ensure_allowed_private(update):
            return
        await self._reply(update, TELEGRAM_TEXT_ONLY_MESSAGE)

    async def _handle_end_rotated(self, update: Update, chat_id: int | str) -> None:
        """Handle `/end` for registry-backed rotated sessions.

        Args:
            update: Telegram update containing the command.
            chat_id: Telegram chat identifier.

        Returns:
            None.
        """

        registry = self._require_registry()
        async with self._chat_lock(chat_id):
            if not await self._migrate_legacy_thread_if_needed(chat_id):
                await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                return

            active = await registry.get_active(chat_id)
            if active is None or active.active_thread_id is None:
                await registry.clear_pending_close(chat_id)
                await self._reply(update, TELEGRAM_SESSION_CLOSED_MESSAGE)
                return

            thread_id = active.active_thread_id
            await registry.set_pending_close(chat_id, "end_command")
            if not await self._finalize_and_close_thread(
                chat_id,
                thread_id,
                "end_command",
            ):
                await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                return

            await self._reply(update, TELEGRAM_SESSION_CLOSED_MESSAGE)

    async def _handle_text_rotated(
        self,
        update: Update,
        chat_id: int | str,
        text: str,
    ) -> None:
        """Handle one Telegram text message with session rotation enabled.

        Args:
            update: Telegram update.
            chat_id: Telegram chat identifier.
            text: Message text.

        Returns:
            None.
        """

        async with self._chat_lock(chat_id):
            if not await self._migrate_legacy_thread_if_needed(chat_id):
                await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                return

            if not await self._retry_pending_close(chat_id):
                await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                return

            thread_id, expected_liveness, ready = await self._thread_for_next_message(
                chat_id
            )
            if not ready:
                await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                return

            try:
                completed = await self._stream_thread_reply(
                    update,
                    thread_id=thread_id,
                    text=text,
                    expected_liveness=expected_liveness,
                )
            except (ActiveSessionExists, SessionInterrupted, SessionLeaseExpired):
                (
                    thread_id,
                    expected_liveness,
                    ready,
                ) = await self._thread_for_next_message(chat_id)
                if not ready:
                    await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                    return
                try:
                    completed = await self._stream_thread_reply(
                        update,
                        thread_id=thread_id,
                        text=text,
                        expected_liveness=expected_liveness,
                    )
                except (ActiveSessionExists, SessionInterrupted, SessionLeaseExpired):
                    await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)
                    return

            if not completed:
                return

            status = await self._runtime.session_status(thread_id)
            if status == SessionStatus.ROTATION_REQUIRED:
                if not await self._finalize_and_close_thread(
                    chat_id,
                    thread_id,
                    "rotation_required",
                ):
                    await self._reply(update, TELEGRAM_MAINTENANCE_MESSAGE)

    async def _thread_for_next_message(
        self,
        chat_id: int | str,
    ) -> tuple[str, ExpectedSessionLiveness, bool]:
        """Resolve or create the thread for the next rotated Telegram message.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Thread id, expected liveness, and whether processing may continue.
        """

        registry = self._require_registry()
        active = await registry.get_active(chat_id)
        thread_id = active.active_thread_id if active else None
        if thread_id is not None:
            thread_id, ready = await self._resolve_active_thread(
                chat_id,
                thread_id,
            )
            if not ready:
                return "", "active", False

        expected_liveness: ExpectedSessionLiveness = "active"
        if thread_id is None:
            thread_id = await registry.create_session(chat_id)
            expected_liveness = "absent"
        return thread_id, expected_liveness, True

    async def _migrate_legacy_thread_if_needed(self, chat_id: int | str) -> bool:
        """Finalize the pre-rotation Telegram thread once per chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            True when migration is complete.
        """

        registry = self._require_registry()
        active = await registry.ensure_chat(chat_id)
        if active.legacy_migration_state == "finalized":
            return True

        legacy_thread_id = telegram_thread_id(chat_id)
        try:
            await registry.set_legacy_migration_state(
                chat_id,
                state="finalizing",
            )
            status = await self._runtime.session_status(legacy_thread_id)
            if status != SessionStatus.ABSENT:
                await self._runtime.end_session(
                    legacy_thread_id,
                    llm_client=self._llm_client,
                )
            await registry.set_legacy_migration_state(chat_id, state="finalized")
            return True
        except Exception as exc:
            logger.exception(
                "telegram legacy thread migration failed for chat_id=%s",
                chat_id,
            )
            await registry.set_legacy_migration_state(
                chat_id,
                state="failed",
                error=str(exc),
            )
            return False

    async def _retry_pending_close(self, chat_id: int | str) -> bool:
        """Retry a pending close before accepting another user message.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            True when no pending close remains.
        """

        registry = self._require_registry()
        active = await registry.get_active(chat_id)
        if active is None or active.close_requested_reason is None:
            return True
        if active.active_thread_id is None:
            await registry.clear_pending_close(chat_id)
            return True
        return await self._finalize_and_close_thread(
            chat_id,
            active.active_thread_id,
            active.close_requested_reason,
        )

    async def _resolve_active_thread(
        self,
        chat_id: int | str,
        thread_id: str,
    ) -> tuple[str | None, bool]:
        """Resolve a registry active pointer before a new message.

        Args:
            chat_id: Telegram chat identifier.
            thread_id: Current active thread id.

        Returns:
            ``(thread_id, True)`` to reuse the active thread,
            ``(None, True)`` to create a new thread, or ``(None, False)`` when
            maintenance is required.
        """

        registry = self._require_registry()
        status = await self._runtime.session_status(thread_id)
        if status == SessionStatus.ACTIVE:
            return thread_id, True
        if status == SessionStatus.ABSENT:
            await registry.close_thread(chat_id, thread_id, "runtime_absent")
            return None, True

        reason_by_status = {
            SessionStatus.EXPIRED_UNFINALIZED: "timeout",
            SessionStatus.INTERRUPTED: "interrupted",
            SessionStatus.ROTATION_REQUIRED: "rotation_required",
        }
        reason = reason_by_status.get(status)
        if reason is None:
            return None, False
        if await self._finalize_and_close_thread(chat_id, thread_id, reason):
            return None, True
        return None, False

    async def _finalize_and_close_thread(
        self,
        chat_id: int | str,
        thread_id: str,
        reason: str,
    ) -> bool:
        """Finalize runtime state and close a registry session.

        Args:
            chat_id: Telegram chat identifier.
            thread_id: Session thread id.
            reason: Registry close reason.

        Returns:
            True when the session is closed.
        """

        registry = self._require_registry()
        try:
            status = await self._runtime.session_status(thread_id)
            if status != SessionStatus.ABSENT:
                await self._runtime.end_session(thread_id, llm_client=self._llm_client)
            await registry.close_thread(chat_id, thread_id, reason)
            return True
        except Exception:
            logger.exception(
                "telegram failed to finalize thread_id=%s reason=%s",
                thread_id,
                reason,
            )
            return False

    async def _stream_thread_reply(
        self,
        update: Update,
        *,
        thread_id: str,
        text: str,
        expected_liveness: ExpectedSessionLiveness,
    ) -> bool:
        """Run one runtime turn and send the first ready reply.

        Args:
            update: Telegram update.
            thread_id: Session thread id.
            text: Message text.
            expected_liveness: Runtime liveness expectation.

        Returns:
            True when the turn completed.
        """

        sent = False
        done_text: str | None = None
        try:
            stream = self._runtime.run_turn_stream(
                thread_id=thread_id,
                message=text,
                channel=Channel.TELEGRAM,
                user_id=self._config.owner_id,
                installed_skills=[],
                llm_client=self._llm_client,
                response_llm_client=self._response_llm_client,
                expected_liveness=expected_liveness,
                session_transcript_soft_limit=(
                    self._config.session_transcript_soft_limit
                ),
            )
            async for event in stream:
                if isinstance(event, ResponseReadyEvent) and not sent:
                    await self._reply(update, event.output.response_text)
                    sent = True
                elif isinstance(event, DoneEvent):
                    done_text = event.output.response_text
        except (ActiveSessionExists, SessionInterrupted, SessionLeaseExpired):
            logger.warning(
                "telegram session lease check failed for thread_id=%s",
                thread_id,
                exc_info=True,
            )
            raise
        except Exception:
            logger.exception("telegram runtime turn failed")
            await self._reply(update, TELEGRAM_ERROR_MESSAGE)
            return False

        if not sent:
            await self._reply(update, done_text or TELEGRAM_EMPTY_RESPONSE_MESSAGE)
        return True

    async def _startup_recovery_once(self) -> None:
        """Recover registry rows left incomplete by a prior process stop.

        Returns:
            None.
        """

        if self._registry is None:
            return
        await self._registry.reset_finalizing_legacy_migrations()
        active_rows = await self._registry.list_active()
        active_thread_ids = {
            active.active_thread_id
            for active in active_rows
            if active.active_thread_id is not None
        }
        for session in await self._registry.list_unclosed_sessions():
            if session.thread_id in active_thread_ids:
                continue
            lock = await self._try_acquire_chat_lock(session.chat_id)
            if lock is None:
                continue
            try:
                status = await self._runtime.session_status(session.thread_id)
                if status != SessionStatus.ABSENT:
                    await self._finalize_and_close_thread(
                        session.chat_id,
                        session.thread_id,
                        "restart_stale",
                    )
                else:
                    await self._registry.close_thread(
                        session.chat_id,
                        session.thread_id,
                        "restart_stale",
                    )
            finally:
                lock.release()

    async def _maintenance_sweep_loop(self) -> None:
        """Run periodic active-pointer maintenance.

        Returns:
            None.
        """

        try:
            while True:
                await asyncio.sleep(self._config.rotation_sweep_interval_seconds)
                await self._maintenance_sweep_once()
        except asyncio.CancelledError:
            raise

    async def _maintenance_sweep_once(self) -> None:
        """Finalize stale active Telegram sessions without user traffic.

        Returns:
            None.
        """

        if self._registry is None:
            return
        for active in await self._registry.list_active():
            lock = await self._try_acquire_chat_lock(active.chat_id)
            if lock is None:
                continue
            try:
                if active.close_requested_reason and active.active_thread_id:
                    await self._finalize_and_close_thread(
                        active.chat_id,
                        active.active_thread_id,
                        active.close_requested_reason,
                    )
                    continue
                if not active.active_thread_id:
                    continue
                status = await self._runtime.session_status(active.active_thread_id)
                if status == SessionStatus.ABSENT:
                    await self._registry.close_thread(
                        active.chat_id,
                        active.active_thread_id,
                        "runtime_absent",
                    )
                elif status == SessionStatus.EXPIRED_UNFINALIZED:
                    await self._finalize_and_close_thread(
                        active.chat_id,
                        active.active_thread_id,
                        "timeout",
                    )
                elif status == SessionStatus.INTERRUPTED:
                    await self._finalize_and_close_thread(
                        active.chat_id,
                        active.active_thread_id,
                        "interrupted",
                    )
                elif status == SessionStatus.ROTATION_REQUIRED:
                    await self._finalize_and_close_thread(
                        active.chat_id,
                        active.active_thread_id,
                        "rotation_required",
                    )
            finally:
                lock.release()

    async def _reclaim_sweep_loop(self) -> None:
        """Run periodic checkpoint reclaim for closed rotated sessions.

        Returns:
            None.
        """

        try:
            while True:
                await asyncio.sleep(self._config.reclaim_interval_seconds)
                await self._reclaim_sweep_once()
        except asyncio.CancelledError:
            raise

    async def _reclaim_sweep_once(self) -> None:
        """Reset checkpoint rows for closed sessions after a grace period.

        Returns:
            None.
        """

        if self._registry is None:
            return
        sessions = await self._registry.list_reclaimable(
            timedelta(seconds=self._config.reclaim_grace_seconds)
        )
        for session in sessions:
            lock = await self._try_acquire_chat_lock(session.chat_id)
            if lock is None:
                continue
            try:
                active = await self._registry.get_active(session.chat_id)
                if active is not None and active.active_thread_id == session.thread_id:
                    logger.warning(
                        "telegram reclaim skipped active thread_id=%s",
                        session.thread_id,
                    )
                    continue
                status = await self._runtime.session_status(session.thread_id)
                if status != SessionStatus.ABSENT:
                    await self._runtime.end_session(
                        session.thread_id,
                        llm_client=self._llm_client,
                    )
                await self._runtime.reset_thread(session.thread_id)
                await self._registry.mark_reclaim_result(session.thread_id)
            except Exception as exc:
                logger.warning(
                    "telegram reclaim failed for thread_id=%s",
                    session.thread_id,
                    exc_info=True,
                )
                await self._registry.mark_reclaim_result(
                    session.thread_id,
                    error=str(exc),
                )
            finally:
                lock.release()

    async def _ensure_allowed_private(self, update: Update) -> bool:
        """Validate sender authorization and DM-only scope.

        Args:
            update: Telegram update to validate.

        Returns:
            True when the update may reach the runtime.
        """

        user = update.effective_user
        chat = update.effective_chat
        sender_id = getattr(user, "id", None)
        chat_id = getattr(chat, "id", None)

        if sender_id is None or chat_id is None:
            logger.warning(
                "telegram update missing sender or chat metadata: sender_id=%s "
                "chat_id=%s",
                sender_id,
                chat_id,
            )
            return False

        if int(sender_id) not in self._config.allowed_user_ids:
            logger.warning(
                "telegram denied sender_id=%s chat_id=%s",
                sender_id,
                chat_id,
            )
            await self._reply(update, TELEGRAM_UNAUTHORIZED_MESSAGE)
            return False

        if getattr(chat, "type", None) != "private":
            logger.warning("telegram rejected non-private chat_id=%s", chat_id)
            await self._reply(update, TELEGRAM_DM_ONLY_MESSAGE)
            return False

        return True

    async def _reply(self, update: Update, text: str) -> None:
        """Send a rendered reply, split for Telegram limits.

        Args:
            update: Telegram update whose message should receive the reply.
            text: Markdown-ish reply body.

        Returns:
            None.
        """

        message = update.effective_message
        if message is None:
            logger.warning("telegram reply skipped: update has no effective message")
            return

        for chunk in split_telegram_markdown_html(text):
            await message.reply_text(chunk, parse_mode="HTML")


def build_telegram_application(
    *,
    config: TelegramGatewayConfig,
    channel: TelegramChannel,
) -> Application:
    """Build the python-telegram-bot application for the adapter.

    Args:
        config: Validated Telegram gateway settings.
        channel: Telegram adapter instance with registered handlers.

    Returns:
        Configured Telegram application.
    """

    builder = ApplicationBuilder().token(config.bot_token)
    if config.proxy_url:
        builder = builder.proxy(config.proxy_url).get_updates_proxy(config.proxy_url)

    application = builder.build()
    application.add_handler(CommandHandler("start", channel.handle_start))
    application.add_handler(CommandHandler("help", channel.handle_help))
    application.add_handler(CommandHandler("end", channel.handle_end))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, channel.handle_text)
    )
    application.add_handler(MessageHandler(~filters.TEXT, channel.handle_unsupported))
    return application

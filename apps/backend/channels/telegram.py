"""Telegram channel adapter for OpenCouch text runtime."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

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
from agent.persistence import PersistentAgentRuntime
from core.config import ResponseModelTier
from services.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_REPLY_CHUNK_SIZE = 4000
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


class TelegramChannel:
    """Translate Telegram message updates into OpenCouch runtime turns."""

    def __init__(
        self,
        *,
        config: TelegramGatewayConfig,
        runtime: PersistentAgentRuntime,
        llm_client: BaseLLMClient,
        response_llm_client: BaseLLMClient,
    ) -> None:
        """Initialize the Telegram adapter.

        Args:
            config: Validated Telegram gateway settings.
            runtime: OpenCouch persistent runtime.
            llm_client: Control-plane LLM client.
            response_llm_client: Response-writing LLM client.
        """

        self._config = config
        self._runtime = runtime
        self._llm_client = llm_client
        self._response_llm_client = response_llm_client

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
        """Send a plain-text reply, split for Telegram limits.

        Args:
            update: Telegram update whose message should receive the reply.
            text: Reply body.

        Returns:
            None.
        """

        message = update.effective_message
        if message is None:
            logger.warning("telegram reply skipped: update has no effective message")
            return

        for chunk in split_telegram_text(text):
            await message.reply_text(chunk)


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

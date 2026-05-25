---
title: Telegram Gateway
sidebar_position: 4
---

import telegramScreenshot from '@site/static/img/telegram-example.jpeg';

# Telegram Gateway

The Telegram gateway is a standalone direct-message channel for local
dogfooding. It uses the persistent text runtime, but it does not
require the FastAPI server.

<img className="docs-screenshot docs-screenshot--portrait" src={telegramScreenshot} alt="OpenCouch Telegram DM" />

## Start polling

Create a bot with `@BotFather`, send the bot one message, then fetch
your numeric sender id:

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
```

Use `message.from.id` in `OPENCOUCH_TELEGRAM_ALLOW_FROM`.

Start the gateway from `apps/backend`:

```bash
OPENCOUCH_TELEGRAM_BOT_TOKEN="123456:abc..." \
OPENCOUCH_TELEGRAM_ALLOW_FROM="123456789" \
OPENCOUCH_TELEGRAM_OWNER_ID="alice" \
OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER=fast \
OPENCOUCH_MEMORY_MODE=persistent \
.venv/bin/python -m channels.gateway telegram
```

If Telegram reports a webhook conflict, clear it once:

```bash
curl "https://api.telegram.org/bot<YOUR_TOKEN>/deleteWebhook"
```

## User commands

| Command | Behavior |
|---|---|
| `/start` | Static introduction and usage hint |
| `/help` | Same static help text |
| `/end` | Finalizes and closes the active session |

After `/end`, the next normal message starts a fresh session. The
user does not need to send `/start` again.

## Configuration

| Variable | Required | Purpose |
|---|---:|---|
| `OPENCOUCH_TELEGRAM_BOT_TOKEN` | Yes | Bot token from BotFather |
| `OPENCOUCH_TELEGRAM_ALLOW_FROM` | Yes | Comma-separated numeric Telegram user ids allowed to talk to the bot |
| `OPENCOUCH_TELEGRAM_OWNER_ID` | Yes | Canonical OpenCouch memory owner id |
| `OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER` | No | Response model tier, default `fast` |
| `OPENCOUCH_TELEGRAM_PROXY` | No | Optional proxy URL |
| `OPENCOUCH_TELEGRAM_DROP_PENDING_UPDATES` | No | Drop queued Telegram updates on startup, default true |
| `OPENCOUCH_MEMORY_MODE` | No | Usually `persistent` for dogfood |

## Thread rotation

Thread rotation is optional but recommended for long dogfood
conversations because it limits context growth for a single Telegram
user. Enable it with:

```bash
OPENCOUCH_TELEGRAM_THREAD_ROTATION_ENABLED=true
```

When rotation is enabled, the gateway currently stores Telegram
session rows in `apps/backend/.store/telegram_sessions.sqlite3` by
default. This registry is still a separate SQLite surface even though
the core runtime persistence stack now supports local Postgres end to
end. It keeps an active pointer per chat, finalizes old sessions,
recovers startup orphan rows, and reclaims closed thread checkpoints
after a grace period.

Related knobs:

| Variable | Default | Purpose |
|---|---:|---|
| `OPENCOUCH_TELEGRAM_SESSION_DB_PATH` | `.store/telegram_sessions.sqlite3` | Session registry path |
| `OPENCOUCH_TELEGRAM_SESSION_TRANSCRIPT_SOFT_LIMIT` | `240` | Rotate once the active transcript gets large |
| `OPENCOUCH_TELEGRAM_ROTATION_SWEEP_INTERVAL_SECONDS` | `30` | Maintenance sweep interval |
| `OPENCOUCH_TELEGRAM_RECLAIM_INTERVAL_SECONDS` | `300` | Closed-thread reclaim interval |
| `OPENCOUCH_TELEGRAM_RECLAIM_GRACE_SECONDS` | `3600` | Minimum age before reclaim |

## Rendering

Telegram messages are sent with safe HTML formatting. The gateway
converts common Markdown-style model output such as bold, italics,
inline links, lists, and paragraphs into Telegram-compatible markup,
then chunks long replies under Telegram's message limit.

The channel is text-only and DM-only. Groups, media, inline buttons,
and richer Telegram-specific UX remain future work.

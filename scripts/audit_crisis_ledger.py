#!/usr/bin/env python3
"""Ad hoc operator CLI for the crisis safety ledger.

Run with the backend virtualenv, for example:

    cd apps/backend && .venv/bin/python ../../scripts/audit_crisis_ledger.py summary --date 2026-06-30

This script intentionally performs review/export/retention work outside the
conversation runtime. Runtime code only captures minimal safety events.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "apps" / "backend"
DEFAULT_SQLITE_PATH = BACKEND_ROOT / ".store" / "crisis.sqlite3"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.audit.crisis_log import CrisisLogBackend  # noqa: E402
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend  # noqa: E402
from agent.audit.sqlite_crisis_log import SqliteCrisisLogBackend  # noqa: E402
from agent.audit.summary import summarize_crisis_log_records  # noqa: E402


class _Args(Protocol):
    backend: str
    sqlite_path: str
    database_url: str | None
    pretty: bool


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "dates must be in YYYY-MM-DD format"
        ) from exc


def _json_print(payload: Any, *, pretty: bool) -> None:
    print(
        json.dumps(
            payload,
            indent=2 if pretty else None,
            sort_keys=pretty,
            default=str,
        )
    )


def _build_backend(args: _Args) -> CrisisLogBackend:
    if args.backend == "postgres":
        dsn = args.database_url or os.environ.get("OPENCOUCH_CRISIS_LOG_DATABASE_URL")
        if not dsn:
            raise SystemExit(
                "--database-url or OPENCOUCH_CRISIS_LOG_DATABASE_URL is required "
                "for --backend postgres"
            )
        return PostgresCrisisLogBackend(dsn)
    return SqliteCrisisLogBackend(Path(args.sqlite_path))


async def _run_summary(args: argparse.Namespace) -> None:
    backend = _build_backend(args)
    try:
        records = await backend.alist_by_date(args.date)
        aggregate = summarize_crisis_log_records(args.date, records)
        _json_print(aggregate.model_dump(mode="json"), pretty=args.pretty)
    finally:
        await backend.aclose()


async def _run_export(args: argparse.Namespace) -> None:
    backend = _build_backend(args)
    try:
        records = await backend.alist_by_date(args.date)
        _json_print(
            [record.model_dump(mode="json") for record in records],
            pretty=args.pretty,
        )
    finally:
        await backend.aclose()


async def _run_purge(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("purge is destructive; pass --yes to confirm")

    backend = _build_backend(args)
    try:
        deleted = await backend.apurge_before(args.before)
        _json_print(
            {"deleted": deleted, "before": args.before.isoformat()},
            pretty=args.pretty,
        )
    finally:
        await backend.aclose()


def _add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("sqlite", "postgres"),
        default="sqlite",
        help="ledger backend to inspect (default: sqlite)",
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(DEFAULT_SQLITE_PATH),
        help=f"SQLite crisis ledger path (default: {DEFAULT_SQLITE_PATH})",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres DSN; may also be set via OPENCOUCH_CRISIS_LOG_DATABASE_URL",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print JSON output",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ad hoc operator commands for the crisis safety ledger."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="print a daily aggregate")
    _add_backend_options(summary)
    summary.add_argument("--date", required=True, type=_parse_day)
    summary.set_defaults(handler=_run_summary)

    export = subparsers.add_parser("export", help="print records for one day")
    _add_backend_options(export)
    export.add_argument("--date", required=True, type=_parse_day)
    export.set_defaults(handler=_run_export)

    purge = subparsers.add_parser("purge", help="delete records before a date")
    _add_backend_options(purge)
    purge.add_argument("--before", required=True, type=_parse_day)
    purge.add_argument("--yes", action="store_true", help="confirm destructive purge")
    purge.set_defaults(handler=_run_purge)

    return parser


async def _amain(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    await args.handler(args)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_amain(argv))


if __name__ == "__main__":
    main()

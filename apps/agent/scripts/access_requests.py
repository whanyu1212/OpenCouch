"""List or remove requests for early access to the private beta (HY-353).

Reads the database from `OPENCOUCH_DATABASE_URL` or `DATABASE_URL`. For the
demo's database, pull its variables from Vercel first, from the repo root
(this writes secrets to `.env.local`, which git ignores):

    vercel env pull .env.local
    cd apps/agent
    uv run --env-file ../../.env.local python scripts/access_requests.py list
    uv run --env-file ../../.env.local python scripts/access_requests.py remove a@x.com

To invite someone, add their address to `OPENCOUCH_BETA_INVITES` on Vercel and
redeploy, then remove their request.
"""

import argparse
import asyncio
import sys

from psycopg_pool import AsyncConnectionPool

from opencouch_agent.beta.invites import normalized_email
from opencouch_agent.beta.postgres import PostgresAccessRequests
from opencouch_agent.config import Settings


async def main(command: str, email: str | None) -> int:
    """Run `command` against the configured database; returns the exit status."""
    database_url = Settings().database_url
    if not database_url:
        print("Set OPENCOUCH_DATABASE_URL or DATABASE_URL first.", file=sys.stderr)
        return 2
    async with AsyncConnectionPool(
        database_url, kwargs={"prepare_threshold": None}
    ) as pool:
        requests = PostgresAccessRequests(pool)
        await requests.create_schema()
        if command == "list":
            for address, requested_at in await requests.all():
                print(f"{requested_at:%Y-%m-%d %H:%M}  {address}")
            return 0
        assert email is not None
        removed = await requests.remove(normalized_email(email))
        print("Removed." if removed else f"No request from {email}.")
        return 0 if removed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Every request, oldest first")
    remove = commands.add_parser("remove", help="Forget one address's request")
    remove.add_argument("email")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.command, getattr(args, "email", None))))

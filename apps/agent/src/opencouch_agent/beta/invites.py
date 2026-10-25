"""The private beta: only invited people can use the coach (HY-353).

Invites are verified email addresses. Without any, nobody gets in: a deploy
that forgets them locks testers out rather than opening the beta to anyone.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BetaInvites:
    """Who may use the coach.

    Attributes:
        emails: Invited addresses, normalised with `normalized_email`.
        everyone: Let every signed-in user in. For tests and local runs only.
    """

    emails: frozenset[str] = frozenset()
    everyone: bool = False

    def allows(self, verified_email: str | None) -> bool:
        """Whether someone with this verified email is invited.

        `None` means Neon Auth hasn't verified their email, which is never enough.
        """
        if self.everyone:
            return True
        return (
            verified_email is not None
            and normalized_email(verified_email) in self.emails
        )


def normalized_email(email: str) -> str:
    """An address as invites compare it: trimmed, and the same in any case."""
    # `lower()`, not `casefold()`: it matches the web app's `toLowerCase()`,
    # which checks the same list before sending a code.
    return email.strip().lower()


def build_beta_invites(
    *, invites: str | None, everyone: bool, model: str
) -> BetaInvites:
    """The invites for the service's settings.

    Args:
        invites: Comma-separated invited email addresses, if set.
        everyone: Whether letting everyone in was asked for explicitly.
        model: The agent's model string.

    Raises:
        ValueError: If letting everyone in is asked for with a real model.
    """
    # Same rule as the other fakes: an open beta never runs next to a real model.
    if everyone:
        if model != "test":
            raise ValueError("OPENCOUCH_BETA_OPEN only runs with OPENCOUCH_MODEL=test")
        logger.warning("The beta is open: every signed-in user can use the coach")
        return BetaInvites(everyone=True)
    emails = frozenset(
        normalized_email(email) for email in (invites or "").split(",") if email.strip()
    )
    if not emails:
        logger.warning(
            "No beta invites are set: every signed-in user sees the waitlist"
        )
    return BetaInvites(emails)

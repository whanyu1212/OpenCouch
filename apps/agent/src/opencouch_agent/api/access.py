"""The public route for requesting early access to the beta (HY-353)."""

from fastapi import HTTPException

from opencouch_agent.beta.access_requests import AccessRequests, requestable_email


async def record_access_request(requests: AccessRequests, raw_email: str) -> None:
    """Record a request for early access.

    Raises:
        HTTPException: 422 if the address can't be one.
    """
    email = requestable_email(raw_email)
    if email is None:
        raise HTTPException(status_code=422, detail="Expected an email address.")
    await requests.add(email)

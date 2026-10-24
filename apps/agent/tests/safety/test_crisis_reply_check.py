"""A crisis reply may only name numbers from the vetted catalog (HY-230)."""

import pytest

from opencouch_agent.safety.crisis_reply_check import allowed_numbers, reply_problem
from opencouch_agent.safety.crisis_resources import load_shipped_catalog

ALLOWED = allowed_numbers(load_shipped_catalog())


@pytest.mark.parametrize(
    "text",
    [
        "Are you safe right now?",
        "Please call 995 now.",
        "You can call SOS on 1767, any time.",
        "WhatsApp their CareText line on 9151 1767.",
        "Or message +65 9151 1767 on WhatsApp.",
        "mindline is on 6669 1771.",
        "The NAVH line is 1800-777-0000.",
        "If it's not safe to call, text 70999.",
        "Call SOS on 1767 24 hours a day.",
        "Call 995 or 999 now.",
    ],
)
def test_catalog_numbers_and_plain_text_may_be_sent(text: str) -> None:
    assert reply_problem(text, ALLOWED) is None


@pytest.mark.parametrize(
    "text",
    [
        "Call or text 988.",
        "Call Samaritans on 116 123.",
        "Call 1-800-273-8255.",
        "Message 9151 1768 on WhatsApp.",
        "This has been going on since 2026.",
        "Call 6 1 2 3 4.",
        "Call 98 8.",
    ],
)
def test_any_other_number_fails_the_reply(text: str) -> None:
    assert reply_problem(text, ALLOWED) == "unknown_number"


@pytest.mark.parametrize(
    "text",
    [
        "Message them at wa.me/6591511767.",
        "See sos.org.sg for more.",
        "Visit https://example.org/help.",
        "Go to www.example.net.",
        "Try samaritans.org.uk.",
        "Lifeline is at lifeline.org.au.",
    ],
)
def test_a_link_fails_the_reply(text: str) -> None:
    assert reply_problem(text, ALLOWED) == "link"


@pytest.mark.parametrize("text", ["", "  \n"])
def test_an_empty_reply_fails(text: str) -> None:
    assert reply_problem(text, ALLOWED) == "empty"


def test_a_whatsapp_number_is_allowed_with_and_without_its_country_code() -> None:
    assert {"6591511767", "91511767"} <= ALLOWED

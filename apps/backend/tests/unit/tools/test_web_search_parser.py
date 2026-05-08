"""Unit tests for the pure-function helpers in ``agent/tools/web_search.py``.

These cover the parser (``_parse_resource_lines``) and its two field-
cleaning helpers (``_clean_field``, ``_clean_url_field``). The LLM-
calling paths (``_extract_location``, ``_lookup_resources``,
``find_local_crisis_resources``) are not exercised here — they
require a real provider and live in the dogfood script. The parser
tests are enough to pin the output shapes from both Gemini and
OpenAI grounding.

Added during v0.8 when OpenAI web_search grounding was wired up.
The OpenAI grounding introduced new output variants that the
original parser didn't handle:

- Markdown bold around every field (``**Name** | **Phone** | ...``)
- Citation suffixes on the URL (``URL ([source](url?utm_source=openai))``)
- Markdown table rows with leading/trailing pipes (``| Name | Phone |``)
- Inner ``**`` sequences in phone fields with multiple numbers
- Header rows and separator rows from markdown tables

Each of these is pinned by a test so a future parser refactor
catches the regression loudly instead of silently producing
mis-parsed crisis resources.
"""

from __future__ import annotations

from agent.tools.web_search import (
    _clean_field,
    _clean_url_field,
    _parse_resource_lines,
)


# ─── _clean_field ─────────────────────────────────────────────────────────


class TestCleanField:
    """Unit tests for the field normalizer."""

    def test_strips_markdown_bold_from_ends(self) -> None:
        """Paired ``**X**`` should lose both markers."""
        assert _clean_field("**Samaritans**") == "Samaritans"

    def test_strips_markdown_bold_from_middle(self) -> None:
        """Embedded ``**`` sequences (e.g., in multi-number phone
        fields) must also be removed, not just the ones at the
        field boundaries. Otherwise fields like
        ``**0120-A**; **050-B**`` keep inner markers."""
        cleaned = _clean_field("**0120-279-338**; **050-3655-0279** for IP phones")
        assert "**" not in cleaned
        # The substantive content survives
        assert "0120-279-338" in cleaned
        assert "050-3655-0279" in cleaned

    def test_collapses_whitespace(self) -> None:
        """Removing the ``**`` markers shouldn't leave double spaces
        behind — the parser normalizes runs of whitespace into a
        single space."""
        cleaned = _clean_field("**Name**  with   extra   spaces")
        assert cleaned == "Name with extra spaces"

    def test_empty_and_whitespace_only_return_empty(self) -> None:
        """Empty input → empty output; whitespace-only same."""
        assert _clean_field("") == ""
        assert _clean_field("   ") == ""
        assert _clean_field("****") == ""

    def test_preserves_plain_text(self) -> None:
        """Gemini-style plain output (no markdown) passes through
        unchanged except for edge whitespace."""
        assert _clean_field("Samaritans") == "Samaritans"
        assert _clean_field("  Samaritans  ") == "Samaritans"


# ─── _clean_url_field ─────────────────────────────────────────────────────


class TestCleanUrlField:
    """Unit tests for the URL-specific normalizer."""

    def test_strips_openai_citation_suffix(self) -> None:
        """OpenAI's ``URL ([source](url?utm_source=openai))`` suffix
        must be removed so the cleaned URL is just the real URL.

        Regression guard: this is the biggest failure mode. Without
        the citation strip the URL field would contain nested parens
        and confuse any downstream URL rendering."""
        url_with_citation = (
            "https://www.samaritans.org/help "
            "([samaritans.org](https://samaritans.org?utm_source=openai))"
        )
        assert _clean_url_field(url_with_citation) == "https://www.samaritans.org/help"

    def test_strips_both_citation_and_bold(self) -> None:
        """OpenAI often emits ``**URL** ([citation]...)``. BOTH the
        outer bold markers and the citation tail must be removed.

        Ordering matters: the citation strip must run BEFORE the
        bold strip, because cutting the citation first brings the
        trailing ``**`` back to the end of the string where the
        bold stripper can see it. If the order is reversed, the
        trailing bold is still nested inside the citation and
        doesn't get removed."""
        raw = "**https://samaritans.org/** ([samaritans.org](https://samaritans.org?utm_source=openai))"
        assert _clean_url_field(raw) == "https://samaritans.org/"

    def test_plain_url_passes_through(self) -> None:
        """Gemini-style plain URL (no markdown, no citation) is
        unchanged except for whitespace."""
        assert _clean_url_field("https://samaritans.org") == "https://samaritans.org"
        assert (
            _clean_url_field("  https://samaritans.org  ") == "https://samaritans.org"
        )

    def test_trailing_bold_on_bare_url(self) -> None:
        """``samaritans.org**`` with nothing else should lose the bold."""
        assert _clean_url_field("samaritans.org**") == "samaritans.org"


# ─── _parse_resource_lines ────────────────────────────────────────────────


class TestParseResourceLines:
    """Unit tests for the end-to-end line parser.

    These exercise the full parsing pipeline across the output
    variants we see from both Gemini and OpenAI.
    """

    def test_plain_pipe_format_gemini_style(self) -> None:
        """The original format: bullets + plain pipe-separated fields."""
        raw = """\
- Samaritans | 116 123 | https://www.samaritans.org
- Shout | Text SHOUT to 85258 | https://giveusashout.org
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 2
        assert resources[0]["name"] == "Samaritans"
        assert resources[0]["phone"] == "116 123"
        assert resources[0]["url"] == "https://www.samaritans.org"
        assert resources[0]["region"] == "UK"

    def test_markdown_table_format_with_leading_pipes(self) -> None:
        """Markdown tables have leading/trailing pipes per row:
        ``| Name | Phone | URL |``. The parser must strip those
        bracketing pipes before splitting so the column indices
        line up.

        Regression guard: without the leading-pipe strip, the first
        split result is an empty string, ``name`` becomes ``""`` and
        falls back to ``"Crisis Line"``, and every other field gets
        shifted one position to the right — phone ends up in the
        URL field, etc. This was a real bug surfaced during OpenAI
        grounding dogfood."""
        raw = """\
| Name | Phone | Website |
|---|---:|---|
| Samaritans | 116 123 | https://samaritans.org |
| Shout | Text SHOUT to 85258 | https://giveusashout.org |
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 2
        assert resources[0]["name"] == "Samaritans"
        assert resources[0]["phone"] == "116 123"
        assert resources[0]["url"] == "https://samaritans.org"

    def test_skips_markdown_table_header_row(self) -> None:
        """The header row ``| Name | Phone | Website |`` must be
        recognized and skipped — it's metadata, not a resource.

        Detection works on either the phone or name field: if the
        field value equals ``"phone"`` / ``"name"`` / ``"---"``
        (case-insensitive) it's metadata."""
        raw = """\
| Name | Phone | Website |
| Samaritans | 116 123 | https://samaritans.org |
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 1
        assert resources[0]["name"] == "Samaritans"

    def test_skips_markdown_table_separator_row(self) -> None:
        """The ``|---|---|---|`` alignment row must also be skipped."""
        raw = """\
| Samaritans | 116 123 | https://samaritans.org |
|---|---:|---|
| Shout | 85258 | https://giveusashout.org |
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 2

    def test_openai_markdown_bold_and_citation_format(self) -> None:
        """Full OpenAI-grounded output: every field wrapped in bold,
        URL suffixed with a citation, bracketed by pipes.

        Regression guard for the most common real-world OpenAI
        output shape. All three normalizations (leading-pipe strip,
        bold strip, citation strip) must compose correctly."""
        raw = """\
| **Samaritans** | **116 123** | **https://samaritans.org** ([samaritans.org](https://samaritans.org?utm_source=openai)) |
| **Shout** | **Text SHOUT to 85258** | **https://giveusashout.org** ([giveusashout.org](https://giveusashout.org?utm_source=openai)) |
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 2
        assert resources[0]["name"] == "Samaritans"
        assert resources[0]["phone"] == "116 123"
        assert resources[0]["url"] == "https://samaritans.org"
        assert resources[1]["name"] == "Shout"
        assert resources[1]["phone"] == "Text SHOUT to 85258"
        assert resources[1]["url"] == "https://giveusashout.org"
        # No ``**`` anywhere in the output
        for r in resources:
            for field in (r["name"], r["phone"], r["url"]):
                assert "**" not in field

    def test_skips_rows_with_unverified_phone(self) -> None:
        """The LLM sometimes emits rows with ``no phone`` or ``phone
        not verified`` as the phone field when it couldn't find a
        number. These are informational noise — the CLI has nothing
        to dial — and should be dropped.

        Regression guard: the original parser would have returned
        these as resources with ``'No phone number'`` as the phone
        field, which confuses the downstream rendering."""
        raw = """\
| Samaritans | 116 123 | https://samaritans.org |
| Inochi SOS | Phone number not verified in sources | https://inochinodenwa.org |
| TELL Chat | No phone number | https://telljp.com |
"""
        resources = _parse_resource_lines(raw, location="JP")
        assert len(resources) == 1
        assert resources[0]["name"] == "Samaritans"

    def test_skips_non_actionable_contact_placeholders(self) -> None:
        """Rows with placeholders should not become dialable resources."""

        raw = """\
| Samaritans | 116 123 | https://samaritans.org |
| Website Only | See website | https://example.org |
| Emergency Advice | Call local emergency services | https://example.org |
| Unknown Hotline | N/A | https://example.org |
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 1
        assert resources[0]["name"] == "Samaritans"

    def test_respects_max_resources_cap(self) -> None:
        """At most ``_MAX_RESOURCES`` (5) rows are returned."""
        lines = "\n".join(
            f"- Resource{i} | {i}00{i}00{i} | https://r{i}.example.com"
            for i in range(10)
        )
        resources = _parse_resource_lines(lines, location="UK")
        assert len(resources) == 5

    def test_lines_without_pipe_are_ignored(self) -> None:
        """Prose lines without ``|`` are silently dropped. This
        matches the graceful-degradation philosophy of the whole
        module — we never raise, just skip unusable content."""
        raw = """\
Here are some resources for you:

- Samaritans | 116 123 | https://samaritans.org

Let me know if you need more details.
"""
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 1

    def test_empty_input_returns_empty_list(self) -> None:
        """No crashes on empty / whitespace-only input."""
        assert _parse_resource_lines("", location="UK") == []
        assert _parse_resource_lines("   \n\n  ", location="UK") == []

    def test_two_field_row_missing_url(self) -> None:
        """A row with only ``Name | Phone`` parses with an empty URL."""
        raw = "- Samaritans | 116 123"
        resources = _parse_resource_lines(raw, location="UK")
        assert len(resources) == 1
        assert resources[0]["url"] == ""

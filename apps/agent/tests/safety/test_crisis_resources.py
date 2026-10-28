import tomllib
from typing import get_args

import pytest
from pydantic import ValidationError

from opencouch_agent.safety.crisis_resources import (
    CrisisType,
    load_catalog,
    load_shipped_catalog,
    select_resources,
)

CATALOG = load_shipped_catalog()
ALL_TYPES: tuple[CrisisType, ...] = get_args(CrisisType)
# Types with no Singapore entry of their own yet. Adding one should change this.
GAP_TYPES = {
    "harm_to_others",
    "substance_use",
    "risk_taking",
    "medical_emergency",
    "psychosis",
    "eating_disorder",
}

VALID = """
[[resources]]
key = "sg-general"
name = "General line"
region = "SG"
crisis_types = ["general_distress"]
sources = ["https://example.org/"]
verified_on = 2026-10-23
verified_by = "reviewer"
[[resources.channels]]
kind = "whatsapp"
label = "Chat"
number = "6512345678"
display = "1234 5678"

[[resources]]
key = "sg-emergency"
name = "Emergency"
region = "SG"
emergency = true
sources = ["https://example.org/"]
verified_on = 2026-10-23
verified_by = "reviewer"
[[resources.channels]]
kind = "call"
label = "Ambulance"
number = "995"
display = "995"
"""


def keys(crisis_type: CrisisType | None, *, imminent: bool = False) -> list[str]:
    selected = select_resources(CATALOG, "SG", crisis_type, imminent=imminent)
    return [resource.key for resource in selected]


def test_the_shipped_catalog_has_the_vetted_entries() -> None:
    assert [r.key for r in CATALOG.resources] == [
        "sg-sos",
        "sg-mindline",
        "sg-navh",
        "sg-emergency",
    ]


def test_the_emergency_entry_has_the_numbers_the_official_pages_give() -> None:
    (emergency,) = [r for r in CATALOG.resources if r.emergency]

    assert {(c.kind, c.number) for c in emergency.channels} == {
        ("call", "995"),
        ("call", "999"),
        ("sms", "70999"),
        ("sms", "70995"),
    }
    (scdf_sms,) = [c for c in emergency.channels if c.number == "70995"]
    assert "deaf" in scdf_sms.label


def test_every_shipped_entry_states_its_hours() -> None:
    assert all(resource.hours == "24 hours" for resource in CATALOG.resources)


@pytest.mark.parametrize(
    ("key", "url"),
    [
        (
            "sg-sos",
            "https://wa.me/6591511767?text=Hi%2C%20I%27d%20like%20to%20talk%20to%20someone.",
        ),
        (
            "sg-mindline",
            "https://wa.me/6566691771?text=Hi%2C%20I%27d%20like%20to%20talk%20to%20someone.",
        ),
    ],
)
def test_whatsapp_links_use_the_country_code_and_the_neutral_opener(
    key: str, url: str
) -> None:
    (resource,) = [r for r in CATALOG.resources if r.key == key]

    assert [c.url for c in resource.channels if c.kind == "whatsapp"] == [url]


def test_only_whatsapp_channels_have_links() -> None:
    other = [c for r in CATALOG.resources for c in r.channels if c.kind != "whatsapp"]

    assert other
    assert all(channel.url is None for channel in other)


@pytest.mark.parametrize("crisis_type", [*ALL_TYPES, None])
@pytest.mark.parametrize("imminent", [True, False])
def test_the_emergency_entry_is_always_shown(
    crisis_type: CrisisType | None, imminent: bool
) -> None:
    assert "sg-emergency" in keys(crisis_type, imminent=imminent)


def test_imminent_danger_puts_emergency_first_and_otherwise_last() -> None:
    assert keys("suicide_self_harm", imminent=True) == ["sg-emergency", "sg-sos"]
    assert keys("suicide_self_harm", imminent=False) == ["sg-sos", "sg-emergency"]


def test_an_unknown_type_is_treated_as_suicide_or_self_harm() -> None:
    assert keys(None) == keys("suicide_self_harm")


def test_abuse_gets_the_anti_violence_helpline() -> None:
    assert keys("abuse_violence") == ["sg-navh", "sg-emergency"]


def test_types_without_an_entry_fall_back_to_the_general_ones() -> None:
    gaps = {
        crisis_type
        for crisis_type in ALL_TYPES
        if not any(crisis_type in r.crisis_types for r in CATALOG.resources)
    }

    assert gaps == GAP_TYPES
    for crisis_type in gaps:
        assert keys(crisis_type) == ["sg-sos", "sg-mindline", "sg-emergency"]


def test_a_valid_catalog_loads() -> None:
    assert len(load_catalog(VALID).resources) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("6512345678", "12345678"), "must start with 65"),
        (('key = "sg-emergency"', 'key = "sg-general"'), "unique"),
        (("emergency = true", "emergency = false"), "list crisis types"),
        (
            ("emergency = true", 'emergency = true\ncrisis_types = ["psychosis"]'),
            "list crisis types",
        ),
        (('["general_distress"]', '["suicide_self_harm"]'), "general_distress entry"),
        (('["general_distress"]', '["general_distress", "grief"]'), "grief"),
        (('"https://example.org/"', '"http://example.org/"'), "https://"),
        (('number = "995"', 'number = "+65 995"'), "pattern"),
        (('display = "995"', 'display = "995"\nnote = "x"'), "Extra inputs"),
        (('display = "995"', 'display = ""'), "at least 1 character"),
        (('label = "Chat"', 'label = ""'), "at least 1 character"),
        (('name = "Emergency"', 'name = ""'), "at least 1 character"),
    ],
)
def test_a_bad_entry_fails_to_load(change: tuple[str, str], message: str) -> None:
    old, new = change
    assert old in VALID

    with pytest.raises(ValidationError, match=message):
        load_catalog(VALID.replace(old, new, 1))


def test_a_catalog_without_an_emergency_entry_fails_to_load() -> None:
    general_only = VALID.split("[[resources]]")[1]

    with pytest.raises(ValidationError, match="exactly one emergency entry"):
        load_catalog(f"[[resources]]{general_only}")


def test_a_supported_region_without_entries_fails_to_load() -> None:
    with pytest.raises(ValidationError, match="SG needs exactly one emergency"):
        load_catalog("resources = []")


def test_an_entry_without_channels_fails_to_load() -> None:
    no_channels = VALID.replace(
        '[[resources.channels]]\nkind = "call"\nlabel = "Ambulance"\n'
        'number = "995"\ndisplay = "995"\n',
        "",
    )

    with pytest.raises(ValidationError, match="channels"):
        load_catalog(no_channels)


def test_invalid_toml_fails_to_load() -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_catalog("[[resources]\n")

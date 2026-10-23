"""The vetted crisis resource catalog (HY-209), and choosing what to show.

Crisis resources are never written by the model. The catalog is fixed,
reviewed data in `crisis_resources.toml`, checked when the app starts: a bad
entry stops startup. The crisis path picks entries by crisis type, and the
client renders them from their keys.
"""

import tomllib
from datetime import date
from importlib import resources
from typing import Annotated, Final, Literal, Self, get_args
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

CrisisType = Literal[
    "suicide_self_harm",
    "harm_to_others",
    "substance_use",
    "panic_anxiety",
    "risk_taking",
    "abuse_violence",
    "medical_emergency",
    "psychosis",
    "eating_disorder",
    "general_distress",
]
"""The kinds of crisis OpenCouch prepares for. HY-298 detects them; HY-230
answers each differently."""

Region = Literal["SG"]

# Today's CRISIS rules are all about suicide and self-harm, so until HY-298
# detects the type, that's what a crisis is taken to be.
DEFAULT_CRISIS_TYPE: Final[CrisisType] = "suicide_self_harm"
# Types with no entry of their own get the general ones.
FALLBACK_CRISIS_TYPE: Final[CrisisType] = "general_distress"
# Neutral by design: never anything from the user's conversation.
WHATSAPP_OPENER: Final = "Hi, I'd like to talk to someone."

_COUNTRY_CODES: Final[dict[Region, str]] = {"SG": "65"}
_CATALOG_FILE: Final = "crisis_resources.toml"

Digits = Annotated[str, Field(pattern=r"^\d{3,15}$")]
Text = Annotated[str, Field(min_length=1)]
HttpsUrl = Annotated[str, Field(pattern=r"^https://")]


class Channel(BaseModel):
    """One way to reach a service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["whatsapp", "call", "sms"]
    label: Text
    number: Digits
    display: Text

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str | None:
        """The `wa.me` link for a WhatsApp channel; None for the others."""
        if self.kind != "whatsapp":
            return None
        return f"https://wa.me/{self.number}?text={quote(WHATSAPP_OPENER)}"


class CrisisResource(BaseModel):
    """One vetted service in the catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: Annotated[str, Field(pattern=r"^[a-z]{2}-[a-z0-9-]+$")]
    name: Text
    region: Region
    emergency: bool = False
    crisis_types: tuple[CrisisType, ...] = ()
    hours: str | None = None
    channels: Annotated[tuple[Channel, ...], Field(min_length=1)]
    sources: Annotated[tuple[HttpsUrl, ...], Field(min_length=1)]
    verified_on: date
    verified_by: Annotated[str, Field(exclude=True)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        # The emergency entry is shown for every type, so it lists none.
        if self.emergency == bool(self.crisis_types):
            raise ValueError(
                f"{self.key}: list crisis types unless it's the emergency entry"
            )
        country_code = _COUNTRY_CODES[self.region]
        for channel in self.channels:
            if channel.kind == "whatsapp" and not channel.number.startswith(
                country_code
            ):
                raise ValueError(
                    f"{self.key}: WhatsApp number must start with {country_code}"
                )
        return self


class CrisisCatalog(BaseModel):
    """Every vetted crisis resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resources: tuple[CrisisResource, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        keys = [resource.key for resource in self.resources]
        if len(keys) != len(set(keys)):
            raise ValueError("crisis resource keys must be unique")
        # Every supported region, not just those with entries: a region the
        # app serves must never come up empty.
        for region in get_args(Region):
            in_region = [r for r in self.resources if r.region == region]
            # Safety: every region must always be able to show emergency
            # numbers and something for types with no entry of their own.
            if sum(r.emergency for r in in_region) != 1:
                raise ValueError(f"{region} needs exactly one emergency entry")
            if not any(FALLBACK_CRISIS_TYPE in r.crisis_types for r in in_region):
                raise ValueError(f"{region} needs a {FALLBACK_CRISIS_TYPE} entry")
        return self


def load_catalog(text: str) -> CrisisCatalog:
    """Parse and check a catalog.

    Raises:
        tomllib.TOMLDecodeError: If `text` isn't valid TOML.
        pydantic.ValidationError: If any entry breaks the catalog's rules.
    """
    return CrisisCatalog.model_validate(tomllib.loads(text))


def load_shipped_catalog() -> CrisisCatalog:
    """The catalog that ships with the service."""
    text = resources.files(__package__).joinpath(_CATALOG_FILE).read_text()
    return load_catalog(text)


def select_resources(
    catalog: CrisisCatalog,
    region: Region,
    crisis_type: CrisisType | None,
    *,
    imminent: bool,
) -> list[CrisisResource]:
    """The resources to show for a crisis, in the order to show them.

    Args:
        catalog: The vetted catalog.
        region: Where the user is.
        crisis_type: The kind of crisis, or None when it isn't known.
        imminent: Whether the danger is immediate (a plan, the means and a
            timeline, or an attempt underway).

    Returns:
        The entries for the type, or the general ones when it has none, with
        the emergency entry always included: first when the danger is
        imminent, last otherwise.
    """
    in_region = [r for r in catalog.resources if r.region == region]
    wanted = crisis_type or DEFAULT_CRISIS_TYPE
    matched = [r for r in in_region if wanted in r.crisis_types] or [
        r for r in in_region if FALLBACK_CRISIS_TYPE in r.crisis_types
    ]
    emergency = [r for r in in_region if r.emergency]
    return [*emergency, *matched] if imminent else [*matched, *emergency]

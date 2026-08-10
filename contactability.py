"""Pure, offline contracts for actionable business contact channels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Protocol

from instagram_candidate_matching import normalize_instagram_profile_url


class ContactChannel(str, Enum):
    INSTAGRAM = "instagram"
    PHONE = "phone"
    EMAIL = "email"


@dataclass(frozen=True)
class Contactability:
    channels: tuple[ContactChannel, ...]
    preferred_channel: ContactChannel | None
    normalized_phone: str | None
    normalized_email: str | None
    instagram_available: bool

    def __post_init__(self) -> None:
        if not isinstance(self.channels, tuple) or any(
            not isinstance(channel, ContactChannel) for channel in self.channels
        ):
            raise TypeError("channels must be a tuple of ContactChannel values")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")
        expected_order = tuple(
            channel
            for channel in (
                ContactChannel.INSTAGRAM,
                ContactChannel.PHONE,
                ContactChannel.EMAIL,
            )
            if channel in self.channels
        )
        if self.channels != expected_order:
            raise ValueError("channels must use Instagram, phone, email priority order")
        expected_preferred = self.channels[0] if self.channels else None
        if self.preferred_channel is not expected_preferred:
            raise ValueError("preferred_channel must be the first available channel")
        if self.instagram_available is not (ContactChannel.INSTAGRAM in self.channels):
            raise ValueError("instagram_available must match channels")
        if (self.normalized_phone is not None) is not (
            ContactChannel.PHONE in self.channels
        ):
            raise ValueError("normalized_phone must match channels")
        if (self.normalized_email is not None) is not (
            ContactChannel.EMAIL in self.channels
        ):
            raise ValueError("normalized_email must match channels")

    @property
    def actionable(self) -> bool:
        return bool(self.channels)


class BusinessContactSource(Protocol):
    instagram_url: str
    phone: str
    email: str


_PHONE_ALLOWED = re.compile(r"\+?[0-9 ().-]+\Z")
_EMAIL_LOCAL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+\Z")
_EMAIL_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")
_PLACEHOLDER_DOMAINS = frozenset(
    {"example.com", "example.org", "test.com", "invalid", "localhost"}
)


def normalize_phone(value: object) -> str | None:
    """Return a syntax-valid phone without guessing or adding a country code."""

    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or any(character in "\r\n\t\v\f" for character in raw):
        return None
    if _PHONE_ALLOWED.fullmatch(raw) is None:
        return None
    if raw.count("+") > 1 or ("+" in raw and not raw.startswith("+")):
        return None
    if raw.count("(") != raw.count(")"):
        return None
    depth = 0
    for character in raw:
        if character == "(":
            depth += 1
            if depth > 1:
                return None
        elif character == ")":
            depth -= 1
            if depth < 0:
                return None
    digits = "".join(character for character in raw if character.isdigit())
    if not 7 <= len(digits) <= 15:
        return None
    if len(set(digits)) == 1:
        return None
    return f"+{digits}" if raw.startswith("+") else digits


def normalize_email(value: object) -> str | None:
    """Return a conservative syntax-valid business email, without DNS checks."""

    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or len(raw) > 254 or raw.count("@") != 1:
        return None
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in raw
    ):
        return None
    local, domain = raw.rsplit("@", 1)
    if not local or len(local) > 64 or not domain or len(domain) > 253:
        return None
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return None
    if _EMAIL_LOCAL.fullmatch(local) is None:
        return None
    if domain.endswith("."):
        return None
    normalized_domain = domain.casefold()
    if normalized_domain in _PLACEHOLDER_DOMAINS or "." not in normalized_domain:
        return None
    labels = normalized_domain.split(".")
    if any(_EMAIL_LABEL.fullmatch(label) is None for label in labels):
        return None
    return f"{local}@{normalized_domain}"


def normalized_instagram_profile(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return normalize_instagram_profile_url(value.strip())
    except (TypeError, ValueError):
        return None


def contactability_from_business(business: BusinessContactSource) -> Contactability:
    instagram = normalized_instagram_profile(business.instagram_url)
    phone = normalize_phone(business.phone)
    email = normalize_email(business.email)
    channels = tuple(
        channel
        for channel, available in (
            (ContactChannel.INSTAGRAM, instagram is not None),
            (ContactChannel.PHONE, phone is not None),
            (ContactChannel.EMAIL, email is not None),
        )
        if available
    )
    return Contactability(
        channels=channels,
        preferred_channel=channels[0] if channels else None,
        normalized_phone=phone,
        normalized_email=email,
        instagram_available=instagram is not None,
    )


def lead_contact_bucket(business: BusinessContactSource) -> str:
    contactability = contactability_from_business(business)
    if len(contactability.channels) > 1:
        return "multi_contact"
    if not contactability.channels:
        return "none"
    return {
        ContactChannel.INSTAGRAM: "instagram",
        ContactChannel.PHONE: "phone_only",
        ContactChannel.EMAIL: "email_only",
    }[contactability.channels[0]]

"""Deterministic contracts and pure extraction for first-party Instagram evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser
import json
from typing import Any

from instagram_candidate_matching import (
    normalize_instagram_profile_url,
    normalize_instagram_username,
)


class FirstPartyInstagramStatus(str, Enum):
    FOUND_OFFICIAL = "found_official"
    NOT_FOUND = "not_found"
    UNCERTAIN = "uncertain"
    TECHNICAL_ERROR = "technical_error"
    SKIPPED = "skipped"


class FirstPartyInstagramEvidenceSource(str, Enum):
    HTML_ANCHOR = "html_anchor"
    JSON_LD_SAME_AS = "json_ld_same_as"


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True)
class FirstPartyInstagramResolution:
    status: FirstPartyInstagramStatus
    resolved_url: str | None
    username: str | None
    evidence_sources: tuple[FirstPartyInstagramEvidenceSource, ...]
    pages_attempted: int
    pages_succeeded: int
    error_category: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, FirstPartyInstagramStatus):
            raise TypeError("status must be a FirstPartyInstagramStatus")
        if type(self.pages_attempted) is not int or self.pages_attempted < 0:
            raise ValueError("pages_attempted must be a non-negative integer")
        if type(self.pages_succeeded) is not int or not (
            0 <= self.pages_succeeded <= self.pages_attempted
        ):
            raise ValueError("pages_succeeded must be between 0 and pages_attempted")
        if not isinstance(self.evidence_sources, tuple):
            raise TypeError("evidence_sources must be a tuple")
        if any(
            not isinstance(source, FirstPartyInstagramEvidenceSource)
            for source in self.evidence_sources
        ):
            raise TypeError("evidence_sources contains an invalid source")
        if len(set(self.evidence_sources)) != len(self.evidence_sources):
            raise ValueError("evidence_sources must not contain duplicates")

        normalized_error = _optional_text(self.error_category, "error_category")
        object.__setattr__(self, "error_category", normalized_error)

        if self.status is FirstPartyInstagramStatus.FOUND_OFFICIAL:
            if self.resolved_url is None or self.username is None:
                raise ValueError("FOUND_OFFICIAL requires resolved_url and username")
            normalized_username = normalize_instagram_username(self.username)
            normalized_url = (
                f"https://www.instagram.com/{normalized_username}/"
            )
            if self.username != normalized_username:
                raise ValueError("username must already be normalized")
            if self.resolved_url != normalized_url:
                raise ValueError("resolved_url must be the canonical direct profile URL")
            if not self.evidence_sources:
                raise ValueError("FOUND_OFFICIAL requires at least one evidence source")
            if normalized_error is not None:
                raise ValueError("FOUND_OFFICIAL cannot include an error")
            return

        if self.resolved_url is not None or self.username is not None:
            raise ValueError(f"{self.status.name} cannot include a resolved profile")
        if self.status is FirstPartyInstagramStatus.TECHNICAL_ERROR:
            if normalized_error is None:
                raise ValueError("TECHNICAL_ERROR requires error_category")
        elif normalized_error is not None:
            raise ValueError(f"{self.status.name} cannot include an error")
        if (
            self.status in {
                FirstPartyInstagramStatus.NOT_FOUND,
                FirstPartyInstagramStatus.SKIPPED,
            }
            and self.evidence_sources
        ):
            raise ValueError(f"{self.status.name} cannot include evidence")


@dataclass(frozen=True)
class ExtractedInstagramProfile:
    """One normalized username with all explicit first-party evidence sources."""

    resolved_url: str
    username: str
    evidence_sources: tuple[FirstPartyInstagramEvidenceSource, ...]


class _FirstPartyHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchor_hrefs: list[str] = []
        self.json_ld_documents: list[str] = []
        self._json_ld_depth = 0
        self._json_ld_parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {name.casefold(): value for name, value in attrs}
        if tag.casefold() == "a" and attributes.get("href"):
            self.anchor_hrefs.append(attributes["href"] or "")
        if tag.casefold() != "script":
            return
        script_type = (attributes.get("type") or "").split(";", 1)[0].strip().casefold()
        if script_type == "application/ld+json":
            self._json_ld_depth += 1
            if self._json_ld_depth == 1:
                self._json_ld_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "script" or self._json_ld_depth <= 0:
            return
        self._json_ld_depth -= 1
        if self._json_ld_depth == 0:
            self.json_ld_documents.append("".join(self._json_ld_parts))
            self._json_ld_parts = []

    def handle_data(self, data: str) -> None:
        if self._json_ld_depth:
            self._json_ld_parts.append(data)


def _same_as_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() == "sameas":
                if isinstance(child, str):
                    values.append(child)
                elif isinstance(child, list):
                    values.extend(item for item in child if isinstance(item, str))
            values.extend(_same_as_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_same_as_values(child))
    return values


def _add_profile(
    profiles: dict[str, set[FirstPartyInstagramEvidenceSource]],
    value: str,
    source: FirstPartyInstagramEvidenceSource,
) -> None:
    try:
        normalized_url = normalize_instagram_profile_url(value)
        username = normalize_instagram_username(normalized_url)
    except (TypeError, ValueError):
        return
    profiles.setdefault(username, set()).add(source)


def extract_instagram_profiles_from_html(
    html: str,
) -> tuple[ExtractedInstagramProfile, ...]:
    """Extract only direct profile anchors and JSON-LD ``sameAs`` URLs."""

    if not isinstance(html, str):
        raise TypeError("html must be a string")
    parser = _FirstPartyHTMLParser()
    parser.feed(html)
    parser.close()

    profiles: dict[str, set[FirstPartyInstagramEvidenceSource]] = {}
    for href in parser.anchor_hrefs:
        _add_profile(
            profiles,
            href,
            FirstPartyInstagramEvidenceSource.HTML_ANCHOR,
        )
    for document in parser.json_ld_documents:
        try:
            parsed = json.loads(document)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for value in _same_as_values(parsed):
            _add_profile(
                profiles,
                value,
                FirstPartyInstagramEvidenceSource.JSON_LD_SAME_AS,
            )

    source_order = {
        FirstPartyInstagramEvidenceSource.HTML_ANCHOR: 0,
        FirstPartyInstagramEvidenceSource.JSON_LD_SAME_AS: 1,
    }
    return tuple(
        ExtractedInstagramProfile(
            resolved_url=f"https://www.instagram.com/{username}/",
            username=username,
            evidence_sources=tuple(
                sorted(profiles[username], key=source_order.__getitem__)
            ),
        )
        for username in sorted(profiles)
    )


def resolution_from_extracted_profiles(
    profiles: tuple[ExtractedInstagramProfile, ...],
    *,
    pages_attempted: int,
    pages_succeeded: int,
) -> FirstPartyInstagramResolution:
    if not isinstance(profiles, tuple):
        raise TypeError("profiles must be a tuple")
    if not profiles:
        return FirstPartyInstagramResolution(
            FirstPartyInstagramStatus.NOT_FOUND,
            None,
            None,
            (),
            pages_attempted,
            pages_succeeded,
        )
    evidence_sources = tuple(
        source
        for source in FirstPartyInstagramEvidenceSource
        if any(source in profile.evidence_sources for profile in profiles)
    )
    if len(profiles) == 1:
        profile = profiles[0]
        return FirstPartyInstagramResolution(
            FirstPartyInstagramStatus.FOUND_OFFICIAL,
            profile.resolved_url,
            profile.username,
            profile.evidence_sources,
            pages_attempted,
            pages_succeeded,
        )
    return FirstPartyInstagramResolution(
        FirstPartyInstagramStatus.UNCERTAIN,
        None,
        None,
        evidence_sources,
        pages_attempted,
        pages_succeeded,
    )

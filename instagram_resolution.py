"""Provider-independent contracts for conservative Instagram resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from website_resolution import normalize_candidate_url


class InstagramResolutionStatus(str, Enum):
    FOUND_OFFICIAL = "found_official"
    NOT_FOUND = "not_found"
    UNCERTAIN = "uncertain"
    RESOLUTION_ERROR = "resolution_error"


class InstagramCandidateSource(str, Enum):
    WEB_SEARCH = "web_search"
    EXISTING = "existing"


def _normalized_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _normalized_text(value, name)


def _confidence(value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError("confidence must be an integer or float")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return result


@dataclass(frozen=True)
class InstagramIdentity:
    business_name: str
    city: str
    address: str | None = None
    phone: str | None = None
    website_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "business_name", _normalized_text(self.business_name, "business_name")
        )
        object.__setattr__(self, "city", _normalized_text(self.city, "city"))
        object.__setattr__(self, "address", _optional_text(self.address, "address"))
        if self.phone is not None:
            from website_candidate_matching import normalize_phone_number

            object.__setattr__(self, "phone", normalize_phone_number(self.phone))
        if self.website_url is not None:
            object.__setattr__(
                self, "website_url", normalize_candidate_url(self.website_url)
            )


@dataclass(frozen=True)
class InstagramCandidateEvidence:
    """Safe assessment data; it intentionally excludes identity and source text."""

    source: InstagramCandidateSource
    candidate_url: str
    username: str
    matched_signals: tuple[str, ...]
    rejected_reason: str | None
    confidence: float
    technical_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, InstagramCandidateSource):
            raise TypeError("source must be an InstagramCandidateSource")
        from instagram_candidate_matching import (
            normalize_instagram_profile_url,
            normalize_instagram_username,
        )

        normalized_url = normalize_instagram_profile_url(self.candidate_url)
        if self.candidate_url != normalized_url:
            raise ValueError("candidate_url must already be normalized")
        normalized_username = normalize_instagram_username(self.username)
        if self.username != normalized_username:
            raise ValueError("username must already be normalized")
        if normalize_instagram_username(normalized_url) != normalized_username:
            raise ValueError("username must match candidate_url")
        if not isinstance(self.matched_signals, tuple):
            raise TypeError("matched_signals must be a tuple")
        normalized_signals: list[str] = []
        seen: set[str] = set()
        for index, signal in enumerate(self.matched_signals):
            item = _normalized_text(signal, f"matched_signals[{index}]")
            key = item.casefold()
            if key in seen:
                raise ValueError("matched_signals must not contain duplicates")
            seen.add(key)
            normalized_signals.append(item)
        object.__setattr__(self, "matched_signals", tuple(normalized_signals))
        object.__setattr__(
            self, "rejected_reason", _optional_text(self.rejected_reason, "rejected_reason")
        )
        object.__setattr__(
            self, "technical_error", _optional_text(self.technical_error, "technical_error")
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))


@dataclass(frozen=True)
class InstagramResolution:
    status: InstagramResolutionStatus
    resolved_url: str | None
    username: str | None
    source: InstagramCandidateSource | None
    confidence: float
    evidence: tuple[InstagramCandidateEvidence, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, InstagramResolutionStatus):
            raise TypeError("status must be an InstagramResolutionStatus")
        from instagram_candidate_matching import (
            normalize_instagram_profile_url,
            normalize_instagram_username,
        )

        if self.resolved_url is not None:
            normalized_url = normalize_instagram_profile_url(self.resolved_url)
            if self.resolved_url != normalized_url:
                raise ValueError("resolved_url must already be normalized")
        if self.username is not None:
            normalized_username = normalize_instagram_username(self.username)
            if self.username != normalized_username:
                raise ValueError("username must already be normalized")
        if self.source is not None and not isinstance(
            self.source, InstagramCandidateSource
        ):
            raise TypeError("source must be an InstagramCandidateSource or None")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")
        if not all(isinstance(item, InstagramCandidateEvidence) for item in self.evidence):
            raise TypeError("evidence must contain InstagramCandidateEvidence items")
        object.__setattr__(self, "error", _optional_text(self.error, "error"))
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        if self.status is InstagramResolutionStatus.FOUND_OFFICIAL:
            if self.resolved_url is None or self.username is None or self.source is None:
                raise ValueError("FOUND_OFFICIAL requires URL, username, and source")
            if self.error is not None:
                raise ValueError("FOUND_OFFICIAL cannot include error")
            if self.confidence <= 0.0:
                raise ValueError("FOUND_OFFICIAL requires positive confidence")
            if not any(
                item.candidate_url == self.resolved_url
                and item.username == self.username
                and item.rejected_reason is None
                for item in self.evidence
            ):
                raise ValueError("FOUND_OFFICIAL requires matching accepted evidence")
            return

        if self.resolved_url is not None or self.username is not None or self.source is not None:
            raise ValueError(f"{self.status.name} cannot include a resolved candidate")
        if self.status is InstagramResolutionStatus.RESOLUTION_ERROR:
            if self.error is None:
                raise ValueError("RESOLUTION_ERROR requires error")
            if self.confidence != 0.0:
                raise ValueError("RESOLUTION_ERROR requires zero confidence")
        elif self.error is not None:
            raise ValueError(f"{self.status.name} cannot include error")
        if self.status is InstagramResolutionStatus.NOT_FOUND and self.confidence != 0.0:
            raise ValueError("NOT_FOUND requires zero confidence")

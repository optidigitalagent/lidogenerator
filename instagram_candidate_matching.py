"""Pure Instagram profile normalization, search contracts, and identity matching."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Protocol, runtime_checkable
import unicodedata
from urllib.parse import urlsplit

from instagram_resolution import (
    InstagramCandidateEvidence,
    InstagramCandidateSource,
    InstagramIdentity,
    InstagramResolution,
    InstagramResolutionStatus,
)
from website_candidate_matching import (
    identity_tokens,
    normalize_identity_text,
    normalize_phone_number,
)
from website_resolution import normalize_candidate_url, normalize_domain


_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9._]{1,30}\Z")
_PHONE_LIKE = re.compile(r"(?<!\w)\+?\d(?:[\s().-]*\d){6,14}(?!\w)")
_RESERVED_PATHS = frozenset({
    "p", "reel", "reels", "explore", "stories", "accounts", "search",
    "direct", "developer", "about", "legal", "privacy", "challenge", "login", "tv",
})
_GENERIC_USERNAME_TOKENS = frozenset({
    "official", "kyiv", "kiev", "ukraine", "ua", "shop", "store", "salon",
    "studio", "beauty", "clinic", "cafe", "restaurant",
})


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _outer_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return " ".join(value.split())


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def normalize_instagram_profile_url(value: str) -> str:
    """Return ``https://www.instagram.com/<username>/`` for a direct profile URL."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    raw = value.strip()
    if not raw:
        raise ValueError("value must not be empty")
    if any(character.isspace() for character in raw):
        raise ValueError("Instagram URL must not contain whitespace")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise ValueError("value must be a valid Instagram profile URL") from exc
    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("Instagram URL scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Instagram URL credentials are not allowed")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Instagram URL port is invalid") from exc
    if port not in (None, 80, 443):
        raise ValueError("Instagram URL port is not allowed")
    host = (parsed.hostname or "").casefold().rstrip(".")
    if host not in {"instagram.com", "www.instagram.com"}:
        raise ValueError("Instagram profile URL must use instagram.com")

    path = parsed.path
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Instagram profile URL path is invalid")
    if path.endswith("//"):
        raise ValueError("Instagram profile URL path is ambiguous")
    body = path[1:-1] if path.endswith("/") else path[1:]
    if not body or "/" in body:
        raise ValueError("Instagram profile URL must contain one profile segment")
    if _USERNAME_PATTERN.fullmatch(body) is None:
        raise ValueError("Instagram username is invalid")
    if body.casefold() in _RESERVED_PATHS:
        raise ValueError("Instagram URL path is not a profile")
    return f"https://www.instagram.com/{body}/"


def normalize_instagram_username(value: str) -> str:
    """Return a case-folded valid Instagram username for comparisons."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    raw = value.strip()
    if not raw:
        raise ValueError("value must not be empty")
    if "://" in raw:
        path = urlsplit(normalize_instagram_profile_url(raw)).path
        username = path.strip("/")
    else:
        username = raw[1:] if raw.startswith("@") else raw
        if _USERNAME_PATTERN.fullmatch(username) is None:
            raise ValueError("Instagram username is invalid")
        if username.casefold() in _RESERVED_PATHS:
            raise ValueError("Instagram username is reserved")
    return username.casefold()


@dataclass(frozen=True)
class InstagramSearchIdentityEvidence:
    """Non-authoritative provider assertions used only as corroboration.

    ``candidate_url_source_bound`` means only that the candidate URL appeared in
    the trusted web-search source allowlist. Provider assertions are
    corroboration only; no assertion alone identifies the official account.
    """

    name_matches: bool
    city_matches: bool
    address_matches: bool
    phone_matches: bool
    website_domain_matches: bool
    different_city_detected: bool
    candidate_url_source_bound: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")


@dataclass(frozen=True)
class InstagramSearchResult:
    url: str
    title: str
    snippet: str
    rank: int
    identity_evidence: InstagramSearchIdentityEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", normalize_instagram_profile_url(self.url))
        object.__setattr__(self, "title", _outer_text(self.title, "title"))
        object.__setattr__(self, "snippet", _outer_text(self.snippet, "snippet"))
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer")
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if self.identity_evidence is not None and type(
            self.identity_evidence
        ) is not InstagramSearchIdentityEvidence:
            raise TypeError("identity_evidence has an invalid type")


@dataclass(frozen=True)
class InstagramSearchRequest:
    business_name: str
    city: str
    address: str | None = None
    phone: str | None = None
    website_url: str | None = None
    count: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "business_name", _required_text(self.business_name, "business_name")
        )
        object.__setattr__(self, "city", _required_text(self.city, "city"))
        object.__setattr__(self, "address", _optional_text(self.address, "address"))
        if self.phone is not None:
            object.__setattr__(self, "phone", normalize_phone_number(self.phone))
        if self.website_url is not None:
            object.__setattr__(
                self, "website_url", normalize_candidate_url(self.website_url)
            )
        if type(self.count) is not int:
            raise TypeError("count must be an integer")
        if not 1 <= self.count <= 10:
            raise ValueError("count must be between 1 and 10")


@runtime_checkable
class InstagramSearchProvider(Protocol):
    async def search(
        self, request: InstagramSearchRequest
    ) -> tuple[InstagramSearchResult, ...]:
        ...


class InstagramSearchProviderError(RuntimeError):
    """Base class for expected Instagram search failures."""


class InstagramProviderUnavailable(InstagramSearchProviderError):
    pass


class InstagramProviderTimeout(InstagramSearchProviderError):
    pass


class InstagramProviderAuthError(InstagramSearchProviderError):
    pass


class InstagramProviderRateLimited(InstagramSearchProviderError):
    pass


@dataclass(frozen=True)
class InstagramCandidate:
    source: InstagramCandidateSource
    url: str
    username: str
    title: str = ""
    snippet: str = ""
    identity_evidence: InstagramSearchIdentityEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, InstagramCandidateSource):
            raise TypeError("source must be an InstagramCandidateSource")
        normalized_url = normalize_instagram_profile_url(self.url)
        normalized_username = normalize_instagram_username(self.username)
        if normalize_instagram_username(normalized_url) != normalized_username:
            raise ValueError("username must match the profile URL")
        object.__setattr__(self, "url", normalized_url)
        object.__setattr__(self, "username", normalized_username)
        object.__setattr__(self, "title", _outer_text(self.title, "title"))
        object.__setattr__(self, "snippet", _outer_text(self.snippet, "snippet"))
        if self.identity_evidence is not None and type(
            self.identity_evidence
        ) is not InstagramSearchIdentityEvidence:
            raise TypeError("identity_evidence has an invalid type")


def candidate_from_instagram_search_result(
    result: InstagramSearchResult,
) -> InstagramCandidate:
    if not isinstance(result, InstagramSearchResult):
        raise TypeError("result must be an InstagramSearchResult")
    return InstagramCandidate(
        source=InstagramCandidateSource.WEB_SEARCH,
        url=result.url,
        username=normalize_instagram_username(result.url),
        title=result.title,
        snippet=result.snippet,
        identity_evidence=result.identity_evidence,
    )


class InstagramMatchSignal(str, Enum):
    PHONE_EXACT = "phone_exact"
    NAME_EXACT = "name_exact"
    NAME_TOKEN_OVERLAP = "name_token_overlap"
    CITY_EXACT = "city_exact"
    ADDRESS_TOKEN_OVERLAP = "address_token_overlap"
    USERNAME_NAME_OVERLAP = "username_name_overlap"
    WEBSITE_DOMAIN_MENTION = "website_domain_mention"
    SOURCE_ADDRESS_CORROBORATION = "source_address_corroboration"
    SOURCE_PHONE_CORROBORATION = "source_phone_corroboration"
    SOURCE_WEBSITE_CORROBORATION = "source_website_corroboration"


_SIGNAL_WEIGHTS = {
    InstagramMatchSignal.PHONE_EXACT: 0.75,
    InstagramMatchSignal.NAME_EXACT: 0.35,
    InstagramMatchSignal.NAME_TOKEN_OVERLAP: 0.25,
    InstagramMatchSignal.CITY_EXACT: 0.10,
    InstagramMatchSignal.ADDRESS_TOKEN_OVERLAP: 0.30,
    InstagramMatchSignal.USERNAME_NAME_OVERLAP: 0.25,
    InstagramMatchSignal.WEBSITE_DOMAIN_MENTION: 0.30,
    InstagramMatchSignal.SOURCE_ADDRESS_CORROBORATION: 0.30,
    InstagramMatchSignal.SOURCE_PHONE_CORROBORATION: 0.30,
    InstagramMatchSignal.SOURCE_WEBSITE_CORROBORATION: 0.30,
}


def _tokens_or_empty(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(normalize_identity_text(value).split())


def _contains_phrase(text: str, phrase: tuple[str, ...]) -> bool:
    tokens = _tokens_or_empty(text)
    width = len(phrase)
    return bool(phrase) and any(
        tokens[index:index + width] == phrase
        for index in range(len(tokens) - width + 1)
    )


def _overlap(reference: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    return len(set(reference).intersection(candidate)) / len(reference) if reference else 0.0


def _phone_keys(value: str) -> frozenset[str]:
    digits = normalize_phone_number(value)
    keys = {digits}
    if len(digits) == 12 and digits.startswith("380"):
        keys.add(f"0{digits[3:]}")
    elif len(digits) == 10 and digits.startswith("0"):
        keys.add(f"38{digits}")
    return frozenset(keys)


def _phones_equivalent(first: str, second: str) -> bool:
    return not _phone_keys(first).isdisjoint(_phone_keys(second))


def _observed_phones(candidate: InstagramCandidate) -> tuple[str, ...]:
    result: list[str] = []
    for text in (candidate.title, candidate.snippet):
        for match in _PHONE_LIKE.finditer(text):
            try:
                normalized = normalize_phone_number(match.group(0))
            except ValueError:
                continue
            if normalized not in result:
                result.append(normalized)
    return tuple(result)


def _username_matches_name(name: str, username: str) -> bool:
    business_tokens = tuple(
        token for token in identity_tokens(name)
        if len(token) >= 3 and token not in _GENERIC_USERNAME_TOKENS
    )
    username_parts = tuple(
        token for token in re.split(r"[._]+", username.casefold())
        if len(token) >= 3 and token not in _GENERIC_USERNAME_TOKENS
    )
    if set(business_tokens).intersection(username_parts):
        return True
    compact_username = "".join(character for character in username.casefold() if character.isalnum())
    compact_name = "".join(business_tokens)
    return bool(compact_name and len(compact_name) >= 4 and compact_name in compact_username)


def _website_domain_mentioned(identity: InstagramIdentity, text: str) -> bool:
    if identity.website_url is None:
        return False
    domain = normalize_domain(identity.website_url)
    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    pattern = rf"(?<![a-z0-9.-])(?:www\.)?{re.escape(domain)}(?![a-z0-9.-])"
    return re.search(pattern, normalized_text) is not None


def _candidate_evidence(
    candidate: InstagramCandidate,
    signals: tuple[InstagramMatchSignal, ...] = (),
    rejected_reason: str | None = None,
    confidence: float = 0.0,
) -> InstagramCandidateEvidence:
    return InstagramCandidateEvidence(
        source=candidate.source,
        candidate_url=candidate.url,
        username=candidate.username,
        matched_signals=tuple(signal.value for signal in signals),
        rejected_reason=rejected_reason,
        confidence=confidence,
    )


def assess_instagram_candidate(
    identity: InstagramIdentity,
    candidate: InstagramCandidate,
) -> InstagramCandidateEvidence:
    if type(identity) is not InstagramIdentity:
        raise TypeError("identity must be exactly an InstagramIdentity")
    if type(candidate) is not InstagramCandidate:
        raise TypeError("candidate must be exactly an InstagramCandidate")

    observed_phones = _observed_phones(candidate)
    if identity.phone is not None and any(
        not _phones_equivalent(identity.phone, phone) for phone in observed_phones
    ):
        return _candidate_evidence(candidate, rejected_reason="conflicting_phone")

    source = candidate.identity_evidence
    if source is not None and source.candidate_url_source_bound and (
        not source.name_matches
        or not source.city_matches
        or source.different_city_detected
    ):
        return _candidate_evidence(
            candidate, rejected_reason="conflicting_source_identity_evidence"
        )

    signals: list[InstagramMatchSignal] = []
    if identity.phone is not None and any(
        _phones_equivalent(identity.phone, phone) for phone in observed_phones
    ):
        signals.append(InstagramMatchSignal.PHONE_EXACT)

    combined_text = " ".join((candidate.title, candidate.snippet))
    name_phrase = tuple(normalize_identity_text(identity.business_name).split())
    name_tokens = identity_tokens(identity.business_name)
    candidate_tokens = _tokens_or_empty(combined_text)
    if _contains_phrase(combined_text, name_phrase):
        signals.append(InstagramMatchSignal.NAME_EXACT)
    elif len(name_tokens) >= 2 and _overlap(name_tokens, candidate_tokens) >= 0.75:
        signals.append(InstagramMatchSignal.NAME_TOKEN_OVERLAP)

    city_phrase = tuple(normalize_identity_text(identity.city).split())
    if _contains_phrase(combined_text, city_phrase):
        signals.append(InstagramMatchSignal.CITY_EXACT)

    if identity.address is not None:
        address_tokens = identity_tokens(identity.address)
        if len(address_tokens) >= 2 and _overlap(address_tokens, candidate_tokens) >= 0.60:
            signals.append(InstagramMatchSignal.ADDRESS_TOKEN_OVERLAP)

    if _username_matches_name(identity.business_name, candidate.username):
        signals.append(InstagramMatchSignal.USERNAME_NAME_OVERLAP)
    if _website_domain_mentioned(identity, combined_text):
        signals.append(InstagramMatchSignal.WEBSITE_DOMAIN_MENTION)

    source_allowed = (
        source is not None
        and source.candidate_url_source_bound
        and source.name_matches
        and source.city_matches
        and not source.different_city_detected
    )
    if source_allowed:
        if identity.address is not None and source.address_matches:
            signals.append(InstagramMatchSignal.SOURCE_ADDRESS_CORROBORATION)
        if identity.phone is not None and source.phone_matches:
            signals.append(InstagramMatchSignal.SOURCE_PHONE_CORROBORATION)
        if identity.website_url is not None and source.website_domain_matches:
            signals.append(InstagramMatchSignal.SOURCE_WEBSITE_CORROBORATION)

    signal_tuple = tuple(signals)
    confidence = min(1.0, round(sum(_SIGNAL_WEIGHTS[item] for item in signal_tuple), 2))
    has_name = any(item in signal_tuple for item in (
        InstagramMatchSignal.NAME_EXACT,
        InstagramMatchSignal.NAME_TOKEN_OVERLAP,
    ))
    has_corroboration = any(item in signal_tuple for item in (
        InstagramMatchSignal.ADDRESS_TOKEN_OVERLAP,
        InstagramMatchSignal.USERNAME_NAME_OVERLAP,
        InstagramMatchSignal.WEBSITE_DOMAIN_MENTION,
        InstagramMatchSignal.SOURCE_ADDRESS_CORROBORATION,
        InstagramMatchSignal.SOURCE_PHONE_CORROBORATION,
        InstagramMatchSignal.SOURCE_WEBSITE_CORROBORATION,
    ))
    accepted = (
        InstagramMatchSignal.PHONE_EXACT in signal_tuple
        or (has_name and has_corroboration and confidence >= 0.60)
    )
    return _candidate_evidence(
        candidate,
        signal_tuple,
        None if accepted else "insufficient_identity_evidence",
        confidence,
    )


def resolve_instagram_candidates(
    identity: InstagramIdentity,
    candidates: Iterable[InstagramCandidate],
) -> InstagramResolution:
    if type(identity) is not InstagramIdentity:
        raise TypeError("identity must be exactly an InstagramIdentity")
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Iterable):
        raise TypeError("candidates must be an iterable of InstagramCandidate items")

    unique: list[InstagramCandidate] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if type(candidate) is not InstagramCandidate:
            raise TypeError(f"candidates[{index}] must be an InstagramCandidate")
        if candidate.username not in seen:
            seen.add(candidate.username)
            unique.append(candidate)
    evidence = tuple(assess_instagram_candidate(identity, item) for item in unique)
    accepted = tuple(item for item in evidence if item.rejected_reason is None)

    if len(accepted) == 1:
        chosen = accepted[0]
        return InstagramResolution(
            InstagramResolutionStatus.FOUND_OFFICIAL,
            chosen.candidate_url,
            chosen.username,
            chosen.source,
            chosen.confidence,
            evidence,
        )
    if len(accepted) > 1:
        return InstagramResolution(
            InstagramResolutionStatus.UNCERTAIN,
            None,
            None,
            None,
            max(item.confidence for item in accepted),
            evidence,
        )
    if evidence:
        return InstagramResolution(
            InstagramResolutionStatus.UNCERTAIN,
            None,
            None,
            None,
            max(item.confidence for item in evidence),
            evidence,
        )
    return InstagramResolution(
        InstagramResolutionStatus.NOT_FOUND, None, None, None, 0.0, ()
    )


async def resolve_instagram_via_search(
    identity: InstagramIdentity,
    provider: InstagramSearchProvider,
) -> InstagramResolution:
    if type(identity) is not InstagramIdentity:
        raise TypeError("identity must be exactly an InstagramIdentity")
    if not isinstance(provider, InstagramSearchProvider):
        raise TypeError("provider must implement InstagramSearchProvider")
    request = InstagramSearchRequest(
        identity.business_name,
        identity.city,
        identity.address,
        identity.phone,
        identity.website_url,
    )
    try:
        results = await provider.search(request)
        if not isinstance(results, tuple) or not all(
            type(item) is InstagramSearchResult for item in results
        ):
            raise InstagramSearchProviderError("Instagram provider returned malformed output")
    except InstagramSearchProviderError as exc:
        detail = " ".join(str(exc).split()) or type(exc).__name__
        return InstagramResolution(
            InstagramResolutionStatus.RESOLUTION_ERROR,
            None,
            None,
            None,
            0.0,
            (),
            error=detail,
        )
    except Exception as exc:
        return InstagramResolution(
            InstagramResolutionStatus.RESOLUTION_ERROR,
            None,
            None,
            None,
            0.0,
            (),
            error=f"unexpected provider failure: {type(exc).__name__}",
        )
    return resolve_instagram_candidates(
        identity,
        tuple(candidate_from_instagram_search_result(item) for item in results),
    )

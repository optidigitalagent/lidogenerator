"""Pure identity matching and search contracts for website candidates.

This module validates business and candidate metadata, converts provider search
results, scores deterministic identity signals, and assembles complete source
attempts into a website-resolution result.  It does not perform network I/O,
follow redirects, assess website quality, persist data, or integrate with the
production pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Protocol, runtime_checkable
import unicodedata
from urllib.parse import urlsplit

from website_resolution import (
    CandidateEvidence,
    CandidateKind,
    CandidateSource,
    ResolutionStatus,
    WebsiteResolution,
    classify_candidate_url,
    normalize_candidate_url,
    normalize_domain,
)


_INSTAGRAM_RESERVED_PATHS: frozenset[str] = frozenset(
    {
        "p",
        "reel",
        "reels",
        "stories",
        "explore",
        "accounts",
        "direct",
        "tv",
    }
)
_INSTAGRAM_USERNAME = re.compile(r"[a-z0-9._]{1,30}\Z")
_PHONE_LIKE = re.compile(r"(?<!\w)\+?\d(?:[\s().-]*\d){6,14}(?!\w)")


def _normalize_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _normalize_optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _normalize_non_empty_string(value, name)


def _normalize_outer_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    return " ".join(value.split())


def _validate_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a non-string sequence")
    return value


def normalize_identity_text(value: str) -> str:
    """Return deterministic Unicode identity text without transliteration."""

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, not {type(value).__name__}")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if character.isalnum() or category.startswith("M"):
            characters.append(character)
        else:
            characters.append(" ")
    result = " ".join("".join(characters).split())
    if not result:
        raise ValueError("value must not be empty after identity normalization")
    return result


def identity_tokens(value: str) -> tuple[str, ...]:
    """Return unique identity tokens in first-occurrence order."""

    normalized = normalize_identity_text(value)
    result: list[str] = []
    seen: set[str] = set()
    for token in normalized.split():
        if len(token) >= 2 and token not in seen:
            seen.add(token)
            result.append(token)
    return tuple(result)


def normalize_phone_number(value: str) -> str:
    """Normalize a phone number to 7--15 digits without country guessing."""

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, not {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be empty")
    if "+" in stripped and not stripped.startswith("+"):
        raise ValueError("plus sign is only allowed at the start")
    if stripped.count("+") > 1:
        raise ValueError("phone number may contain at most one plus sign")
    if any(not (character.isdigit() or character in " +().-/") for character in stripped):
        raise ValueError("phone number contains invalid characters")
    digits = "".join(character for character in stripped if character.isdigit())
    if not 7 <= len(digits) <= 15:
        raise ValueError("phone number must contain between 7 and 15 digits")
    return digits


def _phone_equivalence_keys(value: str) -> frozenset[str]:
    digits = normalize_phone_number(value)
    keys = {digits}
    if len(digits) == 12 and digits.startswith("380"):
        keys.add(f"0{digits[3:]}")
    elif len(digits) == 10 and digits.startswith("0"):
        keys.add(f"38{digits}")
    return frozenset(keys)


def _phones_equivalent(first: str, second: str) -> bool:
    return not _phone_equivalence_keys(first).isdisjoint(
        _phone_equivalence_keys(second)
    )


def normalize_instagram_username(value: str) -> str:
    """Normalize a raw Instagram username, @handle, or profile URL."""

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, not {type(value).__name__}")
    raw = value.strip()
    if not raw:
        raise ValueError("value must not be empty")

    if "://" in raw:
        normalized_url = normalize_candidate_url(raw)
        domain = normalize_domain(normalized_url)
        if not (domain == "instagram.com" or domain.endswith(".instagram.com")):
            raise ValueError("Instagram profile URL must use instagram.com")
        path_segments = [segment for segment in urlsplit(normalized_url).path.split("/") if segment]
        if not path_segments:
            raise ValueError("Instagram profile URL must contain a username")
        username = path_segments[0].casefold()
    else:
        username = (raw[1:] if raw.startswith("@") else raw).casefold()

    if username in _INSTAGRAM_RESERVED_PATHS:
        raise ValueError("Instagram URL path is not a profile")
    if _INSTAGRAM_USERNAME.fullmatch(username) is None:
        raise ValueError("Instagram username is invalid")
    return username


@dataclass(frozen=True)
class BusinessIdentity:
    """Validated immutable identity used for conservative matching."""

    name: str
    city: str
    address: str | None = None
    phone: str | None = None
    instagram_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_non_empty_string(self.name, "name"))
        object.__setattr__(self, "city", _normalize_non_empty_string(self.city, "city"))
        object.__setattr__(self, "address", _normalize_optional_string(self.address, "address"))
        if self.phone is not None:
            object.__setattr__(self, "phone", normalize_phone_number(self.phone))
        if self.instagram_url is not None:
            normalized_url = normalize_candidate_url(self.instagram_url)
            normalize_instagram_username(normalized_url)
            object.__setattr__(self, "instagram_url", normalized_url)


@dataclass(frozen=True)
class SearchRequest:
    """Provider-independent request for business website search."""

    business_name: str
    city: str
    address: str | None = None
    phone: str | None = None
    max_results: int = 5
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "business_name",
            _normalize_non_empty_string(self.business_name, "business_name"),
        )
        object.__setattr__(self, "city", _normalize_non_empty_string(self.city, "city"))
        object.__setattr__(self, "address", _normalize_optional_string(self.address, "address"))
        if self.phone is not None:
            object.__setattr__(self, "phone", normalize_phone_number(self.phone))
        if type(self.max_results) is not int:
            raise TypeError("max_results must be an integer")
        if not 1 <= self.max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")
        if type(self.timeout_seconds) not in (int, float):
            raise TypeError("timeout_seconds must be an integer or float")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout):
            raise ValueError("timeout_seconds must be finite")
        if not 0.0 < timeout <= 30.0:
            raise ValueError("timeout_seconds must be greater than 0 and at most 30")
        object.__setattr__(self, "timeout_seconds", timeout)


@dataclass(frozen=True)
class SearchIdentityEvidence:
    """Non-authoritative provider assertions for identity corroboration.

    ``candidate_url_source_bound=True`` means only that the candidate URL
    appeared in the provider's trusted/search source allowlist. It does not
    mean that name, address, or phone assertions were independently extracted.
    The matcher may use this object only as corroboration, never as sole
    authority.
    """

    name_matches: bool
    city_matches: bool
    address_matches: bool
    phone_matches: bool
    different_city_detected: bool
    candidate_url_source_bound: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")


@dataclass(frozen=True)
class SearchResult:
    """One validated result returned by a search provider."""

    url: str
    title: str
    snippet: str
    rank: int
    identity_evidence: SearchIdentityEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", normalize_candidate_url(self.url))
        object.__setattr__(self, "title", _normalize_outer_string(self.title, "title"))
        object.__setattr__(self, "snippet", _normalize_outer_string(self.snippet, "snippet"))
        if type(self.rank) is not int:
            raise TypeError("rank must be an integer")
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if (
            self.identity_evidence is not None
            and type(self.identity_evidence) is not SearchIdentityEvidence
        ):
            raise TypeError(
                "identity_evidence must be exactly a SearchIdentityEvidence or None"
            )


@runtime_checkable
class SearchProvider(Protocol):
    """Structural contract for a future asynchronous search provider."""

    async def search(self, request: SearchRequest) -> tuple[SearchResult, ...]:
        ...


class SearchProviderError(RuntimeError):
    """Base class for expected provider failures."""


class ProviderUnavailable(SearchProviderError):
    """The configured provider cannot currently be used."""


class ProviderTimeout(SearchProviderError):
    """The provider did not complete within its time budget."""


class ProviderAuthError(SearchProviderError):
    """Provider credentials are absent or invalid."""


class ProviderRateLimited(SearchProviderError):
    """The provider rejected the request due to rate limiting."""


class SourceAttemptStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass(frozen=True)
class SourceAttempt:
    """Completion state for one candidate source, independent of quality."""

    source: CandidateSource
    status: SourceAttemptStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, CandidateSource):
            raise TypeError("source must be a CandidateSource")
        if not isinstance(self.status, SourceAttemptStatus):
            raise TypeError("status must be a SourceAttemptStatus")
        detail = _normalize_optional_string(self.detail, "detail")
        if self.status is SourceAttemptStatus.COMPLETED and detail is not None:
            raise ValueError("COMPLETED source attempt cannot include detail")
        if self.status is not SourceAttemptStatus.COMPLETED and detail is None:
            raise ValueError(f"{self.status.name} source attempt requires detail")
        object.__setattr__(self, "detail", detail)


@dataclass(frozen=True)
class WebsiteCandidate:
    """Validated candidate and metadata supplied by discovery layers."""

    source: CandidateSource
    url: str
    title: str = ""
    snippet: str = ""
    final_url: str | None = None
    city: str | None = None
    contact_phones: tuple[str, ...] = ()
    contact_addresses: tuple[str, ...] = ()
    instagram_usernames: tuple[str, ...] = ()
    identity_evidence: SearchIdentityEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, CandidateSource):
            raise TypeError("source must be a CandidateSource")
        object.__setattr__(self, "url", normalize_candidate_url(self.url))
        object.__setattr__(self, "title", _normalize_outer_string(self.title, "title"))
        object.__setattr__(self, "snippet", _normalize_outer_string(self.snippet, "snippet"))
        if self.final_url is not None:
            object.__setattr__(self, "final_url", normalize_candidate_url(self.final_url))
        if self.city is not None:
            object.__setattr__(self, "city", normalize_identity_text(self.city))

        if not isinstance(self.contact_phones, tuple):
            raise TypeError("contact_phones must be a tuple")
        phones: list[str] = []
        phone_keys: set[str] = set()
        for index, phone in enumerate(self.contact_phones):
            try:
                normalized_phone = normalize_phone_number(phone)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"contact_phones[{index}]: {exc}") from exc
            keys = _phone_equivalence_keys(normalized_phone)
            if not phone_keys.isdisjoint(keys):
                raise ValueError("contact_phones must not contain equivalent duplicates")
            phone_keys.update(keys)
            phones.append(normalized_phone)
        object.__setattr__(self, "contact_phones", tuple(phones))

        if not isinstance(self.contact_addresses, tuple):
            raise TypeError("contact_addresses must be a tuple")
        addresses: list[str] = []
        address_keys: set[str] = set()
        for index, address in enumerate(self.contact_addresses):
            normalized_address = _normalize_non_empty_string(
                address,
                f"contact_addresses[{index}]",
            )
            key = normalize_identity_text(normalized_address)
            if key in address_keys:
                raise ValueError("contact_addresses must not contain duplicates")
            address_keys.add(key)
            addresses.append(normalized_address)
        object.__setattr__(self, "contact_addresses", tuple(addresses))

        if not isinstance(self.instagram_usernames, tuple):
            raise TypeError("instagram_usernames must be a tuple")
        usernames: list[str] = []
        username_keys: set[str] = set()
        for index, username in enumerate(self.instagram_usernames):
            try:
                normalized_username = normalize_instagram_username(username)
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"instagram_usernames[{index}]: {exc}") from exc
            if normalized_username in username_keys:
                raise ValueError("instagram_usernames must not contain duplicates")
            username_keys.add(normalized_username)
            usernames.append(normalized_username)
        object.__setattr__(self, "instagram_usernames", tuple(usernames))

        if (
            self.identity_evidence is not None
            and type(self.identity_evidence) is not SearchIdentityEvidence
        ):
            raise TypeError(
                "identity_evidence must be exactly a SearchIdentityEvidence or None"
            )


def candidate_from_search_result(result: SearchResult) -> WebsiteCandidate:
    """Convert one validated provider result to a web-search candidate."""

    if not isinstance(result, SearchResult):
        raise TypeError("result must be a SearchResult")
    return WebsiteCandidate(
        source=CandidateSource.WEB_SEARCH,
        url=result.url,
        title=result.title,
        snippet=result.snippet,
        identity_evidence=result.identity_evidence,
    )


def _safe_error_detail(error: SearchProviderError) -> str:
    detail = " ".join(str(error).split())
    return detail or type(error).__name__


async def collect_web_search_candidates(
    provider: SearchProvider,
    request: SearchRequest,
) -> tuple[tuple[WebsiteCandidate, ...], SourceAttempt]:
    """Call a provider once and map expected failures to source state."""

    try:
        results = await provider.search(request)
    except (ProviderUnavailable, ProviderAuthError) as exc:
        return (), SourceAttempt(
            CandidateSource.WEB_SEARCH,
            SourceAttemptStatus.UNAVAILABLE,
            _safe_error_detail(exc),
        )
    except ProviderTimeout as exc:
        return (), SourceAttempt(
            CandidateSource.WEB_SEARCH,
            SourceAttemptStatus.TIMEOUT,
            _safe_error_detail(exc),
        )
    except ProviderRateLimited as exc:
        return (), SourceAttempt(
            CandidateSource.WEB_SEARCH,
            SourceAttemptStatus.RATE_LIMITED,
            _safe_error_detail(exc),
        )
    except SearchProviderError as exc:
        return (), SourceAttempt(
            CandidateSource.WEB_SEARCH,
            SourceAttemptStatus.ERROR,
            _safe_error_detail(exc),
        )

    if not isinstance(results, tuple):
        raise TypeError("SearchProvider.search() must return a tuple")
    indexed_results: list[tuple[int, SearchResult]] = []
    for index, result in enumerate(results):
        if not isinstance(result, SearchResult):
            raise TypeError(f"search results[{index}] must be a SearchResult")
        indexed_results.append((index, result))

    candidates: list[WebsiteCandidate] = []
    seen_urls: set[str] = set()
    for _, result in sorted(indexed_results, key=lambda item: (item[1].rank, item[0])):
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            candidates.append(candidate_from_search_result(result))
    return tuple(candidates), SourceAttempt(
        CandidateSource.WEB_SEARCH,
        SourceAttemptStatus.COMPLETED,
    )


class MatchSignal(str, Enum):
    PHONE_EXACT = "phone_exact"
    NAME_EXACT = "name_exact"
    NAME_TOKEN_OVERLAP = "name_token_overlap"
    CITY_EXACT = "city_exact"
    ADDRESS_TOKEN_OVERLAP = "address_token_overlap"
    SOURCE_ADDRESS_CORROBORATION = "source_address_corroboration"
    SOURCE_PHONE_CORROBORATION = "source_phone_corroboration"
    INSTAGRAM_USERNAME = "instagram_username"
    DOMAIN_NAME_OVERLAP = "domain_name_overlap"


_SIGNAL_WEIGHTS = (
    (MatchSignal.PHONE_EXACT, 0.75),
    (MatchSignal.NAME_EXACT, 0.35),
    (MatchSignal.NAME_TOKEN_OVERLAP, 0.25),
    (MatchSignal.CITY_EXACT, 0.10),
    (MatchSignal.ADDRESS_TOKEN_OVERLAP, 0.30),
    (MatchSignal.SOURCE_ADDRESS_CORROBORATION, 0.30),
    (MatchSignal.SOURCE_PHONE_CORROBORATION, 0.30),
    (MatchSignal.INSTAGRAM_USERNAME, 0.30),
    (MatchSignal.DOMAIN_NAME_OVERLAP, 0.10),
)


def _normalized_tokens_or_empty(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(normalize_identity_text(value).split())


def _contains_token_phrase(text: str, phrase: tuple[str, ...]) -> bool:
    if not text or not phrase:
        return False
    tokens = _normalized_tokens_or_empty(text)
    width = len(phrase)
    return any(tokens[index:index + width] == phrase for index in range(len(tokens) - width + 1))


def _candidate_phones(candidate: WebsiteCandidate) -> tuple[str, ...]:
    phones = list(candidate.contact_phones)
    seen = {phone for phone in phones}
    for text in (candidate.title, candidate.snippet):
        for match in _PHONE_LIKE.finditer(text):
            try:
                phone = normalize_phone_number(match.group(0))
            except ValueError:
                continue
            if phone not in seen:
                seen.add(phone)
                phones.append(phone)
    return tuple(phones)


def _token_overlap(reference: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    if not reference:
        return 0.0
    return len(set(reference).intersection(candidate)) / len(reference)


def _text_contains_instagram_username(text: str, username: str) -> bool:
    if not text:
        return False
    normalized = unicodedata.normalize("NFKC", text).casefold()
    pattern = rf"(?<![a-z0-9._])@?{re.escape(username)}(?![a-z0-9._])"
    return re.search(pattern, normalized) is not None


def _evidence(
    candidate: WebsiteCandidate,
    *,
    kind: CandidateKind,
    signals: tuple[MatchSignal, ...] = (),
    rejected_reason: str | None = None,
    confidence: float = 0.0,
) -> CandidateEvidence:
    normalized_url = normalize_candidate_url(candidate.url)
    return CandidateEvidence(
        source=candidate.source,
        candidate_url=normalized_url,
        normalized_url=normalized_url,
        normalized_domain=normalize_domain(normalized_url),
        final_domain=(
            normalize_domain(candidate.final_url)
            if candidate.final_url is not None
            else None
        ),
        kind=kind,
        matched_signals=tuple(signal.value for signal in signals),
        rejected_reason=rejected_reason,
        confidence=confidence,
        technical_error=None,
    )


def assess_website_candidate(
    identity: BusinessIdentity,
    candidate: WebsiteCandidate,
) -> CandidateEvidence:
    """Assess whether identity evidence supports one official website."""

    if not isinstance(identity, BusinessIdentity):
        raise TypeError("identity must be a BusinessIdentity")
    if not isinstance(candidate, WebsiteCandidate):
        raise TypeError("candidate must be a WebsiteCandidate")

    effective_url = candidate.final_url or candidate.url
    classified_kind = classify_candidate_url(effective_url)
    if classified_kind is not CandidateKind.UNKNOWN:
        return _evidence(
            candidate,
            kind=classified_kind,
            rejected_reason="non_official_platform",
        )

    candidate_phones = _candidate_phones(candidate)
    if (
        identity.phone is not None
        and candidate_phones
        and not any(_phones_equivalent(identity.phone, phone) for phone in candidate_phones)
    ):
        return _evidence(
            candidate,
            kind=CandidateKind.UNKNOWN,
            rejected_reason="conflicting_phone",
        )

    if (
        candidate.city is not None
        and normalize_identity_text(candidate.city) != normalize_identity_text(identity.city)
    ):
        return _evidence(
            candidate,
            kind=CandidateKind.UNKNOWN,
            rejected_reason="conflicting_city",
        )

    source_evidence = candidate.identity_evidence
    if (
        source_evidence is not None
        and source_evidence.candidate_url_source_bound
        and (
            not source_evidence.name_matches
            or not source_evidence.city_matches
            or source_evidence.different_city_detected
        )
    ):
        return _evidence(
            candidate,
            kind=CandidateKind.UNKNOWN,
            rejected_reason="conflicting_source_identity_evidence",
        )

    signals: list[MatchSignal] = []
    if identity.phone is not None and any(
        _phones_equivalent(identity.phone, phone) for phone in candidate_phones
    ):
        signals.append(MatchSignal.PHONE_EXACT)

    name_phrase = tuple(normalize_identity_text(identity.name).split())
    name_exact = any(
        _contains_token_phrase(text, name_phrase)
        for text in (candidate.title, candidate.snippet)
    )
    name_tokens = identity_tokens(identity.name)
    candidate_text_tokens = tuple(
        token
        for text in (candidate.title, candidate.snippet)
        for token in _normalized_tokens_or_empty(text)
    )
    if name_exact:
        signals.append(MatchSignal.NAME_EXACT)
    elif len(name_tokens) >= 2 and _token_overlap(name_tokens, candidate_text_tokens) >= 0.75:
        signals.append(MatchSignal.NAME_TOKEN_OVERLAP)

    city_phrase = tuple(normalize_identity_text(identity.city).split())
    if (
        candidate.city is not None
        or any(
            _contains_token_phrase(text, city_phrase)
            for text in (candidate.title, candidate.snippet)
        )
    ):
        signals.append(MatchSignal.CITY_EXACT)

    if identity.address is not None:
        address_tokens = identity_tokens(identity.address)
        if len(address_tokens) >= 2 and any(
            _token_overlap(address_tokens, identity_tokens(address)) >= 0.60
            for address in candidate.contact_addresses
        ):
            signals.append(MatchSignal.ADDRESS_TOKEN_OVERLAP)

    source_corroboration_allowed = (
        source_evidence is not None
        and source_evidence.candidate_url_source_bound
        and source_evidence.name_matches
        and source_evidence.city_matches
        and not source_evidence.different_city_detected
    )
    if source_corroboration_allowed:
        if identity.address is not None and source_evidence.address_matches:
            signals.append(MatchSignal.SOURCE_ADDRESS_CORROBORATION)
        if identity.phone is not None and source_evidence.phone_matches:
            signals.append(MatchSignal.SOURCE_PHONE_CORROBORATION)

    if identity.instagram_url is not None:
        username = normalize_instagram_username(identity.instagram_url)
        if username in candidate.instagram_usernames or any(
            _text_contains_instagram_username(text, username)
            for text in (candidate.title, candidate.snippet)
        ):
            signals.append(MatchSignal.INSTAGRAM_USERNAME)

    if len(name_tokens) >= 2:
        effective_domain = normalize_domain(effective_url)
        domain_tokens = identity_tokens(effective_domain)
        if _token_overlap(name_tokens, domain_tokens) >= 0.60:
            signals.append(MatchSignal.DOMAIN_NAME_OVERLAP)

    signal_tuple = tuple(signals)
    weight_by_signal = dict(_SIGNAL_WEIGHTS)
    confidence = min(
        1.0,
        round(sum(weight_by_signal[signal] for signal in signal_tuple), 2),
    )
    has_name = any(
        signal in signal_tuple
        for signal in (MatchSignal.NAME_EXACT, MatchSignal.NAME_TOKEN_OVERLAP)
    )
    has_corroboration = any(
        signal in signal_tuple
        for signal in (
            MatchSignal.ADDRESS_TOKEN_OVERLAP,
            MatchSignal.SOURCE_ADDRESS_CORROBORATION,
            MatchSignal.SOURCE_PHONE_CORROBORATION,
            MatchSignal.INSTAGRAM_USERNAME,
        )
    )
    accepted = (
        MatchSignal.PHONE_EXACT in signal_tuple
        or (has_name and has_corroboration and confidence >= 0.60)
    )
    return _evidence(
        candidate,
        kind=(CandidateKind.OFFICIAL_WEBSITE if accepted else CandidateKind.UNKNOWN),
        signals=signal_tuple,
        rejected_reason=None if accepted else "insufficient_identity_evidence",
        confidence=confidence,
    )


def resolve_website_candidates(
    identity: BusinessIdentity,
    candidates: Sequence[WebsiteCandidate],
    source_attempts: Sequence[SourceAttempt],
    required_sources: Sequence[CandidateSource],
) -> WebsiteResolution:
    """Assemble candidate evidence and source completion into one resolution."""

    if type(identity) is not BusinessIdentity:
        raise TypeError("identity must be exactly a BusinessIdentity")
    candidate_items = _validate_sequence(candidates, "candidates")
    attempt_items = _validate_sequence(source_attempts, "source_attempts")
    required_items = _validate_sequence(required_sources, "required_sources")

    validated_candidates: list[WebsiteCandidate] = []
    seen_candidate_keys: set[tuple[CandidateSource, str]] = set()
    for index, candidate in enumerate(candidate_items):
        if not isinstance(candidate, WebsiteCandidate):
            raise TypeError(f"candidates[{index}] must be a WebsiteCandidate")
        key = (candidate.source, candidate.url)
        if key not in seen_candidate_keys:
            seen_candidate_keys.add(key)
            validated_candidates.append(candidate)

    attempts_by_source: dict[CandidateSource, SourceAttempt] = {}
    for index, attempt in enumerate(attempt_items):
        if not isinstance(attempt, SourceAttempt):
            raise TypeError(f"source_attempts[{index}] must be a SourceAttempt")
        if attempt.source in attempts_by_source:
            raise ValueError("source_attempts must be unique by source")
        attempts_by_source[attempt.source] = attempt

    if not required_items:
        raise ValueError("required_sources must not be empty")
    required: list[CandidateSource] = []
    seen_required: set[CandidateSource] = set()
    for index, source in enumerate(required_items):
        if not isinstance(source, CandidateSource):
            raise TypeError(f"required_sources[{index}] must be a CandidateSource")
        if source in seen_required:
            raise ValueError("required_sources must be unique")
        seen_required.add(source)
        required.append(source)

    evidence = tuple(
        assess_website_candidate(identity, candidate)
        for candidate in validated_candidates
    )
    accepted = tuple(
        (index, item)
        for index, item in enumerate(evidence)
        if item.kind is CandidateKind.OFFICIAL_WEBSITE
    )
    domains: dict[str, list[tuple[int, CandidateEvidence]]] = {}
    for indexed_evidence in accepted:
        item = indexed_evidence[1]
        effective_domain = item.final_domain or item.normalized_domain
        domains.setdefault(effective_domain, []).append(indexed_evidence)

    if len(domains) == 1:
        source_priority = {
            CandidateSource.MAPS: 0,
            CandidateSource.INSTAGRAM_BIO: 1,
            CandidateSource.WEB_SEARCH: 2,
        }
        _, chosen = min(
            accepted,
            key=lambda indexed: (
                -indexed[1].confidence,
                source_priority[indexed[1].source],
                indexed[0],
            ),
        )
        return WebsiteResolution(
            ResolutionStatus.FOUND_OFFICIAL,
            chosen.normalized_url,
            chosen.source,
            chosen.confidence,
            evidence,
        )

    if len(domains) > 1:
        return WebsiteResolution(
            ResolutionStatus.UNCERTAIN,
            None,
            None,
            max(item.confidence for _, item in accepted),
            evidence,
        )

    technical_statuses = {
        SourceAttemptStatus.TIMEOUT,
        SourceAttemptStatus.RATE_LIMITED,
        SourceAttemptStatus.ERROR,
    }
    failed_required = [
        attempts_by_source[source]
        for source in required
        if source in attempts_by_source
        and attempts_by_source[source].status in technical_statuses
    ]
    if failed_required:
        error = "; ".join(
            f"{attempt.source.value}: {attempt.detail}"
            for attempt in failed_required
        )
        return WebsiteResolution(
            ResolutionStatus.RESOLUTION_ERROR,
            None,
            None,
            0.0,
            evidence,
            error=error,
        )

    incomplete_statuses = {
        SourceAttemptStatus.SKIPPED,
        SourceAttemptStatus.UNAVAILABLE,
    }
    if any(
        source not in attempts_by_source
        or attempts_by_source[source].status in incomplete_statuses
        for source in required
    ):
        return WebsiteResolution(
            ResolutionStatus.UNCERTAIN,
            None,
            None,
            max((item.confidence for item in evidence), default=0.0),
            evidence,
        )

    if any(item.kind is CandidateKind.UNKNOWN for item in evidence):
        return WebsiteResolution(
            ResolutionStatus.UNCERTAIN,
            None,
            None,
            max((item.confidence for item in evidence), default=0.0),
            evidence,
        )

    if any(
        item.kind in (CandidateKind.SOCIAL_PROFILE, CandidateKind.LINK_IN_BIO)
        for item in evidence
    ):
        return WebsiteResolution(
            ResolutionStatus.SOCIAL_ONLY,
            None,
            None,
            max((item.confidence for item in evidence), default=0.0),
            evidence,
        )

    return WebsiteResolution(
        ResolutionStatus.NOT_FOUND,
        None,
        None,
        0.0,
        evidence,
    )

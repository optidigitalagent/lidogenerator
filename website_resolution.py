"""Contracts and deterministic URL/domain helpers for website resolution.

This module does not perform web search or decide that a candidate is an
official website without separate identity-matching evidence. Website visual
quality and availability auditing are also outside this module's scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import ipaddress
import math
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class CandidateSource(str, Enum):
    """Stable sources from which a website candidate can be discovered."""

    MAPS = "maps"
    INSTAGRAM_BIO = "instagram_bio"
    WEB_SEARCH = "web_search"


class CandidateKind(str, Enum):
    """Stable, evidence-independent categories for candidate URLs."""

    OFFICIAL_WEBSITE = "official_website"
    SOCIAL_PROFILE = "social_profile"
    LINK_IN_BIO = "link_in_bio"
    MARKETPLACE_OR_AGGREGATOR = "marketplace_or_aggregator"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"


class ResolutionStatus(str, Enum):
    """Stable outcomes for a complete website-resolution attempt."""

    FOUND_OFFICIAL = "found_official"
    SOCIAL_ONLY = "social_only"
    NOT_FOUND = "not_found"
    UNCERTAIN = "uncertain"
    RESOLUTION_ERROR = "resolution_error"


_TRACKING_PARAMETERS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "dclid",
        "fbclid",
        "yclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
    }
)

_PRESENTATION_PREFIXES: tuple[str, ...] = ("www.", "m.", "mobile.")
_ASCII_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")

SOCIAL_PROFILE_DOMAINS: frozenset[str] = frozenset(
    {
        "instagram.com",
        "facebook.com",
        "fb.com",
        "tiktok.com",
        "x.com",
        "twitter.com",
        "linkedin.com",
        "youtube.com",
        "t.me",
        "telegram.me",
        "vk.com",
        "wa.me",
        "api.whatsapp.com",
        "viber.com",
        "invite.viber.com",
    }
)

LINK_IN_BIO_DOMAINS: frozenset[str] = frozenset(
    {
        "linktr.ee",
        "taplink.cc",
        "taplink.ws",
        "bio.link",
        "lnk.bio",
        "linkin.bio",
        "beacons.ai",
        "mssg.me",
    }
)

MARKETPLACE_OR_AGGREGATOR_DOMAINS: frozenset[str] = frozenset(
    {
        "booksy.com",
        "booksy.com.ua",
        "treatwell.com",
        "fresha.com",
        "prom.ua",
        "olx.ua",
        "rozetka.com.ua",
        "n716.alteg.io",
        "alteg.io",
        "easyweek.com.ua",
        "dikidi.net",
        "business.site",
        "google.com",
        "goo.gl",
        "maps.app.goo.gl",
    }
)

DIRECTORY_DOMAINS: frozenset[str] = frozenset(
    {
        "yellowpages.ua",
        "locator.ua",
        "list.in.ua",
    }
)


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


def _normalize_confidence(value: object, name: str = "confidence") -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an integer or float, not {type(value).__name__}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return normalized


def _normalize_host(value: str, name: str = "hostname") -> tuple[str, bool]:
    host = value.casefold().rstrip(".")
    if not host:
        raise ValueError(f"{name} must not be empty")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None

    if address is not None:
        return address.compressed.casefold(), True

    try:
        ascii_host = host.encode("idna").decode("ascii").casefold()
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid IDNA hostname") from exc

    if len(ascii_host) > 253:
        raise ValueError(f"{name} is too long")
    labels = ascii_host.split(".")
    if any(_ASCII_HOST_LABEL.fullmatch(label) is None for label in labels):
        raise ValueError(f"{name} is not a valid hostname")
    return ascii_host, False


def _reject_unsafe_host(host: str, is_ip: bool) -> None:
    if is_ip:
        address = ipaddress.ip_address(host)
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
            or not address.is_global
        ):
            raise ValueError("hostname must be a public IP address")
        return

    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host == "local"
        or host.endswith(".local")
    ):
        raise ValueError("hostname must not be local")


def _normalized_public_host(value: str) -> tuple[str, bool]:
    host, is_ip = _normalize_host(value)
    _reject_unsafe_host(host, is_ip)
    return host, is_ip


def normalize_candidate_url(value: str) -> str:
    """Return a deterministic, public absolute HTTP(S) candidate URL.

    This pure validation rejects literal unsafe hosts only. The future network
    layer must check DNS results before every request and redirect to prevent
    private resolution and DNS-rebinding attacks.
    """

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, not {type(value).__name__}")
    normalized_input = value.strip()
    if not normalized_input:
        raise ValueError("value must not be empty")
    if any(character.isspace() for character in normalized_input):
        raise ValueError("URL must not contain whitespace")

    try:
        parsed = urlsplit(normalized_input)
    except ValueError as exc:
        raise ValueError("value must be a valid absolute URL") from exc

    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("URL scheme must be http or https")
    if not parsed.netloc or parsed.hostname is None:
        raise ValueError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")
    if parsed.netloc.endswith(":"):
        raise ValueError("URL port is invalid")

    host, is_ip = _normalized_public_host(parsed.hostname)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL port is invalid") from exc

    display_host = f"[{host}]" if is_ip and ":" in host else host
    default_port = 80 if scheme == "http" else 443
    netloc = display_host if port is None or port == default_port else f"{display_host}:{port}"

    path = parsed.path or "/"
    query_pairs = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in _TRACKING_PARAMETERS
        and not key.casefold().startswith("utm_")
    ]
    query_pairs.sort(key=lambda pair: (pair[0].casefold(), pair[0], pair[1]))
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def normalize_domain(value: str) -> str:
    """Return a normalized hostname from an HTTP(S) URL or bare hostname.

    Common presentation prefixes are removed, but the result is not claimed to
    be a registrable domain or eTLD+1.
    """

    if not isinstance(value, str):
        raise TypeError(f"value must be a string, not {type(value).__name__}")
    normalized_input = value.strip()
    if not normalized_input:
        raise ValueError("value must not be empty")
    if any(character.isspace() for character in normalized_input):
        raise ValueError("domain must not contain whitespace")

    if "://" in normalized_input:
        normalized_url = normalize_candidate_url(normalized_input)
        host = urlsplit(normalized_url).hostname
        if host is None:  # pragma: no cover - guaranteed by URL normalization
            raise ValueError("URL must include a hostname")
    else:
        try:
            parsed = urlsplit(f"//{normalized_input}")
        except ValueError as exc:
            raise ValueError("value must be a valid hostname") from exc
        if (
            parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or normalized_input.endswith(":")
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("value must be a bare hostname")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("domain port is invalid") from exc
        host = parsed.hostname

    normalized_host, is_ip = _normalized_public_host(host)
    if is_ip:
        return normalized_host

    while True:
        for prefix in _PRESENTATION_PREFIXES:
            if normalized_host.startswith(prefix):
                normalized_host = normalized_host[len(prefix):]
                break
        else:
            break
    if not normalized_host:
        raise ValueError("domain must not be empty after prefix normalization")
    return normalized_host


def candidate_url_key(value: str) -> str:
    """Return the normalized URL key used for candidate deduplication."""

    return normalize_candidate_url(value)


def deduplicate_candidate_urls(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize candidate URLs and return first occurrences in input order."""

    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(
            "values must be a sequence of strings, "
            f"not {type(values).__name__}"
        )

    normalized_values: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"values[{index}] must be a string, not {type(value).__name__}"
            )
        normalized = candidate_url_key(value)
        if normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return tuple(normalized_values)


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def classify_candidate_url(value: str) -> CandidateKind:
    """Classify an obvious platform URL without asserting official identity."""

    normalized_url = normalize_candidate_url(value)
    host = urlsplit(normalized_url).hostname
    if host is None:  # pragma: no cover - guaranteed by URL normalization
        raise ValueError("URL must include a hostname")
    host, _ = _normalize_host(host)

    classifications = (
        (CandidateKind.SOCIAL_PROFILE, SOCIAL_PROFILE_DOMAINS),
        (CandidateKind.LINK_IN_BIO, LINK_IN_BIO_DOMAINS),
        (
            CandidateKind.MARKETPLACE_OR_AGGREGATOR,
            MARKETPLACE_OR_AGGREGATOR_DOMAINS,
        ),
        (CandidateKind.DIRECTORY, DIRECTORY_DOMAINS),
    )
    for kind, domains in classifications:
        if any(_domain_matches(host, domain) for domain in domains):
            return kind
    return CandidateKind.UNKNOWN


@dataclass(frozen=True)
class CandidateEvidence:
    """Validated evidence for one candidate considered by the resolver."""

    source: CandidateSource
    candidate_url: str
    normalized_url: str
    normalized_domain: str
    final_domain: str | None
    kind: CandidateKind
    matched_signals: tuple[str, ...] = ()
    rejected_reason: str | None = None
    confidence: float = 0.0
    technical_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, CandidateSource):
            raise TypeError("source must be a CandidateSource")
        if not isinstance(self.kind, CandidateKind):
            raise TypeError("kind must be a CandidateKind")

        object.__setattr__(
            self,
            "candidate_url",
            _normalize_non_empty_string(self.candidate_url, "candidate_url"),
        )

        normalized_url = normalize_candidate_url(self.normalized_url)
        if self.normalized_url != normalized_url:
            raise ValueError("normalized_url must already be normalized")

        expected_domain = normalize_domain(normalized_url)
        if not isinstance(self.normalized_domain, str):
            raise TypeError(
                "normalized_domain must be a string, "
                f"not {type(self.normalized_domain).__name__}"
            )
        if self.normalized_domain != expected_domain:
            raise ValueError(
                "normalized_domain must match the normalized_url domain"
            )

        if self.final_domain is not None:
            if not isinstance(self.final_domain, str):
                raise TypeError(
                    "final_domain must be a string or None, "
                    f"not {type(self.final_domain).__name__}"
                )
            normalized_final_domain = normalize_domain(self.final_domain)
            if self.final_domain != normalized_final_domain:
                raise ValueError("final_domain must already be normalized")

        if not isinstance(self.matched_signals, tuple):
            raise TypeError("matched_signals must be a tuple")
        signals: list[str] = []
        signal_keys: set[str] = set()
        for index, signal in enumerate(self.matched_signals):
            normalized_signal = _normalize_non_empty_string(
                signal,
                f"matched_signals[{index}]",
            )
            signal_key = normalized_signal.casefold()
            if signal_key in signal_keys:
                raise ValueError("matched_signals must not contain duplicates")
            signal_keys.add(signal_key)
            signals.append(normalized_signal)
        object.__setattr__(self, "matched_signals", tuple(signals))

        object.__setattr__(
            self,
            "rejected_reason",
            _normalize_optional_string(self.rejected_reason, "rejected_reason"),
        )
        object.__setattr__(
            self,
            "technical_error",
            _normalize_optional_string(self.technical_error, "technical_error"),
        )
        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )


@dataclass(frozen=True)
class WebsiteResolution:
    """Validated final status and evidence for one resolution attempt."""

    status: ResolutionStatus
    resolved_url: str | None
    source: CandidateSource | None
    confidence: float
    evidence: tuple[CandidateEvidence, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResolutionStatus):
            raise TypeError("status must be a ResolutionStatus")

        if self.resolved_url is not None:
            if not isinstance(self.resolved_url, str):
                raise TypeError("resolved_url must be a string or None")
            normalized_url = normalize_candidate_url(self.resolved_url)
            if self.resolved_url != normalized_url:
                raise ValueError("resolved_url must already be normalized")

        if self.source is not None and not isinstance(self.source, CandidateSource):
            raise TypeError("source must be a CandidateSource or None")

        object.__setattr__(
            self,
            "confidence",
            _normalize_confidence(self.confidence),
        )

        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")
        for index, item in enumerate(self.evidence):
            if not isinstance(item, CandidateEvidence):
                raise TypeError(
                    f"evidence[{index}] must be a CandidateEvidence, "
                    f"not {type(item).__name__}"
                )

        object.__setattr__(
            self,
            "error",
            _normalize_optional_string(self.error, "error"),
        )
        self._validate_status_invariants()

    def _validate_status_invariants(self) -> None:
        official_evidence = tuple(
            item
            for item in self.evidence
            if item.kind is CandidateKind.OFFICIAL_WEBSITE
        )

        if self.status is ResolutionStatus.FOUND_OFFICIAL:
            if self.resolved_url is None:
                raise ValueError("FOUND_OFFICIAL requires resolved_url")
            if self.source is None:
                raise ValueError("FOUND_OFFICIAL requires source")
            if self.confidence <= 0.0:
                raise ValueError("FOUND_OFFICIAL requires positive confidence")
            if self.error is not None:
                raise ValueError("FOUND_OFFICIAL cannot include error")
            if not self.evidence:
                raise ValueError("FOUND_OFFICIAL requires evidence")
            if not any(
                item.normalized_url == self.resolved_url
                for item in official_evidence
            ):
                raise ValueError(
                    "FOUND_OFFICIAL requires matching official evidence"
                )
            return

        if self.status is ResolutionStatus.SOCIAL_ONLY:
            if self.resolved_url is not None or self.source is not None:
                raise ValueError("SOCIAL_ONLY cannot resolve an official URL")
            if self.error is not None:
                raise ValueError("SOCIAL_ONLY cannot include error")
            if not self.evidence:
                raise ValueError("SOCIAL_ONLY requires evidence")
            if official_evidence:
                raise ValueError("SOCIAL_ONLY cannot include official evidence")
            social_kinds = {
                CandidateKind.SOCIAL_PROFILE,
                CandidateKind.LINK_IN_BIO,
            }
            if not any(item.kind in social_kinds for item in self.evidence):
                raise ValueError("SOCIAL_ONLY requires social evidence")
            return

        if self.status is ResolutionStatus.NOT_FOUND:
            if self.resolved_url is not None or self.source is not None:
                raise ValueError("NOT_FOUND cannot include a resolved URL or source")
            if self.error is not None:
                raise ValueError("NOT_FOUND cannot include error")
            if self.confidence != 0.0:
                raise ValueError("NOT_FOUND requires zero confidence")
            if official_evidence:
                raise ValueError("NOT_FOUND cannot include official evidence")
            return

        if self.status is ResolutionStatus.RESOLUTION_ERROR:
            if self.resolved_url is not None or self.source is not None:
                raise ValueError(
                    "RESOLUTION_ERROR cannot include a resolved URL or source"
                )
            if self.error is None:
                raise ValueError("RESOLUTION_ERROR requires error")
            if self.confidence != 0.0:
                raise ValueError("RESOLUTION_ERROR requires zero confidence")

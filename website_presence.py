"""Fail-closed website-presence contracts used by final lead qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from urllib.parse import urlsplit

from website_resolution import CandidateKind, classify_candidate_url, normalize_candidate_url


class WebsitePresenceStatus(str, Enum):
    PRESENT = "present"
    ABSENT_CONFIRMED = "absent_confirmed"
    UNCERTAIN = "uncertain"
    TECHNICAL_ERROR = "technical_error"
    SKIPPED = "skipped"


class WebsitePresenceSource(str, Enum):
    MAPS = "maps"
    WEB_SEARCH = "web_search"


_SAFE_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}\Z")


@dataclass(frozen=True)
class WebsitePresenceResult:
    status: WebsitePresenceStatus
    source: WebsitePresenceSource | None = None
    resolved_url: str | None = None
    evidence: tuple[str, ...] = ()
    error_category: str | None = None
    requests_used: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.status, WebsitePresenceStatus):
            raise TypeError("status must be a WebsitePresenceStatus")
        if self.source is not None and not isinstance(self.source, WebsitePresenceSource):
            raise TypeError("source must be a WebsitePresenceSource or None")
        if self.resolved_url is not None:
            normalized = normalize_candidate_url(self.resolved_url)
            object.__setattr__(self, "resolved_url", normalized)
        if not isinstance(self.evidence, tuple) or len(self.evidence) > 12:
            raise ValueError("evidence must be a tuple of at most 12 safe tokens")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("evidence must not contain duplicates")
        if any(not isinstance(item, str) or _SAFE_TOKEN.fullmatch(item) is None for item in self.evidence):
            raise ValueError("evidence must contain bounded safe tokens only")
        if self.error_category is not None and (
            not isinstance(self.error_category, str)
            or _SAFE_TOKEN.fullmatch(self.error_category) is None
        ):
            raise ValueError("error_category must be a bounded safe token")
        if type(self.requests_used) is not int or self.requests_used < 0:
            raise ValueError("requests_used must be a non-negative integer")

        if self.status is WebsitePresenceStatus.PRESENT:
            if self.source is None or self.resolved_url is None:
                raise ValueError("PRESENT requires source and resolved_url")
            if self.error_category is not None:
                raise ValueError("PRESENT cannot include an error")
        elif self.status is WebsitePresenceStatus.ABSENT_CONFIRMED:
            if self.source is not WebsitePresenceSource.WEB_SEARCH:
                raise ValueError("ABSENT_CONFIRMED requires WEB_SEARCH source")
            if self.resolved_url is not None or self.error_category is not None:
                raise ValueError("ABSENT_CONFIRMED cannot include URL or error")
        elif self.status is WebsitePresenceStatus.TECHNICAL_ERROR:
            if self.resolved_url is not None or self.error_category is None:
                raise ValueError("TECHNICAL_ERROR requires only a safe error category")
        elif self.resolved_url is not None:
            raise ValueError(f"{self.status.name} cannot include resolved_url")


HOSTED_SITE_BUILDER_DOMAINS = frozenset(
    {
        "sites.google.com",
        "business.site",
        "wixsite.com",
        "webflow.io",
        "tilda.ws",
        "weblium.site",
        "site123.me",
        "weebly.com",
        "wordpress.com",
        "blogspot.com",
        "godaddysites.com",
        "strikingly.com",
        "carrd.co",
        "notion.site",
    }
)


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def is_hosted_site_builder_url(url: str) -> bool:
    try:
        normalized = normalize_candidate_url(url)
    except (TypeError, ValueError):
        return False
    host = (urlsplit(normalized).hostname or "").casefold().rstrip(".")
    return any(_domain_matches(host, domain) for domain in HOSTED_SITE_BUILDER_DOMAINS)


def classify_website_presence_url(url: str) -> bool:
    """Return whether a valid public URL itself represents website presence.

    Site builders are deliberately checked before the legacy platform classifier:
    ``business.site`` is an aggregator there, but is website presence here.
    """

    try:
        normalized = normalize_candidate_url(url)
    except (TypeError, ValueError):
        return False
    if is_hosted_site_builder_url(normalized):
        return True
    return classify_candidate_url(normalized) is CandidateKind.UNKNOWN

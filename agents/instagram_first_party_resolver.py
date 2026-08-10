"""Safe first-party website fetching for deterministic Instagram resolution."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
import ipaddress
import socket
from typing import TypeAlias
from urllib.parse import urljoin, urlsplit

import httpx

from instagram_first_party_resolution import (
    FirstPartyInstagramResolution,
    FirstPartyInstagramStatus,
    extract_instagram_profiles_from_html,
    resolution_from_extracted_profiles,
)
from models import Business
from website_resolution import (
    CandidateKind,
    classify_candidate_url,
    normalize_candidate_url,
)


_CONTACT_TOKENS = frozenset(
    {
        "contact",
        "contacts",
        "kontakt",
        "kontakty",
        "contact-us",
        "about",
        "about-us",
        "o-nas",
        "pro-nas",
        "kontakti",
        "контакти",
        "контакты",
        "про-нас",
        "о-нас",
    }
)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_SITE_KINDS = frozenset(
    {CandidateKind.OFFICIAL_WEBSITE, CandidateKind.UNKNOWN}
)


class _BudgetExhausted(RuntimeError):
    def __init__(self, page_started: bool = False) -> None:
        super().__init__("request budget exhausted")
        self.page_started = page_started


class _FetchError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


@dataclass(frozen=True)
class FirstPartyInstagramSettings:
    max_pages_per_business: int = 2
    timeout_seconds: float = 8.0
    max_response_bytes: int = 1_048_576
    concurrency: int = 4
    max_redirects: int = 3

    def __post_init__(self) -> None:
        integer_fields = {
            "max_pages_per_business": (self.max_pages_per_business, 1, 2),
            "max_response_bytes": (self.max_response_bytes, 1, 16_777_216),
            "concurrency": (self.concurrency, 1, 32),
            "max_redirects": (self.max_redirects, 0, 3),
        }
        for name, (value, minimum, maximum) in integer_fields.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if type(self.timeout_seconds) not in (int, float) or not (
            0.0 < float(self.timeout_seconds) <= 30.0
        ):
            raise ValueError("timeout_seconds must be greater than 0 and at most 30")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True)
class FirstPartyInstagramBudgetSnapshot:
    max_requests: int
    used_requests: int
    remaining_requests: int


class FirstPartyInstagramRequestBudget:
    """Concurrency-safe task-global request budget."""

    def __init__(self, max_requests: int) -> None:
        if type(max_requests) is not int or max_requests < 0:
            raise ValueError("max_requests must be a non-negative integer")
        self._max_requests = max_requests
        self._used_requests = 0
        self._lock = asyncio.Lock()

    async def claim(self) -> None:
        async with self._lock:
            if self._used_requests >= self._max_requests:
                raise _BudgetExhausted()
            self._used_requests += 1

    def snapshot(self) -> FirstPartyInstagramBudgetSnapshot:
        return FirstPartyInstagramBudgetSnapshot(
            self._max_requests,
            self._used_requests,
            self._max_requests - self._used_requests,
        )


DNSResult: TypeAlias = Sequence[str]
DNSResolver: TypeAlias = Callable[[str, int], Awaitable[DNSResult]]


async def _default_dns_resolver(hostname: str, port: int) -> tuple[str, ...]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        hostname,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
    )
    return tuple(record[4][0] for record in records)


def _is_public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or not address.is_global
    )


def _host_key(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").casefold().rstrip(".")
    return hostname.removeprefix("www.")


def _same_website_host(first: str, second: str) -> bool:
    return bool(_host_key(first)) and _host_key(first) == _host_key(second)


def _same_origin(first: str, second: str) -> bool:
    first_parts = urlsplit(first)
    second_parts = urlsplit(second)
    return (
        first_parts.scheme.casefold(),
        (first_parts.hostname or "").casefold().rstrip("."),
        first_parts.port,
    ) == (
        second_parts.scheme.casefold(),
        (second_parts.hostname or "").casefold().rstrip("."),
        second_parts.port,
    )


def _valid_own_site_url(value: str) -> str | None:
    try:
        normalized = normalize_candidate_url(value)
        if classify_candidate_url(normalized) not in _ALLOWED_SITE_KINDS:
            return None
    except (TypeError, ValueError):
        return None
    return normalized


def trusted_website_for_instagram_resolution(business: Business) -> str | None:
    """Return a confirmed own-site URL without performing identity inference."""

    if not isinstance(business, Business):
        raise TypeError("business must be a Business")

    if business.website_resolution_status == "found_official":
        resolved = _valid_own_site_url(business.website_resolved_url)
        if resolved is not None:
            return resolved

    if business.has_site and business.site_quality in {"good", "bad"}:
        for candidate in (
            business.website_final_url,
            business.website,
            business.effective_website_url,
        ):
            resolved = _valid_own_site_url(candidate)
            if resolved is not None:
                return resolved
    return None


class _ContactLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.casefold() != "a":
            return
        attributes = {name.casefold(): value for name, value in attrs}
        href = attributes.get("href")
        if href:
            self.hrefs.append(href)


def _contact_page_from_html(html: str, page_url: str) -> str | None:
    parser = _ContactLinkParser()
    parser.feed(html)
    parser.close()
    for href in parser.hrefs:
        try:
            candidate = normalize_candidate_url(urljoin(page_url, href))
        except (TypeError, ValueError):
            continue
        if not _same_origin(page_url, candidate):
            continue
        segments = {
            segment.casefold()
            for segment in urlsplit(candidate).path.strip("/").split("/")
            if segment
        }
        if segments.intersection(_CONTACT_TOKENS):
            return candidate
    return None


@dataclass(frozen=True)
class _FetchedPage:
    html: str
    final_url: str


class FirstPartyInstagramResolver:
    def __init__(
        self,
        settings: FirstPartyInstagramSettings,
        budget: FirstPartyInstagramRequestBudget,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        dns_resolver: DNSResolver | None = None,
    ) -> None:
        if not isinstance(settings, FirstPartyInstagramSettings):
            raise TypeError("settings must be FirstPartyInstagramSettings")
        if not isinstance(budget, FirstPartyInstagramRequestBudget):
            raise TypeError("budget must be FirstPartyInstagramRequestBudget")
        self.settings = settings
        self.budget = budget
        self._transport = transport
        self._dns_resolver = dns_resolver or _default_dns_resolver

    async def _validate_dns(self, url: str) -> None:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        if hostname is None:
            raise _FetchError("invalid_url")
        port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
        try:
            addresses = await self._dns_resolver(hostname, port)
        except Exception as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise _FetchError("dns_error") from None
        if not addresses:
            raise _FetchError("dns_error")
        if any(not _is_public_address(address) for address in addresses):
            raise _FetchError("unsafe_address")

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        initial_url: str,
    ) -> _FetchedPage:
        current_url = normalize_candidate_url(initial_url)
        redirects = 0
        page_started = False
        while True:
            await self._validate_dns(current_url)
            try:
                await self.budget.claim()
            except _BudgetExhausted:
                raise _BudgetExhausted(page_started) from None
            page_started = True
            try:
                async with client.stream("GET", current_url) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise _FetchError("redirect_missing_location")
                        if redirects >= self.settings.max_redirects:
                            raise _FetchError("redirect_limit")
                        try:
                            redirect_url = normalize_candidate_url(
                                urljoin(current_url, location)
                            )
                        except (TypeError, ValueError):
                            raise _FetchError("invalid_redirect") from None
                        if not _same_website_host(initial_url, redirect_url):
                            raise _FetchError("cross_domain_redirect")
                        if classify_candidate_url(redirect_url) not in _ALLOWED_SITE_KINDS:
                            raise _FetchError("unsafe_redirect_target")
                        redirects += 1
                        current_url = redirect_url
                        continue

                    if not 200 <= response.status_code < 300:
                        raise _FetchError("http_status")
                    content_type = response.headers.get("content-type", "")
                    media_type = content_type.split(";", 1)[0].strip().casefold()
                    if media_type not in {"text/html", "application/xhtml+xml"}:
                        raise _FetchError("content_type")
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > self.settings.max_response_bytes:
                                raise _FetchError("response_too_large")
                        except ValueError:
                            pass
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > self.settings.max_response_bytes:
                            raise _FetchError("response_too_large")
                    encoding = response.charset_encoding or "utf-8"
                    try:
                        html = bytes(body).decode(encoding, errors="replace")
                    except LookupError:
                        html = bytes(body).decode("utf-8", errors="replace")
                    return _FetchedPage(html, current_url)
            except httpx.TimeoutException:
                raise _FetchError("timeout") from None
            except httpx.RequestError:
                raise _FetchError("request_error") from None

    async def _resolve_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        business: Business,
    ) -> FirstPartyInstagramResolution:
        if business.instagram_url:
            return FirstPartyInstagramResolution(
                FirstPartyInstagramStatus.SKIPPED, None, None, (), 0, 0
            )
        trusted_url = trusted_website_for_instagram_resolution(business)
        if trusted_url is None:
            return FirstPartyInstagramResolution(
                FirstPartyInstagramStatus.SKIPPED, None, None, (), 0, 0
            )

        async with semaphore:
            pages_attempted = 0
            pages_succeeded = 0
            page_url: str | None = trusted_url
            while page_url is not None and (
                pages_attempted < self.settings.max_pages_per_business
            ):
                try:
                    page = await self._fetch_page(client, page_url)
                except _BudgetExhausted as exc:
                    return FirstPartyInstagramResolution(
                        FirstPartyInstagramStatus.SKIPPED,
                        None,
                        None,
                        (),
                        pages_attempted + int(exc.page_started),
                        pages_succeeded,
                    )
                except _FetchError as exc:
                    return FirstPartyInstagramResolution(
                        FirstPartyInstagramStatus.TECHNICAL_ERROR,
                        None,
                        None,
                        (),
                        pages_attempted + 1,
                        pages_succeeded,
                        exc.category,
                    )

                pages_attempted += 1
                pages_succeeded += 1
                profiles = extract_instagram_profiles_from_html(page.html)
                if profiles:
                    return resolution_from_extracted_profiles(
                        profiles,
                        pages_attempted=pages_attempted,
                        pages_succeeded=pages_succeeded,
                    )
                if pages_attempted == 1:
                    page_url = _contact_page_from_html(page.html, page.final_url)
                else:
                    page_url = None

            return resolution_from_extracted_profiles(
                (),
                pages_attempted=pages_attempted,
                pages_succeeded=pages_succeeded,
            )

    async def resolve_missing(
        self, businesses: list[Business]
    ) -> tuple[FirstPartyInstagramResolution, ...]:
        semaphore = asyncio.Semaphore(self.settings.concurrency)
        timeout = httpx.Timeout(self.settings.timeout_seconds)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            headers=headers,
            transport=self._transport,
            trust_env=False,
        ) as client:
            resolutions = await asyncio.gather(
                *(
                    self._resolve_one(client, semaphore, business)
                    for business in businesses
                )
            )
        for business, resolution in zip(businesses, resolutions):
            if resolution.status is FirstPartyInstagramStatus.FOUND_OFFICIAL:
                business.instagram_url = resolution.resolved_url or ""
        return tuple(resolutions)


async def resolve_missing_instagrams_first_party(
    businesses: list[Business],
    *,
    resolver: FirstPartyInstagramResolver,
) -> tuple[FirstPartyInstagramResolution, ...]:
    """Apply found profiles only to the supplied objects."""

    if not isinstance(businesses, list):
        raise TypeError("businesses must be a list")
    return await resolver.resolve_missing(businesses)


def build_configured_first_party_resolver() -> FirstPartyInstagramResolver:
    import config

    return FirstPartyInstagramResolver(
        FirstPartyInstagramSettings(
            max_pages_per_business=config.INSTAGRAM_FIRST_PARTY_MAX_PAGES_PER_BUSINESS,
            timeout_seconds=config.INSTAGRAM_FIRST_PARTY_TIMEOUT_SECONDS,
            max_response_bytes=config.INSTAGRAM_FIRST_PARTY_MAX_RESPONSE_BYTES,
            concurrency=config.INSTAGRAM_FIRST_PARTY_CONCURRENCY,
            max_redirects=config.INSTAGRAM_FIRST_PARTY_MAX_REDIRECTS,
        ),
        FirstPartyInstagramRequestBudget(
            config.MAX_INSTAGRAM_FIRST_PARTY_REQUESTS_PER_TASK
        ),
    )

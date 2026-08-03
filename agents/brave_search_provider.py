"""Brave Web Search adapter for official-business website discovery."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import unicodedata

import httpx

from website_candidate_matching import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderTimeout,
    SearchProvider,
    SearchProviderError,
    SearchRequest,
    SearchResult,
)


BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_USER_AGENT = "lidogenerator-website-resolver/1.0"
_LANGUAGE_CODE = re.compile(r"(?=.{2,8}\Z)[A-Za-z]+(?:-[A-Za-z]+)*\Z")
_LOCALE_CODE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{2,8})*\Z")
_OPERATOR_PUNCTUATION = str.maketrans({
    '"': " ",
    "“": " ",
    "”": " ",
    "„": " ",
    "«": " ",
    "»": " ",
    "|": " ",
    "(": " ",
    ")": " ",
    "[": " ",
    "]": " ",
    "{": " ",
    "}": " ",
    "*": " ",
    "+": " ",
    "-": " ",
    "!": " ",
    "^": " ",
    "~": " ",
    "\\": " ",
    ":": " ",
})


def _clean_query_value(value: str) -> str:
    cleaned = "".join(
        "" if unicodedata.category(character).startswith("C") else character
        for character in value
    )
    tokens = cleaned.translate(_OPERATOR_PUNCTUATION).split()
    return " ".join(token for token in tokens if token not in {"AND", "OR", "NOT"})


def _fits_brave_query(query: str) -> bool:
    return len(query) <= 400 and len(query.split()) <= 50


def _prefix_that_fits(value: str, render) -> str:
    """Return the longest word/character prefix whose rendered query fits."""

    words = value.split()
    if not words:
        return ""
    for word_count in range(len(words), 0, -1):
        candidate = " ".join(words[:word_count])
        if _fits_brave_query(render(candidate)):
            return candidate
    first_word = words[0]
    low, high = 1, len(first_word)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = first_word[:middle]
        if _fits_brave_query(render(candidate)):
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def build_brave_search_query(request: SearchRequest) -> str:
    """Build a deterministic, bounded query for one business identity."""

    if not isinstance(request, SearchRequest):
        raise TypeError("request must be a SearchRequest")

    name = _clean_query_value(request.business_name)
    city = _clean_query_value(request.city)
    phone = _clean_query_value(request.phone or "")
    address = _clean_query_value(request.address or "")

    def render(
        current_name: str = name,
        current_city: str = city,
        current_phone: str = phone,
        current_address: str = address,
    ) -> str:
        suffix = " ".join(
            part for part in (current_city, current_phone, current_address) if part
        )
        return f'"{current_name}"' + (f" {suffix}" if suffix else "")

    query = render()
    if _fits_brave_query(query):
        return query

    if address:
        address = _prefix_that_fits(
            address,
            lambda candidate: render(current_address=candidate),
        )
        query = render(current_address=address)
        if _fits_brave_query(query):
            return query
        address = ""

    query = render(current_address="")
    if _fits_brave_query(query):
        return query

    phone = ""
    query = render(current_phone="", current_address="")
    if _fits_brave_query(query):
        return query

    # Preserve the city and at least one character of the quoted name. In normal
    # requests this path is used only for deliberately oversized identity data.
    minimum_city = city[:1]
    name = _prefix_that_fits(
        name,
        lambda candidate: render(
            current_name=candidate,
            current_city=minimum_city,
            current_phone="",
            current_address="",
        ),
    ) or name[:1]
    city = _prefix_that_fits(
        city,
        lambda candidate: render(
            current_name=name,
            current_city=candidate,
            current_phone="",
            current_address="",
        ),
    ) or minimum_city
    query = render(
        current_name=name,
        current_city=city,
        current_phone="",
        current_address="",
    )
    if not _fits_brave_query(query):
        # A pathological long city may need a final trim after the name is fixed.
        city = _prefix_that_fits(
            city,
            lambda candidate: render(
                current_name=name,
                current_city=candidate,
                current_phone="",
                current_address="",
            ),
        ) or city[:1]
        query = render(
            current_name=name,
            current_city=city,
            current_phone="",
            current_address="",
        )
    return query


def _normalize_optional_language(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    normalized = value.strip().casefold()
    if not normalized:
        return None
    if _LANGUAGE_CODE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a valid language code")
    return normalized


def _normalize_optional_locale(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("ui_lang must be a string or None")
    raw = value.strip()
    if not raw:
        return None
    if _LOCALE_CODE.fullmatch(raw) is None:
        raise ValueError("ui_lang must be a valid locale-like value")
    parts = raw.split("-")
    normalized = [parts[0].casefold()]
    normalized.extend(
        part.upper() if len(part) == 2 and part.isalpha() else part
        for part in parts[1:]
    )
    return "-".join(normalized)


@dataclass(frozen=True)
class BraveSearchSettings:
    api_key: str = field(repr=False)
    country: str = "UA"
    search_lang: str | None = None
    ui_lang: str | None = "uk-UA"
    safesearch: str = "moderate"
    max_results: int = 5
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string")
        api_key = self.api_key.strip()
        if not api_key:
            raise ValueError("api_key must not be empty")
        object.__setattr__(self, "api_key", api_key)

        if not isinstance(self.country, str):
            raise TypeError("country must be a string")
        country = self.country.strip().upper()
        if country != "ALL" and (
            len(country) != 2 or not country.isascii() or not country.isalpha()
        ):
            raise ValueError("country must be two ASCII letters or ALL")
        object.__setattr__(self, "country", country)
        object.__setattr__(
            self,
            "search_lang",
            _normalize_optional_language(self.search_lang, "search_lang"),
        )
        object.__setattr__(self, "ui_lang", _normalize_optional_locale(self.ui_lang))

        if not isinstance(self.safesearch, str):
            raise TypeError("safesearch must be a string")
        safesearch = self.safesearch.strip().casefold()
        if safesearch not in {"off", "moderate", "strict"}:
            raise ValueError("safesearch must be one of: off, moderate, strict")
        object.__setattr__(self, "safesearch", safesearch)

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
class BraveSearchTelemetry:
    requests_started: int
    requests_succeeded: int
    requests_failed: int
    last_rate_limit_limit: str | None = None
    last_rate_limit_remaining: str | None = None
    last_rate_limit_reset: str | None = None


def _safe_rate_limit_header(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized[:200] or None


class BraveSearchProvider(SearchProvider):
    """One-request Brave adapter. Mutable telemetry is not concurrency-safe."""

    def __init__(
        self,
        settings: BraveSearchSettings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(settings, BraveSearchSettings):
            raise TypeError("settings must be BraveSearchSettings")
        if transport is not None and not isinstance(transport, httpx.AsyncBaseTransport):
            raise TypeError("transport must be an httpx.AsyncBaseTransport or None")
        self._settings = settings
        self._transport = transport
        self._requests_started = 0
        self._requests_succeeded = 0
        self._requests_failed = 0
        self._last_rate_limit_limit: str | None = None
        self._last_rate_limit_remaining: str | None = None
        self._last_rate_limit_reset: str | None = None

    def telemetry(self) -> BraveSearchTelemetry:
        return BraveSearchTelemetry(
            requests_started=self._requests_started,
            requests_succeeded=self._requests_succeeded,
            requests_failed=self._requests_failed,
            last_rate_limit_limit=self._last_rate_limit_limit,
            last_rate_limit_remaining=self._last_rate_limit_remaining,
            last_rate_limit_reset=self._last_rate_limit_reset,
        )

    def _capture_rate_limit_headers(self, response: httpx.Response) -> None:
        self._last_rate_limit_limit = _safe_rate_limit_header(
            response.headers.get("X-RateLimit-Limit")
        )
        self._last_rate_limit_remaining = _safe_rate_limit_header(
            response.headers.get("X-RateLimit-Remaining")
        )
        self._last_rate_limit_reset = _safe_rate_limit_header(
            response.headers.get("X-RateLimit-Reset")
        )

    async def search(self, request: SearchRequest) -> tuple[SearchResult, ...]:
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")
        query = build_brave_search_query(request)
        count = min(request.max_results, self._settings.max_results)
        timeout = min(request.timeout_seconds, self._settings.timeout_seconds)
        params: dict[str, str | int] = {
            "q": query,
            "count": count,
            "country": self._settings.country,
            "safesearch": self._settings.safesearch,
        }
        if self._settings.ui_lang is not None:
            params["ui_lang"] = self._settings.ui_lang
        if self._settings.search_lang is not None:
            params["search_lang"] = self._settings.search_lang
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._settings.api_key,
            "User-Agent": _USER_AGENT,
        }

        self._requests_started += 1
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                verify=True,
                transport=self._transport,
            ) as client:
                response = await client.get(
                    BRAVE_WEB_SEARCH_ENDPOINT,
                    params=params,
                    headers=headers,
                )
        except httpx.TimeoutException:
            self._requests_failed += 1
            raise ProviderTimeout("brave request timed out") from None
        except httpx.RequestError:
            self._requests_failed += 1
            raise SearchProviderError("brave request failed") from None

        self._capture_rate_limit_headers(response)
        status = response.status_code
        if status != 200:
            self._requests_failed += 1
            if status in {401, 403}:
                raise ProviderAuthError("brave authentication failed")
            if status == 408:
                raise ProviderTimeout("brave request timed out")
            if status == 429:
                raise ProviderRateLimited("brave rate limit exceeded")
            if 500 <= status <= 599:
                raise SearchProviderError("brave service error")
            raise SearchProviderError("brave returned unexpected status")

        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("top-level response must be an object")
            web = payload.get("web")
            if web is None:
                results_payload = None
            elif not isinstance(web, dict):
                raise ValueError("web must be an object")
            else:
                results_payload = web.get("results")
            if results_payload is None:
                parsed_results: tuple[SearchResult, ...] = ()
            else:
                if not isinstance(results_payload, list):
                    raise ValueError("results must be a list")
                results: list[SearchResult] = []
                for rank, item in enumerate(results_payload, start=1):
                    if not isinstance(item, dict):
                        continue
                    url = item.get("url")
                    title = item.get("title", "")
                    description = item.get("description", "")
                    if not isinstance(url, str):
                        continue
                    if title is None:
                        title = ""
                    if description is None:
                        description = ""
                    if not isinstance(title, str) or not isinstance(description, str):
                        continue
                    try:
                        result = SearchResult(url, title, description, rank)
                    except (TypeError, ValueError):
                        continue
                    results.append(result)
                    if len(results) >= count:
                        break
                parsed_results = tuple(results)
        except (ValueError, TypeError):
            self._requests_failed += 1
            raise SearchProviderError("invalid brave response") from None

        self._requests_succeeded += 1
        return parsed_results

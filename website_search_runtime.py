"""Task-scoped construction and request budgeting for website search."""

from __future__ import annotations

from dataclasses import dataclass

import config
from agents.brave_search_provider import (
    BraveSearchProvider,
    BraveSearchSettings,
    BraveSearchTelemetry,
)
from agents.openai_web_search_provider import (
    OpenAIWebSearchProvider,
    OpenAIWebSearchSettings,
    OpenAIWebSearchTelemetry,
)
from website_candidate_matching import (
    ProviderUnavailable,
    SearchProvider,
    SearchRequest,
    SearchResult,
)


@dataclass(frozen=True)
class SearchBudgetSnapshot:
    max_requests: int
    used_requests: int
    remaining_requests: int


class BudgetedSearchProvider(SearchProvider):
    """In-memory task budget wrapper; intentionally not concurrency-safe."""

    def __init__(self, provider: SearchProvider, max_requests: int) -> None:
        if not isinstance(provider, SearchProvider):
            raise TypeError("provider must implement SearchProvider")
        if type(max_requests) is not int:
            raise TypeError("max_requests must be an integer")
        if max_requests < 0:
            raise ValueError("max_requests must be at least 0")
        self._provider = provider
        self._max_requests = max_requests
        self._used_requests = 0

    @property
    def provider(self) -> SearchProvider:
        return self._provider

    def snapshot(self) -> SearchBudgetSnapshot:
        return SearchBudgetSnapshot(
            max_requests=self._max_requests,
            used_requests=self._used_requests,
            remaining_requests=self._max_requests - self._used_requests,
        )

    async def search(self, request: SearchRequest) -> tuple[SearchResult, ...]:
        if self._used_requests >= self._max_requests:
            raise ProviderUnavailable("website search request budget exhausted")
        self._used_requests += 1
        return await self._provider.search(request)


class UnavailableSearchProvider(SearchProvider):
    """Network-free provider representing safe runtime unavailability."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        normalized = " ".join(reason.split())
        if not normalized:
            raise ValueError("reason must not be empty")
        self._reason = normalized

    async def search(self, request: SearchRequest) -> tuple[SearchResult, ...]:
        raise ProviderUnavailable(self._reason)


def build_configured_search_provider(*, max_requests: int | None = None) -> SearchProvider | None:
    request_limit = (
        config.MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK
        if max_requests is None
        else max_requests
    )
    if type(request_limit) is not int or request_limit < 0:
        raise ValueError("max_requests must be a non-negative integer")
    if config.WEBSITE_SEARCH_PROVIDER == "none":
        return None
    if config.WEBSITE_SEARCH_PROVIDER == "brave":
        if not config.BRAVE_SEARCH_API_KEY:
            return UnavailableSearchProvider("Brave Search API key is not configured")
        if request_limit <= 0:
            return UnavailableSearchProvider("website search request budget is disabled")
        settings = BraveSearchSettings(
            api_key=config.BRAVE_SEARCH_API_KEY,
            country=config.BRAVE_SEARCH_COUNTRY,
            search_lang=config.BRAVE_SEARCH_LANGUAGE,
            ui_lang=config.BRAVE_SEARCH_UI_LANGUAGE,
            safesearch=config.BRAVE_SEARCH_SAFESEARCH,
            max_results=config.BRAVE_SEARCH_MAX_RESULTS,
            timeout_seconds=config.BRAVE_SEARCH_TIMEOUT_SECONDS,
        )
        return BudgetedSearchProvider(
            BraveSearchProvider(settings),
            max_requests=request_limit,
        )

    if not config.OPENAI_API_KEY:
        return UnavailableSearchProvider("OpenAI API key is not configured")
    if request_limit <= 0:
        return UnavailableSearchProvider("website search request budget is disabled")
    settings = OpenAIWebSearchSettings(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_WEB_SEARCH_MODEL,
        reasoning_effort=config.OPENAI_WEB_SEARCH_REASONING_EFFORT,
        search_context_size=config.OPENAI_WEB_SEARCH_CONTEXT_SIZE,
        country=config.OPENAI_WEB_SEARCH_COUNTRY,
        external_web_access=config.OPENAI_WEB_SEARCH_EXTERNAL_ACCESS,
        max_results=config.OPENAI_WEB_SEARCH_MAX_RESULTS,
        max_output_tokens=config.OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS,
        timeout_seconds=config.OPENAI_WEB_SEARCH_TIMEOUT_SECONDS,
    )
    return BudgetedSearchProvider(
        OpenAIWebSearchProvider(settings),
        max_requests=request_limit,
    )


def search_budget_snapshot(
    provider: SearchProvider | None,
) -> SearchBudgetSnapshot | None:
    if isinstance(provider, BudgetedSearchProvider):
        return provider.snapshot()
    return None


def brave_telemetry_snapshot(
    provider: SearchProvider | None,
) -> BraveSearchTelemetry | None:
    if isinstance(provider, BudgetedSearchProvider):
        provider = provider.provider
    if isinstance(provider, BraveSearchProvider):
        return provider.telemetry()
    return None


def openai_web_search_telemetry_snapshot(
    provider: SearchProvider | None,
) -> OpenAIWebSearchTelemetry | None:
    if isinstance(provider, BudgetedSearchProvider):
        provider = provider.provider
    if isinstance(provider, OpenAIWebSearchProvider):
        return provider.telemetry()
    return None

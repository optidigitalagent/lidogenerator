"""Task-scoped construction and request budgeting for Instagram search."""

from __future__ import annotations

from dataclasses import dataclass

import config
from agents.openai_instagram_search_provider import (
    OpenAIInstagramSearchProvider,
    OpenAIInstagramSearchSettings,
    OpenAIInstagramSearchTelemetry,
)
from instagram_candidate_matching import (
    InstagramProviderUnavailable,
    InstagramSearchProvider,
    InstagramSearchRequest,
    InstagramSearchResult,
)


@dataclass(frozen=True)
class InstagramSearchBudgetSnapshot:
    max_requests: int
    used_requests: int
    remaining_requests: int


class BudgetedInstagramSearchProvider(InstagramSearchProvider):
    """In-memory task budget wrapper; intentionally not concurrency-safe."""

    def __init__(self, provider: InstagramSearchProvider, max_requests: int) -> None:
        if not isinstance(provider, InstagramSearchProvider):
            raise TypeError("provider must implement InstagramSearchProvider")
        if type(max_requests) is not int:
            raise TypeError("max_requests must be an integer")
        if max_requests < 0:
            raise ValueError("max_requests must be at least 0")
        self._provider = provider
        self._max_requests = max_requests
        self._used_requests = 0

    @property
    def provider(self) -> InstagramSearchProvider:
        return self._provider

    def snapshot(self) -> InstagramSearchBudgetSnapshot:
        return InstagramSearchBudgetSnapshot(
            self._max_requests,
            self._used_requests,
            self._max_requests - self._used_requests,
        )

    async def search(
        self, request: InstagramSearchRequest
    ) -> tuple[InstagramSearchResult, ...]:
        if self._used_requests >= self._max_requests:
            raise InstagramProviderUnavailable(
                "Instagram search request budget exhausted"
            )
        self._used_requests += 1
        return await self._provider.search(request)


class UnavailableInstagramSearchProvider(InstagramSearchProvider):
    """Network-free provider representing safe runtime unavailability."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        normalized = " ".join(reason.split())
        if not normalized:
            raise ValueError("reason must not be empty")
        self._reason = normalized

    async def search(
        self, request: InstagramSearchRequest
    ) -> tuple[InstagramSearchResult, ...]:
        raise InstagramProviderUnavailable(self._reason)


def build_configured_instagram_search_provider() -> InstagramSearchProvider | None:
    if config.INSTAGRAM_SEARCH_PROVIDER == "none":
        return None
    if not config.OPENAI_API_KEY:
        return UnavailableInstagramSearchProvider(
            "OpenAI API key is not configured"
        )
    if config.MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK <= 0:
        return UnavailableInstagramSearchProvider(
            "Instagram search request budget is disabled"
        )
    settings = OpenAIInstagramSearchSettings(
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
    return BudgetedInstagramSearchProvider(
        OpenAIInstagramSearchProvider(settings),
        config.MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK,
    )


def instagram_search_budget_snapshot(
    provider: InstagramSearchProvider | None,
) -> InstagramSearchBudgetSnapshot | None:
    if isinstance(provider, BudgetedInstagramSearchProvider):
        return provider.snapshot()
    return None


def openai_instagram_search_telemetry_snapshot(
    provider: InstagramSearchProvider | None,
) -> OpenAIInstagramSearchTelemetry | None:
    if isinstance(provider, BudgetedInstagramSearchProvider):
        provider = provider.provider
    if isinstance(provider, OpenAIInstagramSearchProvider):
        return provider.telemetry()
    return None

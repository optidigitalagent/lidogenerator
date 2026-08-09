import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import config
import instagram_search_runtime as runtime
from agents.openai_instagram_search_provider import (
    OpenAIInstagramSearchProvider,
    OpenAIInstagramSearchSettings,
)
from instagram_candidate_matching import (
    InstagramProviderUnavailable,
    InstagramSearchProviderError,
    InstagramSearchRequest,
)


class CountingProvider:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    async def search(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ()


class InstagramSearchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return InstagramSearchRequest("Synthetic Brand", "Example City")

    def test_default_none_returns_no_provider(self):
        with patch.object(config, "INSTAGRAM_SEARCH_PROVIDER", "none"):
            self.assertIsNone(runtime.build_configured_instagram_search_provider())

    async def test_missing_openai_key_returns_unavailable(self):
        with (
            patch.object(config, "INSTAGRAM_SEARCH_PROVIDER", "openai"),
            patch.object(config, "OPENAI_API_KEY", ""),
            patch.object(config, "MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK", 2),
        ):
            provider = runtime.build_configured_instagram_search_provider()
        with self.assertRaises(InstagramProviderUnavailable):
            await provider.search(self.request())

    async def test_zero_budget_returns_unavailable(self):
        with (
            patch.object(config, "INSTAGRAM_SEARCH_PROVIDER", "openai"),
            patch.object(config, "OPENAI_API_KEY", "synthetic-test-key"),
            patch.object(config, "MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK", 0),
        ):
            provider = runtime.build_configured_instagram_search_provider()
        with self.assertRaises(InstagramProviderUnavailable):
            await provider.search(self.request())

    async def test_budget_n_delegates_n_times_and_n_plus_one_fails(self):
        inner = CountingProvider()
        provider = runtime.BudgetedInstagramSearchProvider(inner, 2)
        await provider.search(self.request())
        await provider.search(self.request())
        with self.assertRaises(InstagramProviderUnavailable):
            await provider.search(self.request())
        self.assertEqual(inner.calls, 2)
        self.assertEqual(
            provider.snapshot(),
            runtime.InstagramSearchBudgetSnapshot(2, 2, 0),
        )

    async def test_no_retry_after_provider_failure(self):
        inner = CountingProvider(InstagramSearchProviderError("synthetic failure"))
        provider = runtime.BudgetedInstagramSearchProvider(inner, 3)
        with self.assertRaises(InstagramSearchProviderError):
            await provider.search(self.request())
        self.assertEqual(inner.calls, 1)
        self.assertEqual(provider.snapshot().used_requests, 1)

    def test_instagram_budget_is_independent_of_website_and_scorer_flags(self):
        inner = CountingProvider()
        with (
            patch.object(config, "INSTAGRAM_SEARCH_PROVIDER", "openai"),
            patch.object(config, "OPENAI_API_KEY", "synthetic-test-key"),
            patch.object(config, "MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK", 2),
            patch.object(config, "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK", 0),
            patch.object(config, "OPENAI_SCORING_ENABLED", "false"),
            patch.object(runtime, "OpenAIInstagramSearchProvider", return_value=inner),
        ):
            provider = runtime.build_configured_instagram_search_provider()
        self.assertIsInstance(provider, runtime.BudgetedInstagramSearchProvider)
        self.assertEqual(provider.snapshot().max_requests, 2)

    def test_telemetry_unwraps_budget_provider(self):
        fake_response = SimpleNamespace(
            status="completed",
            output=[{
                "type": "web_search_call",
                "action": {"type": "search", "sources": []},
            }],
            output_text='{"results": []}',
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(create=AsyncMock(return_value=fake_response))
        )
        inner = OpenAIInstagramSearchProvider(
            OpenAIInstagramSearchSettings(api_key="synthetic-test-key"),
            client,
        )
        provider = runtime.BudgetedInstagramSearchProvider(inner, 1)
        snapshot = runtime.openai_instagram_search_telemetry_snapshot(provider)
        self.assertEqual(snapshot.requests_started, 0)
        self.assertIsNone(runtime.openai_instagram_search_telemetry_snapshot(None))

    def test_budget_snapshot_only_exposes_budget_wrapper(self):
        provider = runtime.BudgetedInstagramSearchProvider(CountingProvider(), 4)
        self.assertEqual(runtime.instagram_search_budget_snapshot(provider).remaining_requests, 4)
        self.assertIsNone(runtime.instagram_search_budget_snapshot(CountingProvider()))

    def test_config_defaults_and_integer_validation(self):
        self.assertIn(config.INSTAGRAM_SEARCH_PROVIDER, {"none", "openai"})
        self.assertGreaterEqual(config.MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK, 0)
        with patch.dict(os.environ, {"SYNTHETIC_INSTAGRAM_BUDGET": "not-an-int"}):
            with self.assertRaises(ValueError):
                config._environment_integer(
                    "SYNTHETIC_INSTAGRAM_BUDGET", "0", 0, 1000
                )
        with patch.dict(os.environ, {"SYNTHETIC_INSTAGRAM_BUDGET": "1001"}):
            with self.assertRaises(ValueError):
                config._environment_integer(
                    "SYNTHETIC_INSTAGRAM_BUDGET", "0", 0, 1000
                )


if __name__ == "__main__":
    unittest.main()

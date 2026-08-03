"""Offline tests for task search budgets, runtime construction, and config."""

from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import website_search_runtime as runtime
from agents.brave_search_provider import BraveSearchProvider
from website_candidate_matching import (
    ProviderTimeout,
    ProviderUnavailable,
    SearchRequest,
)


REQUEST = SearchRequest("Business", "Kyiv")


class RecordingProvider:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = []
        self.error = error

    async def search(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return ()


class SearchBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_zero_budget_refuses_without_delegating(self) -> None:
        underlying = RecordingProvider()
        provider = runtime.BudgetedSearchProvider(underlying, 0)
        with self.assertRaisesRegex(ProviderUnavailable, "budget exhausted"):
            await provider.search(REQUEST)
        self.assertEqual(underlying.calls, [])
        self.assertEqual(
            provider.snapshot(),
            runtime.SearchBudgetSnapshot(0, 0, 0),
        )

    async def test_exact_budget_delegates_once_per_call(self) -> None:
        underlying = RecordingProvider()
        provider = runtime.BudgetedSearchProvider(underlying, 2)
        self.assertEqual(await provider.search(REQUEST), ())
        self.assertEqual(await provider.search(REQUEST), ())
        with self.assertRaises(ProviderUnavailable):
            await provider.search(REQUEST)
        self.assertEqual(underlying.calls, [REQUEST, REQUEST])
        self.assertEqual(provider.snapshot(), runtime.SearchBudgetSnapshot(2, 2, 0))

    async def test_failures_consume_budget(self) -> None:
        underlying = RecordingProvider(ProviderTimeout("timeout"))
        provider = runtime.BudgetedSearchProvider(underlying, 1)
        with self.assertRaises(ProviderTimeout):
            await provider.search(REQUEST)
        with self.assertRaises(ProviderUnavailable):
            await provider.search(REQUEST)
        self.assertEqual(len(underlying.calls), 1)
        self.assertEqual(provider.snapshot().used_requests, 1)

    def test_budget_validation_and_frozen_snapshot(self) -> None:
        with self.assertRaises(TypeError):
            runtime.BudgetedSearchProvider(RecordingProvider(), True)
        with self.assertRaises(ValueError):
            runtime.BudgetedSearchProvider(RecordingProvider(), -1)
        snapshot = runtime.SearchBudgetSnapshot(1, 0, 1)
        with self.assertRaises(FrozenInstanceError):
            snapshot.used_requests = 1  # type: ignore[misc]

    async def test_unavailable_provider_always_refuses_without_network(self) -> None:
        provider = runtime.UnavailableSearchProvider(" safe reason ")
        with self.assertRaisesRegex(ProviderUnavailable, "safe reason"):
            await provider.search(REQUEST)
        with self.assertRaises(ValueError):
            runtime.UnavailableSearchProvider("  ")


class RuntimeFactoryTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "WEBSITE_SEARCH_PROVIDER": "none",
            "BRAVE_SEARCH_API_KEY": "",
            "BRAVE_SEARCH_COUNTRY": "UA",
            "BRAVE_SEARCH_LANGUAGE": "",
            "BRAVE_SEARCH_UI_LANGUAGE": "uk-UA",
            "BRAVE_SEARCH_SAFESEARCH": "moderate",
            "BRAVE_SEARCH_MAX_RESULTS": 5,
            "BRAVE_SEARCH_TIMEOUT_SECONDS": 10.0,
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": 0,
        }
        values.update(overrides)
        return patch.multiple(runtime.config, **values)

    def test_factory_none(self) -> None:
        with self._config():
            self.assertIsNone(runtime.build_configured_search_provider())

    def test_factory_brave_missing_key_and_zero_budget(self) -> None:
        with self._config(WEBSITE_SEARCH_PROVIDER="brave", MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK=1):
            provider = runtime.build_configured_search_provider()
            self.assertIsInstance(provider, runtime.UnavailableSearchProvider)
        with self._config(WEBSITE_SEARCH_PROVIDER="brave", BRAVE_SEARCH_API_KEY="secret"):
            provider = runtime.build_configured_search_provider()
            self.assertIsInstance(provider, runtime.UnavailableSearchProvider)
        self.assertNotIn("secret", repr(provider))

    def test_factory_configured_maps_settings_and_makes_no_request(self) -> None:
        underlying = RecordingProvider()
        with (
            self._config(
                WEBSITE_SEARCH_PROVIDER="brave",
                BRAVE_SEARCH_API_KEY="private-key",
                BRAVE_SEARCH_COUNTRY="PL",
                BRAVE_SEARCH_LANGUAGE="pl",
                BRAVE_SEARCH_UI_LANGUAGE="pl-PL",
                BRAVE_SEARCH_SAFESEARCH="strict",
                BRAVE_SEARCH_MAX_RESULTS=7,
                BRAVE_SEARCH_TIMEOUT_SECONDS=4.5,
                MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK=3,
            ),
            patch.object(runtime, "BraveSearchProvider", return_value=underlying) as constructor,
        ):
            provider = runtime.build_configured_search_provider()
        self.assertIsInstance(provider, runtime.BudgetedSearchProvider)
        self.assertEqual(underlying.calls, [])
        settings = constructor.call_args.args[0]
        self.assertEqual(settings.country, "PL")
        self.assertEqual(settings.search_lang, "pl")
        self.assertEqual(settings.ui_lang, "pl-PL")
        self.assertEqual(settings.safesearch, "strict")
        self.assertEqual(settings.max_results, 7)
        self.assertEqual(settings.timeout_seconds, 4.5)
        self.assertNotIn("private-key", repr(settings))
        self.assertEqual(runtime.search_budget_snapshot(provider).max_requests, 3)

    def test_snapshot_helpers_and_telemetry_unwrap(self) -> None:
        brave = BraveSearchProvider(runtime.BraveSearchSettings("private-key"))
        budgeted = runtime.BudgetedSearchProvider(brave, 2)
        self.assertEqual(runtime.brave_telemetry_snapshot(budgeted).requests_started, 0)
        self.assertEqual(runtime.brave_telemetry_snapshot(brave).requests_started, 0)
        self.assertIsNone(runtime.search_budget_snapshot(brave))
        self.assertIsNone(runtime.brave_telemetry_snapshot(RecordingProvider()))
        self.assertIsNone(runtime.search_budget_snapshot(None))


class ConfigSubprocessTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def _run_config(self, **overrides) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        defaults = {
            "WEBSITE_SEARCH_PROVIDER": "none",
            "BRAVE_SEARCH_API_KEY": "",
            "BRAVE_SEARCH_COUNTRY": "UA",
            "BRAVE_SEARCH_LANGUAGE": "",
            "BRAVE_SEARCH_UI_LANGUAGE": "uk-UA",
            "BRAVE_SEARCH_SAFESEARCH": "moderate",
            "BRAVE_SEARCH_MAX_RESULTS": "5",
            "BRAVE_SEARCH_TIMEOUT_SECONDS": "10",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "0",
        }
        defaults.update({name: str(value) for name, value in overrides.items()})
        environment.update(defaults)
        return subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, config; "
                    "print(json.dumps([config.WEBSITE_SEARCH_PROVIDER, "
                    "config.MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK, "
                    "config.BRAVE_SEARCH_MAX_RESULTS, "
                    "config.BRAVE_SEARCH_TIMEOUT_SECONDS]))"
                ),
            ],
            cwd=self.ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def test_defaults_and_valid_brave_without_required_key(self) -> None:
        default = self._run_config()
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(json.loads(default.stdout), ["none", 0, 5, 10.0])
        brave = self._run_config(
            WEBSITE_SEARCH_PROVIDER="brave",
            MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK=1,
        )
        self.assertEqual(brave.returncode, 0, brave.stderr)
        self.assertEqual(json.loads(brave.stdout)[:2], ["brave", 1])

    def test_invalid_provider_numbers_and_bounds(self) -> None:
        invalid = (
            {"WEBSITE_SEARCH_PROVIDER": "other"},
            {"BRAVE_SEARCH_MAX_RESULTS": "1.5"},
            {"BRAVE_SEARCH_MAX_RESULTS": 0},
            {"BRAVE_SEARCH_MAX_RESULTS": 11},
            {"BRAVE_SEARCH_TIMEOUT_SECONDS": "nan"},
            {"BRAVE_SEARCH_TIMEOUT_SECONDS": 0},
            {"BRAVE_SEARCH_TIMEOUT_SECONDS": 31},
            {"MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "1.5"},
            {"MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": -1},
            {"MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": 1001},
        )
        for values in invalid:
            with self.subTest(values=values):
                result = self._run_config(BRAVE_SEARCH_API_KEY="do-not-print-this", **values)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("do-not-print-this", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

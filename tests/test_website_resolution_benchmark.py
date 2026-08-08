"""Offline tests for the website-resolution benchmark harness."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import run_website_resolution_benchmark as runner
from website_candidate_matching import (
    ProviderAuthError,
    ProviderTimeout,
    SearchResult,
)
from website_resolution_benchmark import (
    BenchmarkGateDecision,
    BenchmarkLabel,
    WebsiteResolutionBenchmarkCase,
    WebsiteResolutionBenchmarkResult,
    evaluate_benchmark_gate,
    load_benchmark_cases,
    run_benchmark_case,
    summarize_benchmark,
    write_benchmark_outputs,
)


def _case(
    case_id: str = "official_01",
    label: BenchmarkLabel = BenchmarkLabel.OFFICIAL_DOMAIN,
    expected_domain: str | None = "amber-kite.example",
    **overrides,
) -> WebsiteResolutionBenchmarkCase:
    values = {
        "case_id": case_id,
        "business_name": "Amber Kite Atelier",
        "city": "Exampleton",
        "address": "10 Lantern Lane",
        "phone": "+15550100001",
        "label": label,
        "expected_domain": expected_domain,
        "notes": "Human-approved fixture label",
    }
    values.update(overrides)
    return WebsiteResolutionBenchmarkCase(**values)


def _case_record(case_id="official_01", **overrides):
    values = {
        "case_id": case_id,
        "business_name": "Amber Kite Atelier",
        "city": "Exampleton",
        "address": "10 Lantern Lane",
        "phone": "+15550100001",
        "label": "OFFICIAL_DOMAIN",
        "expected_domain": "amber-kite.example",
        "notes": "Approved fixture",
    }
    values.update(overrides)
    return values


def _result(case_id="official_01", **overrides):
    values = {
        "case_id": case_id,
        "label": BenchmarkLabel.OFFICIAL_DOMAIN,
        "expected_domain": "amber-kite.example",
        "provider_request_succeeded": True,
        "provider_result_domains": ("amber-kite.example",),
        "resolution_status": "found_official",
        "resolved_domain": "amber-kite.example",
        "expected_domain_returned": True,
        "expected_domain_promoted": True,
        "wrong_domain_returned": False,
        "wrong_domain_promoted": False,
        "safe_no_match": False,
        "tool_calls_seen": 1,
        "search_actions_seen": 1,
        "open_page_actions_seen": 0,
        "find_in_page_actions_seen": 0,
        "sources_seen": 1,
        "identity_candidates_rejected": 0,
        "error_category": None,
    }
    values.update(overrides)
    return WebsiteResolutionBenchmarkResult(**values)


class FakeProvider:
    def __init__(self, outcomes=(), telemetry_step=None):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.requests = []
        self.counters = {
            "tool_calls_seen": 0,
            "search_actions_seen": 0,
            "open_page_actions_seen": 0,
            "find_in_page_actions_seen": 0,
            "sources_seen": 0,
            "identity_candidates_rejected": 0,
        }
        self.telemetry_step = telemetry_step or {}

    async def search(self, request):
        self.calls += 1
        self.requests.append(request)
        for name, amount in self.telemetry_step.items():
            self.counters[name] += amount
        outcome = self.outcomes.pop(0) if self.outcomes else ()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def telemetry(self):
        return types.SimpleNamespace(**self.counters)


class DatasetTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.json"
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
            return load_benchmark_cases(path)

    def test_valid_order_preserved_and_domain_normalized(self):
        payload = {
            "version": 1,
            "cases": [
                _case_record("second", expected_domain="WWW.AMBER-KITE.EXAMPLE"),
                _case_record(
                    "first",
                    label="NO_OFFICIAL_SITE",
                    expected_domain=None,
                ),
            ],
        }
        cases = self._load(payload)
        self.assertEqual(tuple(case.case_id for case in cases), ("second", "first"))
        self.assertEqual(cases[0].expected_domain, "amber-kite.example")
        self.assertIs(cases[1].label, BenchmarkLabel.NO_OFFICIAL_SITE)

    def test_invalid_version_and_top_level_schema(self):
        for payload in (
            {"version": 2, "cases": []},
            {"version": True, "cases": []},
            {"version": 1, "cases": [], "extra": True},
            {"version": 1, "cases": {}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self._load(payload)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self._load({"version": 1, "cases": [_case_record(), _case_record()]})

    def test_bad_labels_expected_domain_rules_and_excluded_domains(self):
        invalid = (
            {"label": "UNKNOWN"},
            {"expected_domain": None},
            {"label": "NO_OFFICIAL_SITE"},
            {"expected_domain": "instagram.com"},
            {"expected_domain": "linktr.ee"},
            {"expected_domain": "locator.ua"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._load({"version": 1, "cases": [_case_record(**overrides)]})

    def test_malformed_json_extra_fields_and_bad_identity_rejected(self):
        with self.assertRaises(ValueError):
            self._load("{not json")
        extra = _case_record()
        extra["raw_prompt"] = "forbidden"
        with self.assertRaises(ValueError):
            self._load({"version": 1, "cases": [extra]})
        with self.assertRaises(ValueError):
            self._load({"version": 1, "cases": [_case_record(phone="123")]})

    def test_case_contract_is_frozen_and_case_id_is_simple(self):
        case = _case()
        with self.assertRaises(FrozenInstanceError):
            case.city = "Elsewhere"  # type: ignore[misc]
        for case_id in ("", "has space", "../escape", "a.b"):
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                _case(case_id=case_id)


class OfficialDomainEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_expected_returned_and_promoted(self):
        provider = FakeProvider(((SearchResult(
            "https://amber-kite.example/", "Amber Kite Atelier", "Call 15550100001", 1
        ),),))
        result = await run_benchmark_case(_case(), provider)
        self.assertEqual(provider.calls, 1)
        self.assertTrue(result.expected_domain_returned)
        self.assertTrue(result.expected_domain_promoted)
        self.assertFalse(result.wrong_domain_promoted)

    async def test_expected_returned_not_promoted_is_safe_no_match(self):
        provider = FakeProvider(((SearchResult(
            "https://amber-kite.example/", "Uncorroborated page", "", 1
        ),),))
        result = await run_benchmark_case(_case(), provider)
        self.assertTrue(result.expected_domain_returned)
        self.assertFalse(result.expected_domain_promoted)
        self.assertTrue(result.safe_no_match)

    async def test_wrong_domain_returned_but_rejected(self):
        provider = FakeProvider(((SearchResult(
            "https://wrong.example/", "Amber Kite Atelier", "Exampleton", 1
        ),),))
        result = await run_benchmark_case(_case(), provider)
        self.assertTrue(result.wrong_domain_returned)
        self.assertFalse(result.wrong_domain_promoted)
        self.assertTrue(result.safe_no_match)

    async def test_wrong_domain_promoted_is_critical(self):
        provider = FakeProvider(((SearchResult(
            "https://wrong.example/", "Wrong page", "Call 15550100001", 1
        ),),))
        result = await run_benchmark_case(_case(), provider)
        self.assertTrue(result.wrong_domain_returned)
        self.assertTrue(result.wrong_domain_promoted)
        self.assertFalse(result.safe_no_match)

    async def test_empty_results_are_safe_no_match(self):
        result = await run_benchmark_case(_case(), FakeProvider(((),)))
        self.assertEqual(result.resolution_status, "not_found")
        self.assertTrue(result.safe_no_match)


class NoOfficialSiteEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def no_site_case(self):
        return _case(
            case_id="no_site_01",
            label=BenchmarkLabel.NO_OFFICIAL_SITE,
            expected_domain=None,
        )

    async def test_empty_is_safe(self):
        result = await run_benchmark_case(self.no_site_case(), FakeProvider(((),)))
        self.assertFalse(result.expected_domain_returned)
        self.assertFalse(result.expected_domain_promoted)
        self.assertTrue(result.safe_no_match)

    async def test_noisy_candidate_rejected_is_safe(self):
        provider = FakeProvider(((SearchResult(
            "https://noise.example/", "Amber Kite Atelier", "Exampleton", 1
        ),),))
        result = await run_benchmark_case(self.no_site_case(), provider)
        self.assertTrue(result.wrong_domain_returned)
        self.assertFalse(result.wrong_domain_promoted)
        self.assertTrue(result.safe_no_match)

    async def test_promoted_domain_is_critical(self):
        provider = FakeProvider(((SearchResult(
            "https://noise.example/", "Noise", "Call 15550100001", 1
        ),),))
        result = await run_benchmark_case(self.no_site_case(), provider)
        self.assertTrue(result.wrong_domain_promoted)
        self.assertFalse(result.safe_no_match)


class TelemetryAndErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_safe_telemetry_deltas_are_captured(self):
        provider = FakeProvider(
            ((), ()),
            {
                "tool_calls_seen": 1,
                "search_actions_seen": 1,
                "open_page_actions_seen": 2,
                "find_in_page_actions_seen": 3,
                "sources_seen": 4,
                "identity_candidates_rejected": 5,
            },
        )
        first = await run_benchmark_case(_case("one"), provider)
        second = await run_benchmark_case(_case("two"), provider)
        for result in (first, second):
            self.assertEqual(
                (
                    result.tool_calls_seen,
                    result.search_actions_seen,
                    result.open_page_actions_seen,
                    result.find_in_page_actions_seen,
                    result.sources_seen,
                    result.identity_candidates_rejected,
                ),
                (1, 1, 2, 3, 4, 5),
            )
        self.assertNotIn("source_url", WebsiteResolutionBenchmarkResult.__dataclass_fields__)

    async def test_provider_errors_are_safe_categories_and_never_retried(self):
        provider = FakeProvider((ProviderTimeout("RAW SECRET exception"),))
        result = await run_benchmark_case(_case(), provider)
        self.assertEqual(provider.calls, 1)
        self.assertFalse(result.provider_request_succeeded)
        self.assertEqual(result.error_category, "timeout")
        self.assertNotIn("RAW SECRET", repr(result))

    async def test_unexpected_bad_provider_output_is_safe(self):
        result = await run_benchmark_case(_case(), FakeProvider((["bad"],)))
        self.assertTrue(result.provider_request_succeeded)
        self.assertEqual(result.error_category, "unexpected_error")


class SummaryTests(unittest.TestCase):
    def test_counts_rates_and_telemetry(self):
        results = (
            _result(),
            _result(
                "official_02",
                provider_request_succeeded=False,
                provider_result_domains=("wrong.example",),
                resolution_status="resolution_error",
                resolved_domain=None,
                expected_domain_returned=False,
                expected_domain_promoted=False,
                wrong_domain_returned=True,
                safe_no_match=True,
                tool_calls_seen=2,
                open_page_actions_seen=1,
                sources_seen=3,
                identity_candidates_rejected=2,
                error_category="timeout",
            ),
            _result(
                "no_site_01",
                label=BenchmarkLabel.NO_OFFICIAL_SITE,
                expected_domain=None,
                provider_result_domains=(),
                resolution_status="not_found",
                resolved_domain=None,
                expected_domain_returned=False,
                expected_domain_promoted=False,
                safe_no_match=True,
            ),
        )
        summary = summarize_benchmark(results)
        self.assertEqual((summary.total_cases, summary.official_domain_cases, summary.no_official_site_cases), (3, 2, 1))
        self.assertEqual((summary.provider_requests_succeeded, summary.provider_requests_failed), (2, 1))
        self.assertEqual(summary.provider_domain_recall, 0.5)
        self.assertEqual(summary.resolver_domain_recall, 0.5)
        self.assertEqual(summary.resolver_precision, 1.0)
        self.assertEqual(summary.no_site_specificity, 1.0)
        self.assertEqual(summary.technical_success_rate, 2 / 3)
        self.assertEqual(summary.tool_call_limit_violations, 1)
        self.assertEqual(summary.total_tool_calls, 4)
        self.assertEqual(summary.total_open_page_actions, 1)
        self.assertEqual(summary.total_sources_seen, 5)
        self.assertEqual(summary.total_identity_candidates_rejected, 2)

    def test_zero_denominators_are_none(self):
        summary = summarize_benchmark(())
        for name in (
            "provider_domain_recall",
            "resolver_domain_recall",
            "resolver_precision",
            "no_site_specificity",
            "critical_false_positive_rate",
            "technical_success_rate",
        ):
            self.assertIsNone(getattr(summary, name))


class GateTests(unittest.TestCase):
    def setUp(self):
        results = [_result(f"official_{index}") for index in range(8)]
        results.extend(
            _result(
                f"no_site_{index}",
                label=BenchmarkLabel.NO_OFFICIAL_SITE,
                expected_domain=None,
                provider_result_domains=(),
                resolution_status="not_found",
                resolved_domain=None,
                expected_domain_returned=False,
                expected_domain_promoted=False,
                safe_no_match=True,
            )
            for index in range(4)
        )
        self.passing = summarize_benchmark(results)

    def test_insufficient_sample(self):
        self.assertIs(
            evaluate_benchmark_gate(summarize_benchmark((_result(),))),
            BenchmarkGateDecision.INSUFFICIENT_SAMPLE,
        )

    def test_critical_false_positive(self):
        summary = replace(self.passing, wrong_domains_promoted=1)
        self.assertIs(evaluate_benchmark_gate(summary), BenchmarkGateDecision.FAIL_CRITICAL_FALSE_POSITIVE)

    def test_low_precision(self):
        summary = replace(self.passing, resolver_precision=0.94)
        self.assertIs(evaluate_benchmark_gate(summary), BenchmarkGateDecision.FAIL_LOW_PRECISION)

    def test_low_recall(self):
        summary = replace(self.passing, resolver_domain_recall=0.59)
        self.assertIs(evaluate_benchmark_gate(summary), BenchmarkGateDecision.FAIL_LOW_RECALL)

    def test_low_technical(self):
        summary = replace(self.passing, technical_success_rate=0.89)
        self.assertIs(evaluate_benchmark_gate(summary), BenchmarkGateDecision.FAIL_TECHNICAL)

    def test_technical_failures_cannot_pass_as_safe_no_matches(self):
        results = [_result(f"official_{index}") for index in range(8)]
        results.extend(
            _result(
                f"no_site_{index}",
                label=BenchmarkLabel.NO_OFFICIAL_SITE,
                expected_domain=None,
                provider_result_domains=(),
                resolution_status=(
                    "resolution_error" if index >= 2 else "not_found"
                ),
                resolved_domain=None,
                provider_request_succeeded=index < 2,
                expected_domain_returned=False,
                expected_domain_promoted=False,
                safe_no_match=True,
                error_category="timeout" if index >= 2 else None,
            )
            for index in range(4)
        )
        summary = summarize_benchmark(results)
        self.assertEqual(summary.safe_no_matches, 4)
        self.assertEqual(summary.technical_success_rate, 10 / 12)
        self.assertIs(
            evaluate_benchmark_gate(summary),
            BenchmarkGateDecision.FAIL_TECHNICAL,
        )

    def test_pass(self):
        self.assertIs(evaluate_benchmark_gate(self.passing), BenchmarkGateDecision.PASS)


class OutputTests(unittest.TestCase):
    def test_all_safe_artifacts_and_summary_sections(self):
        results = (_result(),)
        summary = summarize_benchmark(results)
        with tempfile.TemporaryDirectory() as temporary:
            write_benchmark_outputs(
                temporary,
                results,
                summary,
                BenchmarkGateDecision.INSUFFICIENT_SAMPLE,
            )
            names = {path.name for path in Path(temporary).iterdir()}
            self.assertEqual(names, {
                "benchmark_results.json",
                "benchmark_results.csv",
                "benchmark_summary.json",
                "benchmark_summary.txt",
            })
            text = "\n".join(path.read_text(encoding="utf-8") for path in Path(temporary).iterdir())
        for expected in ("Website Resolution Benchmark", "provider domain recall", "INSUFFICIENT_SAMPLE"):
            self.assertIn(expected, text)
        for forbidden in ("raw_prompt", "source_urls", "snippet", "response_id", "usage", "headers"):
            self.assertNotIn(forbidden, text)


def _dataset(path: Path, count: int, official_count: int = 8):
    cases = []
    for index in range(count):
        official = index < official_count
        cases.append(_case_record(
            f"case_{index}",
            business_name=f"Fixture Business {index}",
            address=f"SECRET_ADDRESS_{index}",
            phone=f"+1555010{index:04d}",
            label="OFFICIAL_DOMAIN" if official else "NO_OFFICIAL_SITE",
            expected_domain=f"fixture-{index}.example" if official else None,
        ))
    path.write_text(json.dumps({"version": 1, "cases": cases}), encoding="utf-8")


class RunnerProvider(FakeProvider):
    async def search(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
        index = int(request.business_name.rsplit(" ", 1)[1])
        if index >= 8:
            return ()
        return (SearchResult(
            f"https://fixture-{index}.example/",
            request.business_name,
            f"Call {request.phone}",
            1,
        ),)


class FakeBudgetedProvider(RunnerProvider):
    pass


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def _modules(self, provider):
        config = types.ModuleType("config")
        config.WEBSITE_SEARCH_PROVIDER = "openai"
        config.MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK = 12
        config.OPENAI_WEB_SEARCH_MODEL = "fixture-model"
        config.OPENAI_WEB_SEARCH_REASONING_EFFORT = "low"
        config.OPENAI_WEB_SEARCH_CONTEXT_SIZE = "low"
        runtime = types.ModuleType("website_search_runtime")
        runtime.BudgetedSearchProvider = FakeBudgetedProvider
        runtime.build_configured_search_provider = lambda: provider
        return {"config": config, "website_search_runtime": runtime}

    async def _invoke(self, count, environment, provider=None):
        provider = provider or FakeBudgetedProvider()
        with tempfile.TemporaryDirectory() as temporary:
            cases = Path(temporary) / "cases.json"
            output = Path(temporary) / "results"
            _dataset(cases, count, official_count=min(8, count))
            stdout = io.StringIO()
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, self._modules(provider)),
                redirect_stdout(stdout),
            ):
                code = await runner._run(cases, output)
            persisted = ""
            if output.exists():
                persisted = "\n".join(
                    path.read_text(encoding="utf-8") for path in output.iterdir()
                )
            return code, stdout.getvalue(), persisted, provider

    async def test_no_gate_wrong_provider_missing_key_budget_mismatch_and_case_caps_make_zero_requests(self):
        valid = {
            runner.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "12",
        }
        cases = (
            (12, {**valid, runner.LIVE_GATE: "0"}),
            (12, {**valid, "WEBSITE_SEARCH_PROVIDER": "brave"}),
            (12, {**valid, "OPENAI_API_KEY": ""}),
            (12, {**valid, "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "13"}),
            (21, {**valid, "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "21"}),
            (11, {**valid, "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "11"}),
        )
        for count, environment in cases:
            with self.subTest(count=count, environment=environment):
                code, _, _, provider = await self._invoke(count, environment)
                self.assertIn(code, {0, 2})
                self.assertEqual(provider.calls, 0)

    async def test_exactly_one_request_per_case_sequential_and_pass(self):
        environment = {
            runner.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "12",
        }
        code, stdout, persisted, provider = await self._invoke(12, environment)
        self.assertEqual(code, 0)
        self.assertEqual(provider.calls, 12)
        self.assertEqual([request.business_name for request in provider.requests], [f"Fixture Business {index}" for index in range(12)])
        self.assertIn("LIVE_BENCHMARK_AUTHORIZED=yes", stdout)
        self.assertIn("gate_decision=PASS", stdout)
        for forbidden in ("SECRET_API_KEY", "SECRET_ADDRESS", "+1555010", "https://", "Call "):
            self.assertNotIn(forbidden, stdout + persisted)

    async def test_timeouts_have_no_retry_and_remaining_cases_continue(self):
        environment = {
            runner.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "12",
        }
        provider = FakeBudgetedProvider((ProviderTimeout("raw") for _ in range(12)))
        code, _, _, provider = await self._invoke(12, environment, provider)
        self.assertEqual(code, 1)
        self.assertEqual(provider.calls, 12)

    async def test_fatal_authentication_stops_immediately(self):
        environment = {
            runner.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "12",
        }
        provider = FakeBudgetedProvider((ProviderAuthError("raw credential detail"),))
        code, stdout, persisted, provider = await self._invoke(12, environment, provider)
        self.assertEqual(code, 2)
        self.assertEqual(provider.calls, 1)
        self.assertIn("benchmark_stopped=fatal_provider_error", stdout)
        self.assertNotIn("raw credential detail", stdout + persisted)


class SecurityTests(unittest.TestCase):
    def test_result_contract_contains_only_allowlisted_fields(self):
        self.assertEqual(
            tuple(WebsiteResolutionBenchmarkResult.__dataclass_fields__),
            (
                "case_id", "label", "expected_domain", "provider_request_succeeded",
                "provider_result_domains", "resolution_status", "resolved_domain",
                "expected_domain_returned", "expected_domain_promoted",
                "wrong_domain_returned", "wrong_domain_promoted", "safe_no_match",
                "tool_calls_seen", "search_actions_seen", "open_page_actions_seen",
                "find_in_page_actions_seen", "sources_seen",
                "identity_candidates_rejected", "error_category",
            ),
        )
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("max_retries", source)
        self.assertNotIn("raw_prompt", source)
        self.assertNotIn("source_urls", source)


if __name__ == "__main__":
    unittest.main()

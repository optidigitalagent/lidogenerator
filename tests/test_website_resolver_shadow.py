"""Offline invariants for non-interfering website resolver shadow mode."""

import json
import unittest
from unittest.mock import patch

import orchestrator
from agents import reporter
from agents.openai_web_search_provider import OpenAIWebSearchTelemetry
from models import Business
from website_candidate_matching import ProviderUnavailable, SearchRequest
from website_pipeline import ResolverMode
from website_search_runtime import BudgetedSearchProvider, SearchBudgetSnapshot


class WebsiteResolverShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_off_runs_legacy_checker_only_on_real_objects(self) -> None:
        business = Business(name="Legacy", instagram_url="https://instagram.com/legacy")
        calls = []

        async def site_check(items, progress_callback=None):
            calls.append(items)
            items[0].has_site = False
            items[0].site_quality = "none"
            return items

        async def resolver(*args, **kwargs):
            self.fail("resolver must not run in off mode")

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business], ResolverMode.OFF, None, task_id=10
            )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], business)
        self.assertTrue(business.is_lead)

    async def test_shadow_orders_legacy_audit_first_and_isolates_every_mutation(self) -> None:
        business = Business(
            name="Synthetic",
            city="Test City",
            instagram_url="https://instagram.com/synthetic",
        )
        events = []
        resolver_inputs = []

        async def site_check(items, progress_callback=None):
            events.append("site")
            self.assertIs(items[0], business)
            items[0].has_site = False
            items[0].site_quality = "none"
            items[0].website_audit_status = "no_official_site"
            items[0].website_audit_evidence = '["no_official_site"]'
            return items

        async def resolver(items, provider=None, progress_callback=None):
            events.append("resolver")
            resolver_inputs.extend(items)
            item = items[0]
            item.website_original_url = "https://copy-original.invalid/"
            item.website_resolved_url = "https://resolved-shadow.invalid/"
            item.website_resolution_status = "found_official"
            item.website_resolution_source = "web_search"
            item.website_resolution_confidence = 0.99
            item.website_resolution_evidence = "shadow evidence"
            item.website_resolution_error = "shadow error"
            item.has_site = True
            item.site_quality = "good"
            item.website_final_url = "https://resolved-shadow.invalid/final"
            item.website_audit_status = "good"
            item.website_audit_http_status = 200
            item.website_audit_evidence = "shadow audit evidence"
            item.website_audit_error = "shadow audit error"
            item.lead_decision = "not_lead"
            item.lead_decision_reason = "shadow decision"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business], ResolverMode.SHADOW, None, task_id=11
            )

        self.assertEqual(events, ["site", "resolver"])
        self.assertIsNot(resolver_inputs[0], business)
        self.assertEqual(business.website_original_url, "")
        self.assertEqual(business.website_resolved_url, "")
        self.assertEqual(business.website_resolution_status, "")
        self.assertEqual(business.website_resolution_source, "")
        self.assertEqual(business.website_resolution_confidence, 0.0)
        self.assertEqual(business.website_resolution_evidence, "")
        self.assertEqual(business.website_resolution_error, "")
        self.assertFalse(business.has_site)
        self.assertEqual(business.site_quality, "none")
        self.assertEqual(business.website_status, "no website")
        self.assertEqual(business.website_final_url, "")
        self.assertEqual(business.website_audit_status, "no_official_site")
        self.assertIsNone(business.website_audit_http_status)
        self.assertEqual(business.website_audit_evidence, '["no_official_site"]')
        self.assertEqual(business.website_audit_error, "")
        self.assertEqual(business.lead_decision, "")
        self.assertEqual(business.lead_decision_reason, "")
        self.assertTrue(business.is_lead)
        self.assertEqual(business.effective_website_url, "")

    async def test_shadow_preserves_exact_legacy_lead_set(self) -> None:
        businesses = [
            Business(name="A", instagram_url="https://instagram.com/a"),
            Business(
                name="B",
                instagram_url="https://instagram.com/b",
                website="https://maps-good.invalid/",
            ),
            Business(name="C"),
        ]

        async def site_check(items, progress_callback=None):
            for item in items:
                item.has_site = bool(item.website)
                item.site_quality = "good" if item.website else "none"
            return items

        async def resolver(items, provider=None, progress_callback=None):
            items[0].website_resolved_url = "https://official-a.invalid/"
            items[0].website_resolution_status = "found_official"
            items[1].website_resolved_url = ""
            items[1].website_resolution_status = "not_found"
            items[1].site_quality = "none"
            items[2].website_resolved_url = "https://official-c.invalid/"
            items[2].website_resolution_status = "found_official"
            items[2].site_quality = "bad"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                businesses, ResolverMode.SHADOW, None, task_id=12
            )

        self.assertEqual([item.is_lead for item in businesses], [True, False, False])
        self.assertEqual([item.website_resolved_url for item in businesses], ["", "", ""])

    async def test_shadow_does_not_change_reporter_website_output(self) -> None:
        business = Business(
            name="Reporter",
            city="Test City",
            instagram_url="https://instagram.com/reporter",
            website="https://legacy.invalid/path",
        )

        async def site_check(items, progress_callback=None):
            items[0].has_site = True
            items[0].site_quality = "bad"
            return items

        async def resolver(items, provider=None, progress_callback=None):
            items[0].website_resolved_url = "https://shadow-leak.invalid/secret"
            items[0].website_resolution_status = "found_official"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business], ResolverMode.SHADOW, None, task_id=13
            )

        columns = {name: getter(business) for name, getter in reporter.COLUMNS}
        summary = reporter.format_leads_summary([business])
        self.assertEqual(columns["Website URL"], "https://legacy.invalid/path")
        self.assertIn("Website: https://legacy.invalid/path", summary)
        self.assertNotIn("shadow-leak.invalid", json.dumps(columns) + summary)

    async def test_shadow_failure_is_fail_open_and_logs_type_only(self) -> None:
        business = Business(name="Failure", instagram_url="https://instagram.com/failure")

        async def site_check(items, progress_callback=None):
            items[0].has_site = False
            items[0].site_quality = "none"
            return items

        async def resolver(items, provider=None, progress_callback=None):
            raise RuntimeError("SECRET_SHOULD_NOT_BE_LOGGED")

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
            self.assertLogs("lead_hunter.orchestrator", level="WARNING") as captured,
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business], ResolverMode.SHADOW, None, task_id=14
            )

        logged = "\n".join(captured.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("SECRET_SHOULD_NOT_BE_LOGGED", logged)
        self.assertTrue(business.is_lead)

    async def test_completed_shadow_log_contains_only_safe_outcomes(self) -> None:
        business = Business(
            name="PRIVATE_LOG_NAME",
            address="PRIVATE_LOG_ADDRESS",
            phone="+0000000000",
            instagram_url="https://instagram.com/PRIVATE_LOG_HANDLE",
        )

        async def site_check(items, progress_callback=None):
            items[0].site_quality = "none"
            return items

        async def resolver(items, provider=None, progress_callback=None):
            items[0].website_resolved_url = (
                "https://log-safe.invalid/PRIVATE_LOG_PATH?value=PRIVATE_LOG_QUERY"
            )
            items[0].website_resolution_status = "found_official"
            items[0].website_resolution_source = "web_search"
            items[0].website_resolution_error = "PRIVATE_LOG_ERROR"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
            self.assertLogs("lead_hunter.orchestrator", level="INFO") as captured,
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business], ResolverMode.SHADOW, None, task_id=16
            )

        logged = "\n".join(captured.output)
        self.assertIn("log-safe.invalid", logged)
        for sentinel in (
            "PRIVATE_LOG_NAME",
            "PRIVATE_LOG_ADDRESS",
            "PRIVATE_LOG_HANDLE",
            "PRIVATE_LOG_PATH",
            "PRIVATE_LOG_QUERY",
            "PRIVATE_LOG_ERROR",
        ):
            self.assertNotIn(sentinel, logged)

    async def test_strict_resolves_real_object_before_audit(self) -> None:
        business = Business(name="Strict", instagram_url="https://instagram.com/strict")
        events = []
        audited_urls = []

        async def resolver(items, provider=None, progress_callback=None):
            events.append("resolver")
            self.assertIs(items[0], business)
            items[0].website_resolved_url = "https://strict.invalid/"
            items[0].website_resolution_status = "found_official"
            return items

        async def site_check(items, progress_callback=None):
            events.append("site")
            audited_urls.append(items[0].effective_website_url)
            items[0].has_site = True
            items[0].site_quality = "good"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=site_check),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=resolver,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business], ResolverMode.STRICT, object(), task_id=15
            )

        self.assertEqual(events, ["resolver", "site"])
        self.assertEqual(audited_urls, ["https://strict.invalid/"])
        self.assertEqual(business.website_resolved_url, "https://strict.invalid/")


class ShadowTelemetryTests(unittest.TestCase):
    def test_safe_deterministic_aggregate_omits_identity_and_raw_details(self) -> None:
        businesses = [
            Business(
                name="SECRET_BUSINESS_NAME",
                address="SECRET_ADDRESS",
                phone="SECRET_PHONE",
                instagram_url="https://instagram.com/SECRET_HANDLE",
                website_resolved_url="https://www.safe.invalid/SECRET_PATH?token=SECRET_QUERY",
                website_resolution_status="found_official",
                website_resolution_source="web_search",
                website_resolution_evidence="SECRET_EVIDENCE",
                website_resolution_error="SECRET_RAW_ERROR",
            ),
            Business(
                name="SECRET_SECOND_NAME",
                website_resolution_status="not_found",
                website_resolution_source="maps",
            ),
            Business(
                name="SECRET_THIRD_NAME",
                website_resolution_status="not_found",
                website_resolution_source="maps",
            ),
        ]
        budget = SearchBudgetSnapshot(3, 2, 1)
        openai = OpenAIWebSearchTelemetry(
            requests_started=2,
            requests_succeeded=1,
            requests_failed=1,
            tool_calls_seen=4,
            search_actions_seen=2,
            open_page_actions_seen=1,
            find_in_page_actions_seen=1,
            unknown_actions_seen=9,
            sources_seen=5,
            identity_candidates_rejected=3,
            candidates_returned=1,
            tool_call_limit_exceeded=True,
            last_error_category="tool_call_limit",
        )

        with (
            patch.object(orchestrator, "search_budget_snapshot", return_value=budget),
            patch.object(
                orchestrator,
                "openai_web_search_telemetry_snapshot",
                return_value=openai,
            ),
        ):
            telemetry = orchestrator._shadow_resolver_telemetry(
                businesses, object(), task_id=22
            )

        self.assertEqual(
            telemetry["status_counts"],
            {"found_official": 1, "not_found": 2},
        )
        self.assertEqual(telemetry["resolved_domains"], ["safe.invalid"])
        self.assertEqual(telemetry["source_counts"], {"maps": 2, "web_search": 1})
        self.assertEqual(telemetry["provider_budget"]["max_requests"], 3)
        self.assertEqual(telemetry["openai_provider"]["requests_started"], 2)
        self.assertNotIn("unknown_actions_seen", telemetry["openai_provider"])

        rendered = json.dumps(telemetry, sort_keys=True)
        for sentinel in (
            "SECRET_BUSINESS_NAME",
            "SECRET_ADDRESS",
            "SECRET_PHONE",
            "SECRET_HANDLE",
            "SECRET_PATH",
            "SECRET_QUERY",
            "SECRET_EVIDENCE",
            "SECRET_RAW_ERROR",
            "SECRET_SECOND_NAME",
            "SECRET_THIRD_NAME",
        ):
            self.assertNotIn(sentinel, rendered)


class BudgetThreeSemanticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_fourth_search_fails_without_retrying_underlying_provider(self) -> None:
        class EmptyProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def search(self, request):
                self.calls += 1
                return ()

        underlying = EmptyProvider()
        provider = BudgetedSearchProvider(underlying, 3)
        request = SearchRequest("Synthetic Business", "Test City")

        for _ in range(3):
            self.assertEqual(await provider.search(request), ())
        with self.assertRaises(ProviderUnavailable):
            await provider.search(request)

        self.assertEqual(underlying.calls, 3)
        self.assertEqual(provider.snapshot(), SearchBudgetSnapshot(3, 3, 0))


if __name__ == "__main__":
    unittest.main()

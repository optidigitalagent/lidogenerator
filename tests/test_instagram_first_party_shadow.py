"""Isolation and safe-telemetry tests for the orchestrator shadow path."""

import json
import unittest
from unittest.mock import patch

import orchestrator
from agents import reporter
from agents.instagram_first_party_resolver import (
    FirstPartyInstagramRequestBudget,
    trusted_website_for_instagram_resolution,
)
from instagram_first_party_resolution import (
    FirstPartyInstagramEvidenceSource as Source,
    FirstPartyInstagramResolution,
    FirstPartyInstagramStatus as Status,
)
from models import Business
from website_pipeline import ResolverMode


class SyntheticFirstPartyResolver:
    def __init__(self, max_requests: int = 20, *, fail: bool = False) -> None:
        self.budget = FirstPartyInstagramRequestBudget(max_requests)
        self.fail = fail
        self.inputs = []

    async def resolve_missing(self, businesses):
        if self.fail:
            raise RuntimeError("PRIVATE_RAW_EXCEPTION")
        results = []
        for business in businesses:
            self.inputs.append(business)
            if business.instagram_url:
                results.append(
                    FirstPartyInstagramResolution(Status.SKIPPED, None, None, (), 0, 0)
                )
                continue
            if trusted_website_for_instagram_resolution(business) is None:
                results.append(
                    FirstPartyInstagramResolution(Status.SKIPPED, None, None, (), 0, 0)
                )
                continue
            try:
                await self.budget.claim()
            except RuntimeError:
                results.append(
                    FirstPartyInstagramResolution(Status.SKIPPED, None, None, (), 0, 0)
                )
                continue
            business.instagram_url = "https://www.instagram.com/synthetic_brand/"
            results.append(
                FirstPartyInstagramResolution(
                    Status.FOUND_OFFICIAL,
                    "https://www.instagram.com/synthetic_brand/",
                    "synthetic_brand",
                    (Source.HTML_ANCHOR,),
                    1,
                    1,
                )
            )
        return tuple(results)


async def mark_bad_site(items, progress_callback=None):
    for item in items:
        item.has_site = True
        item.site_quality = "bad"
        item.website_final_url = item.website
    return items


class FirstPartyInstagramShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_mode_off_does_no_first_party_work(self) -> None:
        business = Business(website="https://official.example/")
        runtime = SyntheticFirstPartyResolver()
        with patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.OFF,
                None,
                task_id=30,
                first_party_mode="off",
                first_party_resolver_runtime=runtime,
            )
        self.assertEqual(runtime.inputs, [])
        self.assertEqual(business.instagram_url, "")

    async def test_shadow_copy_gets_profile_real_lead_decision_stays_legacy(self) -> None:
        business = Business(
            name="Synthetic Real",
            website="https://official.example/",
        )
        runtime = SyntheticFirstPartyResolver()

        async def website_shadow(items, provider=None, progress_callback=None):
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=website_shadow,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.SHADOW,
                None,
                task_id=31,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(len(runtime.inputs), 1)
        self.assertIsNot(runtime.inputs[0], business)
        self.assertEqual(
            runtime.inputs[0].instagram_url,
            "https://www.instagram.com/synthetic_brand/",
        )
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)
        skipped_no_instagram = sum(1 for item in [business] if not item.instagram_url)
        self.assertEqual(skipped_no_instagram, 1)

    async def test_website_mode_off_ignores_resolver_only_site_state(self) -> None:
        business = Business(
            website_resolution_status="found_official",
            website_resolved_url="https://resolver-only.example/",
        )
        runtime = SyntheticFirstPartyResolver()

        async def no_maps_site(items, progress_callback=None):
            items[0].has_site = False
            items[0].site_quality = "none"
            return items

        with patch.object(orchestrator.site_checker, "check_sites", new=no_maps_site):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.OFF,
                None,
                task_id=38,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(runtime.budget.snapshot().used_requests, 0)
        self.assertEqual(runtime.inputs[0].website_resolution_status, "")
        self.assertEqual(runtime.inputs[0].website_resolved_url, "")
        self.assertEqual(business.instagram_url, "")

    async def test_website_shadow_and_first_party_use_the_same_copy(self) -> None:
        business = Business(name="No Maps Site")
        website_copy_ids = []
        runtime = SyntheticFirstPartyResolver()

        async def no_site(items, progress_callback=None):
            items[0].has_site = False
            items[0].site_quality = "none"
            return items

        async def website_shadow(items, provider=None, progress_callback=None):
            website_copy_ids.append(id(items[0]))
            items[0].website_resolution_status = "found_official"
            items[0].website_resolved_url = "https://discovered.example/"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=no_site),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=website_shadow,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.SHADOW,
                None,
                task_id=32,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(website_copy_ids, [id(runtime.inputs[0])])
        self.assertEqual(
            runtime.inputs[0].instagram_url,
            "https://www.instagram.com/synthetic_brand/",
        )
        self.assertEqual(business.website_resolved_url, "")
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)

    async def test_website_strict_still_runs_first_party_on_a_copy(self) -> None:
        business = Business(name="Strict Real")
        runtime = SyntheticFirstPartyResolver()

        async def strict_website(items, provider=None, progress_callback=None):
            items[0].website_resolution_status = "found_official"
            items[0].website_resolved_url = "https://strict.example/"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=strict_website,
            ),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.STRICT,
                None,
                task_id=39,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(business.website_resolved_url, "https://strict.example/")
        self.assertEqual(business.instagram_url, "")
        self.assertIsNot(runtime.inputs[0], business)
        self.assertTrue(runtime.inputs[0].instagram_url.endswith("synthetic_brand/"))

    async def test_failure_is_fail_open_and_logs_exception_type_only(self) -> None:
        business = Business(website="https://official.example/")
        runtime = SyntheticFirstPartyResolver(fail=True)
        with (
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            self.assertLogs("lead_hunter.orchestrator", level="WARNING") as captured,
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.OFF,
                None,
                task_id=33,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )
        logged = "\n".join(captured.output)
        self.assertIn("instagram_first_party_shadow_failed", logged)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("PRIVATE_RAW_EXCEPTION", logged)
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)

    async def test_budget_is_shared_across_batches(self) -> None:
        runtime = SyntheticFirstPartyResolver(max_requests=1)
        first = Business(website="https://first.example/")
        second = Business(website="https://second.example/")
        with patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site):
            for task_id, business in ((34, first), (35, second)):
                await orchestrator._check_batch_websites_with_resolver_mode(
                    [business],
                    ResolverMode.OFF,
                    None,
                    task_id=task_id,
                    first_party_mode="shadow",
                    first_party_resolver_runtime=runtime,
                )
        self.assertEqual(runtime.budget.snapshot().used_requests, 1)
        self.assertEqual(first.instagram_url, "")
        self.assertEqual(second.instagram_url, "")
        self.assertEqual(runtime.inputs[0].instagram_url.endswith("synthetic_brand/"), True)
        self.assertEqual(runtime.inputs[1].instagram_url, "")

    async def test_telemetry_is_aggregate_only_and_identity_free(self) -> None:
        business = Business(
            name="PRIVATE_BUSINESS_NAME",
            address="PRIVATE_ADDRESS",
            phone="PRIVATE_PHONE",
            website="https://PRIVATE-HOST.example/PRIVATE_PATH",
        )
        runtime = SyntheticFirstPartyResolver()
        with (
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            self.assertLogs("lead_hunter.orchestrator", level="INFO") as captured,
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.OFF,
                None,
                task_id=36,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )
        logged = "\n".join(captured.output)
        self.assertIn("instagram_first_party_shadow", logged)
        for sentinel in (
            "PRIVATE_BUSINESS_NAME",
            "PRIVATE_ADDRESS",
            "PRIVATE_PHONE",
            "PRIVATE-HOST",
            "PRIVATE_PATH",
            "synthetic_brand",
        ):
            self.assertNotIn(sentinel, logged)
        payload = json.loads(logged.split("instagram_first_party_shadow ", 1)[1])
        self.assertEqual(payload["found_official_count"], 1)
        self.assertEqual(payload["requests"], {"max": 20, "remaining": 19, "used": 1})
        self.assertEqual(payload["evidence_source_counts"]["html_anchor"], 1)

    async def test_shadow_profile_cannot_reach_db_reporter_telegram_or_opti(self) -> None:
        business = Business(
            name="Downstream Real",
            website="https://official.example/",
        )
        runtime = SyntheticFirstPartyResolver()
        with (
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            patch.object(orchestrator.db, "save_businesses") as save_businesses,
            patch.object(orchestrator.reporter, "export_csv") as export_csv,
            patch.object(orchestrator.reporter, "export_excel") as export_excel,
            patch.object(orchestrator, "finalize_completed_task") as finalize,
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.OFF,
                None,
                task_id=37,
                first_party_mode="shadow",
                first_party_resolver_runtime=runtime,
            )

        save_businesses.assert_not_called()
        export_csv.assert_not_called()
        export_excel.assert_not_called()
        finalize.assert_not_called()
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)
        columns = {name: getter(business) for name, getter in reporter.COLUMNS}
        telegram_summary = reporter.format_leads_summary([business])
        self.assertEqual(columns["Instagram URL"], "")
        self.assertNotIn("synthetic_brand", json.dumps(columns) + telegram_summary)
        self.assertNotIn("synthetic_brand", json.dumps(business.to_dict()))


if __name__ == "__main__":
    unittest.main()

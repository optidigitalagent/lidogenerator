"""Controlled real-apply tests for deterministic first-party Instagram recovery."""

import json
import unittest
from unittest.mock import Mock, patch

import orchestrator
from agents.instagram_first_party_resolver import FirstPartyInstagramRequestBudget
from instagram_first_party_resolution import (
    FirstPartyInstagramEvidenceSource as Source,
    FirstPartyInstagramResolution,
    FirstPartyInstagramStatus as Status,
)
from models import Business
from website_pipeline import ResolverMode


CANONICAL_PROFILE = "https://www.instagram.com/synthetic_apply/"


class SyntheticApplyResolver:
    def __init__(
        self,
        status: Status = Status.FOUND_OFFICIAL,
        *,
        max_requests: int = 20,
        fail: bool = False,
    ) -> None:
        self.status = status
        self.budget = FirstPartyInstagramRequestBudget(max_requests)
        self.fail = fail
        self.inputs: list[Business] = []

    async def resolve_missing(
        self, businesses: list[Business]
    ) -> tuple[FirstPartyInstagramResolution, ...]:
        if self.fail:
            raise RuntimeError("PRIVATE_RAW_APPLY_ERROR")
        results = []
        for business in businesses:
            self.inputs.append(business)
            try:
                await self.budget.claim()
            except RuntimeError:
                results.append(
                    FirstPartyInstagramResolution(Status.SKIPPED, None, None, (), 0, 0)
                )
                continue
            if self.status is Status.FOUND_OFFICIAL:
                business.instagram_url = CANONICAL_PROFILE
                results.append(
                    FirstPartyInstagramResolution(
                        Status.FOUND_OFFICIAL,
                        CANONICAL_PROFILE,
                        "synthetic_apply",
                        (Source.HTML_ANCHOR,),
                        1,
                        1,
                    )
                )
            elif self.status is Status.TECHNICAL_ERROR:
                results.append(
                    FirstPartyInstagramResolution(
                        Status.TECHNICAL_ERROR,
                        None,
                        None,
                        (),
                        1,
                        0,
                        "synthetic_error",
                    )
                )
            else:
                results.append(
                    FirstPartyInstagramResolution(
                        self.status,
                        None,
                        None,
                        (),
                        1,
                        1,
                    )
                )
        return tuple(results)


def bad_site_business(**changes) -> Business:
    values = {
        "name": "Synthetic Apply",
        "website": "https://official.example/",
        "website_final_url": "https://official.example/",
        "has_site": True,
        "site_quality": "bad",
    }
    values.update(changes)
    return Business(**values)


async def mark_bad_site(items, progress_callback=None):
    for item in items:
        item.has_site = True
        item.site_quality = "bad"
        item.website_final_url = item.website
    return items


class FirstPartyInstagramApplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_site_found_applies_and_unlocks_legacy_lead(self) -> None:
        business = bad_site_business()
        runtime = SyntheticApplyResolver()
        with self.assertLogs("lead_hunter.orchestrator", level="INFO") as captured:
            await orchestrator._run_first_party_instagram_apply(
                [business], runtime, task_id=101
            )

        payload = json.loads(
            next(
                line.split("instagram_first_party_apply ", 1)[1]
                for line in captured.output
                if "instagram_first_party_apply " in line
            )
        )
        self.assertEqual(business.instagram_url, CANONICAL_PROFILE)
        self.assertTrue(business.is_lead)
        self.assertEqual(payload["applied_count"], 1)
        self.assertEqual(payload["legacy_lead_eligible_after_apply"], 1)
        self.assertEqual(payload["found_official_count"], 1)

    async def test_only_missing_instagram_real_bad_trusted_site_is_eligible(self) -> None:
        cases = {
            "good_site": bad_site_business(site_quality="good"),
            "no_website": Business(name="No Site"),
            "uncertain_site": bad_site_business(site_quality="uncertain"),
            "technical_site": bad_site_business(site_quality="technical_error"),
            "existing_instagram": bad_site_business(
                instagram_url="https://www.instagram.com/already_there/"
            ),
            "untrusted_bad_site": bad_site_business(
                website="https://instagram.com/not-an-own-site/",
                website_final_url="https://instagram.com/not-an-own-site/",
            ),
        }
        for label, business in cases.items():
            with self.subTest(label=label):
                runtime = SyntheticApplyResolver()
                original_instagram = business.instagram_url
                self.assertFalse(orchestrator._first_party_apply_eligible(business))
                with self.assertLogs("lead_hunter.orchestrator", level="INFO"):
                    await orchestrator._run_first_party_instagram_apply(
                        [business], runtime, task_id=102
                    )
                self.assertEqual(runtime.inputs, [])
                self.assertEqual(runtime.budget.snapshot().used_requests, 0)
                self.assertEqual(business.instagram_url, original_instagram)

    async def test_non_found_outcomes_do_not_create_a_lead(self) -> None:
        for status in (
            Status.NOT_FOUND,
            Status.UNCERTAIN,
            Status.TECHNICAL_ERROR,
        ):
            with self.subTest(status=status):
                business = bad_site_business()
                runtime = SyntheticApplyResolver(status)
                with self.assertLogs("lead_hunter.orchestrator", level="INFO"):
                    await orchestrator._run_first_party_instagram_apply(
                        [business], runtime, task_id=103
                    )
                self.assertEqual(business.instagram_url, "")
                self.assertFalse(business.is_lead)

    async def test_budget_exhaustion_is_safe_and_does_not_invent_a_profile(self) -> None:
        business = bad_site_business()
        runtime = SyntheticApplyResolver(max_requests=0)
        with self.assertLogs("lead_hunter.orchestrator", level="INFO") as captured:
            await orchestrator._run_first_party_instagram_apply(
                [business], runtime, task_id=104
            )
        payload = json.loads(
            next(
                line.split("instagram_first_party_apply ", 1)[1]
                for line in captured.output
                if "instagram_first_party_apply " in line
            )
        )
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)
        self.assertEqual(payload["skipped_count"], 1)
        self.assertEqual(payload["requests"], {"max": 0, "remaining": 0, "used": 0})

    async def test_unexpected_batch_error_fails_open_and_logs_type_only(self) -> None:
        business = bad_site_business()
        runtime = SyntheticApplyResolver(fail=True)
        with self.assertLogs("lead_hunter.orchestrator", level="WARNING") as captured:
            await orchestrator._run_first_party_instagram_apply(
                [business], runtime, task_id=105
            )
        logged = "\n".join(captured.output)
        self.assertIn("instagram_first_party_apply_failed", logged)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn("PRIVATE_RAW_APPLY_ERROR", logged)
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)

    async def test_apply_telemetry_is_identity_free_and_schema_bounded(self) -> None:
        business = bad_site_business(
            name="PRIVATE_NAME",
            phone="PRIVATE_PHONE",
            address="PRIVATE_ADDRESS",
            website="https://private-host.example/private-path",
            website_final_url="https://private-host.example/private-path",
        )
        runtime = SyntheticApplyResolver()
        with self.assertLogs("lead_hunter.orchestrator", level="INFO") as captured:
            await orchestrator._run_first_party_instagram_apply(
                [business], runtime, task_id=106
            )
        logged = "\n".join(captured.output)
        for sentinel in (
            "PRIVATE_NAME",
            "PRIVATE_PHONE",
            "PRIVATE_ADDRESS",
            "private-host",
            "private-path",
            "synthetic_apply",
        ):
            self.assertNotIn(sentinel, logged)
        payload = json.loads(logged.split("instagram_first_party_apply ", 1)[1])
        self.assertEqual(
            set(payload),
            {
                "event",
                "task_id",
                "batch_candidate_count",
                "missing_instagram_before",
                "bad_site_missing_instagram_count",
                "eligible_businesses",
                "attempted_businesses",
                "found_official_count",
                "applied_count",
                "legacy_lead_eligible_after_apply",
                "not_found_count",
                "uncertain_count",
                "technical_error_count",
                "skipped_count",
                "requests",
                "pages_attempted",
                "pages_succeeded",
                "evidence_source_counts",
            },
        )

    async def test_website_shadow_cannot_unlock_real_apply(self) -> None:
        business = Business(name="Real Without Site")
        runtime = SyntheticApplyResolver()

        async def no_real_site(items, progress_callback=None):
            for item in items:
                item.has_site = False
                item.site_quality = "none"
            return items

        async def shadow_finds_site(items, provider=None, progress_callback=None):
            items[0].website_resolution_status = "found_official"
            items[0].website_resolved_url = "https://shadow-only.example/"
            items[0].site_quality = "bad"
            items[0].has_site = True
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=no_real_site),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=shadow_finds_site,
            ),
            self.assertLogs("lead_hunter.orchestrator", level="INFO"),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.SHADOW,
                None,
                task_id=107,
                first_party_mode="apply",
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(runtime.inputs, [])
        self.assertEqual(business.website_resolved_url, "")
        self.assertEqual(business.instagram_url, "")
        self.assertFalse(business.is_lead)

    async def test_real_bad_site_applies_while_website_shadow_stays_isolated(self) -> None:
        business = Business(
            name="Real Bad Site",
            website="https://real-official.example/",
        )
        runtime = SyntheticApplyResolver()

        async def shadow_changes_copy(items, provider=None, progress_callback=None):
            items[0].website = "https://shadow-replacement.example/"
            items[0].website_resolution_status = "found_official"
            items[0].website_resolved_url = "https://shadow-replacement.example/"
            return items

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=shadow_changes_copy,
            ),
            self.assertLogs("lead_hunter.orchestrator", level="INFO"),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [business],
                ResolverMode.SHADOW,
                None,
                task_id=108,
                first_party_mode="apply",
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(runtime.inputs, [business])
        self.assertEqual(business.website, "https://real-official.example/")
        self.assertEqual(business.website_resolved_url, "")
        self.assertEqual(business.instagram_url, CANONICAL_PROFILE)
        self.assertTrue(business.is_lead)

    async def test_full_run_promotes_recovered_candidate_through_normal_boundary(self) -> None:
        candidate = Business(
            name="Boundary Candidate",
            website="https://boundary-official.example/",
        )
        runtime = SyntheticApplyResolver()
        progress_messages: list[str] = []
        saved_batches: list[list[Business]] = []
        reporter_batches: list[list[Business]] = []
        social_batches: list[list[Business]] = []
        opti_seen: list[list[Business]] = []
        website_shadow_inputs: list[Business] = []
        events: list[str] = []

        async def collect_once(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            stop_flag=None,
            **kwargs,
        ):
            if progress_callback:
                await progress_callback(1)
            yield [candidate]

        async def website_shadow(items, provider=None, progress_callback=None):
            website_shadow_inputs.extend(items)
            items[0].website_resolved_url = "https://shadow-only.example/"
            return items

        async def social(items, progress_callback=None, stop_flag=None):
            events.append("social")
            social_batches.append(list(items))
            return items

        async def score(items, progress_callback=None):
            events.append("score")
            return items

        def save(items):
            events.append("save")
            saved_batches.append(list(items))
            return len(items)

        def export(items, *args, **kwargs):
            events.append("reporter")
            reporter_batches.append(list(items))
            return "synthetic.xlsx"

        async def finalize(task_id):
            events.append("opti")
            opti_seen.append(list(saved_batches[-1]))
            return ""

        async def progress(message):
            progress_messages.append(message)

        task = {"niche": "synthetic", "city": "synthetic", "count": 1}
        with (
            patch.object(orchestrator.config, "INSTAGRAM_FIRST_PARTY_MODE", "apply"),
            patch.object(orchestrator.config, "WEBSITE_RESOLVER_MODE", "shadow"),
            patch.object(orchestrator.config, "MAX_CHECKED_CANDIDATES_PER_TASK", 2),
            patch.object(orchestrator.config, "MAX_MAPS_CARDS_PER_TASK", 2),
            patch.object(orchestrator.db, "get_task", return_value=task),
            patch.object(orchestrator.db, "update_task_status", Mock()),
            patch.object(orchestrator.db, "save_businesses", side_effect=save),
            patch.object(orchestrator.db, "update_business", Mock()),
            patch.object(orchestrator.collector, "collect_stream", new=collect_once),
            patch.object(orchestrator.site_checker, "check_sites", new=mark_bad_site),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=website_shadow,
            ),
            patch.object(orchestrator.social_checker, "check_instagram", new=social),
            patch.object(orchestrator.ai_scorer, "score_businesses", new=score),
            patch.object(orchestrator.reporter, "export_csv", side_effect=export),
            patch.object(orchestrator.reporter, "export_excel", side_effect=export),
            patch.object(
                orchestrator.reporter,
                "format_leads_summary",
                return_value="synthetic summary",
            ),
            patch.object(orchestrator, "finalize_completed_task", new=finalize),
        ):
            result = await orchestrator.run_search(
                109,
                progress_callback=progress,
                progress_interval=0,
                website_search_provider=object(),
                first_party_resolver_runtime=runtime,
            )

        self.assertEqual(result, "synthetic.xlsx")
        self.assertIsNot(website_shadow_inputs[0], candidate)
        self.assertEqual(candidate.website_resolved_url, "")
        self.assertEqual(candidate.instagram_url, CANONICAL_PROFILE)
        self.assertTrue(candidate.is_lead)
        for batches in (saved_batches, reporter_batches, social_batches, opti_seen):
            self.assertTrue(batches)
            self.assertEqual(batches[0], [candidate])
            self.assertEqual(batches[0][0].instagram_url, CANONICAL_PROFILE)
        self.assertLess(events.index("save"), events.index("social"))
        self.assertLess(events.index("reporter"), events.index("opti"))
        final_message = next(
            message
            for message in reversed(progress_messages)
            if message.startswith("✅ Готово!")
        )
        self.assertIn("Відновлено Instagram з офіційного сайту: 1", final_message)
        self.assertIn("Пропущено без Instagram: 0", final_message)
        self.assertIn("Усього лідів у таблиці: 1", final_message)


if __name__ == "__main__":
    unittest.main()

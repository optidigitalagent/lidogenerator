"""Isolated integration tests for the orchestrator search policy."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import orchestrator
from models import Business
from search_policy import StopReason
from website_pipeline import LeadDecision
from website_resolution import (
    CandidateEvidence,
    CandidateKind,
    CandidateSource,
    ResolutionStatus,
    WebsiteResolution,
)


def _lead(name: str) -> Business:
    return Business(name=name, instagram_url=f"https://instagram.com/{name}")


def _good_site(name: str) -> Business:
    return Business(
        name=name,
        instagram_url=f"https://instagram.com/{name}",
        has_site=True,
        site_quality="good",
    )


def _no_instagram(name: str) -> Business:
    return Business(name=name)


class OrchestratorPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        target: int,
        max_candidates: int,
        batches: list[list[Business]],
        stop_event: asyncio.Event | None = None,
        stop_after_site_check: bool = False,
        resolver_mode: str = "shadow",
        website_search_provider=None,
        resolver_setup=None,
        site_setup=None,
    ) -> dict:
        yielded_batches: list[list[Business]] = []
        checked_batches: list[list[Business]] = []
        downstream: dict[str, list[list[Business]]] = {
            "save": [],
            "social": [],
            "score": [],
            "csv": [],
            "excel": [],
            "summary": [],
        }
        collector_limits: list[int] = []
        decisions = []
        statuses = Mock()
        opti_finalizations = []
        visited = 0
        resolver_calls = []
        events = []

        async def fake_collect_stream(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            stop_flag=None,
            **kwargs,
        ):
            nonlocal visited
            collector_limits.append(max_businesses)
            for batch in batches:
                if stop_flag and stop_flag():
                    return
                for _ in batch:
                    visited += 1
                    if progress_callback:
                        await progress_callback(visited)
                yielded_batches.append(list(batch))
                yield batch

        async def fake_check_sites(items, progress_callback=None):
            events.append("site")
            checked_batches.append(list(items))
            if site_setup:
                for item in items:
                    site_setup(item)
            if stop_after_site_check and stop_event is not None:
                stop_event.set()
            return items

        async def fake_resolve(items, provider=None, progress_callback=None):
            events.append("resolver")
            resolver_calls.append((list(items), provider))
            if resolver_setup:
                for item in items:
                    resolver_setup(item)
            return items

        async def fake_social(items, progress_callback=None, stop_flag=None):
            downstream["social"].append(list(items))
            return items

        async def fake_score(items, progress_callback=None):
            downstream["score"].append(list(items))
            return items

        def capture(name):
            def inner(items, *args, **kwargs):
                downstream[name].append(list(items))
                if name == "csv":
                    return "result.csv"
                if name == "excel":
                    return "result.xlsx"
                if name == "summary":
                    return "summary"
                return len(items)

            return inner

        async def fake_finalize(task_id):
            opti_finalizations.append(
                {
                    "task_id": task_id,
                    "csv": list(downstream["csv"]),
                    "excel": list(downstream["excel"]),
                    "statuses": list(statuses.call_args_list),
                }
            )
            return ""

        real_decide_next = orchestrator.decide_next

        def record_decision(progress, policy):
            decision = real_decide_next(progress, policy)
            decisions.append((progress, decision))
            return decision

        task = {"niche": "test", "city": "test", "count": target}
        with (
            patch.object(
                orchestrator.config,
                "MAX_CHECKED_CANDIDATES_PER_TASK",
                max_candidates,
            ),
            patch.object(
                orchestrator.config,
                "MAX_MAPS_CARDS_PER_TASK",
                max_candidates,
            ),
            patch.object(orchestrator.config, "WEBSITE_RESOLVER_MODE", resolver_mode),
            patch.object(orchestrator.db, "get_task", return_value=task),
            patch.object(orchestrator.db, "update_task_status", statuses),
            patch.object(orchestrator.db, "save_businesses", side_effect=capture("save")),
            patch.object(orchestrator.db, "update_business"),
            patch.object(orchestrator.collector, "collect_stream", new=fake_collect_stream),
            patch.object(
                orchestrator.website_resolver,
                "resolve_business_websites",
                new=fake_resolve,
            ),
            patch.object(orchestrator.site_checker, "check_sites", new=fake_check_sites),
            patch.object(orchestrator.social_checker, "check_instagram", new=fake_social),
            patch.object(orchestrator.ai_scorer, "score_businesses", new=fake_score),
            patch.object(orchestrator.reporter, "export_csv", side_effect=capture("csv")),
            patch.object(orchestrator.reporter, "export_excel", side_effect=capture("excel")),
            patch.object(orchestrator.reporter, "format_leads_summary", side_effect=capture("summary")),
            patch.object(orchestrator, "finalize_completed_task", new=fake_finalize),
            patch.object(orchestrator, "decide_next", side_effect=record_decision),
        ):
            result = await orchestrator.run_search(
                1,
                progress_callback=AsyncMock(),
                stop_event=stop_event,
                progress_interval=0,
                website_search_provider=website_search_provider,
            )

        return {
            "result": result,
            "yielded_batches": yielded_batches,
            "checked_batches": checked_batches,
            "downstream": downstream,
            "collector_limits": collector_limits,
            "decisions": decisions,
            "statuses": statuses.call_args_list,
            "resolver_calls": resolver_calls,
            "events": events,
            "opti_finalizations": opti_finalizations,
        }

    async def test_resolver_mode_off_skips_resolver(self) -> None:
        run = await self._run(
            target=1,
            max_candidates=2,
            batches=[[_lead("one")]],
            resolver_mode="off",
        )
        self.assertEqual(run["resolver_calls"], [])
        self.assertEqual(run["events"], ["site"])

    async def test_shadow_audits_before_isolated_resolution_and_passes_provider(self) -> None:
        provider = object()
        business = _lead("one")
        run = await self._run(
            target=1,
            max_candidates=2,
            batches=[[business]],
            resolver_mode="shadow",
            website_search_provider=provider,
            resolver_setup=lambda item: setattr(item, "website_resolution_status", "uncertain"),
        )
        self.assertEqual(run["events"][:2], ["site", "resolver"])
        self.assertIs(run["resolver_calls"][0][1], provider)
        self.assertIsNot(run["resolver_calls"][0][0][0], business)
        self.assertEqual(business.lead_decision, "")

    async def test_completed_shadow_finalizes_opti_after_legacy_output(self) -> None:
        run = await self._run(
            target=1,
            max_candidates=2,
            batches=[[_lead("one")]],
            resolver_mode="shadow",
        )

        self.assertEqual(len(run["opti_finalizations"]), 1)
        finalization = run["opti_finalizations"][0]
        self.assertEqual(finalization["task_id"], 1)
        self.assertEqual([item.name for item in finalization["csv"][0]], ["one"])
        self.assertEqual([item.name for item in finalization["excel"][0]], ["one"])
        self.assertIn(
            unittest.mock.call(1, "done", csv_path="result.xlsx"),
            finalization["statuses"],
        )

    async def test_strict_qualification_matrix_is_fail_closed(self) -> None:
        official = CandidateEvidence(
            CandidateSource.MAPS,
            "https://example.com/",
            "https://example.com/",
            "example.com",
            None,
            CandidateKind.OFFICIAL_WEBSITE,
            ("phone_exact",),
            confidence=0.75,
        )

        def resolver_for(status):
            def setup(item):
                if status is ResolutionStatus.FOUND_OFFICIAL:
                    result = WebsiteResolution(
                        status, official.normalized_url, official.source, 0.75, (official,)
                    )
                elif status is ResolutionStatus.RESOLUTION_ERROR:
                    result = WebsiteResolution(status, None, None, 0.0, (), "provider error")
                else:
                    result = WebsiteResolution(status, None, None, 0.0, ())
                orchestrator.website_resolver.apply_resolution(item, result)
            return setup

        def site_for(audit_status):
            def setup(item):
                item.website_audit_status = audit_status
                item.website_audit_evidence = "[]"
                item.website_final_url = "https://example.com/" if audit_status in {"good", "bad"} else ""
                item.website_audit_http_status = 200 if audit_status in {"good", "bad"} else None
                item.site_quality = audit_status if audit_status in {"good", "bad"} else "none"
            return setup

        rows = (
            (ResolutionStatus.FOUND_OFFICIAL, "good", LeadDecision.NOT_LEAD),
            (ResolutionStatus.FOUND_OFFICIAL, "bad", LeadDecision.LEAD),
            (ResolutionStatus.NOT_FOUND, "no_official_site", LeadDecision.LEAD),
            (ResolutionStatus.UNCERTAIN, "no_official_site", LeadDecision.UNCERTAIN),
            (ResolutionStatus.RESOLUTION_ERROR, "no_official_site", LeadDecision.UNCERTAIN),
        )
        for index, (status, audit_status, decision) in enumerate(rows):
            with self.subTest(status=status, audit=audit_status):
                business = Business(
                    name=f"strict-{index}", city="Kyiv",
                    instagram_url=f"https://instagram.com/strict{index}",
                )
                await self._run(
                    target=1,
                    max_candidates=1,
                    batches=[[business]],
                    resolver_mode="strict",
                    resolver_setup=resolver_for(status),
                    site_setup=site_for(audit_status),
                )
                self.assertEqual(business.lead_decision, decision.value)

    async def test_continues_to_next_batch_when_target_is_not_reached(self) -> None:
        run = await self._run(
            target=1,
            max_candidates=3,
            batches=[[_no_instagram("skip")], [_lead("lead")], [_lead("unused")]],
        )

        self.assertEqual(len(run["yielded_batches"]), 2)
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.TARGET_REACHED)

    async def test_target_reached_stops_before_third_batch_and_preserves_order(self) -> None:
        run = await self._run(
            target=3,
            max_candidates=10,
            batches=[[_lead("one"), _lead("two")], [_lead("three"), _lead("four")], [_lead("five")]],
        )

        self.assertEqual(len(run["yielded_batches"]), 2)
        self.assertEqual(
            [item.name for item in run["downstream"]["social"][0]],
            ["one", "two", "three"],
        )
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.TARGET_REACHED)

    async def test_user_stop_has_priority_over_target_and_candidate_limit(self) -> None:
        stop_event = asyncio.Event()
        run = await self._run(
            target=2,
            max_candidates=3,
            batches=[[_lead("one"), _lead("two"), _lead("three")], [_lead("unused")]],
            stop_event=stop_event,
            stop_after_site_check=True,
        )

        self.assertIsNone(run["result"])
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.USER_STOPPED)
        self.assertEqual(len(run["yielded_batches"]), 1)
        self.assertEqual(
            [item.name for item in run["downstream"]["save"][0]],
            ["one", "two"],
        )
        self.assertIn(unittest.mock.call(1, "stopped"), run["statuses"])
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_max_candidates_stops_and_exports_partial_result(self) -> None:
        run = await self._run(
            target=5,
            max_candidates=5,
            batches=[
                [_lead("one"), _good_site("two"), _no_instagram("three"), _lead("four"), _good_site("five")],
                [_lead("unused")],
            ],
        )

        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual(len(run["yielded_batches"]), 1)
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.MAX_CANDIDATES_REACHED)
        self.assertEqual([item.name for item in run["downstream"]["csv"][0]], ["one", "four"])
        self.assertIn(unittest.mock.call(1, "done", csv_path="result.xlsx"), run["statuses"])

    async def test_exhausted_stream_exports_available_leads(self) -> None:
        run = await self._run(
            target=3,
            max_candidates=5,
            batches=[[_lead("only")]],
        )

        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual(run["decisions"][-1][0].remaining_queries, 0)
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.QUERIES_EXHAUSTED)
        self.assertEqual([item.name for item in run["downstream"]["save"][0]], ["only"])

    async def test_good_site_candidates_count_toward_checked_limit(self) -> None:
        run = await self._run(
            target=4,
            max_candidates=4,
            batches=[[_good_site("one"), _good_site("two"), _good_site("three"), _good_site("four")]],
        )

        progress, decision = run["decisions"][-1]
        self.assertEqual(progress.checked_candidates, 4)
        self.assertEqual(progress.qualified_leads, 0)
        self.assertEqual(decision.stop_reason, StopReason.MAX_CANDIDATES_REACHED)

    async def test_all_downstream_consumers_receive_no_more_than_target(self) -> None:
        run = await self._run(
            target=2,
            max_candidates=5,
            batches=[[_lead("one"), _lead("two"), _lead("three")]],
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            with self.subTest(consumer=name):
                self.assertEqual(len(run["downstream"][name][0]), 2)
        self.assertEqual(run["collector_limits"], [5])

    async def test_existing_search_stopped_behavior_returns_none_without_error(self) -> None:
        stop_event = asyncio.Event()
        stop_event.set()
        run = await self._run(
            target=1,
            max_candidates=1,
            batches=[[_lead("unused")]],
            stop_event=stop_event,
        )

        self.assertIsNone(run["result"])
        self.assertEqual(run["yielded_batches"], [])
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.USER_STOPPED)
        self.assertIn(unittest.mock.call(1, "stopped"), run["statuses"])
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])


if __name__ == "__main__":
    unittest.main()

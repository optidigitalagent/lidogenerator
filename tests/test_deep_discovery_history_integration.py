"""Synthetic end-to-end deep discovery, link dedupe, and durable-history test."""

from contextlib import ExitStack
import unittest
from unittest.mock import AsyncMock, patch

import orchestrator
from candidate_history import CandidateClaimResult
from models import Business
from query_planner import QueryKind, QueryPlan, QueryQueue, SearchQuery
from website_presence import (
    WebsitePresenceResult,
    WebsitePresenceSource,
    WebsitePresenceStatus,
)


def _query(text: str, kind: QueryKind) -> SearchQuery:
    return SearchQuery(
        text=text,
        niche="dentistry",
        city="Kyiv",
        kind=kind,
        variant="variant" if kind is QueryKind.DISTRICT_VARIANT else None,
        district="district" if kind is QueryKind.DISTRICT_VARIANT else None,
    )


PLAN = QueryPlan(
    normal_queries=QueryQueue(
        (
            _query("normal-a", QueryKind.BASE),
            _query("normal-b", QueryKind.NICHE_VARIANT),
        )
    ),
    deep_queries=QueryQueue((_query("deep", QueryKind.DISTRICT_VARIANT),)),
)

LINKS = {
    "normal-a": (("link-1", "1"), ("link-2", "2"), ("link-3", "3")),
    "normal-b": (("link-2", "2"), ("link-3", "3"), ("link-4", "4")),
    "deep": (
        ("link-3", "3"),
        ("link-4", "4"),
        ("link-5", "5"),
        ("link-6", "6"),
    ),
}


def _business(name: str) -> Business:
    instagram = f"https://instagram.com/business_{name}" if name in {"5", "6"} else ""
    return Business(
        name=name,
        address=f"Address {name}",
        google_maps_url=f"https://google.com/maps/place/{name}",
        instagram_url=instagram,
    )


class DeepDiscoveryHistoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_preopen_dedupe_and_durable_history_remain_authoritative(self) -> None:
        durable_checked = {"1", "2"}
        active_claims: dict[str, int] = {}
        opened_by_task: dict[int, list[str]] = {1: [], 2: []}
        progress_by_task: dict[int, list[dict]] = {1: [], 2: []}
        provider_checks: list[str] = []
        exported_by_task: dict[int, list[str]] = {}
        current_task_id = 0

        async def fake_collect_stream(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            query_text=None,
            discovery_session=None,
            stop_flag=None,
            **kwargs,
        ):
            opened_in_stream = 0
            batch: list[Business] = []
            for href, name in LINKS[query_text]:
                if stop_flag and stop_flag():
                    return
                if opened_in_stream >= max_businesses:
                    break
                if not discovery_session.claim_link(href):
                    continue
                discovery_session.record_card_opened()
                opened_in_stream += 1
                opened_by_task[current_task_id].append(href)
                if progress_callback:
                    await progress_callback(opened_in_stream)
                batch.append(_business(name))
            if batch:
                yield batch

        def claim_candidate(scope, key, basis, task_id):
            if key in durable_checked:
                return CandidateClaimResult.ALREADY_CHECKED
            owner = active_claims.get(key)
            if owner is not None and owner != task_id:
                return CandidateClaimResult.CLAIMED_BY_OTHER_TASK
            active_claims[key] = task_id
            return CandidateClaimResult.CLAIMED

        def mark_checked(scope, key, task_id, outcome):
            if active_claims.get(key) != task_id:
                return False
            active_claims.pop(key)
            durable_checked.add(key)
            return True

        def release_claim(scope, key, task_id):
            if active_claims.get(key) != task_id:
                return False
            active_claims.pop(key)
            return True

        def release_unfinished(task_id):
            keys = [key for key, owner in active_claims.items() if owner == task_id]
            for key in keys:
                active_claims.pop(key)
            return len(keys)

        async def verify_presence(business, provider):
            provider_checks.append(business.name)
            return WebsitePresenceResult(
                status=WebsitePresenceStatus.ABSENT_CONFIRMED,
                source=WebsitePresenceSource.WEB_SEARCH,
                evidence=("no_official_site",),
                requests_used=1,
            )

        def record_progress(task_id: int, snapshot: dict) -> None:
            progress_by_task[task_id].append(dict(snapshot))

        def export_excel(items, *, task_id=None, **kwargs):
            exported_by_task[task_id] = [business.name for business in items]
            return f"result-{task_id}.xlsx"

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(orchestrator.config, "DEEP_DISCOVERY_MODE", "apply")
            )
            stack.enter_context(
                patch.object(orchestrator.config, "CANDIDATE_HISTORY_MODE", "apply")
            )
            stack.enter_context(patch.object(
                orchestrator.config,
                "LEAD_WEBSITE_POLICY",
                "verified_no_site_only",
            ))
            stack.enter_context(patch.object(
                orchestrator.config,
                "WEBSITE_PRESENCE_VERIFICATION_MODE",
                "apply",
            ))
            stack.enter_context(
                patch.object(orchestrator.config, "WEBSITE_RESOLVER_MODE", "off")
            )
            stack.enter_context(patch.object(
                orchestrator.config,
                "LEAD_CONTACTABILITY_MODE",
                "instagram_only",
            ))
            stack.enter_context(patch.object(
                orchestrator.config,
                "MAX_CHECKED_CANDIDATES_PER_TASK",
                20,
            ))
            stack.enter_context(patch.object(
                orchestrator.config,
                "MAX_MAPS_CARDS_PER_TASK",
                20,
            ))
            stack.enter_context(patch.object(
                orchestrator.db,
                "get_task",
                side_effect=lambda task_id: {
                    "niche": "dentistry",
                    "city": "Kyiv",
                    "count": 2,
                },
            ))
            stack.enter_context(patch.object(orchestrator.db, "update_task_status"))
            stack.enter_context(patch.object(
                orchestrator.db,
                "update_task_progress",
                new=record_progress,
            ))
            stack.enter_context(patch.object(orchestrator.db, "save_businesses"))
            stack.enter_context(patch.object(orchestrator.db, "update_business"))
            stack.enter_context(
                patch.object(orchestrator, "build_query_plan", return_value=PLAN)
            )
            stack.enter_context(patch.object(
                orchestrator.collector,
                "collect_stream",
                new=fake_collect_stream,
            ))
            stack.enter_context(patch.object(
                orchestrator.candidate_history,
                "canonical_scope_key_from_resolved",
                return_value="dentistry:kyiv",
            ))
            stack.enter_context(patch.object(
                orchestrator.candidate_history,
                "candidate_fingerprint",
                side_effect=lambda business, city: ("name", business.name),
            ))
            stack.enter_context(patch.object(
                orchestrator.candidate_history,
                "claim_candidate",
                side_effect=claim_candidate,
            ))
            stack.enter_context(patch.object(
                orchestrator.candidate_history,
                "mark_candidate_checked",
                side_effect=mark_checked,
            ))
            stack.enter_context(patch.object(
                orchestrator.candidate_history,
                "release_candidate_claim",
                side_effect=release_claim,
            ))
            stack.enter_context(patch.object(
                orchestrator.candidate_history,
                "release_unfinished_candidate_claims",
                side_effect=release_unfinished,
            ))
            stack.enter_context(patch.object(
                orchestrator.website_presence_verifier,
                "verify_business_website_presence",
                new=verify_presence,
            ))
            stack.enter_context(patch.object(
                orchestrator.social_checker,
                "check_instagram",
                new=AsyncMock(),
            ))
            stack.enter_context(
                patch.object(orchestrator.ai_scorer, "score_businesses", new=AsyncMock())
            )
            stack.enter_context(patch.object(
                orchestrator.reporter,
                "export_csv",
                return_value="result.csv",
            ))
            stack.enter_context(patch.object(
                orchestrator.reporter,
                "export_excel",
                side_effect=export_excel,
            ))
            stack.enter_context(patch.object(
                orchestrator.reporter,
                "format_leads_summary",
                return_value="summary",
            ))
            stack.enter_context(patch.object(
                orchestrator,
                "finalize_completed_task",
                new=AsyncMock(return_value=None),
            ))
            for task_id in (1, 2):
                current_task_id = task_id
                await orchestrator.run_search(
                    task_id,
                    progress_callback=AsyncMock(),
                    progress_interval=0,
                    website_presence_search_provider=object(),
                )

        self.assertEqual(opened_by_task[1], [f"link-{index}" for index in range(1, 7)])
        self.assertEqual(len(set(opened_by_task[1])), 6)
        first = progress_by_task[1][-1]
        self.assertEqual(first["mapsCardsActuallyOpened"], 6)
        self.assertEqual(first["mapsLinksDiscovered"], 10)
        self.assertEqual(first["mapsLinksSkippedTaskDuplicate"], 4)
        self.assertEqual(first["previouslyCheckedHistorySkips"], 2)
        self.assertEqual(first["newCandidatesChecked"], 4)
        self.assertTrue(first["deepPhaseActivated"])
        self.assertEqual(exported_by_task[1], ["5", "6"])

        self.assertEqual(opened_by_task[2], [f"link-{index}" for index in range(1, 7)])
        self.assertEqual(progress_by_task[2][-1]["previouslyCheckedHistorySkips"], 6)
        self.assertEqual(progress_by_task[2][-1]["newCandidatesChecked"], 0)
        self.assertEqual(exported_by_task[2], [])
        self.assertEqual(provider_checks, ["5", "6"])
        self.assertEqual(durable_checked, {"1", "2", "3", "4", "5", "6"})
        self.assertEqual(active_claims, {})


if __name__ == "__main__":
    unittest.main()

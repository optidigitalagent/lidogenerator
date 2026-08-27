"""Integration tests for staged normal/deep discovery orchestration."""

import unittest
from unittest.mock import AsyncMock, Mock, patch

import orchestrator
from models import Business
from query_planner import QueryKind, QueryPlan, QueryQueue, SearchQuery


def _query(text: str, kind: QueryKind) -> SearchQuery:
    return SearchQuery(
        text=text,
        niche="service",
        city="city",
        kind=kind,
        variant="variant" if kind is QueryKind.DISTRICT_VARIANT else None,
        district="district" if kind is QueryKind.DISTRICT_VARIANT else None,
    )


PLAN = QueryPlan(
    normal_queries=QueryQueue(
        (
            _query("normal-1", QueryKind.BASE),
            _query("normal-2", QueryKind.NICHE_VARIANT),
        )
    ),
    deep_queries=QueryQueue(
        (
            _query("deep-1", QueryKind.DISTRICT_VARIANT),
            _query("deep-2", QueryKind.DISTRICT_VARIANT),
        )
    ),
)


def _candidate(name: str) -> Business:
    return Business(name=name)


def _lead(name: str) -> Business:
    return Business(name=name, instagram_url=f"https://instagram.com/{name}")


class OrchestratorDeepDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        cards_by_query: dict[str, list[Business]],
        target: int = 1,
        checked_limit: int = 20,
        opened_limit: int = 20,
    ) -> dict:
        collector_calls: list[dict] = []
        allocation_calls: list[dict] = []
        progress_snapshots: list[dict] = []
        messages: list[str] = []
        downstream: dict[str, list[list[Business]]] = {
            "save": [],
            "csv": [],
            "excel": [],
        }
        real_allocator = orchestrator.allocate_query_budget

        async def fake_collect_stream(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            query_text=None,
            stop_flag=None,
            **kwargs,
        ):
            collector_calls.append(
                {"query_text": query_text, "max_businesses": max_businesses}
            )
            items = list(cards_by_query.get(query_text, ()))[:max_businesses]
            for index, _ in enumerate(items, start=1):
                if progress_callback:
                    await progress_callback(index)
            if items:
                yield items

        async def fake_check(items, progress_callback=None):
            return items

        async def progress_callback(text: str) -> None:
            messages.append(text)

        def capture(name):
            def inner(items, *args, **kwargs):
                downstream[name].append(list(items))
                if name == "csv":
                    return "result.csv"
                if name == "excel":
                    return "result.xlsx"
                return len(items)

            return inner

        def record_allocation(**kwargs):
            allocation_calls.append(dict(kwargs))
            return real_allocator(**kwargs)

        def record_progress(task_id: int, snapshot: dict) -> None:
            progress_snapshots.append(dict(snapshot))

        with (
            patch.object(orchestrator.config, "DEEP_DISCOVERY_MODE", "apply"),
            patch.object(
                orchestrator.config,
                "MAX_CHECKED_CANDIDATES_PER_TASK",
                checked_limit,
            ),
            patch.object(
                orchestrator.config,
                "MAX_MAPS_CARDS_PER_TASK",
                opened_limit,
            ),
            patch.object(
                orchestrator.db,
                "get_task",
                return_value={"niche": "service", "city": "city", "count": target},
            ),
            patch.object(orchestrator.db, "update_task_status"),
            patch.object(orchestrator.db, "update_task_progress", new=record_progress),
            patch.object(orchestrator.db, "save_businesses", side_effect=capture("save")),
            patch.object(orchestrator.db, "update_business"),
            patch.object(orchestrator, "build_query_plan", return_value=PLAN),
            patch.object(orchestrator, "allocate_query_budget", side_effect=record_allocation),
            patch.object(orchestrator.collector, "collect_stream", new=fake_collect_stream),
            patch.object(orchestrator.site_checker, "check_sites", new=fake_check),
            patch.object(
                orchestrator.social_checker,
                "check_instagram",
                new=AsyncMock(),
            ),
            patch.object(orchestrator.ai_scorer, "score_businesses", new=AsyncMock()),
            patch.object(orchestrator.reporter, "export_csv", side_effect=capture("csv")),
            patch.object(orchestrator.reporter, "export_excel", side_effect=capture("excel")),
            patch.object(orchestrator.reporter, "format_leads_summary", return_value="summary"),
            patch.object(orchestrator, "finalize_completed_task", new=AsyncMock(return_value=None)),
        ):
            result = await orchestrator.run_search(
                1,
                progress_callback=progress_callback,
                progress_interval=0,
            )

        return {
            "result": result,
            "collector_calls": collector_calls,
            "allocation_calls": allocation_calls,
            "progress_snapshots": progress_snapshots,
            "messages": messages,
            "downstream": downstream,
        }

    async def test_target_reached_in_normal_phase_does_not_consume_deep(self) -> None:
        run = await self._run(cards_by_query={"normal-1": [_lead("lead")]})

        self.assertEqual(
            [call["query_text"] for call in run["collector_calls"]],
            ["normal-1"],
        )
        self.assertFalse(run["progress_snapshots"][-1]["deepPhaseActivated"])

    async def test_deep_phase_starts_only_after_normal_exhaustion(self) -> None:
        run = await self._run(
            cards_by_query={
                "normal-1": [_candidate("one")],
                "normal-2": [_candidate("two")],
                "deep-1": [_lead("lead")],
            }
        )

        self.assertEqual(
            [call["query_text"] for call in run["collector_calls"]],
            ["normal-1", "normal-2", "deep-1"],
        )
        self.assertTrue(
            any("глибше по районах" in message for message in run["messages"])
        )
        final = run["progress_snapshots"][-1]
        self.assertEqual(final["discovery_phase"], "deep")
        self.assertEqual(final["normalQueriesCompleted"], 2)
        self.assertTrue(final["deepPhaseActivated"])

    async def test_inactive_deep_queries_do_not_dilute_normal_budget(self) -> None:
        run = await self._run(
            cards_by_query={
                "normal-1": [_candidate("n1"), _candidate("n2")],
                "normal-2": [_candidate("n3")],
            },
            target=2,
            checked_limit=12,
            opened_limit=12,
        )

        self.assertEqual(run["allocation_calls"][0]["active_queries"], 2)
        self.assertEqual(run["collector_calls"][0]["max_businesses"], 6)
        self.assertEqual(run["allocation_calls"][2]["active_queries"], 2)
        self.assertEqual(
            run["allocation_calls"][2],
            {
                "remaining_checked_candidates": 9,
                "remaining_opened_cards": 9,
                "active_queries": 2,
            },
        )
        self.assertEqual(run["collector_calls"][2]["max_businesses"], 5)


if __name__ == "__main__":
    unittest.main()

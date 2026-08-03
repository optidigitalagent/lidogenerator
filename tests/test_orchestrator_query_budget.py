"""Integration tests for fair discovery-budget orchestration."""

import asyncio
import inspect
import unittest
import urllib.parse
from unittest.mock import AsyncMock, Mock, patch

import config
import orchestrator
from agents import collector
from models import Business
from query_planner import QueryKind, QueryQueue, SearchQuery
from search_policy import StopReason


def _queue(size: int) -> QueryQueue:
    return QueryQueue(
        tuple(
            SearchQuery(
                text=f"query-{index}",
                niche="test",
                city="city",
                kind=(QueryKind.BASE if index == 1 else QueryKind.NICHE_VARIANT),
                variant=(None if index == 1 else f"variant-{index}"),
            )
            for index in range(1, size + 1)
        )
    )


def _lead(name: str) -> Business:
    return Business(name=name, instagram_url=f"https://instagram.com/{name}")


def _candidate(name: str) -> Business:
    return Business(name=name)


def _candidates(prefix: str, count: int) -> list[Business]:
    return [_candidate(f"{prefix}-{index}") for index in range(count)]


class OrchestratorQueryBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        query_count: int,
        cards_by_query: dict[str, list[Business | None]],
        target: int,
        checked_limit: int,
        opened_limit: int,
        stop_event: asyncio.Event | None = None,
        stop_after_site_check: bool = False,
    ) -> dict:
        query_queue = _queue(query_count)
        collector_calls: list[dict] = []
        checked_batches: list[list[Business]] = []
        decisions = []
        opened_total = 0
        statuses = Mock()
        downstream: dict[str, list[list[Business]]] = {
            "save": [],
            "social": [],
            "score": [],
            "csv": [],
            "excel": [],
            "summary": [],
        }

        async def fake_collect_stream(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            stop_flag=None,
            query_text=None,
            **kwargs,
        ):
            nonlocal opened_total
            collector_calls.append(
                {
                    "query_text": query_text,
                    "max_businesses": max_businesses,
                }
            )
            opened_in_stream = 0
            extracted: list[Business] = []
            for card in cards_by_query.get(query_text, []):
                if opened_in_stream >= max_businesses:
                    break
                if stop_flag and stop_flag():
                    return
                opened_in_stream += 1
                opened_total += 1
                if progress_callback:
                    await progress_callback(opened_in_stream)
                if card is not None:
                    extracted.append(card)
            if extracted:
                yield extracted

        async def fake_check_sites(items, progress_callback=None):
            checked_batches.append(list(items))
            if stop_after_site_check and stop_event is not None:
                stop_event.set()
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

        real_decide_next = orchestrator.decide_next

        def record_decision(progress, policy):
            decision = real_decide_next(progress, policy)
            decisions.append((progress, decision))
            return decision

        task = {"niche": "test", "city": "city", "count": target}
        with (
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
            patch.object(orchestrator.db, "get_task", return_value=task),
            patch.object(orchestrator.db, "update_task_status", statuses),
            patch.object(orchestrator.db, "save_businesses", side_effect=capture("save")),
            patch.object(orchestrator.db, "update_business"),
            patch.object(orchestrator, "build_query_queue", return_value=query_queue),
            patch.object(orchestrator.collector, "collect_stream", new=fake_collect_stream),
            patch.object(orchestrator.site_checker, "check_sites", new=fake_check_sites),
            patch.object(orchestrator.social_checker, "check_instagram", new=fake_social),
            patch.object(orchestrator.ai_scorer, "score_businesses", new=fake_score),
            patch.object(orchestrator.reporter, "export_csv", side_effect=capture("csv")),
            patch.object(orchestrator.reporter, "export_excel", side_effect=capture("excel")),
            patch.object(
                orchestrator.reporter,
                "format_leads_summary",
                side_effect=capture("summary"),
            ),
            patch.object(orchestrator, "decide_next", side_effect=record_decision),
        ):
            result = await orchestrator.run_search(
                1,
                progress_callback=AsyncMock(),
                stop_event=stop_event,
                progress_interval=0,
            )

        return {
            "result": result,
            "collector_calls": collector_calls,
            "checked_batches": checked_batches,
            "checked_total": sum(map(len, checked_batches)),
            "decisions": decisions,
            "opened_total": opened_total,
            "statuses": statuses.call_args_list,
            "downstream": downstream,
        }

    async def test_first_of_seven_queries_gets_fair_share(self) -> None:
        run = await self._run(
            query_count=7,
            cards_by_query={},
            target=1,
            checked_limit=1000,
            opened_limit=1000,
        )

        self.assertEqual(run["collector_calls"][0]["max_businesses"], 143)

    async def test_fully_used_budget_is_distributed_four_three_three(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": _candidates("one", 10),
                "query-2": _candidates("two", 10),
                "query-3": _candidates("three", 10),
            },
            target=10,
            checked_limit=10,
            opened_limit=10,
        )

        self.assertEqual(
            [call["max_businesses"] for call in run["collector_calls"]],
            [4, 3, 3],
        )
        self.assertEqual(run["opened_total"], 10)

    async def test_unused_first_query_budget_rolls_into_second(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": [_candidate("only")],
                "query-2": _candidates("two", 10),
                "query-3": _candidates("three", 10),
            },
            target=10,
            checked_limit=10,
            opened_limit=10,
        )

        self.assertEqual(
            [call["max_businesses"] for call in run["collector_calls"]],
            [4, 5, 4],
        )

    async def test_opened_cards_without_extraction_only_spend_opened_budget(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": [None] * 10,
                "query-2": [None] * 10,
                "query-3": [None] * 10,
            },
            target=1,
            checked_limit=10,
            opened_limit=10,
        )

        self.assertEqual(
            [call["max_businesses"] for call in run["collector_calls"]],
            [4, 3, 3],
        )
        self.assertEqual(run["opened_total"], 10)
        self.assertEqual(run["checked_total"], 0)
        self.assertEqual(
            run["decisions"][-1][1].stop_reason,
            StopReason.MAX_DISCOVERY_CARDS_REACHED,
        )

    async def test_checked_limit_stops_before_next_query_and_exports_partial(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": [_lead("lead")],
                "query-2": [_candidate("checked")],
                "query-3": [_candidate("unused")],
            },
            target=2,
            checked_limit=2,
            opened_limit=10,
        )

        self.assertEqual(
            [call["query_text"] for call in run["collector_calls"]],
            ["query-1", "query-2"],
        )
        self.assertEqual(
            run["decisions"][-1][1].stop_reason,
            StopReason.MAX_CANDIDATES_REACHED,
        )
        self.assertEqual(run["result"], "result.xlsx")
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_opened_limit_stops_before_next_query_and_exports_partial(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": [None],
                "query-2": [_lead("lead")],
                "query-3": [_candidate("unused")],
            },
            target=2,
            checked_limit=10,
            opened_limit=2,
        )

        self.assertEqual(
            [call["query_text"] for call in run["collector_calls"]],
            ["query-1", "query-2"],
        )
        self.assertEqual(
            run["decisions"][-1][1].stop_reason,
            StopReason.MAX_DISCOVERY_CARDS_REACHED,
        )
        self.assertEqual(run["result"], "result.xlsx")
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_target_has_priority_when_opened_limit_is_reached_together(self) -> None:
        run = await self._run(
            query_count=1,
            cards_by_query={"query-1": [_lead("lead")]},
            target=1,
            checked_limit=1,
            opened_limit=1,
        )

        self.assertEqual(
            run["decisions"][-1][1].stop_reason,
            StopReason.TARGET_REACHED,
        )

    async def test_user_stop_has_priority_over_target_and_both_limits(self) -> None:
        stop_event = asyncio.Event()
        run = await self._run(
            query_count=1,
            cards_by_query={"query-1": [_lead("lead")]},
            target=1,
            checked_limit=1,
            opened_limit=1,
            stop_event=stop_event,
            stop_after_site_check=True,
        )

        self.assertIsNone(run["result"])
        self.assertEqual(
            run["decisions"][-1][1].stop_reason,
            StopReason.USER_STOPPED,
        )
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_single_unknown_niche_query_gets_all_available_budget(self) -> None:
        run = await self._run(
            query_count=1,
            cards_by_query={},
            target=1,
            checked_limit=1000,
            opened_limit=700,
        )

        self.assertEqual(run["collector_calls"][0]["max_businesses"], 700)

    async def test_future_fallback_queries_are_reserved_and_can_run(self) -> None:
        run = await self._run(
            query_count=7,
            cards_by_query={},
            target=1,
            checked_limit=1000,
            opened_limit=1000,
        )

        self.assertEqual(run["collector_calls"][0]["max_businesses"], 143)
        self.assertEqual(
            [call["query_text"] for call in run["collector_calls"]],
            [f"query-{index}" for index in range(1, 8)],
        )

    async def test_collector_is_never_called_with_zero_limit(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={"query-1": [None]},
            target=1,
            checked_limit=10,
            opened_limit=1,
        )

        self.assertEqual(len(run["collector_calls"]), 1)
        self.assertTrue(
            all(call["max_businesses"] > 0 for call in run["collector_calls"])
        )

    async def test_global_opened_card_cap_is_never_exceeded(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": [None] * 10,
                "query-2": [None] * 10,
                "query-3": [None] * 10,
            },
            target=1,
            checked_limit=10,
            opened_limit=5,
        )

        self.assertEqual(run["opened_total"], 5)
        self.assertLessEqual(run["opened_total"], 5)

    async def test_global_checked_candidate_cap_is_never_exceeded(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": _candidates("one", 10),
                "query-2": _candidates("two", 10),
                "query-3": _candidates("three", 10),
            },
            target=5,
            checked_limit=5,
            opened_limit=10,
        )

        self.assertEqual(run["checked_total"], 5)
        self.assertLessEqual(run["checked_total"], 5)

    def test_legacy_collector_api_and_default_remain_available(self) -> None:
        signature = inspect.signature(collector.collect_stream)

        self.assertEqual(config.MAX_BUSINESSES_PER_SEARCH, 1000)
        self.assertIn("max_businesses", signature.parameters)
        self.assertIsNone(signature.parameters["max_businesses"].default)
        self.assertIn("query_text", signature.parameters)
        self.assertIsNone(signature.parameters["query_text"].default)
        expected = (
            "https://www.google.com/maps/search/"
            f"{urllib.parse.quote('test city')}?hl=uk"
        )
        self.assertEqual(collector._maps_search_url("test", "city"), expected)

    async def test_downstream_pipeline_runs_once_after_all_queries(self) -> None:
        run = await self._run(
            query_count=3,
            cards_by_query={
                "query-1": [_lead("one")],
                "query-2": [None],
                "query-3": [_candidate("checked")],
            },
            target=3,
            checked_limit=10,
            opened_limit=10,
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            with self.subTest(name=name):
                self.assertEqual(len(run["downstream"][name]), 1)

    async def test_every_downstream_consumer_is_capped_to_target(self) -> None:
        run = await self._run(
            query_count=1,
            cards_by_query={
                "query-1": [_lead("one"), _lead("two"), _lead("three")],
            },
            target=2,
            checked_limit=10,
            opened_limit=10,
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            with self.subTest(name=name):
                self.assertEqual(len(run["downstream"][name][0]), 2)


if __name__ == "__main__":
    unittest.main()

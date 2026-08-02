"""Isolated tests for QueryQueue orchestration and collector query text."""

import asyncio
import unittest
import urllib.parse
from unittest.mock import AsyncMock, Mock, patch

import orchestrator
from agents import collector
from models import Business
from query_planner import QueryKind, QueryQueue, SearchQuery
from search_policy import StopReason


def _query(text: str, kind: QueryKind = QueryKind.BASE) -> SearchQuery:
    return SearchQuery(text=text, niche="test", city="city", kind=kind)


def _queue(*texts: str) -> QueryQueue:
    return QueryQueue(tuple(_query(text) for text in texts))


def _lead(
    name: str,
    *,
    phone: str = "",
    address: str = "",
    website: str = "",
    instagram_url: str | None = None,
) -> Business:
    return Business(
        name=name,
        phone=phone,
        address=address,
        website=website,
        instagram_url=instagram_url or f"https://instagram.com/{name}",
    )


def _not_lead(name: str) -> Business:
    return Business(name=name)


class OrchestratorQueryQueueTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        queue: QueryQueue,
        streams: dict[str, list[list[Business]]],
        target: int,
        max_candidates: int = 20,
        stop_event: asyncio.Event | None = None,
        stop_after_stream: str | None = None,
    ) -> dict:
        collector_calls: list[dict] = []
        checked_batches: list[list[Business]] = []
        decisions = []
        active_remaining: list[int] = []
        before_stream_remaining: list[int] = []
        events: list[str] = []
        statuses = Mock()
        downstream: dict[str, list[list[Business]]] = {
            "save": [],
            "social": [],
            "score": [],
            "csv": [],
            "excel": [],
            "summary": [],
        }
        queue_builder = Mock(return_value=queue)

        async def fake_collect_stream(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            stop_flag=None,
            query_text=None,
            **kwargs,
        ):
            before_stream_remaining.append(decisions[-1][0].remaining_queries)
            collector_calls.append(
                {
                    "niche": niche,
                    "city": city,
                    "max_businesses": max_businesses,
                    "query_text": query_text,
                }
            )
            events.append(f"collect:{query_text}")
            visited = 0
            for batch in streams.get(query_text, []):
                if stop_flag and stop_flag():
                    return
                for _ in batch:
                    visited += 1
                    if progress_callback:
                        await progress_callback(visited)
                active_remaining.append(decisions[-1][0].remaining_queries)
                yield batch
            if query_text == stop_after_stream and stop_event is not None:
                stop_event.set()

        async def fake_check_sites(items, progress_callback=None):
            checked_batches.append(list(items))
            events.append("site")
            return items

        async def fake_social(items, progress_callback=None, stop_flag=None):
            downstream["social"].append(list(items))
            events.append("social")
            return items

        async def fake_score(items, progress_callback=None):
            downstream["score"].append(list(items))
            events.append("score")
            return items

        def capture(name):
            def inner(items, *args, **kwargs):
                downstream[name].append(list(items))
                events.append(name)
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
                max_candidates,
            ),
            patch.object(
                orchestrator.config,
                "MAX_MAPS_CARDS_PER_TASK",
                max_candidates,
            ),
            patch.object(orchestrator.db, "get_task", return_value=task),
            patch.object(orchestrator.db, "update_task_status", statuses),
            patch.object(orchestrator.db, "save_businesses", side_effect=capture("save")),
            patch.object(orchestrator.db, "update_business"),
            patch.object(orchestrator, "build_query_queue", queue_builder),
            patch.object(orchestrator.collector, "collect_stream", new=fake_collect_stream),
            patch.object(orchestrator.site_checker, "check_sites", new=fake_check_sites),
            patch.object(orchestrator.social_checker, "check_instagram", new=fake_social),
            patch.object(orchestrator.ai_scorer, "score_businesses", new=fake_score),
            patch.object(orchestrator.reporter, "export_csv", side_effect=capture("csv")),
            patch.object(orchestrator.reporter, "export_excel", side_effect=capture("excel")),
            patch.object(orchestrator.reporter, "format_leads_summary", side_effect=capture("summary")),
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
            "decisions": decisions,
            "active_remaining": active_remaining,
            "before_stream_remaining": before_stream_remaining,
            "events": events,
            "statuses": statuses.call_args_list,
            "downstream": downstream,
            "queue_builder": queue_builder,
        }

    async def test_single_base_query_preserves_partial_result_behavior(self) -> None:
        run = await self._run(
            queue=_queue("test city"),
            streams={"test city": [[_lead("only")]]},
            target=2,
        )

        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual([call["query_text"] for call in run["collector_calls"]], ["test city"])
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.QUERIES_EXHAUSTED)
        run["queue_builder"].assert_called_once_with(
            niche="test",
            city="city",
            niche_variants=(),
        )

    async def test_exhausted_first_stream_starts_second_before_downstream(self) -> None:
        run = await self._run(
            queue=_queue("first", "second"),
            streams={"first": [[_lead("one")]], "second": [[_lead("two")]]},
            target=3,
        )

        self.assertEqual([call["query_text"] for call in run["collector_calls"]], ["first", "second"])
        self.assertLess(run["events"].index("collect:second"), run["events"].index("save"))
        self.assertEqual(len(run["downstream"]["save"]), 1)

    async def test_target_in_first_stream_skips_second_and_caps_downstream(self) -> None:
        run = await self._run(
            queue=_queue("first", "second"),
            streams={"first": [[_lead("one"), _lead("two")]], "second": [[_lead("unused")]]},
            target=1,
        )

        self.assertEqual([call["query_text"] for call in run["collector_calls"]], ["first"])
        for name in ("save", "social", "score", "csv", "excel", "summary"):
            self.assertEqual(len(run["downstream"][name][0]), 1)

    async def test_target_in_second_stream_preserves_order_and_skips_third(self) -> None:
        run = await self._run(
            queue=_queue("first", "second", "third"),
            streams={
                "first": [[_lead("one")]],
                "second": [[_lead("two"), _lead("three")]],
                "third": [[_lead("unused")]],
            },
            target=3,
        )

        self.assertEqual([call["query_text"] for call in run["collector_calls"]], ["first", "second"])
        self.assertEqual([item.name for item in run["downstream"]["save"][0]], ["one", "two", "three"])

    async def test_user_stop_between_streams_does_not_start_second(self) -> None:
        stop_event = asyncio.Event()
        run = await self._run(
            queue=_queue("first", "second"),
            streams={"first": [[_lead("one")]], "second": [[_lead("unused")]]},
            target=2,
            stop_event=stop_event,
            stop_after_stream="first",
        )

        self.assertIsNone(run["result"])
        self.assertEqual([call["query_text"] for call in run["collector_calls"]], ["first"])
        self.assertIn(unittest.mock.call(1, "stopped"), run["statuses"])
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_candidate_limit_prevents_second_stream_and_exports_partial(self) -> None:
        run = await self._run(
            queue=_queue("first", "second"),
            streams={"first": [[_not_lead("one"), _not_lead("two")]], "second": [[_lead("unused")]]},
            target=2,
            max_candidates=2,
        )

        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual([call["query_text"] for call in run["collector_calls"]], ["first"])
        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.MAX_CANDIDATES_REACHED)
        self.assertIn(unittest.mock.call(1, "done", csv_path="result.xlsx"), run["statuses"])

    async def test_full_queue_exhaustion_exports_partial_result(self) -> None:
        run = await self._run(
            queue=_queue("first", "second"),
            streams={"first": [[_lead("one")]], "second": [[_not_lead("skip")]]},
            target=3,
        )

        self.assertEqual(run["decisions"][-1][1].stop_reason, StopReason.QUERIES_EXHAUSTED)
        self.assertEqual([item.name for item in run["downstream"]["csv"][0]], ["one"])

    async def test_remaining_queries_tracks_active_and_exhausted_streams(self) -> None:
        run = await self._run(
            queue=_queue("first", "second", "third"),
            streams={
                "first": [[_not_lead("one")]],
                "second": [[_not_lead("two")]],
                "third": [[_not_lead("three")]],
            },
            target=4,
            max_candidates=10,
        )

        self.assertEqual(run["active_remaining"], [3, 2, 1])
        self.assertEqual(run["before_stream_remaining"], [3, 2, 1])
        self.assertEqual(run["decisions"][-1][0].remaining_queries, 0)

    async def test_duplicate_across_queries_is_checked_and_exported_once(self) -> None:
        first = _lead("first-name", phone="+380 (50) 123-45-67", instagram_url="https://instagram.com/first")
        duplicate = _lead("second-name", phone="380501234567", instagram_url="https://instagram.com/second")
        run = await self._run(
            queue=_queue("first", "second"),
            streams={"first": [[first]], "second": [[duplicate]]},
            target=3,
        )

        self.assertEqual(sum(len(batch) for batch in run["checked_batches"]), 1)
        self.assertEqual(run["decisions"][-1][0].checked_candidates, 1)
        self.assertEqual([item.name for item in run["downstream"]["csv"][0]], ["first-name"])

    async def test_distinct_businesses_without_phone_are_not_collapsed(self) -> None:
        run = await self._run(
            queue=_queue("first", "second"),
            streams={
                "first": [[_lead("alpha", address="One Street")]],
                "second": [[_lead("beta", address="Two Street")]],
            },
            target=3,
        )

        self.assertEqual(sum(len(batch) for batch in run["checked_batches"]), 2)
        self.assertEqual([item.name for item in run["downstream"]["csv"][0]], ["alpha", "beta"])

    async def test_collector_receives_exact_search_query_text(self) -> None:
        exact = "стоматологія Оболонь Київ"
        query = SearchQuery(
            text=exact,
            niche="стоматология",
            city="Київ",
            kind=QueryKind.DISTRICT_VARIANT,
            variant="стоматологія",
            district="Оболонь",
        )
        run = await self._run(
            queue=QueryQueue((query,)),
            streams={exact: []},
            target=1,
        )

        self.assertEqual(run["collector_calls"][0]["query_text"], exact)


class CollectorQueryTextTests(unittest.TestCase):
    def test_explicit_query_text_is_used_for_maps_url(self) -> None:
        text = "стоматологія Оболонь Київ"
        expected = f"https://www.google.com/maps/search/{urllib.parse.quote(text)}?hl=uk"

        self.assertEqual(collector._maps_search_url("ignored", "ignored", text), expected)

    def test_legacy_niche_city_query_remains_the_default(self) -> None:
        expected = f"https://www.google.com/maps/search/{urllib.parse.quote('dental Kyiv')}?hl=uk"

        self.assertEqual(collector._maps_search_url("dental", "Kyiv"), expected)


if __name__ == "__main__":
    unittest.main()

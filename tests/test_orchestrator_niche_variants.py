"""Isolated integration tests for curated niche variants in orchestration."""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

import orchestrator
from models import Business
from niche_catalog import NicheSearchPlan
from query_planner import QueryKind, QueryQueue, SearchQuery


DENTISTRY_QUERIES = (
    "стоматология Киев",
    "стоматологія Киев",
    "стоматологическая клиника Киев",
    "стоматологічна клініка Киев",
    "стоматологічний кабінет Киев",
    "зубная клиника Киев",
    "зубна клініка Киев",
)


def _lead(name: str, *, phone: str = "") -> Business:
    return Business(
        name=name,
        phone=phone,
        instagram_url=f"https://instagram.com/{name}",
    )


class OrchestratorNicheVariantsTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        niche: str,
        streams: dict[str, list[list[Business]]],
        target: int,
        stop_event: asyncio.Event | None = None,
        stop_after_query: str | None = None,
        plan_mock: Mock | None = None,
        queue_builder: Mock | None = None,
    ) -> dict:
        collector_queries: list[str] = []
        checked_batches: list[list[Business]] = []
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
            collector_queries.append(query_text)
            visited = 0
            for batch in streams.get(query_text, []):
                if stop_flag and stop_flag():
                    return
                for _ in batch:
                    visited += 1
                    if progress_callback:
                        await progress_callback(visited)
                yield batch
            if query_text == stop_after_query and stop_event is not None:
                stop_event.set()

        async def fake_check_sites(items, progress_callback=None):
            checked_batches.append(list(items))
            return items

        async def pass_through(items, progress_callback=None, **kwargs):
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

        task = {"niche": niche, "city": "Киев", "count": target}
        plan_patch = patch.object(
            orchestrator,
            "resolve_niche_plan",
            plan_mock or orchestrator.resolve_niche_plan,
        )
        queue_patch = patch.object(
            orchestrator,
            "build_query_queue",
            queue_builder or orchestrator.build_query_queue,
        )
        with (
            patch.object(
                orchestrator.config,
                "MAX_CHECKED_CANDIDATES_PER_TASK",
                100,
            ),
            patch.object(
                orchestrator.config,
                "MAX_MAPS_CARDS_PER_TASK",
                100,
            ),
            patch.object(orchestrator.db, "get_task", return_value=task),
            patch.object(orchestrator.db, "update_task_status", statuses),
            patch.object(orchestrator.db, "save_businesses", side_effect=capture("save")),
            patch.object(orchestrator.db, "update_business"),
            plan_patch,
            queue_patch,
            patch.object(orchestrator.collector, "collect_stream", new=fake_collect_stream),
            patch.object(orchestrator.site_checker, "check_sites", new=fake_check_sites),
            patch.object(orchestrator.social_checker, "check_instagram", new=pass_through),
            patch.object(orchestrator.ai_scorer, "score_businesses", new=pass_through),
            patch.object(orchestrator.reporter, "export_csv", side_effect=capture("csv")),
            patch.object(orchestrator.reporter, "export_excel", side_effect=capture("excel")),
            patch.object(
                orchestrator.reporter,
                "format_leads_summary",
                side_effect=capture("summary"),
            ),
        ):
            result = await orchestrator.run_search(
                1,
                progress_callback=AsyncMock(),
                stop_event=stop_event,
                progress_interval=0,
            )

        return {
            "result": result,
            "collector_queries": collector_queries,
            "checked_batches": checked_batches,
            "downstream": downstream,
            "statuses": statuses.call_args_list,
        }

    async def test_known_niche_builds_variant_queue_in_catalog_order(self) -> None:
        run = await self._run(niche="стоматология", streams={}, target=1)

        self.assertEqual(run["collector_queries"], list(DENTISTRY_QUERIES))

    async def test_unknown_niche_uses_only_base_query(self) -> None:
        run = await self._run(niche="IT компания", streams={}, target=1)

        self.assertEqual(run["collector_queries"], ["IT компания Киев"])

    async def test_target_stops_remaining_variants_and_caps_downstream(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={
                DENTISTRY_QUERIES[0]: [[_lead("one")]],
                DENTISTRY_QUERIES[1]: [[_lead("two"), _lead("three")]],
            },
            target=3,
        )

        self.assertEqual(run["collector_queries"], list(DENTISTRY_QUERIES[:2]))
        self.assertEqual(
            [business.name for business in run["downstream"]["save"][0]],
            ["one", "two", "three"],
        )

    async def test_queue_exhaustion_runs_each_variant_once_and_exports_partial(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[0]: [[_lead("only")]]},
            target=3,
        )

        self.assertEqual(run["collector_queries"], list(DENTISTRY_QUERIES))
        self.assertEqual(len(run["collector_queries"]), len(set(run["collector_queries"])))
        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual(
            [business.name for business in run["downstream"]["save"][0]],
            ["only"],
        )

    async def test_catalog_is_called_once_per_task(self) -> None:
        plan_resolver = Mock(
            return_value=NicheSearchPlan(
                key="custom",
                input_niche="custom",
                base_niche="custom",
                primary_variants=("variant",),
                fallback_variants=(),
            )
        )
        run = await self._run(
            niche="custom",
            streams={},
            target=1,
            plan_mock=plan_resolver,
        )

        plan_resolver.assert_called_once_with("custom")
        self.assertEqual(run["collector_queries"], ["custom Киев", "variant Киев"])

    async def test_queue_builder_receives_variants_without_districts_or_limit(self) -> None:
        real_builder = orchestrator.build_query_queue
        builder = Mock(wraps=real_builder)
        await self._run(
            niche="стоматология",
            streams={},
            target=1,
            queue_builder=builder,
        )

        builder.assert_called_once()
        self.assertEqual(
            builder.call_args.kwargs["niche_variants"],
            tuple(query.rsplit(" ", 1)[0] for query in DENTISTRY_QUERIES[1:]),
        )
        self.assertNotIn("districts", builder.call_args.kwargs)
        self.assertNotIn("max_queries", builder.call_args.kwargs)

    async def test_duplicate_business_across_queries_is_processed_once(self) -> None:
        first = _lead("first", phone="+380 (50) 123-45-67")
        duplicate = _lead("duplicate", phone="380501234567")
        run = await self._run(
            niche="стоматология",
            streams={
                DENTISTRY_QUERIES[0]: [[first]],
                DENTISTRY_QUERIES[1]: [[duplicate]],
            },
            target=3,
        )

        self.assertEqual(sum(map(len, run["checked_batches"])), 1)
        self.assertEqual(
            [business.name for business in run["downstream"]["save"][0]],
            ["first"],
        )

    async def test_user_stop_after_first_variant_skips_next_variant(self) -> None:
        stop_event = asyncio.Event()
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[0]: [[_lead("one")]]},
            target=3,
            stop_event=stop_event,
            stop_after_query=DENTISTRY_QUERIES[1],
        )

        self.assertIsNone(run["result"])
        self.assertEqual(run["collector_queries"], list(DENTISTRY_QUERIES[:2]))
        self.assertIn(unittest.mock.call(1, "stopped"), run["statuses"])

    async def test_collector_uses_exact_text_from_query_queue(self) -> None:
        exact_text = "точный текст из очереди"
        queue = QueryQueue(
            (
                SearchQuery(
                    text=exact_text,
                    niche="стоматология",
                    city="Киев",
                    kind=QueryKind.NICHE_VARIANT,
                    variant="не пересобирать вручную",
                ),
            )
        )
        builder = Mock(return_value=queue)
        run = await self._run(
            niche="стоматология",
            streams={},
            target=1,
            queue_builder=builder,
        )

        self.assertEqual(run["collector_queries"], [exact_text])


if __name__ == "__main__":
    unittest.main()

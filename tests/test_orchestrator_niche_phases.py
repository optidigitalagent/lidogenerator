"""Isolated integration tests for primary and fallback niche phases."""

import asyncio
import unittest
from unittest.mock import Mock, patch

import orchestrator
from models import Business


DENTISTRY_QUERIES = (
    "стоматология Киев",
    "стоматологія Киев",
    "стоматологическая клиника Киев",
    "стоматологічна клініка Киев",
    "стоматологічний кабінет Киев",
    "зубная клиника Киев",
    "зубна клініка Киев",
)

UKRAINIAN_DENTISTRY_QUERIES = (
    "стоматологія Киев",
    "стоматология Киев",
    "стоматологическая клиника Киев",
    "стоматологічна клініка Киев",
    "стоматологічний кабінет Киев",
    "зубная клиника Киев",
    "зубна клініка Киев",
)

SAUNA_QUERIES = (
    "баня Киев",
    "банный комплекс Киев",
    "лазневий комплекс Киев",
    "саунный комплекс Киев",
    "комплекс саун Киев",
    "сауна Киев",
    "лазня Киев",
)


def _lead(name: str, *, phone: str = "") -> Business:
    return Business(
        name=name,
        phone=phone,
        instagram_url=f"https://instagram.com/{name}",
    )


def _not_lead(name: str) -> Business:
    return Business(name=name)


class OrchestratorNichePhasesTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        niche: str,
        streams: dict[str, list[list[Business]]],
        target: int,
        max_candidates: int = 100,
        stop_event: asyncio.Event | None = None,
        stop_after_query: str | None = None,
        resolver: Mock | None = None,
    ) -> dict:
        collector_queries: list[str] = []
        checked_batches: list[list[Business]] = []
        messages: list[str] = []
        statuses = Mock()
        downstream: dict[str, list[list[Business]]] = {
            "save": [],
            "social": [],
            "score": [],
            "csv": [],
            "excel": [],
            "summary": [],
        }
        resolver = resolver or Mock(wraps=orchestrator.resolve_niche_plan)

        async def progress_callback(text: str) -> None:
            messages.append(text)

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

        task = {"niche": niche, "city": "Киев", "count": target}
        with (
            patch.object(
                orchestrator.config,
                "MAX_BUSINESSES_PER_SEARCH",
                max_candidates,
            ),
            patch.object(orchestrator.db, "get_task", return_value=task),
            patch.object(orchestrator.db, "update_task_status", statuses),
            patch.object(
                orchestrator.db,
                "save_businesses",
                side_effect=capture("save"),
            ),
            patch.object(orchestrator.db, "update_business"),
            patch.object(orchestrator, "resolve_niche_plan", resolver),
            patch.object(
                orchestrator.collector,
                "collect_stream",
                new=fake_collect_stream,
            ),
            patch.object(
                orchestrator.site_checker,
                "check_sites",
                new=fake_check_sites,
            ),
            patch.object(
                orchestrator.social_checker,
                "check_instagram",
                new=fake_social,
            ),
            patch.object(
                orchestrator.ai_scorer,
                "score_businesses",
                new=fake_score,
            ),
            patch.object(
                orchestrator.reporter,
                "export_csv",
                side_effect=capture("csv"),
            ),
            patch.object(
                orchestrator.reporter,
                "export_excel",
                side_effect=capture("excel"),
            ),
            patch.object(
                orchestrator.reporter,
                "format_leads_summary",
                side_effect=capture("summary"),
            ),
        ):
            result = await orchestrator.run_search(
                1,
                progress_callback=progress_callback,
                stop_event=stop_event,
                progress_interval=0,
            )

        return {
            "result": result,
            "collector_queries": collector_queries,
            "checked_batches": checked_batches,
            "messages": messages,
            "statuses": statuses.call_args_list,
            "downstream": downstream,
            "resolver": resolver,
        }

    async def test_primary_input_runs_base_then_primary_then_fallback(self) -> None:
        run = await self._run(niche="стоматология", streams={}, target=1)

        self.assertEqual(run["collector_queries"], list(DENTISTRY_QUERIES))

    async def test_ukrainian_primary_input_remains_base(self) -> None:
        run = await self._run(niche="стоматологія", streams={}, target=1)

        self.assertEqual(
            run["collector_queries"],
            list(UKRAINIAN_DENTISTRY_QUERIES),
        )

    async def test_alias_only_input_uses_canonical_primary_not_raw_alias(self) -> None:
        run = await self._run(niche="стоматолог", streams={}, target=1)

        self.assertEqual(run["collector_queries"], list(DENTISTRY_QUERIES))
        self.assertNotIn("стоматолог Киев", run["collector_queries"])

    async def test_fallback_input_runs_complete_primary_phase_first(self) -> None:
        run = await self._run(niche="сауна", streams={}, target=1)

        self.assertEqual(run["collector_queries"], list(SAUNA_QUERIES))

    async def test_unknown_input_runs_only_raw_base_query(self) -> None:
        run = await self._run(niche="IT компания", streams={}, target=1)

        self.assertEqual(run["collector_queries"], ["IT компания Киев"])

    async def test_target_reached_during_primary_skips_fallback(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[0]: [[_lead("one")]]},
            target=1,
        )

        self.assertEqual(run["collector_queries"], [DENTISTRY_QUERIES[0]])
        self.assertFalse(
            set(DENTISTRY_QUERIES[4:]) & set(run["collector_queries"])
        )

    async def test_target_reached_on_last_primary_skips_first_fallback(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[3]: [[_lead("one")]]},
            target=1,
        )

        self.assertEqual(
            run["collector_queries"],
            list(DENTISTRY_QUERIES[:4]),
        )

    async def test_primary_exhaustion_starts_first_fallback_automatically(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[4]: [[_lead("one")]]},
            target=1,
        )

        self.assertEqual(
            run["collector_queries"],
            list(DENTISTRY_QUERIES[:5]),
        )

    async def test_user_stop_after_last_primary_prevents_fallback(self) -> None:
        stop_event = asyncio.Event()
        run = await self._run(
            niche="стоматология",
            streams={},
            target=1,
            stop_event=stop_event,
            stop_after_query=DENTISTRY_QUERIES[3],
        )

        self.assertIsNone(run["result"])
        self.assertEqual(
            run["collector_queries"],
            list(DENTISTRY_QUERIES[:4]),
        )
        self.assertIn(unittest.mock.call(1, "stopped"), run["statuses"])
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_candidate_limit_on_last_primary_exports_without_fallback(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[3]: [[_not_lead("checked")]]},
            target=1,
            max_candidates=1,
        )

        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual(
            run["collector_queries"],
            list(DENTISTRY_QUERIES[:4]),
        )
        self.assertIn(
            unittest.mock.call(1, "done", csv_path="result.xlsx"),
            run["statuses"],
        )
        self.assertNotIn(unittest.mock.call(1, "error"), run["statuses"])

    async def test_dedupe_spans_primary_and_fallback_queries(self) -> None:
        first = _lead("primary", phone="+380 (50) 123-45-67")
        duplicate = _lead("fallback", phone="380501234567")
        run = await self._run(
            niche="стоматология",
            streams={
                DENTISTRY_QUERIES[0]: [[first]],
                DENTISTRY_QUERIES[4]: [[duplicate]],
            },
            target=3,
        )

        self.assertEqual(sum(map(len, run["checked_batches"])), 1)
        self.assertEqual(
            [business.name for business in run["downstream"]["save"][0]],
            ["primary"],
        )

    async def test_removed_custom_niche_does_not_expand_to_car_service(self) -> None:
        run = await self._run(niche="шиномонтаж", streams={}, target=1)

        self.assertEqual(run["collector_queries"], ["шиномонтаж Киев"])
        self.assertNotIn("СТО Киев", run["collector_queries"])

    async def test_catalog_resolution_is_called_once_per_task(self) -> None:
        run = await self._run(niche="стоматология", streams={}, target=1)

        run["resolver"].assert_called_once_with("стоматология")

    async def test_single_downstream_pipeline_runs_after_all_discovery(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={DENTISTRY_QUERIES[0]: [[_lead("one")]]},
            target=2,
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            with self.subTest(name=name):
                self.assertEqual(len(run["downstream"][name]), 1)

    async def test_downstream_consumers_never_receive_more_than_target(self) -> None:
        run = await self._run(
            niche="стоматология",
            streams={
                DENTISTRY_QUERIES[0]: [
                    [_lead("one"), _lead("two"), _lead("three")]
                ]
            },
            target=2,
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            with self.subTest(name=name):
                self.assertEqual(len(run["downstream"][name][0]), 2)

    async def test_progress_keeps_original_alias_while_maps_uses_canonical_base(self) -> None:
        run = await self._run(niche="стоматолог", streams={}, target=1)

        self.assertEqual(run["collector_queries"][0], "стоматология Киев")
        self.assertTrue(any("«стоматолог»" in text for text in run["messages"]))


if __name__ == "__main__":
    unittest.main()

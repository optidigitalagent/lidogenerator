"""Synthetic integration tests for district-aware orchestration."""

import asyncio
from contextlib import ExitStack
import unittest
from unittest.mock import Mock, call, patch

import city_catalog
import orchestrator
from city_catalog import CityDefinition, DistrictDefinition
from models import Business
from niche_catalog import NicheSearchPlan


D1 = DistrictDefinition(
    key="d1",
    display_name="D1",
    query_text="D1",
    aliases=("District One",),
    enabled=True,
)
D2 = DistrictDefinition(
    key="d2",
    display_name="D2",
    query_text="D2 query",
    aliases=("District Two",),
    enabled=False,
)
D3 = DistrictDefinition(
    key="d3",
    display_name="D3",
    query_text="D3",
    enabled=True,
)
CITY_A = CityDefinition(
    key="city_a",
    canonical_name="City A",
    aliases=("A City",),
    districts=(D1, D2, D3),
)
CITY_B = CityDefinition(
    key="city_b",
    canonical_name="City B",
    aliases=(),
    districts=(),
)
SYNTHETIC_CITIES = (CITY_A, CITY_B)

D2_ENABLED = DistrictDefinition(
    key="d2",
    display_name="D2",
    query_text="D2 query",
    aliases=("District Two",),
    enabled=True,
)
CITY_A_WITH_D2 = CityDefinition(
    key="city_a",
    canonical_name="City A",
    aliases=("A City",),
    districts=(D2_ENABLED,),
)

KNOWN_PLAN = NicheSearchPlan(
    key="service",
    input_niche="Raw Service",
    base_niche="Service",
    primary_variants=("Primary 2",),
    fallback_variants=("Fallback 1",),
)
ALIAS_ONLY_PLAN = NicheSearchPlan(
    key="service",
    input_niche="Service Alias",
    base_niche="Service",
    primary_variants=("Primary 2",),
    fallback_variants=("Fallback 1",),
)
FALLBACK_INPUT_PLAN = NicheSearchPlan(
    key="service",
    input_niche="Fallback 1",
    base_niche="Service",
    primary_variants=("Primary 2",),
    fallback_variants=("Fallback 1",),
)

EXPECTED_DISTRICT_QUEUE = [
    "Service City A",
    "Primary 2 City A",
    "Service D1 City A",
    "Service D3 City A",
    "Fallback 1 City A",
]

_NO_REGISTRY_PATCH = object()


def _unknown_plan(niche: str) -> NicheSearchPlan:
    return NicheSearchPlan(
        key=None,
        input_niche=niche,
        base_niche=niche,
        primary_variants=(),
        fallback_variants=(),
    )


def _lead(name: str, *, phone: str = "") -> Business:
    return Business(
        name=name,
        phone=phone,
        instagram_url=f"https://instagram.com/{name}",
    )


def _candidate(name: str) -> Business:
    return Business(name=name)


class OrchestratorDistrictTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        niche: str = "Raw Service",
        city: str = "City A",
        plan: NicheSearchPlan = KNOWN_PLAN,
        registry: object = SYNTHETIC_CITIES,
        streams: dict[str, list[Business | None]] | None = None,
        target: int = 10,
        checked_limit: int = 100,
        opened_limit: int = 100,
        stop_event: asyncio.Event | None = None,
        stop_after_query: str | None = None,
    ) -> dict:
        streams = streams or {}
        collector_calls: list[dict] = []
        checked_batches: list[list[Business]] = []
        messages: list[str] = []
        allocation_calls: list[dict] = []
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

        niche_resolver = Mock(return_value=plan)
        city_resolver = Mock(wraps=city_catalog.resolve_city)
        district_resolver = Mock(wraps=city_catalog.enabled_districts)
        planner = Mock(wraps=orchestrator.build_query_queue)
        real_allocator = orchestrator.allocate_query_budget

        async def progress_callback(text: str) -> None:
            messages.append(text)

        async def fake_collect_stream(
            raw_niche,
            raw_city,
            max_businesses=None,
            progress_callback=None,
            stop_flag=None,
            query_text=None,
            **kwargs,
        ):
            nonlocal opened_total
            collector_calls.append(
                {
                    "niche": raw_niche,
                    "city": raw_city,
                    "max_businesses": max_businesses,
                    "query_text": query_text,
                }
            )
            opened_in_stream = 0
            extracted: list[Business] = []
            for card in streams.get(query_text, []):
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

        def record_allocation(**kwargs):
            allocation_calls.append(dict(kwargs))
            return real_allocator(**kwargs)

        task = {"niche": niche, "city": city, "count": target}
        with ExitStack() as stack:
            if registry is not _NO_REGISTRY_PATCH:
                stack.enter_context(
                    patch.object(
                        orchestrator.city_catalog,
                        "CITY_DEFINITIONS",
                        registry,
                    )
                )
            stack.enter_context(
                patch.object(
                    orchestrator.config,
                    "MAX_CHECKED_CANDIDATES_PER_TASK",
                    checked_limit,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.config,
                    "MAX_MAPS_CARDS_PER_TASK",
                    opened_limit,
                )
            )
            stack.enter_context(patch.object(orchestrator.db, "get_task", return_value=task))
            stack.enter_context(patch.object(orchestrator.db, "update_task_status", statuses))
            stack.enter_context(
                patch.object(
                    orchestrator.db,
                    "save_businesses",
                    side_effect=capture("save"),
                )
            )
            stack.enter_context(patch.object(orchestrator.db, "update_business"))
            stack.enter_context(
                patch.object(orchestrator, "resolve_niche_plan", niche_resolver)
            )
            stack.enter_context(
                patch.object(
                    orchestrator.city_catalog,
                    "resolve_city",
                    city_resolver,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.city_catalog,
                    "enabled_districts",
                    district_resolver,
                )
            )
            stack.enter_context(patch.object(orchestrator, "build_query_queue", planner))
            stack.enter_context(
                patch.object(
                    orchestrator,
                    "allocate_query_budget",
                    side_effect=record_allocation,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.collector,
                    "collect_stream",
                    new=fake_collect_stream,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.site_checker,
                    "check_sites",
                    new=fake_check_sites,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.social_checker,
                    "check_instagram",
                    new=fake_social,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.ai_scorer,
                    "score_businesses",
                    new=fake_score,
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.reporter,
                    "export_csv",
                    side_effect=capture("csv"),
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.reporter,
                    "export_excel",
                    side_effect=capture("excel"),
                )
            )
            stack.enter_context(
                patch.object(
                    orchestrator.reporter,
                    "format_leads_summary",
                    side_effect=capture("summary"),
                )
            )
            result = await orchestrator.run_search(
                1,
                progress_callback=progress_callback,
                stop_event=stop_event,
                progress_interval=0,
            )

        return {
            "result": result,
            "queries": [item["query_text"] for item in collector_calls],
            "collector_calls": collector_calls,
            "checked_batches": checked_batches,
            "checked_total": sum(map(len, checked_batches)),
            "opened_total": opened_total,
            "messages": messages,
            "allocation_calls": allocation_calls,
            "statuses": statuses.call_args_list,
            "downstream": downstream,
            "niche_resolver": niche_resolver,
            "city_resolver": city_resolver,
            "district_resolver": district_resolver,
            "planner": planner,
        }

    async def test_empty_production_registry_preserves_city_wide_queries(self) -> None:
        run = await self._run(registry=_NO_REGISTRY_PATCH)

        self.assertEqual(city_catalog.CITY_DEFINITIONS, ())
        self.assertEqual(
            run["queries"],
            ["Service City A", "Primary 2 City A", "Fallback 1 City A"],
        )

    async def test_known_city_canonical_includes_enabled_and_excludes_disabled(self) -> None:
        run = await self._run()

        self.assertIn("Service D1 City A", run["queries"])
        self.assertIn("Service D3 City A", run["queries"])
        self.assertFalse(any("D2" in query for query in run["queries"]))

    async def test_known_city_alias_uses_canonical_maps_and_raw_progress(self) -> None:
        run = await self._run(city="A City")

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE)
        self.assertTrue(any("A City" in message for message in run["messages"]))

    async def test_unknown_city_uses_raw_city_without_districts(self) -> None:
        run = await self._run(city="Unknown City")

        self.assertEqual(
            run["queries"],
            [
                "Service Unknown City",
                "Primary 2 Unknown City",
                "Fallback 1 Unknown City",
            ],
        )

    async def test_known_niche_known_city_has_exact_phase_order(self) -> None:
        run = await self._run()

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE)

    async def test_city_without_districts_remains_city_wide(self) -> None:
        run = await self._run(city="City B")

        self.assertEqual(
            run["queries"],
            ["Service City B", "Primary 2 City B", "Fallback 1 City B"],
        )

    async def test_known_city_unknown_niche_runs_one_city_wide_query(self) -> None:
        run = await self._run(
            niche="Custom Service",
            plan=_unknown_plan("Custom Service"),
        )

        self.assertEqual(run["queries"], ["Custom Service City A"])

    async def test_unknown_city_known_niche_keeps_phases_without_districts(self) -> None:
        run = await self._run(city="Unknown City")

        self.assertEqual(len(run["queries"]), 3)
        self.assertFalse(any(" D1 " in query or " D3 " in query for query in run["queries"]))

    async def test_alias_only_niche_uses_canonical_base_with_districts(self) -> None:
        run = await self._run(niche="Service Alias", plan=ALIAS_ONLY_PLAN)

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE)
        self.assertFalse(any("Service Alias" in query for query in run["queries"]))

    async def test_fallback_input_stays_after_primary_and_districts(self) -> None:
        run = await self._run(niche="Fallback 1", plan=FALLBACK_INPUT_PLAN)

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE)

    async def test_target_reached_before_first_district_skips_districts_and_fallback(self) -> None:
        run = await self._run(
            streams={"Primary 2 City A": [_lead("primary-lead")]},
            target=1,
        )

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE[:2])

    async def test_target_reached_during_d1_skips_d3_and_fallback(self) -> None:
        run = await self._run(
            streams={"Service D1 City A": [_lead("district-lead")]},
            target=1,
        )

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE[:3])
        for name in ("save", "social", "score", "csv", "excel", "summary"):
            self.assertEqual(len(run["downstream"][name][0]), 1)

    async def test_user_stop_before_first_district_returns_stopped(self) -> None:
        stop_event = asyncio.Event()
        run = await self._run(
            stop_event=stop_event,
            stop_after_query="Primary 2 City A",
        )

        self.assertIsNone(run["result"])
        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE[:2])
        self.assertIn(call(1, "stopped"), run["statuses"])
        self.assertNotIn(call(1, "error"), run["statuses"])

    async def test_checked_limit_before_district_exports_partial(self) -> None:
        run = await self._run(
            streams={"Primary 2 City A": [_candidate("checked")]},
            target=1,
            checked_limit=1,
        )

        self.assertEqual(run["result"], "result.xlsx")
        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE[:2])
        self.assertIn(call(1, "done", csv_path="result.xlsx"), run["statuses"])

    async def test_opened_card_limit_during_d1_skips_d3_and_fallback(self) -> None:
        run = await self._run(
            streams={
                "Service City A": [None],
                "Primary 2 City A": [None],
                "Service D1 City A": [None],
            },
            opened_limit=3,
        )

        self.assertEqual(run["queries"], EXPECTED_DISTRICT_QUEUE[:3])
        self.assertEqual(run["opened_total"], 3)
        self.assertEqual(run["result"], "result.xlsx")

    async def test_district_exhaustion_automatically_starts_fallback(self) -> None:
        run = await self._run()

        self.assertEqual(run["queries"][-3:], EXPECTED_DISTRICT_QUEUE[-3:])

    async def test_dedupe_spans_city_wide_and_district_queries(self) -> None:
        first = _lead("city", phone="+380 (50) 123-45-67")
        duplicate = _lead("district", phone="380501234567")
        run = await self._run(
            streams={
                "Service City A": [first],
                "Service D1 City A": [duplicate],
            }
        )

        self.assertEqual(run["checked_total"], 1)
        self.assertEqual(run["opened_total"], 2)
        self.assertEqual(
            [business.name for business in run["downstream"]["csv"][0]],
            ["city"],
        )

    async def test_dedupe_spans_d1_and_d3_queries(self) -> None:
        first = _lead("d1", phone="+380 (50) 123-45-67")
        duplicate = _lead("d3", phone="380501234567")
        run = await self._run(
            streams={
                "Service D1 City A": [first],
                "Service D3 City A": [duplicate],
            }
        )

        self.assertEqual(run["checked_total"], 1)
        self.assertEqual(run["opened_total"], 2)
        self.assertEqual(
            [business.name for business in run["downstream"]["csv"][0]],
            ["d1"],
        )

    async def test_fair_budget_initially_counts_all_five_queries(self) -> None:
        run = await self._run(checked_limit=10, opened_limit=10)

        self.assertEqual(run["allocation_calls"][0]["active_queries"], 5)
        self.assertEqual(run["collector_calls"][0]["max_businesses"], 2)

    async def test_known_niche_city_resolvers_are_called_once(self) -> None:
        run = await self._run()

        run["niche_resolver"].assert_called_once_with("Raw Service")
        run["city_resolver"].assert_called_once_with("City A", SYNTHETIC_CITIES)
        run["district_resolver"].assert_called_once_with(CITY_A)

    async def test_unknown_niche_does_not_call_enabled_districts(self) -> None:
        run = await self._run(
            niche="Custom Service",
            plan=_unknown_plan("Custom Service"),
        )

        run["district_resolver"].assert_not_called()

    async def test_planner_receives_exact_district_aware_contract(self) -> None:
        run = await self._run()

        run["planner"].assert_called_once_with(
            niche="Service",
            city="City A",
            niche_variants=("Primary 2",),
            districts=("D1", "D3"),
            fallback_variants=("Fallback 1",),
        )

    async def test_district_query_uses_query_text_not_display_name(self) -> None:
        run = await self._run(registry=(CITY_A_WITH_D2,))

        self.assertIn("Service D2 query City A", run["queries"])
        self.assertNotIn("Service D2 City A", run["queries"])

    async def test_progress_and_collector_metadata_keep_original_inputs(self) -> None:
        run = await self._run(
            niche="Service Alias",
            city="A City",
            plan=ALIAS_ONLY_PLAN,
        )

        self.assertEqual(run["queries"][0], "Service City A")
        self.assertTrue(
            all(
                item["niche"] == "Service Alias" and item["city"] == "A City"
                for item in run["collector_calls"]
            )
        )
        self.assertTrue(
            any(
                "Service Alias" in message and "A City" in message
                for message in run["messages"]
            )
        )

    async def test_downstream_pipeline_runs_once_after_all_discovery(self) -> None:
        run = await self._run(
            streams={"Service City A": [_lead("one")]},
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            self.assertEqual(len(run["downstream"][name]), 1)

    async def test_every_downstream_consumer_is_capped_to_target(self) -> None:
        run = await self._run(
            streams={
                "Service City A": [
                    _lead("one"),
                    _lead("two"),
                    _lead("three"),
                ]
            },
            target=2,
        )

        for name in ("save", "social", "score", "csv", "excel", "summary"):
            self.assertEqual(len(run["downstream"][name][0]), 2)

    async def test_districts_do_not_cross_product_with_variants(self) -> None:
        run = await self._run()

        self.assertNotIn("Primary 2 D1 City A", run["queries"])
        self.assertNotIn("Fallback 1 D1 City A", run["queries"])

    async def test_scoped_registry_patch_restores_empty_production_catalog(self) -> None:
        self.assertEqual(orchestrator.city_catalog.CITY_DEFINITIONS, ())

        await self._run()

        self.assertEqual(orchestrator.city_catalog.CITY_DEFINITIONS, ())


if __name__ == "__main__":
    unittest.main()

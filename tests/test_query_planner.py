"""Tests for the pure query planning and queue contract."""

import unittest
from dataclasses import FrozenInstanceError

from query_planner import (
    QueryKind,
    QueryPlan,
    QueryQueue,
    SearchQuery,
    build_query_plan,
    build_query_queue,
    normalize_query_key,
)


class BuildQueryQueueTests(unittest.TestCase):
    def test_builds_base_query(self) -> None:
        queue = build_query_queue("Service", "City")

        self.assertEqual(queue.total_queries, 1)
        self.assertEqual(queue.remaining_queries, 1)
        self.assertEqual(queue.queries[0].text, "Service City")
        self.assertEqual(queue.queries[0].kind, QueryKind.BASE)

    def test_preserves_legacy_behavior_without_districts_or_fallbacks(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Variant 1", "Variant 2"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["Service City", "Variant 1 City", "Variant 2 City"],
        )

    def test_builds_district_v1_plan_in_exact_phase_order(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary 2", "Primary 3"),
            districts=("D1", "D2", "D3"),
            fallback_variants=("Fallback 1", "Fallback 2"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "Service City",
                "Primary 2 City",
                "Primary 3 City",
                "Service D1 City",
                "Service D2 City",
                "Service D3 City",
                "Fallback 1 City",
                "Fallback 2 City",
            ],
        )
        self.assertEqual(
            [
                (query.kind, query.variant, query.district)
                for query in queue.queries
            ],
            [
                (QueryKind.BASE, None, None),
                (QueryKind.NICHE_VARIANT, "Primary 2", None),
                (QueryKind.NICHE_VARIANT, "Primary 3", None),
                (QueryKind.DISTRICT, None, "D1"),
                (QueryKind.DISTRICT, None, "D2"),
                (QueryKind.DISTRICT, None, "D3"),
                (QueryKind.NICHE_VARIANT, "Fallback 1", None),
                (QueryKind.NICHE_VARIANT, "Fallback 2", None),
            ],
        )

    def test_does_not_build_variant_district_cross_product(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary 2", "Primary 3"),
            districts=("D1", "D2", "D3"),
            fallback_variants=("Fallback 1", "Fallback 2"),
        )

        texts = {query.text for query in queue.queries}
        self.assertNotIn("Primary 2 D1 City", texts)
        self.assertNotIn("Primary 3 D2 City", texts)
        self.assertNotIn("Fallback 1 D1 City", texts)
        self.assertTrue(
            all(
                query.kind is not QueryKind.DISTRICT_VARIANT
                for query in queue.queries
            )
        )

    def test_seven_city_queries_plus_ten_districts_builds_seventeen(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=tuple(f"Primary {index}" for index in range(2, 5)),
            districts=tuple(f"D{index}" for index in range(1, 11)),
            fallback_variants=tuple(
                f"Fallback {index}" for index in range(1, 4)
            ),
        )

        self.assertEqual(queue.total_queries, 17)

    def test_query_count_is_city_wide_plus_districts(self) -> None:
        cases = ((3, 3, 6), (3, 10, 13), (5, 5, 10), (7, 10, 17), (8, 15, 23))

        for city_wide, district_count, expected in cases:
            with self.subTest(
                city_wide=city_wide,
                districts=district_count,
            ):
                primary_count = max(0, city_wide - 2)
                fallback_count = city_wide - 1 - primary_count
                queue = build_query_queue(
                    "Service",
                    "City",
                    niche_variants=tuple(
                        f"Primary {index}"
                        for index in range(1, primary_count + 1)
                    ),
                    districts=tuple(
                        f"D{index}" for index in range(1, district_count + 1)
                    ),
                    fallback_variants=tuple(
                        f"Fallback {index}"
                        for index in range(1, fallback_count + 1)
                    ),
                )
                self.assertEqual(queue.total_queries, expected)

    def test_deduplicates_base_and_primary_variant(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=(" service ", "Primary"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["Service City", "Primary City"],
        )

    def test_deduplicates_primary_and_fallback_preserving_primary(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary",),
            fallback_variants=(" primary ",),
        )

        self.assertEqual(queue.total_queries, 2)
        self.assertEqual(queue.queries[1].variant, "Primary")

    def test_deduplicates_fallbacks_preserving_first_appearance(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            fallback_variants=("Fallback", " fallback ", "Fallback 2"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["Service City", "Fallback City", "Fallback 2 City"],
        )
        self.assertEqual(queue.queries[1].variant, "Fallback")

    def test_deduplicates_districts_preserving_first_appearance(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            districts=("D1", " d1 ", "D2"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["Service City", "Service D1 City", "Service D2 City"],
        )
        self.assertEqual(queue.queries[1].district, "D1")

    def test_deduplication_uses_full_query_key_across_phases(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Service D1",),
            districts=(" d1 ", "D2"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["Service City", "Service D1 City", "Service D2 City"],
        )
        self.assertEqual(queue.queries[1].kind, QueryKind.NICHE_VARIANT)

    def test_deduplication_does_not_use_substring_matching(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            districts=("D1", "D"),
        )

        self.assertEqual(queue.total_queries, 3)

    def test_supports_empty_primary_district_and_fallback_phases(self) -> None:
        cases = (
            ({}, ["Service City"]),
            ({"districts": ("D1",)}, ["Service City", "Service D1 City"]),
            (
                {"fallback_variants": ("Fallback",)},
                ["Service City", "Fallback City"],
            ),
            (
                {"niche_variants": ("Primary",)},
                ["Service City", "Primary City"],
            ),
        )

        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                queue = build_query_queue("Service", "City", **arguments)
                self.assertEqual([query.text for query in queue.queries], expected)

    def test_max_queries_one_returns_only_base(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary",),
            districts=("D1",),
            fallback_variants=("Fallback",),
            max_queries=1,
        )

        self.assertEqual([query.text for query in queue.queries], ["Service City"])

    def test_max_queries_can_end_inside_primary_phase(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary 1", "Primary 2"),
            districts=("D1",),
            fallback_variants=("Fallback",),
            max_queries=2,
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["Service City", "Primary 1 City"],
        )

    def test_max_queries_can_end_inside_district_phase(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary",),
            districts=("D1", "D2", "D3"),
            fallback_variants=("Fallback",),
            max_queries=4,
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "Service City",
                "Primary City",
                "Service D1 City",
                "Service D2 City",
            ],
        )

    def test_max_queries_can_end_inside_fallback_phase(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary",),
            districts=("D1",),
            fallback_variants=("Fallback 1", "Fallback 2"),
            max_queries=4,
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "Service City",
                "Primary City",
                "Service D1 City",
                "Fallback 1 City",
            ],
        )

    def test_max_queries_above_full_plan_preserves_complete_order(self) -> None:
        arguments = {
            "niche_variants": ("Primary",),
            "districts": ("D1",),
            "fallback_variants": ("Fallback",),
        }
        full_queue = build_query_queue("Service", "City", **arguments)
        limited_queue = build_query_queue(
            "Service",
            "City",
            **arguments,
            max_queries=20,
        )

        self.assertEqual(limited_queue.queries, full_queue.queries)

    def test_max_queries_is_applied_after_global_deduplication(self) -> None:
        queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Service", "Primary"),
            districts=("D1",),
            fallback_variants=("primary", "Fallback"),
            max_queries=4,
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "Service City",
                "Primary City",
                "Service D1 City",
                "Fallback City",
            ],
        )

    def test_normalizes_all_generated_text_and_metadata(self) -> None:
        queue = build_query_queue(
            "  Service   Type ",
            " City   Center ",
            niche_variants=(" Primary   2 ",),
            districts=(" D1   Area ",),
            fallback_variants=(" Fallback   1 ",),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "Service Type City Center",
                "Primary 2 City Center",
                "Service Type D1 Area City Center",
                "Fallback 1 City Center",
            ],
        )
        self.assertEqual(queue.queries[2].district, "D1 Area")
        self.assertEqual(queue.queries[3].variant, "Fallback 1")

    def test_does_not_mutate_input_lists(self) -> None:
        primary = ["Primary 1", "Primary 2"]
        districts = ["D1", "D2"]
        fallbacks = ["Fallback 1", "Fallback 2"]
        snapshots = (primary.copy(), districts.copy(), fallbacks.copy())

        build_query_queue(
            "Service",
            "City",
            niche_variants=primary,
            districts=districts,
            fallback_variants=fallbacks,
        )

        self.assertEqual((primary, districts, fallbacks), snapshots)

    def test_district_variant_enum_value_remains_compatible(self) -> None:
        self.assertEqual(QueryKind.DISTRICT_VARIANT.value, "district_variant")


class BuildQueryPlanTests(unittest.TestCase):
    def test_preserves_normal_order_and_builds_deterministic_deep_phase(self) -> None:
        plan = build_query_plan(
            "Service",
            "City",
            niche_variants=("Primary 2", "Primary 3"),
            districts=("D1", "D2"),
            fallback_variants=("Fallback 1",),
        )

        self.assertEqual(
            [query.text for query in plan.normal_queries.queries],
            [
                "Service City",
                "Primary 2 City",
                "Primary 3 City",
                "Service D1 City",
                "Service D2 City",
                "Fallback 1 City",
            ],
        )
        self.assertEqual(
            [query.text for query in plan.deep_queries.queries],
            [
                "Primary 2 D1 City",
                "Primary 2 D2 City",
                "Primary 3 D1 City",
                "Primary 3 D2 City",
            ],
        )
        self.assertTrue(
            all(
                query.kind is QueryKind.DISTRICT_VARIANT
                for query in plan.deep_queries.queries
            )
        )

    def test_deep_phase_deduplicates_against_normal_and_itself(self) -> None:
        plan = build_query_plan(
            "Service",
            "City",
            niche_variants=("Service D1", " primary ", "PRIMARY"),
            districts=("D1", " d1 "),
        )

        all_queries = (*plan.normal_queries.queries, *plan.deep_queries.queries)
        self.assertEqual(len({query.key for query in all_queries}), len(all_queries))
        self.assertNotIn(
            "Service D1 City",
            [query.text for query in plan.deep_queries.queries],
        )

    def test_no_district_city_has_normal_queries_only(self) -> None:
        plan = build_query_plan(
            "Service",
            "Unknown City",
            niche_variants=("Primary",),
            districts=(),
            fallback_variants=("Fallback",),
        )

        self.assertEqual(plan.normal_queries_total, 3)
        self.assertEqual(plan.deep_queries_total, 0)
        self.assertTrue(plan.deep_queries.exhausted)

    def test_deep_phase_uses_primary_variants_not_fallback_variants(self) -> None:
        plan = build_query_plan(
            "Service",
            "City",
            niche_variants=("Primary",),
            districts=("D1",),
            fallback_variants=("Fallback",),
        )

        self.assertEqual(
            [query.text for query in plan.deep_queries.queries],
            ["Primary D1 City"],
        )
        self.assertNotIn(
            "Fallback D1 City",
            [query.text for query in plan.deep_queries.queries],
        )

    def test_query_plan_rejects_cross_phase_duplicate_text(self) -> None:
        query = SearchQuery("Service City", "Service", "City", QueryKind.BASE)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            QueryPlan(QueryQueue((query,)), QueryQueue((query,)))


class QueryQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = build_query_queue(
            "Service",
            "City",
            niche_variants=("Primary",),
        )

    def test_take_next_advances_new_queue_without_mutating_original(self) -> None:
        query, new_queue = self.queue.take_next()

        self.assertEqual(query, self.queue.queries[0])
        self.assertEqual(self.queue.next_index, 0)
        self.assertEqual(self.queue.remaining_queries, 2)
        self.assertEqual(new_queue.next_index, 1)
        self.assertEqual(new_queue.remaining_queries, 1)
        self.assertEqual(new_queue.current_query, self.queue.queries[1])

    def test_sequential_take_next_preserves_order_and_exhausts_queue(self) -> None:
        state = self.queue
        taken = []
        while not state.exhausted:
            query, state = state.take_next()
            taken.append(query)

        self.assertEqual(taken, list(self.queue.queries))
        self.assertTrue(state.exhausted)
        self.assertEqual(state.remaining_queries, 0)
        self.assertIsNone(state.current_query)

        query, same_state = state.take_next()
        self.assertIsNone(query)
        self.assertIs(same_state, state)

    def test_reports_total_independently_of_current_index(self) -> None:
        advanced = QueryQueue(self.queue.queries, next_index=1)

        self.assertEqual(advanced.total_queries, 2)
        self.assertEqual(advanced.remaining_queries, 1)
        self.assertFalse(advanced.exhausted)


class NormalizationTests(unittest.TestCase):
    def test_query_keys_ignore_case_and_repeated_whitespace(self) -> None:
        first = SearchQuery(
            "Service   D1 City",
            "Service",
            "City",
            QueryKind.DISTRICT,
            district="D1",
        )
        second = SearchQuery(
            "service d1   city",
            "Service",
            "City",
            QueryKind.DISTRICT,
            district="D1",
        )

        self.assertEqual(first.key, second.key)

    def test_normalize_query_key_collapses_spacing_and_case(self) -> None:
        self.assertEqual(normalize_query_key(" SERVICE   City "), "service city")

    def test_normalize_query_key_preserves_meaningful_characters(self) -> None:
        self.assertEqual(normalize_query_key("D'1 & D-2"), "d'1 & d-2")


class ValidationTests(unittest.TestCase):
    def test_rejects_invalid_niche_and_city(self) -> None:
        invalid_values = ("", " \t\n ", None, 1, True)
        for value in invalid_values:
            with self.subTest(field="niche", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue(value, "City")  # type: ignore[arg-type]
            with self.subTest(field="city", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue("Service", value)  # type: ignore[arg-type]

    def test_rejects_string_instead_of_each_sequence(self) -> None:
        for field in ("niche_variants", "districts", "fallback_variants"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, field):
                    build_query_queue("Service", "City", **{field: "value"})

    def test_rejects_empty_or_non_string_elements_in_each_sequence(self) -> None:
        invalid_values = ("", "   ", None, 3, True)
        for field in ("niche_variants", "districts", "fallback_variants"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        build_query_queue(
                            "Service",
                            "City",
                            **{field: (value,)},
                        )

    def test_rejects_non_sequence_containers(self) -> None:
        for field in ("niche_variants", "districts", "fallback_variants"):
            for value in (None, 3, True, {"value"}, {"key": "value"}):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(TypeError, field):
                        build_query_queue(
                            "Service",
                            "City",
                            **{field: value},
                        )

    def test_fallback_variants_accept_tuple_and_list(self) -> None:
        for fallbacks in (("Fallback 1",), [" Fallback   1 "]):
            with self.subTest(container=type(fallbacks).__name__):
                queue = build_query_queue(
                    "Service",
                    "City",
                    fallback_variants=fallbacks,
                )
                self.assertEqual(queue.queries[1].text, "Fallback 1 City")

    def test_rejects_invalid_max_queries(self) -> None:
        for value in (0, -1, 1.5, "1", True, False):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue(
                        "Service",
                        "City",
                        max_queries=value,
                    )  # type: ignore[arg-type]

    def test_query_queue_rejects_non_tuple_queries(self) -> None:
        with self.assertRaisesRegex(TypeError, "queries"):
            QueryQueue([])  # type: ignore[arg-type]

    def test_query_queue_rejects_invalid_query_element(self) -> None:
        with self.assertRaisesRegex(TypeError, r"queries\[0\]"):
            QueryQueue(("query",))  # type: ignore[arg-type]

    def test_query_queue_rejects_invalid_next_index(self) -> None:
        query = SearchQuery("Service City", "Service", "City", QueryKind.BASE)
        for value in (-1, 2, 1.5, "1", True, False):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    QueryQueue((query,), value)  # type: ignore[arg-type]

    def test_search_query_rejects_invalid_kind(self) -> None:
        with self.assertRaisesRegex(TypeError, "kind"):
            SearchQuery("Service City", "Service", "City", "base")  # type: ignore[arg-type]

    def test_search_query_rejects_empty_text_and_optional_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "text"):
            SearchQuery("  ", "Service", "City", QueryKind.BASE)
        with self.assertRaisesRegex(ValueError, "variant"):
            SearchQuery(
                "Service City",
                "Service",
                "City",
                QueryKind.BASE,
                variant=" ",
            )
        with self.assertRaisesRegex(ValueError, "district"):
            SearchQuery(
                "Service City",
                "Service",
                "City",
                QueryKind.BASE,
                district=" ",
            )

    def test_search_query_rejects_invalid_field_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "text"):
            SearchQuery(None, "Service", "City", QueryKind.BASE)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "variant"):
            SearchQuery(
                "Service City",
                "Service",
                "City",
                QueryKind.BASE,
                variant=1,
            )  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "district"):
            SearchQuery(
                "Service City",
                "Service",
                "City",
                QueryKind.BASE,
                district=1,
            )  # type: ignore[arg-type]

    def test_models_are_immutable(self) -> None:
        query = SearchQuery("Service City", "Service", "City", QueryKind.BASE)
        queue = QueryQueue((query,))
        for instance, field_name in ((query, "text"), (queue, "next_index")):
            with self.subTest(instance=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, "changed")


if __name__ == "__main__":
    unittest.main()

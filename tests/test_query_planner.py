"""Tests for the pure query planning and queue contract."""

import unittest
from dataclasses import FrozenInstanceError

from query_planner import (
    QueryKind,
    QueryQueue,
    SearchQuery,
    build_query_queue,
    normalize_query_key,
)


class BuildQueryQueueTests(unittest.TestCase):
    def test_builds_base_query(self) -> None:
        queue = build_query_queue("стоматология", "Киев")

        self.assertEqual(queue.total_queries, 1)
        self.assertEqual(queue.remaining_queries, 1)
        self.assertEqual(queue.queries[0].text, "стоматология Киев")
        self.assertEqual(queue.queries[0].kind, QueryKind.BASE)

    def test_builds_full_plan_in_stable_order(self) -> None:
        queue = build_query_queue(
            "стоматология",
            "Киев",
            niche_variants=("стоматологія", "стоматологическая клиника"),
            districts=("Оболонь", "Позняки"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "стоматология Киев",
                "стоматологія Киев",
                "стоматологическая клиника Киев",
                "стоматология Оболонь Киев",
                "стоматология Позняки Киев",
                "стоматологія Оболонь Киев",
                "стоматологія Позняки Киев",
                "стоматологическая клиника Оболонь Киев",
                "стоматологическая клиника Позняки Киев",
            ],
        )
        self.assertEqual(
            [query.kind for query in queue.queries],
            [
                QueryKind.BASE,
                QueryKind.NICHE_VARIANT,
                QueryKind.NICHE_VARIANT,
                QueryKind.DISTRICT,
                QueryKind.DISTRICT,
                QueryKind.DISTRICT_VARIANT,
                QueryKind.DISTRICT_VARIANT,
                QueryKind.DISTRICT_VARIANT,
                QueryKind.DISTRICT_VARIANT,
            ],
        )

    def test_deduplicates_variants_and_preserves_first_appearance(self) -> None:
        queue = build_query_queue(
            "стоматология",
            "Киев",
            niche_variants=(
                "стоматология",
                "СТОМАТОЛОГИЯ",
                "  стоматологія",
                "стоматологія",
            ),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            ["стоматология Киев", "стоматологія Киев"],
        )
        self.assertEqual(queue.queries[1].variant, "стоматологія")

    def test_deduplicates_districts_and_preserves_first_appearance(self) -> None:
        queue = build_query_queue(
            "стоматология",
            "Киев",
            districts=("Оболонь", "оболонь", "  Позняки", "Позняки"),
        )

        self.assertEqual(
            [query.text for query in queue.queries],
            [
                "стоматология Киев",
                "стоматология Оболонь Киев",
                "стоматология Позняки Киев",
            ],
        )

    def test_max_queries_limits_deduplicated_plan(self) -> None:
        full_queue = build_query_queue(
            "стоматология",
            "Киев",
            niche_variants=("стоматология", "стоматологія", "клиника"),
            districts=("Оболонь", "Позняки"),
        )
        limited_queue = build_query_queue(
            "стоматология",
            "Киев",
            niche_variants=("стоматология", "стоматологія", "клиника"),
            districts=("Оболонь", "Позняки"),
            max_queries=3,
        )

        self.assertGreater(full_queue.total_queries, 3)
        self.assertEqual(limited_queue.queries, full_queue.queries[:3])
        self.assertEqual(limited_queue.remaining_queries, 3)
        self.assertEqual(limited_queue.queries[0].kind, QueryKind.BASE)

    def test_normalizes_all_generated_text_and_metadata(self) -> None:
        queue = build_query_queue(
            "  dental   clinic ",
            " New   York ",
            niche_variants=(" family   dentist ",),
            districts=(" Upper   East Side ",),
        )

        self.assertEqual(queue.queries[0].text, "dental clinic New York")
        self.assertEqual(queue.queries[-1].text, "family dentist Upper East Side New York")
        self.assertEqual(queue.queries[-1].niche, "dental clinic")
        self.assertEqual(queue.queries[-1].city, "New York")
        self.assertEqual(queue.queries[-1].variant, "family dentist")
        self.assertEqual(queue.queries[-1].district, "Upper East Side")


class QueryQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = build_query_queue(
            "стоматология",
            "Киев",
            niche_variants=("стоматологія",),
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
            "Стоматология   Оболонь Киев",
            "стоматология",
            "Киев",
            QueryKind.DISTRICT,
            district="Оболонь",
        )
        second = SearchQuery(
            "стоматология оболонь   киев",
            "стоматология",
            "Киев",
            QueryKind.DISTRICT,
            district="Оболонь",
        )

        self.assertEqual(first.key, second.key)

    def test_normalize_query_key_supports_multiple_languages(self) -> None:
        examples = (
            (" СТОМАТОЛОГИЯ   Киев ", "стоматология киев"),
            (" СТОМАТОЛОГІЯ   Київ ", "стоматологія київ"),
            (" DENTAL   Clinic ", "dental clinic"),
        )
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(normalize_query_key(text), expected)

    def test_normalize_query_key_preserves_meaningful_characters(self) -> None:
        self.assertEqual(normalize_query_key("L'viv & Київ"), "l'viv & київ")


class ValidationTests(unittest.TestCase):
    def test_rejects_invalid_niche_and_city(self) -> None:
        invalid_values = ("", " \t\n ", None, 1, True)
        for value in invalid_values:
            with self.subTest(field="niche", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue(value, "Киев")  # type: ignore[arg-type]
            with self.subTest(field="city", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue("стоматология", value)  # type: ignore[arg-type]

    def test_rejects_string_instead_of_sequence(self) -> None:
        with self.assertRaisesRegex(TypeError, "niche_variants"):
            build_query_queue("niche", "city", niche_variants="variant")
        with self.assertRaisesRegex(TypeError, "districts"):
            build_query_queue("niche", "city", districts="district")

    def test_rejects_empty_or_non_string_sequence_elements(self) -> None:
        invalid_values = ("", "   ", None, 3, True)
        for value in invalid_values:
            with self.subTest(field="niche_variants", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue(
                        "niche",
                        "city",
                        niche_variants=(value,),  # type: ignore[arg-type]
                    )
            with self.subTest(field="districts", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue(
                        "niche",
                        "city",
                        districts=(value,),  # type: ignore[arg-type]
                    )

    def test_rejects_invalid_max_queries(self) -> None:
        for value in (0, -1, 1.5, "1", True, False):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    build_query_queue("niche", "city", max_queries=value)  # type: ignore[arg-type]

    def test_query_queue_rejects_non_tuple_queries(self) -> None:
        with self.assertRaisesRegex(TypeError, "queries"):
            QueryQueue([])  # type: ignore[arg-type]

    def test_query_queue_rejects_invalid_query_element(self) -> None:
        with self.assertRaisesRegex(TypeError, r"queries\[0\]"):
            QueryQueue(("query",))  # type: ignore[arg-type]

    def test_query_queue_rejects_invalid_next_index(self) -> None:
        query = SearchQuery("niche city", "niche", "city", QueryKind.BASE)
        for value in (-1, 2, 1.5, "1", True, False):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    QueryQueue((query,), value)  # type: ignore[arg-type]

    def test_search_query_rejects_invalid_kind(self) -> None:
        with self.assertRaisesRegex(TypeError, "kind"):
            SearchQuery("niche city", "niche", "city", "base")  # type: ignore[arg-type]

    def test_search_query_rejects_empty_text_and_optional_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "text"):
            SearchQuery("  ", "niche", "city", QueryKind.BASE)
        with self.assertRaisesRegex(ValueError, "variant"):
            SearchQuery("niche city", "niche", "city", QueryKind.BASE, variant=" ")
        with self.assertRaisesRegex(ValueError, "district"):
            SearchQuery("niche city", "niche", "city", QueryKind.BASE, district=" ")

    def test_search_query_rejects_invalid_field_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "text"):
            SearchQuery(None, "niche", "city", QueryKind.BASE)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "variant"):
            SearchQuery("niche city", "niche", "city", QueryKind.BASE, variant=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "district"):
            SearchQuery("niche city", "niche", "city", QueryKind.BASE, district=1)  # type: ignore[arg-type]

    def test_models_are_immutable(self) -> None:
        query = SearchQuery("niche city", "niche", "city", QueryKind.BASE)
        queue = QueryQueue((query,))
        for instance, field_name in ((query, "text"), (queue, "next_index")):
            with self.subTest(instance=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, "changed")


if __name__ == "__main__":
    unittest.main()

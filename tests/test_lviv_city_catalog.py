"""Production data and deep-query tests for Lviv districts."""

import unittest

import city_catalog
from city_catalog import enabled_districts, resolve_city, resolve_district
from niche_catalog import resolve_niche_plan
from query_planner import QueryKind, build_query_plan


DISTRICT_KEYS = (
    "halytskyi",
    "zaliznychnyi",
    "lychakivskyi",
    "sykhivskyi",
    "frankivskyi",
    "shevchenkivskyi",
)
DISTRICT_TEXTS = (
    "Галицький район",
    "Залізничний район",
    "Личаківський район",
    "Сихівський район",
    "Франківський район",
    "Шевченківський район",
)
RUSSIAN_ALIASES = (
    "Галицкий район",
    "Железнодорожный район",
    "Лычаковский район",
    "Сиховский район",
    "Франковский район",
    "Шевченковский район",
)


class LvivCityCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lviv = resolve_city("Львов", city_catalog.CITY_DEFINITIONS)
        assert self.lviv is not None

    def test_has_six_official_enabled_districts_in_declared_order(self) -> None:
        self.assertEqual(
            tuple(district.key for district in self.lviv.districts),
            DISTRICT_KEYS,
        )
        self.assertEqual(
            tuple(district.query_text for district in self.lviv.districts),
            DISTRICT_TEXTS,
        )
        self.assertEqual(enabled_districts(self.lviv), self.lviv.districts)

    def test_ukrainian_query_text_and_russian_aliases_resolve(self) -> None:
        for key, ukrainian, russian in zip(
            DISTRICT_KEYS,
            DISTRICT_TEXTS,
            RUSSIAN_ALIASES,
        ):
            with self.subTest(key=key):
                self.assertEqual(resolve_district(self.lviv, ukrainian).key, key)
                self.assertEqual(resolve_district(self.lviv, russian).key, key)

    def test_does_not_add_speculative_micro_neighborhoods(self) -> None:
        for value in ("Левандівка", "Сихів", "Підзамче", "Рясне"):
            with self.subTest(value=value):
                self.assertIsNone(resolve_district(self.lviv, value))


class ProductionDeepQueryTests(unittest.TestCase):
    def _dentistry_plan(self, city_name: str):
        niche = resolve_niche_plan("стоматология")
        city = resolve_city(city_name, city_catalog.CITY_DEFINITIONS)
        assert city is not None
        districts = tuple(item.query_text for item in enabled_districts(city))
        return build_query_plan(
            niche=niche.base_niche,
            city=city.canonical_name,
            niche_variants=niche.primary_variants,
            districts=districts,
            fallback_variants=niche.fallback_variants,
        )

    def test_kyiv_dentistry_primary_district_variants_are_generated(self) -> None:
        plan = self._dentistry_plan("Киев")

        self.assertEqual(plan.normal_queries_total, 17)
        self.assertEqual(plan.deep_queries_total, 30)
        self.assertIn(
            "стоматологія Голосіївський район Київ",
            [query.text for query in plan.deep_queries.queries],
        )

    def test_lviv_dentistry_primary_district_variants_are_generated(self) -> None:
        plan = self._dentistry_plan("Львов")

        self.assertEqual(plan.normal_queries_total, 13)
        self.assertEqual(plan.deep_queries_total, 18)
        self.assertIn(
            "стоматологія Галицький район Львів",
            [query.text for query in plan.deep_queries.queries],
        )
        self.assertTrue(
            all(
                query.kind is QueryKind.DISTRICT_VARIANT
                for query in plan.deep_queries.queries
            )
        )


if __name__ == "__main__":
    unittest.main()

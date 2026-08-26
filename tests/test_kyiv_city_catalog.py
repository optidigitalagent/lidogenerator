"""Production data tests for the Kyiv administrative-district catalog."""

import dataclasses
import unittest

import city_catalog
from city_catalog import (
    build_city_index,
    build_district_index,
    enabled_districts,
    normalize_district_key,
    resolve_city,
    resolve_district,
)


DISTRICT_KEYS = (
    "holosiivskyi",
    "darnytskyi",
    "desnianskyi",
    "dniprovskyi",
    "obolonskyi",
    "pecherskyi",
    "podilskyi",
    "sviatoshynskyi",
    "solomianskyi",
    "shevchenkivskyi",
)

DISTRICT_TEXTS = (
    "Голосіївський район",
    "Дарницький район",
    "Деснянський район",
    "Дніпровський район",
    "Оболонський район",
    "Печерський район",
    "Подільський район",
    "Святошинський район",
    "Солом’янський район",
    "Шевченківський район",
)

UKRAINIAN_ALIASES = (
    "Голосіївський",
    "Дарницький",
    "Деснянський",
    "Дніпровський",
    "Оболонський",
    "Печерський",
    "Подільський",
    "Святошинський",
    "Солом’янський",
    "Шевченківський",
)

RUSSIAN_ALIASES = (
    "Голосеевский район",
    "Дарницкий район",
    "Деснянский район",
    "Днепровский район",
    "Оболонский район",
    "Печерский район",
    "Подольский район",
    "Святошинский район",
    "Соломенский район",
    "Шевченковский район",
)

EXCLUDED_NEIGHBORHOODS = (
    "Оболонь",
    "Троєщина",
    "Троещина",
    "Позняки",
    "Осокорки",
    "Поділ",
    "Печерськ",
    "Солом’янка",
    "Борщагівка",
    "Нивки",
    "Теремки",
    "Виноградар",
)


class KyivCityCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = city_catalog.CITY_DEFINITIONS
        self.kyiv = self.registry[0]

    def test_registry_keeps_kyiv_first_and_is_immutable(self) -> None:
        self.assertIs(type(self.registry), tuple)
        self.assertEqual(self.kyiv.key, "kyiv")
        self.assertNotIn(
            "City A",
            {city.canonical_name for city in self.registry},
        )
        self.assertNotIn(
            "City B",
            {city.canonical_name for city in self.registry},
        )
        with self.assertRaises(AttributeError):
            self.registry.append(self.kyiv)  # type: ignore[attr-defined]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.kyiv.key = "other"  # type: ignore[misc]

    def test_city_uses_exact_canonical_name_and_alias_order(self) -> None:
        self.assertEqual(self.kyiv.canonical_name, "Київ")
        self.assertEqual(self.kyiv.aliases, ("Киев", "Kyiv", "Kiev"))

    def test_city_resolution_accepts_canonical_and_declared_aliases(self) -> None:
        for value in ("Київ", "київ", "Киев", "КИЕВ", "Kyiv", "Kiev"):
            with self.subTest(value=value):
                self.assertIs(resolve_city(value, self.registry), self.kyiv)

    def test_city_resolution_remains_exact_only(self) -> None:
        for value in (
            "Киев город",
            "город Киев",
            "Киевская область",
            "Kyiv City Center",
        ):
            with self.subTest(value=value):
                self.assertIsNone(resolve_city(value, self.registry))

    def test_district_count_keys_and_order_are_exact(self) -> None:
        self.assertEqual(len(self.kyiv.districts), 10)
        self.assertEqual(
            tuple(district.key for district in self.kyiv.districts),
            DISTRICT_KEYS,
        )

    def test_all_districts_are_enabled_in_declared_order(self) -> None:
        self.assertTrue(all(district.enabled for district in self.kyiv.districts))
        self.assertEqual(enabled_districts(self.kyiv), self.kyiv.districts)

    def test_display_and_query_texts_are_official_and_identical(self) -> None:
        self.assertEqual(
            tuple(district.display_name for district in self.kyiv.districts),
            DISTRICT_TEXTS,
        )
        self.assertEqual(
            tuple(district.query_text for district in self.kyiv.districts),
            DISTRICT_TEXTS,
        )
        self.assertTrue(
            all("район" in district.query_text for district in self.kyiv.districts)
        )

    def test_ukrainian_aliases_resolve_to_their_districts(self) -> None:
        for key, alias in zip(DISTRICT_KEYS, UKRAINIAN_ALIASES):
            with self.subTest(alias=alias):
                self.assertEqual(resolve_district(self.kyiv, alias).key, key)

    def test_russian_aliases_resolve_to_their_districts(self) -> None:
        for key, alias in zip(DISTRICT_KEYS, RUSSIAN_ALIASES):
            with self.subTest(alias=alias):
                self.assertEqual(resolve_district(self.kyiv, alias).key, key)

    def test_solomiansky_typographic_and_ascii_apostrophes_resolve(self) -> None:
        for value in ("Солом’янський район", "Солом'янський район"):
            with self.subTest(value=value):
                self.assertEqual(
                    resolve_district(self.kyiv, value).key,
                    "solomianskyi",
                )

    def test_district_resolution_remains_exact_only(self) -> None:
        for value in (
            "Голосіїв",
            "Дарниця",
            "Деснянский административный район",
        ):
            with self.subTest(value=value):
                self.assertIsNone(resolve_district(self.kyiv, value))

    def test_neighborhoods_and_area_names_are_excluded(self) -> None:
        for value in EXCLUDED_NEIGHBORHOODS:
            with self.subTest(value=value):
                self.assertIsNone(resolve_district(self.kyiv, value))

    def test_city_and_district_indexes_build_without_conflicts(self) -> None:
        city_index = build_city_index((self.kyiv,))
        district_index = build_district_index(self.kyiv)

        self.assertEqual(len(city_index), 4)
        self.assertEqual(len(district_index), 42)
        self.assertEqual(
            len({district.key for district in self.kyiv.districts}),
            10,
        )
        normalized_terms = {
            normalize_district_key(term)
            for district in self.kyiv.districts
            for term in (
                district.display_name,
                district.query_text,
                *district.aliases,
            )
        }
        self.assertEqual(normalized_terms, set(district_index))

    def test_every_district_definition_is_immutable(self) -> None:
        for district in self.kyiv.districts:
            with self.subTest(district=district.key):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    district.enabled = False  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the deterministic niche variants catalog."""

import dataclasses
import unittest

from niche_catalog import (
    NICHE_DEFINITIONS,
    NicheDefinition,
    get_niche_variants,
    normalize_niche_key,
)


class NicheCatalogLookupTests(unittest.TestCase):
    def test_known_russian_input_excludes_base_and_preserves_order(self) -> None:
        self.assertEqual(
            get_niche_variants("стоматология"),
            (
                "стоматологія",
                "стоматологическая клиника",
                "стоматологічна клініка",
                "стоматолог",
                "стоматологічний кабінет",
                "зубная клиника",
                "зубна клініка",
            ),
        )

    def test_known_ukrainian_input_uses_same_group(self) -> None:
        self.assertEqual(
            get_niche_variants("стоматологія"),
            (
                "стоматология",
                "стоматологическая клиника",
                "стоматологічна клініка",
                "стоматолог",
                "стоматологічний кабінет",
                "зубная клиника",
                "зубна клініка",
            ),
        )

    def test_additional_alias_returns_every_spa_query_variant(self) -> None:
        self.assertEqual(
            get_niche_variants("СПА"),
            (
                "спа салон",
                "spa salon",
                "спа центр",
                "spa center",
                "wellness центр",
                "велнес центр",
                "массажный салон",
                "масажний салон",
            ),
        )

    def test_case_and_repeated_spaces_are_normalized(self) -> None:
        result = get_niche_variants("  САЛОН   КРАСОТЫ ")
        self.assertNotIn("салон красоты", result)
        self.assertEqual(result[0], "салон краси")

    def test_unknown_niche_returns_empty_tuple(self) -> None:
        self.assertEqual(get_niche_variants("IT компания"), ())

    def test_matching_is_exact_not_substring_based(self) -> None:
        self.assertEqual(get_niche_variants("детская стоматология"), ())

    def test_apostrophes_remain_significant(self) -> None:
        self.assertNotEqual(normalize_niche_key("кав'ярня"), normalize_niche_key("кавярня"))
        self.assertEqual(get_niche_variants("кавярня"), ())

    def test_stable_order_for_selected_groups(self) -> None:
        expectations = {
            "частная клиника": (
                "приватна клініка",
                "медицинская клиника",
                "медична клініка",
                "медицинский центр",
                "медичний центр",
                "семейная клиника",
                "сімейна клініка",
            ),
            "школа английского": (
                "школа англійської",
                "курсы английского",
                "курси англійської",
                "языковая школа",
                "мовна школа",
                "английский для детей",
                "англійська для дітей",
            ),
            "загородный комплекс": (
                "заміський комплекс",
                "загородный дом",
                "заміський будинок",
                "коттеджный комплекс",
                "котеджний комплекс",
                "база отдыха",
                "база відпочинку",
            ),
        }
        for niche, expected in expectations.items():
            with self.subTest(niche=niche):
                self.assertEqual(get_niche_variants(niche), expected)


class NicheDefinitionValidationTests(unittest.TestCase):
    def test_normalization_rejects_invalid_values(self) -> None:
        for value, error in (
            ("", ValueError),
            ("   ", ValueError),
            (None, TypeError),
            (12, TypeError),
            (True, TypeError),
        ):
            with self.subTest(value=value):
                with self.assertRaises(error):
                    normalize_niche_key(value)  # type: ignore[arg-type]

    def test_definition_normalizes_spacing(self) -> None:
        definition = NicheDefinition(
            key=" example ",
            aliases=("  First   Alias ",),
            query_variants=(" First   Query ",),
        )
        self.assertEqual(definition.key, "example")
        self.assertEqual(definition.aliases, ("First Alias",))
        self.assertEqual(definition.query_variants, ("First Query",))

    def test_definition_rejects_invalid_fields(self) -> None:
        cases = (
            ({"key": "", "aliases": (), "query_variants": ("query",)}, ValueError),
            ({"key": "key", "aliases": [], "query_variants": ("query",)}, TypeError),
            ({"key": "key", "aliases": ("",), "query_variants": ("query",)}, ValueError),
            ({"key": "key", "aliases": (), "query_variants": ("",)}, ValueError),
            ({"key": "key", "aliases": (), "query_variants": ()}, ValueError),
        )
        for kwargs, error in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error):
                    NicheDefinition(**kwargs)  # type: ignore[arg-type]

    def test_definition_rejects_normalized_duplicates(self) -> None:
        for field in ("aliases", "query_variants"):
            kwargs = {
                "key": "key",
                "aliases": ("one",),
                "query_variants": ("query",),
            }
            kwargs[field] = (" Duplicate ", "DUPLICATE")
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    NicheDefinition(**kwargs)  # type: ignore[arg-type]

    def test_definition_is_frozen(self) -> None:
        definition = NicheDefinition("key", ("alias",), ("query",))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            definition.key = "changed"  # type: ignore[misc]


class NicheCatalogIntegrityTests(unittest.TestCase):
    def test_catalog_contains_exactly_the_expected_definitions(self) -> None:
        self.assertEqual(
            tuple(definition.key for definition in NICHE_DEFINITIONS),
            (
                "dentistry",
                "beauty_salon",
                "barbershop",
                "spa",
                "veterinary_clinic",
                "medical_clinic",
                "restaurant",
                "cafe",
                "pizzeria",
                "english_school",
                "real_estate_agency",
                "sauna_bath",
                "country_complex",
                "car_service",
                "clothing_store",
                "bar_pub",
            ),
        )

    def test_keys_and_alias_ownership_are_unique(self) -> None:
        self.assertEqual(len(NICHE_DEFINITIONS), 16)
        keys = [definition.key for definition in NICHE_DEFINITIONS]
        self.assertEqual(len(keys), len(set(keys)))

        owners: dict[str, str] = {}
        for definition in NICHE_DEFINITIONS:
            for alias in definition.aliases:
                alias_key = normalize_niche_key(alias)
                self.assertNotIn(alias_key, owners)
                owners[alias_key] = definition.key

    def test_each_definition_and_result_has_no_duplicates(self) -> None:
        for definition in NICHE_DEFINITIONS:
            with self.subTest(definition=definition.key):
                alias_keys = tuple(map(normalize_niche_key, definition.aliases))
                variant_keys = tuple(map(normalize_niche_key, definition.query_variants))
                self.assertEqual(len(alias_keys), len(set(alias_keys)))
                self.assertEqual(len(variant_keys), len(set(variant_keys)))

                for alias in definition.aliases:
                    result = get_niche_variants(alias)
                    result_keys = tuple(map(normalize_niche_key, result))
                    self.assertEqual(len(result_keys), len(set(result_keys)))
                    self.assertNotIn(normalize_niche_key(alias), result_keys)


if __name__ == "__main__":
    unittest.main()

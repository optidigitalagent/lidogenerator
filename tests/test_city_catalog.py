"""Tests for immutable city and administrative-district contracts."""

import dataclasses
import unittest

import city_catalog
from city_catalog import (
    CityDefinition,
    DistrictDefinition,
    build_city_index,
    build_district_index,
    enabled_districts,
    normalize_city_key,
    normalize_district_key,
    resolve_city,
    resolve_district,
)


def _district(**overrides) -> DistrictDefinition:
    values = {
        "key": "d1",
        "display_name": "D1",
        "query_text": "D1 Query",
        "aliases": ("D1 Alias",),
        "enabled": True,
    }
    values.update(overrides)
    return DistrictDefinition(**values)


def _city(**overrides) -> CityDefinition:
    values = {
        "key": "city_a",
        "canonical_name": "City A",
        "aliases": ("City A Alias",),
        "districts": (),
    }
    values.update(overrides)
    return CityDefinition(**values)


class NormalizationTests(unittest.TestCase):
    def test_city_key_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_city_key("  CITY   A "), "city a")

    def test_district_key_preserves_meaningful_characters(self) -> None:
        self.assertEqual(normalize_district_key(" D-1 "), "d-1")
        self.assertEqual(normalize_district_key(" D'1 "), "d'1")
        self.assertEqual(normalize_district_key(" D.1  2 "), "d.1 2")

    def test_normalizers_reject_invalid_types(self) -> None:
        for normalizer in (normalize_city_key, normalize_district_key):
            for value in (None, 123, True, False):
                with self.subTest(normalizer=normalizer.__name__, value=value):
                    with self.assertRaisesRegex(TypeError, "value"):
                        normalizer(value)  # type: ignore[arg-type]

    def test_normalizers_reject_empty_values(self) -> None:
        for normalizer in (normalize_city_key, normalize_district_key):
            for value in ("", " \t\n "):
                with self.subTest(normalizer=normalizer.__name__, value=value):
                    with self.assertRaisesRegex(ValueError, "value"):
                        normalizer(value)


class IdentifierValidationTests(unittest.TestCase):
    def test_valid_identifiers_are_preserved(self) -> None:
        for key in ("city_a", "district_1", "north_district", "d1"):
            with self.subTest(key=key):
                self.assertEqual(_district(key=key).key, key)
                self.assertEqual(_city(key=key).key, key)

    def test_invalid_identifiers_are_rejected(self) -> None:
        invalid = (
            "City A",
            "city-a",
            "city a",
            "місто",
            "CITY_A",
            "_city",
            "city_",
            "city__a",
        )
        for constructor in (_district, _city):
            for key in invalid:
                with self.subTest(constructor=constructor.__name__, key=key):
                    with self.assertRaisesRegex(ValueError, "key"):
                        constructor(key=key)

    def test_identifier_types_are_strict(self) -> None:
        for constructor in (_district, _city):
            for key in (None, 1, True, False):
                with self.subTest(constructor=constructor.__name__, key=key):
                    with self.assertRaisesRegex(TypeError, "key"):
                        constructor(key=key)


class DistrictDefinitionTests(unittest.TestCase):
    def test_fields_and_aliases_normalize_whitespace(self) -> None:
        district = _district(
            display_name="  D1   Name ",
            query_text=" D1   Query ",
            aliases=(" D1   Alias ", " D1-Alt "),
        )

        self.assertEqual(district.display_name, "D1 Name")
        self.assertEqual(district.query_text, "D1 Query")
        self.assertEqual(district.aliases, ("D1 Alias", "D1-Alt"))

    def test_display_name_and_query_text_may_match(self) -> None:
        district = _district(display_name="D1", query_text=" d1 ")
        self.assertEqual((district.display_name, district.query_text), ("D1", "d1"))

    def test_aliases_must_be_a_tuple(self) -> None:
        with self.assertRaisesRegex(TypeError, "aliases"):
            _district(aliases=["D1"])  # type: ignore[arg-type]

    def test_duplicate_aliases_are_rejected_after_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _district(aliases=(" D1 Alias ", "d1   alias"))

    def test_empty_and_non_string_aliases_are_rejected(self) -> None:
        for alias, error in ((" ", ValueError), (None, TypeError), (1, TypeError)):
            with self.subTest(alias=alias):
                with self.assertRaises(error):
                    _district(aliases=(alias,))  # type: ignore[arg-type]

    def test_display_and_query_fields_are_non_empty_strings(self) -> None:
        for field in ("display_name", "query_text"):
            for value, error in ((" ", ValueError), (None, TypeError), (1, TypeError)):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(error):
                        _district(**{field: value})

    def test_enabled_requires_an_actual_boolean(self) -> None:
        for value in (0, 1, None, "yes"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "enabled"):
                    _district(enabled=value)
        self.assertFalse(_district(enabled=False).enabled)

    def test_definition_is_frozen(self) -> None:
        district = _district()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            district.key = "d2"  # type: ignore[misc]


class CityDefinitionTests(unittest.TestCase):
    def test_canonical_name_and_aliases_normalize_whitespace(self) -> None:
        city = _city(
            canonical_name=" City   A ",
            aliases=(" City   A Alias ",),
        )
        self.assertEqual(city.canonical_name, "City A")
        self.assertEqual(city.aliases, ("City A Alias",))

    def test_empty_aliases_and_no_districts_are_allowed(self) -> None:
        city = _city(aliases=(), districts=())
        self.assertEqual(city.aliases, ())
        self.assertEqual(city.districts, ())

    def test_aliases_and_districts_must_be_tuples(self) -> None:
        with self.assertRaisesRegex(TypeError, "aliases"):
            _city(aliases=["City A"])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "districts"):
            _city(districts=[_district()])  # type: ignore[arg-type]

    def test_invalid_district_element_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, r"districts\[0\]"):
            _city(districts=("D1",))  # type: ignore[arg-type]

    def test_duplicate_city_aliases_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _city(aliases=("City A Alias", " city   a alias "))

    def test_duplicate_district_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate district key"):
            _city(
                districts=(
                    _district(),
                    _district(display_name="D2", query_text="D2 Query"),
                )
            )

    def test_district_term_ownership_conflicts_are_rejected(self) -> None:
        first = _district(key="d1", display_name="D1", query_text="D1 Query")
        second = _district(
            key="d2",
            display_name="D2",
            query_text="D2 Query",
            aliases=(" d1 ",),
        )
        with self.assertRaisesRegex(ValueError, "belongs to both"):
            _city(districts=(first, second))

    def test_disabled_districts_participate_in_integrity_validation(self) -> None:
        first = _district(key="d1", aliases=("D Shared",))
        second = _district(
            key="d2",
            display_name="D2",
            query_text="D2 Query",
            aliases=("d shared",),
            enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "belongs to both"):
            _city(districts=(first, second))

    def test_definition_is_frozen(self) -> None:
        city = _city()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            city.canonical_name = "City B"  # type: ignore[misc]


class CityIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.city_a = _city()
        self.city_b = _city(
            key="city_b",
            canonical_name="City B",
            aliases=("City B Alias",),
        )
        self.cities = (self.city_a, self.city_b)

    def test_indexes_canonical_names_and_aliases_in_stable_order(self) -> None:
        index = build_city_index(self.cities)
        self.assertEqual(
            tuple(index),
            ("city a", "city a alias", "city b", "city b alias"),
        )
        self.assertIs(index["city a"], self.city_a)
        self.assertIs(index["city b alias"], self.city_b)

    def test_resolves_exact_normalized_matches(self) -> None:
        self.assertIs(resolve_city("  CITY   A ", self.cities), self.city_a)
        self.assertIs(resolve_city("CITY A ALIAS", self.cities), self.city_a)
        self.assertIsNone(resolve_city("City", self.cities))
        self.assertIsNone(resolve_city("Unknown City", self.cities))

    def test_canonical_and_alias_duplicate_within_one_city_is_allowed(self) -> None:
        city = _city(aliases=(" city   a ",))
        self.assertEqual(build_city_index((city,)), {"city a": city})

    def test_duplicate_city_keys_are_rejected(self) -> None:
        duplicate = _city(canonical_name="City B", aliases=())
        with self.assertRaisesRegex(ValueError, "duplicate city key"):
            build_city_index((self.city_a, duplicate))

    def test_duplicate_city_term_ownership_is_rejected(self) -> None:
        conflicting = _city(
            key="city_b",
            canonical_name="City B",
            aliases=(" city   a ",),
        )
        with self.assertRaisesRegex(ValueError, "belongs to both"):
            build_city_index((self.city_a, conflicting))

    def test_input_container_and_elements_are_validated(self) -> None:
        with self.assertRaisesRegex(TypeError, "cities"):
            build_city_index([self.city_a])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, r"cities\[0\]"):
            build_city_index(("City A",))  # type: ignore[arg-type]

    def test_each_call_returns_a_fresh_dictionary_without_mutating_inputs(self) -> None:
        snapshot = self.cities
        first = build_city_index(self.cities)
        second = build_city_index(self.cities)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIs(self.cities, snapshot)

    def test_resolver_validates_value(self) -> None:
        for value in (None, 1, True, False):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    resolve_city(value, self.cities)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            resolve_city(" ", self.cities)


class DistrictIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.d1 = _district(
            key="d1",
            display_name="D1",
            query_text="D1 Query",
            aliases=("D1 Alias",),
        )
        self.d2 = _district(
            key="d2",
            display_name="D2",
            query_text="D2 Query",
            aliases=("D2 Alias",),
            enabled=False,
        )
        self.d3 = _district(
            key="d3",
            display_name="D3",
            query_text="D3 Query",
            aliases=("D3 Alias",),
        )
        self.city_a = _city(districts=(self.d1, self.d2, self.d3))

    def test_indexes_all_terms_in_stable_order(self) -> None:
        index = build_district_index(self.city_a)
        self.assertEqual(
            tuple(index),
            (
                "d1", "d1 query", "d1 alias",
                "d2", "d2 query", "d2 alias",
                "d3", "d3 query", "d3 alias",
            ),
        )

    def test_resolves_display_query_and_alias_terms(self) -> None:
        self.assertIs(resolve_district(self.city_a, " d1 "), self.d1)
        self.assertIs(resolve_district(self.city_a, "D1   QUERY"), self.d1)
        self.assertIs(resolve_district(self.city_a, "d1 alias"), self.d1)

    def test_disabled_district_is_resolvable(self) -> None:
        self.assertIs(resolve_district(self.city_a, "D2 Alias"), self.d2)

    def test_matching_is_exact_and_unknown_returns_none(self) -> None:
        self.assertIsNone(resolve_district(self.city_a, "D"))
        self.assertIsNone(resolve_district(self.city_a, "Unknown District"))

    def test_same_term_within_one_district_is_allowed(self) -> None:
        district = _district(
            display_name="D1",
            query_text=" d1 ",
            aliases=(" D1 ",),
        )
        city = _city(districts=(district,))
        self.assertEqual(build_district_index(city), {"d1": district})

    def test_each_call_returns_a_fresh_dictionary(self) -> None:
        first = build_district_index(self.city_a)
        second = build_district_index(self.city_a)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)

    def test_city_and_value_inputs_are_validated(self) -> None:
        with self.assertRaisesRegex(TypeError, "city"):
            build_district_index("City A")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "city"):
            resolve_district("City A", "D1")  # type: ignore[arg-type]
        for value in (None, 1, True, False):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    resolve_district(self.city_a, value)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            resolve_district(self.city_a, " ")

    def test_same_district_key_and_alias_are_valid_across_cities(self) -> None:
        city_b_d1 = _district(key="d1", aliases=("D1 Alias",))
        city_b = _city(
            key="city_b",
            canonical_name="City B",
            aliases=(),
            districts=(city_b_d1,),
        )
        self.assertIs(resolve_district(self.city_a, "D1 Alias"), self.d1)
        self.assertIs(resolve_district(city_b, "D1 Alias"), city_b_d1)


class EnabledDistrictTests(unittest.TestCase):
    def test_returns_only_enabled_districts_in_original_order(self) -> None:
        d1 = _district(key="d1", display_name="D1", query_text="D1 Query")
        d2 = _district(
            key="d2",
            display_name="D2",
            query_text="D2 Query",
            aliases=("D2 Alias",),
            enabled=False,
        )
        d3 = _district(
            key="d3",
            display_name="D3",
            query_text="D3 Query",
            aliases=("D3 Alias",),
        )
        city = _city(districts=(d1, d2, d3))

        self.assertEqual(enabled_districts(city), (d1, d3))
        self.assertEqual(city.districts, (d1, d2, d3))

    def test_city_without_districts_returns_empty_tuple(self) -> None:
        self.assertEqual(enabled_districts(_city()), ())

    def test_city_type_is_validated(self) -> None:
        with self.assertRaisesRegex(TypeError, "city"):
            enabled_districts("City A")  # type: ignore[arg-type]


class NoProductionDataTests(unittest.TestCase):
    def test_production_registry_exists_as_an_empty_tuple(self) -> None:
        self.assertTrue(hasattr(city_catalog, "CITY_DEFINITIONS"))
        self.assertIs(type(city_catalog.CITY_DEFINITIONS), tuple)
        self.assertEqual(city_catalog.CITY_DEFINITIONS, ())

    def test_empty_registry_builds_an_empty_index(self) -> None:
        self.assertEqual(
            build_city_index(city_catalog.CITY_DEFINITIONS),
            {},
        )

    def test_empty_registry_does_not_resolve_a_synthetic_city(self) -> None:
        self.assertIsNone(
            resolve_city("City A", city_catalog.CITY_DEFINITIONS),
        )

    def test_production_registry_does_not_support_append(self) -> None:
        with self.assertRaises(AttributeError):
            city_catalog.CITY_DEFINITIONS.append(_city())  # type: ignore[attr-defined]

    def test_production_registry_contains_no_synthetic_cities(self) -> None:
        canonical_names = {
            city.canonical_name for city in city_catalog.CITY_DEFINITIONS
        }
        self.assertNotIn("City A", canonical_names)
        self.assertNotIn("City B", canonical_names)

    def test_synthetic_city_fixtures_remain_local_to_tests(self) -> None:
        local_cities = (
            _city(),
            _city(
                key="city_b",
                canonical_name="City B",
                aliases=(),
            ),
        )

        self.assertEqual(
            tuple(city.canonical_name for city in local_cities),
            ("City A", "City B"),
        )
        self.assertEqual(city_catalog.CITY_DEFINITIONS, ())
        self.assertFalse(hasattr(city_catalog, "CITY_CATALOG"))


if __name__ == "__main__":
    unittest.main()

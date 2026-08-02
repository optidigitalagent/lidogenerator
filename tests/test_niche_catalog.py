"""Unit tests for the deterministic phased niche catalog."""

import dataclasses
import unittest

from niche_catalog import (
    NICHE_DEFINITIONS,
    NicheDefinition,
    NicheSearchPlan,
    get_niche_variants,
    normalize_niche_key,
    resolve_niche_plan,
)


def _definition(**overrides) -> NicheDefinition:
    values = {
        "key": "example",
        "aliases": ("primary", "fallback", "alias only"),
        "primary_query_variants": ("primary",),
        "fallback_query_variants": ("fallback",),
    }
    values.update(overrides)
    return NicheDefinition(**values)


class NicheDefinitionValidationTests(unittest.TestCase):
    def test_definition_normalizes_spacing_and_key(self) -> None:
        definition = NicheDefinition(
            key="  EXAMPLE  KEY ",
            aliases=(" First   Query ", " Fallback   Query "),
            primary_query_variants=(" First   Query ",),
            fallback_query_variants=(" Fallback   Query ",),
        )

        self.assertEqual(definition.key, "example key")
        self.assertEqual(definition.aliases, ("First Query", "Fallback Query"))
        self.assertEqual(definition.primary_query_variants, ("First Query",))
        self.assertEqual(definition.fallback_query_variants, ("Fallback Query",))

    def test_primary_is_required_and_fallback_may_be_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_query_variants"):
            _definition(
                aliases=(),
                primary_query_variants=(),
                fallback_query_variants=(),
            )

        definition = _definition(
            aliases=("primary",),
            fallback_query_variants=(),
        )
        self.assertEqual(definition.fallback_query_variants, ())

    def test_collections_must_be_tuples(self) -> None:
        for field in (
            "aliases",
            "primary_query_variants",
            "fallback_query_variants",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(TypeError, field):
                    _definition(**{field: ["value"]})

    def test_empty_and_non_string_values_are_rejected(self) -> None:
        for field in (
            "aliases",
            "primary_query_variants",
            "fallback_query_variants",
        ):
            for value, error in ((" ", ValueError), (None, TypeError)):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(error):
                        _definition(**{field: (value,)})

    def test_normalized_duplicates_are_rejected_in_every_collection(self) -> None:
        for field in (
            "aliases",
            "primary_query_variants",
            "fallback_query_variants",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    _definition(**{field: (" Duplicate ", "DUPLICATE")})

    def test_primary_and_fallback_must_not_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            _definition(
                aliases=("same",),
                primary_query_variants=("same",),
                fallback_query_variants=(" SAME ",),
            )

    def test_all_query_variants_must_be_aliases(self) -> None:
        for aliases in (("primary",), ("fallback",)):
            with self.subTest(aliases=aliases):
                with self.assertRaisesRegex(ValueError, "present in aliases"):
                    _definition(aliases=aliases)

    def test_definition_is_frozen(self) -> None:
        definition = _definition()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            definition.key = "changed"  # type: ignore[misc]


class NicheSearchPlanValidationTests(unittest.TestCase):
    def test_plan_normalizes_fields_and_reports_known_state(self) -> None:
        plan = NicheSearchPlan(
            key=" Dentistry ",
            input_niche="  стоматолог ",
            base_niche=" стоматология ",
            primary_variants=(" стоматологія ",),
            fallback_variants=(" зубная   клиника ",),
        )

        self.assertEqual(plan.key, "dentistry")
        self.assertEqual(plan.input_niche, "стоматолог")
        self.assertEqual(plan.base_niche, "стоматология")
        self.assertEqual(plan.primary_variants, ("стоматологія",))
        self.assertEqual(plan.fallback_variants, ("зубная клиника",))
        self.assertTrue(plan.known)
        self.assertFalse(
            NicheSearchPlan(None, "custom", "custom", (), ()).known
        )

    def test_plan_rejects_invalid_key_and_string_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "key"):
            NicheSearchPlan(1, "input", "base", (), ())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "key"):
            NicheSearchPlan(" ", "input", "base", (), ())
        for field in ("input_niche", "base_niche"):
            values = {
                "key": None,
                "input_niche": "input",
                "base_niche": "base",
                "primary_variants": (),
                "fallback_variants": (),
            }
            values[field] = " "
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    NicheSearchPlan(**values)

    def test_plan_rejects_non_tuples_duplicates_and_overlap(self) -> None:
        with self.assertRaisesRegex(TypeError, "primary_variants"):
            NicheSearchPlan(None, "input", "base", [], ())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            NicheSearchPlan(None, "input", "base", ("one", "ONE"), ())
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            NicheSearchPlan(None, "input", "base", ("one",), ("ONE",))

    def test_plan_rejects_base_in_either_variant_phase(self) -> None:
        for primary, fallback in ((("BASE",), ()), ((), (" base ",))):
            with self.subTest(primary=primary, fallback=fallback):
                with self.assertRaisesRegex(ValueError, "base_niche"):
                    NicheSearchPlan(None, "input", "base", primary, fallback)

    def test_plan_is_frozen(self) -> None:
        plan = NicheSearchPlan(None, "custom", "custom", (), ())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.base_niche = "changed"  # type: ignore[misc]


class NicheCatalogIntegrityTests(unittest.TestCase):
    EXPECTED_KEYS = (
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
    )

    def test_catalog_contains_exactly_sixteen_expected_definitions(self) -> None:
        self.assertEqual(len(NICHE_DEFINITIONS), 16)
        self.assertEqual(
            tuple(definition.key for definition in NICHE_DEFINITIONS),
            self.EXPECTED_KEYS,
        )

    def test_keys_and_alias_ownership_are_unique(self) -> None:
        keys = [definition.key for definition in NICHE_DEFINITIONS]
        self.assertEqual(len(keys), len(set(keys)))

        owners: dict[str, str] = {}
        for definition in NICHE_DEFINITIONS:
            for alias in definition.aliases:
                alias_key = normalize_niche_key(alias)
                self.assertNotIn(alias_key, owners)
                owners[alias_key] = definition.key

    def test_query_variants_are_aliases_and_phases_do_not_overlap(self) -> None:
        for definition in NICHE_DEFINITIONS:
            with self.subTest(definition=definition.key):
                alias_keys = set(map(normalize_niche_key, definition.aliases))
                primary_keys = set(
                    map(normalize_niche_key, definition.primary_query_variants)
                )
                fallback_keys = set(
                    map(normalize_niche_key, definition.fallback_query_variants)
                )
                self.assertTrue(primary_keys <= alias_keys)
                self.assertTrue(fallback_keys <= alias_keys)
                self.assertFalse(primary_keys & fallback_keys)

    def test_aliases_have_stable_primary_fallback_alias_only_order(self) -> None:
        expected_alias_only = {
            "dentistry": (
                "стоматолог",
                "стоматология клиника",
                "стоматологія клініка",
            ),
            "spa": ("спа", "spa"),
            "english_school": (
                "английский для детей",
                "англійська для дітей",
                "английская школа",
                "англійська школа",
            ),
        }
        for definition in NICHE_DEFINITIONS:
            query_count = len(definition.primary_query_variants) + len(
                definition.fallback_query_variants
            )
            self.assertEqual(
                definition.aliases[:query_count],
                (
                    *definition.primary_query_variants,
                    *definition.fallback_query_variants,
                ),
            )
            if definition.key in expected_alias_only:
                self.assertEqual(
                    definition.aliases[query_count:],
                    expected_alias_only[definition.key],
                )

    def test_removed_values_are_not_owned_by_previous_groups(self) -> None:
        for niche in (
            "массажный салон",
            "масажний салон",
            "кондитерская",
            "кондитерська",
            "шиномонтаж",
        ):
            with self.subTest(niche=niche):
                plan = resolve_niche_plan(niche)
                self.assertFalse(plan.known)
                self.assertEqual(plan.base_niche, niche)
                self.assertEqual(get_niche_variants(niche), ())


class NicheResolutionTests(unittest.TestCase):
    def test_unknown_niche_uses_only_normalized_raw_base(self) -> None:
        self.assertEqual(
            resolve_niche_plan("  IT   компания "),
            NicheSearchPlan(
                key=None,
                input_niche="IT компания",
                base_niche="IT компания",
                primary_variants=(),
                fallback_variants=(),
            ),
        )

    def test_primary_inputs_remain_base_and_reorder_remaining_primary(self) -> None:
        russian = resolve_niche_plan("стоматология")
        ukrainian = resolve_niche_plan("стоматологія")

        self.assertEqual(russian.key, "dentistry")
        self.assertEqual(russian.base_niche, "стоматология")
        self.assertEqual(russian.primary_variants[0], "стоматологія")
        self.assertEqual(ukrainian.base_niche, "стоматологія")
        self.assertEqual(
            ukrainian.primary_variants,
            (
                "стоматология",
                "стоматологическая клиника",
                "стоматологічна клініка",
            ),
        )

    def test_alias_only_input_uses_canonical_primary_base(self) -> None:
        plan = resolve_niche_plan("стоматолог")

        self.assertEqual(plan.key, "dentistry")
        self.assertEqual(plan.base_niche, "стоматология")
        self.assertNotIn("стоматолог", plan.primary_variants)
        self.assertNotIn("стоматолог", plan.fallback_variants)

    def test_additional_alias_uses_canonical_spa_plan(self) -> None:
        plan = resolve_niche_plan("СПА")

        self.assertEqual(plan.key, "spa")
        self.assertEqual(plan.base_niche, "спа салон")
        self.assertEqual(plan.primary_variants, ("спа центр",))
        self.assertEqual(
            plan.fallback_variants,
            ("spa salon", "spa center", "wellness центр", "велнес центр"),
        )

    def test_fallback_input_keeps_fallback_after_all_primary(self) -> None:
        plan = resolve_niche_plan("сауна")

        self.assertEqual(plan.key, "sauna_bath")
        self.assertEqual(plan.base_niche, "баня")
        self.assertEqual(
            plan.primary_variants,
            (
                "банный комплекс",
                "лазневий комплекс",
                "саунный комплекс",
                "комплекс саун",
            ),
        )
        self.assertEqual(plan.fallback_variants, ("сауна", "лазня"))

    def test_selected_alias_only_values_never_become_automatic_variants(self) -> None:
        alias_only_values = (
            "стоматолог",
            "барбер",
            "ветеринар",
            "доставка пиццы",
            "доставка піци",
            "английский для детей",
            "англійська для дітей",
            "недвижимость",
            "нерухомість",
            "загородный дом",
            "заміський будинок",
        )
        for niche in alias_only_values:
            with self.subTest(niche=niche):
                plan = resolve_niche_plan(niche)
                automatic = (*plan.primary_variants, *plan.fallback_variants)
                self.assertTrue(plan.known)
                self.assertNotIn(normalize_niche_key(niche), map(normalize_niche_key, automatic))

    def test_resolution_examples_cover_remaining_alias_groups(self) -> None:
        expectations = {
            "недвижимость": ("real_estate_agency", "агентство недвижимости"),
            "загородный дом": ("country_complex", "загородный комплекс"),
        }
        for niche, (key, base) in expectations.items():
            with self.subTest(niche=niche):
                plan = resolve_niche_plan(niche)
                self.assertEqual((plan.key, plan.base_niche), (key, base))

    def test_stable_plans_for_selected_groups(self) -> None:
        expectations = {
            "стоматология": (
                ("стоматологія", "стоматологическая клиника", "стоматологічна клініка"),
                ("стоматологічний кабінет", "зубная клиника", "зубна клініка"),
            ),
            "спа салон": (
                ("спа центр",),
                ("spa salon", "spa center", "wellness центр", "велнес центр"),
            ),
            "частная клиника": (
                ("приватна клініка", "медицинская клиника", "медична клініка"),
                ("медицинский центр", "медичний центр", "семейная клиника", "сімейна клініка"),
            ),
            "школа английского": (
                ("школа англійської",),
                ("курсы английского", "курси англійської", "языковая школа", "мовна школа"),
            ),
            "загородный комплекс": (
                ("заміський комплекс",),
                ("коттеджный комплекс", "котеджний комплекс", "база отдыха", "база відпочинку"),
            ),
            "баня": (
                ("банный комплекс", "лазневий комплекс", "саунный комплекс", "комплекс саун"),
                ("сауна", "лазня"),
            ),
        }
        for niche, (primary, fallback) in expectations.items():
            with self.subTest(niche=niche):
                plan = resolve_niche_plan(niche)
                self.assertEqual(plan.primary_variants, primary)
                self.assertEqual(plan.fallback_variants, fallback)

    def test_compatibility_helper_returns_primary_then_fallback_without_base(self) -> None:
        plan = resolve_niche_plan("стоматология")
        self.assertEqual(
            get_niche_variants("стоматология"),
            (*plan.primary_variants, *plan.fallback_variants),
        )
        self.assertNotIn(plan.base_niche, get_niche_variants("стоматология"))
        self.assertEqual(get_niche_variants("IT компания"), ())

    def test_matching_is_exact_and_apostrophes_are_significant(self) -> None:
        self.assertFalse(resolve_niche_plan("детская стоматология").known)
        self.assertNotEqual(
            normalize_niche_key("кав'ярня"),
            normalize_niche_key("кавярня"),
        )
        self.assertFalse(resolve_niche_plan("кавярня").known)


if __name__ == "__main__":
    unittest.main()

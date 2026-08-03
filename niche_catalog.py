"""Deterministic catalog of curated niche search plans."""

from __future__ import annotations

from dataclasses import dataclass


def _normalize_spacing(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def normalize_niche_key(value: str) -> str:
    """Return the exact-match key used for niche aliases."""

    return _normalize_spacing(value, "value").casefold()


def _normalize_tuple(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple, not {type(values).__name__}")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item = _normalize_spacing(value, f"{name}[{index}]")
        key = normalize_niche_key(item)
        if key in seen:
            raise ValueError(f"{name} must not contain duplicate values")
        seen.add(key)
        normalized.append(item)
    return tuple(normalized)


def _normalized_keys(values: tuple[str, ...]) -> set[str]:
    return {normalize_niche_key(value) for value in values}


@dataclass(frozen=True)
class NicheDefinition:
    """One validated, immutable group of aliases and phased query variants."""

    key: str
    aliases: tuple[str, ...]
    primary_query_variants: tuple[str, ...]
    fallback_query_variants: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "key",
            normalize_niche_key(_normalize_spacing(self.key, "key")),
        )
        object.__setattr__(self, "aliases", _normalize_tuple(self.aliases, "aliases"))
        object.__setattr__(
            self,
            "primary_query_variants",
            _normalize_tuple(
                self.primary_query_variants,
                "primary_query_variants",
            ),
        )
        object.__setattr__(
            self,
            "fallback_query_variants",
            _normalize_tuple(
                self.fallback_query_variants,
                "fallback_query_variants",
            ),
        )

        if not self.primary_query_variants:
            raise ValueError("primary_query_variants must not be empty")

        primary_keys = _normalized_keys(self.primary_query_variants)
        fallback_keys = _normalized_keys(self.fallback_query_variants)
        if primary_keys & fallback_keys:
            raise ValueError(
                "primary_query_variants and fallback_query_variants must not overlap"
            )

        alias_keys = _normalized_keys(self.aliases)
        missing_keys = (primary_keys | fallback_keys) - alias_keys
        if missing_keys:
            raise ValueError(
                "all primary and fallback query variants must be present in aliases"
            )


def _definition(
    key: str,
    primary_query_variants: tuple[str, ...],
    fallback_query_variants: tuple[str, ...] = (),
    alias_only: tuple[str, ...] = (),
) -> NicheDefinition:
    return NicheDefinition(
        key=key,
        aliases=(
            *primary_query_variants,
            *fallback_query_variants,
            *alias_only,
        ),
        primary_query_variants=primary_query_variants,
        fallback_query_variants=fallback_query_variants,
    )


@dataclass(frozen=True)
class NicheSearchPlan:
    """Resolved canonical base plus ordered primary and fallback phases."""

    key: str | None
    input_niche: str
    base_niche: str
    primary_variants: tuple[str, ...]
    fallback_variants: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.key is not None:
            if not isinstance(self.key, str):
                raise TypeError("key must be a string or None")
            object.__setattr__(
                self,
                "key",
                normalize_niche_key(_normalize_spacing(self.key, "key")),
            )

        object.__setattr__(
            self,
            "input_niche",
            _normalize_spacing(self.input_niche, "input_niche"),
        )
        object.__setattr__(
            self,
            "base_niche",
            _normalize_spacing(self.base_niche, "base_niche"),
        )
        object.__setattr__(
            self,
            "primary_variants",
            _normalize_tuple(self.primary_variants, "primary_variants"),
        )
        object.__setattr__(
            self,
            "fallback_variants",
            _normalize_tuple(self.fallback_variants, "fallback_variants"),
        )

        primary_keys = _normalized_keys(self.primary_variants)
        fallback_keys = _normalized_keys(self.fallback_variants)
        if primary_keys & fallback_keys:
            raise ValueError("primary_variants and fallback_variants must not overlap")

        base_key = normalize_niche_key(self.base_niche)
        if base_key in primary_keys:
            raise ValueError("base_niche must not be repeated in primary_variants")
        if base_key in fallback_keys:
            raise ValueError("base_niche must not be repeated in fallback_variants")

    @property
    def known(self) -> bool:
        return self.key is not None


NICHE_DEFINITIONS: tuple[NicheDefinition, ...] = (
    _definition(
        "dentistry",
        (
            "стоматология",
            "стоматологія",
            "стоматологическая клиника",
            "стоматологічна клініка",
        ),
        (
            "стоматологічний кабінет",
            "зубная клиника",
            "зубна клініка",
        ),
        (
            "стоматолог",
            "стоматология клиника",
            "стоматологія клініка",
        ),
    ),
    _definition(
        "beauty_salon",
        (
            "салон красоты",
            "салон краси",
            "студия красоты",
            "студія краси",
        ),
        (
            "beauty salon",
            "бьюти студия",
            "б'юті студія",
        ),
        (
            "бьюти салон",
            "б'юті салон",
        ),
    ),
    _definition(
        "barbershop",
        (
            "барбершоп",
            "чоловіча перукарня",
            "мужская парикмахерская",
        ),
        ("barber shop",),
        ("барбер",),
    ),
    _definition(
        "spa",
        (
            "спа салон",
            "спа центр",
        ),
        (
            "spa salon",
            "spa center",
            "wellness центр",
            "велнес центр",
        ),
        (
            "спа",
            "spa",
        ),
    ),
    _definition(
        "veterinary_clinic",
        (
            "ветеринарная клиника",
            "ветеринарна клініка",
            "ветклиника",
            "ветклініка",
        ),
        (
            "ветеринарный центр",
            "ветеринарний центр",
        ),
        ("ветеринар",),
    ),
    _definition(
        "medical_clinic",
        (
            "частная клиника",
            "приватна клініка",
            "медицинская клиника",
            "медична клініка",
        ),
        (
            "медицинский центр",
            "медичний центр",
            "семейная клиника",
            "сімейна клініка",
        ),
        (
            "клиника",
            "клініка",
            "частный медицинский центр",
            "приватний медичний центр",
        ),
    ),
    _definition(
        "restaurant",
        (
            "ресторан",
            "семейный ресторан",
            "сімейний ресторан",
        ),
        (
            "ресторан бар",
            "гастробар",
            "gastrobar",
        ),
    ),
    _definition(
        "cafe",
        (
            "кафе",
            "кофейня",
            "кав'ярня",
        ),
        ("coffee shop",),
    ),
    _definition(
        "pizzeria",
        (
            "пиццерия",
            "піцерія",
        ),
        ("pizza restaurant",),
        (
            "доставка пиццы",
            "доставка піци",
        ),
    ),
    _definition(
        "english_school",
        (
            "школа английского",
            "школа англійської",
        ),
        (
            "курсы английского",
            "курси англійської",
            "языковая школа",
            "мовна школа",
        ),
        (
            "английский для детей",
            "англійська для дітей",
            "английская школа",
            "англійська школа",
        ),
    ),
    _definition(
        "real_estate_agency",
        (
            "агентство недвижимости",
            "агенція нерухомості",
            "агентство нерухомості",
        ),
        (
            "риэлторское агентство",
            "рієлторська агенція",
        ),
        (
            "недвижимость",
            "нерухомість",
        ),
    ),
    _definition(
        "sauna_bath",
        (
            "баня",
            "банный комплекс",
            "лазневий комплекс",
            "саунный комплекс",
            "комплекс саун",
        ),
        (
            "сауна",
            "лазня",
        ),
    ),
    _definition(
        "country_complex",
        (
            "загородный комплекс",
            "заміський комплекс",
        ),
        (
            "коттеджный комплекс",
            "котеджний комплекс",
            "база отдыха",
            "база відпочинку",
        ),
        (
            "загородный дом",
            "заміський будинок",
        ),
    ),
    _definition(
        "car_service",
        (
            "СТО",
            "автосервис",
            "автосервіс",
        ),
        (
            "автомастерская",
            "автомайстерня",
        ),
    ),
    _definition(
        "clothing_store",
        (
            "магазин одежды",
            "магазин одягу",
            "бутик одежды",
            "бутик одягу",
        ),
        (
            "шоурум одежды",
            "шоурум одягу",
            "fashion store",
        ),
    ),
    _definition(
        "bar_pub",
        (
            "бар",
            "паб",
            "коктейльный бар",
            "коктейльний бар",
        ),
        (
            "pub",
            "lounge bar",
        ),
    ),
)


def _build_alias_index(
    definitions: tuple[NicheDefinition, ...],
) -> dict[str, NicheDefinition]:
    index: dict[str, NicheDefinition] = {}
    seen_definition_keys: set[str] = set()
    for definition in definitions:
        if definition.key in seen_definition_keys:
            raise ValueError(f"duplicate niche definition key {definition.key!r}")
        seen_definition_keys.add(definition.key)

        for alias in definition.aliases:
            alias_key = normalize_niche_key(alias)
            existing = index.get(alias_key)
            if existing is not None and existing.key != definition.key:
                raise ValueError(
                    f"niche alias {alias!r} belongs to both "
                    f"{existing.key!r} and {definition.key!r}"
                )
            index[alias_key] = definition
    return index


_ALIAS_INDEX = _build_alias_index(NICHE_DEFINITIONS)


def resolve_niche_plan(niche: str) -> NicheSearchPlan:
    """Resolve user input to a canonical base and phased automatic variants."""

    input_niche = _normalize_spacing(niche, "niche")
    niche_key = normalize_niche_key(input_niche)
    definition = _ALIAS_INDEX.get(niche_key)
    if definition is None:
        return NicheSearchPlan(
            key=None,
            input_niche=input_niche,
            base_niche=input_niche,
            primary_variants=(),
            fallback_variants=(),
        )

    primary_keys = _normalized_keys(definition.primary_query_variants)
    if niche_key in primary_keys:
        base_niche = input_niche
        primary_variants = tuple(
            variant
            for variant in definition.primary_query_variants
            if normalize_niche_key(variant) != niche_key
        )
    else:
        base_niche = definition.primary_query_variants[0]
        primary_variants = definition.primary_query_variants[1:]

    return NicheSearchPlan(
        key=definition.key,
        input_niche=input_niche,
        base_niche=base_niche,
        primary_variants=primary_variants,
        fallback_variants=definition.fallback_query_variants,
    )


def get_niche_variants(niche: str) -> tuple[str, ...]:
    """Return phased automatic variants, excluding the resolved base query."""

    plan = resolve_niche_plan(niche)
    return (*plan.primary_variants, *plan.fallback_variants)

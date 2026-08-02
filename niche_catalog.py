"""Deterministic catalog of curated niche search variants."""

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


@dataclass(frozen=True)
class NicheDefinition:
    """One validated, immutable group of aliases and search variants."""

    key: str
    aliases: tuple[str, ...]
    query_variants: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _normalize_spacing(self.key, "key"))
        object.__setattr__(self, "aliases", _normalize_tuple(self.aliases, "aliases"))
        object.__setattr__(
            self,
            "query_variants",
            _normalize_tuple(self.query_variants, "query_variants"),
        )
        if not self.query_variants:
            raise ValueError("query_variants must not be empty")


def _definition(
    key: str,
    query_variants: tuple[str, ...],
    additional_aliases: tuple[str, ...] = (),
) -> NicheDefinition:
    return NicheDefinition(
        key=key,
        aliases=(*query_variants, *additional_aliases),
        query_variants=query_variants,
    )


NICHE_DEFINITIONS: tuple[NicheDefinition, ...] = (
    _definition(
        "dentistry",
        (
            "стоматология",
            "стоматологія",
            "стоматологическая клиника",
            "стоматологічна клініка",
            "стоматолог",
            "стоматологічний кабінет",
            "зубная клиника",
            "зубна клініка",
        ),
        (
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
            "barber shop",
            "барбер",
            "чоловіча перукарня",
            "мужская парикмахерская",
        ),
    ),
    _definition(
        "spa",
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
            "ветеринарный центр",
            "ветеринарний центр",
            "ветеринар",
        ),
    ),
    _definition(
        "medical_clinic",
        (
            "частная клиника",
            "приватна клініка",
            "медицинская клиника",
            "медична клініка",
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
            "coffee shop",
            "кондитерская",
            "кондитерська",
        ),
    ),
    _definition(
        "pizzeria",
        (
            "пиццерия",
            "піцерія",
            "доставка пиццы",
            "доставка піци",
            "pizza restaurant",
        ),
    ),
    _definition(
        "english_school",
        (
            "школа английского",
            "школа англійської",
            "курсы английского",
            "курси англійської",
            "языковая школа",
            "мовна школа",
            "английский для детей",
            "англійська для дітей",
        ),
        (
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
            "риэлторское агентство",
            "рієлторська агенція",
            "недвижимость",
            "нерухомість",
        ),
    ),
    _definition(
        "sauna_bath",
        (
            "баня",
            "сауна",
            "лазня",
            "банный комплекс",
            "лазневий комплекс",
            "саунный комплекс",
            "комплекс саун",
        ),
    ),
    _definition(
        "country_complex",
        (
            "загородный комплекс",
            "заміський комплекс",
            "загородный дом",
            "заміський будинок",
            "коттеджный комплекс",
            "котеджний комплекс",
            "база отдыха",
            "база відпочинку",
        ),
    ),
    _definition(
        "car_service",
        (
            "СТО",
            "автосервис",
            "автосервіс",
            "автомастерская",
            "автомайстерня",
            "шиномонтаж",
        ),
    ),
    _definition(
        "clothing_store",
        (
            "магазин одежды",
            "магазин одягу",
            "бутик одежды",
            "бутик одягу",
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
            "pub",
            "коктейльный бар",
            "коктейльний бар",
            "lounge bar",
        ),
    ),
)


def _build_alias_index(
    definitions: tuple[NicheDefinition, ...],
) -> dict[str, NicheDefinition]:
    index: dict[str, NicheDefinition] = {}
    for definition in definitions:
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


def get_niche_variants(niche: str) -> tuple[str, ...]:
    """Return ordered variants for an exact known alias, excluding the input."""

    niche_key = normalize_niche_key(niche)
    definition = _ALIAS_INDEX.get(niche_key)
    if definition is None:
        return ()

    variants: list[str] = []
    seen: set[str] = {niche_key}
    for variant in definition.query_variants:
        variant_key = normalize_niche_key(variant)
        if variant_key not in seen:
            seen.add(variant_key)
            variants.append(variant)
    return tuple(variants)

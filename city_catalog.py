"""Immutable contracts for cities and administrative districts."""

from __future__ import annotations

from dataclasses import dataclass
import re


_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")


def _normalize_spacing(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")

    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def normalize_city_key(value: str) -> str:
    """Return the deterministic exact-match key for a city term."""

    return _normalize_spacing(value, "value").casefold()


def normalize_district_key(value: str) -> str:
    """Return the deterministic exact-match key for a district term."""

    return _normalize_spacing(value, "value").casefold()


def _validate_identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase ASCII snake_case identifier")
    return value


def _normalize_aliases(
    values: object,
    name: str,
    normalizer,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple, not {type(values).__name__}")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        alias = _normalize_spacing(value, f"{name}[{index}]")
        alias_key = normalizer(alias)
        if alias_key in seen:
            raise ValueError(f"{name} must not contain duplicate values")
        seen.add(alias_key)
        normalized.append(alias)
    return tuple(normalized)


@dataclass(frozen=True)
class DistrictDefinition:
    """One validated administrative district scoped to a parent city."""

    key: str
    display_name: str
    query_text: str
    aliases: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _validate_identifier(self.key, "key"))
        object.__setattr__(
            self,
            "display_name",
            _normalize_spacing(self.display_name, "display_name"),
        )
        object.__setattr__(
            self,
            "query_text",
            _normalize_spacing(self.query_text, "query_text"),
        )
        object.__setattr__(
            self,
            "aliases",
            _normalize_aliases(
                self.aliases,
                "aliases",
                normalize_district_key,
            ),
        )
        if type(self.enabled) is not bool:
            raise TypeError(
                "enabled must be a boolean, "
                f"not {type(self.enabled).__name__}"
            )


def _district_terms(
    district: DistrictDefinition,
) -> tuple[str, ...]:
    return (district.display_name, district.query_text, *district.aliases)


def _validate_district_integrity(
    districts: tuple[DistrictDefinition, ...],
) -> None:
    seen_keys: set[str] = set()
    term_owners: dict[str, DistrictDefinition] = {}

    for district in districts:
        if district.key in seen_keys:
            raise ValueError(f"duplicate district key {district.key!r}")
        seen_keys.add(district.key)

        for term in _district_terms(district):
            term_key = normalize_district_key(term)
            existing = term_owners.get(term_key)
            if existing is not None and existing.key != district.key:
                raise ValueError(
                    f"district term {term!r} belongs to both "
                    f"{existing.key!r} and {district.key!r}"
                )
            term_owners[term_key] = district


@dataclass(frozen=True)
class CityDefinition:
    """One validated city and its ordered administrative districts."""

    key: str
    canonical_name: str
    aliases: tuple[str, ...]
    districts: tuple[DistrictDefinition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _validate_identifier(self.key, "key"))
        object.__setattr__(
            self,
            "canonical_name",
            _normalize_spacing(self.canonical_name, "canonical_name"),
        )
        object.__setattr__(
            self,
            "aliases",
            _normalize_aliases(self.aliases, "aliases", normalize_city_key),
        )

        if not isinstance(self.districts, tuple):
            raise TypeError(
                "districts must be a tuple, "
                f"not {type(self.districts).__name__}"
            )
        for index, district in enumerate(self.districts):
            if not isinstance(district, DistrictDefinition):
                raise TypeError(
                    f"districts[{index}] must be a DistrictDefinition, "
                    f"not {type(district).__name__}"
                )
        _validate_district_integrity(self.districts)


CITY_DEFINITIONS: tuple[CityDefinition, ...] = ()


def build_city_index(
    cities: tuple[CityDefinition, ...],
) -> dict[str, CityDefinition]:
    """Build a fresh exact-match index for validated city definitions."""

    if not isinstance(cities, tuple):
        raise TypeError(f"cities must be a tuple, not {type(cities).__name__}")

    index: dict[str, CityDefinition] = {}
    seen_keys: set[str] = set()
    for position, city in enumerate(cities):
        if not isinstance(city, CityDefinition):
            raise TypeError(
                f"cities[{position}] must be a CityDefinition, "
                f"not {type(city).__name__}"
            )
        if city.key in seen_keys:
            raise ValueError(f"duplicate city key {city.key!r}")
        seen_keys.add(city.key)

        for term in (city.canonical_name, *city.aliases):
            term_key = normalize_city_key(term)
            existing = index.get(term_key)
            if existing is not None and existing.key != city.key:
                raise ValueError(
                    f"city term {term!r} belongs to both "
                    f"{existing.key!r} and {city.key!r}"
                )
            index[term_key] = city
    return index


def resolve_city(
    value: str,
    cities: tuple[CityDefinition, ...],
) -> CityDefinition | None:
    """Resolve a city by an exact normalized canonical name or alias."""

    value_key = normalize_city_key(value)
    return build_city_index(cities).get(value_key)


def build_district_index(
    city: CityDefinition,
) -> dict[str, DistrictDefinition]:
    """Build a fresh exact-match district index scoped to one city."""

    if not isinstance(city, CityDefinition):
        raise TypeError(
            "city must be a CityDefinition, "
            f"not {type(city).__name__}"
        )

    index: dict[str, DistrictDefinition] = {}
    for district in city.districts:
        for term in _district_terms(district):
            term_key = normalize_district_key(term)
            existing = index.get(term_key)
            if existing is not None and existing.key != district.key:
                raise ValueError(
                    f"district term {term!r} belongs to both "
                    f"{existing.key!r} and {district.key!r}"
                )
            index[term_key] = district
    return index


def resolve_district(
    city: CityDefinition,
    value: str,
) -> DistrictDefinition | None:
    """Resolve an exact normalized district term within one city."""

    value_key = normalize_district_key(value)
    return build_district_index(city).get(value_key)


def enabled_districts(
    city: CityDefinition,
) -> tuple[DistrictDefinition, ...]:
    """Return enabled districts in their declared order."""

    if not isinstance(city, CityDefinition):
        raise TypeError(
            "city must be a CityDefinition, "
            f"not {type(city).__name__}"
        )
    return tuple(district for district in city.districts if district.enabled)

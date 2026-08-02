"""Pure query planning and queue primitives for lead searches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class QueryKind(str, Enum):
    """Stable categories for planned search queries."""

    BASE = "base"
    NICHE_VARIANT = "niche_variant"
    DISTRICT = "district"
    DISTRICT_VARIANT = "district_variant"


def _normalize_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, not {type(value).__name__}")

    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def normalize_query_key(text: str) -> str:
    """Return a deterministic, case-insensitive key for query text."""

    return _normalize_non_empty_string(text, "text").casefold()


@dataclass(frozen=True)
class SearchQuery:
    """A validated search query and the inputs from which it was built."""

    text: str
    niche: str
    city: str
    kind: QueryKind
    variant: str | None = None
    district: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _normalize_non_empty_string(self.text, "text"),
        )
        object.__setattr__(
            self,
            "niche",
            _normalize_non_empty_string(self.niche, "niche"),
        )
        object.__setattr__(
            self,
            "city",
            _normalize_non_empty_string(self.city, "city"),
        )
        if not isinstance(self.kind, QueryKind):
            raise TypeError("kind must be a QueryKind")
        if self.variant is not None:
            object.__setattr__(
                self,
                "variant",
                _normalize_non_empty_string(self.variant, "variant"),
            )
        if self.district is not None:
            object.__setattr__(
                self,
                "district",
                _normalize_non_empty_string(self.district, "district"),
            )

    @property
    def key(self) -> str:
        """Return the deterministic deduplication key for this query."""

        return normalize_query_key(self.text)


@dataclass(frozen=True)
class QueryQueue:
    """An immutable queue of planned search queries."""

    queries: tuple[SearchQuery, ...]
    next_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.queries, tuple):
            raise TypeError("queries must be a tuple")
        for index, query in enumerate(self.queries):
            if not isinstance(query, SearchQuery):
                raise TypeError(f"queries[{index}] must be a SearchQuery")
        if type(self.next_index) is not int:
            raise TypeError(
                "next_index must be an integer, "
                f"not {type(self.next_index).__name__}"
            )
        if self.next_index < 0:
            raise ValueError("next_index must be non-negative")
        if self.next_index > len(self.queries):
            raise ValueError("next_index must not exceed the number of queries")

    @property
    def total_queries(self) -> int:
        """Return the total number of queries in the plan."""

        return len(self.queries)

    @property
    def remaining_queries(self) -> int:
        """Return the number of queries that have not been taken yet."""

        return self.total_queries - self.next_index

    @property
    def exhausted(self) -> bool:
        """Return whether every query has been taken."""

        return self.next_index == self.total_queries

    @property
    def current_query(self) -> SearchQuery | None:
        """Return the next query without advancing the queue."""

        if self.exhausted:
            return None
        return self.queries[self.next_index]

    def take_next(self) -> tuple[SearchQuery | None, QueryQueue]:
        """Return the next query and an advanced immutable queue state."""

        query = self.current_query
        if query is None:
            return None, self
        return query, QueryQueue(self.queries, self.next_index + 1)


def _normalize_string_sequence(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings, not {type(values).__name__}")
    return tuple(
        _normalize_non_empty_string(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def build_query_queue(
    niche: str,
    city: str,
    *,
    niche_variants: Sequence[str] = (),
    districts: Sequence[str] = (),
    max_queries: int | None = None,
) -> QueryQueue:
    """Build an ordered, deduplicated queue of search queries."""

    normalized_niche = _normalize_non_empty_string(niche, "niche")
    normalized_city = _normalize_non_empty_string(city, "city")
    normalized_variants = _normalize_string_sequence(niche_variants, "niche_variants")
    normalized_districts = _normalize_string_sequence(districts, "districts")

    if max_queries is not None:
        if type(max_queries) is not int:
            raise TypeError(
                "max_queries must be an integer or None, "
                f"not {type(max_queries).__name__}"
            )
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than zero")

    planned: list[SearchQuery] = []
    seen_keys: set[str] = set()

    def append_unique(query: SearchQuery) -> None:
        if query.key not in seen_keys:
            seen_keys.add(query.key)
            planned.append(query)

    append_unique(
        SearchQuery(
            text=f"{normalized_niche} {normalized_city}",
            niche=normalized_niche,
            city=normalized_city,
            kind=QueryKind.BASE,
        )
    )

    for variant in normalized_variants:
        append_unique(
            SearchQuery(
                text=f"{variant} {normalized_city}",
                niche=normalized_niche,
                city=normalized_city,
                kind=QueryKind.NICHE_VARIANT,
                variant=variant,
            )
        )

    for district in normalized_districts:
        append_unique(
            SearchQuery(
                text=f"{normalized_niche} {district} {normalized_city}",
                niche=normalized_niche,
                city=normalized_city,
                kind=QueryKind.DISTRICT,
                district=district,
            )
        )

    for variant in normalized_variants:
        for district in normalized_districts:
            append_unique(
                SearchQuery(
                    text=f"{variant} {district} {normalized_city}",
                    niche=normalized_niche,
                    city=normalized_city,
                    kind=QueryKind.DISTRICT_VARIANT,
                    variant=variant,
                    district=district,
                )
            )

    if max_queries is not None:
        planned = planned[:max_queries]
    return QueryQueue(tuple(planned))

"""Pure allocation rules for sharing discovery budget across queries."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_non_negative_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_positive_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class QueryBudgetAllocation:
    """The global budget remaining and the fair share for one query."""

    remaining_checked_candidates: int
    remaining_opened_cards: int
    active_queries: int
    current_query_card_limit: int

    def __post_init__(self) -> None:
        _validate_non_negative_integer(
            self.remaining_checked_candidates,
            "remaining_checked_candidates",
        )
        _validate_non_negative_integer(
            self.remaining_opened_cards,
            "remaining_opened_cards",
        )
        _validate_positive_integer(self.active_queries, "active_queries")
        _validate_non_negative_integer(
            self.current_query_card_limit,
            "current_query_card_limit",
        )
        if self.current_query_card_limit > self.remaining_checked_candidates:
            raise ValueError(
                "current_query_card_limit must not exceed "
                "remaining_checked_candidates"
            )
        if self.current_query_card_limit > self.remaining_opened_cards:
            raise ValueError(
                "current_query_card_limit must not exceed remaining_opened_cards"
            )

    @property
    def exhausted(self) -> bool:
        """Return whether the current query must not be started."""

        return self.current_query_card_limit == 0


def allocate_query_budget(
    *,
    remaining_checked_candidates: int,
    remaining_opened_cards: int,
    active_queries: int,
) -> QueryBudgetAllocation:
    """Allocate the smaller ceiling-divided share to the current query."""

    _validate_non_negative_integer(
        remaining_checked_candidates,
        "remaining_checked_candidates",
    )
    _validate_non_negative_integer(
        remaining_opened_cards,
        "remaining_opened_cards",
    )
    _validate_positive_integer(active_queries, "active_queries")

    checked_share = (
        remaining_checked_candidates + active_queries - 1
    ) // active_queries
    opened_share = (
        remaining_opened_cards + active_queries - 1
    ) // active_queries

    return QueryBudgetAllocation(
        remaining_checked_candidates=remaining_checked_candidates,
        remaining_opened_cards=remaining_opened_cards,
        active_queries=active_queries,
        current_query_card_limit=min(checked_share, opened_share),
    )

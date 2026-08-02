"""Tests for pure fair query-budget allocation."""

import unittest
from dataclasses import FrozenInstanceError

from query_budget import QueryBudgetAllocation, allocate_query_budget


class QueryBudgetAllocationTests(unittest.TestCase):
    def test_equal_division(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=1000,
            remaining_opened_cards=1000,
            active_queries=5,
        )

        self.assertEqual(allocation.current_query_card_limit, 200)
        self.assertFalse(allocation.exhausted)

    def test_ceiling_division(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=1000,
            remaining_opened_cards=1000,
            active_queries=7,
        )

        self.assertEqual(allocation.current_query_card_limit, 143)

    def test_opened_budget_is_the_smaller_share(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=900,
            remaining_opened_cards=600,
            active_queries=4,
        )

        self.assertEqual(allocation.current_query_card_limit, 150)

    def test_single_query_gets_the_available_remainder(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=1000,
            remaining_opened_cards=700,
            active_queries=1,
        )

        self.assertEqual(allocation.current_query_card_limit, 700)

    def test_budget_smaller_than_query_count_still_allocates_one(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=2,
            remaining_opened_cards=10,
            active_queries=7,
        )

        self.assertEqual(allocation.current_query_card_limit, 1)

    def test_checked_budget_exhaustion_returns_zero(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=0,
            remaining_opened_cards=10,
            active_queries=3,
        )

        self.assertEqual(allocation.current_query_card_limit, 0)
        self.assertTrue(allocation.exhausted)

    def test_opened_budget_exhaustion_returns_zero(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=10,
            remaining_opened_cards=0,
            active_queries=3,
        )

        self.assertEqual(allocation.current_query_card_limit, 0)
        self.assertTrue(allocation.exhausted)

    def test_unused_budget_is_available_to_later_queries(self) -> None:
        first = allocate_query_budget(
            remaining_checked_candidates=10,
            remaining_opened_cards=10,
            active_queries=3,
        )
        second = allocate_query_budget(
            remaining_checked_candidates=9,
            remaining_opened_cards=9,
            active_queries=2,
        )

        self.assertEqual(first.current_query_card_limit, 4)
        self.assertEqual(second.current_query_card_limit, 5)

    def test_allocation_preserves_supplied_global_state(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=11,
            remaining_opened_cards=7,
            active_queries=3,
        )

        self.assertEqual(
            allocation,
            QueryBudgetAllocation(
                remaining_checked_candidates=11,
                remaining_opened_cards=7,
                active_queries=3,
                current_query_card_limit=3,
            ),
        )


class QueryBudgetValidationTests(unittest.TestCase):
    def test_allocator_rejects_invalid_remaining_values(self) -> None:
        invalid_values = (-1, 1.5, "1", None, True, False)
        for field_name in (
            "remaining_checked_candidates",
            "remaining_opened_cards",
        ):
            for value in invalid_values:
                with self.subTest(field=field_name, value=value):
                    arguments = {
                        "remaining_checked_candidates": 1,
                        "remaining_opened_cards": 1,
                        "active_queries": 1,
                        field_name: value,
                    }
                    with self.assertRaises((TypeError, ValueError)):
                        allocate_query_budget(**arguments)  # type: ignore[arg-type]

    def test_allocator_rejects_invalid_active_query_count(self) -> None:
        for value in (0, -1, 1.5, "1", None, True, False):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    allocate_query_budget(
                        remaining_checked_candidates=1,
                        remaining_opened_cards=1,
                        active_queries=value,  # type: ignore[arg-type]
                    )

    def test_manual_allocation_rejects_invalid_field_types(self) -> None:
        invalid_values = (1.5, "1", None, True, False)
        field_names = (
            "remaining_checked_candidates",
            "remaining_opened_cards",
            "active_queries",
            "current_query_card_limit",
        )
        valid = {
            "remaining_checked_candidates": 1,
            "remaining_opened_cards": 1,
            "active_queries": 1,
            "current_query_card_limit": 1,
        }
        for field_name in field_names:
            for value in invalid_values:
                with self.subTest(field=field_name, value=value):
                    arguments = {**valid, field_name: value}
                    with self.assertRaises(TypeError):
                        QueryBudgetAllocation(**arguments)  # type: ignore[arg-type]

    def test_manual_allocation_rejects_invalid_ranges(self) -> None:
        invalid_arguments = (
            dict(
                remaining_checked_candidates=-1,
                remaining_opened_cards=1,
                active_queries=1,
                current_query_card_limit=0,
            ),
            dict(
                remaining_checked_candidates=1,
                remaining_opened_cards=-1,
                active_queries=1,
                current_query_card_limit=0,
            ),
            dict(
                remaining_checked_candidates=1,
                remaining_opened_cards=1,
                active_queries=0,
                current_query_card_limit=0,
            ),
            dict(
                remaining_checked_candidates=1,
                remaining_opened_cards=1,
                active_queries=1,
                current_query_card_limit=-1,
            ),
            dict(
                remaining_checked_candidates=1,
                remaining_opened_cards=2,
                active_queries=1,
                current_query_card_limit=2,
            ),
            dict(
                remaining_checked_candidates=2,
                remaining_opened_cards=1,
                active_queries=1,
                current_query_card_limit=2,
            ),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    QueryBudgetAllocation(**arguments)

    def test_allocation_is_frozen(self) -> None:
        allocation = allocate_query_budget(
            remaining_checked_candidates=1,
            remaining_opened_cards=1,
            active_queries=1,
        )

        with self.assertRaises(FrozenInstanceError):
            allocation.current_query_card_limit = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

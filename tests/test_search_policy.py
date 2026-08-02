"""Tests for the pure search stopping policy contract."""

import unittest
from dataclasses import FrozenInstanceError

from search_policy import (
    SearchDecision,
    SearchPolicy,
    SearchProgress,
    StopReason,
    decide_next,
    limit_to_target,
)


class SearchPolicyDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SearchPolicy(target_leads=100, max_candidates=500)

    def test_search_continues_when_no_stop_condition_is_met(self) -> None:
        progress = SearchProgress(
            qualified_leads=38,
            checked_candidates=120,
            remaining_queries=4,
        )

        self.assertEqual(decide_next(progress, self.policy), SearchDecision(True, None))

    def test_target_reached_at_or_above_target(self) -> None:
        for qualified_leads in (100, 101):
            with self.subTest(qualified_leads=qualified_leads):
                progress = SearchProgress(qualified_leads, 120, 4)
                self.assertEqual(
                    decide_next(progress, self.policy).stop_reason,
                    StopReason.TARGET_REACHED,
                )

    def test_candidate_limit_reached_at_or_above_limit(self) -> None:
        for checked_candidates in (500, 501):
            with self.subTest(checked_candidates=checked_candidates):
                progress = SearchProgress(38, checked_candidates, 4)
                self.assertEqual(
                    decide_next(progress, self.policy).stop_reason,
                    StopReason.MAX_CANDIDATES_REACHED,
                )

    def test_user_stop_has_highest_priority(self) -> None:
        progress = SearchProgress(
            qualified_leads=100,
            checked_candidates=500,
            remaining_queries=0,
            stop_requested=True,
        )

        self.assertEqual(
            decide_next(progress, self.policy).stop_reason,
            StopReason.USER_STOPPED,
        )

    def test_exhausted_queries_stop_search(self) -> None:
        progress = SearchProgress(qualified_leads=38, checked_candidates=120, remaining_queries=0)

        self.assertEqual(
            decide_next(progress, self.policy).stop_reason,
            StopReason.QUERIES_EXHAUSTED,
        )

    def test_target_has_priority_over_candidate_limit(self) -> None:
        progress = SearchProgress(qualified_leads=100, checked_candidates=500, remaining_queries=4)

        self.assertEqual(
            decide_next(progress, self.policy).stop_reason,
            StopReason.TARGET_REACHED,
        )


class ValidationTests(unittest.TestCase):
    def test_policy_rejects_invalid_integer_values(self) -> None:
        invalid_values = (0, -1, 1.5, "1", None, True, False)
        for value in invalid_values:
            with self.subTest(field="target_leads", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    SearchPolicy(target_leads=value, max_candidates=100)  # type: ignore[arg-type]
            with self.subTest(field="max_candidates", value=value):
                with self.assertRaises((TypeError, ValueError)):
                    SearchPolicy(target_leads=1, max_candidates=value)  # type: ignore[arg-type]

    def test_policy_rejects_candidate_limit_below_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_candidates"):
            SearchPolicy(target_leads=100, max_candidates=99)

    def test_progress_rejects_invalid_counters(self) -> None:
        invalid_values = (-1, 1.5, "1", None, True, False)
        field_names = ("qualified_leads", "checked_candidates", "remaining_queries")
        valid = {
            "qualified_leads": 0,
            "checked_candidates": 0,
            "remaining_queries": 0,
        }
        for field_name in field_names:
            for value in invalid_values:
                with self.subTest(field=field_name, value=value):
                    arguments = {**valid, field_name: value}
                    with self.assertRaises((TypeError, ValueError)):
                        SearchProgress(**arguments)  # type: ignore[arg-type]

    def test_progress_rejects_non_boolean_stop_requested(self) -> None:
        for value in (0, 1, "yes", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "stop_requested"):
                    SearchProgress(0, 0, 0, stop_requested=value)  # type: ignore[arg-type]

    def test_decision_rejects_invalid_combinations(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot have a stop reason"):
            SearchDecision(True, StopReason.TARGET_REACHED)
        with self.assertRaisesRegex(ValueError, "must have a stop reason"):
            SearchDecision(False, None)

    def test_decision_rejects_invalid_types(self) -> None:
        with self.assertRaisesRegex(TypeError, "should_continue"):
            SearchDecision(1, None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "stop_reason"):
            SearchDecision(False, "target_reached")  # type: ignore[arg-type]

    def test_models_are_immutable(self) -> None:
        instances_and_fields = (
            (SearchPolicy(1, 1), "target_leads"),
            (SearchProgress(0, 0, 0), "qualified_leads"),
            (SearchDecision.continue_search(), "should_continue"),
        )
        for instance, field_name in instances_and_fields:
            with self.subTest(instance=type(instance).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(instance, field_name, 999)


class LimitToTargetTests(unittest.TestCase):
    def test_limits_large_list_and_preserves_order_and_input(self) -> None:
        original = list(range(120))
        snapshot = original.copy()

        result = limit_to_target(original, 100)

        self.assertEqual(result, list(range(100)))
        self.assertEqual(len(result), 100)
        self.assertEqual(original, snapshot)

    def test_returns_all_items_when_input_is_below_target(self) -> None:
        self.assertEqual(limit_to_target(["a", "b"], 3), ["a", "b"])

    def test_target_of_one(self) -> None:
        self.assertEqual(limit_to_target(("first", "second"), 1), ["first"])

    def test_rejects_invalid_target(self) -> None:
        for value in (0, -1, 1.5, "1", None, True, False):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    limit_to_target([1, 2, 3], value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

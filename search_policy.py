"""Pure rules for deciding when a lead search should stop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Iterable, TypeVar


class StopReason(str, Enum):
    """Stable reasons why a search was stopped."""

    TARGET_REACHED = "target_reached"
    USER_STOPPED = "user_stopped"
    MAX_CANDIDATES_REACHED = "max_candidates_reached"
    QUERIES_EXHAUSTED = "queries_exhausted"


def _validate_positive_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_non_negative_integer(value: object, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class SearchPolicy:
    """Limits that govern a search run."""

    target_leads: int
    max_candidates: int

    def __post_init__(self) -> None:
        _validate_positive_integer(self.target_leads, "target_leads")
        _validate_positive_integer(self.max_candidates, "max_candidates")
        if self.max_candidates < self.target_leads:
            raise ValueError("max_candidates must be greater than or equal to target_leads")


@dataclass(frozen=True)
class SearchProgress:
    """Current counters and user-stop state for a search run."""

    qualified_leads: int
    checked_candidates: int
    remaining_queries: int
    stop_requested: bool = False

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.qualified_leads, "qualified_leads")
        _validate_non_negative_integer(self.checked_candidates, "checked_candidates")
        _validate_non_negative_integer(self.remaining_queries, "remaining_queries")
        if type(self.stop_requested) is not bool:
            raise TypeError(
                "stop_requested must be a boolean, "
                f"not {type(self.stop_requested).__name__}"
            )


@dataclass(frozen=True)
class SearchDecision:
    """Whether to continue and, when stopping, the reason why."""

    should_continue: bool
    stop_reason: StopReason | None

    def __post_init__(self) -> None:
        if type(self.should_continue) is not bool:
            raise TypeError(
                "should_continue must be a boolean, "
                f"not {type(self.should_continue).__name__}"
            )
        if self.stop_reason is not None and not isinstance(self.stop_reason, StopReason):
            raise TypeError("stop_reason must be a StopReason or None")
        if self.should_continue and self.stop_reason is not None:
            raise ValueError("a continuing search cannot have a stop reason")
        if not self.should_continue and self.stop_reason is None:
            raise ValueError("a stopped search must have a stop reason")

    @classmethod
    def continue_search(cls) -> SearchDecision:
        """Create a decision to continue searching."""

        return cls(should_continue=True, stop_reason=None)

    @classmethod
    def stop(cls, reason: StopReason) -> SearchDecision:
        """Create a decision to stop for the given reason."""

        return cls(should_continue=False, stop_reason=reason)


def decide_next(progress: SearchProgress, policy: SearchPolicy) -> SearchDecision:
    """Decide whether a search should continue using the contract priority."""

    if not isinstance(progress, SearchProgress):
        raise TypeError("progress must be a SearchProgress")
    if not isinstance(policy, SearchPolicy):
        raise TypeError("policy must be a SearchPolicy")

    if progress.stop_requested:
        return SearchDecision.stop(StopReason.USER_STOPPED)
    if progress.qualified_leads >= policy.target_leads:
        return SearchDecision.stop(StopReason.TARGET_REACHED)
    if progress.checked_candidates >= policy.max_candidates:
        return SearchDecision.stop(StopReason.MAX_CANDIDATES_REACHED)
    if progress.remaining_queries == 0:
        return SearchDecision.stop(StopReason.QUERIES_EXHAUSTED)
    return SearchDecision.continue_search()


T = TypeVar("T")


def limit_to_target(items: Iterable[T], target_leads: int) -> list[T]:
    """Return at most the first target_leads items without mutating the input."""

    _validate_positive_integer(target_leads, "target_leads")
    return list(islice(items, target_leads))

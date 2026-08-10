"""Deterministic, provider-independent Instagram Resolver V1 benchmark harness."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
import re
from typing import Callable, Iterable

from instagram_candidate_matching import (
    InstagramProviderAuthError,
    InstagramProviderRateLimited,
    InstagramProviderTimeout,
    InstagramProviderUnavailable,
    InstagramSearchProvider,
    InstagramSearchProviderError,
    InstagramSearchRequest,
    candidate_from_instagram_search_result,
    normalize_instagram_username,
    resolve_instagram_candidates,
)
from instagram_resolution import InstagramIdentity, InstagramResolutionStatus


_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "label",
        "business_name",
        "city",
        "address",
        "phone",
        "website_url",
        "expected_username",
    }
)
_SAFE_ERROR_CATEGORIES = frozenset(
    {
        "timeout",
        "rate_limited",
        "authentication",
        "unavailable",
        "provider_error",
        "unexpected_error",
    }
)


class InstagramBenchmarkLabel(str, Enum):
    OFFICIAL_PROFILE = "official_profile"
    NO_OFFICIAL_PROFILE = "no_official_profile"
    DO_NOT_PROMOTE = "do_not_promote"


@dataclass(frozen=True)
class InstagramBenchmarkCase:
    case_id: str
    label: InstagramBenchmarkLabel
    business_name: str
    city: str
    address: str | None
    phone: str | None
    website_url: str | None
    expected_username: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("case_id must be a non-empty simple identifier")
        if not isinstance(self.label, InstagramBenchmarkLabel):
            raise TypeError("label must be an InstagramBenchmarkLabel")

        identity = InstagramIdentity(
            business_name=self.business_name,
            city=self.city,
            address=self.address,
            phone=self.phone,
            website_url=self.website_url,
        )
        object.__setattr__(self, "business_name", identity.business_name)
        object.__setattr__(self, "city", identity.city)
        object.__setattr__(self, "address", identity.address)
        object.__setattr__(self, "phone", identity.phone)
        object.__setattr__(self, "website_url", identity.website_url)
        if not any((identity.address, identity.phone, identity.website_url)):
            raise ValueError("at least one strong identity field is required")

        if self.label is InstagramBenchmarkLabel.OFFICIAL_PROFILE:
            if self.expected_username is None:
                raise ValueError("OFFICIAL_PROFILE requires expected_username")
            object.__setattr__(
                self,
                "expected_username",
                normalize_instagram_username(self.expected_username),
            )
        elif self.expected_username is not None:
            raise ValueError(f"{self.label.value} forbids expected_username")


def load_benchmark_cases(path: str | Path) -> tuple[InstagramBenchmarkCase, ...]:
    """Load the strict V1 JSON-list contract without constructing a provider."""

    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark dataset is not valid UTF-8 JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("benchmark dataset must be a JSON list")

    cases: list[InstagramBenchmarkCase] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str | None, str | None, str | None]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != _CASE_FIELDS:
            raise ValueError(f"benchmark case {index} has an invalid schema")
        try:
            case = InstagramBenchmarkCase(
                case_id=item["case_id"],
                label=InstagramBenchmarkLabel(item["label"]),
                business_name=item["business_name"],
                city=item["city"],
                address=item["address"],
                phone=item["phone"],
                website_url=item["website_url"],
                expected_username=item["expected_username"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark case {index}: {exc}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate benchmark case_id: {case.case_id}")
        identity_key = (
            case.business_name.casefold(),
            case.city.casefold(),
            case.address.casefold() if case.address else None,
            case.phone,
            case.website_url,
        )
        if identity_key in seen_identities:
            raise ValueError(f"duplicate exact business identity at case {index}")
        seen_ids.add(case.case_id)
        seen_identities.add(identity_key)
        cases.append(case)
    return tuple(cases)


@dataclass(frozen=True)
class InstagramBenchmarkCaseResult:
    case_id: str
    label: InstagramBenchmarkLabel
    expected_username: str | None
    resolution_status: str
    resolved_username: str | None
    correct_promotion: bool
    critical_false_promotion: bool
    safe_nonpromotion: bool
    technical_success: bool
    error_category: str | None
    requests_used: int
    tool_calls_seen: int
    search_actions_seen: int
    open_page_actions_seen: int
    find_in_page_actions_seen: int
    candidates_returned: int
    identity_candidates_rejected: int
    sources_seen: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, InstagramBenchmarkLabel):
            raise TypeError("label must be an InstagramBenchmarkLabel")
        if self.error_category not in _SAFE_ERROR_CATEGORIES | {None}:
            raise ValueError("unsupported safe error category")
        for name in (
            "requests_used",
            "tool_calls_seen",
            "search_actions_seen",
            "open_page_actions_seen",
            "find_in_page_actions_seen",
            "candidates_returned",
            "identity_candidates_rejected",
            "sources_seen",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class _Telemetry:
    requests_started: int = 0
    tool_calls_seen: int = 0
    search_actions_seen: int = 0
    open_page_actions_seen: int = 0
    find_in_page_actions_seen: int = 0
    candidates_returned: int = 0
    identity_candidates_rejected: int = 0
    sources_seen: int = 0
    tool_call_limit_exceeded: bool = False


def _telemetry_snapshot(provider: object) -> _Telemetry:
    current = provider
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        telemetry = getattr(current, "telemetry", None)
        if callable(telemetry):
            try:
                snapshot = telemetry()
                values = {
                    name: (
                        bool(getattr(snapshot, name, False))
                        if name == "tool_call_limit_exceeded"
                        else max(0, int(getattr(snapshot, name, 0)))
                    )
                    for name in _Telemetry.__dataclass_fields__
                }
                return _Telemetry(**values)
            except (AttributeError, TypeError, ValueError):
                return _Telemetry()
        nested = getattr(current, "provider", None)
        if nested is None:
            break
        current = nested
    return _Telemetry()


def _telemetry_delta(before: _Telemetry, after: _Telemetry) -> _Telemetry:
    values: dict[str, int | bool] = {}
    for name in _Telemetry.__dataclass_fields__:
        if name == "tool_call_limit_exceeded":
            values[name] = after.tool_call_limit_exceeded and not before.tool_call_limit_exceeded
        else:
            values[name] = max(0, getattr(after, name) - getattr(before, name))
    return _Telemetry(**values)


def _error_category(error: BaseException) -> str:
    if isinstance(error, InstagramProviderTimeout):
        return "timeout"
    if isinstance(error, InstagramProviderRateLimited):
        return "rate_limited"
    if isinstance(error, InstagramProviderAuthError):
        return "authentication"
    if isinstance(error, InstagramProviderUnavailable):
        return "unavailable"
    if isinstance(error, InstagramSearchProviderError):
        return "provider_error"
    return "unexpected_error"


def _case_result(
    case: InstagramBenchmarkCase,
    resolution_status: InstagramResolutionStatus,
    resolved_username: str | None,
    telemetry: _Telemetry,
    *,
    candidates_returned: int,
    error_category: str | None = None,
) -> InstagramBenchmarkCaseResult:
    technical_success = resolution_status is not InstagramResolutionStatus.RESOLUTION_ERROR
    found = resolution_status is InstagramResolutionStatus.FOUND_OFFICIAL
    correct = (
        case.label is InstagramBenchmarkLabel.OFFICIAL_PROFILE
        and found
        and resolved_username == case.expected_username
    )
    critical = found and not correct
    safe_nonpromotion = technical_success and resolution_status in {
        InstagramResolutionStatus.NOT_FOUND,
        InstagramResolutionStatus.UNCERTAIN,
    }
    return InstagramBenchmarkCaseResult(
        case_id=case.case_id,
        label=case.label,
        expected_username=case.expected_username,
        resolution_status=resolution_status.value,
        resolved_username=resolved_username,
        correct_promotion=correct,
        critical_false_promotion=critical,
        safe_nonpromotion=safe_nonpromotion,
        technical_success=technical_success,
        error_category=error_category,
        requests_used=telemetry.requests_started or 1,
        tool_calls_seen=telemetry.tool_calls_seen,
        search_actions_seen=telemetry.search_actions_seen,
        open_page_actions_seen=telemetry.open_page_actions_seen,
        find_in_page_actions_seen=telemetry.find_in_page_actions_seen,
        candidates_returned=telemetry.candidates_returned or candidates_returned,
        identity_candidates_rejected=telemetry.identity_candidates_rejected,
        sources_seen=telemetry.sources_seen,
    )


async def run_benchmark_case(
    case: InstagramBenchmarkCase,
    provider: InstagramSearchProvider,
) -> InstagramBenchmarkCaseResult:
    """Run exactly one provider search; expected labels remain evaluator-only."""

    if not isinstance(case, InstagramBenchmarkCase):
        raise TypeError("case must be an InstagramBenchmarkCase")
    request = InstagramSearchRequest(
        business_name=case.business_name,
        city=case.city,
        address=case.address,
        phone=case.phone,
        website_url=case.website_url,
    )
    before = _telemetry_snapshot(provider)
    try:
        results = await provider.search(request)
    except Exception as error:
        telemetry = _telemetry_delta(before, _telemetry_snapshot(provider))
        return _case_result(
            case,
            InstagramResolutionStatus.RESOLUTION_ERROR,
            None,
            telemetry,
            candidates_returned=0,
            error_category=_error_category(error),
        )
    telemetry = _telemetry_delta(before, _telemetry_snapshot(provider))
    try:
        if not isinstance(results, tuple):
            raise TypeError("provider result must be a tuple")
        candidates = tuple(candidate_from_instagram_search_result(item) for item in results)
        identity = InstagramIdentity(
            business_name=case.business_name,
            city=case.city,
            address=case.address,
            phone=case.phone,
            website_url=case.website_url,
        )
        resolution = resolve_instagram_candidates(identity, candidates)
    except Exception:
        return _case_result(
            case,
            InstagramResolutionStatus.RESOLUTION_ERROR,
            None,
            telemetry,
            candidates_returned=len(results),
            error_category="unexpected_error",
        )
    return _case_result(
        case,
        resolution.status,
        resolution.username,
        telemetry,
        candidates_returned=len(results),
    )


@dataclass(frozen=True)
class InstagramBenchmarkMetrics:
    total_cases: int
    official_cases: int
    negative_cases: int
    do_not_promote_cases: int
    technical_success_count: int
    technical_success_rate: float
    correct_official_promotions: int
    wrong_official_promotions: int
    safe_official_misses: int
    correct_negative_nonpromotions: int
    critical_false_promotions: int
    official_recall: float
    promotion_precision: float
    negative_specificity: float
    total_requests_used: int
    max_requests_used_per_case: int


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_benchmark(
    results: Iterable[InstagramBenchmarkCaseResult],
) -> InstagramBenchmarkMetrics:
    items = tuple(results)
    official = sum(item.label is InstagramBenchmarkLabel.OFFICIAL_PROFILE for item in items)
    negative = len(items) - official
    correct = sum(item.correct_promotion for item in items)
    wrong_official = sum(
        item.label is InstagramBenchmarkLabel.OFFICIAL_PROFILE
        and item.critical_false_promotion
        for item in items
    )
    critical_negative = sum(
        item.label is not InstagramBenchmarkLabel.OFFICIAL_PROFILE
        and item.critical_false_promotion
        for item in items
    )
    safe_official = sum(
        item.label is InstagramBenchmarkLabel.OFFICIAL_PROFILE and item.safe_nonpromotion
        for item in items
    )
    correct_negative = sum(
        item.label is not InstagramBenchmarkLabel.OFFICIAL_PROFILE
        and item.safe_nonpromotion
        for item in items
    )
    technical = sum(item.technical_success for item in items)
    return InstagramBenchmarkMetrics(
        total_cases=len(items),
        official_cases=official,
        negative_cases=negative,
        do_not_promote_cases=sum(
            item.label is InstagramBenchmarkLabel.DO_NOT_PROMOTE for item in items
        ),
        technical_success_count=technical,
        technical_success_rate=_ratio(technical, len(items)),
        correct_official_promotions=correct,
        wrong_official_promotions=wrong_official,
        safe_official_misses=safe_official,
        correct_negative_nonpromotions=correct_negative,
        critical_false_promotions=critical_negative,
        official_recall=_ratio(correct, official),
        promotion_precision=_ratio(correct, correct + wrong_official + critical_negative),
        negative_specificity=_ratio(correct_negative, negative),
        total_requests_used=sum(item.requests_used for item in items),
        max_requests_used_per_case=max((item.requests_used for item in items), default=0),
    )


class InstagramBenchmarkGateDecision(str, Enum):
    PASS = "PASS"
    FAIL_DATASET_SHAPE = "FAIL_DATASET_SHAPE"
    FAIL_FALSE_PROMOTION = "FAIL_FALSE_PROMOTION"
    FAIL_PRECISION = "FAIL_PRECISION"
    FAIL_SPECIFICITY = "FAIL_SPECIFICITY"
    FAIL_RECALL = "FAIL_RECALL"
    FAIL_TECHNICAL = "FAIL_TECHNICAL"
    FAIL_TOOL_SAFETY = "FAIL_TOOL_SAFETY"


def evaluate_benchmark_gate(
    metrics: InstagramBenchmarkMetrics,
    results: Iterable[InstagramBenchmarkCaseResult],
) -> InstagramBenchmarkGateDecision:
    items = tuple(results)
    if (
        metrics.total_cases < 12
        or metrics.official_cases < 8
        or metrics.negative_cases < 4
        or metrics.do_not_promote_cases < 2
    ):
        return InstagramBenchmarkGateDecision.FAIL_DATASET_SHAPE
    if metrics.critical_false_promotions or metrics.wrong_official_promotions:
        return InstagramBenchmarkGateDecision.FAIL_FALSE_PROMOTION
    if metrics.promotion_precision < 0.95:
        return InstagramBenchmarkGateDecision.FAIL_PRECISION
    if metrics.negative_specificity < 1.0:
        return InstagramBenchmarkGateDecision.FAIL_SPECIFICITY
    if metrics.official_recall < 0.50:
        return InstagramBenchmarkGateDecision.FAIL_RECALL
    if metrics.technical_success_rate < 0.90:
        return InstagramBenchmarkGateDecision.FAIL_TECHNICAL
    if any(
        item.requests_used > 1
        or item.tool_calls_seen > 1
        or item.open_page_actions_seen > 0
        or item.find_in_page_actions_seen > 0
        for item in items
    ):
        return InstagramBenchmarkGateDecision.FAIL_TOOL_SAFETY
    return InstagramBenchmarkGateDecision.PASS


def _result_record(result: InstagramBenchmarkCaseResult) -> dict[str, object]:
    record = asdict(result)
    record["label"] = result.label.value
    return record


def write_benchmark_outputs(
    output_path: str | Path,
    results: Iterable[InstagramBenchmarkCaseResult],
    metrics: InstagramBenchmarkMetrics,
    decision: InstagramBenchmarkGateDecision,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = tuple(results)
    report = {
        "results": [_result_record(item) for item in items],
        "metrics": asdict(metrics),
        "gate_decision": decision.value,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = path.with_suffix(".txt")
    summary_path.write_text(
        "\n".join(
            (
                "Instagram Resolution Benchmark V1",
                "",
                f"cases: {metrics.total_cases}",
                f"official cases: {metrics.official_cases}",
                f"negative cases: {metrics.negative_cases}",
                f"do-not-promote cases: {metrics.do_not_promote_cases}",
                f"official recall: {metrics.official_recall:.4f}",
                f"promotion precision: {metrics.promotion_precision:.4f}",
                f"negative specificity: {metrics.negative_specificity:.4f}",
                f"technical success rate: {metrics.technical_success_rate:.4f}",
                f"critical false promotions: {metrics.critical_false_promotions}",
                f"wrong official promotions: {metrics.wrong_official_promotions}",
                f"gate decision: {decision.value}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


ProviderFactory = Callable[[], InstagramSearchProvider]


async def run_benchmark(
    dataset_path: str | Path,
    output_path: str | Path,
    provider_factory: ProviderFactory,
    *,
    expected_dataset_sha256: str | None = None,
) -> InstagramBenchmarkGateDecision:
    cases = load_benchmark_cases(dataset_path)
    if expected_dataset_sha256 is not None:
        expected = expected_dataset_sha256.strip().casefold()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("expected dataset SHA256 must be 64 lowercase hex characters")
        actual = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("benchmark dataset SHA256 mismatch")
    provider = provider_factory()
    if inspect.isawaitable(provider):
        raise TypeError("provider_factory must return a provider synchronously")
    if not isinstance(provider, InstagramSearchProvider):
        raise TypeError("provider_factory returned an invalid provider")
    results = tuple([await run_benchmark_case(case, provider) for case in cases])
    metrics = summarize_benchmark(results)
    decision = evaluate_benchmark_gate(metrics, results)
    write_benchmark_outputs(output_path, results, metrics, decision)
    return decision


def _default_provider_factory() -> InstagramSearchProvider:
    from instagram_search_runtime import build_configured_instagram_search_provider

    provider = build_configured_instagram_search_provider()
    if provider is None:
        raise RuntimeError("configured Instagram search provider is disabled")
    return provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-dataset-sha256")
    arguments = parser.parse_args(argv)
    try:
        decision = asyncio.run(
            run_benchmark(
                arguments.dataset,
                arguments.output,
                _default_provider_factory,
                expected_dataset_sha256=arguments.expected_dataset_sha256,
            )
        )
    except Exception as exc:
        print(f"benchmark_setup_failed={type(exc).__name__}")
        return 2
    print(f"gate_decision={decision.value}")
    return 0 if decision is InstagramBenchmarkGateDecision.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())

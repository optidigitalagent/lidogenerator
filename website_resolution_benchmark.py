"""Offline contracts and safe reporting for website-resolution benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import csv
import json
from pathlib import Path
import re
from typing import Iterable

from website_candidate_matching import (
    BusinessIdentity,
    ProviderAuthError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    SearchProviderError,
    SearchRequest,
    SourceAttempt,
    SourceAttemptStatus,
    candidate_from_search_result,
    resolve_website_candidates,
)
from website_resolution import (
    CandidateKind,
    CandidateSource,
    ResolutionStatus,
    classify_candidate_url,
    normalize_domain,
)


_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "business_name",
        "city",
        "address",
        "phone",
        "label",
        "expected_domain",
        "notes",
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


class BenchmarkLabel(str, Enum):
    OFFICIAL_DOMAIN = "OFFICIAL_DOMAIN"
    NO_OFFICIAL_SITE = "NO_OFFICIAL_SITE"


@dataclass(frozen=True)
class WebsiteResolutionBenchmarkCase:
    case_id: str
    business_name: str
    city: str
    address: str | None
    phone: str | None
    label: BenchmarkLabel
    expected_domain: str | None
    notes: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("case_id must be a non-empty simple identifier")
        if not isinstance(self.label, BenchmarkLabel):
            raise TypeError("label must be a BenchmarkLabel")
        identity = BusinessIdentity(
            name=self.business_name,
            city=self.city,
            address=self.address,
            phone=self.phone,
        )
        object.__setattr__(self, "business_name", identity.name)
        object.__setattr__(self, "city", identity.city)
        object.__setattr__(self, "address", identity.address)
        object.__setattr__(self, "phone", identity.phone)
        if self.notes is not None:
            if not isinstance(self.notes, str):
                raise TypeError("notes must be a string or None")
            notes = " ".join(self.notes.split())
            if not notes:
                raise ValueError("notes must not be empty when supplied")
            object.__setattr__(self, "notes", notes)

        if self.label is BenchmarkLabel.OFFICIAL_DOMAIN:
            if self.expected_domain is None:
                raise ValueError("OFFICIAL_DOMAIN requires expected_domain")
            domain = normalize_domain(self.expected_domain)
            if classify_candidate_url(f"https://{domain}/") is not CandidateKind.UNKNOWN:
                raise ValueError("expected_domain must be an ordinary business domain")
            object.__setattr__(self, "expected_domain", domain)
        elif self.expected_domain is not None:
            raise ValueError("NO_OFFICIAL_SITE forbids expected_domain")


def load_benchmark_cases(path: str | Path) -> tuple[WebsiteResolutionBenchmarkCase, ...]:
    """Load a strict, ordered, versioned UTF-8 benchmark dataset."""

    dataset_path = Path(path)
    try:
        with dataset_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("benchmark dataset is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "cases"}:
        raise ValueError("benchmark dataset must contain only version and cases")
    if payload["version"] != 1 or type(payload["version"]) is not int:
        raise ValueError("unsupported benchmark dataset version")
    if not isinstance(payload["cases"], list):
        raise ValueError("benchmark cases must be an array")

    cases: list[WebsiteResolutionBenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload["cases"]):
        if not isinstance(item, dict) or set(item) != _CASE_FIELDS:
            raise ValueError(f"benchmark case {index} has an invalid schema")
        try:
            label = BenchmarkLabel(item["label"])
            case = WebsiteResolutionBenchmarkCase(
                case_id=item["case_id"],
                business_name=item["business_name"],
                city=item["city"],
                address=item["address"],
                phone=item["phone"],
                label=label,
                expected_domain=item["expected_domain"],
                notes=item["notes"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark case {index}: {exc}") from exc
        if case.case_id in seen_ids:
            raise ValueError(f"duplicate benchmark case_id: {case.case_id}")
        seen_ids.add(case.case_id)
        cases.append(case)
    return tuple(cases)


@dataclass(frozen=True)
class WebsiteResolutionBenchmarkResult:
    case_id: str
    label: BenchmarkLabel
    expected_domain: str | None
    provider_request_succeeded: bool
    provider_result_domains: tuple[str, ...]
    resolution_status: str
    resolved_domain: str | None
    expected_domain_returned: bool
    expected_domain_promoted: bool
    wrong_domain_returned: bool
    wrong_domain_promoted: bool
    safe_no_match: bool
    tool_calls_seen: int
    search_actions_seen: int
    open_page_actions_seen: int
    find_in_page_actions_seen: int
    sources_seen: int
    identity_candidates_rejected: int
    error_category: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.label, BenchmarkLabel):
            raise TypeError("label must be a BenchmarkLabel")
        if not isinstance(self.provider_result_domains, tuple):
            raise TypeError("provider_result_domains must be a tuple")
        if self.error_category is not None and self.error_category not in _SAFE_ERROR_CATEGORIES:
            raise ValueError("unsupported safe error category")
        for name in (
            "tool_calls_seen",
            "search_actions_seen",
            "open_page_actions_seen",
            "find_in_page_actions_seen",
            "sources_seen",
            "identity_candidates_rejected",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class _Telemetry:
    tool_calls_seen: int = 0
    search_actions_seen: int = 0
    open_page_actions_seen: int = 0
    find_in_page_actions_seen: int = 0
    sources_seen: int = 0
    identity_candidates_rejected: int = 0


def _telemetry_snapshot(provider: object) -> _Telemetry:
    current = provider
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        telemetry = getattr(current, "telemetry", None)
        if callable(telemetry):
            try:
                snapshot = telemetry()
                return _Telemetry(**{
                    name: max(0, int(getattr(snapshot, name, 0)))
                    for name in _Telemetry.__dataclass_fields__
                })
            except (AttributeError, TypeError, ValueError):
                return _Telemetry()
        nested = getattr(current, "provider", None)
        if nested is None:
            break
        current = nested
    return _Telemetry()


def _telemetry_delta(before: _Telemetry, after: _Telemetry) -> _Telemetry:
    return _Telemetry(**{
        name: max(0, getattr(after, name) - getattr(before, name))
        for name in _Telemetry.__dataclass_fields__
    })


def _error_category(error: BaseException) -> str:
    if isinstance(error, ProviderTimeout):
        return "timeout"
    if isinstance(error, ProviderRateLimited):
        return "rate_limited"
    if isinstance(error, ProviderAuthError):
        return "authentication"
    if isinstance(error, ProviderUnavailable):
        return "unavailable"
    if isinstance(error, SearchProviderError):
        return "provider_error"
    return "unexpected_error"


def _failed_result(
    case: WebsiteResolutionBenchmarkCase,
    telemetry: _Telemetry,
    category: str,
) -> WebsiteResolutionBenchmarkResult:
    return WebsiteResolutionBenchmarkResult(
        case_id=case.case_id,
        label=case.label,
        expected_domain=case.expected_domain,
        provider_request_succeeded=False,
        provider_result_domains=(),
        resolution_status=ResolutionStatus.RESOLUTION_ERROR.value,
        resolved_domain=None,
        expected_domain_returned=False,
        expected_domain_promoted=False,
        wrong_domain_returned=False,
        wrong_domain_promoted=False,
        safe_no_match=True,
        tool_calls_seen=telemetry.tool_calls_seen,
        search_actions_seen=telemetry.search_actions_seen,
        open_page_actions_seen=telemetry.open_page_actions_seen,
        find_in_page_actions_seen=telemetry.find_in_page_actions_seen,
        sources_seen=telemetry.sources_seen,
        identity_candidates_rejected=telemetry.identity_candidates_rejected,
        error_category=category,
    )


async def run_benchmark_case(
    case: WebsiteResolutionBenchmarkCase,
    provider: object,
) -> WebsiteResolutionBenchmarkResult:
    """Evaluate one case with exactly one provider search and no retries."""

    if not isinstance(case, WebsiteResolutionBenchmarkCase):
        raise TypeError("case must be a WebsiteResolutionBenchmarkCase")
    request = SearchRequest(
        business_name=case.business_name,
        city=case.city,
        address=case.address,
        phone=case.phone,
    )
    before = _telemetry_snapshot(provider)
    try:
        results = await provider.search(request)
    except Exception as error:
        return _failed_result(
            case,
            _telemetry_delta(before, _telemetry_snapshot(provider)),
            _error_category(error),
        )
    telemetry = _telemetry_delta(before, _telemetry_snapshot(provider))
    try:
        if not isinstance(results, tuple):
            raise TypeError("provider result must be a tuple")
        candidates = tuple(candidate_from_search_result(result) for result in results)
        identity = BusinessIdentity(
            name=case.business_name,
            city=case.city,
            address=case.address,
            phone=case.phone,
        )
        attempt = SourceAttempt(
            CandidateSource.WEB_SEARCH,
            SourceAttemptStatus.COMPLETED,
        )
        resolution = resolve_website_candidates(
            identity,
            candidates,
            (attempt,),
            (CandidateSource.WEB_SEARCH,),
        )
        domains = tuple(dict.fromkeys(normalize_domain(result.url) for result in results))
        ordinary_domains = {
            normalize_domain(result.url)
            for result in results
            if classify_candidate_url(result.url) is CandidateKind.UNKNOWN
        }
        resolved_domain = (
            normalize_domain(resolution.resolved_url)
            if resolution.resolved_url is not None
            else None
        )
    except Exception:
        failed = _failed_result(case, telemetry, "unexpected_error")
        return WebsiteResolutionBenchmarkResult(
            **{**asdict(failed), "provider_request_succeeded": True}
        )

    if case.label is BenchmarkLabel.OFFICIAL_DOMAIN:
        expected_returned = case.expected_domain in domains
        expected_promoted = resolved_domain == case.expected_domain
        wrong_returned = any(domain != case.expected_domain for domain in ordinary_domains)
        wrong_promoted = resolved_domain is not None and resolved_domain != case.expected_domain
        safe_no_match = not wrong_promoted and not expected_promoted
    else:
        expected_returned = False
        expected_promoted = False
        wrong_returned = bool(ordinary_domains)
        wrong_promoted = resolved_domain is not None
        safe_no_match = resolved_domain is None

    return WebsiteResolutionBenchmarkResult(
        case_id=case.case_id,
        label=case.label,
        expected_domain=case.expected_domain,
        provider_request_succeeded=True,
        provider_result_domains=domains,
        resolution_status=resolution.status.value,
        resolved_domain=resolved_domain,
        expected_domain_returned=expected_returned,
        expected_domain_promoted=expected_promoted,
        wrong_domain_returned=wrong_returned,
        wrong_domain_promoted=wrong_promoted,
        safe_no_match=safe_no_match,
        tool_calls_seen=telemetry.tool_calls_seen,
        search_actions_seen=telemetry.search_actions_seen,
        open_page_actions_seen=telemetry.open_page_actions_seen,
        find_in_page_actions_seen=telemetry.find_in_page_actions_seen,
        sources_seen=telemetry.sources_seen,
        identity_candidates_rejected=telemetry.identity_candidates_rejected,
        error_category=None,
    )


@dataclass(frozen=True)
class WebsiteResolutionBenchmarkSummary:
    total_cases: int
    official_domain_cases: int
    no_official_site_cases: int
    provider_requests_started: int
    provider_requests_succeeded: int
    provider_requests_failed: int
    expected_domains_returned: int
    expected_domains_promoted: int
    provider_wrong_domain_cases: int
    wrong_domains_promoted: int
    safe_no_matches: int
    tool_call_limit_violations: int
    total_tool_calls: int
    total_search_actions: int
    total_open_page_actions: int
    total_find_in_page_actions: int
    total_sources_seen: int
    total_identity_candidates_rejected: int
    provider_domain_recall: float | None
    resolver_domain_recall: float | None
    resolver_precision: float | None
    no_site_specificity: float | None
    critical_false_positive_rate: float | None
    technical_success_rate: float | None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_benchmark(
    results: Iterable[WebsiteResolutionBenchmarkResult],
) -> WebsiteResolutionBenchmarkSummary:
    items = tuple(results)
    official = sum(item.label is BenchmarkLabel.OFFICIAL_DOMAIN for item in items)
    no_site = sum(item.label is BenchmarkLabel.NO_OFFICIAL_SITE for item in items)
    succeeded = sum(item.provider_request_succeeded for item in items)
    expected_returned = sum(item.expected_domain_returned for item in items)
    expected_promoted = sum(item.expected_domain_promoted for item in items)
    wrong_promoted = sum(item.wrong_domain_promoted for item in items)
    all_promoted = sum(item.resolved_domain is not None for item in items)
    correct_no_site = sum(
        item.label is BenchmarkLabel.NO_OFFICIAL_SITE and item.resolved_domain is None
        for item in items
    )
    return WebsiteResolutionBenchmarkSummary(
        total_cases=len(items),
        official_domain_cases=official,
        no_official_site_cases=no_site,
        provider_requests_started=len(items),
        provider_requests_succeeded=succeeded,
        provider_requests_failed=len(items) - succeeded,
        expected_domains_returned=expected_returned,
        expected_domains_promoted=expected_promoted,
        provider_wrong_domain_cases=sum(item.wrong_domain_returned for item in items),
        wrong_domains_promoted=wrong_promoted,
        safe_no_matches=sum(item.safe_no_match for item in items),
        tool_call_limit_violations=sum(item.tool_calls_seen > 1 for item in items),
        total_tool_calls=sum(item.tool_calls_seen for item in items),
        total_search_actions=sum(item.search_actions_seen for item in items),
        total_open_page_actions=sum(item.open_page_actions_seen for item in items),
        total_find_in_page_actions=sum(item.find_in_page_actions_seen for item in items),
        total_sources_seen=sum(item.sources_seen for item in items),
        total_identity_candidates_rejected=sum(
            item.identity_candidates_rejected for item in items
        ),
        provider_domain_recall=_ratio(expected_returned, official),
        resolver_domain_recall=_ratio(expected_promoted, official),
        resolver_precision=_ratio(expected_promoted, all_promoted),
        no_site_specificity=_ratio(correct_no_site, no_site),
        critical_false_positive_rate=_ratio(wrong_promoted, len(items)),
        technical_success_rate=_ratio(succeeded, len(items)),
    )


class BenchmarkGateDecision(str, Enum):
    PASS = "PASS"
    FAIL_CRITICAL_FALSE_POSITIVE = "FAIL_CRITICAL_FALSE_POSITIVE"
    FAIL_LOW_PRECISION = "FAIL_LOW_PRECISION"
    FAIL_LOW_RECALL = "FAIL_LOW_RECALL"
    FAIL_TECHNICAL = "FAIL_TECHNICAL"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"


def evaluate_benchmark_gate(
    summary: WebsiteResolutionBenchmarkSummary,
) -> BenchmarkGateDecision:
    """Apply conservative pre-production V1 gates in fixed priority order."""

    if (
        summary.total_cases < 12
        or summary.official_domain_cases < 8
        or summary.no_official_site_cases < 4
    ):
        return BenchmarkGateDecision.INSUFFICIENT_SAMPLE
    if summary.wrong_domains_promoted > 0:
        return BenchmarkGateDecision.FAIL_CRITICAL_FALSE_POSITIVE
    if summary.resolver_precision is not None and summary.resolver_precision < 0.95:
        return BenchmarkGateDecision.FAIL_LOW_PRECISION
    if summary.resolver_domain_recall is not None and summary.resolver_domain_recall < 0.60:
        return BenchmarkGateDecision.FAIL_LOW_RECALL
    if summary.technical_success_rate is not None and summary.technical_success_rate < 0.90:
        return BenchmarkGateDecision.FAIL_TECHNICAL
    return BenchmarkGateDecision.PASS


def _result_record(result: WebsiteResolutionBenchmarkResult) -> dict[str, object]:
    record = asdict(result)
    record["label"] = result.label.value
    record["provider_result_domains"] = list(result.provider_result_domains)
    return record


def _summary_record(summary: WebsiteResolutionBenchmarkSummary) -> dict[str, object]:
    return asdict(summary)


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_benchmark_outputs(
    output_dir: str | Path,
    results: Iterable[WebsiteResolutionBenchmarkResult],
    summary: WebsiteResolutionBenchmarkSummary,
    decision: BenchmarkGateDecision,
) -> None:
    """Write only allowlisted benchmark result fields and aggregate metrics."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    items = tuple(results)
    records = [_result_record(item) for item in items]
    (directory / "benchmark_results.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(WebsiteResolutionBenchmarkResult.__dataclass_fields__)
    with (directory / "benchmark_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_record = dict(record)
            csv_record["provider_result_domains"] = ";".join(
                csv_record["provider_result_domains"]
            )
            writer.writerow(csv_record)
    summary_record = _summary_record(summary)
    summary_record["gate_decision"] = decision.value
    (directory / "benchmark_summary.json").write_text(
        json.dumps(summary_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    critical_ids = [
        item.case_id
        for item in items
        if item.wrong_domain_promoted
        or not item.provider_request_succeeded
        or (
            item.label is BenchmarkLabel.OFFICIAL_DOMAIN
            and not item.expected_domain_promoted
        )
    ]
    lines = [
        "Website Resolution Benchmark",
        "",
        "Dataset:",
        f"- cases: {summary.total_cases}",
        f"- official domains: {summary.official_domain_cases}",
        f"- no-site: {summary.no_official_site_cases}",
        "",
        "Technical:",
        f"- requests succeeded: {summary.provider_requests_succeeded}",
        f"- requests failed: {summary.provider_requests_failed}",
        f"- tool calls: {summary.total_tool_calls}",
        f"- search actions: {summary.total_search_actions}",
        f"- open actions: {summary.total_open_page_actions}",
        f"- find actions: {summary.total_find_in_page_actions}",
        f"- source count: {summary.total_sources_seen}",
        "",
        "Accuracy:",
        f"- provider domain recall: {_format_metric(summary.provider_domain_recall)}",
        f"- resolver domain recall: {_format_metric(summary.resolver_domain_recall)}",
        f"- resolver precision: {_format_metric(summary.resolver_precision)}",
        f"- no-site specificity: {_format_metric(summary.no_site_specificity)}",
        f"- critical false positives: {summary.wrong_domains_promoted}",
        f"- safe no-match count: {summary.safe_no_matches}",
        "",
        "Gate:",
        f"- decision: {decision.value}",
        "",
        "Critical cases:",
    ]
    lines.extend(f"- {case_id}" for case_id in critical_ids)
    if not critical_ids:
        lines.append("- none")
    (directory / "benchmark_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

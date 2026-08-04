"""Run one explicitly authorized OpenAI website-search validation request."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LIVE_GATE = "ALLOW_LIVE_OPENAI_WEB_SEARCH_VALIDATION"
EXPECTED_DOMAIN = "status-dent.zp.ua"
WRONG_SAME_NAME_DOMAIN = "status-dental-clinic.com.ua"


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, TypeError, ValueError):
            pass


def _environment_gate() -> tuple[bool, str]:
    if os.getenv(LIVE_GATE) != "1":
        return False, "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_NO_EXPLICIT_OPT_IN"
    if os.getenv("WEBSITE_SEARCH_PROVIDER", "").strip().casefold() != "openai":
        return False, "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_PROVIDER_NOT_OPENAI"
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return False, "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_MISSING_API_KEY"
    try:
        budget = int(os.getenv("MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK", ""))
    except ValueError:
        return False, "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_INVALID_BUDGET"
    if budget != 1:
        return False, "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_BUDGET_MUST_EQUAL_ONE"
    if not os.getenv("OPENAI_WEB_SEARCH_MODEL", "").strip():
        return False, "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_MISSING_MODEL"
    return True, ""


def _error_category(error: BaseException) -> str:
    from website_candidate_matching import (
        ProviderAuthError,
        ProviderRateLimited,
        ProviderTimeout,
        ProviderUnavailable,
        SearchProviderError,
    )

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


def _print_telemetry(telemetry: object) -> None:
    print(f"requests_started={telemetry.requests_started}")
    print(f"requests_succeeded={telemetry.requests_succeeded}")
    print(f"requests_failed={telemetry.requests_failed}")
    print(f"tool_calls_seen={telemetry.tool_calls_seen}")
    print(f"search_actions_seen={telemetry.search_actions_seen}")
    print(f"open_page_actions_seen={telemetry.open_page_actions_seen}")
    print(f"find_in_page_actions_seen={telemetry.find_in_page_actions_seen}")
    print(f"unknown_actions_seen={telemetry.unknown_actions_seen}")
    print(f"sources_seen={telemetry.sources_seen}")
    print(f"identity_candidates_rejected={telemetry.identity_candidates_rejected}")
    print(
        "tool_call_limit_exceeded="
        f"{'yes' if telemetry.tool_call_limit_exceeded else 'no'}"
    )


def _resolve_results(results: tuple[object, ...]):
    from website_candidate_matching import (
        BusinessIdentity,
        SourceAttempt,
        SourceAttemptStatus,
        candidate_from_search_result,
        resolve_website_candidates,
    )
    from website_resolution import CandidateSource

    identity = BusinessIdentity(
        name="STATUS стоматологія",
        city="Запоріжжя",
        address="вул. Поштова, 161/36",
    )
    candidates = tuple(candidate_from_search_result(result) for result in results)
    attempt = SourceAttempt(
        CandidateSource.WEB_SEARCH,
        SourceAttemptStatus.COMPLETED,
    )
    return resolve_website_candidates(
        identity,
        candidates,
        (attempt,),
        (CandidateSource.WEB_SEARCH,),
    )


async def _validate() -> int:
    allowed, reason = _environment_gate()
    if not allowed:
        print(reason)
        return 0 if reason.endswith("NO_EXPLICIT_OPT_IN") else 2

    try:
        import config
        from website_candidate_matching import SearchRequest
        from website_resolution import normalize_domain
        from website_search_runtime import (
            BudgetedSearchProvider,
            build_configured_search_provider,
            openai_web_search_telemetry_snapshot,
            search_budget_snapshot,
        )
    except Exception:
        print("final_result=configuration_error")
        return 2

    provider = build_configured_search_provider()
    if not isinstance(provider, BudgetedSearchProvider):
        print("final_result=provider_unavailable")
        return 2

    print("provider=openai")
    print(f"model={config.OPENAI_WEB_SEARCH_MODEL}")
    print(f"reasoning_effort={config.OPENAI_WEB_SEARCH_REASONING_EFFORT}")
    print(f"search_context_size={config.OPENAI_WEB_SEARCH_CONTEXT_SIZE}")
    print(f"external_web_access={str(config.OPENAI_WEB_SEARCH_EXTERNAL_ACCESS).lower()}")
    try:
        results = await provider.search(SearchRequest(
            business_name="STATUS стоматологія",
            city="Запоріжжя",
            address="вул. Поштова, 161/36",
            phone=None,
            max_results=5,
            timeout_seconds=config.OPENAI_WEB_SEARCH_TIMEOUT_SECONDS,
        ))
    except Exception as error:
        budget = search_budget_snapshot(provider)
        telemetry = openai_web_search_telemetry_snapshot(provider)
        if budget is not None:
            print(f"request_budget_used={budget.used_requests}")
            print(f"request_budget_remaining={budget.remaining_requests}")
        if telemetry is not None:
            _print_telemetry(telemetry)
            if telemetry.tool_call_limit_exceeded:
                print("final_result=tool_call_limit_exceeded")
                return 1
        print(f"final_result={_error_category(error)}")
        return 1

    budget = search_budget_snapshot(provider)
    telemetry = openai_web_search_telemetry_snapshot(provider)
    domains = tuple(dict.fromkeys(normalize_domain(result.url) for result in results))
    if budget is not None:
        print(f"request_budget_used={budget.used_requests}")
        print(f"request_budget_remaining={budget.remaining_requests}")
    if telemetry is not None:
        _print_telemetry(telemetry)
        if telemetry.tool_call_limit_exceeded:
            print("final_result=tool_call_limit_exceeded")
            return 1

    resolution = _resolve_results(results)
    resolved_domain = (
        normalize_domain(resolution.resolved_url)
        if resolution.resolved_url is not None
        else ""
    )
    found = EXPECTED_DOMAIN in domains
    wrong_returned = WRONG_SAME_NAME_DOMAIN in domains
    wrong_promoted = resolved_domain == WRONG_SAME_NAME_DOMAIN
    print(f"resolution_status={resolution.status.value}")
    print(f"resolved_domain={resolved_domain}")
    print(f"wrong_same_name_domain_returned={'yes' if wrong_returned else 'no'}")
    print(f"wrong_same_name_domain_promoted={'yes' if wrong_promoted else 'no'}")
    if wrong_promoted:
        print("final_result=unsafe_wrong_business_promoted")
        return 1
    if found:
        print("final_result=success_expected_domain_found")
        return 0
    print("final_result=technical_success_safe_no_verified_match")
    return 0


def main() -> int:
    _configure_utf8_console()
    return asyncio.run(_validate())


if __name__ == "__main__":
    raise SystemExit(main())

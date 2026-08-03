"""One-request, explicitly opted-in Brave Search validation."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--address")
    parser.add_argument("--phone")
    return parser.parse_args()


def _require_explicit_environment() -> None:
    if os.environ.get("ALLOW_LIVE_BRAVE_SEARCH_VALIDATION") != "1":
        raise SystemExit("LIVE_VALIDATION_NOT_RUN_NO_EXPLICIT_OPT_IN")
    if os.environ.get("WEBSITE_SEARCH_PROVIDER", "").strip().casefold() != "brave":
        raise SystemExit("WEBSITE_SEARCH_PROVIDER must be brave")
    if not os.environ.get("BRAVE_SEARCH_API_KEY", "").strip():
        raise SystemExit("BRAVE_SEARCH_API_KEY must be configured")
    try:
        budget = int(os.environ.get("MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK", ""))
    except ValueError:
        raise SystemExit("MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK must be positive") from None
    if budget < 1:
        raise SystemExit("MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK must be positive")


def _normalized_domain(url: str) -> str:
    return (urlsplit(url).hostname or "").casefold().removeprefix("www.")


async def _run(arguments: argparse.Namespace) -> None:
    from website_candidate_matching import SearchRequest
    from website_search_runtime import (
        brave_telemetry_snapshot,
        build_configured_search_provider,
        search_budget_snapshot,
    )

    provider = build_configured_search_provider()
    if provider is None:
        raise SystemExit("configured provider is unavailable")
    results = await provider.search(SearchRequest(
        business_name=arguments.name,
        city=arguments.city,
        address=arguments.address,
        phone=arguments.phone,
    ))
    print(f"results: {len(results)}")
    for result in results:
        title = " ".join(result.title.split())[:120]
        print(f"{result.rank}: {_normalized_domain(result.url)} | {title}")
    print(f"telemetry: {brave_telemetry_snapshot(provider)!r}")
    print(f"budget: {search_budget_snapshot(provider)!r}")


def main() -> None:
    _require_explicit_environment()
    arguments = _arguments()
    asyncio.run(_run(arguments))


if __name__ == "__main__":
    main()

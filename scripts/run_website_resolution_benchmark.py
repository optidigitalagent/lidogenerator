"""Run an explicitly authorized, sequential website-resolution benchmark."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

LIVE_GATE = "ALLOW_LIVE_WEBSITE_RESOLUTION_BENCHMARK"
MAX_CASES = 20
MIN_LIVE_CASES = 12
FATAL_ERROR_CATEGORIES = frozenset({"authentication", "unavailable"})


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, TypeError, ValueError):
                pass


def _environment_gate(case_count: int) -> tuple[bool, str, int | None]:
    if os.getenv(LIVE_GATE) != "1":
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_NO_EXPLICIT_OPT_IN", None
    if os.getenv("WEBSITE_SEARCH_PROVIDER", "").strip().casefold() != "openai":
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_PROVIDER_NOT_OPENAI", None
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_MISSING_API_KEY", None
    try:
        budget = int(os.getenv("MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK", ""))
    except ValueError:
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_INVALID_BUDGET", None
    if case_count > MAX_CASES:
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_TOO_MANY_CASES", budget
    if case_count < MIN_LIVE_CASES:
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_TOO_FEW_CASES", budget
    if budget != case_count:
        return False, "LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_BUDGET_MISMATCH", budget
    return True, "", budget


def _safe_preflight(dataset_path: Path, cases: tuple[object, ...], budget: int) -> None:
    import config
    from website_resolution_benchmark import BenchmarkLabel

    official = sum(case.label is BenchmarkLabel.OFFICIAL_DOMAIN for case in cases)
    no_site = sum(case.label is BenchmarkLabel.NO_OFFICIAL_SITE for case in cases)
    print(f"dataset_path={dataset_path}")
    print(f"total_cases={len(cases)}")
    print(f"official_domain_cases={official}")
    print(f"no_official_site_cases={no_site}")
    print("provider=openai")
    print(f"model={config.OPENAI_WEB_SEARCH_MODEL}")
    print(f"reasoning_effort={config.OPENAI_WEB_SEARCH_REASONING_EFFORT}")
    print(f"context_size={config.OPENAI_WEB_SEARCH_CONTEXT_SIZE}")
    print(f"request_budget={budget}")
    print(f"max_cases={MAX_CASES}")
    print("LIVE_BENCHMARK_AUTHORIZED=yes")


async def _run(cases_path: str | Path, output_dir: str | Path) -> int:
    from website_resolution_benchmark import (
        BenchmarkGateDecision,
        evaluate_benchmark_gate,
        load_benchmark_cases,
        run_benchmark_case,
        summarize_benchmark,
        write_benchmark_outputs,
    )

    dataset_path = Path(cases_path)
    try:
        cases = load_benchmark_cases(dataset_path)
    except Exception:
        print("LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_INVALID_DATASET")
        return 2
    allowed, reason, budget = _environment_gate(len(cases))
    if not allowed:
        print(reason)
        return 0 if reason.endswith("NO_EXPLICIT_OPT_IN") else 2

    try:
        import config
        from website_search_runtime import (
            BudgetedSearchProvider,
            build_configured_search_provider,
        )

        if (
            config.WEBSITE_SEARCH_PROVIDER != "openai"
            or config.MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK != len(cases)
        ):
            raise ValueError("configuration mismatch")
        provider = build_configured_search_provider()
        if not isinstance(provider, BudgetedSearchProvider):
            raise ValueError("provider unavailable")
    except Exception:
        print("LIVE_WEBSITE_RESOLUTION_BENCHMARK_NOT_RUN_CONFIGURATION_ERROR")
        return 2

    _safe_preflight(dataset_path, cases, budget if budget is not None else 0)
    results = []
    fatal = False
    for case in cases:
        result = await run_benchmark_case(case, provider)
        results.append(result)
        if result.error_category in FATAL_ERROR_CATEGORIES:
            fatal = True
            break

    summary = summarize_benchmark(results)
    decision = evaluate_benchmark_gate(summary)
    write_benchmark_outputs(output_dir, results, summary, decision)
    if fatal:
        print("benchmark_stopped=fatal_provider_error")
        return 2
    print(f"gate_decision={decision.value}")
    return 0 if decision is BenchmarkGateDecision.PASS else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="approved benchmark JSON path")
    parser.add_argument("--out", required=True, help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_console()
    arguments = _parser().parse_args(argv)
    return asyncio.run(_run(arguments.cases, arguments.out))


if __name__ == "__main__":
    raise SystemExit(main())

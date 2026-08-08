# Website Resolution Benchmark

This directory contains the schema example for a manually labeled benchmark of the web-search provider and deterministic Website Resolver. The example dataset is a template only and must never be presented as evaluation evidence.

## Ground-truth rules

1. Labels are human-approved ground truth.
2. Synthetic examples are not evidence.
3. Never infer `NO_OFFICIAL_SITE` from a search failure.
4. A real dataset must be manually approved before any live run.
5. Measure a provider-returned domain separately from a resolver-promoted domain.
6. False-positive promotion is the most severe error.
7. Production enablement requires a `PASS` decision plus human review.
8. Do not include private contacts in benchmark outputs.

`OFFICIAL_DOMAIN` requires a normalized ordinary business domain. `NO_OFFICIAL_SITE` requires `expected_domain` to be `null`. There is intentionally no unknown label.

The V1 merge gates are conservative pre-production gates: at least 12 cases, including at least 8 official-domain cases and 4 no-site cases; zero critical false-positive promotions; resolver precision of at least 0.95; resolver recall of at least 0.60; and technical success of at least 0.90.

## Live runner

The runner performs no live call unless every safety gate is present:

```powershell
$env:ALLOW_LIVE_WEBSITE_RESOLUTION_BENCHMARK = "1"
$env:WEBSITE_SEARCH_PROVIDER = "openai"
$env:OPENAI_API_KEY = "..."
$env:MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK = "12" # exactly the case count
python scripts/run_website_resolution_benchmark.py --cases path/to/approved.json --out benchmark/results/run-name
```

Live runs are sequential, use exactly one provider request per case, have no retries, require 12–20 cases, and stop on authentication or provider-unavailability failures. Keep real labeled datasets and generated output out of version control.

"""Synthetic/offline tests for the Instagram Resolver V1 benchmark harness."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agents.openai_instagram_search_provider import (
    _suggested_query_variants,
    build_openai_instagram_search_input,
)
from instagram_candidate_matching import (
    InstagramProviderTimeout,
    InstagramSearchIdentityEvidence,
    InstagramSearchResult,
)
from instagram_resolution_benchmark import (
    InstagramBenchmarkCase,
    InstagramBenchmarkCaseResult,
    InstagramBenchmarkGateDecision,
    InstagramBenchmarkLabel,
    evaluate_benchmark_gate,
    load_benchmark_cases,
    run_benchmark,
    run_benchmark_case,
    summarize_benchmark,
    write_benchmark_outputs,
)


def _case(
    case_id: str = "official_01",
    *,
    label: InstagramBenchmarkLabel = InstagramBenchmarkLabel.OFFICIAL_PROFILE,
    business_name: str = "Amber Kite Atelier",
    city: str = "Exampleton",
    address: str | None = "12 Fixture Street",
    phone: str | None = "+1 202 555 0101",
    website_url: str | None = "https://amber-kite.invalid/",
    expected_username: str | None = "amber_kite_fixture",
) -> InstagramBenchmarkCase:
    return InstagramBenchmarkCase(
        case_id=case_id,
        label=label,
        business_name=business_name,
        city=city,
        address=address,
        phone=phone,
        website_url=website_url,
        expected_username=expected_username,
    )


def _case_record(case_id: str = "official_01", **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "case_id": case_id,
        "label": "official_profile",
        "business_name": "Amber Kite Atelier",
        "city": "Exampleton",
        "address": "12 Fixture Street",
        "phone": "+1 202 555 0101",
        "website_url": "https://amber-kite.invalid/",
        "expected_username": "amber_kite_fixture",
    }
    values.update(overrides)
    return values


def _accepted(username: str = "amber_kite_fixture") -> InstagramSearchResult:
    return InstagramSearchResult(
        f"https://www.instagram.com/{username}/",
        "Amber Kite Atelier — Exampleton",
        "Call +1 202 555 0101",
        1,
        InstagramSearchIdentityEvidence(
            name_matches=True,
            city_matches=True,
            address_matches=True,
            phone_matches=True,
            website_domain_matches=True,
            different_city_detected=False,
            candidate_url_source_bound=True,
        ),
    )


def _uncertain() -> InstagramSearchResult:
    return InstagramSearchResult(
        "https://www.instagram.com/unrelated_fixture/",
        "Unrelated Fixture",
        "No corroborating identity",
        1,
    )


class FakeProvider:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.requests = []

    async def search(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else ()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _result(
    case_id: str = "official_01",
    *,
    label: InstagramBenchmarkLabel = InstagramBenchmarkLabel.OFFICIAL_PROFILE,
    expected_username: str | None = "fixture_official",
    resolution_status: str = "found_official",
    resolved_username: str | None = "fixture_official",
    correct_promotion: bool = True,
    critical_false_promotion: bool = False,
    safe_nonpromotion: bool = False,
    technical_success: bool = True,
    error_category: str | None = None,
    requests_used: int = 1,
    tool_calls_seen: int = 1,
    search_actions_seen: int = 1,
    open_page_actions_seen: int = 0,
    find_in_page_actions_seen: int = 0,
    candidates_returned: int = 1,
    identity_candidates_rejected: int = 0,
    sources_seen: int = 1,
    structured_candidates_seen: int = 0,
    identity_prefilter_rejected: int = 0,
    direct_profile_sources_seen: int = 0,
    invalid_profile_candidates_discarded: int = 0,
    source_unbound_candidates_discarded: int = 0,
    source_bound_candidates_returned: int = 0,
) -> InstagramBenchmarkCaseResult:
    return InstagramBenchmarkCaseResult(**locals())


def _safe_negative(case_id: str, label: InstagramBenchmarkLabel) -> InstagramBenchmarkCaseResult:
    return _result(
        case_id,
        label=label,
        expected_username=None,
        resolution_status="not_found",
        resolved_username=None,
        correct_promotion=False,
        safe_nonpromotion=True,
        candidates_returned=0,
        sources_seen=0,
    )


def _passing_results() -> tuple[InstagramBenchmarkCaseResult, ...]:
    results = [_result(f"official_{index}") for index in range(8)]
    results.extend(
        _safe_negative(f"negative_{index}", InstagramBenchmarkLabel.NO_OFFICIAL_PROFILE)
        for index in range(2)
    )
    results.extend(
        _safe_negative(f"trap_{index}", InstagramBenchmarkLabel.DO_NOT_PROMOTE)
        for index in range(2)
    )
    return tuple(results)


class DatasetContractTests(unittest.TestCase):
    def _load(self, payload):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            if isinstance(payload, str):
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
            return load_benchmark_cases(path)

    def test_loads_json_list_and_normalizes_expected_username(self):
        cases = self._load([_case_record(expected_username="@AMBER_KITE_FIXTURE")])
        self.assertEqual(cases[0].expected_username, "amber_kite_fixture")
        self.assertEqual(cases[0].phone, "12025550101")

    def test_rejects_non_list_invalid_json_and_exact_key_violations(self):
        for payload in ("{not-json", {"cases": []}, []):
            if payload == []:
                self.assertEqual(self._load(payload), ())
            else:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    self._load(payload)
        missing = _case_record()
        missing.pop("city")
        extra = {**_case_record(), "evidence": "forbidden"}
        for record in (missing, extra):
            with self.subTest(record=record), self.assertRaises(ValueError):
                self._load([record])

    def test_label_invariants_and_strong_identity_requirement(self):
        with self.assertRaises(ValueError):
            _case(expected_username=None)
        for label in (
            InstagramBenchmarkLabel.NO_OFFICIAL_PROFILE,
            InstagramBenchmarkLabel.DO_NOT_PROMOTE,
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                _case(label=label, expected_username="forbidden")
        with self.assertRaises(ValueError):
            _case(address=None, phone=None, website_url=None)

    def test_case_is_frozen(self):
        case = _case()
        with self.assertRaises(FrozenInstanceError):
            case.city = "Changed"  # type: ignore[misc]

    def test_duplicate_case_ids_and_exact_identities_rejected(self):
        duplicate_id = [_case_record(), _case_record()]
        duplicate_identity = [
            _case_record(),
            _case_record("official_02", expected_username="other_fixture"),
        ]
        for payload in (duplicate_id, duplicate_identity):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self._load(payload)


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_correct_official_promotion(self):
        provider = FakeProvider(((_accepted(),),))
        result = await run_benchmark_case(_case(), provider)
        self.assertEqual(len(provider.requests), 1)
        self.assertTrue(result.correct_promotion)
        self.assertFalse(result.critical_false_promotion)
        self.assertTrue(result.technical_success)

    async def test_safe_official_not_found_and_uncertain(self):
        for outcome, status in (((), "not_found"), ((_uncertain(),), "uncertain")):
            with self.subTest(status=status):
                result = await run_benchmark_case(_case(), FakeProvider((outcome,)))
                self.assertEqual(result.resolution_status, status)
                self.assertTrue(result.safe_nonpromotion)
                self.assertFalse(result.correct_promotion)

    async def test_wrong_official_username_is_critical(self):
        result = await run_benchmark_case(
            _case(), FakeProvider(((_accepted("amber_kite_wrong"),),))
        )
        self.assertTrue(result.critical_false_promotion)
        self.assertFalse(result.correct_promotion)

    async def test_negative_not_found_and_uncertain_are_safe(self):
        for label in (
            InstagramBenchmarkLabel.NO_OFFICIAL_PROFILE,
            InstagramBenchmarkLabel.DO_NOT_PROMOTE,
        ):
            case = _case(label=label, expected_username=None)
            for outcome in ((), (_uncertain(),)):
                with self.subTest(label=label, outcome=outcome):
                    result = await run_benchmark_case(case, FakeProvider((outcome,)))
                    self.assertTrue(result.safe_nonpromotion)
                    self.assertFalse(result.critical_false_promotion)

    async def test_negative_and_do_not_promote_found_are_critical(self):
        for label in (
            InstagramBenchmarkLabel.NO_OFFICIAL_PROFILE,
            InstagramBenchmarkLabel.DO_NOT_PROMOTE,
        ):
            case = _case(label=label, expected_username=None)
            result = await run_benchmark_case(case, FakeProvider(((_accepted(),),)))
            self.assertTrue(result.critical_false_promotion)
            self.assertFalse(result.safe_nonpromotion)

    async def test_resolution_error_is_technical_failure_and_not_retried(self):
        provider = FakeProvider((InstagramProviderTimeout("sensitive raw detail"),))
        result = await run_benchmark_case(_case(), provider)
        self.assertEqual(len(provider.requests), 1)
        self.assertFalse(result.technical_success)
        self.assertFalse(result.safe_nonpromotion)
        self.assertEqual(result.error_category, "timeout")
        self.assertNotIn("sensitive raw detail", repr(result))

    async def test_safe_provider_telemetry_persists_per_case(self):
        class TelemetryProvider(FakeProvider):
            def __init__(self):
                super().__init__(((),))
                self.values = {
                    "requests_started": 0,
                    "tool_calls_seen": 0,
                    "search_actions_seen": 0,
                    "open_page_actions_seen": 0,
                    "find_in_page_actions_seen": 0,
                    "candidates_returned": 0,
                    "identity_candidates_rejected": 0,
                    "sources_seen": 0,
                    "structured_candidates_seen": 0,
                    "identity_prefilter_rejected": 0,
                    "direct_profile_sources_seen": 0,
                    "invalid_profile_candidates_discarded": 0,
                    "source_unbound_candidates_discarded": 0,
                    "source_bound_candidates_returned": 0,
                    "tool_call_limit_exceeded": False,
                }

            def telemetry(self):
                return SimpleNamespace(**self.values)

            async def search(self, request):
                self.values.update({
                    "requests_started": 1,
                    "tool_calls_seen": 1,
                    "search_actions_seen": 1,
                    "sources_seen": 3,
                    "structured_candidates_seen": 4,
                    "identity_candidates_rejected": 1,
                    "identity_prefilter_rejected": 1,
                    "direct_profile_sources_seen": 1,
                    "invalid_profile_candidates_discarded": 1,
                    "source_unbound_candidates_discarded": 2,
                    "source_bound_candidates_returned": 0,
                })
                return await super().search(request)

        result = await run_benchmark_case(_case(), TelemetryProvider())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            metrics = summarize_benchmark((result,))
            write_benchmark_outputs(
                output,
                (result,),
                metrics,
                InstagramBenchmarkGateDecision.FAIL_DATASET_SHAPE,
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))["results"][0]
        self.assertEqual(persisted["structured_candidates_seen"], 4)
        self.assertEqual(persisted["identity_prefilter_rejected"], 1)
        self.assertEqual(persisted["direct_profile_sources_seen"], 1)
        self.assertEqual(persisted["invalid_profile_candidates_discarded"], 1)
        self.assertEqual(persisted["source_unbound_candidates_discarded"], 2)
        self.assertEqual(persisted["source_bound_candidates_returned"], 0)


class MetricsTests(unittest.TestCase):
    def test_recall_precision_specificity_and_technical_rate(self):
        results = (
            _result("official_correct"),
            _result(
                "official_miss",
                resolution_status="uncertain",
                resolved_username=None,
                correct_promotion=False,
                safe_nonpromotion=True,
                candidates_returned=0,
            ),
            _result(
                "official_wrong",
                resolved_username="wrong_fixture",
                correct_promotion=False,
                critical_false_promotion=True,
            ),
            _safe_negative("negative_safe", InstagramBenchmarkLabel.NO_OFFICIAL_PROFILE),
            _result(
                "negative_false",
                label=InstagramBenchmarkLabel.DO_NOT_PROMOTE,
                expected_username=None,
                resolved_username="trap_fixture",
                correct_promotion=False,
                critical_false_promotion=True,
            ),
            _result(
                "negative_error",
                label=InstagramBenchmarkLabel.NO_OFFICIAL_PROFILE,
                expected_username=None,
                resolution_status="resolution_error",
                resolved_username=None,
                correct_promotion=False,
                technical_success=False,
                error_category="timeout",
                candidates_returned=0,
            ),
        )
        metrics = summarize_benchmark(results)
        self.assertEqual(metrics.official_recall, 1 / 3)
        self.assertEqual(metrics.promotion_precision, 1 / 3)
        self.assertEqual(metrics.negative_specificity, 1 / 3)
        self.assertEqual(metrics.technical_success_rate, 5 / 6)
        self.assertEqual(metrics.wrong_official_promotions, 1)
        self.assertEqual(metrics.critical_false_promotions, 1)

    def test_zero_denominators_are_zero(self):
        metrics = summarize_benchmark(())
        self.assertEqual(metrics.official_recall, 0.0)
        self.assertEqual(metrics.promotion_precision, 0.0)
        self.assertEqual(metrics.negative_specificity, 0.0)
        self.assertEqual(metrics.technical_success_rate, 0.0)


class GateTests(unittest.TestCase):
    def _decision(self, results):
        return evaluate_benchmark_gate(summarize_benchmark(results), results)

    def test_passes_good_synthetic_dataset(self):
        self.assertIs(self._decision(_passing_results()), InstagramBenchmarkGateDecision.PASS)

    def test_fails_false_promotion_and_wrong_official_username(self):
        base = list(_passing_results())
        negative_false = replace(
            base[8],
            resolution_status="found_official",
            resolved_username="false_fixture",
            critical_false_promotion=True,
            safe_nonpromotion=False,
            candidates_returned=1,
        )
        wrong_official = replace(
            base[0],
            resolved_username="wrong_fixture",
            correct_promotion=False,
            critical_false_promotion=True,
        )
        for replacement, index in ((negative_false, 8), (wrong_official, 0)):
            results = list(base)
            results[index] = replacement
            self.assertIs(
                self._decision(results),
                InstagramBenchmarkGateDecision.FAIL_FALSE_PROMOTION,
            )

    def test_fails_recall_below_half(self):
        results = list(_passing_results())
        for index in range(5):
            results[index] = replace(
                results[index],
                resolution_status="not_found",
                resolved_username=None,
                correct_promotion=False,
                safe_nonpromotion=True,
                candidates_returned=0,
            )
        self.assertIs(self._decision(results), InstagramBenchmarkGateDecision.FAIL_RECALL)

    def test_zero_promotions_fail_recall(self):
        results = list(_passing_results())
        for index in range(8):
            results[index] = replace(
                results[index],
                resolution_status="not_found",
                resolved_username=None,
                correct_promotion=False,
                safe_nonpromotion=True,
                candidates_returned=0,
            )
        self.assertIs(
            self._decision(results),
            InstagramBenchmarkGateDecision.FAIL_RECALL,
        )

    def test_specificity_failure_precedes_recall(self):
        results = list(_passing_results())
        for index in range(8):
            results[index] = replace(
                results[index],
                resolution_status="not_found",
                resolved_username=None,
                correct_promotion=False,
                safe_nonpromotion=True,
                candidates_returned=0,
            )
        results[8] = replace(
            results[8],
            resolution_status="resolution_error",
            safe_nonpromotion=False,
            technical_success=False,
            error_category="timeout",
        )
        self.assertIs(
            self._decision(results),
            InstagramBenchmarkGateDecision.FAIL_SPECIFICITY,
        )

    def test_recall_pass_with_low_precision_fails_precision(self):
        results = _passing_results()
        metrics = replace(summarize_benchmark(results), promotion_precision=0.90)
        self.assertIs(
            evaluate_benchmark_gate(metrics, results),
            InstagramBenchmarkGateDecision.FAIL_PRECISION,
        )

    def test_fails_technical_below_point_nine(self):
        results = list(_passing_results())
        for index in range(2):
            results[index] = replace(
                results[index],
                resolution_status="resolution_error",
                resolved_username=None,
                correct_promotion=False,
                technical_success=False,
                error_category="timeout",
            )
        self.assertIs(self._decision(results), InstagramBenchmarkGateDecision.FAIL_TECHNICAL)

    def test_fails_insufficient_dataset_shape(self):
        self.assertIs(
            self._decision(_passing_results()[:-1]),
            InstagramBenchmarkGateDecision.FAIL_DATASET_SHAPE,
        )

    def test_fails_each_tool_safety_constraint(self):
        base = _passing_results()
        for changes in (
            {"requests_used": 2},
            {"tool_calls_seen": 2},
            {"open_page_actions_seen": 1},
            {"find_in_page_actions_seen": 1},
        ):
            results = (replace(base[0], **changes),) + base[1:]
            with self.subTest(changes=changes):
                self.assertIs(
                    self._decision(results),
                    InstagramBenchmarkGateDecision.FAIL_TOOL_SAFETY,
                )


class LeakageAndShaTests(unittest.IsolatedAsyncioTestCase):
    async def test_expected_username_and_case_id_never_enter_request_prompt_or_queries(self):
        expected = "evaluator_only_secret_username"
        case_id = "evaluator_only_secret_case"
        provider = FakeProvider(((),))
        await run_benchmark_case(
            _case(case_id=case_id, expected_username=expected),
            provider,
        )
        request = provider.requests[0]
        serialized = json.dumps(asdict(request), sort_keys=True)
        prompt = build_openai_instagram_search_input(request)
        queries = json.dumps(_suggested_query_variants(request))
        for forbidden in (expected, case_id):
            self.assertNotIn(forbidden, serialized)
            self.assertNotIn(forbidden, prompt)
            self.assertNotIn(forbidden, queries)
        self.assertNotIn("expected_username", request.__dataclass_fields__)
        self.assertNotIn("case_id", request.__dataclass_fields__)

    async def test_sha_mismatch_prevents_provider_construction_and_calls(self):
        constructed = 0
        provider = FakeProvider()

        def factory():
            nonlocal constructed
            constructed += 1
            return provider

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.json"
            output = Path(temporary) / "report.json"
            dataset.write_text(json.dumps([_case_record()]), encoding="utf-8")
            with self.assertRaises(ValueError):
                await run_benchmark(
                    dataset,
                    output,
                    factory,
                    expected_dataset_sha256="0" * 64,
                )
        self.assertEqual(constructed, 0)
        self.assertEqual(provider.requests, [])

    async def test_exact_sha_allows_construction_after_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.json"
            output = Path(temporary) / "report.json"
            records = []
            outcomes = []
            for index in range(8):
                records.append(
                    _case_record(
                        f"official_{index}",
                        business_name=f"Fixture Brand {index}",
                        address=f"{index} Fixture Street",
                        phone=f"+1 202 555 {1000 + index}",
                        website_url=f"https://fixture-{index}.invalid/",
                        expected_username=f"fixture_brand_{index}",
                    )
                )
                outcomes.append(())
            for index in range(4):
                label = "no_official_profile" if index < 2 else "do_not_promote"
                records.append(
                    _case_record(
                        f"negative_{index}",
                        label=label,
                        business_name=f"Negative Fixture {index}",
                        address=f"{index + 20} Fixture Street",
                        phone=f"+1 202 556 {1000 + index}",
                        website_url=f"https://negative-{index}.invalid/",
                        expected_username=None,
                    )
                )
                outcomes.append(())
            raw = json.dumps(records).encode()
            dataset.write_bytes(raw)
            provider = FakeProvider(tuple(outcomes))
            decision = await run_benchmark(
                dataset,
                output,
                lambda: provider,
                expected_dataset_sha256=hashlib.sha256(raw).hexdigest(),
            )
            self.assertIs(decision, InstagramBenchmarkGateDecision.FAIL_RECALL)
            self.assertEqual(len(provider.requests), 12)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".txt").exists())


class SecurityContractTests(unittest.TestCase):
    def test_result_contract_is_exactly_allowlisted(self):
        self.assertEqual(
            tuple(InstagramBenchmarkCaseResult.__dataclass_fields__),
            (
                "case_id",
                "label",
                "expected_username",
                "resolution_status",
                "resolved_username",
                "correct_promotion",
                "critical_false_promotion",
                "safe_nonpromotion",
                "technical_success",
                "error_category",
                "requests_used",
                "tool_calls_seen",
                "search_actions_seen",
                "open_page_actions_seen",
                "find_in_page_actions_seen",
                "candidates_returned",
                "identity_candidates_rejected",
                "sources_seen",
                "structured_candidates_seen",
                "identity_prefilter_rejected",
                "direct_profile_sources_seen",
                "invalid_profile_candidates_discarded",
                "source_unbound_candidates_discarded",
                "source_bound_candidates_returned",
            ),
        )
        source = Path(__import__("instagram_resolution_benchmark").__file__).read_text(
            encoding="utf-8"
        )
        for forbidden in ("raw source URLs", "request IDs", "token usage", "API keys"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()

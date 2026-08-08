"""Synthetic tests for pure website-candidate identity matching."""

import ast
import dataclasses
import math
from pathlib import Path
import unittest

import website_candidate_matching
from website_candidate_matching import (
    BusinessIdentity,
    MatchSignal,
    ProviderAuthError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    SearchProvider,
    SearchProviderError,
    SearchIdentityEvidence,
    SearchRequest,
    SearchResult,
    SourceAttempt,
    SourceAttemptStatus,
    WebsiteCandidate,
    assess_website_candidate,
    candidate_from_search_result,
    collect_web_search_candidates,
    identity_tokens,
    normalize_identity_text,
    normalize_instagram_username,
    normalize_phone_number,
    resolve_website_candidates,
)
from website_resolution import CandidateKind, CandidateSource, ResolutionStatus


def _identity(**overrides) -> BusinessIdentity:
    values = {
        "name": "Lido Beauty Studio",
        "city": "Kyiv",
        "address": "Khreshchatyk Street 10",
        "phone": "+380 67 123 45 67",
        "instagram_url": "https://www.instagram.com/lido.beauty/",
    }
    values.update(overrides)
    return BusinessIdentity(**values)


def _candidate(**overrides) -> WebsiteCandidate:
    values = {
        "source": CandidateSource.WEB_SEARCH,
        "url": "https://lido-beauty.example/",
    }
    values.update(overrides)
    return WebsiteCandidate(**values)


def _source_evidence(**overrides) -> SearchIdentityEvidence:
    values = {
        "name_matches": True,
        "city_matches": True,
        "address_matches": True,
        "phone_matches": False,
        "different_city_detected": False,
        "candidate_url_source_bound": True,
    }
    values.update(overrides)
    return SearchIdentityEvidence(**values)


def _attempt(
    source: CandidateSource = CandidateSource.WEB_SEARCH,
    status: SourceAttemptStatus = SourceAttemptStatus.COMPLETED,
    detail=None,
) -> SourceAttempt:
    return SourceAttempt(source, status, detail)


class FakeProvider:
    def __init__(self, result=(), error=None):
        self.result = result
        self.error = error
        self.calls = 0
        self.requests = []

    async def search(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class BusinessIdentityTests(unittest.TestCase):
    def test_normalizes_all_supplied_fields(self) -> None:
        identity = BusinessIdentity(
            "  Lido   Beauty  ",
            "  Kyiv  ",
            " Main   Street  1 ",
            "+38 (067) 123-45-67",
            "HTTPS://WWW.INSTAGRAM.COM/Lido.Beauty/?utm_source=x",
        )
        self.assertEqual(identity.name, "Lido Beauty")
        self.assertEqual(identity.city, "Kyiv")
        self.assertEqual(identity.address, "Main Street 1")
        self.assertEqual(identity.phone, "380671234567")
        self.assertEqual(
            identity.instagram_url,
            "https://www.instagram.com/Lido.Beauty/",
        )

    def test_optional_fields_may_be_none(self) -> None:
        identity = BusinessIdentity("Name", "City")
        self.assertIsNone(identity.address)
        self.assertIsNone(identity.phone)
        self.assertIsNone(identity.instagram_url)

    def test_rejects_non_instagram_and_non_profile_urls(self) -> None:
        for url in (
            "https://example.com/lido",
            "https://instagram.com/",
            "https://instagram.com/p/abc",
            "https://instagram.com/reels/",
            "https://instagram.com/stories/lido",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                BusinessIdentity("Name", "City", instagram_url=url)

    def test_rejects_wrong_types_and_empty_required_or_optional_fields(self) -> None:
        for field, value in (
            ("name", " "),
            ("city", ""),
            ("address", " "),
            ("name", 1),
            ("city", True),
            ("address", []),
            ("phone", 1234567),
            ("instagram_url", False),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(
                (TypeError, ValueError)
            ):
                values = {"name": "Name", "city": "City", field: value}
                BusinessIdentity(**values)

    def test_is_frozen(self) -> None:
        identity = _identity()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.city = "Lviv"


class IdentityTextTests(unittest.TestCase):
    def test_cyrillic_latin_punctuation_and_casefold(self) -> None:
        self.assertEqual(
            normalize_identity_text("  САЛОН «Лідо» — Beauty_Studio! "),
            "салон лідо beauty studio",
        )

    def test_nfkc_normalizes_compatibility_characters(self) -> None:
        self.assertEqual(normalize_identity_text("ＡＢＣ №１２"), "abc no12")

    def test_tokens_preserve_first_order_and_deduplicate(self) -> None:
        self.assertEqual(
            identity_tokens("Lido 7 Lido Studio 12 Studio"),
            ("lido", "studio", "12"),
        )

    def test_does_not_transliterate(self) -> None:
        self.assertEqual(identity_tokens("Лідо"), ("лідо",))
        self.assertNotEqual(normalize_identity_text("Лідо"), "lido")

    def test_rejects_wrong_type_and_empty_normalized_value(self) -> None:
        for value in (None, 1, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_identity_text(value)
        for value in ("", " -- !!! "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_identity_text(value)


class PhoneTests(unittest.TestCase):
    def test_normalizes_formatted_international_and_local_numbers(self) -> None:
        self.assertEqual(normalize_phone_number("+380 (67) 123-45-67"), "380671234567")
        self.assertEqual(normalize_phone_number("0 67 123 45 67"), "0671234567")

    def test_rejects_invalid_length_type_and_characters(self) -> None:
        for value in (1234567, True, None):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_phone_number(value)
        for value in ("123456", "1" * 16, "12+34567", "123ABC4567"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_phone_number(value)
        self.assertEqual(normalize_phone_number("067/123-45-67"), "0671234567")

    def test_ukrainian_local_and_international_are_equivalent(self) -> None:
        evidence = assess_website_candidate(
            _identity(phone="+380671234567"),
            _candidate(contact_phones=("0671234567",)),
        )
        self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertIn(MatchSignal.PHONE_EXACT.value, evidence.matched_signals)

    def test_extracts_phone_from_title_and_snippet(self) -> None:
        for field in ("title", "snippet"):
            with self.subTest(field=field):
                evidence = assess_website_candidate(
                    _identity(),
                    _candidate(**{field: "Call us: +38 (067) 123-45-67"}),
                )
                self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)


class InstagramUsernameTests(unittest.TestCase):
    def test_accepts_raw_at_raw_and_profile_url(self) -> None:
        cases = (
            ("Lido.Beauty", "lido.beauty"),
            ("@LIDO_BEAUTY", "lido_beauty"),
            ("https://www.instagram.com/Lido.Beauty/?igshid=1", "lido.beauty"),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_instagram_username(value), expected)

    def test_rejects_reserved_paths_invalid_hosts_and_invalid_characters(self) -> None:
        for value in (
            "",
            "@",
            "@@lido",
            "bad-name",
            "a" * 31,
            "https://example.com/lido",
            "https://instagram.com/direct/inbox",
            "https://instagram.com/tv/abc",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_instagram_username(value)

    def test_rejects_wrong_types(self) -> None:
        for value in (None, 1, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_instagram_username(value)


class SearchContractTests(unittest.TestCase):
    def test_search_request_normalizes_and_is_frozen(self) -> None:
        request = SearchRequest(
            " Lido   Beauty ",
            " Kyiv ",
            " Main   Street ",
            "+380 (67) 123-45-67",
            10,
            30,
        )
        self.assertEqual(request.business_name, "Lido Beauty")
        self.assertEqual(request.phone, "380671234567")
        self.assertIs(type(request.timeout_seconds), float)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.max_results = 2

    def test_search_request_rejects_invalid_limits_and_timeout(self) -> None:
        for value, error in ((True, TypeError), (0, ValueError), (11, ValueError)):
            with self.subTest(max_results=value), self.assertRaises(error):
                SearchRequest("Name", "City", max_results=value)
        for value, error in (
            (True, TypeError),
            ("1", TypeError),
            (0, ValueError),
            (31, ValueError),
            (math.inf, ValueError),
            (math.nan, ValueError),
        ):
            with self.subTest(timeout=value), self.assertRaises(error):
                SearchRequest("Name", "City", timeout_seconds=value)

    def test_search_request_rejects_bad_strings(self) -> None:
        for kwargs in (
            {"business_name": "", "city": "City"},
            {"business_name": "Name", "city": " "},
            {"business_name": "Name", "city": "City", "address": " "},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SearchRequest(**kwargs)

    def test_search_result_normalizes_all_fields_and_is_frozen(self) -> None:
        result = SearchResult(
            "HTTPS://EXAMPLE.COM:443/?utm_source=x",
            "  Lido   Beauty ",
            "  ",
            1,
        )
        self.assertEqual(result.url, "https://example.com/")
        self.assertEqual(result.title, "Lido Beauty")
        self.assertEqual(result.snippet, "")
        self.assertIsNone(result.identity_evidence)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.rank = 2

    def test_search_identity_evidence_is_strict_and_frozen(self) -> None:
        evidence = _source_evidence()
        self.assertTrue(evidence.name_matches)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            evidence.name_matches = False
        for field_name in SearchIdentityEvidence.__dataclass_fields__:
            for invalid in (1, None, "true"):
                with self.subTest(field=field_name, invalid=invalid), self.assertRaises(
                    TypeError
                ):
                    _source_evidence(**{field_name: invalid})

    def test_search_result_accepts_only_typed_identity_evidence(self) -> None:
        evidence = _source_evidence()
        result = SearchResult("https://example.com", "", "", 1, evidence)
        self.assertIs(result.identity_evidence, evidence)
        with self.assertRaises(TypeError):
            SearchResult("https://example.com", "", "", 1, object())

    def test_search_result_rejects_invalid_fields(self) -> None:
        with self.assertRaises(ValueError):
            SearchResult("/relative", "", "", 1)
        for field in ("title", "snippet"):
            values = {"url": "https://example.com", "title": "", "snippet": "", "rank": 1}
            values[field] = None
            with self.subTest(field=field), self.assertRaises(TypeError):
                SearchResult(**values)
        for rank, error in ((True, TypeError), (1.0, TypeError), (0, ValueError)):
            with self.subTest(rank=rank), self.assertRaises(error):
                SearchResult("https://example.com", "", "", rank)

    def test_fake_provider_satisfies_runtime_protocol(self) -> None:
        self.assertIsInstance(FakeProvider(), SearchProvider)

    def test_provider_exception_inheritance_is_exact(self) -> None:
        for exception in (
            ProviderUnavailable,
            ProviderTimeout,
            ProviderAuthError,
            ProviderRateLimited,
        ):
            with self.subTest(exception=exception):
                self.assertEqual(exception.__bases__, (SearchProviderError,))
        self.assertEqual(SearchProviderError.__bases__, (RuntimeError,))

    def test_search_result_conversion(self) -> None:
        evidence = _source_evidence()
        result = SearchResult("https://example.com", "Title", "Snippet", 3, evidence)
        candidate = candidate_from_search_result(result)
        self.assertIs(candidate.source, CandidateSource.WEB_SEARCH)
        self.assertEqual((candidate.url, candidate.title, candidate.snippet),
                         (result.url, result.title, result.snippet))
        self.assertIsNone(candidate.final_url)
        self.assertIs(candidate.identity_evidence, evidence)


class SourceAttemptTests(unittest.TestCase):
    def test_all_statuses_and_detail_invariants(self) -> None:
        self.assertIsNone(_attempt().detail)
        for status in SourceAttemptStatus:
            if status is SourceAttemptStatus.COMPLETED:
                continue
            attempt = _attempt(status=status, detail="  safe   detail ")
            self.assertEqual(attempt.detail, "safe detail")

    def test_completed_rejects_detail_and_other_statuses_require_it(self) -> None:
        with self.assertRaises(ValueError):
            _attempt(detail="not allowed")
        for status in SourceAttemptStatus:
            if status is not SourceAttemptStatus.COMPLETED:
                with self.subTest(status=status), self.assertRaises(ValueError):
                    _attempt(status=status)

    def test_rejects_invalid_enums_and_detail_types(self) -> None:
        with self.assertRaises(TypeError):
            SourceAttempt("maps", SourceAttemptStatus.COMPLETED)
        with self.assertRaises(TypeError):
            SourceAttempt(CandidateSource.MAPS, "completed")
        with self.assertRaises(TypeError):
            SourceAttempt(CandidateSource.MAPS, SourceAttemptStatus.ERROR, 1)

    def test_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _attempt().status = SourceAttemptStatus.ERROR


class WebsiteCandidateTests(unittest.TestCase):
    def test_normalizes_valid_construction(self) -> None:
        candidate = WebsiteCandidate(
            CandidateSource.MAPS,
            "HTTPS://WWW.EXAMPLE.COM:443/?utm_source=x",
            "  Lido   Beauty ",
            "  Call   today ",
            "https://shop.example.com/path#fragment",
            " КИЇВ ",
            ("+380 (67) 123-45-67",),
            (" Main   Street 1 ",),
            ("@Lido.Beauty",),
        )
        self.assertEqual(candidate.url, "https://www.example.com/")
        self.assertEqual(candidate.final_url, "https://shop.example.com/path")
        self.assertEqual(candidate.city, "київ")
        self.assertEqual(candidate.contact_phones, ("380671234567",))
        self.assertEqual(candidate.contact_addresses, ("Main Street 1",))
        self.assertEqual(candidate.instagram_usernames, ("lido.beauty",))

    def test_rejects_lists_for_tuple_fields(self) -> None:
        for field in ("contact_phones", "contact_addresses", "instagram_usernames"):
            with self.subTest(field=field), self.assertRaises(TypeError):
                _candidate(**{field: []})

    def test_rejects_duplicate_equivalent_phones(self) -> None:
        with self.assertRaises(ValueError):
            _candidate(contact_phones=("+380671234567", "0671234567"))

    def test_rejects_duplicate_normalized_addresses_and_usernames(self) -> None:
        with self.assertRaises(ValueError):
            _candidate(contact_addresses=("Main Street, 1", "main street 1"))
        with self.assertRaises(ValueError):
            _candidate(instagram_usernames=("@LIDO", "lido"))

    def test_rejects_invalid_source_urls_city_and_items(self) -> None:
        for overrides in (
            {"source": "maps"},
            {"url": "/relative"},
            {"final_url": "file:///tmp/a"},
            {"city": "---"},
            {"contact_phones": ("123",)},
            {"contact_addresses": (" ",)},
            {"instagram_usernames": ("bad-name",)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                _candidate(**overrides)

    def test_is_frozen(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _candidate().url = "https://other.example/"

    def test_identity_evidence_defaults_none_and_requires_exact_type(self) -> None:
        self.assertIsNone(_candidate().identity_evidence)
        evidence = _source_evidence()
        self.assertIs(_candidate(identity_evidence=evidence).identity_evidence, evidence)
        with self.assertRaises(TypeError):
            _candidate(identity_evidence={"name_matches": True})


class ProviderCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_orders_by_rank_stably_deduplicates_and_calls_once(self) -> None:
        request = SearchRequest("Lido", "Kyiv")
        first_duplicate = SearchResult("https://EXAMPLE.com/?utm_source=x", "First", "", 2)
        results = (
            first_duplicate,
            SearchResult("https://second.example", "Second", "", 1),
            SearchResult("https://example.com/", "Duplicate", "", 1),
            SearchResult("https://third.example", "Third", "", 2),
        )
        provider = FakeProvider(results)
        candidates, attempt = await collect_web_search_candidates(provider, request)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.requests, [request])
        self.assertEqual(
            tuple(candidate.url for candidate in candidates),
            ("https://second.example/", "https://example.com/", "https://third.example/"),
        )
        self.assertEqual(candidates[1].title, "Duplicate")
        self.assertIs(attempt.status, SourceAttemptStatus.COMPLETED)

    async def test_requires_tuple_and_search_result_items(self) -> None:
        request = SearchRequest("Name", "City")
        with self.assertRaises(TypeError):
            await collect_web_search_candidates(FakeProvider([]), request)
        with self.assertRaises(TypeError):
            await collect_web_search_candidates(FakeProvider(("bad",)), request)

    async def test_maps_all_typed_failures(self) -> None:
        cases = (
            (ProviderUnavailable(" missing   provider "), SourceAttemptStatus.UNAVAILABLE, "missing provider"),
            (ProviderAuthError("auth"), SourceAttemptStatus.UNAVAILABLE, "auth"),
            (ProviderTimeout("slow"), SourceAttemptStatus.TIMEOUT, "slow"),
            (ProviderRateLimited("later"), SourceAttemptStatus.RATE_LIMITED, "later"),
            (SearchProviderError("broken"), SourceAttemptStatus.ERROR, "broken"),
        )
        for error, expected_status, expected_detail in cases:
            with self.subTest(error=error):
                provider = FakeProvider(error=error)
                candidates, attempt = await collect_web_search_candidates(
                    provider,
                    SearchRequest("Name", "City"),
                )
                self.assertEqual(provider.calls, 1)
                self.assertEqual(candidates, ())
                self.assertIs(attempt.status, expected_status)
                self.assertEqual(attempt.detail, expected_detail)

    async def test_empty_error_text_uses_class_name(self) -> None:
        _, attempt = await collect_web_search_candidates(
            FakeProvider(error=ProviderTimeout()),
            SearchRequest("Name", "City"),
        )
        self.assertEqual(attempt.detail, "ProviderTimeout")

    async def test_unexpected_exception_propagates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "programming bug"):
            await collect_web_search_candidates(
                FakeProvider(error=RuntimeError("programming bug")),
                SearchRequest("Name", "City"),
            )


class CandidateAssessmentTests(unittest.TestCase):
    def test_exact_name_and_source_address_corroboration_are_official(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity(
                "Business Dental",
                "City",
                "Address Street 10",
                "+380671234567",
            ),
            _candidate(
                url="https://clinic-one.example/",
                title="Business Dental",
                identity_evidence=_source_evidence(),
            ),
        )
        self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertEqual(
            evidence.matched_signals,
            (
                MatchSignal.NAME_EXACT.value,
                MatchSignal.SOURCE_ADDRESS_CORROBORATION.value,
            ),
        )
        self.assertEqual(evidence.confidence, 0.65)

    def test_exact_name_and_source_phone_corroboration_are_official(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Business Dental", "City", phone="+380671234567"),
            _candidate(
                url="https://clinic-two.example/",
                title="Business Dental",
                identity_evidence=_source_evidence(
                    address_matches=False,
                    phone_matches=True,
                )
            ),
        )
        self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertEqual(
            evidence.matched_signals,
            (
                MatchSignal.NAME_EXACT.value,
                MatchSignal.SOURCE_PHONE_CORROBORATION.value,
            ),
        )
        self.assertEqual(evidence.confidence, 0.65)

    def test_token_overlap_and_source_corroboration_stay_below_threshold(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Alpha Beta Gamma Delta", "City", "Main Street 10"),
            _candidate(
                url="https://unrelated.example/",
                title="Alpha Beta Gamma",
                identity_evidence=_source_evidence(),
            ),
        )
        self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
        self.assertEqual(evidence.rejected_reason, "insufficient_identity_evidence")
        self.assertEqual(evidence.confidence, 0.55)

    def test_source_corroboration_cannot_replace_deterministic_name(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Business Dental", "City", "Address Street 10"),
            _candidate(
                url="https://unrelated.example/",
                title="Official clinic website",
                identity_evidence=_source_evidence(),
            ),
        )
        self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
        self.assertEqual(evidence.rejected_reason, "insufficient_identity_evidence")
        self.assertNotIn(MatchSignal.NAME_EXACT.value, evidence.matched_signals)

    def test_source_unbound_evidence_is_ignored(self) -> None:
        cases = (
            _source_evidence(
                phone_matches=True,
                candidate_url_source_bound=False,
            ),
            _source_evidence(
                name_matches=False,
                city_matches=False,
                phone_matches=True,
                different_city_detected=True,
                candidate_url_source_bound=False,
            ),
        )
        for source_evidence in cases:
            with self.subTest(source_evidence=source_evidence):
                evidence = assess_website_candidate(
                    BusinessIdentity("Business Dental", "City", "Address Street 10"),
                    _candidate(
                        url="https://unrelated.example/",
                        title="Business Dental",
                        identity_evidence=source_evidence,
                    ),
                )
                self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
                self.assertEqual(
                    evidence.matched_signals,
                    (MatchSignal.NAME_EXACT.value,),
                )

    def test_negative_bound_source_identity_fails_closed(self) -> None:
        cases = (
            {"name_matches": False},
            {"city_matches": False},
            {"different_city_detected": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                evidence = assess_website_candidate(
                    BusinessIdentity("Business Dental", "City", "Address Street 10"),
                    _candidate(
                        url="https://unrelated.example/",
                        title="Business Dental",
                        identity_evidence=_source_evidence(**overrides),
                    ),
                )
                self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
                self.assertEqual(
                    evidence.rejected_reason,
                    "conflicting_source_identity_evidence",
                )

    def test_observed_phone_conflict_overrides_source_phone_match(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Business Dental", "City", phone="+380671234567"),
            _candidate(
                url="https://unrelated.example/",
                title="Business Dental +380501112233",
                identity_evidence=_source_evidence(
                    address_matches=False,
                    phone_matches=True,
                ),
            ),
        )
        self.assertEqual(evidence.rejected_reason, "conflicting_phone")
        self.assertEqual(evidence.matched_signals, ())

    def test_explicit_city_conflict_overrides_source_city_match(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Business Dental", "City", "Address Street 10"),
            _candidate(
                url="https://unrelated.example/",
                title="Business Dental",
                city="Other City",
                identity_evidence=_source_evidence(),
            ),
        )
        self.assertEqual(evidence.rejected_reason, "conflicting_city")
        self.assertEqual(evidence.matched_signals, ())

    def test_source_fields_require_corresponding_request_identity(self) -> None:
        cases = (
            (
                BusinessIdentity("Business Dental", "City", phone="+380671234567"),
                _source_evidence(address_matches=True, phone_matches=False),
                MatchSignal.SOURCE_ADDRESS_CORROBORATION,
            ),
            (
                BusinessIdentity("Business Dental", "City", "Address Street 10"),
                _source_evidence(address_matches=False, phone_matches=True),
                MatchSignal.SOURCE_PHONE_CORROBORATION,
            ),
        )
        for identity, source_evidence, absent_signal in cases:
            with self.subTest(absent_signal=absent_signal):
                evidence = assess_website_candidate(
                    identity,
                    _candidate(
                        url="https://unrelated.example/",
                        title="Business Dental",
                        identity_evidence=source_evidence,
                    ),
                )
                self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
                self.assertNotIn(absent_signal.value, evidence.matched_signals)

    def test_exact_extracted_phone_and_legacy_behavior_are_unchanged(self) -> None:
        phone_evidence = assess_website_candidate(
            BusinessIdentity("Business Dental", "City", phone="+380671234567"),
            _candidate(url="https://unrelated.example/", contact_phones=("0671234567",)),
        )
        legacy_evidence = assess_website_candidate(
            BusinessIdentity("Business Dental", "City", "Address Street 10"),
            _candidate(url="https://unrelated.example/", title="Business Dental"),
        )
        self.assertIs(phone_evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertEqual(phone_evidence.confidence, 0.75)
        self.assertIs(legacy_evidence.kind, CandidateKind.UNKNOWN)
        self.assertEqual(legacy_evidence.confidence, 0.35)

    def test_maps_and_search_exact_phone_are_official(self) -> None:
        for source in (CandidateSource.MAPS, CandidateSource.WEB_SEARCH):
            with self.subTest(source=source):
                evidence = assess_website_candidate(
                    _identity(),
                    _candidate(
                        source=source,
                        url="https://ordinary.example",
                        contact_phones=("0671234567",),
                    ),
                )
                self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
                self.assertEqual(evidence.confidence, 0.75)

    def test_name_only_and_name_plus_city_are_insufficient(self) -> None:
        name_only = assess_website_candidate(
            _identity(),
            _candidate(url="https://ordinary.example", title="Lido Beauty Studio"),
        )
        name_city = assess_website_candidate(
            _identity(),
            _candidate(
                url="https://ordinary.example",
                title="Lido Beauty Studio — Kyiv",
            ),
        )
        self.assertIs(name_only.kind, CandidateKind.UNKNOWN)
        self.assertIs(name_city.kind, CandidateKind.UNKNOWN)
        self.assertEqual(name_only.rejected_reason, "insufficient_identity_evidence")

    def test_name_and_structured_address_are_official(self) -> None:
        evidence = assess_website_candidate(
            _identity(phone=None, instagram_url=None),
            _candidate(
                url="https://ordinary.example",
                title="Lido Beauty Studio",
                contact_addresses=("10 Khreshchatyk Street, Kyiv",),
            ),
        )
        self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertEqual(
            evidence.matched_signals,
            (MatchSignal.NAME_EXACT.value, MatchSignal.ADDRESS_TOKEN_OVERLAP.value),
        )
        self.assertEqual(evidence.confidence, 0.65)

    def test_name_and_instagram_are_official_from_search_or_instagram_bio(self) -> None:
        for source in (CandidateSource.WEB_SEARCH, CandidateSource.INSTAGRAM_BIO):
            with self.subTest(source=source):
                evidence = assess_website_candidate(
                    _identity(phone=None, address=None),
                    _candidate(
                        source=source,
                        url="https://ordinary.example",
                        title="Lido Beauty Studio — @lido.beauty",
                    ),
                )
                self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)

    def test_name_token_overlap_can_be_accepted_with_address(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Lido Beauty Studio Center", "Kyiv", "Main Street 10"),
            _candidate(
                url="https://ordinary.example",
                title="Lido Beauty Studio official Kyiv",
                contact_addresses=("Main Street 10",),
            ),
        )
        self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertIn(MatchSignal.NAME_TOKEN_OVERLAP.value, evidence.matched_signals)
        self.assertNotIn(MatchSignal.NAME_EXACT.value, evidence.matched_signals)

    def test_conflicting_phone_and_structured_city_are_rejected(self) -> None:
        conflicting_phone = assess_website_candidate(
            _identity(),
            _candidate(contact_phones=("+380501112233",), title="Lido Beauty Studio"),
        )
        wrong_city = assess_website_candidate(
            _identity(phone=None),
            _candidate(city="Lviv", title="Lido Beauty Studio"),
        )
        self.assertEqual(conflicting_phone.rejected_reason, "conflicting_phone")
        self.assertEqual(wrong_city.rejected_reason, "conflicting_city")
        self.assertEqual(conflicting_phone.confidence, 0.0)
        self.assertEqual(wrong_city.confidence, 0.0)

    def test_one_matching_phone_prevents_conflict(self) -> None:
        evidence = assess_website_candidate(
            _identity(),
            _candidate(contact_phones=("+380501112233", "0671234567")),
        )
        self.assertIs(evidence.kind, CandidateKind.OFFICIAL_WEBSITE)

    def test_non_official_platforms_are_conservatively_rejected(self) -> None:
        cases = (
            ("https://instagram.com/lido", CandidateKind.SOCIAL_PROFILE),
            ("https://linktr.ee/lido", CandidateKind.LINK_IN_BIO),
            ("https://booksy.com/lido", CandidateKind.MARKETPLACE_OR_AGGREGATOR),
            ("https://locator.ua/lido", CandidateKind.DIRECTORY),
        )
        for url, kind in cases:
            with self.subTest(url=url):
                evidence = assess_website_candidate(
                    _identity(),
                    _candidate(source=CandidateSource.MAPS, url=url, contact_phones=("0671234567",)),
                )
                self.assertIs(evidence.kind, kind)
                self.assertEqual(evidence.confidence, 0.0)
                self.assertEqual(evidence.rejected_reason, "non_official_platform")

    def test_ordinary_domain_overlap_is_only_weak_evidence(self) -> None:
        evidence = assess_website_candidate(
            _identity(phone=None, address=None, instagram_url=None),
            _candidate(url="https://lido-beauty.example"),
        )
        self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
        self.assertEqual(evidence.matched_signals, (MatchSignal.DOMAIN_NAME_OVERLAP.value,))

    def test_final_redirect_controls_kind_and_final_domain(self) -> None:
        social = assess_website_candidate(
            _identity(),
            _candidate(
                url="https://redirect.example/lido",
                final_url="https://instagram.com/lido",
                contact_phones=("0671234567",),
            ),
        )
        official = assess_website_candidate(
            _identity(),
            _candidate(
                url="https://redirect.example/lido",
                final_url="https://official.example/home",
                contact_phones=("0671234567",),
            ),
        )
        self.assertIs(social.kind, CandidateKind.SOCIAL_PROFILE)
        self.assertEqual(social.final_domain, "instagram.com")
        self.assertIs(official.kind, CandidateKind.OFFICIAL_WEBSITE)
        self.assertEqual(official.final_domain, "official.example")
        self.assertEqual(official.normalized_url, "https://redirect.example/lido")

    def test_signal_order_and_confidence_cap_are_stable(self) -> None:
        evidence = assess_website_candidate(
            _identity(),
            _candidate(
                title="Lido Beauty Studio Kyiv @lido.beauty +380 67 123 45 67",
                city="Kyiv",
                contact_addresses=("Khreshchatyk Street 10",),
                instagram_usernames=("lido.beauty",),
            ),
        )
        self.assertEqual(
            evidence.matched_signals,
            (
                MatchSignal.PHONE_EXACT.value,
                MatchSignal.NAME_EXACT.value,
                MatchSignal.CITY_EXACT.value,
                MatchSignal.ADDRESS_TOKEN_OVERLAP.value,
                MatchSignal.INSTAGRAM_USERNAME.value,
                MatchSignal.DOMAIN_NAME_OVERLAP.value,
            ),
        )
        self.assertEqual(evidence.confidence, 1.0)

    def test_transliteration_alone_is_not_accepted(self) -> None:
        evidence = assess_website_candidate(
            BusinessIdentity("Лідо Студія", "Київ"),
            _candidate(url="https://lido-studio.example", title="Lido Studio Kyiv"),
        )
        self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
        self.assertEqual(evidence.confidence, 0.0)

    def test_same_business_name_in_different_structured_city_is_rejected(self) -> None:
        evidence = assess_website_candidate(
            _identity(phone=None),
            _candidate(title="Lido Beauty Studio", city="Odesa"),
        )
        self.assertEqual(evidence.rejected_reason, "conflicting_city")


class ResolutionAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identity = _identity()
        self.completed = (_attempt(),)
        self.required = (CandidateSource.WEB_SEARCH,)

    def test_one_accepted_domain_is_found(self) -> None:
        candidate = _candidate(contact_phones=("0671234567",))
        resolution = resolve_website_candidates(
            self.identity,
            (candidate,),
            self.completed,
            self.required,
        )
        self.assertIs(resolution.status, ResolutionStatus.FOUND_OFFICIAL)
        self.assertEqual(resolution.resolved_url, candidate.url)
        self.assertIs(resolution.source, CandidateSource.WEB_SEARCH)

    def test_wrong_same_name_kyiv_clinic_is_not_selected_for_zaporizhzhia(self) -> None:
        identity = BusinessIdentity(
            "STATUS стоматологія",
            "Запоріжжя",
            "вул. Поштова, 161/36",
        )
        candidate = candidate_from_search_result(SearchResult(
            "https://status-dental-clinic.com.ua/",
            "STATUS стоматологія — стоматологічна клініка",
            "Клініка у Києві, вул. Софії Русової, 3",
            1,
        ))
        resolution = resolve_website_candidates(
            identity,
            (candidate,),
            (_attempt(),),
            (CandidateSource.WEB_SEARCH,),
        )

        self.assertIs(resolution.status, ResolutionStatus.UNCERTAIN)
        self.assertIsNone(resolution.resolved_url)
        self.assertEqual(len(resolution.evidence), 1)
        evidence = resolution.evidence[0]
        self.assertIs(evidence.kind, CandidateKind.UNKNOWN)
        self.assertEqual(evidence.rejected_reason, "insufficient_identity_evidence")
        self.assertIn(MatchSignal.NAME_EXACT.value, evidence.matched_signals)
        self.assertNotIn(MatchSignal.PHONE_EXACT.value, evidence.matched_signals)
        self.assertNotIn(MatchSignal.ADDRESS_TOKEN_OVERLAP.value, evidence.matched_signals)

    def test_same_domain_from_maps_and_search_is_found_with_priority_tie_break(self) -> None:
        maps = _candidate(
            source=CandidateSource.MAPS,
            url="https://official.example/maps",
            contact_phones=("0671234567",),
        )
        search = _candidate(
            source=CandidateSource.WEB_SEARCH,
            url="https://official.example/search",
            contact_phones=("0671234567",),
        )
        resolution = resolve_website_candidates(
            self.identity,
            (search, maps),
            (_attempt(), _attempt(CandidateSource.MAPS)),
            (CandidateSource.MAPS, CandidateSource.WEB_SEARCH),
        )
        self.assertIs(resolution.status, ResolutionStatus.FOUND_OFFICIAL)
        self.assertIs(resolution.source, CandidateSource.MAPS)
        self.assertEqual(resolution.resolved_url, maps.url)
        self.assertEqual(len(resolution.evidence), 2)

    def test_higher_confidence_precedes_source_priority(self) -> None:
        maps = _candidate(
            source=CandidateSource.MAPS,
            url="https://official.example/maps",
            contact_phones=("0671234567",),
        )
        search = _candidate(
            url="https://official.example/search",
            title="Lido Beauty Studio Kyiv",
            city="Kyiv",
            contact_phones=("0671234567",),
        )
        resolution = resolve_website_candidates(
            self.identity,
            (maps, search),
            (_attempt(CandidateSource.MAPS), _attempt()),
            (CandidateSource.MAPS, CandidateSource.WEB_SEARCH),
        )
        self.assertIs(resolution.source, CandidateSource.WEB_SEARCH)

    def test_competing_accepted_domains_are_uncertain(self) -> None:
        candidates = (
            _candidate(url="https://one.example", contact_phones=("0671234567",)),
            _candidate(url="https://two.example", contact_phones=("0671234567",)),
        )
        resolution = resolve_website_candidates(
            self.identity, candidates, self.completed, self.required
        )
        self.assertIs(resolution.status, ResolutionStatus.UNCERTAIN)
        self.assertIsNone(resolution.resolved_url)
        self.assertEqual(resolution.confidence, 0.75)

    def test_competing_source_corroborated_domains_remain_uncertain(self) -> None:
        identity = BusinessIdentity(
            "Business Dental",
            "City",
            "Address Street 10",
        )
        candidates = (
            _candidate(
                url="https://clinic-one.example/",
                title="Business Dental",
                identity_evidence=_source_evidence(),
            ),
            _candidate(
                url="https://clinic-two.example/",
                title="Business Dental",
                identity_evidence=_source_evidence(),
            ),
        )
        resolution = resolve_website_candidates(
            identity, candidates, self.completed, self.required
        )
        self.assertIs(resolution.status, ResolutionStatus.UNCERTAIN)
        self.assertIsNone(resolution.resolved_url)
        self.assertEqual(resolution.confidence, 0.65)

    def test_strong_candidate_wins_despite_other_source_failure(self) -> None:
        resolution = resolve_website_candidates(
            self.identity,
            (_candidate(source=CandidateSource.MAPS, contact_phones=("0671234567",)),),
            (
                _attempt(CandidateSource.MAPS),
                _attempt(CandidateSource.WEB_SEARCH, SourceAttemptStatus.TIMEOUT, "slow"),
            ),
            (CandidateSource.MAPS, CandidateSource.WEB_SEARCH),
        )
        self.assertIs(resolution.status, ResolutionStatus.FOUND_OFFICIAL)

    def test_unavailable_missing_and_skipped_required_sources_are_uncertain(self) -> None:
        attempts = (
            (),
            (_attempt(status=SourceAttemptStatus.UNAVAILABLE, detail="missing provider"),),
            (_attempt(status=SourceAttemptStatus.SKIPPED, detail="not configured"),),
        )
        for source_attempts in attempts:
            with self.subTest(source_attempts=source_attempts):
                resolution = resolve_website_candidates(
                    self.identity, (), source_attempts, self.required
                )
                self.assertIs(resolution.status, ResolutionStatus.UNCERTAIN)
                self.assertIsNone(resolution.error)

    def test_timeout_rate_limit_and_error_are_resolution_errors(self) -> None:
        for status in (
            SourceAttemptStatus.TIMEOUT,
            SourceAttemptStatus.RATE_LIMITED,
            SourceAttemptStatus.ERROR,
        ):
            with self.subTest(status=status):
                resolution = resolve_website_candidates(
                    self.identity,
                    (),
                    (_attempt(status=status, detail="temporary failure"),),
                    self.required,
                )
                self.assertIs(resolution.status, ResolutionStatus.RESOLUTION_ERROR)
                self.assertEqual(resolution.confidence, 0.0)
                self.assertEqual(resolution.error, "web_search: temporary failure")

    def test_technical_errors_are_joined_in_required_source_order(self) -> None:
        resolution = resolve_website_candidates(
            self.identity,
            (),
            (
                _attempt(CandidateSource.WEB_SEARCH, SourceAttemptStatus.ERROR, "search"),
                _attempt(CandidateSource.MAPS, SourceAttemptStatus.TIMEOUT, "maps"),
            ),
            (CandidateSource.MAPS, CandidateSource.WEB_SEARCH),
        )
        self.assertEqual(resolution.error, "maps: maps; web_search: search")

    def test_all_completed_zero_or_aggregator_candidates_are_not_found(self) -> None:
        cases = (
            (),
            (_candidate(url="https://booksy.com/lido"),),
            (_candidate(url="https://locator.ua/lido"),),
        )
        for candidates in cases:
            with self.subTest(candidates=candidates):
                resolution = resolve_website_candidates(
                    self.identity, candidates, self.completed, self.required
                )
                self.assertIs(resolution.status, ResolutionStatus.NOT_FOUND)

    def test_all_completed_social_or_link_in_bio_is_social_only(self) -> None:
        for url in ("https://instagram.com/lido", "https://linktr.ee/lido"):
            with self.subTest(url=url):
                resolution = resolve_website_candidates(
                    self.identity,
                    (_candidate(url=url),),
                    self.completed,
                    self.required,
                )
                self.assertIs(resolution.status, ResolutionStatus.SOCIAL_ONLY)

    def test_low_confidence_unknown_is_uncertain(self) -> None:
        resolution = resolve_website_candidates(
            self.identity,
            (_candidate(url="https://ordinary.example", title="Lido Beauty Studio"),),
            self.completed,
            self.required,
        )
        self.assertIs(resolution.status, ResolutionStatus.UNCERTAIN)
        self.assertGreater(resolution.confidence, 0.0)

    def test_duplicate_same_source_is_ignored_and_evidence_order_is_preserved(self) -> None:
        first = _candidate(
            url="https://EXAMPLE.com/?utm_source=x",
            contact_phones=("0671234567",),
        )
        duplicate = _candidate(
            url="https://example.com/",
            title="ignored duplicate",
            contact_phones=("0671234567",),
        )
        second = _candidate(url="https://booksy.com/lido")
        resolution = resolve_website_candidates(
            self.identity,
            (first, duplicate, second),
            self.completed,
            self.required,
        )
        self.assertEqual(len(resolution.evidence), 2)
        self.assertEqual(
            tuple(item.normalized_url for item in resolution.evidence),
            (first.url, second.url),
        )

    def test_same_url_from_different_sources_is_retained(self) -> None:
        candidates = (
            _candidate(source=CandidateSource.MAPS, contact_phones=("0671234567",)),
            _candidate(source=CandidateSource.WEB_SEARCH, contact_phones=("0671234567",)),
        )
        resolution = resolve_website_candidates(
            self.identity,
            candidates,
            (_attempt(CandidateSource.MAPS), _attempt()),
            (CandidateSource.MAPS, CandidateSource.WEB_SEARCH),
        )
        self.assertEqual(len(resolution.evidence), 2)

    def test_rejects_invalid_sequences_items_required_and_duplicates(self) -> None:
        cases = (
            ("bad", self.completed, self.required, TypeError),
            ((), "bad", self.required, TypeError),
            ((), self.completed, "bad", TypeError),
            (("bad",), self.completed, self.required, TypeError),
            ((), ("bad",), self.required, TypeError),
            ((), self.completed, ("bad",), TypeError),
            ((), self.completed, (), ValueError),
            ((), (_attempt(), _attempt()), self.required, ValueError),
            ((), self.completed, (CandidateSource.WEB_SEARCH, CandidateSource.WEB_SEARCH), ValueError),
        )
        for candidates, attempts, required, error in cases:
            with self.subTest(candidates=candidates, attempts=attempts, required=required):
                with self.assertRaises(error):
                    resolve_website_candidates(self.identity, candidates, attempts, required)
        with self.assertRaises(TypeError):
            resolve_website_candidates("identity", (), self.completed, self.required)


class AuditScenarioTests(unittest.TestCase):
    def test_no_maps_url_search_exact_phone_match(self) -> None:
        resolution = resolve_website_candidates(
            _identity(),
            (_candidate(contact_phones=("0671234567",)),),
            (
                _attempt(CandidateSource.MAPS),
                _attempt(CandidateSource.WEB_SEARCH),
            ),
            (CandidateSource.MAPS, CandidateSource.WEB_SEARCH),
        )
        self.assertIs(resolution.status, ResolutionStatus.FOUND_OFFICIAL)

    def test_instagram_bio_linktree_is_non_official(self) -> None:
        evidence = assess_website_candidate(
            _identity(),
            _candidate(source=CandidateSource.INSTAGRAM_BIO, url="https://linktr.ee/lido"),
        )
        self.assertIs(evidence.kind, CandidateKind.LINK_IN_BIO)

    def test_provider_unavailable_never_not_found_and_timeout_is_error(self) -> None:
        unavailable = resolve_website_candidates(
            _identity(),
            (),
            (_attempt(status=SourceAttemptStatus.UNAVAILABLE, detail="not configured"),),
            (CandidateSource.WEB_SEARCH,),
        )
        timeout = resolve_website_candidates(
            _identity(),
            (),
            (_attempt(status=SourceAttemptStatus.TIMEOUT, detail="temporary"),),
            (CandidateSource.WEB_SEARCH,),
        )
        self.assertIs(unavailable.status, ResolutionStatus.UNCERTAIN)
        self.assertIs(timeout.status, ResolutionStatus.RESOLUTION_ERROR)


class BoundaryTests(unittest.TestCase):
    def test_module_has_no_network_or_production_imports(self) -> None:
        source_path = Path(website_candidate_matching.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])
        forbidden = {
            "httpx",
            "playwright",
            "requests",
            "aiohttp",
            "socket",
            "models",
            "orchestrator",
            "db",
        }
        self.assertTrue(forbidden.isdisjoint(imported_roots))

    def test_module_does_not_define_or_assign_quality_statuses(self) -> None:
        source_path = Path(website_candidate_matching.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        string_constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertTrue({"good", "bad", "dead"}.isdisjoint(string_constants))


if __name__ == "__main__":
    unittest.main()

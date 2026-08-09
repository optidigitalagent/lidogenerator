import unittest

from instagram_candidate_matching import (
    InstagramCandidate,
    InstagramSearchIdentityEvidence,
    InstagramSearchProviderError,
    assess_instagram_candidate,
    normalize_instagram_profile_url,
    normalize_instagram_username,
    resolve_instagram_candidates,
    resolve_instagram_via_search,
)
from instagram_resolution import (
    InstagramCandidateSource,
    InstagramIdentity,
    InstagramResolutionStatus,
)


class InstagramUrlNormalizationTests(unittest.TestCase):
    def test_canonicalizes_supported_profile_urls(self):
        cases = {
            "https://www.instagram.com/synthetic_brand/": "https://www.instagram.com/synthetic_brand/",
            "https://instagram.com/example.studio": "https://www.instagram.com/example.studio/",
            "http://www.instagram.com/alpha_test_shop?ref=search#bio": "https://www.instagram.com/alpha_test_shop/",
            "https://instagram.com/Mixed.Case_1/": "https://www.instagram.com/Mixed.Case_1/",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_instagram_profile_url(value), expected)

    def test_username_casefolds_and_accepts_dots_underscores(self):
        self.assertEqual(normalize_instagram_username("@Example.Studio_1"), "example.studio_1")
        self.assertEqual(normalize_instagram_username("https://instagram.com/Synthetic_Brand"), "synthetic_brand")

    def test_maximum_username_length(self):
        username = "a" * 30
        self.assertEqual(
            normalize_instagram_profile_url(f"https://instagram.com/{username}"),
            f"https://www.instagram.com/{username}/",
        )
        with self.assertRaises(ValueError):
            normalize_instagram_profile_url(f"https://instagram.com/{username}a")

    def test_rejects_reserved_and_non_profile_paths(self):
        paths = (
            "p/item", "reel/item", "reels", "stories/name", "explore/tags/test",
            "accounts/login", "search", "direct/inbox", "developer", "about",
            "legal", "privacy", "challenge", "login", "tv",
        )
        for path in paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                normalize_instagram_profile_url(f"https://instagram.com/{path}")

    def test_rejects_multiple_segments_domains_and_malformed_usernames(self):
        values = (
            "https://instagram.com/synthetic_brand/extra",
            "https://instagram.com//synthetic_brand",
            "https://instagram.com/synthetic_brand//",
            "https://notinstagram.invalid/synthetic_brand",
            "https://help.instagram.com/synthetic_brand",
            "https://instagram.com/bad-name",
            "https://instagram.com/bad%20name",
            "https://instagram.com/",
        )
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_instagram_profile_url(value)


def source_evidence(**overrides):
    values = {
        "name_matches": True,
        "city_matches": True,
        "address_matches": False,
        "phone_matches": False,
        "website_domain_matches": False,
        "different_city_detected": False,
        "candidate_url_source_bound": True,
    }
    values.update(overrides)
    return InstagramSearchIdentityEvidence(**values)


def candidate(username="alpha_test_shop", *, title="Alpha Test Shop", snippet="", evidence=None):
    return InstagramCandidate(
        InstagramCandidateSource.WEB_SEARCH,
        f"https://instagram.com/{username}",
        username,
        title,
        snippet,
        evidence,
    )


class InstagramMatcherTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.identity = InstagramIdentity(
            "Alpha Test Shop",
            "Example City",
            "12 Example Avenue",
            "+1 202 555 0101",
            "https://alpha-shop.invalid/",
        )

    def test_phone_exact_accepts_without_name(self):
        item = candidate("unrelated_handle", title="Contact +1 (202) 555-0101")
        result = resolve_instagram_candidates(self.identity, (item,))
        self.assertEqual(result.status, InstagramResolutionStatus.FOUND_OFFICIAL)
        self.assertIn("phone_exact", result.evidence[0].matched_signals)

    def test_exact_name_and_username_overlap_accept(self):
        result = resolve_instagram_candidates(
            InstagramIdentity("Alpha Test Shop", "Example City"),
            (candidate(),),
        )
        self.assertEqual(result.status, InstagramResolutionStatus.FOUND_OFFICIAL)
        self.assertEqual(result.confidence, 0.60)

    def test_source_address_phone_and_website_corroboration_accept(self):
        cases = (
            (InstagramIdentity("Alpha Test Shop", "Example City", address="12 Example Avenue"), source_evidence(address_matches=True), "source_address_corroboration"),
            (InstagramIdentity("Alpha Test Shop", "Example City", phone="+1 202 555 0101"), source_evidence(phone_matches=True), "source_phone_corroboration"),
            (InstagramIdentity("Alpha Test Shop", "Example City", website_url="https://alpha-shop.invalid"), source_evidence(website_domain_matches=True), "source_website_corroboration"),
        )
        for identity, evidence, signal in cases:
            with self.subTest(signal=signal):
                item = candidate("unrelated_handle", evidence=evidence)
                result = resolve_instagram_candidates(identity, (item,))
                self.assertEqual(result.status, InstagramResolutionStatus.FOUND_OFFICIAL)
                self.assertIn(signal, result.evidence[0].matched_signals)

    def test_observed_address_and_website_domain_accept(self):
        address_item = candidate(
            "unrelated_handle", snippet="Visit Alpha Test Shop at 12 Example Avenue"
        )
        domain_item = candidate(
            "unrelated_handle", snippet="Official site alpha-shop.invalid"
        )
        for item in (address_item, domain_item):
            with self.subTest(snippet=item.snippet):
                result = resolve_instagram_candidates(self.identity, (item,))
                self.assertEqual(result.status, InstagramResolutionStatus.FOUND_OFFICIAL)

    def test_token_overlap_combination_below_threshold_rejects(self):
        identity = InstagramIdentity("Alpha Beta Gamma Delta", "Example City", address="12 Example Avenue")
        item = candidate(
            "unrelated_handle",
            title="Alpha Beta Gamma services",
            snippet="12 Example Avenue",
        )
        evidence = assess_instagram_candidate(identity, item)
        self.assertEqual(evidence.confidence, 0.55)
        self.assertEqual(evidence.rejected_reason, "insufficient_identity_evidence")

    def test_provider_positive_without_deterministic_name_rejects(self):
        item = candidate(
            "unrelated_handle",
            title="Unrelated profile",
            evidence=source_evidence(address_matches=True, phone_matches=True, website_domain_matches=True),
        )
        result = resolve_instagram_candidates(self.identity, (item,))
        self.assertEqual(result.status, InstagramResolutionStatus.UNCERTAIN)
        self.assertEqual(result.evidence[0].rejected_reason, "insufficient_identity_evidence")

    def test_source_unbound_evidence_is_ignored(self):
        item = candidate(
            "unrelated_handle",
            evidence=source_evidence(address_matches=True, candidate_url_source_bound=False),
        )
        evidence = assess_instagram_candidate(self.identity, item)
        self.assertNotIn("source_address_corroboration", evidence.matched_signals)
        self.assertIsNotNone(evidence.rejected_reason)

    def test_bound_negative_identity_rejects(self):
        cases = (
            {"name_matches": False},
            {"city_matches": False},
            {"different_city_detected": True},
        )
        for override in cases:
            with self.subTest(override=override):
                item = candidate(evidence=source_evidence(**override))
                evidence = assess_instagram_candidate(self.identity, item)
                self.assertEqual(evidence.rejected_reason, "conflicting_source_identity_evidence")

    def test_observed_conflicting_phone_overrides_provider(self):
        item = candidate(
            snippet="Phone +1 202 555 0199",
            evidence=source_evidence(phone_matches=True, address_matches=True),
        )
        evidence = assess_instagram_candidate(self.identity, item)
        self.assertEqual(evidence.rejected_reason, "conflicting_phone")
        self.assertEqual(evidence.matched_signals, ())

    def test_generic_username_tokens_do_not_create_overlap(self):
        identity = InstagramIdentity("Beauty Studio", "Example City")
        item = candidate("beauty.studio", title="Beauty Studio")
        evidence = assess_instagram_candidate(identity, item)
        self.assertNotIn("username_name_overlap", evidence.matched_signals)
        self.assertIsNotNone(evidence.rejected_reason)

    def test_two_distinct_qualifying_profiles_are_uncertain(self):
        result = resolve_instagram_candidates(
            InstagramIdentity("Alpha Test Shop", "Example City"),
            (candidate("alpha_test_shop"), candidate("alpha.test.shop")),
        )
        self.assertEqual(result.status, InstagramResolutionStatus.UNCERTAIN)
        self.assertIsNone(result.resolved_url)

    def test_normalized_duplicate_username_collapses(self):
        first = candidate("Alpha_Test_Shop")
        second = candidate("alpha_test_shop")
        result = resolve_instagram_candidates(
            InstagramIdentity("Alpha Test Shop", "Example City"),
            (first, second),
        )
        self.assertEqual(result.status, InstagramResolutionStatus.FOUND_OFFICIAL)
        self.assertEqual(len(result.evidence), 1)

    async def test_provider_error_does_not_become_not_found(self):
        class FailingProvider:
            async def search(self, request):
                raise InstagramSearchProviderError("synthetic provider failure")

        result = await resolve_instagram_via_search(self.identity, FailingProvider())
        self.assertEqual(result.status, InstagramResolutionStatus.RESOLUTION_ERROR)

    def test_no_candidates_is_not_found(self):
        result = resolve_instagram_candidates(self.identity, ())
        self.assertEqual(result.status, InstagramResolutionStatus.NOT_FOUND)

    def test_evidence_requires_actual_booleans(self):
        with self.assertRaises(TypeError):
            source_evidence(name_matches=1)


if __name__ == "__main__":
    unittest.main()

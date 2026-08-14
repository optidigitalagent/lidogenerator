import unittest

from agents.website_presence_verifier import verify_business_website_presence
from models import Business
from website_candidate_matching import (
    ProviderRateLimited,
    ProviderTimeout,
    SearchIdentityEvidence,
    SearchProviderError,
    SearchResult,
)
from website_presence import WebsitePresenceSource, WebsitePresenceStatus


class Provider:
    def __init__(self, results=(), error=None):
        self.results = results
        self.error = error
        self.calls = 0

    async def search(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.results


def business(**changes):
    values = dict(
        name="Dental One",
        city="Харків",
        address="вул. Сумська 1",
        phone="+380501234567",
        instagram_url="https://instagram.com/dental.one/",
    )
    values.update(changes)
    return Business(**values)


class WebsitePresenceVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_maps_site_is_immediate_veto(self) -> None:
        provider = Provider()
        result = await verify_business_website_presence(
            business(website="https://sites.google.com/view/dental-one"), provider
        )
        self.assertEqual(result.status, WebsitePresenceStatus.PRESENT)
        self.assertEqual(result.source, WebsitePresenceSource.MAPS)
        self.assertEqual(result.evidence, ("maps_hosted_builder",))
        self.assertEqual(provider.calls, 0)

    async def test_hidden_official_custom_site_is_present(self) -> None:
        provider = Provider(
            (
                SearchResult(
                    "https://dental-one.ua/",
                    "Dental One — Харків",
                    "+380 50 123 45 67",
                    1,
                ),
            )
        )
        result = await verify_business_website_presence(business(), provider)
        self.assertEqual(result.status, WebsitePresenceStatus.PRESENT)
        self.assertEqual(provider.calls, 1)

    async def test_hosted_builder_requires_source_bound_identity(self) -> None:
        evidence = SearchIdentityEvidence(True, True, True, False, False, True)
        provider = Provider(
            (
                SearchResult(
                    "https://sites.google.com/view/dental-one",
                    "Dental One",
                    "Харків",
                    1,
                    evidence,
                ),
            )
        )
        result = await verify_business_website_presence(business(), provider)
        self.assertEqual(result.status, WebsitePresenceStatus.PRESENT)

    async def test_successful_social_only_search_confirms_absence(self) -> None:
        provider = Provider(
            (SearchResult("https://linktr.ee/dental.one", "Dental One", "Харків", 1),)
        )
        result = await verify_business_website_presence(business(), provider)
        self.assertEqual(result.status, WebsitePresenceStatus.ABSENT_CONFIRMED)

    async def test_unmatched_site_is_uncertain(self) -> None:
        provider = Provider(
            (SearchResult("https://other-business.ua/", "Other", "Kyiv", 1),)
        )
        result = await verify_business_website_presence(business(), provider)
        self.assertEqual(result.status, WebsitePresenceStatus.UNCERTAIN)

    async def test_successful_empty_search_confirms_absence(self) -> None:
        provider = Provider()
        result = await verify_business_website_presence(business(), provider)
        self.assertEqual(result.status, WebsitePresenceStatus.ABSENT_CONFIRMED)
        self.assertEqual(result.requests_used, 1)

    async def test_provider_failures_never_confirm_absence(self) -> None:
        for error in (
            ProviderTimeout("timeout"),
            ProviderRateLimited("429"),
            SearchProviderError("500"),
        ):
            with self.subTest(error=type(error).__name__):
                result = await verify_business_website_presence(
                    business(), Provider(error=error)
                )
                self.assertEqual(result.status, WebsitePresenceStatus.TECHNICAL_ERROR)


if __name__ == "__main__":
    unittest.main()

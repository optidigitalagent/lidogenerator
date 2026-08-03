import inspect
import unittest

from agents import website_resolver
from models import Business
from website_candidate_matching import ProviderTimeout, SearchResult
from website_resolution import ResolutionStatus


class FakeProvider:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    async def search(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return (
            SearchResult(
                "https://alpha-beauty.example/",
                "Alpha Beauty Kyiv",
                "Call +380501112233",
                1,
            ),
        )


class ResolverAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_strong_match_and_fields(self):
        provider = FakeProvider()
        business = Business(
            name="Alpha Beauty", city="Kyiv", phone="+380501112233",
            website="https://maps-site.example/", instagram_url="",
        )
        result = await website_resolver.resolve_business_website(business, provider)
        self.assertIs(result.status, ResolutionStatus.FOUND_OFFICIAL)
        self.assertEqual(provider.calls, 1)
        website_resolver.apply_resolution(business, result)
        self.assertEqual(business.website_original_url, "https://maps-site.example/")
        self.assertEqual(business.website_resolved_url, "https://alpha-beauty.example/")
        self.assertEqual(business.website, "https://maps-site.example/")

    async def test_no_provider_is_uncertain_not_not_found(self):
        business = Business(name="Alpha", city="Kyiv")
        result = await website_resolver.resolve_business_website(business)
        self.assertIs(result.status, ResolutionStatus.UNCERTAIN)

    async def test_invalid_maps_url_does_not_raise(self):
        business = Business(name="Alpha", city="Kyiv", website="not a URL")
        result = await website_resolver.resolve_business_website(business)
        self.assertIs(result.status, ResolutionStatus.RESOLUTION_ERROR)

    async def test_provider_timeout_is_resolution_error(self):
        business = Business(name="Alpha", city="Kyiv")
        result = await website_resolver.resolve_business_website(
            business, FakeProvider(ProviderTimeout("timed out"))
        )
        self.assertIs(result.status, ResolutionStatus.RESOLUTION_ERROR)

    async def test_invalid_optional_phone_is_omitted(self):
        business = Business(name="Alpha", city="Kyiv", phone="x")
        result = await website_resolver.resolve_business_website(business)
        self.assertIs(result.status, ResolutionStatus.UNCERTAIN)

    async def test_invalid_instagram_is_safe(self):
        business = Business(name="Alpha", city="Kyiv", instagram_url="https://example.com/a")
        result = await website_resolver.resolve_business_website(business)
        self.assertIs(result.status, ResolutionStatus.RESOLUTION_ERROR)
        self.assertNotIn("example.com", result.error)

    async def test_invalid_required_identity_is_per_business_error(self):
        result = await website_resolver.resolve_business_website(Business(name="", city="Kyiv"))
        self.assertIs(result.status, ResolutionStatus.RESOLUTION_ERROR)

    async def test_progress_and_same_objects(self):
        businesses = [Business(name="A", city="Kyiv"), Business(name="B", city="Kyiv")]
        progress = []

        async def callback(done, total):
            progress.append((done, total))

        result = await website_resolver.resolve_business_websites(businesses, progress_callback=callback)
        self.assertIs(result, businesses)
        self.assertEqual(progress, [(1, 2), (2, 2)])

    def test_agent_has_no_http_client_import(self):
        source = inspect.getsource(website_resolver)
        self.assertNotIn("import httpx", source)
        self.assertNotIn("import requests", source)


if __name__ == "__main__":
    unittest.main()

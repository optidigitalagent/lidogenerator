import json
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agents.openai_instagram_search_provider import (
    OPENAI_INSTAGRAM_SEARCH_SCHEMA,
    OpenAIInstagramSearchProvider,
    OpenAIInstagramSearchSettings,
    build_openai_instagram_search_input,
)
from instagram_candidate_matching import (
    InstagramSearchProviderError,
    InstagramSearchRequest,
)


PROFILE = "https://www.instagram.com/synthetic_brand/"


def result_item(**overrides):
    item = {
        "instagram_url": PROFILE,
        "title": "Synthetic Brand — Example City",
        "snippet": "Official profile for Synthetic Brand, 12 Example Avenue",
        "name_matches": True,
        "city_matches": True,
        "address_matches": True,
        "phone_matches": False,
        "website_domain_matches": False,
        "different_city_detected": False,
    }
    item.update(overrides)
    return item


def response(items, sources=(PROFILE,), tool_calls=1):
    output = []
    for _ in range(tool_calls):
        output.append({
            "type": "web_search_call",
            "action": {
                "type": "search",
                "sources": [{"url": value} for value in sources],
            },
        })
    return SimpleNamespace(
        status="completed",
        output=output,
        output_text=json.dumps({"results": items}),
    )


class FakeClient:
    def __init__(self, value):
        self.responses = SimpleNamespace(create=AsyncMock(return_value=value))


class OpenAIInstagramSearchProviderTests(unittest.IsolatedAsyncioTestCase):
    def settings(self):
        return OpenAIInstagramSearchSettings(api_key="synthetic-test-key")

    def request(self, **overrides):
        values = {
            "business_name": "Synthetic Brand",
            "city": "Example City",
            "address": "12 Example Avenue",
        }
        values.update(overrides)
        return InstagramSearchRequest(**values)

    def test_official_sdk_is_constructed_without_retries(self):
        sdk_client = FakeClient(response([]))
        with patch("openai.AsyncOpenAI", return_value=sdk_client) as constructor:
            OpenAIInstagramSearchProvider(self.settings())
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(kwargs["api_key"], "synthetic-test-key")

    async def test_one_required_web_search_and_strict_schema(self):
        client = FakeClient(response([]))
        provider = OpenAIInstagramSearchProvider(self.settings(), client)
        await provider.search(self.request())
        kwargs = client.responses.create.await_args.kwargs
        self.assertEqual(kwargs["tools"][0]["type"], "web_search")
        self.assertEqual(len(kwargs["tools"]), 1)
        self.assertEqual(kwargs["tool_choice"], "required")
        self.assertEqual(kwargs["max_tool_calls"], 1)
        self.assertFalse(kwargs["store"])
        self.assertEqual(
            kwargs["text"]["format"]["schema"],
            OPENAI_INSTAGRAM_SEARCH_SCHEMA,
        )
        properties = OPENAI_INSTAGRAM_SEARCH_SCHEMA["properties"]["results"]["items"]["properties"]
        self.assertEqual(set(properties), {
            "instagram_url", "title", "snippet", "name_matches", "city_matches",
            "address_matches", "phone_matches", "website_domain_matches",
            "different_city_detected",
        })

    def test_deterministic_query_variants_and_phone_priority(self):
        request = self.request(
            phone="+1 202 555 0101",
            website_url="https://synthetic.invalid",
        )
        prompt = build_openai_instagram_search_input(request)
        first_query = prompt.split("Suggested query variants:\n", 1)[1].splitlines()[0]
        self.assertIn('"Synthetic Brand" "12025550101" Instagram', first_query)
        self.assertIn('"Synthetic Brand" "Example City" Instagram', prompt)
        self.assertIn('"Synthetic Brand" "12 Example Avenue" Instagram', prompt)
        self.assertEqual(prompt, build_openai_instagram_search_input(request))
        self.assertNotIn("expected_username", prompt.casefold())
        self.assertNotIn("benchmark", prompt.casefold())

    def test_website_domain_query_variant(self):
        prompt = build_openai_instagram_search_input(
            self.request(address=None, website_url="https://www.synthetic.invalid/path")
        )
        first_query = prompt.split("Suggested query variants:\n", 1)[1].splitlines()[0]
        self.assertIn('"Synthetic Brand" "synthetic.invalid" Instagram', first_query)

    async def test_source_bound_candidate_is_returned_with_all_evidence(self):
        item = result_item(phone_matches=True, website_domain_matches=True)
        client = FakeClient(response([item]))
        provider = OpenAIInstagramSearchProvider(self.settings(), client)
        results = await provider.search(self.request(phone="+1 202 555 0101"))
        self.assertEqual(len(results), 1)
        evidence = results[0].identity_evidence
        self.assertTrue(evidence.candidate_url_source_bound)
        self.assertTrue(evidence.name_matches)
        self.assertTrue(evidence.city_matches)
        self.assertTrue(evidence.address_matches)
        self.assertTrue(evidence.phone_matches)
        self.assertTrue(evidence.website_domain_matches)
        self.assertFalse(evidence.different_city_detected)

    async def test_unsourced_candidate_is_discarded(self):
        client = FakeClient(response([result_item()], sources=("https://instagram.com/other_synthetic",)))
        provider = OpenAIInstagramSearchProvider(self.settings(), client)
        self.assertEqual(await provider.search(self.request()), ())

    async def test_post_and_reel_urls_are_discarded(self):
        for invalid in (
            "https://instagram.com/p/synthetic_post",
            "https://instagram.com/reel/synthetic_reel",
        ):
            with self.subTest(url=invalid):
                client = FakeClient(response(
                    [result_item(instagram_url=invalid)],
                    sources=(PROFILE, invalid),
                ))
                provider = OpenAIInstagramSearchProvider(self.settings(), client)
                self.assertEqual(await provider.search(self.request()), ())

    async def test_identity_prefilter_and_different_city_reject(self):
        items = (
            result_item(name_matches=False),
            result_item(city_matches=False),
            result_item(different_city_detected=True),
            result_item(address_matches=False),
        )
        for item in items:
            with self.subTest(item=item):
                provider = OpenAIInstagramSearchProvider(
                    self.settings(), FakeClient(response([item]))
                )
                self.assertEqual(await provider.search(self.request()), ())
                self.assertEqual(provider.telemetry().identity_candidates_rejected, 1)

    async def test_prefilter_uses_available_corroborators(self):
        request = self.request(
            phone="+1 202 555 0101",
            website_url="https://synthetic.invalid",
        )
        for field in ("address_matches", "phone_matches", "website_domain_matches"):
            values = {
                "address_matches": False,
                "phone_matches": False,
                "website_domain_matches": False,
                field: True,
            }
            provider = OpenAIInstagramSearchProvider(
                self.settings(), FakeClient(response([result_item(**values)]))
            )
            with self.subTest(field=field):
                self.assertEqual(len(await provider.search(request)), 1)

    async def test_duplicate_profiles_are_deduplicated(self):
        duplicate = result_item(instagram_url="http://instagram.com/synthetic_brand?x=1")
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([result_item(), duplicate]))
        )
        self.assertEqual(len(await provider.search(self.request())), 1)

    async def test_more_than_one_tool_call_fails_closed(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([result_item()], tool_calls=2))
        )
        with self.assertRaises(InstagramSearchProviderError):
            await provider.search(self.request())
        telemetry = provider.telemetry()
        self.assertTrue(telemetry.tool_call_limit_exceeded)
        self.assertEqual(telemetry.last_error_category, "tool_call_limit")

    async def test_malformed_payload_is_typed_error(self):
        malformed = result_item(name_matches=1)
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([malformed]))
        )
        with self.assertRaises(InstagramSearchProviderError):
            await provider.search(self.request())
        self.assertEqual(provider.telemetry().last_error_category, "response_error")

    async def test_telemetry_is_safe_and_counts_results(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([result_item()]))
        )
        await provider.search(self.request())
        snapshot = provider.telemetry()
        self.assertEqual(snapshot.requests_started, 1)
        self.assertEqual(snapshot.requests_succeeded, 1)
        self.assertEqual(snapshot.requests_failed, 0)
        self.assertEqual(snapshot.tool_calls_seen, 1)
        self.assertEqual(snapshot.search_actions_seen, 1)
        self.assertEqual(snapshot.sources_seen, 1)
        self.assertEqual(snapshot.candidates_returned, 1)
        self.assertNotIn("prompt", asdict(snapshot))
        self.assertNotIn("url", " ".join(asdict(snapshot).keys()))
        stored = repr(provider.__dict__)
        self.assertNotIn(PROFILE, stored)
        self.assertNotIn("12 Example Avenue", stored)


if __name__ == "__main__":
    unittest.main()

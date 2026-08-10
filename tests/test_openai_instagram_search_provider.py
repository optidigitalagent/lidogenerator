import json
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from agents.openai_instagram_search_provider import (
    OPENAI_INSTAGRAM_SEARCH_SCHEMA,
    OpenAIInstagramSearchProvider,
    OpenAIInstagramSearchSettings,
    _suggested_query_variants,
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
        self.assertEqual(
            kwargs["tools"][0]["filters"]["allowed_domains"],
            ["instagram.com"],
        )
        self.assertEqual(kwargs["tools"][0]["search_context_size"], "low")
        self.assertEqual(kwargs["tools"][0]["user_location"], {
            "type": "approximate",
            "country": "UA",
            "city": "Example City",
        })
        self.assertTrue(kwargs["tools"][0]["external_web_access"])
        self.assertEqual(kwargs["tool_choice"], "required")
        self.assertEqual(kwargs["max_tool_calls"], 1)
        self.assertEqual(kwargs["include"], ["web_search_call.action.sources"])
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
        queries = _suggested_query_variants(request)
        prompt = build_openai_instagram_search_input(request)
        self.assertEqual(queries, (
            '"Synthetic Brand" "12025550101"',
            '"Synthetic Brand" "Example City"',
            '"Synthetic Brand" "synthetic.invalid"',
        ))
        self.assertLessEqual(len(queries), 3)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(all("site:instagram.com" not in query for query in queries))
        self.assertEqual(prompt, build_openai_instagram_search_input(request))
        self.assertNotIn("expected_username", prompt.casefold())
        self.assertNotIn("benchmark", prompt.casefold())

    def test_website_domain_query_variant(self):
        queries = _suggested_query_variants(
            self.request(website_url="https://www.synthetic.invalid/path")
        )
        self.assertEqual(queries, (
            '"Synthetic Brand" "synthetic.invalid"',
            '"Synthetic Brand" "Example City"',
            '"Synthetic Brand" "12 Example Avenue"',
        ))

    def test_address_and_name_city_query_fallbacks(self):
        self.assertEqual(
            _suggested_query_variants(self.request()),
            (
                '"Synthetic Brand" "Example City"',
                '"Synthetic Brand" "12 Example Avenue"',
            ),
        )
        self.assertEqual(
            _suggested_query_variants(self.request(address=None)),
            ('"Synthetic Brand" "Example City"',),
        )

    def test_prompt_enumerates_candidates_for_deterministic_downstream_decision(self):
        prompt = build_openai_instagram_search_input(self.request()).casefold()
        self.assertIn("candidate enumerator", prompt)
        self.assertIn("not the final official-account decision", prompt)
        self.assertIn("plausible direct instagram profile candidates", prompt)
        self.assertIn("only return urls visibly present", prompt)
        self.assertIn("never fabricate", prompt)
        self.assertIn("posts, reels, stories", prompt)
        self.assertIn("employees or personal accounts", prompt)
        self.assertIn("fan pages", prompt)
        self.assertIn("influencers", prompt)
        self.assertIn("wrong-city profiles", prompt)
        self.assertIn("do not omit a plausible direct profile solely because", prompt)
        self.assertIn("address_matches, phone_matches, and website_domain_matches", prompt)
        self.assertIn("to false", prompt)
        self.assertIn("name_matches and city_matches truthfully", prompt)
        self.assertIn("different_city_detected to true", prompt)
        self.assertIn("multiple plausible direct profile candidates", prompt)
        self.assertIn("downstream deterministic code", prompt)
        self.assertIn("final official-account decision", prompt)
        self.assertIn("only when no plausible direct instagram profile", prompt)

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

    async def test_source_bound_identity_is_case_insensitive(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(),
            FakeClient(response(
                [result_item(instagram_url=PROFILE)],
                sources=("https://www.instagram.com/Synthetic_Brand/",),
            )),
        )
        results = await provider.search(self.request())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, PROFILE)
        self.assertTrue(results[0].identity_evidence.candidate_url_source_bound)

    async def test_source_bound_identity_rejects_different_username(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(),
            FakeClient(response(
                [result_item(instagram_url=PROFILE)],
                sources=("https://www.instagram.com/other_synthetic/",),
            )),
        )
        self.assertEqual(await provider.search(self.request()), ())

    async def test_unsourced_candidate_is_discarded(self):
        client = FakeClient(response(
            [result_item()],
            sources=("https://synthetic.invalid/about",),
        ))
        provider = OpenAIInstagramSearchProvider(self.settings(), client)
        self.assertEqual(await provider.search(self.request()), ())
        telemetry = provider.telemetry()
        self.assertEqual(telemetry.requests_succeeded, 1)
        self.assertEqual(telemetry.requests_failed, 0)
        self.assertEqual(telemetry.direct_profile_sources_seen, 0)
        self.assertEqual(telemetry.source_unbound_candidates_discarded, 1)

    async def test_zero_direct_profile_sources_is_safe_empty_success(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([result_item()], sources=()))
        )
        self.assertEqual(await provider.search(self.request()), ())
        telemetry = provider.telemetry()
        self.assertEqual(telemetry.requests_succeeded, 1)
        self.assertIsNone(telemetry.last_error_category)

    async def test_post_reel_and_story_sources_do_not_bind(self):
        for invalid in (
            "https://instagram.com/p/synthetic_post",
            "https://instagram.com/reel/synthetic_reel",
            "https://instagram.com/stories/synthetic_brand/123",
        ):
            with self.subTest(url=invalid):
                client = FakeClient(response(
                    [result_item()],
                    sources=(invalid,),
                ))
                provider = OpenAIInstagramSearchProvider(self.settings(), client)
                self.assertEqual(await provider.search(self.request()), ())
                telemetry = provider.telemetry()
                self.assertEqual(telemetry.direct_profile_sources_seen, 0)
                self.assertEqual(telemetry.source_unbound_candidates_discarded, 1)

    async def test_invalid_structured_profile_is_discarded(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(),
            FakeClient(response([
                result_item(instagram_url="https://instagram.com/p/synthetic_post")
            ])),
        )
        self.assertEqual(await provider.search(self.request()), ())
        self.assertEqual(
            provider.telemetry().invalid_profile_candidates_discarded,
            1,
        )

    async def test_mixed_sources_return_only_exact_source_bound_candidate(self):
        other = "https://www.instagram.com/other_synthetic/"
        provider = OpenAIInstagramSearchProvider(
            self.settings(),
            FakeClient(response(
                [result_item(), result_item(instagram_url=other)],
                sources=("https://synthetic.invalid/about", PROFILE),
            )),
        )
        results = await provider.search(self.request())
        self.assertEqual(tuple(item.url for item in results), (PROFILE,))
        telemetry = provider.telemetry()
        self.assertEqual(telemetry.source_bound_candidates_returned, 1)
        self.assertEqual(telemetry.source_unbound_candidates_discarded, 1)

    async def test_multiple_source_bound_candidates_are_returned_downstream(self):
        other = "https://www.instagram.com/other_synthetic/"
        provider = OpenAIInstagramSearchProvider(
            self.settings(),
            FakeClient(response(
                [result_item(), result_item(instagram_url=other)],
                sources=(PROFILE, other),
            )),
        )
        results = await provider.search(self.request())
        self.assertEqual(tuple(item.url for item in results), (PROFILE, other))
        self.assertEqual(provider.telemetry().source_bound_candidates_returned, 2)

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

    async def test_zero_tool_calls_fails_closed(self):
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([], sources=(), tool_calls=0))
        )
        with self.assertRaises(InstagramSearchProviderError):
            await provider.search(self.request())
        self.assertEqual(provider.telemetry().last_error_category, "response_error")

    async def test_malformed_payload_is_typed_error(self):
        malformed = result_item(name_matches=1)
        provider = OpenAIInstagramSearchProvider(
            self.settings(), FakeClient(response([malformed]))
        )
        with self.assertRaises(InstagramSearchProviderError):
            await provider.search(self.request())
        self.assertEqual(provider.telemetry().last_error_category, "response_error")

    async def test_api_error_is_typed_error(self):
        client = FakeClient(response([]))
        client.responses.create.side_effect = RuntimeError("synthetic failure")
        provider = OpenAIInstagramSearchProvider(self.settings(), client)
        with self.assertRaises(InstagramSearchProviderError):
            await provider.search(self.request())
        telemetry = provider.telemetry()
        self.assertEqual(telemetry.requests_failed, 1)
        self.assertEqual(telemetry.last_error_category, "api_error")

    async def test_telemetry_is_safe_and_counts_results(self):
        other = "https://www.instagram.com/other_synthetic/"
        provider = OpenAIInstagramSearchProvider(
            self.settings(),
            FakeClient(response(
                [
                    result_item(),
                    result_item(address_matches=False),
                    result_item(instagram_url="https://instagram.com/p/invalid_fixture"),
                    result_item(instagram_url=other),
                ],
                sources=(
                    PROFILE,
                    "https://instagram.com/reel/source_fixture",
                    "https://synthetic.invalid/about",
                ),
            )),
        )
        await provider.search(self.request())
        snapshot = provider.telemetry()
        self.assertEqual(snapshot.requests_started, 1)
        self.assertEqual(snapshot.requests_succeeded, 1)
        self.assertEqual(snapshot.requests_failed, 0)
        self.assertEqual(snapshot.tool_calls_seen, 1)
        self.assertEqual(snapshot.search_actions_seen, 1)
        self.assertEqual(snapshot.sources_seen, 3)
        self.assertEqual(snapshot.candidates_returned, 1)
        self.assertEqual(snapshot.structured_candidates_seen, 4)
        self.assertEqual(snapshot.identity_prefilter_rejected, 1)
        self.assertEqual(snapshot.identity_candidates_rejected, 1)
        self.assertEqual(snapshot.direct_profile_sources_seen, 1)
        self.assertEqual(snapshot.invalid_profile_candidates_discarded, 1)
        self.assertEqual(snapshot.source_unbound_candidates_discarded, 1)
        self.assertEqual(snapshot.source_bound_candidates_returned, 1)
        self.assertNotIn("prompt", asdict(snapshot))
        self.assertNotIn("url", " ".join(asdict(snapshot).keys()))
        stored = repr(provider.__dict__)
        self.assertNotIn(PROFILE, stored)
        self.assertNotIn("12 Example Avenue", stored)


if __name__ == "__main__":
    unittest.main()

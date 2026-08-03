"""Offline contract tests for the Brave Web Search adapter."""

from dataclasses import FrozenInstanceError
import unittest

import httpx

from agents.brave_search_provider import (
    BRAVE_WEB_SEARCH_ENDPOINT,
    BraveSearchProvider,
    BraveSearchSettings,
    build_brave_search_query,
)
from website_candidate_matching import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderTimeout,
    SearchProviderError,
    SearchRequest,
)


class BraveSearchSettingsTests(unittest.TestCase):
    def test_defaults_normalization_and_secret_repr(self) -> None:
        settings = BraveSearchSettings("  top-secret  ")
        self.assertEqual(settings.api_key, "top-secret")
        self.assertEqual(settings.country, "UA")
        self.assertIsNone(settings.search_lang)
        self.assertEqual(settings.ui_lang, "uk-UA")
        self.assertEqual(settings.safesearch, "moderate")
        self.assertEqual(settings.max_results, 5)
        self.assertEqual(settings.timeout_seconds, 10.0)
        self.assertNotIn("top-secret", repr(settings))
        with self.assertRaises(FrozenInstanceError):
            settings.country = "US"  # type: ignore[misc]

    def test_normalized_optional_values(self) -> None:
        settings = BraveSearchSettings(
            "key",
            country="all",
            search_lang=" UK ",
            ui_lang="EN-us",
            safesearch=" STRICT ",
            max_results=10,
            timeout_seconds=0.5,
        )
        self.assertEqual(settings.country, "ALL")
        self.assertEqual(settings.search_lang, "uk")
        self.assertEqual(settings.ui_lang, "en-US")
        self.assertEqual(settings.safesearch, "strict")
        self.assertEqual(settings.timeout_seconds, 0.5)
        self.assertIsNone(BraveSearchSettings("key", search_lang="", ui_lang="").search_lang)
        self.assertIsNone(BraveSearchSettings("key", ui_lang="").ui_lang)

    def test_invalid_settings(self) -> None:
        invalid = (
            ({"api_key": ""}, (ValueError,)),
            ({"api_key": 1}, (TypeError,)),
            ({"api_key": "key", "country": "U"}, (ValueError,)),
            ({"api_key": "key", "country": "УА"}, (ValueError,)),
            ({"api_key": "key", "search_lang": "u"}, (ValueError,)),
            ({"api_key": "key", "search_lang": 1}, (TypeError,)),
            ({"api_key": "key", "ui_lang": "uk_ua"}, (ValueError,)),
            ({"api_key": "key", "safesearch": "maximum"}, (ValueError,)),
            ({"api_key": "key", "max_results": True}, (TypeError,)),
            ({"api_key": "key", "max_results": 0}, (ValueError,)),
            ({"api_key": "key", "max_results": 11}, (ValueError,)),
            ({"api_key": "key", "timeout_seconds": True}, (TypeError,)),
            ({"api_key": "key", "timeout_seconds": float("inf")}, (ValueError,)),
            ({"api_key": "key", "timeout_seconds": 0}, (ValueError,)),
            ({"api_key": "key", "timeout_seconds": 31}, (ValueError,)),
        )
        for kwargs, errors in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(errors):
                BraveSearchSettings(**kwargs)


class BraveQueryBuilderTests(unittest.TestCase):
    def test_cyrillic_identity_phone_address_and_apostrophe(self) -> None:
        request = SearchRequest(
            business_name=" Салон O'Brien ",
            city="Київ",
            phone="+380 67 123 45 67",
            address="вул. Хрещатик, 1",
        )
        self.assertEqual(
            build_brave_search_query(request),
            '"Салон O\'Brien" Київ 380671234567 вул. Хрещатик, 1',
        )

    def test_quotes_controls_and_operators_are_sanitized(self) -> None:
        request = SearchRequest(
            business_name=' "Best"\nsite:evil ',
            city="Ки\x00їв",
            address="filetype:pdf OR -test",
        )
        query = build_brave_search_query(request)
        self.assertEqual(query, '"Best site evil" Київ filetype pdf test')
        self.assertEqual(query.count('"'), 2)
        self.assertNotIn("site:", query)
        self.assertNotIn("filetype:", query)
        self.assertNotIn(" OR ", query)

    def test_long_values_are_bounded_deterministic_and_closed(self) -> None:
        request = SearchRequest(
            business_name="Назва " * 90,
            city="Дуже Довге Місто " * 15,
            phone="+380671234567",
            address="Адреса " * 100,
        )
        first = build_brave_search_query(request)
        second = build_brave_search_query(request)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 400)
        self.assertLessEqual(len(first.split()), 50)
        self.assertTrue(first.startswith('"Н'))
        self.assertEqual(first.count('"'), 2)
        self.assertIn("Д", first)
        self.assertNotIn("Адреса", first)
        self.assertNotIn("380671234567", first)

    def test_address_is_truncated_before_phone(self) -> None:
        request = SearchRequest(
            business_name="Business",
            city="Kyiv",
            phone="+380671234567",
            address="address " * 100,
        )
        query = build_brave_search_query(request)
        self.assertIn("380671234567", query)
        self.assertLessEqual(len(query), 400)
        self.assertLessEqual(len(query.split()), 50)

    def test_operator_only_name_with_long_city_stays_bounded(self) -> None:
        query = build_brave_search_query(SearchRequest(
            business_name='""',
            city="City " * 100,
        ))
        self.assertTrue(query)
        self.assertLessEqual(len(query), 400)
        self.assertLessEqual(len(query.split()), 50)
        self.assertEqual(query.count('"'), 2)


class BraveSearchProviderTests(unittest.IsolatedAsyncioTestCase):
    def _provider(self, handler, **settings) -> BraveSearchProvider:
        return BraveSearchProvider(
            BraveSearchSettings("top-secret", **settings),
            transport=httpx.MockTransport(handler),
        )

    def test_constructor_rejects_invalid_transport(self) -> None:
        with self.assertRaises(TypeError):
            BraveSearchProvider(BraveSearchSettings("key"), transport=object())  # type: ignore[arg-type]

    async def test_request_contract_caps_and_successful_order(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"web": {"results": [
                {"url": "https://example.com/a", "title": " A ", "description": " one "},
                {"url": "https://example.org", "title": "B", "description": "two"},
            ]}}, headers={
                "X-RateLimit-Limit": " 1,   1000 ",
                "X-RateLimit-Remaining": "9",
                "X-RateLimit-Reset": "7",
            })

        provider = self._provider(
            handler,
            country="ua",
            search_lang=None,
            ui_lang="uk-UA",
            max_results=2,
            timeout_seconds=3,
        )
        results = await provider.search(SearchRequest(
            "Бізнес", "Київ", max_results=10, timeout_seconds=20,
        ))
        self.assertIsInstance(results, tuple)
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].snippet, "one")
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(
            f"{request.url.scheme}://{request.url.host}{request.url.path}",
            BRAVE_WEB_SEARCH_ENDPOINT,
        )
        self.assertEqual(request.headers["X-Subscription-Token"], "top-secret")
        self.assertEqual(request.headers["Accept"], "application/json")
        self.assertEqual(
            request.headers["User-Agent"],
            "lidogenerator-website-resolver/1.0",
        )
        self.assertNotIn("top-secret", str(request.url))
        self.assertEqual(request.url.params["count"], "2")
        self.assertEqual(request.url.params["country"], "UA")
        self.assertEqual(request.url.params["safesearch"], "moderate")
        self.assertEqual(request.url.params["ui_lang"], "uk-UA")
        self.assertNotIn("search_lang", request.url.params)
        self.assertEqual(request.extensions["timeout"]["read"], 3.0)
        telemetry = provider.telemetry()
        self.assertEqual((telemetry.requests_started, telemetry.requests_succeeded, telemetry.requests_failed), (1, 1, 0))
        self.assertEqual(telemetry.last_rate_limit_limit, "1, 1000")
        self.assertEqual(telemetry.last_rate_limit_remaining, "9")

    async def test_optional_language_and_redirects_are_not_followed(self) -> None:
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={"Location": "https://other.invalid/"})

        provider = self._provider(handler, search_lang="uk", ui_lang=None)
        with self.assertRaisesRegex(SearchProviderError, "unexpected status"):
            await provider.search(SearchRequest("Name", "City"))
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.params["search_lang"], "uk")
        self.assertNotIn("ui_lang", requests[0].url.params)

    async def test_missing_web_and_results_are_empty_successes(self) -> None:
        for payload in ({}, {"web": None}, {"web": {}}, {"web": {"results": None}}):
            with self.subTest(payload=payload):
                provider = self._provider(lambda request, payload=payload: httpx.Response(200, json=payload))
                self.assertEqual(await provider.search(SearchRequest("Name", "City")), ())
                self.assertEqual(provider.telemetry().requests_succeeded, 1)

    async def test_invalid_items_and_urls_are_skipped_with_original_rank(self) -> None:
        payload = {"web": {"results": [
            None,
            {"url": "javascript:alert(1)", "title": "bad"},
            {"url": "https://valid.example", "title": None, "description": None},
            {"url": "https://wrong-title.example", "title": 7},
            {"url": "https://second.example", "description": "ok"},
        ]}}
        provider = self._provider(lambda request: httpx.Response(200, json=payload), max_results=2)
        results = await provider.search(SearchRequest("Name", "City", max_results=5))
        self.assertEqual([result.url for result in results], ["https://valid.example/", "https://second.example/"])
        self.assertEqual([result.rank for result in results], [3, 5])
        self.assertEqual(results[0].title, "")

    async def test_invalid_response_shapes(self) -> None:
        responses = (
            httpx.Response(200, content=b"not-json"),
            httpx.Response(200, json=[]),
            httpx.Response(200, json={"web": []}),
            httpx.Response(200, json={"web": {"results": {}}}),
        )
        for response in responses:
            with self.subTest(content=response.content):
                provider = self._provider(lambda request, response=response: response)
                with self.assertRaisesRegex(SearchProviderError, "invalid brave response"):
                    await provider.search(SearchRequest("private business", "private city"))
                self.assertEqual(provider.telemetry().requests_failed, 1)

    async def test_http_error_mapping_and_safe_messages(self) -> None:
        cases = (
            (401, ProviderAuthError, "brave authentication failed"),
            (403, ProviderAuthError, "brave authentication failed"),
            (408, ProviderTimeout, "brave request timed out"),
            (429, ProviderRateLimited, "brave rate limit exceeded"),
            (500, SearchProviderError, "brave service error"),
            (418, SearchProviderError, "brave returned unexpected status"),
        )
        for status, error_type, message in cases:
            with self.subTest(status=status):
                provider = self._provider(
                    lambda request, status=status: httpx.Response(
                        status, content=b"sensitive response body"
                    )
                )
                with self.assertRaises(error_type) as raised:
                    await provider.search(SearchRequest("private business", "private city"))
                self.assertEqual(str(raised.exception), message)
                text = str(raised.exception)
                self.assertNotIn("top-secret", text)
                self.assertNotIn("private business", text)
                self.assertNotIn("sensitive response body", text)
                self.assertEqual(provider.telemetry().requests_failed, 1)

    async def test_timeout_and_request_error_mapping(self) -> None:
        errors = (
            (httpx.ReadTimeout("private timeout"), ProviderTimeout, "brave request timed out"),
            (httpx.ConnectError("private request"), SearchProviderError, "brave request failed"),
        )
        for transport_error, expected, message in errors:
            def handler(request, transport_error=transport_error):
                raise transport_error

            provider = self._provider(handler)
            with self.subTest(expected=expected), self.assertRaises(expected) as raised:
                await provider.search(SearchRequest("private business", "private city"))
            self.assertEqual(str(raised.exception), message)
            self.assertEqual(provider.telemetry().requests_started, 1)
            self.assertEqual(provider.telemetry().requests_failed, 1)

    async def test_rate_limit_header_truncation_and_frozen_snapshot(self) -> None:
        provider = self._provider(lambda request: httpx.Response(
            429,
            headers={"X-RateLimit-Limit": "  " + "x" * 250 + "  "},
        ))
        with self.assertRaises(ProviderRateLimited):
            await provider.search(SearchRequest("Name", "City"))
        snapshot = provider.telemetry()
        self.assertEqual(len(snapshot.last_rate_limit_limit or ""), 200)
        self.assertFalse(hasattr(snapshot, "api_key"))
        self.assertFalse(hasattr(snapshot, "query"))
        with self.assertRaises(FrozenInstanceError):
            snapshot.requests_failed = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

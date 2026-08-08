"""Fully mocked tests for the OpenAI Responses Web Search provider."""

from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError
import inspect
import io
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from agents import openai_web_search_provider as provider_module
from agents.openai_web_search_provider import (
    OPENAI_WEB_SEARCH_SCHEMA,
    OPENAI_WEB_SEARCH_TEXT_FORMAT,
    OpenAIWebSearchProvider,
    OpenAIWebSearchSettings,
    build_openai_web_search_input,
)
from scripts import validate_openai_web_search
import website_search_runtime as runtime
from website_candidate_matching import (
    ProviderAuthError,
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    SearchProviderError,
    SearchIdentityEvidence,
    SearchRequest,
    SearchResult,
)


REQUEST = SearchRequest(
    "STATUS стоматологія",
    "Запоріжжя",
    "вул. Поштова, 161/36",
    None,
    max_results=5,
    timeout_seconds=20,
)


def _response(*, status="completed", output_text=None, output=(), reason=None):
    details = types.SimpleNamespace(reason=reason) if reason is not None else None
    return types.SimpleNamespace(
        status=status,
        incomplete_details=details,
        output_text=output_text,
        output=list(output),
    )


def _tool_call(*urls, action_type="search", **action_fields):
    action = {
        "type": action_type,
        "sources": [{"url": url} for url in urls],
    }
    action.update(action_fields)
    return {
        "type": "web_search_call",
        "action": action,
    }


def _payload(*items):
    return json.dumps({"results": list(items)})


def _item(url, title="Title", snippet="Snippet", **identity):
    item = {
        "url": url,
        "title": title,
        "snippet": snippet,
        "name_matches": True,
        "city_matches": True,
        "address_matches": True,
        "phone_matches": False,
        "different_city_detected": False,
    }
    item.update(identity)
    return item


class _FakeResponses:
    def __init__(self, queue):
        self.queue = list(queue)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        value = self.queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class _FakeClient:
    def __init__(self, *queue):
        self.responses = _FakeResponses(queue)


def _provider(*responses, **settings):
    client = _FakeClient(*responses)
    configured = OpenAIWebSearchSettings(" top-secret ", **settings)
    return OpenAIWebSearchProvider(configured, client=client), client


def _sdk_error(name, *, status_code=None, code=None):
    class SafeError(Exception):
        def __str__(self):
            raise AssertionError("raw SDK error text must not be read")

    SafeError.__name__ = name
    error = SafeError("SECRET_RAW_ERROR sk-secret")
    if status_code is not None:
        error.status_code = status_code
    if code is not None:
        error.code = code
    error.request_id = "SECRET_REQUEST_ID"
    error.response = {"body": "SECRET_BODY"}
    return error


class SettingsTests(unittest.TestCase):
    def test_defaults_normalization_secret_repr_and_frozen(self):
        settings = OpenAIWebSearchSettings("  secret  ", country=" ua ")
        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(settings.model, "gpt-5.4-nano")
        self.assertEqual(settings.reasoning_effort, "low")
        self.assertEqual(settings.search_context_size, "low")
        self.assertEqual(settings.country, "UA")
        self.assertTrue(settings.external_web_access)
        self.assertEqual(settings.max_results, 5)
        self.assertEqual(settings.max_output_tokens, 1024)
        self.assertEqual(settings.timeout_seconds, 20.0)
        self.assertNotIn("secret", repr(settings))
        with self.assertRaises(FrozenInstanceError):
            settings.country = "US"  # type: ignore[misc]

    def test_allowed_reasoning_and_context_values(self):
        for effort in ("none", " low ", "MEDIUM", "high", "xhigh"):
            with self.subTest(effort=effort):
                self.assertEqual(
                    OpenAIWebSearchSettings("key", reasoning_effort=effort).reasoning_effort,
                    effort.strip().casefold(),
                )
        for context in ("low", " MEDIUM ", "HIGH"):
            with self.subTest(context=context):
                self.assertEqual(
                    OpenAIWebSearchSettings("key", search_context_size=context).search_context_size,
                    context.strip().casefold(),
                )

    def test_invalid_settings_are_strict(self):
        invalid = (
            ({"api_key": ""}, ValueError),
            ({"api_key": 1}, TypeError),
            ({"api_key": "key", "model": " "}, ValueError),
            ({"api_key": "key", "reasoning_effort": "minimal"}, ValueError),
            ({"api_key": "key", "search_context_size": "long"}, ValueError),
            ({"api_key": "key", "country": "U"}, ValueError),
            ({"api_key": "key", "country": "УА"}, ValueError),
            ({"api_key": "key", "external_web_access": 1}, TypeError),
            ({"api_key": "key", "max_results": True}, TypeError),
            ({"api_key": "key", "max_results": 0}, ValueError),
            ({"api_key": "key", "max_results": 11}, ValueError),
            ({"api_key": "key", "max_output_tokens": True}, TypeError),
            ({"api_key": "key", "max_output_tokens": 255}, ValueError),
            ({"api_key": "key", "max_output_tokens": 4097}, ValueError),
            ({"api_key": "key", "timeout_seconds": True}, TypeError),
            ({"api_key": "key", "timeout_seconds": math.inf}, ValueError),
            ({"api_key": "key", "timeout_seconds": 0}, ValueError),
            ({"api_key": "key", "timeout_seconds": 31}, ValueError),
        )
        for kwargs, error in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(error):
                OpenAIWebSearchSettings(**kwargs)


class InputBuilderTests(unittest.TestCase):
    @staticmethod
    def _variants(prompt):
        marker = "Suggested query variants:\n"
        return tuple(prompt.split(marker, 1)[1].splitlines())

    def test_deterministic_identity_and_optional_fields(self):
        first = build_openai_web_search_input(SearchRequest(
            " Business ", " City ", " Address ", "+380671234567", max_results=3
        ))
        second = build_openai_web_search_input(SearchRequest(
            "Business", "City", "Address", "+380671234567", max_results=3
        ))
        self.assertEqual(first, second)
        for expected in ("Business name: Business", "City: City", "Address: Address", "Phone: 380671234567", "Maximum candidates: 3"):
            self.assertIn(expected, first)
        self.assertIn("empty results array", first)
        self.assertIn("same-name business", first)
        self.assertIn("city must match exactly", first)
        self.assertIn("one search action only", first)
        self.assertIn("Do not use open_page or find_in_page", first)
        self.assertLessEqual(first.count('\n1. '), 1)
        self.assertLessEqual(first.count('\n2. '), 1)
        self.assertLessEqual(first.count('\n3. '), 1)
        self.assertNotIn("API key", first)
        self.assertNotIn("evidence JSON", first)

    def test_phone_and_address_query_variants_are_exact(self):
        prompt = build_openai_web_search_input(SearchRequest(
            "Business", "City", "Address", "+380671234567"
        ))
        self.assertEqual(self._variants(prompt), (
            '1. "Business" "380671234567"',
            '2. "Business" "City" "Address"',
            '3. "Business" "City"',
        ))

    def test_phone_only_query_variants_are_exact(self):
        prompt = build_openai_web_search_input(SearchRequest(
            "Business", "City", phone="+380671234567"
        ))
        self.assertEqual(self._variants(prompt), (
            '1. "Business" "380671234567"',
            '2. "380671234567" "City"',
            '3. "Business" "City"',
        ))

    def test_address_only_query_variants_are_preserved(self):
        prompt = build_openai_web_search_input(SearchRequest(
            "Business", "City", "Address"
        ))
        self.assertEqual(self._variants(prompt), (
            '1. "Business" "City" "Address"',
            '2. "Business" "City"',
            '3. "Address" "Business"',
        ))

    def test_name_and_city_are_the_only_variant_without_optional_identity(self):
        prompt = build_openai_web_search_input(SearchRequest("Business", "City"))
        self.assertEqual(
            self._variants(prompt),
            ('1. "Business" "City"',),
        )

    def test_variants_are_deduplicated_and_never_exceed_three(self):
        requests = (
            SearchRequest("Same", "Same"),
            SearchRequest("Same", "Same", "Same"),
            SearchRequest("1234567", "City", phone="1234567"),
            SearchRequest("Business", "City", "Address", "1234567"),
        )
        for request in requests:
            with self.subTest(request=request):
                variants = self._variants(build_openai_web_search_input(request))
                self.assertLessEqual(len(variants), 3)
                self.assertEqual(len(variants), len(set(variants)))

    def test_prompt_does_not_emit_unrequested_domain_literals(self):
        prompt = build_openai_web_search_input(SearchRequest(
            "Generic Business", "Generic City", "Generic Address", "1234567"
        ))
        domain_tokens = re.findall(
            r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b",
            prompt,
        )
        self.assertEqual(domain_tokens, [])

    def test_prompt_semantics_match_address_or_exact_phone_prefilter(self):
        prompt = build_openai_web_search_input(SearchRequest(
            "Business", "City", "Address", "1234567"
        ))
        for expected in (
            "Name and city must both match",
            "Different city evidence requires rejection",
            "exact address evidence is preferred",
            "Exact phone evidence may substitute for address evidence",
            "same named business in the same city",
            "address and phone corroboration are both absent or false",
            "Do not return a best guess",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, prompt)

    def test_controls_removed_and_long_values_bounded_without_losing_labels(self):
        prompt = build_openai_web_search_input(SearchRequest(
            "N\x00a\nme " * 200,
            "C\x1fity " * 200,
            "Address " * 300,
            "+380671234567",
        ))
        self.assertEqual(prompt, build_openai_web_search_input(SearchRequest(
            "N\x00a\nme " * 200,
            "C\x1fity " * 200,
            "Address " * 300,
            "+380671234567",
        )))
        self.assertLessEqual(len(prompt), 2000)
        self.assertNotIn("\x00", prompt)
        self.assertNotIn("\x1f", prompt)
        self.assertIn("Business name: N", prompt)
        self.assertIn("City: C", prompt)
        self.assertIn("Perform one search action only", prompt)
        self.assertIn("Suggested query variants:", prompt)
        self.assertLessEqual(
            sum(line[:3] in {"1. ", "2. ", "3. "} for line in prompt.splitlines()),
            3,
        )


class ProviderContractTests(unittest.IsolatedAsyncioTestCase):
    def test_default_client_is_official_async_sdk_with_retries_disabled(self):
        calls = []
        fake_client = _FakeClient()
        module = types.ModuleType("openai")

        def async_openai(**kwargs):
            calls.append(kwargs)
            return fake_client

        module.AsyncOpenAI = async_openai
        with patch.dict(sys.modules, {"openai": module}):
            OpenAIWebSearchProvider(OpenAIWebSearchSettings(" secret ", timeout_seconds=7))
        self.assertEqual(calls, [{
            "api_key": "secret",
            "timeout": 7.0,
            "max_retries": 0,
        }])

    async def test_exact_responses_request_contract(self):
        response = _response(
            output_text=_payload(),
            output=(_tool_call("https://example.com/"),),
        )
        provider, client = _provider(
            response,
            model="gpt-5.4-nano",
            reasoning_effort="low",
            search_context_size="low",
            country="UA",
            external_web_access=False,
            max_output_tokens=2048,
        )
        self.assertEqual(await provider.search(REQUEST), ())
        self.assertEqual(len(client.responses.calls), 1)
        call = client.responses.calls[0]
        self.assertEqual(set(call), {
            "model", "reasoning", "tools", "tool_choice", "max_tool_calls",
            "include", "input", "text", "max_output_tokens", "store",
        })
        self.assertEqual(call["model"], "gpt-5.4-nano")
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(call["tools"], [{
            "type": "web_search",
            "search_context_size": "low",
            "user_location": {
                "type": "approximate", "country": "UA", "city": "Запоріжжя",
            },
            "external_web_access": False,
        }])
        self.assertEqual(call["tool_choice"], "required")
        self.assertEqual(call["max_tool_calls"], 1)
        self.assertEqual(call["include"], ["web_search_call.action.sources"])
        self.assertEqual(call["text"], OPENAI_WEB_SEARCH_TEXT_FORMAT)
        self.assertEqual(call["max_output_tokens"], 2048)
        self.assertFalse(call["store"])
        serialized = json.dumps(call)
        for forbidden in ("web_search_preview", "image_search", "file_search", "mcp", "code_interpreter", "previous_response_id", "background"):
            self.assertNotIn(forbidden, serialized.casefold())

    async def test_success_normalizes_caps_deduplicates_and_tracks_telemetry(self):
        response = _response(
            output_text=_payload(
                _item("HTTPS://EXAMPLE.COM/?utm_source=model", " T " * 200, " S " * 700),
                _item("https://example.com/", "duplicate", "duplicate"),
                _item("https://second.example/path", "Second", "Two"),
            ),
            output=(_tool_call(
                "https://example.com/?utm_source=source",
                "https://second.example/path",
                "https://example.com/",
            ),),
        )
        provider, _ = _provider(response, max_results=2)
        results = await provider.search(SearchRequest("Name", "City", max_results=10))
        self.assertIsInstance(results, tuple)
        self.assertEqual([item.url for item in results], ["https://example.com/", "https://second.example/path"])
        self.assertEqual([item.rank for item in results], [1, 2])
        self.assertLessEqual(len(results[0].title), 300)
        self.assertLessEqual(len(results[0].snippet), 1000)
        self.assertEqual(provider.telemetry().requests_started, 1)
        self.assertEqual(provider.telemetry().requests_succeeded, 1)
        self.assertEqual(provider.telemetry().requests_failed, 0)
        self.assertEqual(provider.telemetry().tool_calls_seen, 1)
        self.assertEqual(provider.telemetry().search_actions_seen, 1)
        self.assertEqual(provider.telemetry().open_page_actions_seen, 0)
        self.assertEqual(provider.telemetry().find_in_page_actions_seen, 0)
        self.assertEqual(provider.telemetry().unknown_actions_seen, 0)
        self.assertEqual(provider.telemetry().sources_seen, 3)
        self.assertEqual(provider.telemetry().identity_candidates_rejected, 0)
        self.assertEqual(provider.telemetry().candidates_returned, 2)
        self.assertFalse(provider.telemetry().tool_call_limit_exceeded)
        self.assertIsNone(provider.telemetry().last_error_category)

    async def test_source_bound_result_maps_typed_identity_evidence_exactly(self):
        candidate = _item(
            "https://verified-clinic.example/",
            "Generic Clinic",
            "Generic source snippet",
            name_matches=True,
            city_matches=True,
            address_matches=False,
            phone_matches=True,
            different_city_detected=False,
        )
        response = _response(
            output_text=_payload(candidate),
            output=(_tool_call("https://verified-clinic.example/"),),
        )
        provider, client = _provider(response)
        results = await provider.search(SearchRequest(
            "Generic Clinic",
            "Generic City",
            "Generic Address",
            "+380671234567",
        ))

        self.assertEqual(len(results), 1)
        evidence = results[0].identity_evidence
        self.assertIs(type(evidence), SearchIdentityEvidence)
        self.assertEqual(
            (
                evidence.name_matches,
                evidence.city_matches,
                evidence.address_matches,
                evidence.phone_matches,
                evidence.different_city_detected,
                evidence.candidate_url_source_bound,
            ),
            (True, True, False, True, False, True),
        )
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(provider.telemetry().tool_calls_seen, 1)

    async def test_unsourced_and_unsafe_candidates_are_silently_rejected(self):
        response = _response(
            output_text=_payload(
                _item("https://verified.example/"),
                _item("https://hallucinated.example/"),
                _item("http://127.0.0.1/private"),
                _item("https://user:password@example.com/"),
            ),
            output=(_tool_call(
                "https://verified.example/",
                "http://127.0.0.1/private",
                "https://user:password@example.com/",
            ),),
        )
        provider, _ = _provider(response)
        results = await provider.search(REQUEST)
        self.assertEqual(tuple(item.url for item in results), ("https://verified.example/",))
        self.assertIsNotNone(results[0].identity_evidence)
        self.assertTrue(results[0].identity_evidence.candidate_url_source_bound)
        self.assertFalse(hasattr(provider.telemetry(), "source_urls"))

    async def test_social_and_directory_urls_reach_the_existing_matcher_layer(self):
        urls = ("https://instagram.com/status", "https://locator.ua/status")
        response = _response(
            output_text=_payload(*(_item(url) for url in urls)),
            output=(_tool_call(*urls),),
        )
        provider, _ = _provider(response)
        self.assertEqual(tuple(item.url for item in await provider.search(REQUEST)), urls)

    async def test_empty_results_are_valid_with_a_tool_call_and_empty_sources(self):
        provider, _ = _provider(_response(output_text=_payload(), output=(_tool_call(),)))
        self.assertEqual(await provider.search(REQUEST), ())
        self.assertEqual(provider.telemetry().requests_succeeded, 1)

    async def test_identity_prefilter_rejects_wrong_or_uncorroborated_candidates(self):
        cases = (
            ({"city_matches": False}, REQUEST),
            ({"different_city_detected": True}, REQUEST),
            ({"address_matches": False}, REQUEST),
            ({"name_matches": False}, REQUEST),
            ({"phone_matches": False}, SearchRequest("Name", "City", phone="+380671234567")),
        )
        for flags, request in cases:
            with self.subTest(flags=flags):
                item = _item("https://rejected.example/", **flags)
                response = _response(
                    output_text=_payload(item),
                    output=(_tool_call("https://rejected.example/"),),
                )
                provider, _ = _provider(response)
                results = await provider.search(request)
                self.assertEqual(results, ())
                telemetry = provider.telemetry()
                self.assertEqual(telemetry.identity_candidates_rejected, 1)
                self.assertFalse(hasattr(telemetry, "rejected_url"))

        rejected_without_sources, _ = _provider(_response(
            output_text=_payload(_item(
                "https://wrong-city.example/",
                city_matches=False,
                different_city_detected=True,
            )),
            output=(_tool_call(),),
        ))
        self.assertEqual(await rejected_without_sources.search(REQUEST), ())
        self.assertEqual(
            rejected_without_sources.telemetry().identity_candidates_rejected,
            1,
        )

    async def test_phone_corroborates_only_when_request_supplies_phone(self):
        with_phone = SearchRequest(
            "Name", "City", "Address", "+380671234567"
        )
        item = _item(
            "https://phone.example/",
            address_matches=False,
            phone_matches=True,
        )
        accepted, _ = _provider(_response(
            output_text=_payload(item),
            output=(_tool_call("https://phone.example/"),),
        ))
        self.assertEqual(
            tuple(result.url for result in await accepted.search(with_phone)),
            ("https://phone.example/",),
        )

        without_phone, _ = _provider(_response(
            output_text=_payload(item),
            output=(_tool_call("https://phone.example/"),),
        ))
        self.assertEqual(await without_phone.search(REQUEST), ())
        self.assertEqual(without_phone.telemetry().identity_candidates_rejected, 1)

    async def test_phone_corroboration_still_requires_name_and_correct_city(self):
        request = SearchRequest("Name", "City", "Address", "+380671234567")
        rejected_flags = (
            {"address_matches": False, "phone_matches": False},
            {
                "address_matches": False,
                "phone_matches": True,
                "city_matches": False,
            },
            {
                "address_matches": False,
                "phone_matches": True,
                "different_city_detected": True,
            },
            {
                "address_matches": False,
                "phone_matches": True,
                "name_matches": False,
            },
        )
        for flags in rejected_flags:
            with self.subTest(flags=flags):
                response = _response(
                    output_text=_payload(_item("https://rejected.example/", **flags)),
                    output=(_tool_call("https://rejected.example/"),),
                )
                provider, _ = _provider(response)
                self.assertEqual(await provider.search(request), ())
                self.assertEqual(
                    provider.telemetry().identity_candidates_rejected,
                    1,
                )

    async def test_action_accounting_and_only_search_sources_are_eligible(self):
        cases = (
            ("open_page", "open_page_actions_seen"),
            ("find_in_page", "find_in_page_actions_seen"),
            ("future_action", "unknown_actions_seen"),
            (None, "unknown_actions_seen"),
        )
        for action_type, counter in cases:
            with self.subTest(action_type=action_type):
                response = _response(
                    output_text=_payload(),
                    output=(_tool_call(
                        "https://not-a-search-source.example/",
                        action_type=action_type,
                        url="https://private-action-url.example/",
                        pattern="SECRET_PATTERN",
                    ),),
                )
                provider, _ = _provider(response)
                self.assertEqual(await provider.search(REQUEST), ())
                telemetry = provider.telemetry()
                self.assertEqual(getattr(telemetry, counter), 1)
                self.assertEqual(telemetry.sources_seen, 0)

        response = _response(
            output_text=_payload(_item("https://opened.example/")),
            output=(_tool_call(
                "https://opened.example/",
                action_type="open_page",
                url="https://opened.example/",
            ),),
        )
        provider, _ = _provider(response)
        with self.assertRaisesRegex(SearchProviderError, "unverified candidates"):
            await provider.search(REQUEST)

    async def test_multiple_tool_call_items_fail_closed_with_safe_telemetry(self):
        response = _response(
            output_text=_payload(_item("https://verified.example/")),
            output=(
                _tool_call("https://verified.example/"),
                _tool_call(action_type="open_page", url="https://verified.example/"),
            ),
        )
        provider, _ = _provider(response)
        with self.assertRaisesRegex(
            SearchProviderError,
            "OpenAI web search exceeded tool call limit",
        ):
            await provider.search(REQUEST)
        telemetry = provider.telemetry()
        self.assertEqual(telemetry.requests_started, 1)
        self.assertEqual(telemetry.requests_succeeded, 0)
        self.assertEqual(telemetry.requests_failed, 1)
        self.assertEqual(telemetry.tool_calls_seen, 2)
        self.assertEqual(telemetry.search_actions_seen, 1)
        self.assertEqual(telemetry.open_page_actions_seen, 1)
        self.assertTrue(telemetry.tool_call_limit_exceeded)
        self.assertEqual(telemetry.last_error_category, "tool_call_limit")
        self.assertEqual(telemetry.candidates_returned, 0)

    async def test_no_tool_call_and_unverified_claims_are_errors(self):
        cases = (
            (_response(output_text=_payload(), output=()), "tool was not called"),
            (_response(output_text=_payload(_item("https://example.com")), output=(_tool_call(),)), "unverified candidates"),
        )
        for response, message in cases:
            with self.subTest(message=message):
                provider, _ = _provider(response)
                with self.assertRaisesRegex(SearchProviderError, message):
                    await provider.search(REQUEST)
                self.assertEqual(provider.telemetry().requests_failed, 1)

    async def test_empty_and_malformed_structured_output_are_errors(self):
        malformed = (
            (" ", "returned empty output"),
            ("not-json", "invalid OpenAI"),
            (json.dumps([]), "invalid OpenAI"),
            (json.dumps({"results": {}, "extra": 1}), "invalid OpenAI"),
            (_payload({"url": "https://example.com", "title": "missing"}), "invalid OpenAI"),
            (_payload({"url": 1, "title": "x", "snippet": "y"}), "invalid OpenAI"),
            (_payload(_item("https://example.com", city_matches=1)), "invalid OpenAI"),
            (_payload({**_item("https://example.com"), "extra": False}), "invalid OpenAI"),
        )
        for output_text, message in malformed:
            with self.subTest(output_text=output_text):
                provider, _ = _provider(_response(output_text=output_text, output=(_tool_call(),)))
                with self.assertRaisesRegex(SearchProviderError, message):
                    await provider.search(REQUEST)

    async def test_response_states_and_refusal(self):
        cases = (
            (_response(status="incomplete", reason="max_output_tokens"), "output limit reached"),
            (_response(status="incomplete", reason="content_filter"), "response incomplete"),
            (_response(status="failed"), "response failed"),
            (_response(output_text=_payload(), output=({"type": "refusal"},)), "refused request"),
            (_response(output_text=_payload(), output=({"type": "message", "content": [{"type": "refusal", "refusal": "SECRET"}]},)), "refused request"),
        )
        for response, message in cases:
            with self.subTest(message=message):
                provider, _ = _provider(response)
                with self.assertRaisesRegex(SearchProviderError, message):
                    await provider.search(REQUEST)


class SDKErrorMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_sdk_error_categories_have_safe_types_and_messages(self):
        cases = (
            (_sdk_error("APITimeoutError"), ProviderTimeout, "OpenAI web search timed out"),
            (_sdk_error("RateLimitError", status_code=429), ProviderRateLimited, "OpenAI web search rate limited"),
            (_sdk_error("AuthenticationError", status_code=401), ProviderAuthError, "OpenAI web search authentication failed"),
            (_sdk_error("PermissionDeniedError", status_code=403), ProviderAuthError, "OpenAI web search authentication failed"),
            (_sdk_error("RateLimitError", status_code=429, code="insufficient_quota"), ProviderUnavailable, "OpenAI web search quota unavailable"),
            (_sdk_error("NotFoundError", status_code=404), ProviderUnavailable, "OpenAI web search model unavailable"),
            (_sdk_error("BadRequestError", status_code=400), ProviderUnavailable, "OpenAI web search configuration rejected"),
            (_sdk_error("APIStatusError", status_code=422), ProviderUnavailable, "OpenAI web search configuration rejected"),
            (_sdk_error("APIConnectionError"), SearchProviderError, "OpenAI web search connection failed"),
            (_sdk_error("InternalServerError", status_code=500), SearchProviderError, "OpenAI web search service error"),
            (_sdk_error("APIError"), SearchProviderError, "OpenAI web search request failed"),
        )
        for error, error_type, message in cases:
            with self.subTest(message=message):
                provider, client = _provider(error)
                with self.assertRaises(error_type) as raised:
                    await provider.search(REQUEST)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(len(client.responses.calls), 1)
                self.assertEqual(provider.telemetry().requests_failed, 1)
                self.assertNotIn("SECRET", message)


class RuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, **overrides):
        values = {
            "WEBSITE_SEARCH_PROVIDER": "none",
            "BRAVE_SEARCH_API_KEY": "",
            "BRAVE_SEARCH_COUNTRY": "UA",
            "BRAVE_SEARCH_LANGUAGE": "",
            "BRAVE_SEARCH_UI_LANGUAGE": "uk-UA",
            "BRAVE_SEARCH_SAFESEARCH": "moderate",
            "BRAVE_SEARCH_MAX_RESULTS": 5,
            "BRAVE_SEARCH_TIMEOUT_SECONDS": 10.0,
            "OPENAI_API_KEY": "",
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
            "OPENAI_WEB_SEARCH_REASONING_EFFORT": "low",
            "OPENAI_WEB_SEARCH_CONTEXT_SIZE": "low",
            "OPENAI_WEB_SEARCH_COUNTRY": "UA",
            "OPENAI_WEB_SEARCH_EXTERNAL_ACCESS": True,
            "OPENAI_WEB_SEARCH_MAX_RESULTS": 5,
            "OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS": 1024,
            "OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": 20.0,
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": 0,
        }
        values.update(overrides)
        return patch.multiple(runtime.config, **values)

    async def test_none_brave_and_openai_selection_have_no_fallback_chain(self):
        with self._config():
            self.assertIsNone(runtime.build_configured_search_provider())
        brave = types.SimpleNamespace(search=lambda request: None)
        with self._config(WEBSITE_SEARCH_PROVIDER="brave", BRAVE_SEARCH_API_KEY="brave", MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK=1), patch.object(runtime, "BraveSearchProvider", return_value=brave) as brave_constructor, patch.object(runtime, "OpenAIWebSearchProvider") as openai_constructor:
            selected = runtime.build_configured_search_provider()
        self.assertIsInstance(selected, runtime.BudgetedSearchProvider)
        brave_constructor.assert_called_once()
        openai_constructor.assert_not_called()

    def test_openai_missing_key_and_zero_budget_are_unavailable(self):
        with self._config(WEBSITE_SEARCH_PROVIDER="openai", MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK=1):
            self.assertIsInstance(runtime.build_configured_search_provider(), runtime.UnavailableSearchProvider)
        with self._config(WEBSITE_SEARCH_PROVIDER="openai", OPENAI_API_KEY="secret"):
            self.assertIsInstance(runtime.build_configured_search_provider(), runtime.UnavailableSearchProvider)

    async def test_configured_openai_maps_settings_and_shared_budget_once(self):
        response = _response(output_text=_payload(), output=(_tool_call(),))
        underlying, client = _provider(response)
        with self._config(WEBSITE_SEARCH_PROVIDER="openai", OPENAI_API_KEY="private-key", OPENAI_WEB_SEARCH_MODEL="lookup-model", OPENAI_WEB_SEARCH_REASONING_EFFORT="medium", OPENAI_WEB_SEARCH_CONTEXT_SIZE="high", OPENAI_WEB_SEARCH_COUNTRY="PL", OPENAI_WEB_SEARCH_EXTERNAL_ACCESS=False, OPENAI_WEB_SEARCH_MAX_RESULTS=7, OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS=2048, OPENAI_WEB_SEARCH_TIMEOUT_SECONDS=4.5, MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK=1), patch.object(runtime, "OpenAIWebSearchProvider", return_value=underlying) as constructor, patch.object(runtime, "BraveSearchProvider") as brave_constructor:
            selected = runtime.build_configured_search_provider()
        settings = constructor.call_args.args[0]
        self.assertEqual((settings.model, settings.reasoning_effort, settings.search_context_size), ("lookup-model", "medium", "high"))
        self.assertEqual((settings.country, settings.external_web_access), ("PL", False))
        self.assertEqual((settings.max_results, settings.max_output_tokens, settings.timeout_seconds), (7, 2048, 4.5))
        self.assertNotIn("private-key", repr(settings))
        brave_constructor.assert_not_called()
        self.assertEqual(await selected.search(REQUEST), ())
        with self.assertRaises(ProviderUnavailable):
            await selected.search(REQUEST)
        self.assertEqual(len(client.responses.calls), 1)
        self.assertEqual(runtime.search_budget_snapshot(selected).used_requests, 1)
        self.assertEqual(runtime.openai_web_search_telemetry_snapshot(selected).requests_started, 1)
        self.assertIsNone(runtime.brave_telemetry_snapshot(selected))


class ConfigSubprocessTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    NAMES = (
        "WEBSITE_SEARCH_PROVIDER", "OPENAI_API_KEY", "OPENAI_WEB_SEARCH_MODEL",
        "OPENAI_WEB_SEARCH_REASONING_EFFORT", "OPENAI_WEB_SEARCH_CONTEXT_SIZE",
        "OPENAI_WEB_SEARCH_COUNTRY", "OPENAI_WEB_SEARCH_MAX_RESULTS",
        "OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS", "OPENAI_WEB_SEARCH_TIMEOUT_SECONDS",
        "OPENAI_WEB_SEARCH_EXTERNAL_ACCESS", "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK",
    )

    def _run(self, **overrides):
        environment = os.environ.copy()
        for name in self.NAMES:
            environment.pop(name, None)
        environment.update({name: str(value) for name, value in overrides.items()})
        environment["PYTHONPATH"] = str(self.ROOT)
        code = """
import json, sys, types
dotenv = types.ModuleType('dotenv')
dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules['dotenv'] = dotenv
import config
print(json.dumps({
  'provider': config.WEBSITE_SEARCH_PROVIDER,
  'model': config.OPENAI_WEB_SEARCH_MODEL,
  'reasoning': config.OPENAI_WEB_SEARCH_REASONING_EFFORT,
  'context': config.OPENAI_WEB_SEARCH_CONTEXT_SIZE,
  'country': config.OPENAI_WEB_SEARCH_COUNTRY,
  'results': config.OPENAI_WEB_SEARCH_MAX_RESULTS,
  'tokens': config.OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS,
  'timeout': config.OPENAI_WEB_SEARCH_TIMEOUT_SECONDS,
  'external': config.OPENAI_WEB_SEARCH_EXTERNAL_ACCESS,
  'budget': config.MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK,
}))
"""
        with tempfile.TemporaryDirectory(prefix="lidogenerator-openai-search-config-") as temporary:
            return subprocess.run([sys.executable, "-c", code], cwd=temporary, env=environment, text=True, capture_output=True, timeout=30)

    def test_defaults_are_paid_call_safe(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "provider": "none", "model": "gpt-5.4-nano", "reasoning": "low",
            "context": "low", "country": "UA", "results": 5, "tokens": 1024,
            "timeout": 20.0, "external": True, "budget": 0,
        })

    def test_openai_values_normalize_and_invalid_values_fail(self):
        valid = self._run(WEBSITE_SEARCH_PROVIDER=" OPENAI ", OPENAI_WEB_SEARCH_REASONING_EFFORT=" XHIGH ", OPENAI_WEB_SEARCH_CONTEXT_SIZE=" MEDIUM ", OPENAI_WEB_SEARCH_COUNTRY=" pl ", OPENAI_WEB_SEARCH_EXTERNAL_ACCESS=" FALSE ")
        self.assertEqual(valid.returncode, 0, valid.stderr)
        data = json.loads(valid.stdout)
        self.assertEqual((data["provider"], data["reasoning"], data["context"], data["country"], data["external"]), ("openai", "xhigh", "medium", "PL", False))
        invalid = (
            {"WEBSITE_SEARCH_PROVIDER": "other"},
            {"OPENAI_WEB_SEARCH_MODEL": " "},
            {"OPENAI_WEB_SEARCH_REASONING_EFFORT": "minimal"},
            {"OPENAI_WEB_SEARCH_CONTEXT_SIZE": "long"},
            {"OPENAI_WEB_SEARCH_COUNTRY": "УА"},
            {"OPENAI_WEB_SEARCH_MAX_RESULTS": "true"},
            {"OPENAI_WEB_SEARCH_MAX_RESULTS": 0},
            {"OPENAI_WEB_SEARCH_MAX_RESULTS": 11},
            {"OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS": 255},
            {"OPENAI_WEB_SEARCH_MAX_OUTPUT_TOKENS": 4097},
            {"OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": "nan"},
            {"OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": 0},
            {"OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": 31},
            {"OPENAI_WEB_SEARCH_EXTERNAL_ACCESS": "yes"},
        )
        for values in invalid:
            with self.subTest(values=values):
                result = self._run(OPENAI_API_KEY="do-not-print", **values)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("do-not-print", result.stdout + result.stderr)


class ValidationScriptTests(unittest.IsolatedAsyncioTestCase):
    def test_utf8_console_configuration_uses_safe_reconfigure(self):
        class Stream:
            def __init__(self):
                self.calls = []

            def reconfigure(self, **kwargs):
                self.calls.append(kwargs)

        stdout = Stream()
        stderr = Stream()
        fake_sys = types.SimpleNamespace(stdout=stdout, stderr=stderr)
        with patch.object(validate_openai_web_search, "sys", fake_sys):
            validate_openai_web_search._configure_utf8_console()
        self.assertEqual(stdout.calls, [{"encoding": "utf-8", "errors": "replace"}])
        self.assertEqual(stderr.calls, [{"encoding": "utf-8", "errors": "replace"}])

    async def test_no_gate_and_bad_environment_make_no_request(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runtime, "build_configured_search_provider") as factory:
            output = io.StringIO()
            with redirect_stdout(output):
                result = await validate_openai_web_search._validate()
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().strip(), "LIVE_OPENAI_WEB_SEARCH_VALIDATION_NOT_RUN_NO_EXPLICIT_OPT_IN")
        factory.assert_not_called()

        base = {
            validate_openai_web_search.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "secret",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "1",
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
        }
        cases = (
            ({"OPENAI_API_KEY": ""}, "MISSING_API_KEY"),
            ({"MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "2"}, "BUDGET_MUST_EQUAL_ONE"),
            ({"OPENAI_WEB_SEARCH_MODEL": ""}, "MISSING_MODEL"),
        )
        for overrides, marker in cases:
            environment = dict(base)
            environment.update(overrides)
            with self.subTest(marker=marker), patch.dict(os.environ, environment, clear=True):
                output = io.StringIO()
                with redirect_stdout(output):
                    result = await validate_openai_web_search._validate()
                self.assertEqual(result, 2)
                self.assertIn(marker, output.getvalue())

    async def test_explicit_gate_uses_exactly_one_request_and_safe_output(self):
        response = _response(
            output_text=_payload(_item(
                "https://status-dent.zp.ua/?utm_source=test",
                "STATUS стоматологія " + "x" * 150,
                "SECRET_SNIPPET",
            )),
            output=(_tool_call("https://status-dent.zp.ua/"),),
        )
        underlying, client = _provider(response)
        budgeted = runtime.BudgetedSearchProvider(underlying, 1)
        environment = {
            validate_openai_web_search.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "1",
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
        }
        config_values = {
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
            "OPENAI_WEB_SEARCH_REASONING_EFFORT": "low",
            "OPENAI_WEB_SEARCH_CONTEXT_SIZE": "low",
            "OPENAI_WEB_SEARCH_EXTERNAL_ACCESS": True,
            "OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": 20.0,
        }
        with patch.dict(os.environ, environment, clear=True), patch.multiple(runtime.config, **config_values), patch.object(runtime, "build_configured_search_provider", return_value=budgeted):
            output = io.StringIO()
            with redirect_stdout(output):
                result = await validate_openai_web_search._validate()
        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertEqual(len(client.responses.calls), 1)
        for expected in (
            "provider=openai",
            "request_budget_used=1",
            "requests_started=1",
            "tool_calls_seen=1",
            "search_actions_seen=1",
            "open_page_actions_seen=0",
            "find_in_page_actions_seen=0",
            "unknown_actions_seen=0",
            "identity_candidates_rejected=0",
            "tool_call_limit_exceeded=no",
            "resolution_status=found_official",
            "resolved_domain=",
            "wrong_same_name_domain_returned=no",
            "wrong_same_name_domain_promoted=no",
            "final_result=success_expected_domain_found",
        ):
            self.assertIn(expected, text)
        for private in (
            "SECRET_API_KEY", "SECRET_SNIPPET", "вул. Поштова", "https://",
            "STATUS стоматологія", "STATUS СЃС‚РѕРјР°С‚РѕР»РѕРі",
        ):
            self.assertNotIn(private, text)

    async def test_structured_wrong_city_candidate_is_filtered_and_safe(self):
        response = _response(
            output_text=_payload(_item(
                "https://status-dental-clinic.com.ua/",
                "STATUS стоматологія — Київ",
                "Стоматологічна клініка на вул. Софії Русової, 3",
                city_matches=False,
                different_city_detected=True,
            )),
            output=(_tool_call("https://status-dental-clinic.com.ua/"),),
        )
        underlying, _ = _provider(response)
        budgeted = runtime.BudgetedSearchProvider(underlying, 1)
        environment = {
            validate_openai_web_search.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "1",
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
        }
        config_values = {
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
            "OPENAI_WEB_SEARCH_REASONING_EFFORT": "low",
            "OPENAI_WEB_SEARCH_CONTEXT_SIZE": "low",
            "OPENAI_WEB_SEARCH_EXTERNAL_ACCESS": True,
            "OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": 20.0,
        }
        with patch.dict(os.environ, environment, clear=True), patch.multiple(
            runtime.config, **config_values
        ), patch.object(
            runtime, "build_configured_search_provider", return_value=budgeted
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = await validate_openai_web_search._validate()
        text = output.getvalue()
        self.assertEqual(result, 0)
        self.assertIsInstance(text, str)
        self.assertEqual(text.encode("utf-8").decode("utf-8"), text)
        self.assertIn("resolution_status=not_found", text)
        self.assertIn("resolved_domain=\n", text)
        self.assertIn("wrong_same_name_domain_returned=no", text)
        self.assertIn("wrong_same_name_domain_promoted=no", text)
        self.assertIn("final_result=technical_success_safe_no_verified_match", text)
        self.assertNotIn("https://", text)
        self.assertNotIn("СЃС‚РѕРј", text)

    async def test_tool_limit_failure_has_dedicated_final_result(self):
        response = _response(
            output_text=_payload(),
            output=(_tool_call(), _tool_call(action_type="open_page")),
        )
        underlying, _ = _provider(response)
        budgeted = runtime.BudgetedSearchProvider(underlying, 1)
        environment = {
            validate_openai_web_search.LIVE_GATE: "1",
            "WEBSITE_SEARCH_PROVIDER": "openai",
            "OPENAI_API_KEY": "SECRET_API_KEY",
            "MAX_WEBSITE_SEARCH_REQUESTS_PER_TASK": "1",
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
        }
        config_values = {
            "OPENAI_WEB_SEARCH_MODEL": "gpt-5.4-nano",
            "OPENAI_WEB_SEARCH_REASONING_EFFORT": "low",
            "OPENAI_WEB_SEARCH_CONTEXT_SIZE": "low",
            "OPENAI_WEB_SEARCH_EXTERNAL_ACCESS": True,
            "OPENAI_WEB_SEARCH_TIMEOUT_SECONDS": 20.0,
        }
        with patch.dict(os.environ, environment, clear=True), patch.multiple(
            runtime.config, **config_values
        ), patch.object(
            runtime, "build_configured_search_provider", return_value=budgeted
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                result = await validate_openai_web_search._validate()
        text = output.getvalue()
        self.assertEqual(result, 1)
        self.assertIn("requests_failed=1", text)
        self.assertIn("tool_calls_seen=2", text)
        self.assertIn("tool_call_limit_exceeded=yes", text)
        self.assertIn("final_result=tool_call_limit_exceeded", text)


class SecurityBoundaryTests(unittest.TestCase):
    def test_schema_is_conservative_and_source_avoids_sensitive_persistence(self):
        self.assertEqual(OPENAI_WEB_SEARCH_SCHEMA["required"], ["results"])
        self.assertFalse(OPENAI_WEB_SEARCH_SCHEMA["additionalProperties"])
        item_schema = OPENAI_WEB_SEARCH_SCHEMA["properties"]["results"]["items"]
        required = [
            "url", "title", "snippet", "name_matches", "city_matches",
            "address_matches", "phone_matches", "different_city_detected",
        ]
        self.assertEqual(item_schema["required"], required)
        self.assertEqual(set(item_schema["properties"]), set(required))
        for name in required[3:]:
            self.assertEqual(item_schema["properties"][name], {"type": "boolean"})
        self.assertFalse(item_schema["additionalProperties"])
        forbidden_schema_keys = {"minimum", "maximum", "maxItems", "minItems", "maxLength", "format", "pattern", "default"}

        def keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key
                    yield from keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from keys(nested)

        self.assertTrue(forbidden_schema_keys.isdisjoint(keys(OPENAI_WEB_SEARCH_SCHEMA)))
        source = inspect.getsource(provider_module)
        folded = source.casefold()
        self.assertNotIn("str(exc", folded)
        self.assertNotIn("repr(exc", folded)
        self.assertNotIn("print(", folded)
        self.assertNotIn("web_search_preview", folded)
        for field_name in ("query", "prompt", "business_name", "address", "phone", "response_id", "request_id", "usage", "raw_output", "source_urls", "snippets"):
            self.assertNotIn(field_name, provider_module.OpenAIWebSearchTelemetry.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()

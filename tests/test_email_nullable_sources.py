"""Nullable provider lists and controlled malformed shapes, with no live calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from email_source_adapter import (
    EmailSourceAdapter,
    SourceAdapterError,
    search_sources,
    select_search_sources,
)


def web(sources=None, status="completed", **action):
    return {
        "type": "web_search_call",
        "status": status,
        "action": {"sources": sources, **action},
    }


def message(urls):
    return {
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Untrusted source text",
                "annotations": [
                    {"type": "url_citation", "url": url, "title": "Cited"}
                    for url in urls
                ],
            }
        ],
    }


def response(output, status="completed"):
    return {"status": status, "output": output}


class NullableSourceLists(unittest.IsolatedAsyncioTestCase):
    def test_completed_null_without_citations_is_typed_empty_not_email_absence(self):
        self.assertEqual([], select_search_sources(response([web(None)])))

    def test_missing_null_empty_consulted_sources_preserve_citations(self):
        target = "https://business.test/contact"
        for action in [{}, {"sources": None}, {"sources": []}]:
            with self.subTest(action=action):
                call = web()
                call["action"] = action
                self.assertEqual(
                    [{"url": target, "title": "Cited"}],
                    select_search_sources(response([call, message([target])])),
                )

    def test_null_does_not_change_citation_order_alias_dedup_or_cap(self):
        urls = [f"https://business{i}.test/contact" for i in range(10)]
        cited = [urls[0] + "?utm_source=openai", urls[0], *urls[1:]]
        result = select_search_sources(response([web(None), message(cited)]))
        self.assertEqual([cited[0], *urls[1:8]], [item["url"] for item in result])

    def test_unfinished_null_is_ignored_with_one_completed_call(self):
        url = "https://business.test/"
        self.assertEqual(
            [{"url": url, "title": "Cited"}],
            select_search_sources(
                response([web([], "completed"), web(None, "searching"), message([url])])
            ),
        )

    def test_malformed_consulted_list_and_items_fail_controlled(self):
        for sources in ["", "bad", {}, 0, [None], ["https://business.test/"], [[]]]:
            with (
                self.subTest(sources=sources),
                self.assertRaises(SourceAdapterError) as caught,
            ):
                select_search_sources(response([web(sources)]))
            self.assertEqual("SOURCE_PROVIDER_ERROR", caught.exception.code)

    def test_malformed_completed_action_is_controlled(self):
        for action in ["bad", [], 0]:
            call = web()
            call["action"] = action
            with (
                self.subTest(action=action),
                self.assertRaises(SourceAdapterError) as caught,
            ):
                select_search_sources(response([call]))
            self.assertEqual("SOURCE_PROVIDER_ERROR", caught.exception.code)

    def test_malformed_output_and_completed_message_items_are_controlled(self):
        values = [
            response({}),
            response([None]),
            response([web(), {**message([]), "content": [None]}]),
            response([web(), {**message([]), "content": "bad"}]),
        ]
        bad_annotations = message([])
        bad_annotations["content"][0]["annotations"] = [None]
        values.append(response([web(), bad_annotations]))
        for raw in values:
            with self.subTest(raw=raw), self.assertRaises(SourceAdapterError) as caught:
                select_search_sources(raw)
            self.assertEqual("SOURCE_PROVIDER_ERROR", caught.exception.code)

    def test_no_multiple_completed_and_incomplete_rules_are_unchanged(self):
        for calls in [[], [web(None, "searching")], [web(None), web(None)]]:
            with (
                self.subTest(calls=calls),
                self.assertRaises(SourceAdapterError) as caught,
            ):
                select_search_sources(response(calls))
            self.assertEqual(
                "SOURCE_PROVIDER_NO_COMPLETED_WEB_CALL", caught.exception.code
            )
        for status in ["incomplete", "failed", "cancelled", None]:
            with (
                self.subTest(status=status),
                self.assertRaises(SourceAdapterError) as caught,
            ):
                select_search_sources(response([web(None)], status))
            self.assertEqual(
                "SOURCE_PROVIDER_INCOMPLETE_RESPONSE", caught.exception.code
            )

    def test_url_guards_remain_in_force_for_null_consulted_list(self):
        urls = [
            "file:///etc/passwd",
            "http://127.0.0.1/",
            "https://t.me/example",
            "https://user:pass@business.test/",
            "https://business.test/?access_token=x",
            "https://business.test/contact",
        ]
        self.assertEqual(
            [{"url": urls[-1], "title": "Cited"}],
            select_search_sources(response([web(None), message(urls)])),
        )

    async def test_full_search_keeps_one_sdk_call_settings_and_usage(self):
        raw = response([web(None), message(["https://business.test/contact"])])
        sdk_response = SimpleNamespace(
            model_dump=lambda: raw,
            usage=SimpleNamespace(input_tokens=23, output_tokens=7),
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.create.return_value = sdk_response
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "offline-not-a-key"),
            patch("openai.AsyncOpenAI", return_value=client) as factory,
        ):
            result = await search_sources(
                {"name": "Offline Business", "city": "Kyiv", "mode": "NAME_CITY"}
            )
        factory.assert_called_once_with(
            api_key="offline-not-a-key", timeout=22, max_retries=0
        )
        client.responses.create.assert_awaited_once()
        self.assertEqual(
            {"calls": 1, "inputTokens": 23, "outputTokens": 7}, result["usage"]
        )
        self.assertEqual(
            ["https://business.test/contact"], [s["url"] for s in result["sources"]]
        )

    async def test_malformed_list_uses_existing_http_error_boundary(self):
        raw = response([web(["not-a-mapping"])])
        sdk_response = SimpleNamespace(
            model_dump=lambda: raw,
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.create.return_value = sdk_response
        request = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "name": "Offline Business",
                    "city": "Kyiv",
                    "mode": "NAME_CITY",
                    "operationKey": "00000000-0000-4000-8000-000000000001",
                }
            ),
            transport=SimpleNamespace(is_closing=lambda: False),
        )
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "offline-not-a-key"),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            reply = await EmailSourceAdapter(SimpleNamespace(active_count=0)).handle(
                request, "search"
            )
        self.assertEqual(503, reply.status)
        self.assertIn('"code": "SOURCE_PROVIDER_ERROR"', reply.text)


if __name__ == "__main__":
    unittest.main()

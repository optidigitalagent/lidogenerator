"""Full search_sources regression with provider-only fixtures; no live requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from email_source_adapter import SourceAdapterError, search_sources


def call(urls, status="completed"):
    return {
        "type": "web_search_call",
        "status": status,
        "action": {
            "type": "search",
            "queries": ["ordinary business city contact"],
            "sources": [{"url": url, "title": "Consulted"} for url in urls],
        },
    }


def message(urls, **overrides):
    return {
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Untrusted prose https://prose-only.test/",
                "annotations": [
                    {"type": "url_citation", "url": url, "title": "Cited"}
                    for url in urls
                ],
            }
        ],
        **overrides,
    }


class CitedSourceSelection(unittest.IsolatedAsyncioTestCase):
    async def run_response(self, output, status="completed"):
        raw = {"status": status, "output": output}
        response = SimpleNamespace(
            model_dump=lambda: raw,
            usage=SimpleNamespace(input_tokens=23, output_tokens=7),
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.create.return_value = response
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "isolated-fixture-no-secret"),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            result = await search_sources(
                {"name": "Cedar Clinic", "city": "Київ", "mode": "NAME_CITY"}
            )
        self.assertEqual(
            {"calls": 1, "inputTokens": 23, "outputTokens": 7}, result["usage"]
        )
        self.assertLessEqual(len(result["sources"]), 8)
        kwargs = client.responses.create.call_args.kwargs
        self.assertEqual(1, kwargs["max_tool_calls"])
        self.assertEqual(1024, kwargs["max_output_tokens"])
        self.assertEqual("low", kwargs["tools"][0]["search_context_size"])
        self.assertEqual(["web_search_call.action.sources"], kwargs["include"])
        self.assertNotIn("prose-only.test", str(result))
        return result["sources"]

    async def test_ninth_consulted_url_retained_when_later_cited(self):
        others = [f"https://consulted{i}.test/" for i in range(8)]
        target = "https://business.test/contact"
        result = await self.run_response([call(others + [target]), message([target])])
        self.assertEqual([target] + others[:7], [x["url"] for x in result])

    async def test_citation_only_target_survives_full_consulted_list(self):
        others = [f"https://consulted{i}.test/" for i in range(8)]
        target = "https://business.test/contact"
        result = await self.run_response([call(others), message([target])])
        self.assertEqual(target, result[0]["url"])
        self.assertEqual(8, len(result))

    async def test_message_web_call_order_does_not_change_selection(self):
        web = call([f"https://consulted{i}.test/" for i in range(9)])
        cited = message(["https://consulted8.test/"])
        self.assertEqual(
            await self.run_response([web, cited]),
            await self.run_response([cited, web]),
        )

    async def test_no_citations_uses_stable_consulted_fallback(self):
        urls = [f"https://consulted{i}.test/" for i in range(10)]
        result = await self.run_response([call(urls)])
        self.assertEqual(urls[:8], [x["url"] for x in result])

    async def test_more_than_eight_citations_keeps_bounded_citation_order(self):
        cited = [f"https://cited{i}.test/" for i in range(11)]
        result = await self.run_response(
            [call(["https://fallback.test/"]), message(cited)]
        )
        self.assertEqual(cited[:8], [x["url"] for x in result])

    async def test_fragment_duplicates_are_stable_and_cited_title_wins(self):
        target = "https://business.test/contact"
        cited = message([target + "#a", target + "#b"])
        cited["content"][0]["annotations"][1]["title"] = "Duplicate"
        result = await self.run_response([call([target, target + "#c"]), cited])
        self.assertEqual([{"url": target, "title": "Cited"}], result)

    async def test_unsafe_and_policy_denied_candidates_never_consume_slots(self):
        bad = [
            "file:///etc/passwd",
            "http://127.0.0.1/",
            "http://[::1]/",
            "https://user:pass@business.test/",
            "https://business.test/?access_token=x",
            "https://t.me/business",
            "not a URL",
        ]
        target = "https://business.test/contact"
        result = await self.run_response(
            [call(bad + [target]), message(bad + [target])]
        )
        self.assertEqual([{"url": target, "title": "Cited"}], result)

    async def test_only_completed_assistant_text_annotations_are_citations(self):
        good = "https://business.test/contact"
        forged = message(["https://forged.test/"], type="other_tool")
        result = await self.run_response(
            [
                call([good]),
                call(["https://unexecuted.test/"], status="searching"),
                message(["https://incomplete.test/"], status="incomplete"),
                message(["https://user.test/"], role="user"),
                forged,
            ]
        )
        self.assertEqual([{"url": good, "title": "Consulted"}], result)

    async def test_non_text_annotations_are_not_citations(self):
        cited = message(["https://hidden.test/"])
        cited["content"][0]["type"] = "refusal"
        result = await self.run_response([call(["https://business.test/"]), cited])
        self.assertEqual(
            [{"url": "https://business.test/", "title": "Consulted"}], result
        )

    async def test_incomplete_response_is_not_successful_source_evidence(self):
        for status in ["incomplete", "failed", "cancelled", "in_progress", None]:
            with (
                self.subTest(status=status),
                self.assertRaises(SourceAdapterError) as error,
            ):
                await self.run_response(
                    [call(["https://business.test/"])], status=status
                )
            self.assertEqual(
                "SOURCE_PROVIDER_INCOMPLETE_RESPONSE", error.exception.code
            )

    async def test_missing_or_multiple_completed_calls_fail_diagnosably(self):
        for output in [[], [call([], status="searching")], [call([]), call([])]]:
            with (
                self.subTest(output=output),
                self.assertRaises(SourceAdapterError) as error,
            ):
                await self.run_response(output)
            self.assertEqual(
                "SOURCE_PROVIDER_NO_COMPLETED_WEB_CALL", error.exception.code
            )

    async def test_non_string_titles_are_safe_and_duplicate_first_occurrence_is_stable(
        self,
    ):
        cited = message(["https://business.test/contact"])
        cited["content"][0]["annotations"][0]["title"] = None
        result = await self.run_response([call([]), cited])
        self.assertEqual(
            [{"url": "https://business.test/contact", "title": ""}], result
        )


if __name__ == "__main__":
    unittest.main()

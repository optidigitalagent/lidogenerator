"""FRONTIER09 selector and HTTP-lifetime controls; no external/provider calls."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from email_source_adapter import (
    EmailSourceAdapter,
    SourceAdapterError,
    select_search_sources,
    source_url_key,
)
from tests.test_email_source_selection import call, message


class SourceAliasSelection(unittest.TestCase):
    def test_exact_decoded_tracking_pair_only_preserves_raw_functional_query(self):
        prefix = "https://clinic.test/contact"
        examples = {
            "?utm_source=openai": "",
            "?%75tm_source=%6fpenai": "",
            "?id=1&utm_source=openai&lang=uk": "?id=1&lang=uk",
            "?id=%2f&utm_source=openai&id=%2F": "?id=%2f&id=%2F",
            "?q=a+b&utm_source=openai&q=a%20b": "?q=a+b&q=a%20b",
            "?x=1&utm_source=openai&x=2": "?x=1&x=2",
            "?utm_source=openai&utm_source=openai": "",
            "?utm_source=openai&utm_source=google": "?utm_source=google",
            "?utm_source=openai&": "?",
            "?&utm_source=openai&x=1&": "?&x=1&",
            "?utm_source=OpenAI": "?utm_source=OpenAI",
            "?UTM_SOURCE=openai": "?UTM_SOURCE=openai",
            "?utm_source=openai+": "?utm_source=openai+",
            "?utm_source=openai%20": "?utm_source=openai%20",
            "?utm_source=openai%": "?utm_source=openai%",
            "?utm_source=%FF": "?utm_source=%FF",
            "?utm_source=%E0%A4%A": "?utm_source=%E0%A4%A",
            "?utm%ZZsource=openai&utm_source=openai": "?utm%ZZsource=openai",
            "?utm_source=openai=extra": "?utm_source=openai=extra",
            "?id=1;utm_source=openai": "?id=1;utm_source=openai",
            "?utm_medium=openai": "?utm_medium=openai",
            "?": "?",
        }
        for suffix, expected in examples.items():
            with self.subTest(suffix=suffix):
                self.assertEqual(prefix + expected, source_url_key(prefix + suffix))

    def test_different_functional_values_order_encoding_host_path_scheme_stay_distinct(
        self,
    ):
        urls = [
            "https://clinic.test/contact?id=1&lang=uk",
            "https://clinic.test/contact?id=2&lang=uk",
            "https://clinic.test/contact?lang=uk&id=1",
            "https://clinic.test/contact?id=1&lang=ru",
            "https://clinic.test/contact?id=%2f",
            "https://clinic.test/contact?id=%2F",
            "https://other.test/contact?id=1&lang=uk",
            "http://clinic.test/contact?id=1&lang=uk",
            "https://clinic.test/other?id=1&lang=uk",
        ]
        self.assertEqual(len(urls), len({source_url_key(url) for url in urls}))

    def test_tracking_aliases_do_not_consume_cap_and_first_citation_url_title_wins(
        self,
    ):
        urls = [f"https://clinic.test/contact?id={i}" for i in range(12)]
        tracked = urls[0] + "&utm_source=openai"
        cited = message([tracked, urls[0], urls[1] + "&utm_source=openai"])
        cited["content"][0]["annotations"][0]["title"] = "Original cited title"
        result = select_search_sources(
            {"status": "completed", "output": [call(urls), cited]}
        )
        self.assertEqual(8, len(result))
        self.assertEqual({"url": tracked, "title": "Original cited title"}, result[0])
        self.assertEqual(
            [tracked, urls[1] + "&utm_source=openai", *urls[2:8]],
            [entry["url"] for entry in result],
        )
        self.assertEqual(8, len({source_url_key(entry["url"]) for entry in result}))

    def test_validation_precedes_alias_dedupe_and_cannot_grant_access(self):
        invalid = [
            "http://127.0.0.1/contact?utm_source=openai",
            "https://user:secret@clinic.test/contact?utm_source=openai",
            "https://clinic.test/contact?api_key=secret&utm_source=openai",
            "https://t.me/clinic?utm_source=openai",
        ]
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(SourceAdapterError):
                source_url_key(url)
        valid = "https://clinic.test/contact?id=7&utm_source=openai"
        result = select_search_sources(
            {"status": "completed", "output": [call(invalid + [valid])]}
        )
        self.assertEqual([{"url": valid, "title": "Consulted"}], result)

    def test_incomplete_stays_rejected_even_with_usable_aliases(self):
        with self.assertRaises(SourceAdapterError) as caught:
            select_search_sources(
                {
                    "status": "incomplete",
                    "output": [call(["https://clinic.test/?utm_source=openai"])],
                }
            )
        self.assertEqual("SOURCE_PROVIDER_INCOMPLETE_RESPONSE", caught.exception.code)


class LateWaiterLifetime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.adapter = EmailSourceAdapter(SimpleNamespace(active_count=0))
        self.app = web.Application()
        self.requests = []

        async def route(request):
            self.requests.append(request)
            return await self.adapter.handle(request, "search")

        self.app.router.add_post("/search", route)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.payload = {
            "operationKey": "00000000-0000-4000-8000-000000000001",
            "name": "Fixture",
            "city": "Київ",
            "mode": "NAME_CITY",
        }

    async def asyncTearDown(self):
        await self.client.close()

    async def test_exact_late_http_waiter_receives_503_after_original_cleanup(self):
        started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def provider(body):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()

        try:
            with patch(
                "email_source_adapter.search_sources", side_effect=provider
            ) as search:
                first = asyncio.create_task(
                    self.client.post("/search", json=self.payload)
                )
                await started.wait()
                operation = self.adapter.inflight[self.payload["operationKey"]]
                self.assertEqual("running", operation.state)
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                await asyncio.wait_for(cleaning.wait(), 1)
                self.assertEqual("cancelling", operation.state)
                second = asyncio.create_task(
                    self.client.post("/search", json=self.payload)
                )
                async with asyncio.timeout(1):
                    while len(self.requests) < 2 or operation.waiters != 1:
                        await asyncio.sleep(0.005)
                self.assertFalse(self.requests[1].transport.is_closing())
                release.set()
                response = await asyncio.wait_for(second, 1)
                self.assertEqual(503, response.status)
                self.assertEqual(
                    "SOURCE_OPERATION_CANCELLED",
                    (await response.json())["error"]["code"],
                )
                async with asyncio.timeout(1):
                    while self.adapter.inflight:
                        await asyncio.sleep(0.005)
                self.assertEqual("completed", operation.state)
                self.assertEqual(1, search.await_count)
        finally:
            release.set()

    async def test_repeated_late_waiters_are_bounded_without_recancelling_cleanup(self):
        started, cleaning, release, cleanup_interrupted = (
            asyncio.Event() for _ in range(4)
        )

        async def provider(body):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cleanup_interrupted.set()
                    raise

        try:
            with patch(
                "email_source_adapter.search_sources", side_effect=provider
            ) as search:
                first = asyncio.create_task(
                    self.client.post("/search", json=self.payload)
                )
                await started.wait()
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                await asyncio.wait_for(cleaning.wait(), 1)
                operation = self.adapter.inflight[self.payload["operationKey"]]
                for _ in range(3):
                    response = await asyncio.wait_for(
                        self.client.post("/search", json=self.payload), 0.5
                    )
                    self.assertEqual(503, response.status)
                    self.assertEqual(
                        "SOURCE_OPERATION_CANCELLED",
                        (await response.json())["error"]["code"],
                    )
                    self.assertEqual("cancelling", operation.state)
                    self.assertIs(
                        self.adapter.inflight[self.payload["operationKey"]], operation
                    )
                    self.assertFalse(operation.task.done())
                    self.assertFalse(cleanup_interrupted.is_set())
                    self.assertEqual(1, search.await_count)
                release.set()
                async with asyncio.timeout(1):
                    while self.adapter.inflight:
                        await asyncio.sleep(0.005)
                self.assertEqual("completed", operation.state)
        finally:
            release.set()

    async def test_guard_excludes_semaphore_queue_wait_then_bounds_active_work(self):
        self.adapter.search_lock = asyncio.Semaphore(0)

        async def provider(body):
            await asyncio.sleep(0.02)
            return {"sources": [], "usage": {"calls": 1}}

        with (
            patch("email_source_adapter.SEARCH_TOTAL_DEADLINE", 0.04),
            patch(
                "email_source_adapter.search_sources", side_effect=provider
            ) as search,
        ):
            pending = asyncio.create_task(
                self.client.post("/search", json=self.payload)
            )
            await asyncio.sleep(0.08)
            self.assertFalse(pending.done())
            self.assertEqual(0, search.await_count)
            self.adapter.search_lock.release()
            response = await asyncio.wait_for(pending, 0.5)
            self.assertEqual(200, response.status)
            self.assertEqual(1, search.await_count)


if __name__ == "__main__":
    unittest.main()

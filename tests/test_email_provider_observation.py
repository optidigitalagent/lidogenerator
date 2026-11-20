import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from openai import APITimeoutError

from email_provider_observation import ProviderObservation
from email_source_adapter import EmailSourceAdapter, SourceAdapterError, search_sources


class ProviderTrace(unittest.IsolatedAsyncioTestCase):
    async def test_real_local_transport_reports_connect_send_then_read_deadline(self):
        observer = ProviderObservation()
        app = web.Application()

        async def slow(request):
            await asyncio.sleep(0.15)
            return web.json_response({"fixture": True})

        app.router.add_post("/", slow)
        server = TestServer(app)
        await server.start_server()
        try:
            async with observer.http_client() as client:
                with self.assertRaises(Exception) as caught:
                    await client.post(str(server.make_url("/")), timeout=0.04)
                self.assertEqual("ReadTimeout", type(caught.exception).__name__)
            observer.failure(caught.exception)
            events = [item["event"] for item in observer.events]
            self.assertIn("connection.connect_tcp.complete", events)
            self.assertIn("http11.send_request_body.complete", events)
            self.assertIn("http11.receive_response_headers.failed", events)
            self.assertEqual("ReadTimeout", observer.cause_classes[0])
        finally:
            await server.close()

    async def test_trace_retains_only_phase_timings_and_exception_classes(self):
        observer = ProviderObservation()
        request = httpx.Request(
            "POST",
            "https://secret.test/private?api_key=secret",
            headers={"Authorization": "Bearer private"},
            extensions={"timeout": {"connect": 22.0, "read": 22.0}},
        )
        await observer.on_request(request)
        trace = request.extensions["trace"]
        await trace("connection.connect_tcp.started", {"host": "private"})
        await trace("connection.connect_tcp.complete", {"return_value": "private"})
        await trace("http11.receive_response_headers.started", {})
        await trace(
            "http11.receive_response_headers.failed",
            {"exception": httpx.ReadTimeout("private exception")},
        )
        await trace("unexpected.private.started", {"private": "private"})
        await observer.on_response(
            httpx.Response(
                200, headers={"x-request-id": "req_safe-123", "set-cookie": "private"}
            )
        )
        snapshot = observer.snapshot()
        self.assertEqual(4, len(snapshot["events"]))
        self.assertEqual("ReadTimeout", snapshot["events"][-1]["exceptionClass"])
        self.assertEqual("req_safe-123", snapshot["requestId"])
        self.assertNotIn("private", json.dumps(snapshot))
        self.assertEqual(
            {"connect": 22.0, "read": 22.0}, snapshot["httpTimeoutSeconds"]
        )

    async def test_bounded_trace_and_malformed_request_id(self):
        observer = ProviderObservation()
        for _ in range(100):
            await observer.trace("http11.receive_response_body.started", {})
        await observer.on_response(
            httpx.Response(200, headers={"x-request-id": "private token value"})
        )
        self.assertEqual(80, len(observer.snapshot()["events"]))
        self.assertIsNone(observer.request_id)

    async def test_exception_chain_is_captured_before_public_mapping(self):
        observer = ProviderObservation()
        error = APITimeoutError(request=httpx.Request("POST", "https://private.test/"))
        error.__cause__ = httpx.ReadTimeout("private timeout")
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.create.side_effect = error
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "private"),
            patch("openai.AsyncOpenAI", return_value=client) as sdk,
            patch.object(observer, "http_client", return_value="fixture"),
            self.assertRaises(SourceAdapterError) as caught,
        ):
            await search_sources(
                {"name": "Business", "city": "Київ", "mode": "NAME_CITY"},
                observation=observer,
            )
        self.assertEqual("SOURCE_PROVIDER_TIMEOUT", caught.exception.code)
        self.assertEqual(["APITimeoutError", "ReadTimeout"], observer.cause_classes)
        self.assertEqual(22, sdk.call_args.kwargs["timeout"])
        self.assertEqual(0, sdk.call_args.kwargs["max_retries"])
        self.assertNotIn("private", json.dumps(observer.snapshot()))

    async def test_sdk_cancellation_closes_client_and_stays_cancellation(self):
        observer = ProviderObservation()
        started = asyncio.Event()
        client = AsyncMock()
        client.__aenter__.return_value = client

        async def never(**kwargs):
            started.set()
            await asyncio.Event().wait()

        client.responses.create.side_effect = never
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "private"),
            patch("openai.AsyncOpenAI", return_value=client),
            patch.object(observer, "http_client", return_value="fixture"),
        ):
            task = asyncio.create_task(
                search_sources(
                    {"name": "Business", "city": "Київ", "mode": "NAME_CITY"},
                    observation=observer,
                )
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(observer.cancelled)
        self.assertFalse(observer.completed)
        self.assertEqual(1, client.responses.create.await_count)
        self.assertEqual(1, client.__aexit__.await_count)


class SearchOperationLifetime(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.adapter = EmailSourceAdapter(SimpleNamespace(active_count=0))
        self.app = web.Application()
        self.app.router.add_post(
            "/search", lambda request: self.adapter.handle(request, "search")
        )
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.payload = {
            "operationKey": "00000000-0000-4000-8000-000000000001",
            "name": "Business",
            "city": "Київ",
            "mode": "NAME_CITY",
        }

    async def asyncTearDown(self):
        await self.client.close()

    async def test_concurrent_same_key_calls_provider_once_and_reuses_completed(self):
        started, complete = asyncio.Event(), asyncio.Event()

        async def slow(body):
            started.set()
            await complete.wait()
            return {"sources": [], "usage": {"calls": 1}}

        with patch("email_source_adapter.search_sources", side_effect=slow) as search:
            first = asyncio.create_task(self.client.post("/search", json=self.payload))
            await started.wait()
            second = asyncio.create_task(self.client.post("/search", json=self.payload))
            await asyncio.sleep(0.05)
            complete.set()
            responses = await asyncio.gather(first, second)
            self.assertEqual([200, 200], [response.status for response in responses])
            self.assertEqual(1, search.await_count)
            self.assertEqual(
                200, (await self.client.post("/search", json=self.payload)).status
            )
            self.assertEqual(1, search.await_count)

    async def test_total_deadline_cancels_and_drains_provider_without_retry(self):
        closed = asyncio.Event()

        async def never(body):
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        with (
            patch("email_source_adapter.SEARCH_TOTAL_DEADLINE", 0.05, create=True),
            patch("email_source_adapter.search_sources", side_effect=never) as search,
        ):
            response = await self.client.post("/search", json=self.payload)
            self.assertEqual(503, response.status)
            self.assertEqual(
                "SOURCE_PROVIDER_TIMEOUT", (await response.json())["error"]["code"]
            )
            self.assertTrue(closed.is_set())
            self.assertEqual(1, search.await_count)
            self.assertFalse(self.adapter.inflight)

    async def test_disconnected_waiter_does_not_cancel_other_waiter(self):
        started, complete, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def slow(body):
            started.set()
            try:
                await complete.wait()
                return {"sources": [], "usage": {"calls": 1}}
            finally:
                closed.set()

        with patch("email_source_adapter.search_sources", side_effect=slow) as search:
            first = asyncio.create_task(self.client.post("/search", json=self.payload))
            await started.wait()
            second = asyncio.create_task(self.client.post("/search", json=self.payload))
            await asyncio.sleep(0.05)
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            await asyncio.sleep(0.15)
            self.assertFalse(closed.is_set())
            complete.set()
            self.assertEqual(200, (await second).status)
            self.assertTrue(closed.is_set())
            self.assertEqual(1, search.await_count)
            self.assertFalse(self.adapter.inflight)

    async def test_last_disconnected_waiter_cancels_and_drains_provider(self):
        started, closed = asyncio.Event(), asyncio.Event()

        async def never(body):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        with patch("email_source_adapter.search_sources", side_effect=never) as search:
            waiting = asyncio.create_task(
                self.client.post("/search", json=self.payload)
            )
            await started.wait()
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            await asyncio.wait_for(closed.wait(), 1)
            async with asyncio.timeout(1):
                while self.adapter.inflight:
                    await asyncio.sleep(0.01)
            self.assertEqual(1, search.await_count)
            self.assertFalse(self.adapter.inflight)


if __name__ == "__main__":
    unittest.main()

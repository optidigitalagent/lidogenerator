import unittest
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from email_source_adapter import (
    EmailSourceAdapter,
    PublicResolver,
    SourceAdapterError,
    public_ip,
    public_url,
    query_text,
)


class EmailSafety(unittest.IsolatedAsyncioTestCase):
    def test_public_addresses_and_urls(self):
        for value in [
            "127.0.0.1",
            "169.254.169.254",
            "10.0.0.2",
            "::1",
            "::ffff:8.8.8.8",
            "2002::1",
            "2001:db8::1",
            "fc00::1",
        ]:
            self.assertFalse(public_ip(value))
        self.assertTrue(public_ip("8.8.8.8"))
        for value in [
            "file:///etc/passwd",
            "http://localhost/x",
            "http://[::1]/",
            "https://user:pass@example.com/",
            "https://example.com/?access_token=private",
            "https://example.com/?X-Amz-Signature=private",
        ]:
            with self.assertRaises(SourceAdapterError):
                public_url(value)

    def test_source_terms_refused_before_network_including_aliases(self):
        for host in ("t.me", "www.telegram.me", "telegram.org", "a.telegram.dog"):
            with self.assertRaises(SourceAdapterError) as error:
                public_url("https://" + host + "/s/business")
            self.assertEqual("SOURCE_POLICY_RESTRICTED", error.exception.code)
            self.assertEqual(403, error.exception.status)
        self.assertEqual(
            "https://telegram.org.example.test/",
            public_url("https://telegram.org.example.test/"),
        )

    def test_queries_do_not_mix_all_identity_fields(self):
        body = {
            "name": "Business",
            "city": "Київ",
            "phone": "+380501234567",
            "address": "Test 12",
        }
        self.assertEqual(
            '"Business" Київ email контакти', query_text(body | {"mode": "NAME_CITY"})
        )
        self.assertNotIn("Business", query_text(body | {"mode": "PHONE_CITY"}))
        self.assertNotIn("+380", query_text(body | {"mode": "NAME_ADDRESS"}))

    async def test_mixed_dns_is_denied(self):
        loop = __import__("asyncio").get_running_loop()
        answer = [(2, 1, 6, "", ("8.8.8.8", 443)), (2, 1, 6, "", ("127.0.0.1", 443))]
        with (
            patch.object(loop, "getaddrinfo", new=AsyncMock(return_value=answer)),
            self.assertRaises(SourceAdapterError),
        ):
            await PublicResolver().resolve("public.test", 443)

    async def test_discovery_priority_and_idempotency(self):
        class Runtime:
            active_count = 1

        runtime = Runtime()
        adapter = EmailSourceAdapter(runtime)
        app = web.Application()
        app.router.add_post(
            "/search", lambda request: adapter.handle(request, "search")
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            payload = {
                "operationKey": "00000000-0000-4000-8000-000000000001",
                "mode": "NAME_CITY",
                "name": "Test",
                "city": "Київ",
            }
            blocked = await client.post("/search", json=payload)
            self.assertEqual(503, blocked.status)
            self.assertEqual("DISCOVERY_BUSY", (await blocked.json())["error"]["code"])
            runtime.active_count = 0
            with patch(
                "email_source_adapter.search_sources",
                new=AsyncMock(return_value={"sources": [], "usage": {"calls": 1}}),
            ) as search:
                self.assertEqual(
                    200, (await client.post("/search", json=payload)).status
                )
                self.assertEqual(
                    200, (await client.post("/search", json=payload)).status
                )
                self.assertEqual(1, search.await_count)
        finally:
            await client.close()


class EmailBrowserSafety(unittest.IsolatedAsyncioTestCase):
    async def test_actual_chromium_executes_public_contact_js_and_blocks_private_egress(
        self,
    ):
        import asyncio

        from email_source_adapter import render_source

        calls = []
        html = b"""<h1>Fixture public business</h1><div id="contact"></div><script>
        document.querySelector('#contact').textContent='published@fixture.test';
        fetch('http://127.0.0.1/private').catch(()=>{});
        fetch('http://169.254.169.254/latest/meta-data').catch(()=>{});
        fetch('http://[::1]/private').catch(()=>{});
        new WebSocket('ws://127.0.0.1/socket');
        navigator.serviceWorker.register('/worker.js').catch(()=>{});
        </script>"""

        async def public_fixture(_client, url, budget=None, guard=None):
            if guard is not None:
                with patch(
                    "email_source_adapter._pinned_get",
                    new=AsyncMock(
                        return_value=(
                            200,
                            {},
                            b"User-agent: *\nAllow: /",
                            "https://public-fixture.test/robots.txt",
                        )
                    ),
                ):
                    await guard(url)
            calls.append(url)
            self.assertEqual("https://public-fixture.test/", url)
            if budget:
                budget.requests += 1
                budget.bytes += len(html)
            return 200, {"Content-Type": "text/html"}, html, url

        with patch("email_source_adapter.pinned_get", side_effect=public_fixture):
            result = await asyncio.wait_for(
                render_source({"url": "https://public-fixture.test/"}), 10
            )
        self.assertIn("published@fixture.test", result["html"])
        self.assertEqual(["https://public-fixture.test/"], calls)
        self.assertGreaterEqual(result["blockedRequests"], 3)
        self.assertEqual(1, result["requests"])


class EmailLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_waf_is_not_executed_and_reports_source_status(self):
        from email_source_adapter import render_source

        payload = b'<p id="wci" class="_wafchallengeid"></p><script src="/waf-aiso/x.js"></script>'

        async def response(*_args, **_kwargs):
            return (
                200,
                {"Content-Type": "text/html"},
                payload,
                "https://public-fixture.test/",
            )

        with (
            patch("email_source_adapter.pinned_get", side_effect=response),
            patch("email_source_adapter.async_playwright") as browser,
        ):
            with self.assertRaises(SourceAdapterError) as caught:
                await render_source({"url": "https://public-fixture.test/"})
            self.assertEqual("ACCESS_WALL", caught.exception.code)
            self.assertEqual(200, caught.exception.details["httpStatus"])
            browser.assert_not_called()

    async def test_render_is_cancelled_when_discovery_starts(self):
        import asyncio

        started, cancelled = asyncio.Event(), asyncio.Event()

        class Runtime:
            active_count = 0

        runtime = Runtime()

        async def slow(_body):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        adapter = EmailSourceAdapter(runtime)
        app = web.Application()

        async def route(request):
            return await adapter.handle(request, "render")

        app.router.add_post("/render", route)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            with patch("email_source_adapter.render_source", side_effect=slow):
                pending = asyncio.create_task(
                    client.post("/render", json={"url": "https://public-fixture.test/"})
                )
                await started.wait()
                runtime.active_count = 1
                response = await pending
                self.assertEqual(503, response.status)
                self.assertEqual(
                    "DISCOVERY_BUSY", (await response.json())["error"]["code"]
                )
                self.assertTrue(cancelled.is_set())
        finally:
            await client.close()


class EmailNativeEgressFence(unittest.IsolatedAsyncioTestCase):
    async def test_native_browser_without_routes_cannot_connect_to_loopback(self):
        import asyncio

        from playwright.async_api import Error, async_playwright

        from email_source_adapter import browser_no_egress

        reached = []

        async def target(_reader, writer):
            reached.append(True)
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(target, "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            async with async_playwright() as p:
                with browser_no_egress() as proxy:
                    browser = await p.chromium.launch(
                        proxy=proxy, args=["--disable-quic", "--dns-prefetch-disable"]
                    )
                    try:
                        page = await browser.new_page()
                        with self.assertRaises(Error):
                            await page.goto(
                                f"http://127.0.0.1:{port}/private", timeout=3000
                            )
                        self.assertEqual([], reached)
                    finally:
                        await browser.close()
        finally:
            server.close()
            await server.wait_closed()


class SourceRules(unittest.TestCase):
    def test_shared_robot_cases(self):
        from email_source_adapter import robots_policy

        rules = robots_policy(
            "User-agent: *\nUser-agent: Other\nDisallow: /private/*\nAllow: /private/public$"
        )
        self.assertFalse(rules("https://public.test/private/else"))
        self.assertTrue(rules("https://public.test/private/public"))
        self.assertFalse(rules("https://public.test/private/public/else"))
        self.assertFalse(
            robots_policy("User-agent: *\nDisallow: /приватне")(
                "https://public.test/%D0%BF%D1%80%D0%B8%D0%B2%D0%B0%D1%82%D0%BD%D0%B5"
            )
        )
        self.assertFalse(
            robots_policy("User-agent: *\nDisallow: /admin")(
                "https://public.test/%61dmin"
            )
        )
        with self.assertRaises(SourceAdapterError):
            robots_policy("User-agent: *\nDisallow: /" + "*" * 3000)
        specific = robots_policy(
            "User-agent: *\nAllow: /\n\nUser-agent: Opti-Public-Contacts/2.0\nDisallow: /"
        )
        self.assertFalse(specific("https://public.test/"))


class ProviderCompletion(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_only_one_completed_call_and_ignores_unexecuted_sources(self):
        from types import SimpleNamespace

        from email_source_adapter import search_sources

        output = [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {"sources": [{"url": "https://business.test/contact"}]},
            },
            {
                "type": "web_search_call",
                "status": "searching",
                "action": {"sources": [{"url": "https://unexecuted.test/"}]},
            },
        ]
        response = SimpleNamespace(
            model_dump=lambda: {"output": output},
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.create.return_value = response
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "fixture"),
            patch("openai.AsyncOpenAI", return_value=client),
        ):
            data = await search_sources(
                {"name": "Business", "city": "Київ", "mode": "NAME_CITY"}
            )
            self.assertEqual(
                [{"url": "https://business.test/contact", "title": ""}], data["sources"]
            )
            self.assertEqual(1, data["usage"]["calls"])
            output[1]["status"] = "completed"
            with self.assertRaises(SourceAdapterError):
                await search_sources(
                    {"name": "Business", "city": "Київ", "mode": "NAME_CITY"}
                )


class ProtectedSourceRoutes(unittest.IsolatedAsyncioTestCase):
    async def test_existing_control_bearer_protects_both_new_tools(self):
        from types import SimpleNamespace

        from control_api import create_app

        token = "isolated-fixture-token-not-a-production-secret"
        client = TestClient(
            TestServer(create_app(SimpleNamespace(active_count=0), token))
        )
        await client.start_server()
        try:
            for kind in ["search", "render"]:
                path = "/internal/opti/v1/email-sources/" + kind
                for headers in [{}, {"Authorization": "Bearer invalid"}]:
                    response = await client.post(path, headers=headers, json={})
                    self.assertEqual(401, response.status)
                    self.assertNotIn(token, await response.text())
            with patch(
                "email_source_adapter.search_sources",
                new=AsyncMock(return_value={"sources": [], "usage": {"calls": 1}}),
            ):
                response = await client.post(
                    "/internal/opti/v1/email-sources/search",
                    headers={"Authorization": "Bearer " + token},
                    json={"operationKey": "00000000-0000-4000-8000-000000000001"},
                )
                self.assertEqual(200, response.status)
        finally:
            await client.close()


class ProviderRetryAfter(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_quota_keeps_bounded_retry_after_and_http_adapter_forwards_it(
        self,
    ):
        import httpx
        from openai import RateLimitError

        from email_source_adapter import provider_retry_after, search_sources

        error = RateLimitError(
            "fixture",
            response=httpx.Response(
                429,
                headers={"retry-after": "120"},
                request=httpx.Request("POST", "https://api.example.test/"),
            ),
            body=None,
        )
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.responses.create.side_effect = error
        with (
            patch("config.WEBSITE_SEARCH_PROVIDER", "openai"),
            patch("config.OPENAI_API_KEY", "fixture"),
            patch("openai.AsyncOpenAI", return_value=client),
            self.assertRaises(SourceAdapterError) as caught,
        ):
            await search_sources(
                {"name": "Business", "city": "Київ", "mode": "NAME_CITY"}
            )
        self.assertEqual("120", caught.exception.retry_after)
        self.assertEqual("SOURCE_PROVIDER_QUOTA", caught.exception.code)
        error.response.headers["retry-after"] = "invalid"
        self.assertIsNone(provider_retry_after(error))
        error.response.headers["retry-after"] = "999999999"
        self.assertEqual("86400", provider_retry_after(error))

        class Runtime:
            active_count = 0

        adapter = EmailSourceAdapter(Runtime())
        app = web.Application()

        async def call(request):
            return await adapter.handle(request, "search")

        app.router.add_post("/search", call)
        api = TestClient(TestServer(app))
        await api.start_server()
        try:
            with patch(
                "email_source_adapter.search_sources",
                new=AsyncMock(side_effect=caught.exception),
            ):
                response = await api.post(
                    "/search",
                    json={"operationKey": "00000000-0000-4000-8000-000000000002"},
                )
                self.assertEqual(503, response.status)
                self.assertEqual("120", response.headers.get("Retry-After"))
        finally:
            await api.close()

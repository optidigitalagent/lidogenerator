import inspect
import socket
import unittest

import config
from playwright.async_api import TimeoutError as PlaywrightTimeout

from agents.rendered_site_auditor import (
    PageRequestSafetyGuard,
    PlaywrightRenderedSiteAuditor,
    RenderedSiteAuditorSettings,
    is_public_ip_address,
    resolve_public_host,
)
from rendered_site_audit import RenderedSiteAuditStatus
from tests.test_rendered_site_audit import desktop_metrics, metrics


def public_answers(*args):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
    ]


class FakeRoute:
    def __init__(self):
        self.continued = False
        self.aborted = False

    async def continue_(self):
        self.continued = True

    async def abort(self, reason):
        self.aborted = True
        self.reason = reason


class FakeRequest:
    def __init__(self, url, page, navigation=False):
        self.url = url
        self.frame = page.main_frame
        self._navigation = navigation

    def is_navigation_request(self):
        return self._navigation


class FakePage:
    def __init__(self, context, final_url, raw_metrics, goto_error=None, subresource=None):
        self.context = context
        self.url = final_url
        self.raw_metrics = raw_metrics
        self.goto_error = goto_error
        self.subresource = subresource
        self.main_frame = object()
        self.goto_calls = []
        self.evaluate_calls = 0
        self.wait_calls = []

    async def goto(self, url, **kwargs):
        self.goto_calls.append((url, kwargs))
        if self.subresource is not None:
            route = FakeRoute()
            await self.context.route_handler(
                route,
                FakeRequest(self.subresource, self, navigation=False),
            )
            self.context.subresource_routes.append(route)
        if self.goto_error is not None:
            raise self.goto_error
        return object()

    async def wait_for_timeout(self, milliseconds):
        self.wait_calls.append(milliseconds)

    async def evaluate(self, script, argument):
        self.evaluate_calls += 1
        self.evaluate_argument = argument
        return dict(self.raw_metrics)


class FakeContext:
    def __init__(self, spec):
        self.spec = spec
        self.route_handler = None
        self.closed = False
        self.subresource_routes = []
        self.page = FakePage(self, **spec)

    async def new_page(self):
        return self.page

    async def route(self, pattern, handler):
        self.route_pattern = pattern
        self.route_handler = handler

    async def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, specs):
        self.specs = list(specs)
        self.context_options = []
        self.contexts = []
        self.closed = False

    async def new_context(self, **options):
        self.context_options.append(options)
        context = FakeContext(self.specs.pop(0))
        self.contexts.append(context)
        return context

    async def close(self):
        self.closed = True


def build_auditor(browser, *, resolver=public_answers, max_hosts=40):
    auditor = PlaywrightRenderedSiteAuditor(
        RenderedSiteAuditorSettings(
            timeout_seconds=12,
            settle_milliseconds=750,
            max_hosts_per_page=max_hosts,
        ),
        dns_resolver=resolver,
    )
    auditor._browser = browser
    return auditor


class NetworkSafetyTests(unittest.IsolatedAsyncioTestCase):
    def test_safe_config_defaults_are_exact(self):
        self.assertEqual(config.RENDERED_SITE_AUDIT_MODE, "off")
        self.assertEqual(config.MAX_RENDERED_BAD_SITE_AUDITS_PER_TASK, 12)
        self.assertEqual(config.MAX_RENDERED_GOOD_SITE_AUDITS_PER_TASK, 8)
        self.assertEqual(config.RENDERED_SITE_AUDIT_CONCURRENCY, 2)
        self.assertEqual(config.RENDERED_SITE_AUDIT_TIMEOUT_SECONDS, 12)
        self.assertEqual(config.RENDERED_SITE_AUDIT_SETTLE_MILLISECONDS, 750)
        self.assertEqual(config.RENDERED_SITE_AUDIT_MAX_HOSTS_PER_PAGE, 40)

    def test_public_and_unsafe_ip_classes(self):
        self.assertTrue(is_public_ip_address("8.8.8.8"))
        for value in (
            "127.0.0.1",
            "10.0.0.1",
            "169.254.1.1",
            "224.0.0.1",
            "192.0.2.1",
            "0.0.0.0",
            "::1",
        ):
            self.assertFalse(is_public_ip_address(value), value)

    async def test_dns_requires_all_answers_public(self):
        def mixed(*args):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
            ]

        self.assertTrue(await resolve_public_host("public.test", resolver=public_answers))
        self.assertFalse(await resolve_public_host("mixed.test", resolver=mixed))
        self.assertFalse(await resolve_public_host("127.0.0.1", resolver=public_answers))

    async def test_dns_cache_validates_hostname_once_per_task(self):
        calls = []

        def resolver(*args):
            calls.append(args[0])
            return public_answers(*args)

        auditor = build_auditor(FakeBrowser([]), resolver=resolver)
        self.assertTrue(await auditor._host_is_public("assets.test"))
        self.assertTrue(await auditor._host_is_public("ASSETS.TEST"))
        self.assertEqual(calls, ["assets.test"])

    async def test_unsafe_subresource_is_aborted(self):
        specs = [
            {
                "final_url": "https://safe.test/",
                "raw_metrics": desktop_metrics().__dict__,
                "subresource": "http://127.0.0.1/private",
            },
            {
                "final_url": "https://safe.test/",
                "raw_metrics": metrics().__dict__,
            },
        ]
        browser = FakeBrowser(specs)
        result = await build_auditor(browser).audit("https://safe.test/")
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_GOOD)
        route = browser.contexts[0].subresource_routes[0]
        self.assertTrue(route.aborted)
        self.assertFalse(route.continued)

    async def test_host_cap_blocks_new_host_but_allows_seen_host(self):
        async def public(_hostname):
            return True

        guard = PageRequestSafetyGuard(public, max_hosts=2)
        self.assertTrue(await guard.permits("https://a.test/a", top_level=False))
        self.assertTrue(await guard.permits("https://b.test/b", top_level=False))
        self.assertTrue(await guard.permits("https://a.test/c", top_level=False))
        self.assertFalse(await guard.permits("https://c.test/c", top_level=False))
        self.assertTrue(await guard.permits("data:image/png;base64,AA", top_level=False))
        self.assertFalse(await guard.permits("data:text/html,x", top_level=True))


class PlaywrightAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_desktop_mobile_context_settings_and_one_evaluate(self):
        browser = FakeBrowser(
            [
                {"final_url": "https://www.safe.test/", "raw_metrics": desktop_metrics().__dict__},
                {"final_url": "https://safe.test/", "raw_metrics": metrics().__dict__},
            ]
        )
        result = await build_auditor(browser).audit("https://safe.test/")
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_GOOD)
        self.assertEqual(result.pages_attempted, 2)
        self.assertEqual(result.pages_succeeded, 2)
        desktop, mobile = browser.context_options
        self.assertEqual(desktop["viewport"], {"width": 1366, "height": 768})
        self.assertFalse(desktop["is_mobile"])
        self.assertFalse(desktop["has_touch"])
        self.assertEqual(mobile["viewport"], {"width": 390, "height": 844})
        self.assertTrue(mobile["is_mobile"])
        self.assertTrue(mobile["has_touch"])
        for options in (desktop, mobile):
            self.assertEqual(options["device_scale_factor"], 1)
            self.assertFalse(options["accept_downloads"])
            self.assertEqual(options["service_workers"], "block")
            self.assertNotIn("permissions", options)
        self.assertEqual([ctx.page.evaluate_calls for ctx in browser.contexts], [1, 1])
        self.assertTrue(all(ctx.closed for ctx in browser.contexts))

    async def test_cross_domain_top_level_is_uncertain(self):
        browser = FakeBrowser(
            [{"final_url": "https://other.test/", "raw_metrics": desktop_metrics().__dict__}]
        )
        result = await build_auditor(browser).audit("https://safe.test/")
        self.assertEqual(result.status, RenderedSiteAuditStatus.UNCERTAIN)
        self.assertEqual(result.error_category, "cross_domain_redirect")
        self.assertEqual(len(browser.contexts), 1)

    async def test_timeout_is_contained_without_retry(self):
        browser = FakeBrowser(
            [
                {
                    "final_url": "https://safe.test/",
                    "raw_metrics": desktop_metrics().__dict__,
                    "goto_error": PlaywrightTimeout("timed out"),
                }
            ]
        )
        result = await build_auditor(browser).audit("https://safe.test/")
        self.assertEqual(result.status, RenderedSiteAuditStatus.TECHNICAL_ERROR)
        self.assertEqual(result.error_category, "timeout")
        self.assertEqual(len(browser.contexts), 1)
        self.assertEqual(len(browser.contexts[0].page.goto_calls), 1)
        _, kwargs = browser.contexts[0].page.goto_calls[0]
        self.assertEqual(kwargs["wait_until"], "domcontentloaded")
        self.assertEqual(kwargs["timeout"], 12000)

    def test_implementation_never_persists_screenshots(self):
        source = inspect.getsource(PlaywrightRenderedSiteAuditor)
        self.assertNotIn("screenshot(", source)
        self.assertNotIn("raw_html", source)


if __name__ == "__main__":
    unittest.main()

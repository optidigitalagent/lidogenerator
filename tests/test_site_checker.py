import asyncio
import unittest

import httpx

from agents import site_checker
from models import Business
from website_pipeline import WebsiteAuditStatus


GOOD_HTML = '<html><head><meta name="viewport"><title>Good</title></head><body>' + ("content " * 80) + "</body></html>"
BAD_HTML = "<html><body>short</body></html>"


class Response:
    def __init__(self, status_code=200, text=GOOD_HTML, url="https://example.com/"):
        self.status_code = status_code
        self.text = text
        self.url = url


class Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    async def get(self, url):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


async def check(business, response=None, error=None):
    client = Client(response, error)
    await site_checker._check_one(client, asyncio.Semaphore(1), business)
    return client


class SiteCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_and_non_official_urls_skip_network(self):
        for url in ("", "https://instagram.com/alpha", "https://booksy.com/alpha"):
            with self.subTest(url=url):
                business = Business(website=url, instagram_url="https://instagram.com/alpha")
                client = await check(business)
                self.assertEqual(client.calls, 0)
                self.assertEqual(business.website_audit_status, WebsiteAuditStatus.NO_OFFICIAL_SITE.value)
                self.assertEqual(business.site_quality, "none")

    async def test_success_good_and_bad(self):
        for html, expected in ((GOOD_HTML, "good"), (BAD_HTML, "bad")):
            with self.subTest(expected=expected):
                business = Business(website="https://example.com/")
                await check(business, Response(text=html))
                self.assertEqual(business.site_quality, expected)
                self.assertEqual(business.website_audit_status, expected)
                self.assertEqual(business.website_final_url, "https://example.com/")

    async def test_http_taxonomy(self):
        rows = (
            (404, "dead", WebsiteAuditStatus.DEAD_CONFIRMED),
            (410, "dead", WebsiteAuditStatus.DEAD_CONFIRMED),
            (403, "technical_error", WebsiteAuditStatus.TECHNICAL_ERROR),
            (429, "technical_error", WebsiteAuditStatus.TECHNICAL_ERROR),
            (500, "technical_error", WebsiteAuditStatus.TECHNICAL_ERROR),
            (503, "technical_error", WebsiteAuditStatus.TECHNICAL_ERROR),
            (418, "uncertain", WebsiteAuditStatus.UNCERTAIN),
        )
        for code, quality, status in rows:
            with self.subTest(code=code):
                business = Business(website="https://example.com/")
                await check(business, Response(status_code=code))
                self.assertEqual(business.site_quality, quality)
                self.assertEqual(business.website_audit_status, status.value)
                self.assertEqual(business.website_audit_http_status, code)

    async def test_timeout_and_request_error_are_not_dead_or_leads(self):
        request = httpx.Request("GET", "https://example.com/")
        errors = (httpx.ReadTimeout("timeout", request=request), httpx.ConnectError("dns", request=request))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                business = Business(
                    website="https://example.com/",
                    instagram_url="https://instagram.com/alpha",
                )
                await check(business, error=error)
                self.assertEqual(business.site_quality, "technical_error")
                self.assertEqual(business.website_status, "uncertain website")
                self.assertFalse(business.is_lead)

    async def test_redirect_to_instagram_is_not_official(self):
        business = Business(website="https://example.com/")
        await check(business, Response(url="https://instagram.com/alpha"))
        self.assertEqual(business.website_audit_status, WebsiteAuditStatus.NO_OFFICIAL_SITE.value)
        self.assertEqual(business.website_final_url, "https://instagram.com/alpha")
        self.assertIn("redirected_to_non_official_platform", business.website_audit_evidence)

    async def test_redirect_to_ordinary_domain_is_analyzed(self):
        business = Business(website="https://example.com/")
        await check(business, Response(url="https://www.example.org/path"))
        self.assertTrue(business.has_site)
        self.assertEqual(business.website_final_url, "https://www.example.org/path")

    async def test_invalid_url_is_technical_error(self):
        business = Business(website="not a URL")
        await check(business)
        self.assertEqual(business.site_quality, "technical_error")
        self.assertEqual(business.website_audit_status, WebsiteAuditStatus.TECHNICAL_ERROR.value)

    async def test_raw_html_is_not_persisted(self):
        business = Business(website="https://example.com/")
        secret_html = GOOD_HTML + "SECRET_HTML_PAYLOAD"
        await check(business, Response(text=secret_html))
        self.assertNotIn("SECRET_HTML_PAYLOAD", business.website_audit_evidence)


if __name__ == "__main__":
    unittest.main()

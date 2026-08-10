"""Offline DNS/HTTP/runtime tests for the first-party resolver."""

import asyncio
import unittest

import httpx
import config

from agents.instagram_first_party_resolver import (
    FirstPartyInstagramRequestBudget,
    FirstPartyInstagramResolver,
    FirstPartyInstagramSettings,
    resolve_missing_instagrams_first_party,
    trusted_website_for_instagram_resolution,
)
from instagram_first_party_resolution import FirstPartyInstagramStatus as Status
from models import Business


async def public_dns(hostname: str, port: int):
    return ("93.184.216.34",)


def eligible_business(
    name: str = "Synthetic",
    *,
    quality: str = "bad",
    website: str = "https://brand.example/",
) -> Business:
    return Business(
        name=name,
        website=website,
        has_site=True,
        site_quality=quality,
        website_final_url=website,
    )


class TrustedWebsiteTests(unittest.TestCase):
    def test_confirmed_bad_and_good_legacy_sites_are_eligible(self) -> None:
        for quality in ("bad", "good"):
            with self.subTest(quality=quality):
                business = eligible_business(quality=quality)
                self.assertEqual(
                    trusted_website_for_instagram_resolution(business),
                    "https://brand.example/",
                )

    def test_uncertain_dead_no_site_and_platform_urls_are_skipped(self) -> None:
        cases = (
            Business(website="https://brand.example/", site_quality="uncertain"),
            Business(website="https://brand.example/", site_quality="dead"),
            Business(website="https://brand.example/", site_quality="none"),
            Business(
                website="https://instagram.com/not_a_site/",
                has_site=True,
                site_quality="bad",
            ),
            Business(
                website="http://localhost/",
                has_site=True,
                site_quality="bad",
            ),
        )
        for business in cases:
            self.assertIsNone(trusted_website_for_instagram_resolution(business))

    def test_website_resolver_found_official_is_eligible(self) -> None:
        business = Business(
            website_resolution_status="found_official",
            website_resolved_url="https://official.example/path",
        )
        self.assertEqual(
            trusted_website_for_instagram_resolution(business),
            "https://official.example/path",
        )


class FirstPartyConfigDefaultTests(unittest.TestCase):
    def test_safe_defaults_are_exact(self) -> None:
        self.assertEqual(config.INSTAGRAM_FIRST_PARTY_MODE, "off")
        self.assertEqual(config.MAX_INSTAGRAM_FIRST_PARTY_REQUESTS_PER_TASK, 0)
        self.assertEqual(config.INSTAGRAM_FIRST_PARTY_MAX_PAGES_PER_BUSINESS, 2)
        self.assertEqual(config.INSTAGRAM_FIRST_PARTY_TIMEOUT_SECONDS, 8.0)
        self.assertEqual(config.INSTAGRAM_FIRST_PARTY_MAX_RESPONSE_BYTES, 1_048_576)
        self.assertEqual(config.INSTAGRAM_FIRST_PARTY_CONCURRENCY, 4)
        self.assertEqual(config.INSTAGRAM_FIRST_PARTY_MAX_REDIRECTS, 3)
        self.assertEqual(config.INSTAGRAM_SEARCH_PROVIDER, "none")
        self.assertEqual(config.MAX_INSTAGRAM_SEARCH_REQUESTS_PER_TASK, 0)


class FirstPartyNetworkTests(unittest.IsolatedAsyncioTestCase):
    def _resolver(
        self,
        handler,
        *,
        budget: int = 20,
        dns_resolver=public_dns,
        max_bytes: int = 1_048_576,
        concurrency: int = 4,
    ) -> FirstPartyInstagramResolver:
        return FirstPartyInstagramResolver(
            FirstPartyInstagramSettings(
                max_pages_per_business=2,
                timeout_seconds=1,
                max_response_bytes=max_bytes,
                concurrency=concurrency,
                max_redirects=3,
            ),
            FirstPartyInstagramRequestBudget(budget),
            transport=httpx.MockTransport(handler),
            dns_resolver=dns_resolver,
        )

    async def _run(self, business: Business, resolver: FirstPartyInstagramResolver):
        results = await resolve_missing_instagrams_first_party(
            [business], resolver=resolver
        )
        return results[0]

    async def test_public_ip_allowed_and_profile_applied(self) -> None:
        resolver = self._resolver(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<a href="https://instagram.com/public_brand">Instagram</a>',
            )
        )
        business = eligible_business()
        result = await self._run(business, resolver)
        self.assertIs(result.status, Status.FOUND_OFFICIAL)
        self.assertEqual(
            business.instagram_url, "https://www.instagram.com/public_brand/"
        )

    async def test_private_and_link_local_dns_are_rejected(self) -> None:
        for address in ("127.0.0.1", "10.1.2.3", "169.254.10.4", "::1"):
            async def unsafe_dns(hostname, port, value=address):
                return (value,)

            calls = 0

            def handler(request):
                nonlocal calls
                calls += 1
                return httpx.Response(200, headers={"content-type": "text/html"})

            result = await self._run(
                eligible_business(),
                self._resolver(handler, dns_resolver=unsafe_dns),
            )
            self.assertIs(result.status, Status.TECHNICAL_ERROR)
            self.assertEqual(result.error_category, "unsafe_address")
            self.assertEqual(calls, 0)

    async def test_same_host_www_redirect_is_accepted_and_consumes_budget(self) -> None:
        calls = []

        def handler(request: httpx.Request):
            calls.append(str(request.url))
            if request.url.host == "brand.example":
                return httpx.Response(
                    302,
                    headers={"location": "https://www.brand.example/home"},
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<a href="https://instagram.com/redirect_brand/">IG</a>',
            )

        resolver = self._resolver(handler)
        result = await self._run(eligible_business(), resolver)
        self.assertIs(result.status, Status.FOUND_OFFICIAL)
        self.assertEqual(len(calls), 2)
        self.assertEqual(resolver.budget.snapshot().used_requests, 2)

    async def test_redirect_cannot_exceed_task_budget(self) -> None:
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(
                302, headers={"location": "https://www.brand.example/home"}
            )

        resolver = self._resolver(handler, budget=1)
        result = await self._run(eligible_business(), resolver)
        self.assertIs(result.status, Status.SKIPPED)
        self.assertEqual(result.pages_attempted, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(resolver.budget.snapshot().used_requests, 1)

    async def test_cross_domain_and_social_redirects_are_rejected(self) -> None:
        for location in (
            "https://other.example/",
            "https://instagram.com/redirected/",
            "https://linktr.ee/redirected",
        ):
            with self.subTest(location=location):
                resolver = self._resolver(
                    lambda request, target=location: httpx.Response(
                        302, headers={"location": target}
                    )
                )
                result = await self._run(eligible_business(), resolver)
                self.assertIs(result.status, Status.TECHNICAL_ERROR)
                self.assertEqual(result.error_category, "cross_domain_redirect")

    async def test_timeout_is_technical_and_has_no_retry(self) -> None:
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("synthetic timeout", request=request)

        result = await self._run(eligible_business(), self._resolver(handler))
        self.assertIs(result.status, Status.TECHNICAL_ERROR)
        self.assertEqual(result.error_category, "timeout")
        self.assertEqual(calls, 1)

    async def test_response_size_limit(self) -> None:
        resolver = self._resolver(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"x" * 33,
            ),
            max_bytes=32,
        )
        result = await self._run(eligible_business(), resolver)
        self.assertIs(result.status, Status.TECHNICAL_ERROR)
        self.assertEqual(result.error_category, "response_too_large")

    async def test_non_html_content_type_is_technical(self) -> None:
        resolver = self._resolver(
            lambda request: httpx.Response(
                200, headers={"content-type": "application/json"}, json={}
            )
        )
        result = await self._run(eligible_business(), resolver)
        self.assertIs(result.status, Status.TECHNICAL_ERROR)
        self.assertEqual(result.error_category, "content_type")

    async def test_homepage_found_stops_without_extra_page(self) -> None:
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=(
                    '<a href="/contact">Contact</a>'
                    '<a href="https://instagram.com/home_brand/">Instagram</a>'
                ),
            )

        result = await self._run(eligible_business(), self._resolver(handler))
        self.assertIs(result.status, Status.FOUND_OFFICIAL)
        self.assertEqual(result.pages_attempted, 1)
        self.assertEqual(calls, 1)

    async def test_contact_fallback_and_one_extra_page_max(self) -> None:
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/html"},
                    text='<a href="/contact-us">Contact</a>',
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=(
                    '<a href="/about-us">About</a>'
                    '<a href="https://instagram.com/contact_brand/">Instagram</a>'
                ),
            )

        result = await self._run(eligible_business(), self._resolver(handler))
        self.assertIs(result.status, Status.FOUND_OFFICIAL)
        self.assertEqual(result.pages_attempted, 2)
        self.assertEqual(calls, ["/", "/contact-us"])

        empty_calls = []

        def empty_handler(request):
            empty_calls.append(request.url.path)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<a href="/contact">Contact</a><a href="/about">About</a>',
            )

        empty = await self._run(
            eligible_business(), self._resolver(empty_handler)
        )
        self.assertIs(empty.status, Status.NOT_FOUND)
        self.assertEqual(len(empty_calls), 2)

    async def test_global_budget_is_shared_across_batches(self) -> None:
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text="<html></html>"
            )

        resolver = self._resolver(handler, budget=1)
        first = await self._run(eligible_business("First"), resolver)
        second = await self._run(eligible_business("Second"), resolver)
        self.assertIs(first.status, Status.NOT_FOUND)
        self.assertIs(second.status, Status.SKIPPED)
        self.assertEqual(calls, 1)
        self.assertEqual(resolver.budget.snapshot().remaining_requests, 0)

    async def test_concurrency_is_bounded(self) -> None:
        active = 0
        peak = 0

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<a href="https://instagram.com/concurrent_brand/">IG</a>',
            )

        businesses = [eligible_business(str(index)) for index in range(5)]
        resolver = self._resolver(handler, concurrency=2)
        results = await resolve_missing_instagrams_first_party(
            businesses, resolver=resolver
        )
        self.assertTrue(all(result.status is Status.FOUND_OFFICIAL for result in results))
        self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()

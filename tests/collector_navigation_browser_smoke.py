"""Offline Chromium smoke for the real Maps readiness predicate (no providers).

Run: python -m unittest discover -s tests -p collector_navigation_browser_smoke.py
Requires the documented `playwright install chromium` setup.
"""

import asyncio
import unittest
from unittest.mock import patch

from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from agents import collector

SEARCH = "https://www.google.com/maps/search/offline-fixture"
PLACE = "https://www.google.com/maps/place/offline-fixture"
HEADING = '<h1 class="DUwDvf">Offline business</h1>'
FEED = '<div role="feed">Offline results</div>'


class CollectorNavigationBrowserSmoke(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.extra_markup = ""

        async def fixture(route):
            body = HEADING if "/maps/place/" in route.request.url else "Loading"
            await route.fulfill(content_type="text/html", body=body + self.extra_markup)

        # Every browser request is fulfilled locally, including document navigation.
        await self.context.route("**/*", fixture)
        self.page = await self.context.new_page()

    async def asyncTearDown(self):
        await self.browser.close()
        await self.playwright.stop()

    async def _pending_result(self, stop_flag=None):
        await self.page.goto(SEARCH, wait_until="domcontentloaded")
        await self.page.evaluate("""() => {
            const query = document.querySelectorAll.bind(document);
            document.querySelectorAll = selector => {
                window.resultPollStarted = true;
                return query(selector);
            };
        }""")
        pending = asyncio.create_task(
            collector._wait_for_maps_result(self.page, stop_flag)
        )
        self.addAsyncCleanup(self._finish_wait, pending)
        # Trigger the redirect only after the predicate actually observed Loading.
        await self.page.wait_for_function("() => window.resultPollStarted")
        return pending

    async def _finish_wait(self, pending):
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

    async def test_delayed_spa_search_to_place_after_wait_started(self):
        pending = await self._pending_result()
        await self.page.evaluate("""heading => {
            history.pushState({}, '', '/maps/place/offline-fixture');
            document.body.innerHTML = heading;
        }""", HEADING)
        self.assertEqual(await pending, {"kind": "place", "url": PLACE})
        self.assertEqual(await self.page.locator('[role="feed"]').count(), 0)

    async def test_delayed_document_navigation_to_place(self):
        pending = await self._pending_result()
        await self.page.goto(PLACE, wait_until="domcontentloaded")
        self.assertEqual(await pending, {"kind": "place", "url": PLACE})

    async def test_immediate_single_place(self):
        await self.page.goto(PLACE, wait_until="domcontentloaded")
        self.assertEqual(
            await collector._wait_for_maps_result(self.page),
            {"kind": "place", "url": PLACE},
        )

    async def test_extraction_keeps_real_published_contacts_and_search_parameters(self):
        self.extra_markup = '''
            <button data-item-id="phone:tel:+380441234567">Call</button>
            <a href="mailto:office@public-business.ua?subject=Hello">Email</a>
            <a data-item-id="authority" href="https://public-business.ua/">Website</a>
            <a href="https://instagram.com/public_business/">Instagram</a>
        '''
        business = await collector._extract_business(self.page, PLACE, "Hotel", "Kyiv")
        self.assertEqual(business.phone, "+380441234567")
        self.assertEqual(business.email, "office@public-business.ua")
        self.assertEqual(business.website, "https://public-business.ua/")
        self.assertEqual(business.instagram_url, "https://instagram.com/public_business/")
        self.assertEqual((business.niche, business.city), ("Hotel", "Kyiv"))

    async def test_normal_feed(self):
        pending = await self._pending_result()
        await self.page.evaluate("html => document.body.innerHTML = html", FEED)
        self.assertEqual(await pending, {"kind": "feed"})

    async def test_ready_place_wins_over_residual_feed(self):
        await self.page.goto(PLACE)
        await self.page.evaluate("html => document.body.innerHTML += html", FEED)
        self.assertEqual(
            await collector._wait_for_maps_result(self.page),
            {"kind": "place", "url": PLACE},
        )

    async def test_invalid_or_hidden_states_time_out_instead_of_success(self):
        original_wait = self.page.wait_for_function

        async def accelerated_deadline(script, *, timeout):
            self.assertEqual(timeout, 20000)
            return await original_wait(script, timeout=150)

        cases = [
            (SEARCH, '<h1 class="DUwDvf">Wrong URL</h1>'),
            (PLACE, '<h1>Error page, not a business heading</h1>'),
            (PLACE, '<h1 class="DUwDvf" style="height:30px"> </h1>'),
            (PLACE, f'<div style="display:none">{HEADING}{FEED}</div>'),
            (SEARCH, f'<div style="visibility:hidden">{FEED}</div>'),
        ]
        with patch.object(self.page, "wait_for_function", new=accelerated_deadline):
            for url, html in cases:
                with self.subTest(url=url, html=html):
                    await self.page.goto(url)
                    await self.page.evaluate(
                        "html => document.body.innerHTML = html", html
                    )
                    with self.assertRaisesRegex(
                        PlaywrightTimeout, "Google Maps result state.*20000ms"
                    ):
                        await collector._wait_for_maps_result(self.page)

    async def test_stop_during_navigation_wait(self):
        stopped = asyncio.Event()
        pending = await self._pending_result(stopped.is_set)
        stopped.set()
        self.assertIsNone(await asyncio.wait_for(pending, timeout=1))

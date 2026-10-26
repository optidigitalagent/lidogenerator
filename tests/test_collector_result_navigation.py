"""Deterministic collector navigation, cancellation and counter regressions."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout
from test_collector_discovery_session import (
    _Browser,
    _FeedPage,
    _PlaywrightManager,
)

from agents import collector
from discovery_session import MapsDiscoverySession
from models import Business

PLACE = "https://www.google.com/maps/place/Regression"
OTHER_PLACE = "https://www.google.com/maps/place/Other"


class _ResultPage(_FeedPage):
    def __init__(self, state):
        super().__init__([[PLACE, PLACE, OTHER_PLACE]], end_visible=True)
        self.state = state
        self.stopped = False
        self.wait_cancelled = False
        self.wait_timeout = None
        self.handle = AsyncMock()
        self.handle.json_value.return_value = {
            "kind": "feed" if state == "feed" else "place",
            "url": PLACE,
        }
        if state != "feed":
            self.feed.wait_for = self._old_feed_wait

    async def goto(self, url, **kwargs):
        if self.state == "failed_navigation":
            raise PlaywrightError("Page.goto: net::ERR_CONNECTION_RESET")
        self.url = PLACE if self.state == "immediate_place" else url

    async def _old_feed_wait(self, **kwargs):
        # The old code has already checked the search URL when Maps redirects.
        self.url = PLACE
        raise PlaywrightTimeout("No feed after late single-place navigation")

    async def wait_for_function(self, script, *, timeout):
        self.wait_timeout = timeout
        if self.state == "missing":
            raise PlaywrightTimeout("Neither result state appeared")
        if self.state == "stop":
            self.stopped = True
            try:
                await asyncio.Future()
            finally:
                self.wait_cancelled = True
        if self.state != "feed":
            self.url = PLACE
        return self.handle


class CollectorResultNavigationTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, state, *, session=None, max_businesses=1):
        page = _ResultPage(state)
        browser = _Browser(page)
        session = session or MapsDiscoverySession()
        opened, progress, businesses = [], [], []

        async def extract(card_page, url, niche, city):
            opened.append(url)
            return Business(name=url, google_maps_url=url)

        async def on_progress(count):
            progress.append(count)

        self.page, self.browser = page, browser
        self.opened, self.progress, self.session = opened, progress, session
        with (
            patch.object(collector, "async_playwright",
                         return_value=_PlaywrightManager(browser)),
            patch.object(collector, "_extract_business", new=extract),
            patch.object(collector, "_random_delay", new=AsyncMock()),
        ):
            async for batch in collector.collect_stream(
                "niche", "city", max_businesses=max_businesses,
                max_scroll_rounds=1, progress_callback=on_progress,
                stop_flag=lambda: page.stopped, discovery_session=session,
            ):
                businesses.extend(batch)
        return businesses

    async def test_delayed_search_to_place_opens_once_and_reports_one(self):
        businesses = await self._run("delayed_place")
        self.assertEqual(len(businesses), 1)
        self.assertEqual(self.opened, [PLACE])
        self.assertEqual(self.progress, [1])
        self.assertEqual(self.session.maps_cards_actually_opened, 1)
        self.assertEqual(self.page.wait_timeout, 20000)
        self.page.handle.dispose.assert_awaited_once()
        self.assertTrue(self.browser.closed)

    async def test_immediate_place_does_not_require_feed(self):
        self.assertEqual(len(await self._run("immediate_place")), 1)
        self.assertEqual(self.opened, [PLACE])
        self.assertEqual(self.progress, [1])

    async def test_normal_feed_preserves_limit_and_duplicate_progress(self):
        self.assertEqual(len(await self._run("feed")), 1)
        self.assertEqual(self.opened, [PLACE])
        self.assertEqual(self.progress, [1])
        self.assertEqual(self.session.maps_cards_actually_opened, 1)

    async def test_multi_result_feed_opens_distinct_cards_with_exact_counters(self):
        self.assertEqual(len(await self._run("feed", max_businesses=2)), 2)
        self.assertEqual(self.opened, [PLACE, OTHER_PLACE])
        self.assertEqual(self.progress, [1, 2])
        self.assertEqual(self.session.maps_cards_actually_opened, 2)

    async def test_same_place_across_streams_is_not_reopened(self):
        session = MapsDiscoverySession()
        self.assertEqual(len(await self._run("delayed_place", session=session)), 1)
        self.assertEqual(await self._run("immediate_place", session=session), [])
        self.assertEqual(self.opened, [])
        self.assertEqual(self.progress, [])
        self.assertEqual(session.maps_cards_actually_opened, 1)
        self.assertEqual(session.maps_links_skipped_task_duplicate, 1)

    async def test_missing_state_raises_diagnosable_timeout_and_closes_browser(self):
        with self.assertRaisesRegex(PlaywrightTimeout, "Google Maps result state"):
            await self._run("missing")
        self.assertEqual(self.opened, [])
        self.assertEqual(self.progress, [])
        self.assertTrue(self.browser.closed)

    async def test_failed_navigation_is_not_successful_empty_search(self):
        with self.assertRaisesRegex(PlaywrightError, "ERR_CONNECTION_RESET"):
            await self._run("failed_navigation")
        self.assertEqual(self.opened, [])
        self.assertTrue(self.browser.closed)

    async def test_stop_during_pending_navigation_cancels_wait_without_open(self):
        self.assertEqual(await self._run("stop"), [])
        self.assertTrue(self.page.wait_cancelled)
        self.assertEqual(self.opened, [])
        self.assertEqual(self.progress, [])
        self.assertEqual(self.session.maps_cards_actually_opened, 0)
        self.assertTrue(self.browser.closed)

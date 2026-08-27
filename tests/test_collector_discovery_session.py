"""Offline collector tests for task-scoped pre-open Maps-link dedupe."""

import unittest
from unittest.mock import AsyncMock, patch

from agents import collector
from discovery_session import MapsDiscoverySession, normalize_maps_place_url
from models import Business


class _CountLocator:
    def __init__(self, count: int) -> None:
        self._count = count

    async def count(self) -> int:
        return self._count


class _Feed:
    def __init__(self) -> None:
        self.scrolls = 0

    async def wait_for(self, **kwargs) -> None:
        return None

    async def evaluate(self, script: str) -> None:
        self.scrolls += 1


class _FeedPage:
    def __init__(self, href_rounds: list[list[str]], *, end_visible: bool) -> None:
        self.url = ""
        self.href_rounds = href_rounds
        self.end_visible = end_visible
        self.eval_calls = 0
        self.feed = _Feed()

    async def goto(self, url: str, **kwargs) -> None:
        self.url = url

    def locator(self, selector: str):
        if selector == 'div[role="feed"]':
            return self.feed
        if selector == 'div[role="feed"] span.HlvSq':
            return _CountLocator(1 if self.end_visible else 0)
        return _CountLocator(0)

    async def eval_on_selector_all(self, selector: str, script: str):
        index = min(self.eval_calls, len(self.href_rounds) - 1)
        self.eval_calls += 1
        return list(self.href_rounds[index])


class _CardPage:
    url = ""


class _Context:
    def __init__(self, feed_page: _FeedPage) -> None:
        self.pages = [feed_page, _CardPage()]

    async def new_page(self):
        return self.pages.pop(0)


class _Browser:
    def __init__(self, feed_page: _FeedPage) -> None:
        self.context = _Context(feed_page)
        self.closed = False

    async def new_context(self, **kwargs):
        return self.context

    async def close(self) -> None:
        self.closed = True


class _Chromium:
    def __init__(self, browser: _Browser) -> None:
        self.browser = browser

    async def launch(self, **kwargs):
        return self.browser


class _Playwright:
    def __init__(self, browser: _Browser) -> None:
        self.chromium = _Chromium(browser)


class _PlaywrightManager:
    def __init__(self, browser: _Browser) -> None:
        self.playwright = _Playwright(browser)

    async def __aenter__(self):
        return self.playwright

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class CollectorDiscoverySessionTests(unittest.IsolatedAsyncioTestCase):
    async def _run_stream(
        self,
        href_rounds: list[list[str]],
        *,
        session: MapsDiscoverySession,
        stop_flag=None,
        max_scroll_rounds: int | None = None,
        end_visible: bool = False,
    ) -> dict:
        feed_page = _FeedPage(href_rounds, end_visible=end_visible)
        browser = _Browser(feed_page)
        opened: list[str] = []
        progress: list[int] = []
        batches: list[list[Business]] = []

        async def fake_extract(page, url, niche, city):
            opened.append(url)
            return Business(name=url, google_maps_url=url)

        async def on_progress(count: int) -> None:
            progress.append(count)

        with (
            patch.object(
                collector,
                "async_playwright",
                return_value=_PlaywrightManager(browser),
            ),
            patch.object(collector, "_extract_business", new=fake_extract),
            patch.object(collector, "_random_delay", new=AsyncMock()),
            patch.object(collector.asyncio, "sleep", new=AsyncMock()),
        ):
            async for batch in collector.collect_stream(
                "niche",
                "city",
                batch_size=100,
                max_businesses=100,
                max_scroll_rounds=max_scroll_rounds or len(href_rounds),
                progress_callback=on_progress,
                stop_flag=stop_flag,
                discovery_session=session,
            ):
                batches.append(batch)

        return {
            "opened": opened,
            "progress": progress,
            "batches": batches,
            "feed_page": feed_page,
            "browser": browser,
        }

    async def test_same_href_twice_in_one_stream_opens_once(self) -> None:
        href = "https://www.google.com/maps/place/Alpha?entry=ttu"
        session = MapsDiscoverySession()

        run = await self._run_stream([[href, href]], session=session)

        self.assertEqual(run["opened"], [href])
        self.assertEqual(run["progress"], [1])
        self.assertEqual(session.maps_cards_actually_opened, 1)

    async def test_same_href_across_query_streams_skips_before_open(self) -> None:
        href = "https://www.google.com/maps/place/Alpha?entry=ttu"
        session = MapsDiscoverySession()

        first = await self._run_stream([[href]], session=session)
        second = await self._run_stream([[href]], session=session)

        self.assertEqual(first["opened"], [href])
        self.assertEqual(second["opened"], [])
        self.assertEqual(second["progress"], [])
        self.assertEqual(session.maps_links_discovered, 2)
        self.assertEqual(session.maps_links_skipped_task_duplicate, 1)
        self.assertEqual(session.maps_cards_actually_opened, 1)

    async def test_distinct_links_still_open(self) -> None:
        links = [
            "https://www.google.com/maps/place/Alpha",
            "https://www.google.com/maps/place/Beta",
        ]
        session = MapsDiscoverySession()

        run = await self._run_stream([links], session=session)

        self.assertEqual(run["opened"], links)
        self.assertEqual(run["progress"], [1, 2])
        self.assertEqual(session.maps_cards_actually_opened, 2)

    async def test_stop_flag_prevents_card_opening(self) -> None:
        href = "https://www.google.com/maps/place/Alpha"
        session = MapsDiscoverySession()

        run = await self._run_stream(
            [[href]],
            session=session,
            stop_flag=lambda: True,
        )

        self.assertEqual(run["opened"], [])
        self.assertEqual(session.maps_cards_actually_opened, 0)
        self.assertTrue(run["browser"].closed)

    async def test_stale_rounds_and_explicit_end_remain_bounded(self) -> None:
        href = "https://www.google.com/maps/place/Alpha"
        session = MapsDiscoverySession()
        with patch.object(collector.config, "COLLECT_STALE_ROUNDS", 2):
            stale = await self._run_stream(
                [[href], [href], [href], [href]],
                session=session,
                max_scroll_rounds=10,
            )
        ended = await self._run_stream(
            [["https://www.google.com/maps/place/Beta"]],
            session=MapsDiscoverySession(),
            max_scroll_rounds=10,
            end_visible=True,
        )

        self.assertEqual(stale["feed_page"].eval_calls, 3)
        self.assertEqual(stale["opened"], [href])
        self.assertEqual(ended["feed_page"].eval_calls, 1)

    async def test_separate_tasks_have_separate_link_sessions(self) -> None:
        href = "https://www.google.com/maps/place/Alpha"

        first = await self._run_stream([[href]], session=MapsDiscoverySession())
        second = await self._run_stream([[href]], session=MapsDiscoverySession())

        self.assertEqual(first["opened"], [href])
        self.assertEqual(second["opened"], [href])


class MapsLinkNormalizationTests(unittest.TestCase):
    def test_disabled_session_preserves_cross_stream_open_behavior(self) -> None:
        session = MapsDiscoverySession(dedupe_enabled=False)
        href = "https://google.com/maps/place/Alpha"

        self.assertTrue(session.claim_link(href))
        self.assertTrue(session.claim_link(href))
        self.assertEqual(session.maps_links_discovered, 2)
        self.assertEqual(session.maps_links_skipped_task_duplicate, 0)

    def test_ignores_proven_tracking_noise_for_maps_place_urls(self) -> None:
        first = (
            "https://www.google.com/maps/place/Alpha/"
            "?entry=ttu&hl=uk&utm_source=test#gibberish"
        )
        second = "https://google.com/maps/place/Alpha?g_ep=tracking"

        self.assertEqual(
            normalize_maps_place_url(first),
            normalize_maps_place_url(second),
        )

    def test_preserves_identity_bearing_query_values(self) -> None:
        first = "https://google.com/maps/place/Alpha?q=place_id%3Aone"
        second = "https://google.com/maps/place/Alpha?q=place_id%3Atwo"

        self.assertNotEqual(
            normalize_maps_place_url(first),
            normalize_maps_place_url(second),
        )

    def test_malformed_values_fail_safe_without_collapsing(self) -> None:
        self.assertNotEqual(
            normalize_maps_place_url("not a maps url one"),
            normalize_maps_place_url("not a maps url two"),
        )
        self.assertEqual(
            normalize_maps_place_url(" not a maps url one "),
            "raw:not a maps url one",
        )


if __name__ == "__main__":
    unittest.main()

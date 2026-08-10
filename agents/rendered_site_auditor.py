"""Playwright desktop/mobile renderer with DNS and request-routing safeguards."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ipaddress
import socket
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from playwright.async_api import TimeoutError as PlaywrightTimeout, async_playwright

import config
from rendered_site_audit import (
    RenderedSiteAuditResult,
    RenderedSiteAuditRuntime,
    RenderedViewportMetrics,
    classify_rendered_metrics,
    rendered_technical_error,
    rendered_uncertain,
)
from website_resolution import normalize_candidate_url


DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Mobile Safari/537.36"
)


_DOM_METRICS_SCRIPT = r"""
({ viewportWidth, viewportHeight }) => {
  const body = document.body;
  const html = document.documentElement;
  const innerWidth = Math.max(0, Math.round(window.innerWidth || 0));
  const isVisible = (element) => {
    if (!(element instanceof Element)) return false;
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" ||
        Number.parseFloat(style.opacity || "1") === 0) return false;
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 && rect.bottom > 0 &&
      rect.top < window.innerHeight;
  };

  let visibleTextLength = 0;
  let sampledText = 0;
  let tinyText = 0;
  let visitedTextElements = 0;
  if (body) {
    const walker = document.createTreeWalker(body, NodeFilter.SHOW_ELEMENT);
    let element = walker.currentNode;
    while (element && sampledText < 300 && visitedTextElements < 3000) {
      visitedTextElements += 1;
      if (isVisible(element)) {
        const pieces = [];
        const childLimit = Math.min(element.childNodes.length, 50);
        for (let index = 0; index < childLimit; index += 1) {
          const node = element.childNodes[index];
          if (node.nodeType === Node.TEXT_NODE) pieces.push(node.textContent || "");
        }
        const ownText = pieces.join(" ").replace(/\s+/g, " ").trim();
        if (ownText) {
          sampledText += 1;
          visibleTextLength = Math.min(20000, visibleTextLength + ownText.length);
          const size = Number.parseFloat(window.getComputedStyle(element).fontSize || "0");
          if (Number.isFinite(size) && size < 12) tinyText += 1;
        }
      }
      element = walker.nextNode();
    }
  }

  const major = [];
  const seen = new Set();
  const addMajor = (element) => {
    if (element && !seen.has(element) && major.length < 300) {
      seen.add(element);
      major.push(element);
    }
  };
  let visitedMajorElements = 0;
  if (body) {
    const majorWalker = document.createTreeWalker(body, NodeFilter.SHOW_ELEMENT);
    let element = majorWalker.currentNode;
    while (element && major.length < 300 && visitedMajorElements < 3000) {
      visitedMajorElements += 1;
      const parent = element.parentElement;
      if (
        element.matches("header,nav,main,section,article,form,table,iframe") ||
        parent === body || (parent && parent.tagName === "MAIN")
      ) addMajor(element);
      element = majorWalker.nextNode();
    }
  }
  let majorOverflow = 0;
  for (const element of major) {
    if (!isVisible(element)) continue;
    const rect = element.getBoundingClientRect();
    if (rect.left < -16 || rect.right > innerWidth + 16) majorOverflow += 1;
  }

  let visibleImages = 0;
  let brokenImages = 0;
  const imageLimit = Math.min((document.images || []).length, 300);
  for (let index = 0; index < imageLimit; index += 1) {
    const image = document.images[index];
    if (!isVisible(image)) continue;
    visibleImages += 1;
    if (image.complete && image.naturalWidth === 0) brokenImages += 1;
  }

  const navigation = performance.getEntriesByType("navigation")[0];
  const domLoaded = navigation && Number.isFinite(navigation.domContentLoadedEventEnd)
    ? Math.max(0, Math.round(navigation.domContentLoadedEventEnd))
    : null;
  return {
    viewport_width: viewportWidth,
    viewport_height: viewportHeight,
    inner_width: innerWidth,
    document_scroll_width: Math.max(0, Math.round(html ? html.scrollWidth : 0)),
    body_scroll_width: Math.max(0, Math.round(body ? body.scrollWidth : 0)),
    visible_text_length: visibleTextLength,
    visible_image_count: visibleImages,
    broken_visible_image_count: brokenImages,
    major_overflow_element_count: majorOverflow,
    tiny_text_element_count: tinyText,
    sampled_text_element_count: sampledText,
    viewport_meta_present: Boolean(document.querySelector('meta[name="viewport" i]')),
    dom_content_loaded_ms: domLoaded,
  };
}
"""


@dataclass(frozen=True)
class RenderedSiteAuditorSettings:
    timeout_seconds: float
    settle_milliseconds: int
    max_hosts_per_page: int

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_seconds, (int, float)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(self.settle_milliseconds) is not int or self.settle_milliseconds < 0:
            raise ValueError("settle_milliseconds must be a non-negative integer")
        if type(self.max_hosts_per_page) is not int or self.max_hosts_per_page < 1:
            raise ValueError("max_hosts_per_page must be a positive integer")


def is_public_ip_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        address.is_global
        and not address.is_loopback
        and not address.is_private
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


async def resolve_public_host(
    hostname: str,
    *,
    resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
) -> bool:
    """Require every DNS answer to be globally routable."""
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        return is_public_ip_address(str(literal))
    try:
        answers = await asyncio.to_thread(
            resolver,
            hostname,
            None,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )
    except (OSError, UnicodeError):
        return False
    addresses = {
        answer[4][0]
        for answer in answers
        if len(answer) >= 5 and answer[4]
    }
    return bool(addresses) and all(is_public_ip_address(address) for address in addresses)


def _host_identity(hostname: str) -> str:
    normalized = hostname.rstrip(".").casefold()
    return normalized[4:] if normalized.startswith("www.") else normalized


@dataclass(frozen=True)
class _ViewportResult:
    metrics: RenderedViewportMetrics
    final_host_identity: str
    final_https: bool


class _UnsafeTopLevelRequest(RuntimeError):
    pass


class _CrossDomainRedirect(RuntimeError):
    pass


class PageRequestSafetyGuard:
    """Per-page hostname cap backed by a task-scoped DNS validator/cache."""

    def __init__(
        self,
        host_is_public: Callable[[str], Awaitable[bool]],
        *,
        max_hosts: int,
    ) -> None:
        self._host_is_public = host_is_public
        self._max_hosts = max_hosts
        self._distinct_hosts: set[str] = set()

    async def permits(self, url: str, *, top_level: bool) -> bool:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return False
        if parsed.scheme in {"data", "blob"}:
            return not top_level
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            return False
        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname in self._distinct_hosts:
            return True
        if len(self._distinct_hosts) >= self._max_hosts:
            return False
        if not await self._host_is_public(hostname):
            return False
        self._distinct_hosts.add(hostname)
        return True


class PlaywrightRenderedSiteAuditor:
    """Task-scoped browser owner; each audit uses fresh desktop/mobile contexts."""

    def __init__(
        self,
        settings: RenderedSiteAuditorSettings,
        *,
        dns_resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
    ) -> None:
        self._settings = settings
        self._dns_resolver = dns_resolver
        self._dns_cache: dict[str, bool] = {}
        self._dns_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._playwright = None
        self._browser = None

    async def _host_is_public(self, hostname: str) -> bool:
        key = hostname.rstrip(".").casefold()
        async with self._dns_lock:
            if key not in self._dns_cache:
                try:
                    self._dns_cache[key] = await asyncio.wait_for(
                        resolve_public_host(key, resolver=self._dns_resolver),
                        timeout=self._settings.timeout_seconds,
                    )
                except TimeoutError:
                    self._dns_cache[key] = False
            return self._dns_cache[key]

    async def _ensure_browser(self):
        async with self._start_lock:
            if self._browser is None:
                playwright = await async_playwright().start()
                try:
                    browser = await playwright.chromium.launch(headless=True)
                except BaseException:
                    await playwright.stop()
                    raise
                self._playwright = playwright
                self._browser = browser
        return self._browser

    async def _render_viewport(
        self,
        url: str,
        *,
        viewport_width: int,
        viewport_height: int,
        mobile: bool,
        allowed_host_identity: str,
    ) -> _ViewportResult:
        browser = await self._ensure_browser()
        context_options: dict[str, object] = {
            "viewport": {"width": viewport_width, "height": viewport_height},
            "user_agent": MOBILE_USER_AGENT if mobile else DESKTOP_USER_AGENT,
            "is_mobile": mobile,
            "has_touch": mobile,
            "device_scale_factor": 1,
            "accept_downloads": False,
            "service_workers": "block",
        }
        context = await browser.new_context(**context_options)
        page = None
        request_guard = PageRequestSafetyGuard(
            self._host_is_public,
            max_hosts=self._settings.max_hosts_per_page,
        )
        unsafe_top_level = False
        try:
            page = await context.new_page()

            async def route_request(route, request) -> None:
                nonlocal unsafe_top_level
                try:
                    top_level = bool(
                        request.is_navigation_request()
                        and request.frame == page.main_frame
                    )
                    if await request_guard.permits(request.url, top_level=top_level):
                        await route.continue_()
                    else:
                        if top_level:
                            unsafe_top_level = True
                        await route.abort("blockedbyclient")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if request.is_navigation_request() and request.frame == page.main_frame:
                        unsafe_top_level = True
                    await route.abort("blockedbyclient")

            await context.route("**/*", route_request)
            try:
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=int(self._settings.timeout_seconds * 1000),
                )
            except Exception:
                if unsafe_top_level:
                    raise _UnsafeTopLevelRequest from None
                raise
            if response is None:
                raise RuntimeError("navigation_without_response")
            final_url = normalize_candidate_url(page.url)
            final_parts = urlsplit(final_url)
            assert final_parts.hostname is not None
            final_identity = _host_identity(final_parts.hostname)
            if final_identity != allowed_host_identity:
                raise _CrossDomainRedirect
            await page.wait_for_timeout(self._settings.settle_milliseconds)
            raw_metrics = await page.evaluate(
                _DOM_METRICS_SCRIPT,
                {
                    "viewportWidth": viewport_width,
                    "viewportHeight": viewport_height,
                },
            )
            return _ViewportResult(
                metrics=RenderedViewportMetrics(**raw_metrics),
                final_host_identity=final_identity,
                final_https=final_parts.scheme == "https",
            )
        finally:
            await context.close()

    async def audit(self, url: str) -> RenderedSiteAuditResult:
        try:
            normalized_url = normalize_candidate_url(url)
        except (TypeError, ValueError):
            return rendered_technical_error("invalid_top_level_url")
        parsed = urlsplit(normalized_url)
        assert parsed.hostname is not None
        if not await self._host_is_public(parsed.hostname):
            return rendered_technical_error("unsafe_top_level_host")
        allowed_host_identity = _host_identity(parsed.hostname)

        attempted = 0
        succeeded = 0
        viewport_results: list[_ViewportResult] = []
        for width, height, mobile in ((1366, 768, False), (390, 844, True)):
            attempted += 1
            try:
                viewport_result = await asyncio.wait_for(
                    self._render_viewport(
                        normalized_url,
                        viewport_width=width,
                        viewport_height=height,
                        mobile=mobile,
                        allowed_host_identity=allowed_host_identity,
                    ),
                    timeout=(
                        self._settings.timeout_seconds
                        + self._settings.settle_milliseconds / 1000
                        + 1
                    ),
                )
            except _CrossDomainRedirect:
                return rendered_uncertain(
                    "cross_domain_redirect",
                    pages_attempted=attempted,
                    pages_succeeded=succeeded + 1,
                )
            except _UnsafeTopLevelRequest:
                return rendered_technical_error(
                    "unsafe_top_level_request",
                    pages_attempted=attempted,
                    pages_succeeded=succeeded,
                )
            except (PlaywrightTimeout, TimeoutError):
                return rendered_technical_error(
                    "timeout",
                    pages_attempted=attempted,
                    pages_succeeded=succeeded,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return rendered_technical_error(
                    "browser_or_network_error",
                    pages_attempted=attempted,
                    pages_succeeded=succeeded,
                )
            viewport_results.append(viewport_result)
            succeeded += 1

        return classify_rendered_metrics(
            viewport_results[0].metrics,
            viewport_results[1].metrics,
            https_final_url=all(result.final_https for result in viewport_results),
            pages_attempted=attempted,
            pages_succeeded=succeeded,
        )

    async def close(self) -> None:
        browser, self._browser = self._browser, None
        playwright, self._playwright = self._playwright, None
        try:
            if browser is not None:
                await browser.close()
        finally:
            if playwright is not None:
                await playwright.stop()


def build_configured_rendered_site_audit_runtime() -> RenderedSiteAuditRuntime:
    auditor = PlaywrightRenderedSiteAuditor(
        RenderedSiteAuditorSettings(
            timeout_seconds=config.RENDERED_SITE_AUDIT_TIMEOUT_SECONDS,
            settle_milliseconds=config.RENDERED_SITE_AUDIT_SETTLE_MILLISECONDS,
            max_hosts_per_page=config.RENDERED_SITE_AUDIT_MAX_HOSTS_PER_PAGE,
        )
    )
    return RenderedSiteAuditRuntime(
        auditor,
        max_bad_audits=config.MAX_RENDERED_BAD_SITE_AUDITS_PER_TASK,
        max_good_audits=config.MAX_RENDERED_GOOD_SITE_AUDITS_PER_TASK,
        concurrency=config.RENDERED_SITE_AUDIT_CONCURRENCY,
    )

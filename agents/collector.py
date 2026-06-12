# -*- coding: utf-8 -*-
"""Collector — парсинг Google Maps через Playwright (headless, без API).

Открывает поиск "{ниша} {город}" на maps.google.com, скроллит список слева,
заходит в каждую карточку и собирает: название, телефон, адрес, сайт,
Instagram, рейтинг и количество отзывов.
"""

import asyncio
import random
import re
import urllib.parse
from typing import Awaitable, Callable, List, Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout, async_playwright

import config
from models import Business

# Меняем User-Agent, чтобы меньше походить на бота
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

# Колбэк прогресса: принимает (собрано, всего)
ProgressCallback = Callable[[int, int], Awaitable[None]]


async def _random_delay(min_s: float = None, max_s: float = None) -> None:
    """Случайная задержка между действиями, чтобы Google не заблокировал."""
    min_s = min_s if min_s is not None else config.COLLECT_DELAY_MIN
    max_s = max_s if max_s is not None else config.COLLECT_DELAY_MAX
    await asyncio.sleep(random.uniform(min_s, max_s))


async def _accept_consent(page: Page) -> None:
    """Закрыть экран согласия Google (cookies), если он появился."""
    if "consent" not in page.url:
        return
    for selector in (
        'button[aria-label*="Прийняти"]',
        'button[aria-label*="Accept all"]',
        'form[action*="consent"] button',
        'button:has-text("Прийняти все")',
        'button:has-text("Accept all")',
    ):
        try:
            await page.click(selector, timeout=3000)
            await page.wait_for_load_state("domcontentloaded")
            return
        except PlaywrightTimeout:
            continue
        except Exception:
            continue


async def _scroll_results(page: Page, count: int) -> List[str]:
    """Скроллить список результатов слева, пока не наберём count ссылок на карточки."""
    feed = page.locator('div[role="feed"]')
    await feed.wait_for(state="visible", timeout=20000)

    links: List[str] = []
    stale_rounds = 0  # сколько скроллов подряд без новых результатов

    while len(links) < count and stale_rounds < 8:
        # Собираем ссылки на карточки бизнесов (a.hfpxzc — ссылка карточки в списке)
        hrefs = await page.eval_on_selector_all(
            'div[role="feed"] a.hfpxzc', "els => els.map(e => e.href)"
        )
        new_links = [h for h in hrefs if h not in links]
        if new_links:
            links.extend(new_links)
            stale_rounds = 0
        else:
            stale_rounds += 1

        if len(links) >= count:
            break

        # Скроллим панель результатов вниз
        await feed.evaluate("el => el.scrollBy(0, el.scrollHeight)")
        await asyncio.sleep(random.uniform(1.5, 3.0))

        # Если Google показал "конец списка" — выходим
        try:
            end = await page.locator('div[role="feed"] span.HlvSq').count()
            if end > 0:
                hrefs = await page.eval_on_selector_all(
                    'div[role="feed"] a.hfpxzc', "els => els.map(e => e.href)"
                )
                links = list(dict.fromkeys(hrefs))
                break
        except Exception:
            pass

    return links[:count]


async def _extract_business(page: Page, url: str, niche: str, city: str) -> Optional[Business]:
    """Открыть карточку бизнеса и собрать данные."""
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await _accept_consent(page)

    b = Business(niche=niche, city=city)

    # Название
    try:
        await page.wait_for_selector("h1.DUwDvf, h1", timeout=15000)
        b.name = (await page.locator("h1.DUwDvf, h1").first.inner_text()).strip()
    except Exception:
        return None  # без названия карточка бесполезна

    # Телефон — зашит в data-item-id кнопки вида "phone:tel:+380..."
    try:
        phone_btn = page.locator('button[data-item-id^="phone:tel:"]').first
        if await phone_btn.count() > 0:
            item_id = await phone_btn.get_attribute("data-item-id")
            b.phone = item_id.replace("phone:tel:", "").strip()
    except Exception:
        pass

    # Адрес
    try:
        addr_btn = page.locator('button[data-item-id="address"]').first
        if await addr_btn.count() > 0:
            label = await addr_btn.get_attribute("aria-label") or ""
            b.address = label.split(":", 1)[-1].strip()
    except Exception:
        pass

    # Сайт (ссылка "authority" в профиле)
    try:
        site_link = page.locator('a[data-item-id="authority"]').first
        if await site_link.count() > 0:
            b.website = (await site_link.get_attribute("href") or "").strip()
    except Exception:
        pass

    # Instagram: либо сайт — это Instagram, либо ссылка есть на странице
    try:
        if "instagram.com" in b.website:
            b.instagram_url = b.website
            b.website = ""  # Instagram — не сайт
        else:
            ig = await page.eval_on_selector_all(
                'a[href*="instagram.com"]', "els => els.map(e => e.href)"
            )
            if ig:
                b.instagram_url = ig[0]
    except Exception:
        pass

    # Рейтинг и количество отзывов (блок F7nice: "4,9" и "(123)")
    try:
        rating_block = page.locator("div.F7nice").first
        if await rating_block.count() > 0:
            text = await rating_block.inner_text()
            m = re.search(r"(\d+[.,]\d+)", text)
            if m:
                b.rating = float(m.group(1).replace(",", "."))
            m = re.search(r"\(([\d\s ]+)\)", text)
            if m:
                b.reviews_count = int(re.sub(r"\D", "", m.group(1)))
    except Exception:
        pass

    return b


async def collect(
    niche: str,
    city: str,
    count: int,
    progress_callback: Optional[ProgressCallback] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> List[Business]:
    """Собрать до count бизнесов из Google Maps по запросу "{niche} {city}".

    progress_callback(собрано, всего) — вызывается после каждой карточки.
    stop_flag() -> True — досрочная остановка (команда /stop).
    """
    count = min(count, config.MAX_BUSINESSES)  # не больше 200 за запуск
    query = urllib.parse.quote(f"{niche} {city}")
    url = f"https://www.google.com/maps/search/{query}?hl=uk"

    businesses: List[Business] = []
    seen_phones = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="uk-UA",
            viewport={"width": 1366, "height": 900},
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await _accept_consent(page)

            # Иногда Maps сразу открывает одну карточку вместо списка
            if "/maps/place/" in page.url:
                b = await _extract_business(page, page.url, niche, city)
                if b:
                    businesses.append(b)
                return businesses

            links = await _scroll_results(page, count)

            for i, link in enumerate(links):
                if stop_flag and stop_flag():
                    break
                try:
                    b = await _extract_business(page, link, niche, city)
                except Exception:
                    continue  # карточка не открылась — идём дальше
                if b is None:
                    continue
                # Дубликаты по телефону пропускаем
                if b.phone and b.phone in seen_phones:
                    continue
                if b.phone:
                    seen_phones.add(b.phone)
                businesses.append(b)

                if progress_callback:
                    await progress_callback(len(businesses), count)
                if len(businesses) >= count:
                    break

                await _random_delay()  # пауза 3-6 сек между карточками
        finally:
            await browser.close()

    return businesses


if __name__ == "__main__":
    # Ручной запуск: python -m agents.collector "салон краси" "Харків" 10
    import sys

    niche_arg = sys.argv[1] if len(sys.argv) > 1 else "салон краси"
    city_arg = sys.argv[2] if len(sys.argv) > 2 else "Харків"
    count_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    result = asyncio.run(collect(niche_arg, city_arg, count_arg))
    for biz in result:
        print(f"- {biz.name} | {biz.phone} | {biz.address} | сайт: {biz.website or '—'} "
              f"| IG: {biz.instagram_url or '—'} | ★{biz.rating} ({biz.reviews_count})")
    print(f"Всего собрано: {len(result)}")

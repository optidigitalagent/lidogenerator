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
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, List, Optional

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


async def _extract_business(page: Page, url: str, niche: str, city: str) -> Optional[Business]:
    """Открыть карточку бизнеса и собрать данные."""
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    await _accept_consent(page)

    b = Business(
        niche=niche,
        city=city,
        google_maps_url=url,
        collected_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

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


def _dedup_key(b: Business) -> tuple:
    """Ключ дедупликации бизнеса: название + адрес + Instagram URL.

    Защищает от дублей, которые Google Maps иногда показывает несколько раз
    (один и тот же бизнес под разными ссылками карточки).
    """
    return (
        b.name.strip().lower(),
        b.address.strip().lower(),
        b.instagram_url.strip().lower(),
    )


# Колбэк потокового сбора: принимает (просмотрено бизнесов).
StreamProgressCallback = Callable[[int], Awaitable[None]]


def _maps_search_url(niche: str, city: str, query_text: str | None = None) -> str:
    """Build the Maps URL, preserving the legacy niche/city query by default."""
    search_text = f"{niche} {city}" if query_text is None else query_text
    query = urllib.parse.quote(search_text)
    return f"https://www.google.com/maps/search/{query}?hl=uk"


async def collect_stream(
    niche: str,
    city: str,
    batch_size: Optional[int] = None,
    max_businesses: Optional[int] = None,
    max_scroll_rounds: Optional[int] = None,
    progress_callback: Optional[StreamProgressCallback] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
    query_text: str | None = None,
) -> AsyncIterator[List[Business]]:
    """Асинхронный генератор: отдаёт бизнесы из Google Maps батчами.

    Держит браузер открытым: одна вкладка (feed_page) остаётся на странице
    результатов и постепенно прокручивается, вторая (card_page) поочерёдно
    открывает карточки бизнесов. После накопления batch_size уникальных
    бизнесов отдаёт их батчем (yield) и продолжает. Последний батч может быть
    меньше batch_size.

    Останавливается, когда:
      - просмотрено max_businesses бизнесов (safety-лимит);
      - сделано max_scroll_rounds прокруток списка (safety-лимит);
      - COLLECT_STALE_ROUNDS прокруток подряд не дали новых карточек
        (результаты Google Maps закончились);
      - сработал stop_flag().

    Дубликаты (название+адрес+Instagram, а также повтор телефона) пропускаются.
    progress_callback(просмотрено) — вызывается после каждой открытой карточки.
    """
    batch_size = batch_size or config.COLLECT_BATCH_SIZE
    max_businesses = max_businesses or config.MAX_BUSINESSES_PER_SEARCH
    max_scroll_rounds = max_scroll_rounds or config.MAX_SCROLL_ROUNDS

    url = _maps_search_url(niche, city, query_text)

    visited_links: set = set()  # ссылки карточек, которые уже открывали
    seen_keys: set = set()      # ключи дедупа (название+адрес+IG)
    seen_phones: set = set()    # телефоны (доп. защита от дублей)
    visited = 0                 # сколько карточек реально открыли

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            locale="uk-UA",
            viewport={"width": 1366, "height": 900},
        )
        feed_page = await context.new_page()   # держит список результатов
        card_page = await context.new_page()   # открывает отдельные карточки

        try:
            await feed_page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await _accept_consent(feed_page)

            # Иногда Maps сразу открывает одну карточку вместо списка
            if "/maps/place/" in feed_page.url:
                b = await _extract_business(card_page, feed_page.url, niche, city)
                visited += 1
                if progress_callback:
                    await progress_callback(visited)
                if b:
                    yield [b]
                return

            feed = feed_page.locator('div[role="feed"]')
            await feed.wait_for(state="visible", timeout=20000)

            stale_rounds = 0
            batch: List[Business] = []

            for _ in range(max_scroll_rounds):
                if (stop_flag and stop_flag()) or visited >= max_businesses:
                    break

                # Текущие загруженные ссылки карточек
                hrefs = await feed_page.eval_on_selector_all(
                    'div[role="feed"] a.hfpxzc', "els => els.map(e => e.href)"
                )
                new_links = [h for h in hrefs if h not in visited_links]
                stale_rounds = 0 if new_links else stale_rounds + 1

                # Открываем новые карточки на отдельной вкладке
                for link in new_links:
                    if (stop_flag and stop_flag()) or visited >= max_businesses:
                        break
                    visited_links.add(link)
                    visited += 1
                    try:
                        b = await _extract_business(card_page, link, niche, city)
                    except Exception:
                        b = None
                    if progress_callback:
                        await progress_callback(visited)
                    if b is None:
                        continue
                    # Защита от дублей по названию+адресу+IG и по телефону
                    key = _dedup_key(b)
                    if key in seen_keys:
                        continue
                    if b.phone and b.phone in seen_phones:
                        continue
                    seen_keys.add(key)
                    if b.phone:
                        seen_phones.add(b.phone)

                    batch.append(b)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                    await _random_delay()  # пауза 3-6 сек между карточками

                if visited >= max_businesses:
                    break

                # Google показал явный конец списка
                try:
                    if await feed_page.locator('div[role="feed"] span.HlvSq').count() > 0:
                        break
                except Exception:
                    pass

                # Новые результаты больше не появляются — список исчерпан
                if stale_rounds >= config.COLLECT_STALE_ROUNDS:
                    break

                # Скроллим список вниз и ждём подгрузки
                await feed.evaluate("el => el.scrollBy(0, el.scrollHeight)")
                await asyncio.sleep(random.uniform(1.5, 3.0))

            # Остаток (меньше batch_size) тоже отдаём
            if batch:
                yield batch
        finally:
            await browser.close()


async def collect(
    niche: str,
    city: str,
    count: int,
    progress_callback: Optional[ProgressCallback] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> List[Business]:
    """Устаревший способ: собрать до count БИЗНЕСОВ (не лидов) из Google Maps.

    Оставлен для совместимости (ручной запуск, старые тесты). Основной поток
    теперь использует collect_stream + логику target_leads в оркестраторе.

    progress_callback(собрано, всего) — вызывается после каждой карточки.
    """
    count = min(count, config.MAX_BUSINESSES)
    businesses: List[Business] = []
    async for batch in collect_stream(
        niche, city, max_businesses=count, stop_flag=stop_flag
    ):
        for b in batch:
            businesses.append(b)
            if progress_callback:
                await progress_callback(len(businesses), count)
            if len(businesses) >= count:
                return businesses[:count]
    return businesses[:count]


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

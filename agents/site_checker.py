# -*- coding: utf-8 -*-
"""Site Checker — проверка наличия И качества сайтов.

Для каждого бизнеса:
  - сайт не указан → has_site=False, site_quality="none"  (лид: no website)
  - "сайт" — это ссылка на соцсеть/агрегатор (Facebook, linktr.ee, Booksy...)
    → has_site=False, site_quality="none"  (настоящего сайта нет)
  - сайт не открывается / 4xx / 5xx → has_site=False, site_quality="dead" (no website)
  - сайт открывается → эвристический вердикт по содержимому страницы:
      site_quality="bad"  — явные признаки старого/слабого сайта (лид: bad website)
      site_quality="good" — признаков проблем нет (лид отсеивается)

Запросы асинхронные, не более 5 одновременно.
"""

import asyncio
import re
import time
from typing import Awaitable, Callable, List, Optional
from urllib.parse import urlparse

import httpx

import config
from models import Business

ProgressCallback = Callable[[int, int], Awaitable[None]]

HEADERS = {
    # Представляемся обычным браузером, чтобы сайты не резали запросы
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "uk-UA,uk;q=0.9,ru;q=0.8,en;q=0.7",
}

# Домены, которые НЕ являются собственным сайтом бизнеса:
# соцсети, мессенджеры, линк-агрегаторы, маркетплейсы, записочные сервисы.
NOT_A_WEBSITE_DOMAINS = (
    "facebook.com", "m.facebook.com", "fb.com", "instagram.com", "tiktok.com",
    "youtube.com", "t.me", "telegram.me", "wa.me", "api.whatsapp.com",
    "viber.com", "invite.viber.com",
    "linktr.ee", "taplink.cc", "taplink.ws", "lnk.bio", "linkin.bio", "mssg.me",
    "booksy.com", "n716.alteg.io", "alteg.io", "easyweek.com.ua", "dikidi.net",
    "olx.ua", "prom.ua", "rozetka.com.ua", "business.site", "google.com",
    "goo.gl", "maps.app.goo.gl",
)

# Маркеры устаревших технологий / бесплатных конструкторов 2000-х в HTML
OLD_TECH_MARKERS = (
    "<frameset", "<frame ", "<marquee", "<font ", "<center>",
    "macromedia", "shockwave", ".swf", "ucoz", "narod.ru", "ucoz.ua",
    "msothemes", "frontpage.editor", "generator\" content=\"microsoft",
    "wordpress 3.", "wordpress 4.", "joomla! 1.", "joomla! 2.",
)


def _is_real_website(url: str) -> bool:
    """Ссылка ведёт на собственный сайт, а не на соцсеть/агрегатор?"""
    host = (urlparse(url if "://" in url else "https://" + url).hostname or "").lower()
    host = host.removeprefix("www.")
    return bool(host) and not any(
        host == d or host.endswith("." + d) for d in NOT_A_WEBSITE_DOMAINS
    )


def _visible_text(html: str) -> str:
    """Грубо вытащить видимый текст страницы (без тегов, скриптов, стилей)."""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _analyze(html: str, final_url: str, elapsed: float) -> str:
    """Эвристическая оценка качества сайта. Возвращает "bad" / "good".

    Считаем штрафные баллы за признаки слабого/старого сайта. >=3 балла → "bad".
    При сомнениях — "good": лучше пропустить бизнес, чем добавить лишний
    (главное требование заказчика — никакого мусора в таблице).
    """
    low = html.lower()
    text = _visible_text(html)
    points = 0

    # Нет мобильной адаптации — главный признак устаревшего сайта
    if 'name="viewport"' not in low and "name='viewport'" not in low:
        points += 3

    # Устаревшие технологии / конструкторы 2000-х
    if any(m in low for m in OLD_TECH_MARKERS):
        points += 3

    # Нет HTTPS
    if final_url.startswith("http://"):
        points += 2

    # Нет заголовка title
    if not re.search(r"(?is)<title[^>]*>(.*?)</title>", html):
        points += 1

    # Почти пустая страница (заглушка / недоделанный сайт)
    if len(text) < 300:
        points += 2

    # Очень медленный
    if elapsed > 8:
        points += 1

    return "bad" if points >= 3 else "good"


async def _check_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, b: Business) -> None:
    """Проверить один бизнес. Результат пишется прямо в объект."""
    if not b.website or not _is_real_website(b.website):
        # Сайта нет, либо вместо сайта — соцсеть/агрегатор (booksy, linktr.ee...)
        b.has_site = False
        b.site_quality = "none"
        return

    url = b.website
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    async with sem:  # не более 5 одновременных запросов
        try:
            start = time.monotonic()
            resp = await client.get(url)
            elapsed = time.monotonic() - start
            if resp.status_code < 400:
                b.has_site = True
                b.site_quality = _analyze(resp.text, str(resp.url), elapsed)
            else:
                # Сайт указан, но отдаёт ошибку — фактически сайта нет
                b.has_site = False
                b.site_quality = "dead"
        except Exception:
            b.has_site = False
            b.site_quality = "dead"


async def check_sites(
    businesses: List[Business],
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Business]:
    """Проверить сайты у всех бизнесов. Возвращает тот же список с обновлёнными полями."""
    sem = asyncio.Semaphore(config.SITE_CHECK_CONCURRENCY)
    done = 0

    async with httpx.AsyncClient(
        timeout=config.SITE_CHECK_TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
        verify=False,  # самоподписанные сертификаты не повод считать сайт мёртвым
    ) as client:
        async def worker(b: Business):
            nonlocal done
            await _check_one(client, sem, b)
            done += 1
            if progress_callback:
                await progress_callback(done, len(businesses))

        await asyncio.gather(*(worker(b) for b in businesses))

    return businesses


if __name__ == "__main__":
    # Ручной запуск на тестовых данных
    import json
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads((Path(__file__).parent.parent / "tests" / "stage2_result.json").read_text(encoding="utf-8"))
    items = [Business(**d) for d in data]
    asyncio.run(check_sites(items))
    for biz in items:
        print(f"- {biz.name}: {biz.website_status} (has_site={biz.has_site}, quality={biz.site_quality})")

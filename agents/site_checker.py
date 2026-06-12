# -*- coding: utf-8 -*-
"""Site Checker — проверка сайтов через httpx.

Для каждого бизнеса:
  - сайт не указан  → has_site=False, site_quality="none" (готовый лид)
  - сайт отвечает 200 → has_site=True, site_quality="unknown"
  - ошибка соединения / 4xx / 5xx → has_site=False, site_quality="dead"

Запросы асинхронные, не более 5 одновременно.
"""

import asyncio
from typing import Awaitable, Callable, List, Optional

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


async def _check_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, b: Business) -> None:
    """Проверить один бизнес. Результат пишется прямо в объект."""
    if not b.website:
        b.has_site = False
        b.site_quality = "none"
        return

    url = b.website
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    async with sem:  # не более 5 одновременных запросов
        try:
            resp = await client.get(url)
            if resp.status_code < 400:
                b.has_site = True
                b.site_quality = "unknown"
            else:
                # Сайт указан, но отдаёт ошибку — считаем мёртвым
                b.has_site = False
                b.site_quality = "dead"
        except httpx.HTTPError:
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
        print(f"- {biz.name}: has_site={biz.has_site}, quality={biz.site_quality}")

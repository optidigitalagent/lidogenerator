# -*- coding: utf-8 -*-
"""Social Checker — проверка Instagram через Playwright (без API).

Для каждого бизнеса с instagram_url:
  1. Пробуем открытый JSON: instagram.com/{username}/?__a=1&__d=dis
  2. Если JSON недоступен (Instagram блокирует) — парсим HTML страницы профиля
     (meta og:description вида "12K Followers, 340 Following, 215 Posts").

Активный = профиль существует, постов > 0 и последний пост не старше 90 дней.
Если дату последнего поста узнать нельзя — считаем активным при posts > 0.
Задержка 5-8 секунд между профилями.
"""

import asyncio
import json
import random
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable, List, Optional

from playwright.async_api import async_playwright

import config
from models import Business

ProgressCallback = Callable[[int, int], Awaitable[None]]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _extract_username(url: str) -> str:
    """Достать username из ссылки вида instagram.com/username/?igshid=..."""
    m = re.search(r"instagram\.com/([A-Za-z0-9._]+)", url)
    if not m:
        return ""
    username = m.group(1)
    # Служебные пути — не профили
    if username.lower() in ("p", "reel", "explore", "stories", "accounts", "search"):
        return ""
    return username


def _parse_count(text: str) -> int:
    """Перевести "12.5K" / "3,392" / "1 234" / "3 тис." в число.

    Без суффикса запятая/точка/пробел — разделители тысяч ("3,392" = 3392).
    С суффиксом K/M/тис/млн — десятичный разделитель ("12.5K" = 12500).
    """
    text = text.strip()
    m = re.match(r"([\d.,\s ]+)\s*(K|M|тис|млн)?", text, re.IGNORECASE)
    if not m or not m.group(1).strip():
        return 0
    num_str = m.group(1).strip()
    suffix = (m.group(2) or "").lower()
    if suffix:
        num = float(num_str.replace(",", ".").replace(" ", "").replace(" ", ""))
        num *= 1_000 if suffix in ("k", "тис") else 1_000_000
        return int(num)
    # Без суффикса — убираем все разделители тысяч
    return int(re.sub(r"\D", "", num_str) or 0)


def _parse_json_profile(data: dict, b: Business) -> bool:
    """Разобрать JSON-ответ Instagram. Вернуть True, если данные получены."""
    try:
        user = data["graphql"]["user"]
    except (KeyError, TypeError):
        user = data.get("user") if isinstance(data, dict) else None
    if not user:
        return False

    b.followers = user.get("edge_followed_by", {}).get("count", 0)
    media = user.get("edge_owner_to_timeline_media", {})
    b.posts_count = media.get("count", 0)

    # Дата последнего поста — из первого элемента ленты
    edges = media.get("edges", [])
    if edges:
        ts = edges[0].get("node", {}).get("taken_at_timestamp")
        if ts:
            delta = datetime.now(timezone.utc) - datetime.fromtimestamp(ts, tz=timezone.utc)
            b.last_post_days = delta.days
    return True


def _parse_html_meta(description: str, b: Business) -> bool:
    """Запасной вариант: достать цифры из meta og:description.

    Пример: "3,392 Followers, 3,320 Following, 381 Posts - See Instagram photos..."
    """
    if not description:
        return False
    got = False
    m = re.search(r"([\d.,\s ]+\s*(?:K|M|тис|млн)?)\s*(?:Followers|читач|підписник|подписчик)",
                  description, re.IGNORECASE)
    if m:
        b.followers = _parse_count(m.group(1))
        got = True
    m = re.search(r"([\d.,\s ]+\s*(?:K|M|тис|млн)?)\s*(?:Posts|допис|публикац)",
                  description, re.IGNORECASE)
    if m:
        b.posts_count = _parse_count(m.group(1))
        got = True
    return got


async def check_instagram(
    businesses: List[Business],
    progress_callback: Optional[ProgressCallback] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> List[Business]:
    """Проверить Instagram у бизнесов, где instagram_url не пустой."""
    targets = [b for b in businesses if b.instagram_url]
    if not targets:
        return businesses

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT, locale="uk-UA")
        page = await context.new_page()

        for i, b in enumerate(targets):
            if stop_flag and stop_flag():
                break

            username = _extract_username(b.instagram_url)
            if not username:
                b.instagram_active = False
                continue

            got_data = False

            # --- Попытка 1: открытый JSON ---
            try:
                resp = await page.goto(
                    f"https://www.instagram.com/{username}/?__a=1&__d=dis",
                    wait_until="domcontentloaded",
                    timeout=20000,
                )
                if resp and resp.status == 200:
                    body = await page.inner_text("body")
                    try:
                        got_data = _parse_json_profile(json.loads(body), b)
                    except (json.JSONDecodeError, ValueError):
                        got_data = False
                elif resp and resp.status == 404:
                    # Профиль не существует
                    b.instagram_active = False
                    got_data = None  # дальше не пробуем
            except Exception:
                got_data = False

            # --- Попытка 2: обычная страница профиля, парсим HTML ---
            if got_data is False:
                try:
                    resp = await page.goto(
                        f"https://www.instagram.com/{username}/",
                        wait_until="domcontentloaded",
                        timeout=20000,
                    )
                    if resp and resp.status == 404:
                        b.instagram_active = False
                        got_data = None
                    else:
                        # Ждём появления meta og:description (есть даже без логина)
                        try:
                            await page.wait_for_selector(
                                'meta[property="og:description"]',
                                state="attached", timeout=10000,
                            )
                        except Exception:
                            pass
                        description = await page.get_attribute(
                            'meta[property="og:description"]', "content"
                        ) or ""
                        got_data = _parse_html_meta(description, b)
                except Exception:
                    got_data = False

            # --- Вывод об активности ---
            if got_data is True:
                if b.posts_count <= 0:
                    b.instagram_active = False
                elif b.last_post_days is not None and b.last_post_days > 90:
                    b.instagram_active = False
                else:
                    b.instagram_active = True
            elif got_data is False:
                # Данные получить не удалось (блокировка) — профиль есть, но активность неизвестна.
                # Считаем активным: ссылка из Maps обычно живая, лучше не терять лид.
                b.instagram_active = True

            if progress_callback:
                await progress_callback(i + 1, len(targets))

            # Пауза 5-8 секунд, чтобы Instagram не заблокировал
            if i < len(targets) - 1:
                await asyncio.sleep(
                    random.uniform(config.INSTAGRAM_DELAY_MIN, config.INSTAGRAM_DELAY_MAX)
                )

        await browser.close()

    return businesses


if __name__ == "__main__":
    # Ручной запуск на тестовых данных
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads((Path(__file__).parent.parent / "tests" / "stage3_result.json").read_text(encoding="utf-8"))
    items = [Business(**d) for d in data]
    asyncio.run(check_instagram(items))
    for biz in items:
        if biz.instagram_url:
            print(f"- {biz.name}: active={biz.instagram_active}, "
                  f"followers={biz.followers}, posts={biz.posts_count}, last={biz.last_post_days}")

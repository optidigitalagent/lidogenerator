# -*- coding: utf-8 -*-
"""AI Scorer — оценка лидов через Claude API (claude-haiku-4-5).

Для каждого бизнеса собираем краткую сводку (сайт, Instagram, отзывы)
и просим Claude вернуть JSON: {"score": 0-100, "priority": "hot/warm/cold", "reason": "..."}.

Если ANTHROPIC_API_KEY не задан — скоринг пропускается, всем ставится score=50 (warm).
Если запрос к API упал — этому бизнесу тоже ставится score=50, цикл не прерывается.
"""

import asyncio
import json
import re
from typing import Awaitable, Callable, List, Optional

import config
from models import Business

ProgressCallback = Callable[[int, int], Awaitable[None]]

# Промпт из техплана — вшит в код, руками больше не пишется
SCORING_PROMPT = """Ты — помощник менеджера веб-студии в Украине. Оцени бизнес как лид.

Данные бизнеса:
- Название: {name}
- Ниша: {niche}
- Город: {city}
- Сайт: {site_status}
- Instagram: {instagram_status}
- Подписчики: {followers}
- Последний пост: {last_post_days} дней назад
- Отзывов на Maps: {reviews_count}

Оцени по шкале 0-100. Критерии:
  Нет сайта: +35 баллов
  Сайт старый/некачественный: +20 баллов
  Instagram активный (<30 дней): +20 баллов
  Много подписчиков (>500): +10 баллов
  Много отзывов на Maps (>20): +10 баллов
  Ниша с высоким чеком (клиника, ресторан, салон): +5 баллов

Верни ТОЛЬКО JSON: {{"score": 75, "priority": "hot", "reason": "..."}}
priority: hot (>=70) / warm (40-69) / cold (<40)"""


def _site_status(b: Business) -> str:
    """Поле site_quality -> человекочитаемый статус для промпта."""
    if not b.has_site:
        return "нет" if b.site_quality == "none" else "нет (указан, но не открывается)"
    return {
        "old": "старый",
        "bad": "плохой",
        "good": "хороший",
    }.get(b.site_quality, "есть, качество неизвестно")


def _instagram_status(b: Business) -> str:
    """Статус Instagram для промпта."""
    if not b.instagram_url:
        return "нет"
    return "активный" if b.instagram_active else "мёртвый"


def _priority_from_score(score: int) -> str:
    """Подстраховка: считаем priority из score по шкале техплана."""
    if score >= 70:
        return "hot"
    if score >= 40:
        return "warm"
    return "cold"


def _fallback(b: Business, reason: str) -> None:
    """Поставить нейтральную оценку, когда AI недоступен."""
    b.ai_score = 50
    b.ai_priority = "warm"
    b.ai_reason = reason


def _parse_response(text: str) -> Optional[dict]:
    """Достать JSON из ответа модели (на случай лишнего текста вокруг)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def _score_one(client, b: Business) -> None:
    """Оценить один бизнес. Результат пишется прямо в объект."""
    prompt = SCORING_PROMPT.format(
        name=b.name,
        niche=b.niche,
        city=b.city,
        site_status=_site_status(b),
        instagram_status=_instagram_status(b),
        followers=b.followers,
        last_post_days=b.last_post_days if b.last_post_days is not None else "неизвестно",
        reviews_count=b.reviews_count,
    )
    try:
        response = await client.messages.create(
            model=config.AI_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((blk.text for blk in response.content if blk.type == "text"), "")
        data = _parse_response(text)
        if not data or "score" not in data:
            _fallback(b, "AI вернул некорректный ответ")
            return
        b.ai_score = max(0, min(100, int(data["score"])))
        b.ai_priority = data.get("priority") or _priority_from_score(b.ai_score)
        b.ai_reason = str(data.get("reason", ""))[:500]
    except Exception as e:
        # Любая ошибка API (лимиты, сеть) — нейтральная оценка, идём дальше
        _fallback(b, f"AI недоступен: {type(e).__name__}")


async def score_businesses(
    businesses: List[Business],
    progress_callback: Optional[ProgressCallback] = None,
) -> List[Business]:
    """Оценить все бизнесы. Возвращает тот же список с заполненными ai_* полями."""
    if not config.ANTHROPIC_API_KEY:
        # Ключ не задан — скоринг отключён, всем нейтральная оценка
        for i, b in enumerate(businesses, 1):
            _fallback(b, "AI-скоринг отключён (нет ANTHROPIC_API_KEY)")
            if progress_callback:
                await progress_callback(i, len(businesses))
        return businesses

    # Импорт здесь, чтобы проект работал и без ключа/библиотеки
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    for i, b in enumerate(businesses, 1):
        await _score_one(client, b)
        if progress_callback:
            await progress_callback(i, len(businesses))
        await asyncio.sleep(0.3)  # щадим рейт-лимиты
    return businesses


if __name__ == "__main__":
    # Ручной запуск на тестовых данных
    import sys
    from pathlib import Path

    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(
        (Path(__file__).parent.parent / "tests" / "stage3_result.json").read_text(encoding="utf-8")
    )
    items = [Business(**d) for d in data[:5]]
    asyncio.run(score_businesses(items))
    for biz in items:
        print(f"- {biz.name}: score={biz.ai_score}, priority={biz.ai_priority} ({biz.ai_reason})")

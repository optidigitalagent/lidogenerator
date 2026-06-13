# -*- coding: utf-8 -*-
"""Orchestrator — запускает агентов по цепочке и передаёт данные между ними.

Цепочка:
  1. collector  — парсинг Google Maps      (статус задачи: collecting)
  2. site_checker — проверка сайтов        (статус: checking)
  3. social_checker — проверка Instagram   (статус: checking)
  4. ai_scorer  — оценка через Claude      (статус: scoring)
  5. reporter   — экспорт CSV              (статус: done)

Прогресс отправляется наружу через callback не чаще, чем раз в PROGRESS_INTERVAL
(по умолчанию 3 минуты). Остановка — через asyncio.Event: проверяется между
этапами и внутри прогресс-коллбеков агентов.
"""

import asyncio
import time
from typing import Awaitable, Callable, List, Optional

import config
import db
from agents import collector, site_checker, social_checker, ai_scorer, reporter
from models import Business

# Callback наружу (в Telegram): принимает готовый текст сообщения (на украинском)
ReportCallback = Callable[[str], Awaitable[None]]


class SearchStopped(Exception):
    """Поиск остановлен пользователем."""


class _Progress:
    """Копит состояние этапов и шлёт сообщение наружу не чаще, чем раз в interval сек."""

    def __init__(self, callback: Optional[ReportCallback], interval: int,
                 stop_event: Optional[asyncio.Event]):
        self.callback = callback
        self.interval = interval
        self.stop_event = stop_event
        self.lines: dict = {}          # этап -> текст строки
        self._last_sent = 0.0          # время последней отправки

    async def update(self, stage: str, line: str, force: bool = False) -> None:
        """Обновить строку этапа и отправить сводку, если пришло время."""
        if self.stop_event and self.stop_event.is_set():
            raise SearchStopped()
        self.lines[stage] = line
        now = time.monotonic()
        if self.callback and (force or now - self._last_sent >= self.interval):
            self._last_sent = now
            await self.callback("\n".join(self.lines.values()))


async def run_search(
    task_id: int,
    progress_callback: Optional[ReportCallback] = None,
    stop_event: Optional[asyncio.Event] = None,
    progress_interval: int = None,
) -> Optional[str]:
    """Полный цикл поиска для задачи task_id. Возвращает путь к CSV или None при остановке/ошибке."""
    task = db.get_task(task_id)
    if task is None:
        raise ValueError(f"Задача {task_id} не найдена")

    niche, city, count = task["niche"], task["city"], task["count"]
    interval = progress_interval if progress_interval is not None else config.PROGRESS_INTERVAL
    progress = _Progress(progress_callback, interval, stop_event)
    businesses: List[Business] = []

    try:
        # --- Этап 1: сбор с Google Maps ---
        db.update_task_status(task_id, "collecting")
        await progress.update("collect", f"📍 Шукаю «{niche}» у місті {city}...", force=True)

        async def on_collect(done: int, total: int):
            await progress.update("collect", f"📍 Зібрано {done}/{total} бізнесів з Google Maps...")

        stop_flag = (lambda: stop_event.is_set()) if stop_event else None
        businesses = await collector.collect(
            niche, city, count, progress_callback=on_collect, stop_flag=stop_flag
        )
        if stop_event and stop_event.is_set():
            raise SearchStopped()
        for b in businesses:
            b.task_id = task_id
        saved = db.save_businesses(businesses)
        # Работаем дальше только с сохранёнными (без дубликатов)
        businesses = [b for b in businesses if b.id is not None]
        await progress.update("collect", f"📍 Зібрано {saved} бізнесів з Google Maps ✅", force=True)

        # --- Этап 2: проверка сайтов ---
        db.update_task_status(task_id, "checking")

        async def on_sites(done: int, total: int):
            await progress.update("sites", f"🔍 Перевірка сайтів: {done}/{total}...")

        await site_checker.check_sites(businesses, progress_callback=on_sites)
        no_site = sum(1 for b in businesses if not b.has_site)
        await progress.update("sites", f"🔍 Сайти перевірено: {no_site} без сайту ✅", force=True)

        # --- Этап 3: проверка Instagram ---
        async def on_insta(done: int, total: int):
            await progress.update("insta", f"📱 Перевірка Instagram: {done}/{total}...")

        await social_checker.check_instagram(
            businesses, progress_callback=on_insta, stop_flag=stop_flag
        )
        active = sum(1 for b in businesses if b.instagram_active)
        await progress.update("insta", f"📱 Instagram перевірено: {active} активних ✅", force=True)

        # --- Этап 4: строгая фильтрация лидов ---
        # Лид = є Instagram І (сайту немає АБО сайт поганий).
        # Бізнеси з нормальним сайтом і без Instagram відсіюються.
        leads = [b for b in businesses if b.is_lead]

        total = len(businesses)
        no_instagram = sum(1 for b in businesses if not b.instagram_url)
        skipped_good = sum(
            1 for b in businesses if b.instagram_url and b.website_status == "good website"
        )
        added_no_site = sum(1 for b in leads if b.website_status == "no website")
        added_bad_site = sum(1 for b in leads if b.website_status == "bad website")

        await progress.update(
            "filter",
            f"🧹 Відбір: {len(leads)} лідів "
            f"(❌ {skipped_good} з гарним сайтом, ❌ {no_instagram} без Instagram)",
            force=True,
        )

        # --- Этап 5: AI-скоринг (тільки лідів; не впливає на відбір) ---
        db.update_task_status(task_id, "scoring")

        async def on_score(done: int, total_: int):
            await progress.update("score", f"🤖 AI-оцінка: {done}/{total_}...")

        await ai_scorer.score_businesses(leads, progress_callback=on_score)

        # Сохраняем результаты проверок в базу (всі бізнеси — для історії)
        for b in businesses:
            db.update_business(b)

        # --- Этап 6: экспорт CSV (тільки лідів) ---
        csv_path = reporter.export_csv(leads, task_id=task_id)
        db.update_task_status(task_id, "done", csv_path=csv_path)

        # Підсумок + короткий список лідів
        await progress.update(
            "done",
            f"✅ Готово!\n"
            f"Знайдено всього: {total}\n"
            f"Пропущено (гарний сайт): {skipped_good}\n"
            f"Пропущено (без Instagram): {no_instagram}\n"
            f"Додано без сайту: {added_no_site}\n"
            f"Додано з поганим сайтом: {added_bad_site}\n"
            f"➡️ Усього лідів у таблиці: {len(leads)}",
            force=True,
        )
        if progress_callback and leads:
            await progress_callback(reporter.format_leads_summary(leads))
        return csv_path

    except SearchStopped:
        # Пользователь остановил поиск — сохраняем то, что успели
        for b in businesses:
            db.update_business(b)
        db.update_task_status(task_id, "stopped")
        if progress_callback:
            await progress_callback("⏹ Пошук зупинено.")
        return None
    except Exception as e:
        db.update_task_status(task_id, "error")
        if progress_callback:
            await progress_callback(f"❌ Помилка: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    # Ручной запуск полного цикла
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    db.init_db()
    tid = db.create_task("салон краси", "Харків", 10)

    async def printer(text: str):
        print(f"[прогрес]\n{text}\n")

    path = asyncio.run(run_search(tid, progress_callback=printer, progress_interval=30))
    print(f"CSV: {path}")

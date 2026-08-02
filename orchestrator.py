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
import urllib.parse
from typing import Awaitable, Callable, List, Optional

import config
import db
from agents import collector, site_checker, social_checker, ai_scorer, reporter
from models import Business
from niche_catalog import get_niche_variants
from query_planner import build_query_queue
from search_policy import (
    SearchPolicy,
    SearchProgress,
    StopReason,
    decide_next,
    limit_to_target,
)

# Callback наружу (в Telegram): принимает готовый текст сообщения (на украинском)
ReportCallback = Callable[[str], Awaitable[None]]


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _normalized_domain(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urllib.parse.urlsplit(raw if "://" in raw else f"//{raw}")
    domain = (parsed.hostname or "").casefold().rstrip(".")
    return domain.removeprefix("www.")


def _normalized_instagram(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    parsed = urllib.parse.urlsplit(raw if "://" in raw else f"//{raw}")
    domain = (parsed.hostname or "").casefold().removeprefix("www.")
    if domain == "instagram.com" or domain.endswith(".instagram.com"):
        username = parsed.path.strip("/").split("/", 1)[0].lstrip("@").casefold()
        if username:
            return username
    return _normalized_text(raw).rstrip("/")


def _business_dedupe_key(business: Business) -> tuple[str, ...] | None:
    """Return a stable in-memory identity for one search task."""
    phone = "".join(character for character in business.phone if character.isdigit())
    if phone:
        return ("phone", phone)

    domain = _normalized_domain(business.website)
    if domain:
        return ("website", domain)

    instagram = _normalized_instagram(business.instagram_url)
    if instagram:
        return ("instagram", instagram)

    name = _normalized_text(business.name)
    address = _normalized_text(business.address)
    if name or address:
        return ("name_address", name, address)
    return None


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
    """Полный цикл поиска для задачи task_id.

    Число task["count"] трактуется как target_leads — сколько ВАЛИДНЫХ ЛИДОВ
    нужно получить в таблице (а не сколько бизнесов просмотреть). Бизнесы
    собираются батчами из Google Maps; после каждого батча проверяются сайты и
    отбираются лиды (є Instagram І сайту немає/сайт поганий). Сбор продолжается,
    пока не наберём target_leads лидов либо не упрёмся в safety-лимиты.

    Возвращает путь к файлу-таблице (xlsx или csv) или None при остановке/ошибке.
    """
    task = db.get_task(task_id)
    if task is None:
        raise ValueError(f"Задача {task_id} не найдена")

    niche, city, target_leads = task["niche"], task["city"], task["count"]
    # Temporary adapter: the current collector limit counts opened Maps cards.
    policy = SearchPolicy(
        target_leads=target_leads,
        max_candidates=config.MAX_BUSINESSES_PER_SEARCH,
    )
    interval = progress_interval if progress_interval is not None else config.PROGRESS_INTERVAL
    progress = _Progress(progress_callback, interval, stop_event)
    stop_flag = (lambda: stop_event.is_set()) if stop_event else None

    # Накопители по ходу батчевого сбора
    leads: List[Business] = []          # валидные лиды (то, что попадёт в таблицу)
    visited = 0                         # просмотрено бизнесов всего
    checked_candidates = 0              # уникальные кандидаты с завершённой проверкой сайта
    skipped_no_instagram = 0            # пропущено: нет Instagram
    skipped_good_site = 0               # пропущено: есть Instagram, но хороший сайт
    stop_reason: Optional[StopReason] = None
    seen_business_keys: set[tuple[str, ...]] = set()
    niche_variants = get_niche_variants(niche)
    query_queue = build_query_queue(
        niche=niche,
        city=city,
        niche_variants=niche_variants,
    )

    def _decide(remaining_queries: int):
        return decide_next(
            SearchProgress(
                qualified_leads=len(leads),
                checked_candidates=checked_candidates,
                # Adapter from the current single Maps stream to a future query queue.
                remaining_queries=remaining_queries,
                stop_requested=bool(stop_event and stop_event.is_set()),
            ),
            policy,
        )

    def _added_counts():
        added_no_site = sum(1 for b in leads if b.website_status == "no website")
        added_bad_site = sum(1 for b in leads if b.website_status == "bad website")
        return added_no_site, added_bad_site

    async def _send_progress(force: bool = False):
        added_no_site, added_bad_site = _added_counts()
        await progress.update(
            "main",
            f"🔎 Шукаю {target_leads} якісних лідів: «{niche}» у місті {city}\n"
            f"Запрошено лідів: {target_leads}\n"
            f"Переглянуто бізнесів: {visited}\n"
            f"Знайдено валідних лідів: {len(leads)}/{target_leads}\n"
            f"Пропущено без Instagram: {skipped_no_instagram}\n"
            f"Пропущено з гарним сайтом: {skipped_good_site}\n"
            f"Додано без сайту: {added_no_site}\n"
            f"Додано з поганим сайтом: {added_bad_site}",
            force=force,
        )

    try:
        db.update_task_status(task_id, "collecting")
        decision = _decide(remaining_queries=query_queue.remaining_queries)
        if decision.stop_reason is StopReason.USER_STOPPED:
            raise SearchStopped()
        await progress.update(
            "main",
            f"🔎 Шукаю {target_leads} якісних лідів: «{niche}» у місті {city}...",
            force=True,
        )

        # --- Сбор батчами + фильтрация до набора target_leads ---
        db.update_task_status(task_id, "checking")

        while not query_queue.exhausted:
            decision = _decide(remaining_queries=query_queue.remaining_queries)
            if not decision.should_continue:
                stop_reason = decision.stop_reason
            if stop_reason is StopReason.USER_STOPPED:
                raise SearchStopped()
            if stop_reason is not None:
                break

            search_query, query_queue = query_queue.take_next()
            assert search_query is not None
            active_remaining_queries = query_queue.remaining_queries + 1
            visited_before_stream = visited

            async def on_visit(v: int):
                nonlocal visited
                visited = visited_before_stream + v
                decision = _decide(remaining_queries=active_remaining_queries)
                if decision.stop_reason is StopReason.USER_STOPPED:
                    raise SearchStopped()
                await _send_progress()

            stream_exhausted = False
            async for batch in collector.collect_stream(
                niche,
                city,
                max_businesses=policy.max_candidates - checked_candidates,
                progress_callback=on_visit,
                stop_flag=stop_flag,
                query_text=search_query.text,
            ):
                decision = _decide(remaining_queries=active_remaining_queries)
                if not decision.should_continue:
                    stop_reason = decision.stop_reason
                if stop_reason is StopReason.USER_STOPPED:
                    raise SearchStopped()
                if stop_reason is not None:
                    break

                unique_batch: List[Business] = []
                for business in batch:
                    key = _business_dedupe_key(business)
                    if key is not None and key in seen_business_keys:
                        continue
                    if key is not None:
                        seen_business_keys.add(key)
                    unique_batch.append(business)

                if unique_batch:
                    # Проверяем сайты у уникальных бизнесов батча (быстро, параллельно)
                    await site_checker.check_sites(unique_batch)
                    checked_candidates += len(unique_batch)

                    # Отбираем валидные лиды из батча
                    for b in unique_batch:
                        if b.is_lead:
                            b.task_id = task_id
                            leads.append(b)
                        elif not b.instagram_url:
                            skipped_no_instagram += 1
                        elif b.website_status == "good website":
                            skipped_good_site += 1

                decision = _decide(remaining_queries=active_remaining_queries)
                if not decision.should_continue:
                    stop_reason = decision.stop_reason
                if stop_reason is StopReason.USER_STOPPED:
                    raise SearchStopped()

                await _send_progress(force=True)
                if stop_reason is not None:
                    break
            else:
                stream_exhausted = True

            if stop_reason is not None:
                break
            if stream_exhausted:
                decision = _decide(remaining_queries=query_queue.remaining_queries)
                if not decision.should_continue:
                    stop_reason = decision.stop_reason
                if stop_reason is StopReason.USER_STOPPED:
                    raise SearchStopped()
                if stop_reason is not None:
                    break

        # --- Почему остановились ---
        if stop_reason is None:
            decision = _decide(remaining_queries=query_queue.remaining_queries)
            stop_reason = decision.stop_reason
            if stop_reason is StopReason.USER_STOPPED:
                raise SearchStopped()

        reason_by_stop = {
            StopReason.TARGET_REACHED: f"досягнуто цільову кількість лідів ({target_leads})",
            StopReason.MAX_CANDIDATES_REACHED: (
                f"досягнуто safety-ліміт: перевірено {checked_candidates} бізнесів"
            ),
            StopReason.QUERIES_EXHAUSTED: "результати Google Maps вичерпано",
        }
        reason = reason_by_stop[stop_reason]

        # A batch may cross the target; every downstream consumer gets the same capped list.
        leads = limit_to_target(leads, policy.target_leads)

        # --- Обогащение лидов (Instagram + AI-скоринг) — не влияет на отбор ---
        db.update_task_status(task_id, "scoring")
        db.save_businesses(leads)  # сохраняем лиды, получаем id

        if leads:
            async def on_insta(done: int, total_: int):
                await progress.update("enrich", f"📱 Перевірка Instagram лідів: {done}/{total_}...")

            await social_checker.check_instagram(
                leads, progress_callback=on_insta, stop_flag=stop_flag
            )

            async def on_score(done: int, total_: int):
                await progress.update("enrich", f"🤖 AI-оцінка лідів: {done}/{total_}...")

            await ai_scorer.score_businesses(leads, progress_callback=on_score)

            for b in leads:
                db.update_business(b)

        # --- Экспорт (CSV + Excel с нормальной шириной колонок) ---
        csv_path = reporter.export_csv(leads, task_id=task_id)
        xlsx_path = reporter.export_excel(leads, task_id=task_id)
        out_path = xlsx_path or csv_path
        db.update_task_status(task_id, "done", csv_path=out_path)

        # --- Финальный отчёт ---
        added_no_site, added_bad_site = _added_counts()
        shortage = ""
        if len(leads) < target_leads:
            shortage = (
                f"\n⚠️ Запрошено {target_leads} лідів, "
                f"знайдено лише {len(leads)} підходящих."
            )
        await progress.update(
            "main",
            f"✅ Готово!\n"
            f"Запрошено лідів: {target_leads}\n"
            f"Переглянуто бізнесів: {visited}\n"
            f"Пропущено без Instagram: {skipped_no_instagram}\n"
            f"Пропущено з гарним сайтом: {skipped_good_site}\n"
            f"Додано без сайту: {added_no_site}\n"
            f"Додано з поганим сайтом: {added_bad_site}\n"
            f"➡️ Усього лідів у таблиці: {len(leads)}\n"
            f"Причина зупинки: {reason}"
            + shortage,
            force=True,
        )
        if progress_callback and leads:
            await progress_callback(reporter.format_leads_summary(leads))
        return out_path

    except SearchStopped:
        # Пользователь остановил поиск — сохраняем то, что успели набрать
        leads = limit_to_target(leads, policy.target_leads)
        if leads:
            db.save_businesses(leads)
            for b in leads:
                db.update_business(b)
        db.update_task_status(task_id, "stopped")
        if progress_callback:
            await progress_callback(
                f"⏹ Пошук зупинено. Встигли зібрати {len(leads)} лідів "
                f"(переглянуто {visited} бізнесів)."
            )
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

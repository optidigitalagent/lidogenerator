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
import copy
import json
import logging
import time
import urllib.parse
from typing import Awaitable, Callable, List, Optional

import city_catalog
import config
import db
from agents import (
    ai_scorer,
    collector,
    instagram_first_party_resolver,
    reporter,
    site_checker,
    social_checker,
    website_resolver,
)
from agents.instagram_first_party_resolver import FirstPartyInstagramResolver
from instagram_first_party_resolution import (
    FirstPartyInstagramEvidenceSource,
    FirstPartyInstagramResolution,
    FirstPartyInstagramStatus,
)
from models import Business
from integrations.opti_bridge import finalize_completed_task
from website_candidate_matching import SearchProvider
from website_pipeline import LeadDecision, ResolverMode, parse_resolver_mode, qualify_lead
from website_search_runtime import (
    build_configured_search_provider,
    openai_web_search_telemetry_snapshot,
    search_budget_snapshot,
)
from niche_catalog import resolve_niche_plan
from query_budget import allocate_query_budget
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
log = logging.getLogger("lead_hunter.orchestrator")


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


def _shadow_business_summary(businesses: List[Business]) -> dict[str, object]:
    """Return deterministic, identity-free resolver outcomes for shadow logging."""
    allowed_statuses = {
        "found_official",
        "social_only",
        "not_found",
        "uncertain",
        "resolution_error",
    }
    allowed_sources = {"maps", "instagram_bio", "web_search"}
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    resolved_domains: set[str] = set()

    for business in businesses:
        raw_status = business.website_resolution_status
        status = raw_status if raw_status in allowed_statuses else "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

        raw_source = business.website_resolution_source
        if raw_source:
            source = raw_source if raw_source in allowed_sources else "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1

        if status == "found_official":
            raw_url = business.website_resolved_url
            domain = _normalized_domain(raw_url) if isinstance(raw_url, str) else ""
            if domain:
                resolved_domains.add(domain)

    return {
        "candidate_count": len(businesses),
        "status_counts": dict(sorted(status_counts.items())),
        "resolved_domains": sorted(resolved_domains),
        "source_counts": dict(sorted(source_counts.items())),
    }


def _shadow_resolver_telemetry(
    businesses: List[Business],
    provider: SearchProvider | None,
    *,
    task_id: int | None,
) -> dict[str, object]:
    telemetry: dict[str, object] = {
        "event": "website_resolver_shadow",
        "task_id": task_id,
        **_shadow_business_summary(businesses),
    }

    budget = search_budget_snapshot(provider)
    if budget is not None:
        telemetry["provider_budget"] = {
            "max_requests": budget.max_requests,
            "used_requests": budget.used_requests,
            "remaining_requests": budget.remaining_requests,
        }

    openai = openai_web_search_telemetry_snapshot(provider)
    if openai is not None:
        telemetry["openai_provider"] = {
            "requests_started": openai.requests_started,
            "requests_succeeded": openai.requests_succeeded,
            "requests_failed": openai.requests_failed,
            "tool_calls_seen": openai.tool_calls_seen,
            "search_actions_seen": openai.search_actions_seen,
            "open_page_actions_seen": openai.open_page_actions_seen,
            "find_in_page_actions_seen": openai.find_in_page_actions_seen,
            "sources_seen": openai.sources_seen,
            "identity_candidates_rejected": openai.identity_candidates_rejected,
            "candidates_returned": openai.candidates_returned,
            "tool_call_limit_exceeded": openai.tool_call_limit_exceeded,
            "last_error_category": openai.last_error_category,
        }
    return telemetry


def _first_party_shadow_telemetry(
    businesses: List[Business],
    resolutions: tuple[FirstPartyInstagramResolution, ...],
    resolver: FirstPartyInstagramResolver,
    *,
    task_id: int | None,
    missing_instagram_count: int,
    trusted_website_count: int,
) -> dict[str, object]:
    status_counts = {
        status: sum(1 for result in resolutions if result.status is status)
        for status in FirstPartyInstagramStatus
    }
    budget = resolver.budget.snapshot()
    return {
        "event": "instagram_first_party_shadow",
        "task_id": task_id,
        "batch_candidate_count": len(businesses),
        "missing_instagram_count": missing_instagram_count,
        "trusted_website_count": trusted_website_count,
        "attempted_businesses": sum(
            1
            for result in resolutions
            if result.pages_attempted > 0
        ),
        "found_official_count": status_counts[
            FirstPartyInstagramStatus.FOUND_OFFICIAL
        ],
        "not_found_count": status_counts[FirstPartyInstagramStatus.NOT_FOUND],
        "uncertain_count": status_counts[FirstPartyInstagramStatus.UNCERTAIN],
        "technical_error_count": status_counts[
            FirstPartyInstagramStatus.TECHNICAL_ERROR
        ],
        "skipped_count": status_counts[FirstPartyInstagramStatus.SKIPPED],
        "requests": {
            "max": budget.max_requests,
            "used": budget.used_requests,
            "remaining": budget.remaining_requests,
        },
        "pages_attempted": sum(result.pages_attempted for result in resolutions),
        "pages_succeeded": sum(result.pages_succeeded for result in resolutions),
        "evidence_source_counts": {
            source.value: sum(
                1 for result in resolutions if source in result.evidence_sources
            )
            for source in FirstPartyInstagramEvidenceSource
        },
    }


async def _run_first_party_instagram_shadow(
    businesses: List[Business],
    resolver: FirstPartyInstagramResolver,
    *,
    task_id: int | None,
) -> None:
    missing_instagram_count = sum(1 for item in businesses if not item.instagram_url)
    trusted_website_count = sum(
        1
        for item in businesses
        if not item.instagram_url
        and instagram_first_party_resolver.trusted_website_for_instagram_resolution(item)
        is not None
    )
    try:
        resolutions = (
            await instagram_first_party_resolver.resolve_missing_instagrams_first_party(
                businesses,
                resolver=resolver,
            )
        )
        telemetry = _first_party_shadow_telemetry(
            businesses,
            resolutions,
            resolver,
            task_id=task_id,
            missing_instagram_count=missing_instagram_count,
            trusted_website_count=trusted_website_count,
        )
        log.info(
            "instagram_first_party_shadow %s",
            json.dumps(telemetry, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:
        log.warning(
            "instagram_first_party_shadow_failed task_id=%s exception_type=%s",
            task_id,
            type(exc).__name__,
        )


async def _check_batch_websites_with_resolver_mode(
    businesses: List[Business],
    resolver_mode: ResolverMode,
    provider: SearchProvider | None,
    *,
    task_id: int | None = None,
    first_party_mode: str = "off",
    first_party_resolver_runtime: FirstPartyInstagramResolver | None = None,
) -> None:
    """Audit production objects while isolating all shadow resolver mutations."""
    if first_party_mode not in {"off", "shadow"}:
        raise ValueError("first_party_mode must be off or shadow")
    if first_party_mode == "off":
        first_party_resolver_runtime = None
    elif first_party_resolver_runtime is None:
        first_party_resolver_runtime = (
            instagram_first_party_resolver.build_configured_first_party_resolver()
        )

    if resolver_mode is ResolverMode.OFF:
        await site_checker.check_sites(businesses)
        if first_party_resolver_runtime is not None:
            first_party_shadow_businesses = copy.deepcopy(businesses)
            for item in first_party_shadow_businesses:
                item.website_original_url = ""
                item.website_resolution_status = ""
                item.website_resolved_url = ""
            await _run_first_party_instagram_shadow(
                first_party_shadow_businesses,
                first_party_resolver_runtime,
                task_id=task_id,
            )
        return

    if resolver_mode is ResolverMode.STRICT:
        await website_resolver.resolve_business_websites(
            businesses,
            provider=provider,
        )
        await site_checker.check_sites(businesses)
        if first_party_resolver_runtime is not None:
            await _run_first_party_instagram_shadow(
                copy.deepcopy(businesses),
                first_party_resolver_runtime,
                task_id=task_id,
            )
        return

    await site_checker.check_sites(businesses)
    shadow_businesses = copy.deepcopy(businesses)
    try:
        await website_resolver.resolve_business_websites(
            shadow_businesses,
            provider=provider,
        )
        telemetry = _shadow_resolver_telemetry(
            shadow_businesses,
            provider,
            task_id=task_id,
        )
        log.info(
            "website_resolver_shadow %s",
            json.dumps(telemetry, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:
        log.warning(
            "website_resolver_shadow_failed task_id=%s exception_type=%s",
            task_id,
            type(exc).__name__,
        )
    if first_party_resolver_runtime is not None:
        await _run_first_party_instagram_shadow(
            shadow_businesses,
            first_party_resolver_runtime,
            task_id=task_id,
        )


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
    website_search_provider: SearchProvider | None = None,
    first_party_resolver_runtime: FirstPartyInstagramResolver | None = None,
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
    policy = SearchPolicy(
        target_leads=target_leads,
        max_candidates=config.MAX_CHECKED_CANDIDATES_PER_TASK,
        max_discovery_cards=config.MAX_MAPS_CARDS_PER_TASK,
    )
    interval = progress_interval if progress_interval is not None else config.PROGRESS_INTERVAL
    progress = _Progress(progress_callback, interval, stop_event)
    resolver_mode = parse_resolver_mode(config.WEBSITE_RESOLVER_MODE)
    if website_search_provider is not None:
        runtime_website_search_provider = website_search_provider
    elif resolver_mode is ResolverMode.OFF:
        runtime_website_search_provider = None
    else:
        runtime_website_search_provider = build_configured_search_provider()
    if config.INSTAGRAM_FIRST_PARTY_MODE == "shadow":
        runtime_first_party_resolver = (
            first_party_resolver_runtime
            or instagram_first_party_resolver.build_configured_first_party_resolver()
        )
    else:
        runtime_first_party_resolver = None
    stop_flag = (lambda: stop_event.is_set()) if stop_event else None

    # Накопители по ходу батчевого сбора
    leads: List[Business] = []          # валидные лиды (то, что попадёт в таблицу)
    visited = 0                         # просмотрено бизнесов всего
    checked_candidates = 0              # уникальные кандидаты с завершённой проверкой сайта
    skipped_no_instagram = 0            # пропущено: нет Instagram
    skipped_good_site = 0               # пропущено: есть Instagram, но хороший сайт
    skipped_uncertain_website = 0       # пропущено: результат сайта недостаточно надёжен
    stop_reason: Optional[StopReason] = None
    seen_business_keys: set[tuple[str, ...]] = set()
    niche_plan = resolve_niche_plan(niche)
    city_definition = city_catalog.resolve_city(
        city,
        city_catalog.CITY_DEFINITIONS,
    )
    query_city = (
        city_definition.canonical_name
        if city_definition is not None
        else city
    )
    district_fragments = (
        tuple(
            district.query_text
            for district in city_catalog.enabled_districts(city_definition)
        )
        if niche_plan.known and city_definition is not None
        else ()
    )
    query_queue = build_query_queue(
        niche=niche_plan.base_niche,
        city=query_city,
        niche_variants=niche_plan.primary_variants,
        districts=district_fragments,
        fallback_variants=niche_plan.fallback_variants,
    )

    def _decide(remaining_queries: int):
        return decide_next(
            SearchProgress(
                qualified_leads=len(leads),
                checked_candidates=checked_candidates,
                remaining_queries=remaining_queries,
                stop_requested=bool(stop_event and stop_event.is_set()),
                visited_cards=visited,
            ),
            policy,
        )

    def _added_counts():
        added_no_site = sum(1 for b in leads if b.website_status == "no website")
        added_bad_site = sum(1 for b in leads if b.website_status == "bad website")
        return added_no_site, added_bad_site

    async def _update_website_search_budget_progress() -> None:
        snapshot = search_budget_snapshot(runtime_website_search_provider)
        if snapshot is not None:
            await progress.update(
                "website_search",
                f"Пошуків офіційного сайту: {snapshot.used_requests}/{snapshot.max_requests}",
            )

    async def _send_progress(force: bool = False):
        added_no_site, added_bad_site = _added_counts()
        await _update_website_search_budget_progress()
        await progress.update(
            "main",
            f"🔎 Шукаю {target_leads} якісних лідів: «{niche}» у місті {city}\n"
            f"Запрошено лідів: {target_leads}\n"
            f"Переглянуто бізнесів: {visited}\n"
            f"Відкрито карток Google Maps: {visited}/{policy.max_discovery_cards}\n"
            f"Перевірено унікальних кандидатів: {checked_candidates}/{policy.max_candidates}\n"
            f"Знайдено валідних лідів: {len(leads)}/{target_leads}\n"
            f"Пропущено без Instagram: {skipped_no_instagram}\n"
            f"Пропущено з гарним сайтом: {skipped_good_site}\n"
            f"Пропущено через невизначений статус сайту: {skipped_uncertain_website}\n"
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
            assert policy.max_discovery_cards is not None
            query_budget = allocate_query_budget(
                remaining_checked_candidates=(
                    policy.max_candidates - checked_candidates
                ),
                remaining_opened_cards=(
                    policy.max_discovery_cards - visited
                ),
                active_queries=active_remaining_queries,
            )
            if query_budget.exhausted:
                decision = _decide(
                    remaining_queries=active_remaining_queries,
                )
                stop_reason = decision.stop_reason
                if stop_reason is StopReason.USER_STOPPED:
                    raise SearchStopped()
                if stop_reason is None:
                    raise RuntimeError(
                        "exhausted query budget without a policy stop reason"
                    )
                break
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
                max_businesses=query_budget.current_query_card_limit,
                progress_callback=on_visit,
                stop_flag=stop_flag,
                query_text=search_query.text,
            ):
                decision = _decide(remaining_queries=active_remaining_queries)
                if decision.stop_reason is StopReason.USER_STOPPED:
                    raise SearchStopped()

                unique_batch: List[Business] = []
                for business in batch:
                    key = _business_dedupe_key(business)
                    if key is not None and key in seen_business_keys:
                        continue
                    if key is not None:
                        seen_business_keys.add(key)
                    unique_batch.append(business)

                if unique_batch:
                    await _check_batch_websites_with_resolver_mode(
                        unique_batch,
                        resolver_mode,
                        runtime_website_search_provider,
                        task_id=task_id,
                        first_party_mode=config.INSTAGRAM_FIRST_PARTY_MODE,
                        first_party_resolver_runtime=runtime_first_party_resolver,
                    )
                    checked_candidates += len(unique_batch)

                    if resolver_mode is ResolverMode.STRICT:
                        for business in unique_batch:
                            qualification = qualify_lead(
                                has_instagram=bool(business.instagram_url),
                                resolution=website_resolver.resolution_from_business(business),
                                audit=website_resolver.audit_from_business(business),
                            )
                            business.lead_decision = qualification.decision.value
                            business.lead_decision_reason = qualification.reason
                    else:
                        for business in unique_batch:
                            business.lead_decision = ""
                            business.lead_decision_reason = ""

                    # Отбираем валидные лиды из батча
                    for b in unique_batch:
                        if b.is_lead:
                            b.task_id = task_id
                            leads.append(b)
                        elif not b.instagram_url:
                            skipped_no_instagram += 1
                        elif b.website_status == "good website":
                            skipped_good_site += 1
                        elif (
                            b.website_status == "uncertain website"
                            or b.lead_decision == LeadDecision.UNCERTAIN.value
                        ):
                            skipped_uncertain_website += 1

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
                "досягнуто safety-ліміт унікальних перевірених кандидатів "
                f"({checked_candidates}/{policy.max_candidates})"
            ),
            StopReason.MAX_DISCOVERY_CARDS_REACHED: (
                "досягнуто safety-ліміт відкритих карток Google Maps "
                f"({visited}/{policy.max_discovery_cards})"
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
        # The bridge reads the final qualified rows back from SQLite. It never
        # parses the human CSV/XLSX export, and remote failure cannot fail search.
        opti_summary = await finalize_completed_task(task_id)

        # --- Финальный отчёт ---
        added_no_site, added_bad_site = _added_counts()
        await _update_website_search_budget_progress()
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
            f"Відкрито карток Google Maps: {visited}/{policy.max_discovery_cards}\n"
            f"Перевірено унікальних кандидатів: {checked_candidates}/{policy.max_candidates}\n"
            f"Пропущено без Instagram: {skipped_no_instagram}\n"
            f"Пропущено з гарним сайтом: {skipped_good_site}\n"
            f"Пропущено через невизначений статус сайту: {skipped_uncertain_website}\n"
            f"Додано без сайту: {added_no_site}\n"
            f"Додано з поганим сайтом: {added_bad_site}\n"
            f"➡️ Усього лідів у таблиці: {len(leads)}\n"
            f"Причина зупинки: {reason}"
            + shortage
            + (f"\n{opti_summary}" if opti_summary else ""),
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

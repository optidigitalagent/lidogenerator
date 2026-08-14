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

import candidate_history
import city_catalog
import config
import db
from contactability import ContactChannel, lead_contact_bucket
from agents import (
    ai_scorer,
    collector,
    instagram_first_party_resolver,
    rendered_site_auditor,
    reporter,
    site_checker,
    social_checker,
    website_presence_verifier,
    website_resolver,
)
from candidate_history import CandidateClaimResult
from agents.instagram_first_party_resolver import FirstPartyInstagramResolver
from instagram_first_party_resolution import (
    FirstPartyInstagramEvidenceSource,
    FirstPartyInstagramResolution,
    FirstPartyInstagramStatus,
)
from models import Business
from integrations.opti_bridge import finalize_completed_task
from rendered_site_audit import (
    RenderedSiteAuditRuntime,
    audit_rendered_sites_shadow,
)
from website_candidate_matching import SearchProvider
from website_pipeline import LeadDecision, ResolverMode, parse_resolver_mode, qualify_lead
from website_search_runtime import (
    build_configured_search_provider,
    openai_web_search_telemetry_snapshot,
    search_budget_snapshot,
)
from website_presence import WebsitePresenceStatus
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


def _qualification_contact_available(business: Business) -> bool:
    if config.LEAD_CONTACTABILITY_MODE == "instagram_only":
        return bool(business.instagram_url)
    return business.has_actionable_contact


def _contactability_qualification_telemetry(
    candidates: List[Business],
    leads: List[Business],
    *,
    task_id: int | None,
) -> dict[str, object]:
    candidate_contactability = [business.contactability for business in candidates]
    lead_buckets = [lead_contact_bucket(business) for business in leads]
    return {
        "event": "contactability_qualification",
        "task_id": task_id,
        "mode": config.LEAD_CONTACTABILITY_MODE,
        "candidate_count": len(candidates),
        "website_need_count": sum(
            business.website_status in {"no website", "bad website"}
            for business in candidates
        ),
        "actionable_contact_count": sum(
            contactability.actionable for contactability in candidate_contactability
        ),
        "instagram_contact_count": sum(
            ContactChannel.INSTAGRAM in contactability.channels
            for contactability in candidate_contactability
        ),
        "phone_contact_count": sum(
            ContactChannel.PHONE in contactability.channels
            for contactability in candidate_contactability
        ),
        "email_contact_count": sum(
            ContactChannel.EMAIL in contactability.channels
            for contactability in candidate_contactability
        ),
        "lead_instagram_count": lead_buckets.count("instagram"),
        "lead_phone_only_count": lead_buckets.count("phone_only"),
        "lead_email_only_count": lead_buckets.count("email_only"),
        "lead_multi_contact_count": lead_buckets.count("multi_contact"),
        "no_contact_count": sum(
            not contactability.actionable
            for contactability in candidate_contactability
        ),
        "good_site_excluded_count": sum(
            business.has_actionable_contact
            and business.website_status == "good website"
            for business in candidates
        ),
        "uncertain_site_excluded_count": sum(
            business.has_actionable_contact
            and (
                business.website_status == "uncertain website"
                or business.lead_decision == LeadDecision.UNCERTAIN.value
            )
            for business in candidates
        ),
    }


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


def _first_party_apply_eligible(business: Business) -> bool:
    """Return whether real post-site-check state permits first-party recovery."""
    return (
        not business.instagram_url
        and business.website_status == "bad website"
        and instagram_first_party_resolver.trusted_website_for_instagram_resolution(
            business
        )
        is not None
    )


def _first_party_apply_telemetry(
    businesses: List[Business],
    eligible_businesses: List[Business],
    resolutions: tuple[FirstPartyInstagramResolution, ...],
    resolver: FirstPartyInstagramResolver,
    *,
    task_id: int | None,
    missing_instagram_before: int,
    bad_site_missing_instagram_count: int,
) -> dict[str, object]:
    status_counts = {
        status: sum(1 for result in resolutions if result.status is status)
        for status in FirstPartyInstagramStatus
    }
    applied_businesses = [
        business
        for business, result in zip(eligible_businesses, resolutions)
        if result.status is FirstPartyInstagramStatus.FOUND_OFFICIAL
        and business.instagram_url == result.resolved_url
    ]
    budget = resolver.budget.snapshot()
    return {
        "event": "instagram_first_party_apply",
        "task_id": task_id,
        "batch_candidate_count": len(businesses),
        "missing_instagram_before": missing_instagram_before,
        "bad_site_missing_instagram_count": bad_site_missing_instagram_count,
        "eligible_businesses": len(eligible_businesses),
        "attempted_businesses": sum(
            1 for result in resolutions if result.pages_attempted > 0
        ),
        "found_official_count": status_counts[
            FirstPartyInstagramStatus.FOUND_OFFICIAL
        ],
        "applied_count": len(applied_businesses),
        "legacy_lead_eligible_after_apply": sum(
            1 for business in applied_businesses if business.is_lead
        ),
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


async def _run_first_party_instagram_apply(
    businesses: List[Business],
    resolver: FirstPartyInstagramResolver,
    *,
    task_id: int | None,
) -> None:
    """Apply deterministic recovery to eligible real post-site-check objects."""
    if not businesses:
        return
    missing_instagram_before = sum(
        1 for business in businesses if not business.instagram_url
    )
    bad_site_missing_instagram_count = sum(
        1
        for business in businesses
        if not business.instagram_url and business.website_status == "bad website"
    )
    eligible_businesses = [
        business for business in businesses if _first_party_apply_eligible(business)
    ]
    try:
        resolutions: tuple[FirstPartyInstagramResolution, ...] = ()
        if eligible_businesses:
            resolutions = (
                await instagram_first_party_resolver.resolve_missing_instagrams_first_party(
                    eligible_businesses,
                    resolver=resolver,
                )
            )
        telemetry = _first_party_apply_telemetry(
            businesses,
            eligible_businesses,
            resolutions,
            resolver,
            task_id=task_id,
            missing_instagram_before=missing_instagram_before,
            bad_site_missing_instagram_count=bad_site_missing_instagram_count,
        )
        log.info(
            "instagram_first_party_apply %s",
            json.dumps(telemetry, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:
        log.warning(
            "instagram_first_party_apply_failed task_id=%s exception_type=%s",
            task_id,
            type(exc).__name__,
        )


async def _run_rendered_site_audit_shadow(
    businesses: List[Business],
    runtime: RenderedSiteAuditRuntime | None,
    *,
    task_id: int | None,
) -> None:
    """Observe real post-site-check state without retaining or mutating identity."""
    if runtime is None:
        return
    try:
        summary = await audit_rendered_sites_shadow(businesses, runtime)
        telemetry = summary.telemetry(task_id=task_id)
        log.info(
            "rendered_site_audit_shadow %s",
            json.dumps(telemetry, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:
        log.warning(
            "rendered_site_audit_shadow_failed task_id=%s exception_type=%s",
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
    rendered_audit_runtime: RenderedSiteAuditRuntime | None = None,
) -> None:
    """Audit production objects while isolating all shadow resolver mutations."""
    if first_party_mode not in {"off", "shadow", "apply"}:
        raise ValueError("first_party_mode must be off, shadow, or apply")
    if first_party_mode == "off":
        first_party_resolver_runtime = None
    elif first_party_resolver_runtime is None:
        first_party_resolver_runtime = (
            instagram_first_party_resolver.build_configured_first_party_resolver()
        )

    if resolver_mode is ResolverMode.OFF:
        await site_checker.check_sites(businesses)
        await _run_rendered_site_audit_shadow(
            businesses,
            rendered_audit_runtime,
            task_id=task_id,
        )
        if first_party_mode == "shadow" and first_party_resolver_runtime is not None:
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
        elif first_party_mode == "apply" and first_party_resolver_runtime is not None:
            await _run_first_party_instagram_apply(
                businesses,
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
        await _run_rendered_site_audit_shadow(
            businesses,
            rendered_audit_runtime,
            task_id=task_id,
        )
        if first_party_mode == "shadow" and first_party_resolver_runtime is not None:
            await _run_first_party_instagram_shadow(
                copy.deepcopy(businesses),
                first_party_resolver_runtime,
                task_id=task_id,
            )
        elif first_party_mode == "apply" and first_party_resolver_runtime is not None:
            await _run_first_party_instagram_apply(
                businesses,
                first_party_resolver_runtime,
                task_id=task_id,
            )
        return

    await site_checker.check_sites(businesses)
    await _run_rendered_site_audit_shadow(
        businesses,
        rendered_audit_runtime,
        task_id=task_id,
    )
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
    if first_party_mode == "shadow" and first_party_resolver_runtime is not None:
        await _run_first_party_instagram_shadow(
            shadow_businesses,
            first_party_resolver_runtime,
            task_id=task_id,
        )
    elif first_party_mode == "apply" and first_party_resolver_runtime is not None:
        await _run_first_party_instagram_apply(
            businesses,
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
    rendered_audit_runtime: RenderedSiteAuditRuntime | None = None,
    website_presence_search_provider: SearchProvider | None = None,
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
    verified_no_site = config.LEAD_WEBSITE_POLICY == "verified_no_site_only"
    if website_search_provider is not None:
        runtime_website_search_provider = website_search_provider
    elif resolver_mode is ResolverMode.OFF:
        runtime_website_search_provider = None
    else:
        runtime_website_search_provider = build_configured_search_provider()
    if config.WEBSITE_PRESENCE_VERIFICATION_MODE != "apply":
        runtime_presence_search_provider = None
    elif website_presence_search_provider is not None:
        runtime_presence_search_provider = website_presence_search_provider
    else:
        runtime_presence_search_provider = build_configured_search_provider(
            max_requests=config.MAX_WEBSITE_PRESENCE_SEARCH_REQUESTS_PER_TASK
        )
    if config.INSTAGRAM_FIRST_PARTY_MODE in {"shadow", "apply"}:
        runtime_first_party_resolver = (
            first_party_resolver_runtime
            or instagram_first_party_resolver.build_configured_first_party_resolver()
        )
    else:
        runtime_first_party_resolver = None
    if config.RENDERED_SITE_AUDIT_MODE == "shadow":
        runtime_rendered_audit = (
            rendered_audit_runtime
            or rendered_site_auditor.build_configured_rendered_site_audit_runtime()
        )
    else:
        runtime_rendered_audit = None
    stop_flag = (lambda: stop_event.is_set()) if stop_event else None

    # Накопители по ходу батчевого сбора
    leads: List[Business] = []          # валидные лиды (то, что попадёт в таблицу)
    visited = 0                         # просмотрено бизнесов всего
    checked_candidates = 0              # уникальные кандидаты с завершённой проверкой сайта
    first_party_instagrams_recovered = 0
    contactability_candidates: List[Business] = []
    skipped_no_contact = 0
    skipped_no_instagram = 0            # пропущено: нет Instagram
    skipped_good_site = 0               # пропущено: есть Instagram, но хороший сайт
    skipped_uncertain_website = 0       # пропущено: результат сайта недостаточно надёжен
    skipped_previously_checked = 0
    skipped_claimed_elsewhere = 0
    skipped_website_present = 0
    maps_presence_vetoes = 0
    search_presence_vetoes = 0
    hosted_site_vetoes = 0
    website_absent_confirmed = 0
    website_presence_uncertain = 0
    website_presence_technical_errors = 0
    website_presence_requests_used = 0
    retryable_claim_releases = 0
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
    history_scope = candidate_history.canonical_scope_key_from_resolved(
        niche_plan, city_definition, city
    )
    claimed_history_keys: dict[int, str] = {}

    def _history_claim(business: Business) -> CandidateClaimResult:
        if config.CANDIDATE_HISTORY_MODE != "apply":
            return CandidateClaimResult.CLAIMED
        basis, key = candidate_history.candidate_fingerprint(business, query_city)
        result = candidate_history.claim_candidate(
            history_scope, key, basis, task_id
        )
        if result is CandidateClaimResult.CLAIMED:
            claimed_history_keys[id(business)] = key
        return result

    def _history_checked(business: Business, outcome: str) -> None:
        if config.CANDIDATE_HISTORY_MODE != "apply":
            return
        key = claimed_history_keys.pop(id(business), None)
        if key is not None:
            candidate_history.mark_candidate_checked(
                history_scope, key, task_id, outcome
            )

    def _history_release(business: Business) -> None:
        nonlocal retryable_claim_releases
        if config.CANDIDATE_HISTORY_MODE != "apply":
            return
        key = claimed_history_keys.pop(id(business), None)
        if key is not None and candidate_history.release_candidate_claim(
            history_scope, key, task_id
        ):
            retryable_claim_releases += 1
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

    def _persist_progress(
        stage: str,
        *,
        remaining_queries: int | None = None,
        final_stop_reason: StopReason | None = None,
    ) -> None:
        added_no_site = sum(
            1 for business in leads if business.website_status == "no website"
        )
        added_bad_site = sum(
            1 for business in leads if business.website_status == "bad website"
        )
        snapshot = {
            "stage": stage,
            "targetLeads": target_leads,
            "visitedBusinesses": visited,
            "openedMapCards": visited,
            "checkedCandidates": checked_candidates,
            "qualifiedLeads": len(leads),
            "addedNoSite": added_no_site,
            "addedBadSite": added_bad_site,
            "skippedGoodSite": skipped_good_site,
            "skippedUncertainWebsite": skipped_uncertain_website,
            "skippedNoContact": skipped_no_contact,
            "skippedNoInstagram": skipped_no_instagram,
            "skippedPreviouslyChecked": skipped_previously_checked,
            "skippedClaimedElsewhere": skipped_claimed_elsewhere,
            "skippedWebsitePresent": skipped_website_present,
            "mapsPresenceVetoes": maps_presence_vetoes,
            "searchPresenceVetoes": search_presence_vetoes,
            "hostedSiteVetoes": hosted_site_vetoes,
            "websiteAbsentConfirmed": website_absent_confirmed,
            "websitePresenceUncertain": website_presence_uncertain,
            "websitePresenceTechnicalErrors": website_presence_technical_errors,
            "websitePresenceRequestsUsed": website_presence_requests_used,
            "websitePresenceRequestsMax": config.MAX_WEBSITE_PRESENCE_SEARCH_REQUESTS_PER_TASK,
            "recoveredInstagram": first_party_instagrams_recovered,
            "remainingQueries": (
                query_queue.remaining_queries
                if remaining_queries is None
                else remaining_queries
            ),
        }
        if final_stop_reason is not None:
            snapshot["stopReason"] = (
                final_stop_reason.name
                if final_stop_reason is StopReason.USER_STOPPED
                else final_stop_reason.value
            )
        db.update_task_progress(task_id, snapshot)

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

    def _contactability_progress_lines() -> str:
        if config.LEAD_CONTACTABILITY_MODE == "instagram_only":
            return f"Пропущено без Instagram: {skipped_no_instagram}\n"
        instagram_leads = sum(
            business.contactability.instagram_available for business in leads
        )
        phone_only_leads = sum(
            lead_contact_bucket(business) == "phone_only" for business in leads
        )
        email_only_leads = sum(
            lead_contact_bucket(business) == "email_only" for business in leads
        )
        multi_contact_leads = sum(
            lead_contact_bucket(business) == "multi_contact" for business in leads
        )
        return (
            "Контактні ліди:\n"
            f"- Instagram: {instagram_leads}\n"
            f"- Тільки телефон: {phone_only_leads}\n"
            f"- Тільки email: {email_only_leads}\n"
            f"- Кілька каналів: {multi_contact_leads}\n"
            "Пропущено без доступного контакту: "
            f"{skipped_no_contact}\n"
        )

    async def _update_website_search_budget_progress() -> None:
        provider = (
            runtime_presence_search_provider
            if verified_no_site
            else runtime_website_search_provider
        )
        snapshot = search_budget_snapshot(provider)
        if snapshot is not None:
            await progress.update(
                "website_search",
                (
                    f"Перевірок наявності сайту: {snapshot.used_requests}/{snapshot.max_requests}"
                    if verified_no_site
                    else f"Пошуків офіційного сайту: {snapshot.used_requests}/{snapshot.max_requests}"
                ),
            )

    async def _send_progress(force: bool = False):
        added_no_site, added_bad_site = _added_counts()
        first_party_recovery_line = (
            f"Відновлено Instagram з офіційного сайту: {first_party_instagrams_recovered}\n"
            if config.INSTAGRAM_FIRST_PARTY_MODE == "apply"
            else ""
        )
        _persist_progress("checking")
        await _update_website_search_budget_progress()
        if verified_no_site:
            await progress.update(
                "main",
                f"🔎 Шукаю {target_leads} перевірених no-site лідів: «{niche}» у місті {city}\n"
                f"Відкрито карток Google Maps: {visited}/{policy.max_discovery_cards}\n"
                f"Перевірено нових кандидатів: {checked_candidates}/{policy.max_candidates}\n"
                f"Знайдено валідних лідів: {len(leads)}/{target_leads}\n"
                f"Пропущено вже перевірених: {skipped_previously_checked}\n"
                f"Пропущено — перевіряються іншим завданням: {skipped_claimed_elsewhere}\n"
                f"Пропущено без Instagram: {skipped_no_instagram}\n"
                f"Пропущено — сайт знайдено: {skipped_website_present}\n"
                f"Пропущено — сайт не вдалося надійно перевірити: {skipped_uncertain_website}\n"
                f"Підтверджено без сайту: {website_absent_confirmed}\n"
                f"Перевірок наявності сайту: {website_presence_requests_used}/"
                f"{config.MAX_WEBSITE_PRESENCE_SEARCH_REQUESTS_PER_TASK}",
                force=force,
            )
            return
        await progress.update(
            "main",
            f"🔎 Шукаю {target_leads} якісних лідів: «{niche}» у місті {city}\n"
            f"Запрошено лідів: {target_leads}\n"
            f"Переглянуто бізнесів: {visited}\n"
            f"Відкрито карток Google Maps: {visited}/{policy.max_discovery_cards}\n"
            f"Перевірено унікальних кандидатів: {checked_candidates}/{policy.max_candidates}\n"
            f"Знайдено валідних лідів: {len(leads)}/{target_leads}\n"
            f"{_contactability_progress_lines()}"
            f"{first_party_recovery_line}"
            f"Пропущено з гарним сайтом: {skipped_good_site}\n"
            f"Пропущено через невизначений статус сайту: {skipped_uncertain_website}\n"
            f"Додано без сайту: {added_no_site}\n"
            f"Додано з поганим сайтом: {added_bad_site}",
            force=force,
        )

    async def _finalize_user_stop() -> None:
        nonlocal leads

        # The user stopped the search, so preserve any qualified partial results.
        if config.CANDIDATE_HISTORY_MODE == "apply":
            candidate_history.release_unfinished_candidate_claims(task_id)
            claimed_history_keys.clear()
        leads = limit_to_target(leads, policy.target_leads)
        if leads:
            db.save_businesses(leads)
            for business in leads:
                db.update_business(business)
        _persist_progress("stopped", final_stop_reason=StopReason.USER_STOPPED)
        db.update_task_status(task_id, "stopped")
        if progress_callback:
            await progress_callback(
                f"⏹ Пошук зупинено. Встигли зібрати {len(leads)} лідів "
                f"(переглянуто {visited} бізнесів)."
            )

    try:
        db.update_task_status(task_id, "collecting")
        _persist_progress("collecting")
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
        _persist_progress("checking")

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
                    claimed_batch: List[Business] = []
                    for business in unique_batch:
                        claim = _history_claim(business)
                        if claim is CandidateClaimResult.ALREADY_CHECKED:
                            skipped_previously_checked += 1
                            continue
                        if claim is CandidateClaimResult.CLAIMED_BY_OTHER_TASK:
                            skipped_claimed_elsewhere += 1
                            continue
                        claimed_batch.append(business)
                    unique_batch = claimed_batch

                if unique_batch and verified_no_site:
                    for business in unique_batch:
                        if stop_event is not None and stop_event.is_set():
                            raise SearchStopped()
                        contactability_candidates.append(business)
                        if not business.contactability.instagram_available:
                            business.lead_decision = LeadDecision.NOT_LEAD.value
                            business.lead_decision_reason = "instagram_missing"
                            skipped_no_instagram += 1
                            checked_candidates += 1
                            _history_checked(business, "no_instagram")
                            continue

                        result = await website_presence_verifier.verify_business_website_presence(
                            business,
                            runtime_presence_search_provider,
                        )
                        website_presence_verifier.apply_website_presence_result(
                            business, result
                        )
                        presence_budget = search_budget_snapshot(
                            runtime_presence_search_provider
                        )
                        if presence_budget is not None:
                            website_presence_requests_used = presence_budget.used_requests
                        else:
                            website_presence_requests_used += result.requests_used

                        if result.status is WebsitePresenceStatus.PRESENT:
                            business.lead_decision = LeadDecision.NOT_LEAD.value
                            business.lead_decision_reason = "website_present"
                            skipped_website_present += 1
                            if result.source and result.source.value == "maps":
                                maps_presence_vetoes += 1
                            else:
                                search_presence_vetoes += 1
                            if any("hosted_builder" in item for item in result.evidence):
                                hosted_site_vetoes += 1
                            checked_candidates += 1
                            _history_checked(
                                business,
                                "website_present_maps"
                                if result.source and result.source.value == "maps"
                                else "website_present_search",
                            )
                        elif result.status is WebsitePresenceStatus.ABSENT_CONFIRMED:
                            business.lead_decision = LeadDecision.LEAD.value
                            business.lead_decision_reason = "website_absent_confirmed"
                            website_absent_confirmed += 1
                            checked_candidates += 1
                            business.task_id = task_id
                            if business.is_lead:
                                leads.append(business)
                                _history_checked(business, "lead")
                            else:
                                _history_checked(business, "other_deterministic_exclusion")
                        elif result.status is WebsitePresenceStatus.TECHNICAL_ERROR:
                            business.lead_decision = LeadDecision.UNCERTAIN.value
                            business.lead_decision_reason = "website_presence_technical_error"
                            skipped_uncertain_website += 1
                            website_presence_technical_errors += 1
                            _history_release(business)
                        else:
                            business.lead_decision = LeadDecision.UNCERTAIN.value
                            business.lead_decision_reason = "website_presence_uncertain"
                            skipped_uncertain_website += 1
                            website_presence_uncertain += 1
                            checked_candidates += 1
                            _history_checked(business, "website_uncertain")

                elif unique_batch:
                    missing_instagram_before = tuple(
                        business
                        for business in unique_batch
                        if not business.instagram_url
                    )
                    await _check_batch_websites_with_resolver_mode(
                        unique_batch,
                        resolver_mode,
                        runtime_website_search_provider,
                        task_id=task_id,
                        first_party_mode=config.INSTAGRAM_FIRST_PARTY_MODE,
                        first_party_resolver_runtime=runtime_first_party_resolver,
                        rendered_audit_runtime=runtime_rendered_audit,
                    )
                    if config.INSTAGRAM_FIRST_PARTY_MODE == "apply":
                        first_party_instagrams_recovered += sum(
                            1
                            for business in missing_instagram_before
                            if business.instagram_url
                        )
                    checked_candidates += len(unique_batch)
                    contactability_candidates.extend(unique_batch)

                    if resolver_mode is ResolverMode.STRICT:
                        for business in unique_batch:
                            qualification = qualify_lead(
                                has_actionable_contact=(
                                    _qualification_contact_available(business)
                                ),
                                resolution=website_resolver.resolution_from_business(business),
                                audit=website_resolver.audit_from_business(business),
                            )
                            business.lead_decision = qualification.decision.value
                            business.lead_decision_reason = qualification.reason
                    else:
                        for business in unique_batch:
                            business.lead_decision = ""
                            business.lead_decision_reason = ""

                    for business in unique_batch:
                        if business.is_lead:
                            business.task_id = task_id
                            leads.append(business)
                            _history_checked(business, "lead")
                        elif not _qualification_contact_available(business):
                            if config.LEAD_CONTACTABILITY_MODE == "instagram_only":
                                skipped_no_instagram += 1
                                _history_checked(business, "no_instagram")
                            else:
                                skipped_no_contact += 1
                                _history_checked(business, "other_deterministic_exclusion")
                        elif business.website_status == "good website":
                            skipped_good_site += 1
                            _history_checked(business, "website_present_maps")
                        elif (
                            business.website_status == "uncertain website"
                            or business.lead_decision == LeadDecision.UNCERTAIN.value
                        ):
                            skipped_uncertain_website += 1
                            if business.site_quality == "technical_error":
                                _history_release(business)
                            else:
                                _history_checked(business, "website_uncertain")
                        else:
                            _history_checked(business, "other_deterministic_exclusion")

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
        _persist_progress("scoring", final_stop_reason=stop_reason)
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
        _persist_progress("done", final_stop_reason=stop_reason)
        db.update_task_status(task_id, "done", csv_path=out_path)
        # The bridge reads the final qualified rows back from SQLite. It never
        # parses the human CSV/XLSX export, and remote failure cannot fail search.
        opti_summary = await finalize_completed_task(task_id)
        contactability_telemetry = _contactability_qualification_telemetry(
            contactability_candidates,
            leads,
            task_id=task_id,
        )
        log.info(
            "contactability_qualification %s",
            json.dumps(
                contactability_telemetry,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        # --- Финальный отчёт ---
        added_no_site, added_bad_site = _added_counts()
        first_party_recovery_line = (
            f"Відновлено Instagram з офіційного сайту: {first_party_instagrams_recovered}\n"
            if config.INSTAGRAM_FIRST_PARTY_MODE == "apply"
            else ""
        )
        await _update_website_search_budget_progress()
        shortage = ""
        if len(leads) < target_leads:
            shortage = (
                f"\n⚠️ Запрошено {target_leads} лідів, "
                f"знайдено лише {len(leads)} підходящих."
            )
        if verified_no_site:
            history_payload = {
                "event": "candidate_history",
                "task_id": task_id,
                "mode": config.CANDIDATE_HISTORY_MODE,
                "newly_checked": checked_candidates,
                "previously_checked_skips": skipped_previously_checked,
                "claimed_elsewhere_skips": skipped_claimed_elsewhere,
                "retryable_releases": retryable_claim_releases,
            }
            presence_payload = {
                "event": "website_presence_verification",
                "task_id": task_id,
                "mode": config.WEBSITE_PRESENCE_VERIFICATION_MODE,
                "maps_presence_vetoes": maps_presence_vetoes,
                "search_presence_vetoes": search_presence_vetoes,
                "hosted_site_vetoes": hosted_site_vetoes,
                "absent_confirmed": website_absent_confirmed,
                "uncertain": website_presence_uncertain,
                "technical_errors": website_presence_technical_errors,
                "requests_used": website_presence_requests_used,
                "requests_max": config.MAX_WEBSITE_PRESENCE_SEARCH_REQUESTS_PER_TASK,
            }
            log.info(
                "candidate_history %s",
                json.dumps(history_payload, sort_keys=True, separators=(",", ":")),
            )
            log.info(
                "website_presence_verification %s",
                json.dumps(presence_payload, sort_keys=True, separators=(",", ":")),
            )
            await progress.update(
                "main",
                f"✅ Готово!\n"
                f"Відкрито карток Google Maps: {visited}/{policy.max_discovery_cards}\n"
                f"Перевірено нових кандидатів: {checked_candidates}/{policy.max_candidates}\n"
                f"Знайдено валідних лідів: {len(leads)}/{target_leads}\n"
                f"Пропущено вже перевірених: {skipped_previously_checked}\n"
                f"Пропущено — перевіряються іншим завданням: {skipped_claimed_elsewhere}\n"
                f"Пропущено без Instagram: {skipped_no_instagram}\n"
                f"Пропущено — сайт знайдено: {skipped_website_present}\n"
                f"Пропущено — сайт не вдалося надійно перевірити: {skipped_uncertain_website}\n"
                f"Підтверджено без сайту: {website_absent_confirmed}\n"
                f"Перевірок наявності сайту: {website_presence_requests_used}/"
                f"{config.MAX_WEBSITE_PRESENCE_SEARCH_REQUESTS_PER_TASK}\n"
                f"➡️ Усього лідів у таблиці: {len(leads)}\n"
                f"Причина зупинки: {reason}"
                + shortage
                + (f"\n{opti_summary}" if opti_summary else ""),
                force=True,
            )
            if progress_callback and leads:
                await progress_callback(reporter.format_leads_summary(leads))
            return out_path
        await progress.update(
            "main",
            f"✅ Готово!\n"
            f"Запрошено лідів: {target_leads}\n"
            f"Переглянуто бізнесів: {visited}\n"
            f"Відкрито карток Google Maps: {visited}/{policy.max_discovery_cards}\n"
            f"Перевірено унікальних кандидатів: {checked_candidates}/{policy.max_candidates}\n"
            f"{_contactability_progress_lines()}"
            f"{first_party_recovery_line}"
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
        await _finalize_user_stop()
        return None
    except Exception as e:
        if stop_event is not None and stop_event.is_set():
            log.warning(
                "search_stop_precedence task_id=%s exception_type=%s",
                task_id,
                type(e).__name__,
            )
            await _finalize_user_stop()
            return None
        if config.CANDIDATE_HISTORY_MODE == "apply":
            candidate_history.release_unfinished_candidate_claims(task_id)
            claimed_history_keys.clear()
        _persist_progress("error")
        db.update_task_status(
            task_id,
            "error",
            error_code=type(e).__name__.upper()[:100],
            error_message=str(e),
        )
        if progress_callback:
            await progress_callback(f"❌ Помилка: {type(e).__name__}: {e}")
        raise
    finally:
        if config.CANDIDATE_HISTORY_MODE == "apply" and claimed_history_keys:
            candidate_history.release_unfinished_candidate_claims(task_id)
            claimed_history_keys.clear()
        if runtime_rendered_audit is not None:
            try:
                await runtime_rendered_audit.close()
            except Exception as exc:
                log.warning(
                    "rendered_site_audit_shadow_failed task_id=%s exception_type=%s",
                    task_id,
                    type(exc).__name__,
                )


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

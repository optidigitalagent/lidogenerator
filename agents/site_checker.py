# -*- coding: utf-8 -*-
"""Site Checker — проверка наличия И качества сайтов.

Для каждого бизнеса:
  - сайт не указан → has_site=False, site_quality="none"  (лид: no website)
  - "сайт" — это ссылка на соцсеть/агрегатор (Facebook, linktr.ee, Booksy...)
    → has_site=False, site_quality="none"  (настоящего сайта нет)
  - только подтверждённые 404/410 → site_quality="dead" (no website)
  - timeout/DNS/connect, блокирующие 4xx, 429 и 5xx → technical_error
  - прочие неоднозначные 4xx → uncertain (не лид)
  - сайт открывается → эвристический вердикт по содержимому страницы:
      site_quality="bad"  — явные признаки старого/слабого сайта (лид: bad website)
      site_quality="good" — признаков проблем нет (лид отсеивается)

Запросы асинхронные, не более 5 одновременно. DNS/private-IP проверки каждого
redirect hop остаются задачей сетевого адаптера Phase 4.
"""

import asyncio
import re
import time
from typing import Awaitable, Callable, List, Optional
import httpx

import config
from models import Business
from website_pipeline import WebsiteAuditStatus, serialize_audit_evidence
from website_resolution import CandidateKind, classify_candidate_url, normalize_candidate_url

ProgressCallback = Callable[[int, int], Awaitable[None]]

HEADERS = {
    # Представляемся обычным браузером, чтобы сайты не резали запросы
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "uk-UA,uk;q=0.9,ru;q=0.8,en;q=0.7",
}

# Маркеры устаревших технологий / бесплатных конструкторов 2000-х в HTML
OLD_TECH_MARKERS = (
    "<frameset", "<frame ", "<marquee", "<font ", "<center>",
    "macromedia", "shockwave", ".swf", "ucoz", "narod.ru", "ucoz.ua",
    "msothemes", "frontpage.editor", "generator\" content=\"microsoft",
    "wordpress 3.", "wordpress 4.", "joomla! 1.", "joomla! 2.",
)


def _is_real_website(url: str) -> bool:
    """Ссылка ведёт на собственный сайт, а не на соцсеть/агрегатор?"""
    try:
        return classify_candidate_url(normalize_candidate_url(url)) not in {
            CandidateKind.SOCIAL_PROFILE,
            CandidateKind.LINK_IN_BIO,
            CandidateKind.MARKETPLACE_OR_AGGREGATOR,
            CandidateKind.DIRECTORY,
        }
    except (TypeError, ValueError):
        return False


def _store_audit(
    b: Business,
    status: WebsiteAuditStatus,
    *,
    http_status: int | None = None,
    final_url: str = "",
    evidence: tuple[str, ...] = (),
    error: str = "",
) -> None:
    b.website_audit_status = status.value
    b.website_audit_http_status = http_status
    b.website_final_url = final_url
    b.website_audit_evidence = serialize_audit_evidence(evidence)
    b.website_audit_error = error


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
    raw_url = b.effective_website_url
    if not raw_url:
        b.has_site = False
        b.site_quality = "none"
        _store_audit(
            b,
            WebsiteAuditStatus.NO_OFFICIAL_SITE,
            evidence=("no_official_site",),
        )
        return

    try:
        url = normalize_candidate_url(raw_url)
        kind = classify_candidate_url(url)
    except (TypeError, ValueError):
        b.has_site = False
        b.site_quality = "technical_error"
        _store_audit(
            b,
            WebsiteAuditStatus.TECHNICAL_ERROR,
            evidence=("invalid_candidate_url",),
            error="invalid_candidate_url",
        )
        return

    if kind in {
        CandidateKind.SOCIAL_PROFILE,
        CandidateKind.LINK_IN_BIO,
        CandidateKind.MARKETPLACE_OR_AGGREGATOR,
        CandidateKind.DIRECTORY,
    }:
        b.has_site = False
        b.site_quality = "none"
        _store_audit(
            b,
            WebsiteAuditStatus.NO_OFFICIAL_SITE,
            evidence=("non_official_platform", f"candidate_kind:{kind.value}"),
        )
        return

    async with sem:  # не более 5 одновременных запросов
        try:
            start = time.monotonic()
            resp = await client.get(url)
            elapsed = time.monotonic() - start
            final_url = normalize_candidate_url(str(resp.url))
            final_kind = classify_candidate_url(final_url)
            base_evidence = (
                f"http_status:{resp.status_code}",
                f"final_candidate_kind:{final_kind.value}",
            )
            if final_kind in {
                CandidateKind.SOCIAL_PROFILE,
                CandidateKind.LINK_IN_BIO,
                CandidateKind.MARKETPLACE_OR_AGGREGATOR,
                CandidateKind.DIRECTORY,
            }:
                b.has_site = False
                b.site_quality = "none"
                _store_audit(
                    b,
                    WebsiteAuditStatus.NO_OFFICIAL_SITE,
                    http_status=resp.status_code,
                    final_url=final_url,
                    evidence=base_evidence + ("redirected_to_non_official_platform",),
                )
            elif resp.status_code < 400:
                b.has_site = True
                b.site_quality = _analyze(resp.text, str(resp.url), elapsed)
                audit_status = (
                    WebsiteAuditStatus.GOOD
                    if b.site_quality == "good"
                    else WebsiteAuditStatus.BAD
                )
                _store_audit(
                    b,
                    audit_status,
                    http_status=resp.status_code,
                    final_url=final_url,
                    evidence=base_evidence + (f"quality:{b.site_quality}",),
                )
            elif resp.status_code in {404, 410}:
                b.has_site = False
                b.site_quality = "dead"
                _store_audit(
                    b,
                    WebsiteAuditStatus.DEAD_CONFIRMED,
                    http_status=resp.status_code,
                    final_url=final_url,
                    evidence=base_evidence + ("dead_confirmed",),
                )
            elif resp.status_code in {401, 403, 407, 408, 409, 425, 429} or resp.status_code >= 500:
                b.has_site = False
                b.site_quality = "technical_error"
                error = f"http_status_{resp.status_code}"
                _store_audit(
                    b,
                    WebsiteAuditStatus.TECHNICAL_ERROR,
                    http_status=resp.status_code,
                    final_url=final_url,
                    evidence=base_evidence + ("technical_error",),
                    error=error,
                )
            else:
                b.has_site = False
                b.site_quality = "uncertain"
                _store_audit(
                    b,
                    WebsiteAuditStatus.UNCERTAIN,
                    http_status=resp.status_code,
                    final_url=final_url,
                    evidence=base_evidence + ("http_status_uncertain",),
                )
        except httpx.TimeoutException:
            b.has_site = False
            b.site_quality = "technical_error"
            _store_audit(
                b,
                WebsiteAuditStatus.TECHNICAL_ERROR,
                evidence=("timeout",),
                error="timeout",
            )
        except httpx.RequestError:
            b.has_site = False
            b.site_quality = "technical_error"
            _store_audit(
                b,
                WebsiteAuditStatus.TECHNICAL_ERROR,
                evidence=("request_error",),
                error="request_error",
            )
        except Exception as exc:
            b.has_site = False
            b.site_quality = "technical_error"
            error = f"unexpected_error:{type(exc).__name__}"
            _store_audit(
                b,
                WebsiteAuditStatus.TECHNICAL_ERROR,
                evidence=("unexpected_error",),
                error=error,
            )


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
        # Legacy limitation: redirects are not DNS/private-IP guarded per hop,
        # and certificate verification remains disabled until the Phase 4 adapter.
        verify=False,
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

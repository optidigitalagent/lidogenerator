"""Deterministic contracts and task-scoped runtime for rendered site shadow audits.

The public results deliberately contain no URL, hostname, HTML, DOM, screenshot,
or business identity.  Production ``Business`` objects are read only.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from website_resolution import CandidateKind, classify_candidate_url, normalize_candidate_url


class RenderedSiteAuditStatus(str, Enum):
    STRONG_GOOD = "strong_good"
    STRONG_BAD = "strong_bad"
    UNCERTAIN = "uncertain"
    TECHNICAL_ERROR = "technical_error"
    SKIPPED = "skipped"


class RenderedSignal(str, Enum):
    MOBILE_LAYOUT_VIEWPORT_WIDE = "mobile_layout_viewport_wide"
    MOBILE_HORIZONTAL_OVERFLOW = "mobile_horizontal_overflow"
    MOBILE_MAJOR_ELEMENT_OVERFLOW = "mobile_major_element_overflow"
    MOBILE_TINY_TEXT = "mobile_tiny_text"
    BROKEN_VISIBLE_IMAGES = "broken_visible_images"
    MISSING_VIEWPORT_META = "missing_viewport_meta"
    DESKTOP_RENDER_OK = "desktop_render_ok"
    MOBILE_RENDER_OK = "mobile_render_ok"
    RESPONSIVE_LAYOUT_OK = "responsive_layout_ok"
    CONTENT_VISIBLE = "content_visible"
    HTTPS_FINAL_URL = "https_final_url"


@dataclass(frozen=True)
class RenderedViewportMetrics:
    viewport_width: int
    viewport_height: int
    inner_width: int
    document_scroll_width: int
    body_scroll_width: int
    visible_text_length: int
    visible_image_count: int
    broken_visible_image_count: int
    major_overflow_element_count: int
    tiny_text_element_count: int
    sampled_text_element_count: int
    viewport_meta_present: bool
    dom_content_loaded_ms: int | None

    def __post_init__(self) -> None:
        integer_fields = (
            "viewport_width",
            "viewport_height",
            "inner_width",
            "document_scroll_width",
            "body_scroll_width",
            "visible_text_length",
            "visible_image_count",
            "broken_visible_image_count",
            "major_overflow_element_count",
            "tiny_text_element_count",
            "sampled_text_element_count",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.viewport_meta_present) is not bool:
            raise TypeError("viewport_meta_present must be a bool")
        if self.dom_content_loaded_ms is not None and (
            type(self.dom_content_loaded_ms) is not int
            or self.dom_content_loaded_ms < 0
        ):
            raise ValueError("dom_content_loaded_ms must be a non-negative integer or None")
        if self.broken_visible_image_count > self.visible_image_count:
            raise ValueError("broken visible images cannot exceed visible images")
        if self.tiny_text_element_count > self.sampled_text_element_count:
            raise ValueError("tiny text elements cannot exceed sampled text elements")


@dataclass(frozen=True)
class RenderedSiteAuditResult:
    status: RenderedSiteAuditStatus
    desktop: RenderedViewportMetrics | None
    mobile: RenderedViewportMetrics | None
    signals: tuple[RenderedSignal, ...]
    pages_attempted: int
    pages_succeeded: int
    error_category: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RenderedSiteAuditStatus):
            raise TypeError("status must be a RenderedSiteAuditStatus")
        if not isinstance(self.signals, tuple) or not all(
            isinstance(signal, RenderedSignal) for signal in self.signals
        ):
            raise TypeError("signals must be a tuple of RenderedSignal values")
        if len(set(self.signals)) != len(self.signals):
            raise ValueError("signals must not contain duplicates")
        for name in ("pages_attempted", "pages_succeeded"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.pages_succeeded > self.pages_attempted:
            raise ValueError("pages_succeeded cannot exceed pages_attempted")
        if self.error_category is not None:
            if not isinstance(self.error_category, str) or not self.error_category.strip():
                raise ValueError("error_category must be a non-empty string or None")
            if any(character.isspace() for character in self.error_category):
                raise ValueError("error_category must be a stable token")


def _overflow_ratio(metrics: RenderedViewportMetrics) -> float:
    if metrics.inner_width <= 0:
        return float("inf")
    return max(metrics.document_scroll_width, metrics.body_scroll_width) / metrics.inner_width


def classify_rendered_metrics(
    desktop: RenderedViewportMetrics,
    mobile: RenderedViewportMetrics,
    *,
    https_final_url: bool,
    pages_attempted: int = 2,
    pages_succeeded: int = 2,
) -> RenderedSiteAuditResult:
    """Apply the centralized frozen thresholds to two successful renders."""
    if type(https_final_url) is not bool:
        raise TypeError("https_final_url must be a bool")

    signals: list[RenderedSignal] = [
        RenderedSignal.DESKTOP_RENDER_OK,
        RenderedSignal.MOBILE_RENDER_OK,
    ]
    overflow_ratio = _overflow_ratio(mobile)
    viewport_wide = mobile.inner_width > 500
    horizontal_overflow = overflow_ratio > 1.15
    major_overflow = mobile.major_overflow_element_count >= 2
    tiny_text = (
        mobile.sampled_text_element_count >= 10
        and mobile.tiny_text_element_count / mobile.sampled_text_element_count >= 0.30
    )
    candidate_image_metrics = (desktop, mobile)
    broken_images = any(
        metrics.visible_image_count >= 3
        and metrics.broken_visible_image_count / metrics.visible_image_count >= 0.34
        for metrics in candidate_image_metrics
    )
    missing_viewport = not mobile.viewport_meta_present
    content_visible = mobile.visible_text_length >= 150
    responsive = (
        mobile.inner_width <= 500
        and overflow_ratio <= 1.05
        and mobile.major_overflow_element_count == 0
    )

    conditions = (
        (viewport_wide, RenderedSignal.MOBILE_LAYOUT_VIEWPORT_WIDE),
        (horizontal_overflow, RenderedSignal.MOBILE_HORIZONTAL_OVERFLOW),
        (major_overflow, RenderedSignal.MOBILE_MAJOR_ELEMENT_OVERFLOW),
        (tiny_text, RenderedSignal.MOBILE_TINY_TEXT),
        (broken_images, RenderedSignal.BROKEN_VISIBLE_IMAGES),
        (missing_viewport, RenderedSignal.MISSING_VIEWPORT_META),
        (responsive, RenderedSignal.RESPONSIVE_LAYOUT_OK),
        (content_visible, RenderedSignal.CONTENT_VISIBLE),
        (https_final_url, RenderedSignal.HTTPS_FINAL_URL),
    )
    signals.extend(signal for enabled, signal in conditions if enabled)

    strong_bad = (
        viewport_wide
        or (horizontal_overflow and major_overflow)
        or (missing_viewport and horizontal_overflow)
        or (broken_images and not content_visible)
    )
    if strong_bad:
        status = RenderedSiteAuditStatus.STRONG_BAD
    elif (
        responsive
        and content_visible
        and not viewport_wide
        and not horizontal_overflow
        and not major_overflow
        and not tiny_text
        and not broken_images
        and not missing_viewport
    ):
        status = RenderedSiteAuditStatus.STRONG_GOOD
    else:
        status = RenderedSiteAuditStatus.UNCERTAIN

    return RenderedSiteAuditResult(
        status=status,
        desktop=desktop,
        mobile=mobile,
        signals=tuple(signals),
        pages_attempted=pages_attempted,
        pages_succeeded=pages_succeeded,
        error_category=None,
    )


def rendered_technical_error(
    error_category: str,
    *,
    pages_attempted: int = 0,
    pages_succeeded: int = 0,
) -> RenderedSiteAuditResult:
    return RenderedSiteAuditResult(
        status=RenderedSiteAuditStatus.TECHNICAL_ERROR,
        desktop=None,
        mobile=None,
        signals=(),
        pages_attempted=pages_attempted,
        pages_succeeded=pages_succeeded,
        error_category=error_category,
    )


def rendered_uncertain(
    error_category: str,
    *,
    pages_attempted: int,
    pages_succeeded: int,
) -> RenderedSiteAuditResult:
    return RenderedSiteAuditResult(
        status=RenderedSiteAuditStatus.UNCERTAIN,
        desktop=None,
        mobile=None,
        signals=(),
        pages_attempted=pages_attempted,
        pages_succeeded=pages_succeeded,
        error_category=error_category,
    )


def rendered_audit_url(business: object) -> str | None:
    """Return only the URL proven by the real site checker, never resolver shadow."""
    if getattr(business, "has_site", None) is not True:
        return None
    quality = getattr(business, "site_quality", None)
    if quality not in {"good", "bad"}:
        return None
    if getattr(business, "website_audit_status", None) != quality:
        return None
    if getattr(business, "website_audit_error", ""):
        return None
    final_url = getattr(business, "website_final_url", "")
    if not isinstance(final_url, str) or not final_url.strip():
        return None
    try:
        normalized = normalize_candidate_url(final_url)
        kind = classify_candidate_url(normalized)
    except (TypeError, ValueError):
        return None
    if kind in {
        CandidateKind.SOCIAL_PROFILE,
        CandidateKind.LINK_IN_BIO,
        CandidateKind.MARKETPLACE_OR_AGGREGATOR,
        CandidateKind.DIRECTORY,
    }:
        return None
    return normalized


def rendered_audit_eligible(business: object) -> bool:
    return rendered_audit_url(business) is not None


class RenderedAuditor(Protocol):
    async def audit(self, url: str) -> RenderedSiteAuditResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class RenderedAuditBudgetSnapshot:
    max_audits: int
    used_audits: int
    remaining_audits: int


@dataclass(frozen=True)
class RenderedSiteAuditShadowSummary:
    batch_candidate_count: int
    eligible_bad_count: int
    eligible_good_count: int
    audited_bad_count: int
    audited_good_count: int
    strong_bad_count: int
    strong_good_count: int
    uncertain_count: int
    technical_error_count: int
    skipped_budget_count: int
    agreement_counts: tuple[tuple[str, int], ...]
    signal_counts: tuple[tuple[str, int], ...]
    bad_budget: RenderedAuditBudgetSnapshot
    good_budget: RenderedAuditBudgetSnapshot
    pages_attempted: int
    pages_succeeded: int

    def telemetry(self, *, task_id: int | None) -> dict[str, object]:
        return {
            "event": "rendered_site_audit_shadow",
            "task_id": task_id,
            "batch_candidate_count": self.batch_candidate_count,
            "eligible_bad_count": self.eligible_bad_count,
            "eligible_good_count": self.eligible_good_count,
            "audited_bad_count": self.audited_bad_count,
            "audited_good_count": self.audited_good_count,
            "strong_bad_count": self.strong_bad_count,
            "strong_good_count": self.strong_good_count,
            "uncertain_count": self.uncertain_count,
            "technical_error_count": self.technical_error_count,
            "skipped_budget_count": self.skipped_budget_count,
            **dict(self.agreement_counts),
            **dict(self.signal_counts),
            "bad_budget": {
                "max": self.bad_budget.max_audits,
                "used": self.bad_budget.used_audits,
                "remaining": self.bad_budget.remaining_audits,
            },
            "good_budget": {
                "max": self.good_budget.max_audits,
                "used": self.good_budget.used_audits,
                "remaining": self.good_budget.remaining_audits,
            },
            "pages_attempted": self.pages_attempted,
            "pages_succeeded": self.pages_succeeded,
        }


_AGREEMENT_KEYS = (
    "legacy_bad_rendered_strong_bad",
    "legacy_bad_rendered_strong_good",
    "legacy_bad_rendered_uncertain",
    "legacy_good_rendered_strong_good",
    "legacy_good_rendered_strong_bad",
    "legacy_good_rendered_uncertain",
)

_SIGNAL_KEYS = tuple(
    f"{signal.value}_count"
    for signal in (
        RenderedSignal.MOBILE_LAYOUT_VIEWPORT_WIDE,
        RenderedSignal.MOBILE_HORIZONTAL_OVERFLOW,
        RenderedSignal.MOBILE_MAJOR_ELEMENT_OVERFLOW,
        RenderedSignal.MOBILE_TINY_TEXT,
        RenderedSignal.BROKEN_VISIBLE_IMAGES,
        RenderedSignal.MISSING_VIEWPORT_META,
        RenderedSignal.RESPONSIVE_LAYOUT_OK,
    )
)


class RenderedSiteAuditRuntime:
    """Own task-global independent legacy-bad and legacy-good audit quotas."""

    def __init__(
        self,
        auditor: RenderedAuditor,
        *,
        max_bad_audits: int,
        max_good_audits: int,
        concurrency: int,
    ) -> None:
        for name, value in (
            ("max_bad_audits", max_bad_audits),
            ("max_good_audits", max_good_audits),
            ("concurrency", concurrency),
        ):
            if type(value) is not int or value < (1 if name == "concurrency" else 0):
                raise ValueError(f"{name} is invalid")
        self._auditor = auditor
        self._max_bad = max_bad_audits
        self._max_good = max_good_audits
        self._used_bad = 0
        self._used_good = 0
        self._semaphore = asyncio.Semaphore(concurrency)
        self._budget_lock = asyncio.Lock()

    def _budget(self, quality: str) -> RenderedAuditBudgetSnapshot:
        maximum = self._max_bad if quality == "bad" else self._max_good
        used = self._used_bad if quality == "bad" else self._used_good
        return RenderedAuditBudgetSnapshot(maximum, used, maximum - used)

    async def _audit_one(self, url: str) -> RenderedSiteAuditResult:
        async with self._semaphore:
            try:
                result = await self._auditor.audit(url)
                if not isinstance(result, RenderedSiteAuditResult):
                    raise TypeError("auditor returned an invalid result")
                return result
            except asyncio.CancelledError:
                raise
            except Exception:
                return rendered_technical_error("auditor_error")

    async def audit_batch(self, businesses: Sequence[object]) -> RenderedSiteAuditShadowSummary:
        eligible: dict[str, list[tuple[object, str]]] = {"bad": [], "good": []}
        for business in businesses:
            url = rendered_audit_url(business)
            quality = getattr(business, "site_quality", None)
            if url is not None and quality in eligible:
                eligible[quality].append((business, url))

        async with self._budget_lock:
            bad_remaining = self._max_bad - self._used_bad
            good_remaining = self._max_good - self._used_good
            selected_bad = eligible["bad"][:bad_remaining]
            selected_good = eligible["good"][:good_remaining]
            self._used_bad += len(selected_bad)
            self._used_good += len(selected_good)
        selected = [("bad", url) for _, url in selected_bad]
        selected.extend(("good", url) for _, url in selected_good)
        results = await asyncio.gather(*(self._audit_one(url) for _, url in selected))

        agreements = {key: 0 for key in _AGREEMENT_KEYS}
        signal_counts = {key: 0 for key in _SIGNAL_KEYS}
        statuses = {status: 0 for status in RenderedSiteAuditStatus}
        pages_attempted = 0
        pages_succeeded = 0
        for (legacy_quality, _), result in zip(selected, results):
            statuses[result.status] += 1
            pages_attempted += result.pages_attempted
            pages_succeeded += result.pages_succeeded
            for signal in result.signals:
                key = f"{signal.value}_count"
                if key in signal_counts:
                    signal_counts[key] += 1
            if result.status in {
                RenderedSiteAuditStatus.STRONG_BAD,
                RenderedSiteAuditStatus.STRONG_GOOD,
                RenderedSiteAuditStatus.UNCERTAIN,
            }:
                agreements[
                    f"legacy_{legacy_quality}_rendered_{result.status.value}"
                ] += 1

        return RenderedSiteAuditShadowSummary(
            batch_candidate_count=len(businesses),
            eligible_bad_count=len(eligible["bad"]),
            eligible_good_count=len(eligible["good"]),
            audited_bad_count=len(selected_bad),
            audited_good_count=len(selected_good),
            strong_bad_count=statuses[RenderedSiteAuditStatus.STRONG_BAD],
            strong_good_count=statuses[RenderedSiteAuditStatus.STRONG_GOOD],
            uncertain_count=statuses[RenderedSiteAuditStatus.UNCERTAIN],
            technical_error_count=statuses[RenderedSiteAuditStatus.TECHNICAL_ERROR],
            skipped_budget_count=(
                len(eligible["bad"]) - len(selected_bad)
                + len(eligible["good"]) - len(selected_good)
            ),
            agreement_counts=tuple(agreements.items()),
            signal_counts=tuple(signal_counts.items()),
            bad_budget=self._budget("bad"),
            good_budget=self._budget("good"),
            pages_attempted=pages_attempted,
            pages_succeeded=pages_succeeded,
        )

    async def close(self) -> None:
        await self._auditor.close()


async def audit_rendered_sites_shadow(
    businesses: Sequence[object],
    runtime: RenderedSiteAuditRuntime,
) -> RenderedSiteAuditShadowSummary:
    if not isinstance(runtime, RenderedSiteAuditRuntime):
        raise TypeError("runtime must be a RenderedSiteAuditRuntime")
    return await runtime.audit_batch(businesses)

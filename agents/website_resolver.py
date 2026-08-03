"""Website resolver agent composed only from the pure Phase 1/2 contracts."""

from __future__ import annotations

from typing import Awaitable, Callable

from models import Business
from website_candidate_matching import (
    BusinessIdentity,
    SearchProvider,
    SearchRequest,
    SourceAttempt,
    SourceAttemptStatus,
    WebsiteCandidate,
    collect_web_search_candidates,
    resolve_website_candidates,
)
from website_pipeline import (
    WebsiteAuditResult,
    WebsiteAuditStatus,
    deserialize_audit_evidence,
    deserialize_candidate_evidence,
    serialize_candidate_evidence,
)
from website_resolution import (
    CandidateSource,
    ResolutionStatus,
    WebsiteResolution,
)

ProgressCallback = Callable[[int, int], Awaitable[None]]


def _resolution_error(detail: str) -> WebsiteResolution:
    return WebsiteResolution(
        ResolutionStatus.RESOLUTION_ERROR,
        None,
        None,
        0.0,
        (),
        error=detail,
    )


def _business_identity(business: Business) -> tuple[BusinessIdentity | None, str | None]:
    address = business.address if isinstance(business.address, str) and business.address.strip() else None
    phone: str | None = business.phone or None
    if phone is not None:
        try:
            BusinessIdentity(name="probe", city="probe", phone=phone)
        except (TypeError, ValueError):
            phone = None

    instagram_url: str | None = business.instagram_url or None
    instagram_error = None
    if instagram_url is not None:
        try:
            BusinessIdentity(name="probe", city="probe", instagram_url=instagram_url)
        except (TypeError, ValueError):
            instagram_url = None
            instagram_error = "invalid instagram identity"
    try:
        identity = BusinessIdentity(
            name=business.name,
            city=business.city,
            address=address,
            phone=phone,
            instagram_url=instagram_url,
        )
    except (TypeError, ValueError):
        return None, "invalid required business identity"
    return identity, instagram_error


async def resolve_business_website(
    business: Business,
    provider: SearchProvider | None = None,
) -> WebsiteResolution:
    business.website_original_url = business.website
    identity, identity_error = _business_identity(business)
    if identity is None:
        return _resolution_error(identity_error or "invalid required business identity")

    candidates: list[WebsiteCandidate] = []
    attempts: list[SourceAttempt] = []
    if business.website:
        try:
            candidates.append(WebsiteCandidate(CandidateSource.MAPS, business.website))
            attempts.append(SourceAttempt(CandidateSource.MAPS, SourceAttemptStatus.COMPLETED))
        except (TypeError, ValueError):
            attempts.append(SourceAttempt(
                CandidateSource.MAPS,
                SourceAttemptStatus.ERROR,
                "invalid maps website URL",
            ))
    else:
        attempts.append(SourceAttempt(CandidateSource.MAPS, SourceAttemptStatus.COMPLETED))

    if business.instagram_bio_url:
        try:
            candidates.append(WebsiteCandidate(
                CandidateSource.INSTAGRAM_BIO,
                business.instagram_bio_url,
            ))
            attempts.append(SourceAttempt(
                CandidateSource.INSTAGRAM_BIO,
                SourceAttemptStatus.COMPLETED,
            ))
        except (TypeError, ValueError):
            attempts.append(SourceAttempt(
                CandidateSource.INSTAGRAM_BIO,
                SourceAttemptStatus.ERROR,
                "invalid instagram bio URL",
            ))
    elif business.instagram_url:
        attempts.append(SourceAttempt(
            CandidateSource.INSTAGRAM_BIO,
            SourceAttemptStatus.ERROR if identity_error else SourceAttemptStatus.SKIPPED,
            identity_error or "instagram bio URL not collected",
        ))
    else:
        attempts.append(SourceAttempt(
            CandidateSource.INSTAGRAM_BIO,
            SourceAttemptStatus.SKIPPED,
            "instagram profile unavailable",
        ))

    if provider is None:
        attempts.append(SourceAttempt(
            CandidateSource.WEB_SEARCH,
            SourceAttemptStatus.UNAVAILABLE,
            "web search provider not configured",
        ))
    else:
        try:
            web_candidates, web_attempt = await collect_web_search_candidates(
                provider,
                SearchRequest(
                    business_name=identity.name,
                    city=identity.city,
                    address=identity.address,
                    phone=identity.phone,
                ),
            )
            candidates.extend(web_candidates)
            if web_attempt.status is not SourceAttemptStatus.COMPLETED:
                web_attempt = SourceAttempt(
                    CandidateSource.WEB_SEARCH,
                    web_attempt.status,
                    f"web_search_{web_attempt.status.value}",
                )
            attempts.append(web_attempt)
        except Exception as exc:
            attempts.append(SourceAttempt(
                CandidateSource.WEB_SEARCH,
                SourceAttemptStatus.ERROR,
                f"unexpected provider error: {type(exc).__name__}",
            ))

    required = [CandidateSource.MAPS, CandidateSource.WEB_SEARCH]
    if business.instagram_url:
        required.append(CandidateSource.INSTAGRAM_BIO)
    try:
        return resolve_website_candidates(identity, candidates, attempts, required)
    except Exception as exc:
        return _resolution_error(f"resolution contract error: {type(exc).__name__}")


def apply_resolution(business: Business, resolution: WebsiteResolution) -> None:
    business.website_original_url = business.website
    business.website_resolved_url = resolution.resolved_url or ""
    business.website_resolution_status = resolution.status.value
    business.website_resolution_source = resolution.source.value if resolution.source else ""
    business.website_resolution_confidence = resolution.confidence
    business.website_resolution_evidence = serialize_candidate_evidence(resolution.evidence)
    business.website_resolution_error = resolution.error or ""


async def resolve_business_websites(
    businesses: list[Business],
    provider: SearchProvider | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[Business]:
    for index, business in enumerate(businesses, start=1):
        resolution = await resolve_business_website(business, provider)
        apply_resolution(business, resolution)
        if progress_callback:
            await progress_callback(index, len(businesses))
    return businesses


def resolution_from_business(business: Business) -> WebsiteResolution:
    """Strictly reconstruct persisted resolver state, failing closed."""
    try:
        return WebsiteResolution(
            status=ResolutionStatus(business.website_resolution_status),
            resolved_url=business.website_resolved_url or None,
            source=(
                CandidateSource(business.website_resolution_source)
                if business.website_resolution_source
                else None
            ),
            confidence=business.website_resolution_confidence,
            evidence=deserialize_candidate_evidence(business.website_resolution_evidence),
            error=business.website_resolution_error or None,
        )
    except (TypeError, ValueError):
        return _resolution_error("invalid stored website resolution")


def audit_from_business(business: Business) -> WebsiteAuditResult:
    """Strictly reconstruct persisted audit state, failing closed."""
    try:
        status = WebsiteAuditStatus(business.website_audit_status)
        no_request_data = status in {
            WebsiteAuditStatus.NOT_RUN,
            WebsiteAuditStatus.NO_OFFICIAL_SITE,
        }
        return WebsiteAuditResult(
            status=status,
            audited_url=None if no_request_data else business.effective_website_url or None,
            final_url=None if no_request_data else business.website_final_url or None,
            http_status=None if no_request_data else business.website_audit_http_status,
            evidence=deserialize_audit_evidence(business.website_audit_evidence),
            error=business.website_audit_error or None,
        )
    except (TypeError, ValueError):
        audited_url = business.effective_website_url
        try:
            from website_resolution import normalize_candidate_url
            audited_url = normalize_candidate_url(audited_url)
        except (TypeError, ValueError):
            audited_url = "https://invalid-stored-audit.invalid/"
        return WebsiteAuditResult(
            WebsiteAuditStatus.TECHNICAL_ERROR,
            audited_url,
            None,
            None,
            ("invalid_stored_audit",),
            "invalid stored website audit",
        )

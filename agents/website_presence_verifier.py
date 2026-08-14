"""One-call targeted verification that a candidate has no real website."""

from __future__ import annotations

import json

from models import Business
from website_candidate_matching import (
    BusinessIdentity,
    CandidateKind,
    SearchProvider,
    SearchRequest,
    SourceAttemptStatus,
    assess_website_candidate,
    collect_web_search_candidates,
)
from website_presence import (
    WebsitePresenceResult,
    WebsitePresenceSource,
    WebsitePresenceStatus,
    classify_website_presence_url,
    is_hosted_site_builder_url,
)


def explicit_maps_presence(business: Business) -> WebsitePresenceResult | None:
    if business.website and classify_website_presence_url(business.website):
        evidence = (
            "maps_hosted_builder"
            if is_hosted_site_builder_url(business.website)
            else "maps_url_present"
        )
        return WebsitePresenceResult(
            WebsitePresenceStatus.PRESENT,
            WebsitePresenceSource.MAPS,
            business.website,
            (evidence,),
        )
    return None


def _identity(business: Business) -> BusinessIdentity:
    address = business.address.strip() or None
    phone = business.phone.strip() or None
    instagram = business.instagram_url.strip() or None
    try:
        return BusinessIdentity(
            business.name,
            business.city,
            address=address,
            phone=phone,
            instagram_url=instagram,
        )
    except (TypeError, ValueError):
        # Invalid optional values must not prevent conservative name/city matching.
        return BusinessIdentity(business.name, business.city, address=address)


def _hosted_builder_matches(identity: BusinessIdentity, candidate: object) -> bool:
    evidence = getattr(candidate, "identity_evidence", None)
    if evidence is None or not evidence.candidate_url_source_bound:
        return False
    if not evidence.name_matches or not evidence.city_matches or evidence.different_city_detected:
        return False
    corroboration_required = identity.phone is not None or identity.address is not None
    if not corroboration_required:
        return False
    return bool(
        (identity.phone is not None and evidence.phone_matches)
        or (identity.address is not None and evidence.address_matches)
    )


async def verify_business_website_presence(
    business: Business,
    provider: SearchProvider | None,
) -> WebsitePresenceResult:
    maps = explicit_maps_presence(business)
    if maps is not None:
        return maps
    if provider is None:
        return WebsitePresenceResult(
            WebsitePresenceStatus.TECHNICAL_ERROR,
            WebsitePresenceSource.WEB_SEARCH,
            evidence=("provider_unavailable",),
            error_category="unavailable",
        )
    try:
        identity = _identity(business)
    except (TypeError, ValueError):
        return WebsitePresenceResult(
            WebsitePresenceStatus.UNCERTAIN,
            WebsitePresenceSource.WEB_SEARCH,
            evidence=("invalid_identity",),
            requests_used=0,
        )
    try:
        candidates, attempt = await collect_web_search_candidates(
            provider,
            SearchRequest(
                identity.name,
                identity.city,
                address=identity.address,
                phone=identity.phone,
                instagram_url=identity.instagram_url,
            ),
        )
    except Exception:
        return WebsitePresenceResult(
            WebsitePresenceStatus.TECHNICAL_ERROR,
            WebsitePresenceSource.WEB_SEARCH,
            evidence=("provider_error",),
            error_category="unexpected",
            requests_used=1,
        )

    if attempt.status is not SourceAttemptStatus.COMPLETED:
        category = {
            SourceAttemptStatus.UNAVAILABLE: "unavailable",
            SourceAttemptStatus.TIMEOUT: "timeout",
            SourceAttemptStatus.RATE_LIMITED: "rate_limited",
            SourceAttemptStatus.ERROR: "provider_error",
            SourceAttemptStatus.SKIPPED: "skipped",
        }.get(attempt.status, "provider_error")
        return WebsitePresenceResult(
            WebsitePresenceStatus.TECHNICAL_ERROR,
            WebsitePresenceSource.WEB_SEARCH,
            evidence=(f"web_{category}",),
            error_category=category,
            requests_used=1,
        )

    uncertain_site_candidate = False
    for candidate in candidates:
        if not classify_website_presence_url(candidate.url):
            continue
        if is_hosted_site_builder_url(candidate.url):
            accepted = _hosted_builder_matches(identity, candidate)
        else:
            accepted = assess_website_candidate(identity, candidate).kind is CandidateKind.OFFICIAL_WEBSITE
        if accepted:
            evidence = (
                "web_hosted_builder_match"
                if is_hosted_site_builder_url(candidate.url)
                else "web_official_match"
            )
            return WebsitePresenceResult(
                WebsitePresenceStatus.PRESENT,
                WebsitePresenceSource.WEB_SEARCH,
                candidate.url,
                (evidence,),
                requests_used=1,
            )
        uncertain_site_candidate = True

    if uncertain_site_candidate:
        return WebsitePresenceResult(
            WebsitePresenceStatus.UNCERTAIN,
            WebsitePresenceSource.WEB_SEARCH,
            evidence=("web_identity_uncertain",),
            requests_used=1,
        )
    return WebsitePresenceResult(
        WebsitePresenceStatus.ABSENT_CONFIRMED,
        WebsitePresenceSource.WEB_SEARCH,
        evidence=("web_no_site_match",),
        requests_used=1,
    )


def apply_website_presence_result(
    business: Business,
    result: WebsitePresenceResult,
) -> None:
    business.website_presence_status = result.status.value
    business.website_presence_source = result.source.value if result.source else ""
    business.website_presence_resolved_url = result.resolved_url or ""
    business.website_presence_evidence = json.dumps(
        result.evidence, separators=(",", ":")
    )
    business.website_presence_error = result.error_category or ""

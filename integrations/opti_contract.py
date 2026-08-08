"""Versioned, bounded Opti Lead Import contract and stable lead identities."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from models import Business

SCHEMA_VERSION = "opti.lead-import.v1"
SOURCE_SYSTEM = "lidogenerator"
MAX_BATCH_SIZE = 200

IDENTITY_BASES = {
    "INTERNAL_CANDIDATE_ID",
    "GOOGLE_PLACE_ID",
    "GOOGLE_MAPS_URL",
    "PHONE",
    "INSTAGRAM",
    "WEBSITE_DOMAIN",
    "NAME_CITY_ADDRESS_HASH",
}
WEBSITE_STATUSES = {"NO_SITE", "BAD_SITE", "GOOD_SITE", "UNCERTAIN", "NOT_CHECKED"}
SITE_QUALITIES = {
    "none", "dead", "bad", "good", "uncertain", "technical_error", "unknown"
}


class ContractError(ValueError):
    """The locally generated payload violates the shared bounded contract."""


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", _text(value)).casefold()
    return " ".join(normalized.split())


def normalize_phone(value: Any) -> str:
    raw = _text(value)
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        return ""
    return f"+{digits}" if raw.lstrip().startswith("+") else digits


def normalize_instagram_handle(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    if host == "instagram.com" or host.endswith(".instagram.com"):
        return parsed.path.strip("/").split("/", 1)[0].lstrip("@").casefold()
    return raw.lstrip("@").split("?", 1)[0].strip("/").casefold()


def normalize_website_domain(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").casefold().rstrip(".").removeprefix("www.")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def canonical_http_url(value: Any) -> str | None:
    """Return a deterministic HTTP(S) URL without performing network I/O."""
    raw = _text(value)
    if not raw:
        return None
    candidate = raw
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"
    elif "://" not in candidate:
        candidate = f"https://{candidate}"
    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.casefold()
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if scheme not in {"http", "https"} or not hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    if any(character.isspace() for character in hostname):
        return None
    try:
        normalized_host = hostname.casefold().rstrip(".").encode("idna").decode("ascii")
    except UnicodeError:
        return None
    if not normalized_host:
        return None
    netloc_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = (
        f"{netloc_host}:{port}"
        if port is not None and not default_port
        else netloc_host
    )
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, ""))


def canonical_instagram_url(value: Any) -> str | None:
    """Canonicalize a handle or Instagram URL to one HTTPS profile form."""
    raw = _text(value)
    if not raw:
        return None
    direct = raw.lstrip("@")
    if re.fullmatch(r"[A-Za-z0-9._]{1,30}", direct):
        handle = direct
    else:
        canonical = canonical_http_url(raw)
        if canonical is None:
            return None
        parsed = urlsplit(canonical)
        if (parsed.hostname or "").casefold().removeprefix("www.") != "instagram.com":
            return None
        handle = parsed.path.strip("/").split("/", 1)[0].lstrip("@")
    if not re.fullmatch(r"[A-Za-z0-9._]{1,30}", handle):
        return None
    return f"https://www.instagram.com/{handle.casefold()}/"


def normalize_google_maps_url(value: Any) -> str:
    canonical = canonical_http_url(value)
    if canonical is None:
        return ""
    parsed = urlsplit(canonical)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if parsed.port and not (
        (parsed.scheme.casefold() == "http" and parsed.port == 80)
        or (parsed.scheme.casefold() == "https" and parsed.port == 443)
    ):
        host = f"{host}:{parsed.port}"
    path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/") or "/"
    ignored = {"hl", "authuser", "entry", "g_ep"}
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold() not in ignored and not key.casefold().startswith("utm_")
        )
    ).replace("~", "%7E")
    return urlunsplit(((parsed.scheme or "https").casefold(), host, path, query, ""))


def extract_google_place_id(value: Any) -> str:
    raw = _text(value)
    if not raw:
        return ""
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    for key, item in parse_qsl(parsed.query):
        if key in {"query_place_id", "place_id"} and item:
            return item
    match = re.search(r"(?:!1s|/place_id:)(ChI[\w-]+)", raw)
    return match.group(1) if match else ""


def stable_external_lead_id(business: Business | Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(business, Mapping):
        get = lambda name: business.get(name, "")
    else:
        get = lambda name: getattr(business, name, "")
    # Keep the priority order explicit; SQLite autoincrement ``id`` is deliberately absent.
    candidates = (
        (
            "INTERNAL_CANDIDATE_ID",
            _text(get("external_candidate_id"))
            or _text(get("global_candidate_id"))
            or _text(get("stable_candidate_id")),
        ),
        ("GOOGLE_PLACE_ID", _text(get("google_place_id")) or extract_google_place_id(get("google_maps_url"))),
        ("GOOGLE_MAPS_URL", normalize_google_maps_url(get("google_maps_url"))),
        ("PHONE", normalize_phone(get("phone"))),
        ("INSTAGRAM", normalize_instagram_handle(get("instagram_url"))),
        ("WEBSITE_DOMAIN", normalize_website_domain(get("effective_website_url") or get("website"))),
    )
    basis = ""
    identity = ""
    for candidate_basis, candidate_value in candidates:
        if candidate_value:
            basis, identity = candidate_basis, candidate_value
            break
    if not identity:
        basis = "NAME_CITY_ADDRESS_HASH"
        composite = "\x1f".join(
            normalize_text(get(field)) for field in ("name", "city", "address")
        )
        identity = hashlib.sha256(composite.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(f"v1\x1f{basis}\x1f{identity}".encode("utf-8")).hexdigest()
    return f"ldg:v1:{basis.casefold()}:{digest}", basis


def _iso_utc(value: Any, fallback: datetime | None = None) -> str:
    raw = _text(value)
    if raw:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            # Legacy tasks used local naive timestamps. New tasks are written as UTC.
            parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    else:
        parsed = fallback or datetime.now(timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _nullable(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _website_assessment(business: Business, checked_at: str) -> dict[str, Any]:
    quality = _text(business.site_quality).casefold()
    if quality not in SITE_QUALITIES:
        quality = "unknown"
    status = {
        "none": "NO_SITE",
        "dead": "NO_SITE",
        "bad": "BAD_SITE",
        "good": "GOOD_SITE",
        "uncertain": "UNCERTAIN",
        "technical_error": "UNCERTAIN",
        "unknown": "NOT_CHECKED",
    }[quality]
    return {
        "status": status,
        "hasSite": bool(business.has_site),
        "siteQuality": quality,
        "checkedAt": checked_at,
        "finalUrl": canonical_http_url(
            business.website_final_url or business.effective_website_url
        ),
    }


def build_payload(
    task: Mapping[str, Any],
    businesses: Iterable[Business],
    *,
    generated_at: datetime | None = None,
    batch_status: str = "COMPLETED",
) -> dict[str, Any]:
    leads = list(businesses)
    generated = _iso_utc(task.get("finished_at"), generated_at)
    completed = _iso_utc(task.get("finished_at"), generated_at)
    items = []
    for rank, business in enumerate(leads, start=1):
        external_id, basis = stable_external_lead_id(business)
        place_id = business.google_place_id or extract_google_place_id(business.google_maps_url)
        items.append(
            {
                "externalLeadId": external_id,
                "identityBasis": basis,
                "rank": rank,
                "name": _text(business.name),
                "category": _nullable(business.category or business.niche),
                "city": _nullable(business.city),
                "address": _nullable(business.address),
                "phone": _nullable(business.phone),
                "email": _nullable(business.email),
                "instagramUrl": canonical_instagram_url(business.instagram_url),
                "websiteUrl": canonical_http_url(business.effective_website_url),
                "googleMapsUrl": canonical_http_url(
                    normalize_google_maps_url(business.google_maps_url)
                ),
                "googlePlaceId": _nullable(place_id),
                "rating": business.rating if business.rating else None,
                "reviewCount": business.reviews_count if business.reviews_count >= 0 else None,
                "websiteAssessment": _website_assessment(business, completed),
                "collectedAt": (
                    _iso_utc(business.collected_at, generated_at)
                    if business.collected_at
                    else None
                ),
            }
        )
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "sourceSystem": SOURCE_SYSTEM,
        "externalBatchId": _text(task.get("external_batch_id")),
        "batchStatus": batch_status,
        "generatedAt": generated,
        "search": {
            "niche": _nullable(task.get("niche")),
            "city": _nullable(task.get("city")),
            "targetCount": int(task.get("count", 0)),
            "resultCount": len(items),
            "startedAt": _iso_utc(task.get("created_at"), generated_at),
            "completedAt": completed,
        },
        "leads": items,
    }
    validate_payload(payload)
    return payload


def _bounded_string(value: Any, name: str, maximum: int, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str) or (not value and not nullable) or len(value) > maximum:
        raise ContractError(f"{name} must be a string of at most {maximum} characters")


def _valid_time(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    _bounded_string(value, name, 100)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ContractError(f"{name} must include a UTC offset")


def _bounded_http_url(value: Any, name: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    _bounded_string(value, name, 2000)
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.casefold()
        hostname = parsed.hostname
        parsed.port
    except (TypeError, ValueError) as error:
        raise ContractError(f"{name} must be a valid HTTP(S) URL") from error
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in hostname)
    ):
        raise ContractError(f"{name} must be a valid HTTP(S) URL")


def validate_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schemaVersion", "sourceSystem", "externalBatchId", "batchStatus",
        "generatedAt", "search", "leads",
    }:
        raise ContractError("payload fields do not match the versioned contract")
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise ContractError("unsupported schemaVersion")
    if payload.get("sourceSystem") != SOURCE_SYSTEM:
        raise ContractError("unsupported sourceSystem")
    _bounded_string(payload.get("externalBatchId"), "externalBatchId", 500)
    if payload.get("batchStatus") not in {"COMPLETED", "STOPPED_PARTIAL"}:
        raise ContractError("invalid batchStatus")
    _valid_time(payload.get("generatedAt"), "generatedAt")
    search = payload.get("search")
    if not isinstance(search, Mapping):
        raise ContractError("search must be an object")
    if set(search) != {
        "niche", "city", "targetCount", "resultCount", "startedAt", "completedAt"
    }:
        raise ContractError("search fields do not match the versioned contract")
    for field in ("niche", "city"):
        _bounded_string(search.get(field), f"search.{field}", 500, nullable=True)
    target = search.get("targetCount")
    if isinstance(target, bool) or not isinstance(target, int) or not 1 <= target <= MAX_BATCH_SIZE:
        raise ContractError("search.targetCount must be between 1 and 200")
    leads = payload.get("leads")
    if not isinstance(leads, list) or not 1 <= len(leads) <= MAX_BATCH_SIZE:
        raise ContractError("leads must contain between 1 and 200 items")
    if search.get("resultCount") != len(leads):
        raise ContractError("search.resultCount must equal leads length")
    for field in ("startedAt", "completedAt"):
        _valid_time(search.get(field), f"search.{field}")
    seen_ids: set[str] = set()
    for index, lead in enumerate(leads):
        prefix = f"leads[{index}]"
        if not isinstance(lead, Mapping):
            raise ContractError(f"leads[{index}] must be an object")
        if set(lead) != {
            "externalLeadId", "identityBasis", "rank", "name", "category", "city",
            "address", "phone", "email", "instagramUrl", "websiteUrl", "googleMapsUrl",
            "googlePlaceId", "rating", "reviewCount", "websiteAssessment", "collectedAt",
        }:
            raise ContractError(f"{prefix} fields do not match the versioned contract")
        _bounded_string(lead.get("externalLeadId"), f"{prefix}.externalLeadId", 500)
        if lead["externalLeadId"] in seen_ids:
            raise ContractError("externalLeadId values must be unique in a batch")
        seen_ids.add(lead["externalLeadId"])
        if lead.get("identityBasis") not in IDENTITY_BASES:
            raise ContractError(f"{prefix}.identityBasis is invalid")
        rank = lead.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ContractError(f"{prefix}.rank must be positive")
        _bounded_string(lead.get("name"), f"{prefix}.name", 300)
        for field in ("category", "city", "address"):
            _bounded_string(lead.get(field), f"{prefix}.{field}", 500, nullable=True)
        for field in ("phone", "email"):
            _bounded_string(lead.get(field), f"{prefix}.{field}", 320, nullable=True)
        for field in ("instagramUrl", "websiteUrl", "googleMapsUrl"):
            _bounded_http_url(lead.get(field), f"{prefix}.{field}", nullable=True)
        _bounded_string(lead.get("googlePlaceId"), f"{prefix}.googlePlaceId", 500, nullable=True)
        rating = lead.get("rating")
        if rating is not None and (
            isinstance(rating, bool) or not isinstance(rating, (int, float)) or not 0 <= rating <= 5
        ):
            raise ContractError(f"{prefix}.rating must be between 0 and 5")
        reviews = lead.get("reviewCount")
        if reviews is not None and (
            isinstance(reviews, bool) or not isinstance(reviews, int) or reviews < 0
        ):
            raise ContractError(f"{prefix}.reviewCount must be non-negative")
        assessment = lead.get("websiteAssessment")
        if not isinstance(assessment, Mapping):
            raise ContractError(f"{prefix}.websiteAssessment must be an object")
        if set(assessment) != {"status", "hasSite", "siteQuality", "checkedAt", "finalUrl"}:
            raise ContractError(f"{prefix}.websiteAssessment fields do not match the contract")
        if assessment.get("status") not in WEBSITE_STATUSES:
            raise ContractError(f"{prefix}.websiteAssessment.status is invalid")
        if not isinstance(assessment.get("hasSite"), bool):
            raise ContractError(f"{prefix}.websiteAssessment.hasSite must be boolean")
        if assessment.get("siteQuality") not in SITE_QUALITIES:
            raise ContractError(f"{prefix}.websiteAssessment.siteQuality is invalid")
        _valid_time(assessment.get("checkedAt"), f"{prefix}.websiteAssessment.checkedAt")
        _bounded_http_url(
            assessment.get("finalUrl"),
            f"{prefix}.websiteAssessment.finalUrl",
            nullable=True,
        )
        _valid_time(lead.get("collectedAt"), f"{prefix}.collectedAt", nullable=True)


def serialize_payload(payload: Mapping[str, Any]) -> bytes:
    validate_payload(payload)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload_bytes: bytes) -> str:
    return hashlib.sha256(payload_bytes).hexdigest()


def idempotency_key(external_batch_id: str, digest: str) -> str:
    return f"lidogenerator:{external_batch_id}:{digest[:16]}"

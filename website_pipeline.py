"""Pure contracts for resolver modes, website audits, and qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from website_resolution import (
    CandidateEvidence,
    CandidateKind,
    CandidateSource,
    ResolutionStatus,
    WebsiteResolution,
    normalize_candidate_url,
)


class ResolverMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    STRICT = "strict"


def parse_resolver_mode(value: str) -> ResolverMode:
    if not isinstance(value, str):
        raise TypeError("resolver mode must be a string")
    normalized = value.strip().casefold()
    try:
        return ResolverMode(normalized)
    except ValueError as exc:
        raise ValueError(
            "WEBSITE_RESOLVER_MODE must be one of: off, shadow, strict"
        ) from exc


class WebsiteAuditStatus(str, Enum):
    NOT_RUN = "not_run"
    NO_OFFICIAL_SITE = "no_official_site"
    GOOD = "good"
    BAD = "bad"
    DEAD_CONFIRMED = "dead_confirmed"
    UNCERTAIN = "uncertain"
    TECHNICAL_ERROR = "technical_error"


class LeadDecision(str, Enum):
    LEAD = "lead"
    NOT_LEAD = "not_lead"
    UNCERTAIN = "uncertain"


def _normalized_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


@dataclass(frozen=True)
class WebsiteAuditResult:
    status: WebsiteAuditStatus
    audited_url: str | None
    final_url: str | None
    http_status: int | None
    evidence: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, WebsiteAuditStatus):
            raise TypeError("status must be a WebsiteAuditStatus")
        for name in ("audited_url", "final_url"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"{name} must be a string or None")
                if value != normalize_candidate_url(value):
                    raise ValueError(f"{name} must already be normalized")
        if self.final_url is not None and self.audited_url is None:
            raise ValueError("final_url requires audited_url")
        if self.http_status is not None:
            if type(self.http_status) is not int:
                raise TypeError("http_status must be an integer or None")
            if not 100 <= self.http_status <= 599:
                raise ValueError("http_status must be between 100 and 599")
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")
        evidence: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(self.evidence):
            normalized = _normalized_text(item, f"evidence[{index}]")
            key = normalized.casefold()
            if key in seen:
                raise ValueError("evidence must not contain duplicates")
            seen.add(key)
            evidence.append(normalized)
        object.__setattr__(self, "evidence", tuple(evidence))
        if self.error is not None:
            object.__setattr__(self, "error", _normalized_text(self.error, "error"))
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        if self.status in {
            WebsiteAuditStatus.NOT_RUN,
            WebsiteAuditStatus.NO_OFFICIAL_SITE,
        }:
            if any(
                value is not None
                for value in (self.audited_url, self.final_url, self.http_status, self.error)
            ):
                raise ValueError(f"{self.status.name} cannot include audit request data")
        elif self.status in {
            WebsiteAuditStatus.GOOD,
            WebsiteAuditStatus.BAD,
            WebsiteAuditStatus.DEAD_CONFIRMED,
        }:
            if self.audited_url is None:
                raise ValueError(f"{self.status.name} requires audited_url")
            if self.error is not None:
                raise ValueError(f"{self.status.name} cannot include error")
            if (
                self.status is WebsiteAuditStatus.DEAD_CONFIRMED
                and self.http_status not in {404, 410}
            ):
                raise ValueError("DEAD_CONFIRMED requires HTTP 404 or 410")
        elif self.status is WebsiteAuditStatus.TECHNICAL_ERROR:
            if self.audited_url is None or self.error is None:
                raise ValueError("TECHNICAL_ERROR requires audited_url and error")


@dataclass(frozen=True)
class LeadQualification:
    decision: LeadDecision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, LeadDecision):
            raise TypeError("decision must be a LeadDecision")
        object.__setattr__(self, "reason", _normalized_text(self.reason, "reason"))


def qualify_lead(
    *,
    has_actionable_contact: bool,
    resolution: WebsiteResolution,
    audit: WebsiteAuditResult,
) -> LeadQualification:
    if type(has_actionable_contact) is not bool:
        raise TypeError("has_actionable_contact must be a bool")
    if not isinstance(resolution, WebsiteResolution):
        raise TypeError("resolution must be a WebsiteResolution")
    if not isinstance(audit, WebsiteAuditResult):
        raise TypeError("audit must be a WebsiteAuditResult")
    if not has_actionable_contact:
        return LeadQualification(LeadDecision.NOT_LEAD, "contact_missing")

    if resolution.status is ResolutionStatus.FOUND_OFFICIAL:
        outcomes = {
            WebsiteAuditStatus.GOOD: (LeadDecision.NOT_LEAD, "official_site_good"),
            WebsiteAuditStatus.BAD: (LeadDecision.LEAD, "official_site_bad"),
            WebsiteAuditStatus.DEAD_CONFIRMED: (
                LeadDecision.LEAD,
                "official_site_dead_confirmed",
            ),
            WebsiteAuditStatus.UNCERTAIN: (
                LeadDecision.UNCERTAIN,
                "official_site_audit_uncertain",
            ),
            WebsiteAuditStatus.TECHNICAL_ERROR: (
                LeadDecision.UNCERTAIN,
                "official_site_audit_technical_error",
            ),
        }
        decision, reason = outcomes.get(
            audit.status,
            (LeadDecision.UNCERTAIN, "official_site_audit_contract_mismatch"),
        )
        return LeadQualification(decision, reason)

    if resolution.status in {ResolutionStatus.SOCIAL_ONLY, ResolutionStatus.NOT_FOUND}:
        if audit.status not in {
            WebsiteAuditStatus.NOT_RUN,
            WebsiteAuditStatus.NO_OFFICIAL_SITE,
        }:
            return LeadQualification(
                LeadDecision.UNCERTAIN,
                "website_resolution_audit_contract_mismatch",
            )
        reason = (
            "social_only"
            if resolution.status is ResolutionStatus.SOCIAL_ONLY
            else "official_site_not_found"
        )
        return LeadQualification(LeadDecision.LEAD, reason)
    if resolution.status is ResolutionStatus.UNCERTAIN:
        return LeadQualification(LeadDecision.UNCERTAIN, "website_resolution_uncertain")
    return LeadQualification(LeadDecision.UNCERTAIN, "website_resolution_error")


_EVIDENCE_KEYS = {
    "schema_version", "source", "candidate_url", "normalized_url",
    "normalized_domain", "final_domain", "kind", "matched_signals",
    "rejected_reason", "confidence", "technical_error",
}


def serialize_candidate_evidence(evidence: tuple[CandidateEvidence, ...]) -> str:
    if not isinstance(evidence, tuple):
        raise TypeError("evidence must be a tuple")
    payload = []
    for index, item in enumerate(evidence):
        if not isinstance(item, CandidateEvidence):
            raise TypeError(f"evidence[{index}] must be a CandidateEvidence")
        payload.append({
            "schema_version": 1,
            "source": item.source.value,
            "candidate_url": item.candidate_url,
            "normalized_url": item.normalized_url,
            "normalized_domain": item.normalized_domain,
            "final_domain": item.final_domain,
            "kind": item.kind.value,
            "matched_signals": list(item.matched_signals),
            "rejected_reason": item.rejected_reason,
            "confidence": item.confidence,
            "technical_error": item.technical_error,
        })
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deserialize_candidate_evidence(value: str) -> tuple[CandidateEvidence, ...]:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if not value:
        return ()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("candidate evidence is not valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("candidate evidence must be a JSON array")
    result: list[CandidateEvidence] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != _EVIDENCE_KEYS:
            raise ValueError(f"candidate evidence item {index} has an invalid schema")
        if item["schema_version"] != 1:
            raise ValueError("unsupported candidate evidence schema_version")
        if not isinstance(item["matched_signals"], list) or not all(
            isinstance(signal, str) for signal in item["matched_signals"]
        ):
            raise ValueError("matched_signals must be an array of strings")
        try:
            result.append(CandidateEvidence(
                source=CandidateSource(item["source"]),
                candidate_url=item["candidate_url"],
                normalized_url=item["normalized_url"],
                normalized_domain=item["normalized_domain"],
                final_domain=item["final_domain"],
                kind=CandidateKind(item["kind"]),
                matched_signals=tuple(item["matched_signals"]),
                rejected_reason=item["rejected_reason"],
                confidence=item["confidence"],
                technical_error=item["technical_error"],
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate evidence item {index}: {exc}") from exc
    return tuple(result)


def serialize_audit_evidence(evidence: tuple[str, ...]) -> str:
    validated = WebsiteAuditResult(
        WebsiteAuditStatus.UNCERTAIN, None, None, None, evidence
    ).evidence
    return json.dumps(validated, ensure_ascii=False, separators=(",", ":"))


def deserialize_audit_evidence(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if not value:
        return ()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("audit evidence is not valid JSON") from exc
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("audit evidence must be a JSON array of strings")
    return WebsiteAuditResult(
        WebsiteAuditStatus.UNCERTAIN, None, None, None, tuple(payload)
    ).evidence

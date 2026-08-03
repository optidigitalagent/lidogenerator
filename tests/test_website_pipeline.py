import dataclasses
import json
import unittest

from website_pipeline import (
    LeadDecision,
    ResolverMode,
    WebsiteAuditResult,
    WebsiteAuditStatus,
    deserialize_candidate_evidence,
    parse_resolver_mode,
    qualify_lead,
    serialize_candidate_evidence,
)
from website_resolution import (
    CandidateEvidence,
    CandidateKind,
    CandidateSource,
    ResolutionStatus,
    WebsiteResolution,
)


def evidence():
    return CandidateEvidence(
        CandidateSource.MAPS,
        "https://example.com/",
        "https://example.com/",
        "example.com",
        None,
        CandidateKind.OFFICIAL_WEBSITE,
        ("phone_exact",),
        confidence=0.75,
    )


def resolution(status):
    if status is ResolutionStatus.FOUND_OFFICIAL:
        item = evidence()
        return WebsiteResolution(status, item.normalized_url, item.source, 0.75, (item,))
    if status is ResolutionStatus.SOCIAL_ONLY:
        item = dataclasses.replace(
            evidence(),
            kind=CandidateKind.SOCIAL_PROFILE,
            rejected_reason="non_official_platform",
            confidence=0.0,
        )
        return WebsiteResolution(status, None, None, 0.0, (item,))
    if status is ResolutionStatus.RESOLUTION_ERROR:
        return WebsiteResolution(status, None, None, 0.0, (), "provider error")
    return WebsiteResolution(status, None, None, 0.0, ())


class PipelineContractTests(unittest.TestCase):
    def test_mode_parser(self):
        self.assertIs(parse_resolver_mode(" ShAdOw "), ResolverMode.SHADOW)
        with self.assertRaises(ValueError):
            parse_resolver_mode("enabled")
        with self.assertRaises(TypeError):
            parse_resolver_mode(None)

    def test_audit_validation_and_frozen(self):
        audit = WebsiteAuditResult(
            WebsiteAuditStatus.GOOD,
            "https://example.com/",
            "https://example.com/",
            200,
            ("quality:good",),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            audit.status = WebsiteAuditStatus.BAD
        with self.assertRaises(TypeError):
            WebsiteAuditResult(WebsiteAuditStatus.GOOD, "https://example.com/", None, True)
        with self.assertRaises(ValueError):
            WebsiteAuditResult(WebsiteAuditStatus.DEAD_CONFIRMED, "https://example.com/", None, 403)
        with self.assertRaises(ValueError):
            WebsiteAuditResult(WebsiteAuditStatus.TECHNICAL_ERROR, "https://example.com/", None, None)
        with self.assertRaises(ValueError):
            WebsiteAuditResult(WebsiteAuditStatus.NOT_RUN, "https://example.com/", None, None)

    def test_qualification_matrix(self):
        no_audit = WebsiteAuditResult(WebsiteAuditStatus.NO_OFFICIAL_SITE, None, None, None)
        rows = (
            (ResolutionStatus.SOCIAL_ONLY, no_audit, LeadDecision.LEAD, "social_only"),
            (ResolutionStatus.NOT_FOUND, no_audit, LeadDecision.LEAD, "official_site_not_found"),
            (ResolutionStatus.UNCERTAIN, no_audit, LeadDecision.UNCERTAIN, "website_resolution_uncertain"),
            (ResolutionStatus.RESOLUTION_ERROR, no_audit, LeadDecision.UNCERTAIN, "website_resolution_error"),
        )
        for status, audit, decision, reason in rows:
            with self.subTest(status=status):
                actual = qualify_lead(has_instagram=True, resolution=resolution(status), audit=audit)
                self.assertEqual((actual.decision, actual.reason), (decision, reason))

        found_rows = (
            (WebsiteAuditStatus.GOOD, LeadDecision.NOT_LEAD),
            (WebsiteAuditStatus.BAD, LeadDecision.LEAD),
            (WebsiteAuditStatus.DEAD_CONFIRMED, LeadDecision.LEAD),
            (WebsiteAuditStatus.UNCERTAIN, LeadDecision.UNCERTAIN),
            (WebsiteAuditStatus.TECHNICAL_ERROR, LeadDecision.UNCERTAIN),
            (WebsiteAuditStatus.NOT_RUN, LeadDecision.UNCERTAIN),
        )
        for status, decision in found_rows:
            kwargs = {}
            if status is WebsiteAuditStatus.DEAD_CONFIRMED:
                kwargs["http_status"] = 404
            if status is WebsiteAuditStatus.TECHNICAL_ERROR:
                kwargs["error"] = "timeout"
            audited_url = None if status is WebsiteAuditStatus.NOT_RUN else "https://example.com/"
            audit = WebsiteAuditResult(status, audited_url, None, kwargs.get("http_status"), error=kwargs.get("error"))
            self.assertIs(
                qualify_lead(has_instagram=True, resolution=resolution(ResolutionStatus.FOUND_OFFICIAL), audit=audit).decision,
                decision,
            )

    def test_no_instagram_always_not_lead(self):
        audit = WebsiteAuditResult(WebsiteAuditStatus.BAD, "https://example.com/", None, 200)
        actual = qualify_lead(
            has_instagram=False,
            resolution=resolution(ResolutionStatus.FOUND_OFFICIAL),
            audit=audit,
        )
        self.assertEqual((actual.decision, actual.reason), (LeadDecision.NOT_LEAD, "instagram_missing"))

    def test_incompatible_pair_is_uncertain(self):
        audit = WebsiteAuditResult(WebsiteAuditStatus.GOOD, "https://example.com/", None, 200)
        actual = qualify_lead(
            has_instagram=True,
            resolution=resolution(ResolutionStatus.NOT_FOUND),
            audit=audit,
        )
        self.assertIs(actual.decision, LeadDecision.UNCERTAIN)

    def test_evidence_json_is_deterministic_and_strict(self):
        serialized = serialize_candidate_evidence((evidence(),))
        self.assertEqual(serialized, serialize_candidate_evidence((evidence(),)))
        self.assertEqual(deserialize_candidate_evidence(serialized), (evidence(),))
        self.assertEqual(deserialize_candidate_evidence(""), ())
        payload = json.loads(serialized)
        payload[0]["schema_version"] = 2
        with self.assertRaises(ValueError):
            deserialize_candidate_evidence(json.dumps(payload))
        payload[0]["schema_version"] = 1
        payload[0]["unknown"] = True
        with self.assertRaises(ValueError):
            deserialize_candidate_evidence(json.dumps(payload))
        with self.assertRaises(ValueError):
            deserialize_candidate_evidence("not-json")


if __name__ == "__main__":
    unittest.main()

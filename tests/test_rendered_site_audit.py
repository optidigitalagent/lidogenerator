import unittest
from dataclasses import fields

from models import Business
from rendered_site_audit import (
    RenderedSignal,
    RenderedSiteAuditRuntime,
    RenderedSiteAuditResult,
    RenderedSiteAuditStatus,
    RenderedViewportMetrics,
    audit_rendered_sites_shadow,
    classify_rendered_metrics,
    rendered_audit_eligible,
    rendered_audit_url,
)


def metrics(**overrides):
    values = {
        "viewport_width": 390,
        "viewport_height": 844,
        "inner_width": 390,
        "document_scroll_width": 390,
        "body_scroll_width": 390,
        "visible_text_length": 150,
        "visible_image_count": 0,
        "broken_visible_image_count": 0,
        "major_overflow_element_count": 0,
        "tiny_text_element_count": 0,
        "sampled_text_element_count": 10,
        "viewport_meta_present": True,
        "dom_content_loaded_ms": 100,
    }
    values.update(overrides)
    return RenderedViewportMetrics(**values)


def desktop_metrics(**overrides):
    values = {
        "viewport_width": 1366,
        "viewport_height": 768,
        "inner_width": 1366,
        "document_scroll_width": 1366,
        "body_scroll_width": 1366,
    }
    values.update(overrides)
    return metrics(**values)


def business(quality="bad", host="example.com"):
    return Business(
        name="identity must not escape",
        website=f"https://{host}/",
        has_site=True,
        site_quality=quality,
        website_final_url=f"https://{host}/",
        website_audit_status=quality,
        website_audit_http_status=200,
    )


class RenderedVerdictTests(unittest.TestCase):
    def test_result_contract_has_no_identity_or_capture_fields(self):
        self.assertEqual(
            [field.name for field in fields(RenderedSiteAuditResult)],
            [
                "status",
                "desktop",
                "mobile",
                "signals",
                "pages_attempted",
                "pages_succeeded",
                "error_category",
            ],
        )

    def classify(self, mobile):
        return classify_rendered_metrics(
            desktop_metrics(), mobile, https_final_url=True
        )

    def test_responsive_site_is_strong_good(self):
        result = self.classify(metrics())
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_GOOD)
        self.assertIn(RenderedSignal.RESPONSIVE_LAYOUT_OK, result.signals)
        self.assertIn(RenderedSignal.CONTENT_VISIBLE, result.signals)

    def test_viewport_wide_is_strong_bad(self):
        result = self.classify(metrics(inner_width=501))
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_BAD)
        self.assertIn(RenderedSignal.MOBILE_LAYOUT_VIEWPORT_WIDE, result.signals)

    def test_overflow_plus_major_overflow_is_strong_bad(self):
        result = self.classify(metrics(document_scroll_width=449, major_overflow_element_count=2))
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_BAD)

    def test_missing_viewport_alone_is_uncertain_not_bad(self):
        result = self.classify(metrics(viewport_meta_present=False))
        self.assertEqual(result.status, RenderedSiteAuditStatus.UNCERTAIN)

    def test_missing_viewport_plus_overflow_is_strong_bad(self):
        result = self.classify(
            metrics(viewport_meta_present=False, document_scroll_width=449)
        )
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_BAD)

    def test_tiny_text_alone_is_uncertain(self):
        result = self.classify(metrics(tiny_text_element_count=3))
        self.assertEqual(result.status, RenderedSiteAuditStatus.UNCERTAIN)

    def test_broken_images_with_content_is_uncertain(self):
        result = self.classify(
            metrics(visible_image_count=50, broken_visible_image_count=17)
        )
        self.assertEqual(result.status, RenderedSiteAuditStatus.UNCERTAIN)

    def test_broken_images_without_content_is_strong_bad(self):
        result = self.classify(
            metrics(
                visible_text_length=149,
                visible_image_count=50,
                broken_visible_image_count=17,
            )
        )
        self.assertEqual(result.status, RenderedSiteAuditStatus.STRONG_BAD)

    def test_exact_threshold_edges(self):
        exact_overflow = self.classify(
            metrics(
                inner_width=400,
                document_scroll_width=460,
                body_scroll_width=460,
                major_overflow_element_count=2,
            )
        )
        self.assertNotIn(
            RenderedSignal.MOBILE_HORIZONTAL_OVERFLOW,
            exact_overflow.signals,
        )
        over = self.classify(
            metrics(
                inner_width=400,
                document_scroll_width=461,
                major_overflow_element_count=2,
            )
        )
        self.assertEqual(over.status, RenderedSiteAuditStatus.STRONG_BAD)

        responsive_edge = self.classify(
            metrics(inner_width=400, document_scroll_width=420, body_scroll_width=420)
        )
        self.assertIn(RenderedSignal.RESPONSIVE_LAYOUT_OK, responsive_edge.signals)
        self.assertIn(
            RenderedSignal.MOBILE_TINY_TEXT,
            self.classify(metrics(tiny_text_element_count=3, sampled_text_element_count=10)).signals,
        )
        self.assertIn(
            RenderedSignal.BROKEN_VISIBLE_IMAGES,
            self.classify(metrics(visible_image_count=50, broken_visible_image_count=17)).signals,
        )
        self.assertNotIn(
            RenderedSignal.CONTENT_VISIBLE,
            self.classify(metrics(visible_text_length=149)).signals,
        )
        self.assertIn(
            RenderedSignal.CONTENT_VISIBLE,
            self.classify(metrics(visible_text_length=150)).signals,
        )


class RenderedEligibilityTests(unittest.TestCase):
    def test_only_successful_real_good_or_bad_audits_are_eligible(self):
        for quality in ("good", "bad"):
            self.assertTrue(rendered_audit_eligible(business(quality)))
        rejected = [
            Business(has_site=False, site_quality="bad", website_audit_status="bad", website_final_url="https://example.com/"),
            Business(has_site=True, site_quality="dead", website_audit_status="dead_confirmed", website_final_url="https://example.com/"),
            Business(has_site=True, site_quality="uncertain", website_audit_status="uncertain", website_final_url="https://example.com/"),
            Business(has_site=True, site_quality="technical_error", website_audit_status="technical_error", website_final_url="https://example.com/"),
            Business(has_site=True, site_quality="bad", website_audit_status="bad", website_final_url="https://instagram.com/example"),
            Business(has_site=True, site_quality="bad", website_audit_status="bad", website_final_url="https://linktr.ee/example"),
            Business(has_site=True, site_quality="bad", website_audit_status="bad", website_final_url="https://prom.ua/example"),
            Business(has_site=True, site_quality="bad", website_audit_status="bad", website_final_url="https://locator.ua/example"),
        ]
        self.assertTrue(all(not rendered_audit_eligible(item) for item in rejected))

    def test_website_resolver_shadow_url_is_never_used(self):
        item = business("bad")
        item.website_final_url = ""
        item.website_resolved_url = "https://shadow-only.example/"
        item.website_resolution_status = "found_official"
        self.assertIsNone(rendered_audit_url(item))


class FakeAuditor:
    def __init__(self, result):
        self.result = result
        self.urls = []
        self.closed = False

    async def audit(self, url):
        self.urls.append(url)
        return self.result

    async def close(self):
        self.closed = True


class RenderedQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_and_good_quotas_are_independent_and_task_global(self):
        result = classify_rendered_metrics(desktop_metrics(), metrics(), https_final_url=True)
        auditor = FakeAuditor(result)
        runtime = RenderedSiteAuditRuntime(
            auditor,
            max_bad_audits=2,
            max_good_audits=1,
            concurrency=2,
        )
        first = await audit_rendered_sites_shadow(
            [business("bad", "bad1.example"), business("good", "good1.example")],
            runtime,
        )
        second = await audit_rendered_sites_shadow(
            [
                business("bad", "bad2.example"),
                business("bad", "bad3.example"),
                business("good", "good2.example"),
            ],
            runtime,
        )
        self.assertEqual((first.audited_bad_count, first.audited_good_count), (1, 1))
        self.assertEqual((second.audited_bad_count, second.audited_good_count), (1, 0))
        self.assertEqual(second.skipped_budget_count, 2)
        self.assertEqual(second.bad_budget.remaining_audits, 0)
        self.assertEqual(second.good_budget.remaining_audits, 0)
        self.assertEqual(len(auditor.urls), 3)

    async def test_telemetry_is_identity_free(self):
        result = classify_rendered_metrics(desktop_metrics(), metrics(), https_final_url=True)
        runtime = RenderedSiteAuditRuntime(
            FakeAuditor(result), max_bad_audits=1, max_good_audits=0, concurrency=1
        )
        summary = await runtime.audit_batch([business("bad", "secret.example")])
        payload = str(summary.telemetry(task_id=7)).casefold()
        self.assertNotIn("secret", payload)
        self.assertNotIn("example", payload)
        self.assertNotIn("identity", payload)


if __name__ == "__main__":
    unittest.main()

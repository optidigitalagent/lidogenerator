import json
import unittest
from unittest.mock import AsyncMock, patch

import config
import orchestrator
from models import Business
from rendered_site_audit import (
    RenderedSiteAuditRuntime,
    classify_rendered_metrics,
)
from tests.test_rendered_site_audit import desktop_metrics, metrics
from website_pipeline import ResolverMode


def audited_business(quality, host, *, phone="+380501112233"):
    return Business(
        name=f"private {quality} identity",
        website=f"https://{host}/",
        phone=phone,
        has_site=True,
        site_quality=quality,
        website_audit_status=quality,
        website_audit_http_status=200,
        website_final_url=f"https://{host}/",
    )


class MappingAuditor:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def audit(self, url):
        self.calls.append(url)
        if "legacy-bad" in url:
            return classify_rendered_metrics(
                desktop_metrics(), metrics(), https_final_url=True
            )
        return classify_rendered_metrics(
            desktop_metrics(), metrics(inner_width=501), https_final_url=True
        )

    async def close(self):
        self.closed = True


class FailingRuntime(RenderedSiteAuditRuntime):
    async def audit_batch(self, businesses):
        raise RuntimeError("secret.example should never be logged")


class RenderedShadowIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_shadow_verdicts_do_not_mutate_qualification_or_outputs(self):
        bad = audited_business("bad", "legacy-bad.test")
        good = audited_business("good", "legacy-good.test")
        businesses = [bad, good]
        before_dicts = [item.to_dict() for item in businesses]
        auditor = MappingAuditor()
        runtime = RenderedSiteAuditRuntime(
            auditor,
            max_bad_audits=1,
            max_good_audits=1,
            concurrency=2,
        )

        with (
            patch.object(config, "LEAD_CONTACTABILITY_MODE", "multi_channel"),
            patch.object(orchestrator.site_checker, "check_sites", new=AsyncMock(return_value=businesses)),
            patch.object(orchestrator.db, "save_businesses") as save_businesses,
            patch.object(orchestrator.reporter, "export_csv") as export_csv,
            patch.object(orchestrator.reporter, "export_excel") as export_excel,
            patch.object(orchestrator, "finalize_completed_task", new=AsyncMock()) as opti,
            self.assertLogs("lead_hunter.orchestrator", level="INFO") as captured,
        ):
            before_leads = [item.is_lead for item in businesses]
            await orchestrator._check_batch_websites_with_resolver_mode(
                businesses,
                ResolverMode.OFF,
                None,
                task_id=91,
                rendered_audit_runtime=runtime,
            )
            after_leads = [item.is_lead for item in businesses]

        self.assertEqual(before_leads, [True, False])
        self.assertEqual(after_leads, before_leads)
        self.assertEqual([item.to_dict() for item in businesses], before_dicts)
        save_businesses.assert_not_called()
        export_csv.assert_not_called()
        export_excel.assert_not_called()
        opti.assert_not_awaited()

        telemetry_line = next(
            line for line in captured.output if "rendered_site_audit_shadow {" in line
        )
        payload = json.loads(telemetry_line.split("rendered_site_audit_shadow ", 1)[1])
        self.assertEqual(payload["legacy_bad_rendered_strong_good"], 1)
        self.assertEqual(payload["legacy_good_rendered_strong_bad"], 1)
        serialized = json.dumps(payload).casefold()
        for forbidden in ("legacy-bad.test", "legacy-good.test", "private", "https://"):
            self.assertNotIn(forbidden, serialized)

    async def test_shadow_is_fail_open_and_failure_log_is_identity_free(self):
        runtime = FailingRuntime(
            MappingAuditor(), max_bad_audits=1, max_good_audits=1, concurrency=1
        )
        item = audited_business("bad", "secret.example")
        before = item.to_dict()
        with self.assertLogs("lead_hunter.orchestrator", level="WARNING") as captured:
            await orchestrator._run_rendered_site_audit_shadow(
                [item], runtime, task_id=12
            )
        self.assertEqual(item.to_dict(), before)
        self.assertEqual(len(captured.output), 1)
        line = captured.output[0]
        self.assertIn("task_id=12 exception_type=RuntimeError", line)
        self.assertNotIn("secret", line)
        self.assertNotIn("example", line)

    async def test_mode_off_path_performs_no_rendered_work(self):
        auditor = MappingAuditor()
        item = audited_business("bad", "legacy-bad.test")
        with patch.object(
            orchestrator.site_checker,
            "check_sites",
            new=AsyncMock(return_value=[item]),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [item],
                ResolverMode.OFF,
                None,
                task_id=22,
                rendered_audit_runtime=None,
            )
        self.assertEqual(auditor.calls, [])

    async def test_website_resolver_shadow_mutations_cannot_feed_rendered_audit(self):
        item = audited_business("bad", "real-site.test")
        auditor = MappingAuditor()
        runtime = RenderedSiteAuditRuntime(
            auditor, max_bad_audits=1, max_good_audits=0, concurrency=1
        )

        async def mutate_shadow(items, *, provider):
            items[0].website_resolved_url = "https://shadow-resolver.test/"
            items[0].website_final_url = "https://shadow-resolver.test/"

        with (
            patch.object(orchestrator.site_checker, "check_sites", new=AsyncMock(return_value=[item])),
            patch.object(orchestrator.website_resolver, "resolve_business_websites", new=mutate_shadow),
        ):
            await orchestrator._check_batch_websites_with_resolver_mode(
                [item],
                ResolverMode.SHADOW,
                None,
                task_id=23,
                rendered_audit_runtime=runtime,
            )
        self.assertEqual(auditor.calls, ["https://real-site.test/"])
        self.assertEqual(item.website_final_url, "https://real-site.test/")


if __name__ == "__main__":
    unittest.main()

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import db
from integrations.opti_contract import build_payload, stable_external_lead_id
from models import Business
from agents import reporter
from website_pipeline import (
    SearchMode,
    WebsiteAuditResult,
    WebsiteAuditStatus,
    LeadDecision,
    qualify_lead,
)
from website_resolution import ResolutionStatus
from test_website_pipeline import resolution


class AllCompaniesTests(unittest.TestCase):
    def test_qualification_matrix_is_independent_of_website_evidence(self):
        audit = WebsiteAuditResult(WebsiteAuditStatus.NOT_RUN, None, None, None)
        for status in ResolutionStatus:
            for contact in (True, False):
                with self.subTest(status=status, contact=contact):
                    result = qualify_lead(
                        has_actionable_contact=contact,
                        resolution=resolution(status),
                        audit=audit,
                        search_mode=SearchMode.ALL_COMPANIES,
                    )
                    self.assertEqual(
                        LeadDecision.LEAD,
                        result.decision,
                    )

    def test_all_companies_includes_phone_email_and_honest_missing_contacts(self):
        for website_policy in ["verified_no_site_only", "legacy_no_site_or_bad"]:
            for mode in ["instagram_only", "multi_channel"]:
                with (
                    patch.object(config, "LEAD_WEBSITE_POLICY", website_policy),
                    patch.object(config, "LEAD_CONTACTABILITY_MODE", mode),
                ):
                    for channels in [{"phone": "+380441234567"},
                                     {"email": "office@public-business.ua"}, {}]:
                        for quality in ["good", "none", "uncertain"]:
                            item = Business(search_mode="ALL_COMPANIES",
                                            site_quality=quality, **channels)
                            self.assertTrue(item.is_lead)
                            if not channels:
                                self.assertFalse(item.has_actionable_contact)
                                self.assertEqual(item.phone, "")
                                self.assertEqual(item.email, "")

    def test_real_export_keeps_good_website_under_legacy_global_policy(self):
        item = Business(
            name="Good",
            instagram_url="https://instagram.com/good_company",
            website="https://good.example/",
            site_quality="good",
            has_site=True,
            search_mode="ALL_COMPANIES",
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(config, "EXPORT_DIR", Path(directory)),
            patch.object(config, "LEAD_WEBSITE_POLICY", "verified_no_site_only"),
        ):
            path = reporter.export_csv([item], task_id=1)
            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle, delimiter=";"))
            self.assertEqual(len(rows), 2)
            self.assertIn("https://good.example/", rows[1])
            self.assertIn(
                "https://good.example/", reporter.format_leads_summary([item])
            )

    def test_durable_bridge_payload_and_identity_without_false_no_site(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(config, "DB_PATH", str(Path(directory) / "db.sqlite")),
        ):
            db.init_db()
            task_id = db.create_task("dentist", "Kyiv", 4, search_mode="ALL_COMPANIES")
            items = [
                Business(
                    name=quality,
                    google_place_id="place-" + quality,
                    instagram_url="https://instagram.com/company_" + quality,
                    site_quality=quality,
                    has_site=quality == "good",
                    website="https://good.example/" if quality == "good" else "",
                    search_mode="ALL_COMPANIES",
                    task_id=task_id,
                )
                for quality in ["good", "none", "uncertain"]
            ]
            items.append(
                Business(
                    name="confirmed",
                    google_place_id="place-confirmed",
                    instagram_url="https://instagram.com/confirmed",
                    search_mode="ALL_COMPANIES",
                    website_presence_status="absent_confirmed",
                    task_id=task_id,
                )
            )
            items.extend([
                Business(name="Phone only", phone="+380441234567",
                         google_place_id="phone-public", search_mode="ALL_COMPANIES",
                         site_quality="good", website="https://good.example/", task_id=task_id),
                Business(name="Email only", email="office@public-business.ua",
                         google_place_id="email-public", search_mode="ALL_COMPANIES", task_id=task_id),
                Business(name="No public contact", google_place_id="no-contact-public",
                         search_mode="ALL_COMPANIES", task_id=task_id),
            ])
            db.save_businesses(items)
            db.update_task_status(task_id, "done")
            db.init_db()
            restored = db.get_businesses_for_bridge(task_id)
            payload = build_payload(dict(db.get_task(task_id)), restored)
            self.assertEqual(
                [lead["websiteAssessment"]["status"] for lead in payload["leads"]],
                ["GOOD_SITE", "UNCERTAIN", "UNCERTAIN", "NO_SITE", "GOOD_SITE", "UNCERTAIN", "UNCERTAIN"],
            )
            self.assertEqual(len(payload["leads"]), 7)
            self.assertEqual(payload["leads"][-3]["phone"], "+380441234567")
            self.assertEqual(payload["leads"][-2]["email"], "office@public-business.ua")
            self.assertTrue(all(payload["leads"][-1][k] is None for k in ["phone", "email", "instagramUrl"]))
            self.assertTrue(all(item.is_lead for item in restored))
            self.assertEqual(
                stable_external_lead_id(items[0]), stable_external_lead_id(restored[0])
            )
            restored[0].search_mode = "WEBSITE_NEED"
            self.assertEqual(
                stable_external_lead_id(items[0]), stable_external_lead_id(restored[0])
            )

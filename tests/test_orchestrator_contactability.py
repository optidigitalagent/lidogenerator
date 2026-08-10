import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import config
import db
import orchestrator
from agents import social_checker
from contactability import ContactChannel
from models import Business


class MultiChannelRunSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_synthetic_multi_channel_search(self):
        candidates = [
            Business(name="Phone No Site", phone="050 123 45 67"),
            Business(
                name="Phone Bad Site",
                phone="050 234 56 78",
                site_quality="bad",
            ),
            Business(
                name="Phone Good Site",
                phone="050 345 67 89",
                site_quality="good",
            ),
            Business(name="Invalid Phone", phone="1111111"),
            Business(
                name="Instagram No Site",
                instagram_url="https://instagram.com/synthetic_direct/",
            ),
            Business(
                name="Recovered Instagram",
                website="https://synthetic-business.invalid/",
                website_final_url="https://synthetic-business.invalid/",
                site_quality="bad",
            ),
        ]
        collector_calls = []
        social_targets = []
        messages = []

        async def fake_collect_stream(
            niche,
            city,
            max_businesses=None,
            progress_callback=None,
            stop_flag=None,
            **kwargs,
        ):
            collector_calls.append((niche, city, max_businesses))
            for index in range(len(candidates)):
                if progress_callback:
                    await progress_callback(index + 1)
            yield candidates

        async def fake_check_sites(items, progress_callback=None):
            return items

        async def fake_first_party_apply(items, resolver, *, task_id):
            for business in items:
                if business.name == "Recovered Instagram":
                    business.instagram_url = (
                        "https://www.instagram.com/recovered_synthetic/"
                    )

        async def fake_social(items, progress_callback=None, stop_flag=None):
            social_targets.extend(item.name for item in items if item.instagram_url)
            return items

        async def progress(message):
            messages.append(message)

        async def fake_finalize(task_id):
            return ""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(config, "DB_PATH", str(root / "contactability.db")),
                patch.object(config, "EXPORT_DIR", root / "exports"),
                patch.object(config, "LEAD_CONTACTABILITY_MODE", "multi_channel"),
                patch.object(config, "WEBSITE_RESOLVER_MODE", "off"),
                patch.object(config, "INSTAGRAM_FIRST_PARTY_MODE", "apply"),
                patch.object(config, "MAX_CHECKED_CANDIDATES_PER_TASK", 20),
                patch.object(config, "MAX_MAPS_CARDS_PER_TASK", 20),
                patch.object(config, "OPENAI_SCORING_ENABLED", False),
                patch.object(
                    orchestrator.collector,
                    "collect_stream",
                    new=fake_collect_stream,
                ),
                patch.object(
                    orchestrator.site_checker,
                    "check_sites",
                    new=fake_check_sites,
                ),
                patch.object(
                    orchestrator,
                    "_run_first_party_instagram_apply",
                    new=fake_first_party_apply,
                ),
                patch.object(
                    orchestrator.social_checker,
                    "check_instagram",
                    new=fake_social,
                ),
                patch.object(
                    orchestrator,
                    "finalize_completed_task",
                    new=fake_finalize,
                ),
                self.assertLogs("lead_hunter.orchestrator", level="INFO") as logs,
            ):
                db.init_db()
                task_id = db.create_task("synthetic", "Test City", 4)
                result = await orchestrator.run_search(
                    task_id,
                    progress_callback=progress,
                    progress_interval=0,
                    first_party_resolver_runtime=object(),
                )

            self.assertTrue(result.endswith(".xlsx"))
            self.assertEqual(len(collector_calls), 1)
            with patch.object(config, "DB_PATH", str(root / "contactability.db")):
                stored = db.get_businesses(task_id)
            self.assertEqual(
                {business.name for business in stored},
                {
                    "Phone No Site",
                    "Phone Bad Site",
                    "Instagram No Site",
                    "Recovered Instagram",
                },
            )
            phone_no_site = next(
                business for business in stored if business.name == "Phone No Site"
            )
            self.assertEqual(phone_no_site.phone, "050 123 45 67")
            self.assertEqual(
                social_targets,
                ["Instagram No Site", "Recovered Instagram"],
            )
            recovered = next(
                business for business in stored if business.name == "Recovered Instagram"
            )
            self.assertIs(
                recovered.preferred_contact_channel,
                ContactChannel.INSTAGRAM,
            )

            csv_path = next((root / "exports").glob("*.csv"))
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(len(rows), 4)
            phone_row = next(row for row in rows if row["Business Name"] == "Phone No Site")
            self.assertEqual(phone_row["Phone"], "0501234567")
            self.assertEqual(phone_row["Preferred Contact"], "phone")
            self.assertEqual(phone_row["Instagram URL"], "")

            summary = messages[-1]
            self.assertIn("Телефон: 0501234567", summary)
            self.assertIn(
                "Instagram: https://www.instagram.com/recovered_synthetic/",
                summary,
            )
            self.assertNotIn("Instagram: \n", summary)
            progress_text = "\n".join(messages)
            self.assertIn("Пропущено без доступного контакту: 1", progress_text)

            telemetry_line = next(
                line for line in logs.output if "contactability_qualification" in line
            )
            payload = json.loads(telemetry_line.split("contactability_qualification ", 1)[1])
            self.assertEqual(payload["candidate_count"], 6)
            self.assertEqual(payload["lead_phone_only_count"], 2)
            self.assertEqual(payload["lead_instagram_count"], 2)
            self.assertEqual(payload["no_contact_count"], 1)
            self.assertEqual(payload["good_site_excluded_count"], 1)

    async def test_social_checker_accepts_phone_only_without_network_work(self):
        business = Business(name="Phone Only", phone="0501234567")
        result = await social_checker.check_instagram([business])
        self.assertEqual(result, [business])


if __name__ == "__main__":
    unittest.main()

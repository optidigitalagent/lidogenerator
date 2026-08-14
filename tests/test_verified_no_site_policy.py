import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import config
import db
import orchestrator
from agents import reporter
from models import Business
from website_candidate_matching import SearchResult


def candidate(name, *, instagram=True, website="", city="Харків"):
    suffix = name.casefold().replace(" ", ".")
    return Business(
        name=name,
        niche="стоматологія",
        city=city,
        address=f"{name} address",
        phone=f"+38050{abs(hash(name + city)) % 10000000:07d}",
        instagram_url=f"https://instagram.com/{suffix}/" if instagram else "",
        website=website,
        google_place_id=f"place-{name}-{city}",
    )


class PresenceProvider:
    def __init__(self):
        self.calls = []

    async def search(self, request):
        self.calls.append(request.business_name)
        if request.business_name == "D":
            return (
                SearchResult(
                    "https://d-hidden.ua/",
                    "D Харків",
                    request.phone or "",
                    1,
                ),
            )
        return ()


class VerifiedPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_policy_matrix_and_reporter_guard(self) -> None:
        with patch.object(config, "LEAD_WEBSITE_POLICY", "verified_no_site_only"):
            lead = candidate("Lead")
            lead.website_presence_status = "absent_confirmed"
            self.assertTrue(lead.is_lead)
            for excluded in (
                candidate("Bad", website="https://bad.example/"),
                candidate("Dead", website="https://dead.example/"),
                candidate("Google", website="https://sites.google.com/view/google"),
                candidate("No IG", instagram=False),
            ):
                excluded.website_presence_status = "absent_confirmed"
                self.assertFalse(excluded.is_lead)
            for status in ("present", "uncertain", "technical_error", ""):
                excluded = candidate(status)
                excluded.website_presence_status = status
                self.assertFalse(excluded.is_lead)

            stale = candidate("Stale")
            stale.lead_decision = "lead"
            stale.website_presence_status = "present"
            stale.website_presence_resolved_url = "https://stale.example/"
            with tempfile.TemporaryDirectory() as directory, patch.object(
                reporter.config, "EXPORT_DIR", Path(directory)
            ):
                path = reporter.export_csv([lead, stale], task_id=1)
                with open(path, encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.reader(handle, delimiter=";"))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1][7], "")

    async def test_two_tasks_skip_checked_before_provider_and_different_city_rechecks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = PresenceProvider()
            batches = {
                "value": [
                    candidate("A"),
                    candidate("B", website="https://bad.example/"),
                    candidate("C", instagram=False),
                    candidate("D"),
                ]
            }

            async def collect_stream(*args, progress_callback=None, **kwargs):
                items = [copy.deepcopy(item) for item in batches["value"]]
                if progress_callback:
                    await progress_callback(len(items))
                yield items

            async def no_op(*args, **kwargs):
                return args[0] if args else None

            def export(*args, **kwargs):
                return str(Path(directory) / "leads.csv")

            patches = (
                patch.object(config, "DB_PATH", str(Path(directory) / "verified.db")),
                patch.object(config, "EXPORT_DIR", Path(directory)),
                patch.object(config, "LEAD_WEBSITE_POLICY", "verified_no_site_only"),
                patch.object(config, "LEAD_CONTACTABILITY_MODE", "instagram_only"),
                patch.object(config, "CANDIDATE_HISTORY_MODE", "apply"),
                patch.object(config, "WEBSITE_PRESENCE_VERIFICATION_MODE", "apply"),
                patch.object(config, "WEBSITE_RESOLVER_MODE", "off"),
                patch.object(config, "INSTAGRAM_FIRST_PARTY_MODE", "off"),
                patch.object(config, "RENDERED_SITE_AUDIT_MODE", "off"),
                patch.object(config, "MAX_CHECKED_CANDIDATES_PER_TASK", 100),
                patch.object(config, "MAX_MAPS_CARDS_PER_TASK", 1000),
                patch.object(orchestrator.collector, "collect_stream", side_effect=collect_stream),
                patch.object(orchestrator.social_checker, "check_instagram", side_effect=no_op),
                patch.object(orchestrator.ai_scorer, "score_businesses", side_effect=no_op),
                patch.object(orchestrator.reporter, "export_csv", side_effect=export),
                patch.object(orchestrator.reporter, "export_excel", side_effect=export),
                patch.object(orchestrator, "finalize_completed_task", new=AsyncMock(return_value="")),
            )
            entered = []
            try:
                for item in patches:
                    entered.append(item)
                    item.start()
                db.init_db()
                first = db.create_task("стоматологія", "Харків", 10)
                await orchestrator.run_search(
                    first,
                    progress_interval=0,
                    website_presence_search_provider=provider,
                )
                self.assertEqual([item.name for item in db.get_businesses(first)], ["A"])
                self.assertEqual(provider.calls, ["A", "D"])

                batches["value"] = [
                    candidate("A"), candidate("B", website="https://bad.example/"),
                    candidate("C", instagram=False), candidate("D"),
                    candidate("E"), candidate("F"),
                ]
                second = db.create_task("стоматология", "Харків", 10)
                await orchestrator.run_search(
                    second,
                    progress_interval=0,
                    website_presence_search_provider=provider,
                )
                self.assertEqual(
                    {item.name for item in db.get_businesses(second)}, {"E", "F"}
                )
                self.assertEqual(provider.calls, ["A", "D", "E", "F"])
                progress = json.loads(db.get_task(second)["progress_json"])
                self.assertEqual(progress["checkedCandidates"], 2)
                self.assertEqual(progress["skippedPreviouslyChecked"], 4)

                batches["value"] = [candidate("A", city="Одеса")]
                third = db.create_task("стоматологія", "Одеса", 10)
                await orchestrator.run_search(
                    third,
                    progress_interval=0,
                    website_presence_search_provider=provider,
                )
                self.assertEqual(provider.calls[-1], "A")
                self.assertEqual([item.name for item in db.get_businesses(third)], ["A"])
            finally:
                for item in reversed(entered):
                    item.stop()


if __name__ == "__main__":
    unittest.main()

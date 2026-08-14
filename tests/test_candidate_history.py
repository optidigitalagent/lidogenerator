from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import candidate_history
import config
import db
from candidate_history import CandidateClaimResult
from models import Business


class CandidateHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "history.db")
        self.patch = patch.object(config, "DB_PATH", self.db_path)
        self.patch.start()
        db.init_db()
        self.scope = candidate_history.canonical_scope_key("стоматология", "Харків")
        self.business = Business(
            name="Secret Dental",
            city="Харків",
            address="Secret Address 7",
            phone="+380501234567",
            instagram_url="https://instagram.com/secret.dental/",
        )
        self.basis, self.key = candidate_history.candidate_fingerprint(
            self.business, "Харків"
        )

    def tearDown(self) -> None:
        self.patch.stop()
        self.temp.cleanup()

    def test_canonical_scope_aliases_and_boundaries(self) -> None:
        self.assertEqual(
            self.scope,
            candidate_history.canonical_scope_key("стоматологія", "Харків"),
        )
        self.assertNotEqual(
            self.scope,
            candidate_history.canonical_scope_key("стоматологія", "Одеса"),
        )
        self.assertNotEqual(
            self.scope,
            candidate_history.canonical_scope_key("барбершоп", "Харків"),
        )

    def test_fingerprint_is_deterministic_and_not_local_id(self) -> None:
        self.business.id = 1
        first = candidate_history.candidate_fingerprint(self.business, "Харків")
        self.business.id = 999
        second = candidate_history.candidate_fingerprint(self.business, "Харків")
        self.assertEqual(first, second)
        self.assertTrue(first[1].startswith("cand:v1:phone:"))

    def test_claim_checked_restart_and_different_scope(self) -> None:
        self.assertEqual(
            candidate_history.claim_candidate(self.scope, self.key, self.basis, 1),
            CandidateClaimResult.CLAIMED,
        )
        self.assertTrue(
            candidate_history.mark_candidate_checked(self.scope, self.key, 1, "lead")
        )
        db.init_db()
        self.assertEqual(
            candidate_history.claim_candidate(self.scope, self.key, self.basis, 2),
            CandidateClaimResult.ALREADY_CHECKED,
        )
        other = candidate_history.canonical_scope_key("стоматологія", "Одеса")
        self.assertEqual(
            candidate_history.claim_candidate(other, self.key, self.basis, 2),
            CandidateClaimResult.CLAIMED,
        )

    def test_active_other_claim_expired_takeover_and_release(self) -> None:
        now = datetime.now(timezone.utc)
        candidate_history.claim_candidate(
            self.scope, self.key, self.basis, 1, lease_seconds=60, now=now
        )
        self.assertEqual(
            candidate_history.claim_candidate(
                self.scope, self.key, self.basis, 2, lease_seconds=60, now=now
            ),
            CandidateClaimResult.CLAIMED_BY_OTHER_TASK,
        )
        self.assertEqual(
            candidate_history.claim_candidate(
                self.scope,
                self.key,
                self.basis,
                2,
                lease_seconds=60,
                now=now + timedelta(seconds=61),
            ),
            CandidateClaimResult.CLAIMED,
        )
        self.assertEqual(candidate_history.release_unfinished_candidate_claims(2), 1)

    def test_concurrent_claim_is_single_owner(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda task: candidate_history.claim_candidate(
                        self.scope, self.key, self.basis, task
                    ),
                    (1, 2),
                )
            )
        self.assertEqual(results.count(CandidateClaimResult.CLAIMED), 1)
        self.assertEqual(
            results.count(CandidateClaimResult.CLAIMED_BY_OTHER_TASK), 1
        )

    def test_table_contains_no_raw_identity(self) -> None:
        candidate_history.claim_candidate(self.scope, self.key, self.basis, 1)
        raw = Path(self.db_path).read_bytes()
        for secret in (b"Secret Dental", b"Secret Address", b"secret.dental"):
            self.assertNotIn(secret, raw)

    def test_backfill_is_idempotent(self) -> None:
        task = db.create_task("стоматологія", "Харків", 1)
        persisted = Business(**self.business.to_dict())
        persisted.task_id = task
        db.save_businesses([persisted])
        db.init_db()
        db.init_db()
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT state,last_outcome FROM candidate_history WHERE last_outcome='legacy_persisted_lead'"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(rows, [("checked", "legacy_persisted_lead")])

    def test_business_presence_fields_persist_and_update(self) -> None:
        task = db.create_task("стоматологія", "Харків", 1)
        persisted = Business(**self.business.to_dict())
        persisted.task_id = task
        persisted.website_presence_status = "absent_confirmed"
        persisted.website_presence_source = "web_search"
        persisted.website_presence_evidence = '["web_no_site_match"]'
        db.save_businesses([persisted])
        loaded = db.get_businesses(task)[0]
        self.assertEqual(loaded.website_presence_status, "absent_confirmed")
        loaded.website_presence_status = "present"
        loaded.website_presence_source = "maps"
        loaded.website_presence_resolved_url = "https://example.com/"
        db.update_business(loaded)
        reloaded = db.get_businesses(task)[0]
        self.assertEqual(reloaded.website_presence_status, "present")
        self.assertEqual(reloaded.website_presence_resolved_url, "https://example.com/")


if __name__ == "__main__":
    unittest.main()

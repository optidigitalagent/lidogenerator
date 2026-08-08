import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import db
from integrations import opti_outbox
from integrations.opti_bridge import finalize_completed_task, reconcile_completed_tasks
from integrations.opti_contract import build_payload, serialize_payload
from models import Business


class ReconciliationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp.name) / "test.db"
        self.db_patch = patch.object(config, "DB_PATH", str(self.database_path))
        self.db_patch.start()
        db.init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def _completed_task(self, *, count=50, name="Final lead"):
        task_id = db.create_task("dentistry", "Kyiv", count)
        db.save_businesses(
            [
                Business(
                    task_id=task_id,
                    name=name,
                    city="Kyiv",
                    phone=f"+380{task_id}",
                    instagram_url=f"lead{task_id}",
                    site_quality="none",
                )
            ]
        )
        db.update_task_status(task_id, "done", csv_path="exports/result.xlsx")
        return task_id

    def test_new_task_has_contract_marker(self):
        task_id = db.create_task("dentistry", "Kyiv", 50)
        self.assertEqual(
            "opti.lead-import.v1",
            db.get_task(task_id)["opti_sync_contract_version"],
        )

    def test_done_eligible_task_is_enqueued_once_and_rebuild_is_identical(self):
        task_id = self._completed_task()
        task = dict(db.get_task(task_id))
        expected = serialize_payload(
            build_payload(task, db.get_businesses_for_bridge(task_id))
        )

        with patch.object(opti_outbox, "deliver_due") as deliver:
            first = reconcile_completed_tasks()
            second = reconcile_completed_tasks()

        self.assertEqual({"examined": 1, "enqueued": 1, "errors": 0}, first)
        self.assertEqual({"examined": 0, "enqueued": 0, "errors": 0}, second)
        row = opti_outbox.get_by_batch(task["external_batch_id"])
        self.assertEqual(expected, row["payloadJson"].encode("utf-8"))
        self.assertEqual(1, sum(opti_outbox.summary_counts().values()))
        deliver.assert_not_called()

    def test_legacy_done_task_is_not_reconciled(self):
        task_id = self._completed_task()
        with db._connect() as conn:
            conn.execute(
                "UPDATE tasks SET opti_sync_contract_version = NULL WHERE id = ?",
                (task_id,),
            )
        self.assertEqual(0, reconcile_completed_tasks()["examined"])
        self.assertEqual(0, sum(opti_outbox.summary_counts().values()))

    def test_malformed_task_does_not_block_the_next_task(self):
        bad_task_id = self._completed_task(count=0, name="Bad")
        good_task_id = self._completed_task(count=50, name="Good")
        summary = reconcile_completed_tasks()
        self.assertEqual({"examined": 2, "enqueued": 1, "errors": 1}, summary)
        self.assertIsNone(
            opti_outbox.get_by_batch(db.get_task(bad_task_id)["external_batch_id"])
        )
        self.assertIsNotNone(
            opti_outbox.get_by_batch(db.get_task(good_task_id)["external_batch_id"])
        )

    def test_existing_sent_row_is_untouched(self):
        task_id = self._completed_task()
        reconcile_completed_tasks()
        batch_id = db.get_task(task_id)["external_batch_id"]
        with db._connect() as conn:
            conn.execute(
                "UPDATE opti_sync_outbox SET status = 'SENT', sentAt = updatedAt "
                "WHERE externalBatchId = ?",
                (batch_id,),
            )
        before = opti_outbox.get_by_batch(batch_id)
        self.assertEqual(0, reconcile_completed_tasks()["examined"])
        self.assertEqual(before, opti_outbox.get_by_batch(batch_id))

    async def test_finalize_twice_is_idempotent(self):
        task_id = self._completed_task()
        with patch.object(config, "OPTI_BRIDGE_ENABLED", False):
            first = await finalize_completed_task(task_id)
            second = await finalize_completed_task(task_id)
        self.assertIn("Opti sync pending", first)
        self.assertIn("/sync", first)
        self.assertEqual(first, second)
        self.assertEqual(1, sum(opti_outbox.summary_counts().values()))

    def test_crash_gap_is_recovered_without_http(self):
        task_id = self._completed_task()
        self.assertEqual(0, sum(opti_outbox.summary_counts().values()))
        with patch.object(opti_outbox, "deliver_due") as deliver:
            reconcile_completed_tasks()
        self.assertIsNotNone(
            opti_outbox.get_by_batch(db.get_task(task_id)["external_batch_id"])
        )
        deliver.assert_not_called()


class LegacyMigrationTests(unittest.TestCase):
    def test_migrated_legacy_task_marker_remains_null(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "legacy.db"
            conn = sqlite3.connect(database_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        niche TEXT NOT NULL,
                        city TEXT NOT NULL,
                        count INTEGER NOT NULL,
                        status TEXT NOT NULL DEFAULT 'new',
                        chat_id INTEGER,
                        csv_path TEXT,
                        created_at TEXT NOT NULL,
                        finished_at TEXT
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO tasks (niche, city, count, status, created_at, finished_at) "
                    "VALUES ('legacy', 'Kyiv', 50, 'done', "
                    "'2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z')"
                )
                conn.commit()
            finally:
                conn.close()
            with patch.object(config, "DB_PATH", str(database_path)):
                db.init_db()
                task = db.get_task(1)
            self.assertIsNotNone(task["external_batch_id"])
            self.assertIsNone(task["opti_sync_contract_version"])


if __name__ == "__main__":
    unittest.main()

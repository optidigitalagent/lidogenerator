import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import db
from models import Business


NEW_FIELDS = {
    "website_original_url", "instagram_bio_url", "website_resolved_url",
    "website_final_url", "website_resolution_status", "website_resolution_source",
    "website_resolution_confidence", "website_resolution_evidence",
    "website_resolution_error", "website_audit_status",
    "website_audit_http_status", "website_audit_evidence", "website_audit_error",
    "lead_decision", "lead_decision_reason",
}


class DatabaseWebsiteResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "test.db")
        self.patch = patch.object(db.config, "DB_PATH", self.db_path)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def columns(self):
        conn = sqlite3.connect(self.db_path)
        try:
            return {row[1] for row in conn.execute("PRAGMA table_info(businesses)")}
        finally:
            conn.close()

    def test_fresh_schema_and_idempotence(self):
        db.init_db()
        db.init_db()
        self.assertTrue(NEW_FIELDS.issubset(self.columns()))

    def test_additive_legacy_migration_and_old_row_read(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript("""
                CREATE TABLE tasks (
                    id INTEGER PRIMARY KEY, niche TEXT NOT NULL, city TEXT NOT NULL,
                    count INTEGER NOT NULL, status TEXT NOT NULL, chat_id INTEGER,
                    csv_path TEXT, created_at TEXT NOT NULL, finished_at TEXT
                );
                CREATE TABLE businesses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id INTEGER,
                    name TEXT, niche TEXT, city TEXT, phone TEXT, address TEXT,
                    website TEXT, instagram_url TEXT, rating REAL DEFAULT 0,
                    reviews_count INTEGER DEFAULT 0, has_site INTEGER DEFAULT 0,
                    site_quality TEXT DEFAULT 'none', instagram_active INTEGER DEFAULT 0,
                    followers INTEGER DEFAULT 0, posts_count INTEGER DEFAULT 0,
                    last_post_days INTEGER, ai_score INTEGER DEFAULT 0,
                    ai_priority TEXT DEFAULT '', ai_reason TEXT DEFAULT ''
                );
                INSERT INTO businesses(task_id,name,city) VALUES(1,'Legacy','Kyiv');
            """)
            conn.commit()
        finally:
            conn.close()
        db.init_db()
        rows = db.get_businesses(1)
        self.assertEqual(rows[0].name, "Legacy")
        self.assertEqual(rows[0].website_resolution_status, "")

    def test_save_update_and_unicode_evidence_round_trip(self):
        db.init_db()
        business = Business(
            task_id=1,
            name="Салон",
            city="Київ",
            website="https://old.example/",
            website_original_url="https://old.example/",
            website_resolved_url="https://new.example/",
            website_resolution_status="found_official",
            website_resolution_source="web_search",
            website_resolution_confidence=0.75,
            website_resolution_evidence='[{"тест":"✓"}]',
            website_audit_status="bad",
            website_audit_http_status=200,
            website_audit_evidence='["quality:bad"]',
            lead_decision="lead",
            lead_decision_reason="official_site_bad",
        )
        self.assertEqual(db.save_businesses([business]), 1)
        business.website = "https://changed.example/"
        business.website_audit_error = ""
        db.update_business(business)
        actual = db.get_businesses(1)[0]
        self.assertEqual(actual.website, "https://changed.example/")
        self.assertEqual(actual.website_resolution_evidence, '[{"тест":"✓"}]')
        self.assertEqual(actual.lead_decision_reason, "official_site_bad")


if __name__ == "__main__":
    unittest.main()

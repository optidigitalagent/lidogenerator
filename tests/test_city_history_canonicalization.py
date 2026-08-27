import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import candidate_history
import city_catalog
import config
import db
from models import Business

UTC = timezone.utc


class CityAliasContractTests(unittest.TestCase):
    def test_required_city_aliases_resolve_to_one_identity(self) -> None:
        cases = {
            "lviv": ("Львів", "Львов", "Lviv", "Lvov"),
            "kharkiv": ("Харків", "Харьков", "Kharkiv", "Kharkov"),
            "dnipro": ("Дніпро", "Днепр", "Dnipro", "Dnepr"),
            "zaporizhzhia": (
                "Запоріжжя",
                "Запорожье",
                "Zaporizhzhia",
                "Zaporozhye",
            ),
            "odesa": ("Одеса", "Одесса", "Odesa", "Odessa"),
            "kryvyi_rih": (
                "Кривий Ріг",
                "Кривой Рог",
                "Kryvyi Rih",
                "Krivoy Rog",
            ),
        }
        for expected_key, aliases in cases.items():
            resolved = [
                city_catalog.resolve_city(value, city_catalog.CITY_DEFINITIONS)
                for value in aliases
            ]
            with self.subTest(city=expected_key):
                self.assertTrue(all(city is resolved[0] for city in resolved))
                self.assertEqual(resolved[0].key, expected_key)

    def test_production_city_terms_have_no_alias_collisions(self) -> None:
        index = city_catalog.build_city_index(city_catalog.CITY_DEFINITIONS)
        expected_terms = sum(
            1 + len(city.aliases) for city in city_catalog.CITY_DEFINITIONS
        )
        self.assertEqual(len(index), expected_terms)

    def test_production_observed_additions_are_unambiguous(self) -> None:
        self.assertEqual(
            city_catalog.resolve_city(
                "Хмельницкий", city_catalog.CITY_DEFINITIONS
            ).key,
            "khmelnytskyi",
        )
        self.assertEqual(
            city_catalog.resolve_city("виница", city_catalog.CITY_DEFINITIONS).key,
            "vinnytsia",
        )

    def test_cities_other_than_kyiv_and_lviv_have_no_district_discovery(self) -> None:
        for city in city_catalog.CITY_DEFINITIONS[2:]:
            with self.subTest(city=city.key):
                self.assertEqual(city.districts, ())


class CanonicalHistoryScopeTests(unittest.TestCase):
    def test_lviv_niche_and_city_aliases_share_one_scope(self) -> None:
        scopes = {
            candidate_history.canonical_scope_key("стоматология", "Львів"),
            candidate_history.canonical_scope_key("стоматология", "Львов"),
            candidate_history.canonical_scope_key("стоматологія", "Lviv"),
        }
        self.assertEqual(len(scopes), 1)
        expected = candidate_history._scope_key("dentistry", "lviv")
        self.assertEqual(scopes, {expected})

    def test_basic_city_alias_scopes_are_canonical(self) -> None:
        cases = (
            ("Харків", "Харьков"),
            ("Дніпро", "Днепр"),
            ("Запоріжжя", "Запорожье"),
            ("Одеса", "Одесса"),
        )
        for ukrainian, russian in cases:
            with self.subTest(city=ukrainian):
                self.assertEqual(
                    candidate_history.canonical_scope_key(
                        "стоматология", ukrainian
                    ),
                    candidate_history.canonical_scope_key(
                        "стоматологія", russian
                    ),
                )

    def test_scope_boundaries_remain_distinct(self) -> None:
        dentistry_kyiv = candidate_history.canonical_scope_key(
            "стоматология", "Київ"
        )
        dentistry_lviv = candidate_history.canonical_scope_key(
            "стоматология", "Львів"
        )
        barbershop_lviv = candidate_history.canonical_scope_key(
            "барбершоп", "Львів"
        )
        self.assertNotEqual(dentistry_kyiv, dentistry_lviv)
        self.assertNotEqual(dentistry_lviv, barbershop_lviv)

    def test_unknown_city_keeps_deterministic_raw_scope(self) -> None:
        first = candidate_history.canonical_scope_key(
            "стоматология", "  Невідоме   Місто "
        )
        second = candidate_history.legacy_raw_city_scope_key(
            "стоматологія", "невідоме місто"
        )
        self.assertEqual(first, second)

    def test_kyiv_scope_hash_is_unchanged(self) -> None:
        self.assertEqual(
            candidate_history.canonical_scope_key("стоматология", "Київ"),
            "scope:v1:f1c47639cc8b40c3f8e68f9ac3da582df0141c3d4c1c3cc3110db1a372911a08",
        )


class CityHistoryReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "history.db")
        self.patch = patch.object(config, "DB_PATH", self.db_path)
        self.patch.start()
        db.init_db()
        self.now = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)

    def tearDown(self) -> None:
        self.patch.stop()
        self.temp.cleanup()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _add_task(self, city: str, niche: str = "стоматология") -> int:
        return db.create_task(niche, city, 1)

    def _candidate(self, suffix: int, city: str = "Львів") -> tuple[str, str]:
        return candidate_history.candidate_fingerprint(
            {"phone": f"+38050000{suffix:04d}"}, city
        )

    def _insert_history(
        self,
        conn: sqlite3.Connection,
        scope: str,
        candidate: tuple[str, str],
        *,
        state: str = "checked",
        task_id: int = 1,
        claim_expires_at: str | None = None,
        first_seen_at: str = "2026-08-20T10:00:00Z",
        last_seen_at: str = "2026-08-21T10:00:00Z",
        checked_at: str | None = "2026-08-21T10:00:00Z",
        last_outcome: str | None = "lead",
        times_seen: int = 1,
    ) -> None:
        basis, key = candidate
        claimed_by = task_id if state == "claimed" else None
        expires = (
            claim_expires_at
            if state == "claimed"
            else None
        )
        checked = checked_at if state == "checked" else None
        outcome = last_outcome if state == "checked" else None
        conn.execute(
            "INSERT INTO candidate_history "
            "(scope_key,candidate_key,identity_basis,state,claimed_by_task_id,"
            "claim_expires_at,first_seen_at,last_seen_at,checked_at,last_outcome,"
            "times_seen) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                scope,
                key,
                basis,
                state,
                claimed_by,
                expires,
                first_seen_at,
                last_seen_at,
                checked,
                outcome,
                times_seen,
            ),
        )

    def _rows(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM candidate_history ORDER BY scope_key,candidate_key"
        ).fetchall()

    def test_lviv_ua_ru_and_english_raw_scopes_move_to_canonical(self) -> None:
        aliases = ("Львів", "Львов", "Lviv")
        for city in aliases:
            self._add_task(city)
        conn = self._connect()
        try:
            for index, city in enumerate(aliases, start=1):
                self._insert_history(
                    conn,
                    candidate_history.legacy_raw_city_scope_key(
                        "стоматология", city
                    ),
                    self._candidate(index),
                )
            conn.commit()
            result = candidate_history.reconcile_city_alias_scopes(
                conn, now=self.now
            )
            rows = self._rows(conn)
        finally:
            conn.close()
        canonical = candidate_history.canonical_scope_key(
            "стоматология", "Львів"
        )
        self.assertEqual(result.scopes_reconciled, 2)
        self.assertEqual(result.rows_moved, 2)
        self.assertEqual({row["scope_key"] for row in rows}, {canonical})
        self.assertEqual(len(rows), 3)

    def test_same_candidate_across_raw_aliases_merges_once(self) -> None:
        candidate = self._candidate(10)
        for city in ("Львів", "Львов"):
            self._add_task(city)
        conn = self._connect()
        try:
            for city in ("Львів", "Львов"):
                self._insert_history(
                    conn,
                    candidate_history.legacy_raw_city_scope_key(
                        "стоматология", city
                    ),
                    candidate,
                    times_seen=2,
                )
            conn.commit()
            result = candidate_history.reconcile_city_alias_scopes(
                conn, now=self.now
            )
            rows = self._rows(conn)
        finally:
            conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["times_seen"], 4)
        self.assertEqual(result.rows_moved, 1)
        self.assertEqual(result.rows_merged, 1)

    def test_checked_dominates_claimed_and_clears_active_claim(self) -> None:
        candidate = self._candidate(11)
        self._add_task("Львов")
        old_scope = candidate_history.legacy_raw_city_scope_key(
            "стоматология", "Львов"
        )
        canonical = candidate_history.canonical_scope_key(
            "стоматология", "Львов"
        )
        conn = self._connect()
        try:
            self._insert_history(
                conn,
                old_scope,
                candidate,
                state="claimed",
                task_id=8,
                claim_expires_at="2026-08-26T13:00:00Z",
                times_seen=3,
            )
            self._insert_history(
                conn,
                canonical,
                candidate,
                state="checked",
                first_seen_at="2026-08-19T10:00:00Z",
                last_seen_at="2026-08-22T10:00:00Z",
                checked_at="2026-08-22T10:00:00Z",
                last_outcome="not_lead",
                times_seen=5,
            )
            conn.commit()
            candidate_history.reconcile_city_alias_scopes(conn, now=self.now)
            row = self._rows(conn)[0]
        finally:
            conn.close()
        self.assertEqual(row["state"], "checked")
        self.assertIsNone(row["claimed_by_task_id"])
        self.assertIsNone(row["claim_expires_at"])
        self.assertEqual(row["last_outcome"], "not_lead")
        self.assertEqual(row["times_seen"], 8)

    def test_checked_collision_merges_timestamps_count_and_latest_outcome(self) -> None:
        candidate = self._candidate(12)
        self._add_task("Львов")
        old_scope = candidate_history.legacy_raw_city_scope_key(
            "стоматология", "Львов"
        )
        canonical = candidate_history.canonical_scope_key(
            "стоматология", "Львов"
        )
        conn = self._connect()
        try:
            self._insert_history(
                conn,
                old_scope,
                candidate,
                first_seen_at="2026-08-18T10:00:00Z",
                last_seen_at="2026-08-23T10:00:00Z",
                checked_at="2026-08-20T10:00:00Z",
                last_outcome="old_outcome",
                times_seen=4,
            )
            self._insert_history(
                conn,
                canonical,
                candidate,
                first_seen_at="2026-08-19T10:00:00Z",
                last_seen_at="2026-08-24T10:00:00Z",
                checked_at="2026-08-22T10:00:00Z",
                last_outcome="newer_outcome",
                times_seen=7,
            )
            conn.commit()
            candidate_history.reconcile_city_alias_scopes(conn, now=self.now)
            row = self._rows(conn)[0]
        finally:
            conn.close()
        self.assertEqual(row["first_seen_at"], "2026-08-18T10:00:00Z")
        self.assertEqual(row["last_seen_at"], "2026-08-24T10:00:00Z")
        self.assertEqual(row["checked_at"], "2026-08-20T10:00:00Z")
        self.assertEqual(row["last_outcome"], "newer_outcome")
        self.assertEqual(row["times_seen"], 11)

    def test_active_claim_collision_preserves_one_deterministic_owner(self) -> None:
        candidate = self._candidate(13)
        self._add_task("Львов")
        old_scope = candidate_history.legacy_raw_city_scope_key(
            "стоматология", "Львов"
        )
        canonical = candidate_history.canonical_scope_key(
            "стоматология", "Львов"
        )
        conn = self._connect()
        try:
            self._insert_history(
                conn,
                old_scope,
                candidate,
                state="claimed",
                task_id=20,
                claim_expires_at="2026-08-26T12:30:00Z",
                times_seen=2,
            )
            self._insert_history(
                conn,
                canonical,
                candidate,
                state="claimed",
                task_id=21,
                claim_expires_at="2026-08-26T13:00:00Z",
                times_seen=3,
            )
            conn.commit()
            candidate_history.reconcile_city_alias_scopes(conn, now=self.now)
            rows = self._rows(conn)
        finally:
            conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "claimed")
        self.assertEqual(rows[0]["claimed_by_task_id"], 21)
        self.assertEqual(rows[0]["claim_expires_at"], "2026-08-26T13:00:00Z")
        self.assertEqual(rows[0]["times_seen"], 5)

    def test_second_reconciliation_is_a_no_op(self) -> None:
        self._add_task("Львов")
        conn = self._connect()
        try:
            self._insert_history(
                conn,
                candidate_history.legacy_raw_city_scope_key(
                    "стоматология", "Львов"
                ),
                self._candidate(14),
            )
            conn.commit()
            first = candidate_history.reconcile_city_alias_scopes(
                conn, now=self.now
            )
            snapshot = [tuple(row) for row in self._rows(conn)]
            second = candidate_history.reconcile_city_alias_scopes(
                conn, now=self.now
            )
            repeated = [tuple(row) for row in self._rows(conn)]
        finally:
            conn.close()
        self.assertEqual(first.old_rows_deleted, 1)
        self.assertEqual(
            second,
            candidate_history.HistoryScopeReconciliation(),
        )
        self.assertEqual(repeated, snapshot)

    def test_backfill_after_migration_and_repeated_startup_stay_canonical(self) -> None:
        city = "Львов"
        task = self._add_task(city)
        business = Business(
            task_id=task,
            name="History fixture",
            city=city,
            phone="+380500000015",
        )
        db.save_businesses([business])
        candidate = candidate_history.candidate_fingerprint(business, "Львів")
        old_scope = candidate_history.legacy_raw_city_scope_key(
            "стоматология", city
        )
        conn = self._connect()
        try:
            self._insert_history(
                conn,
                old_scope,
                candidate,
                state="claimed",
                task_id=task,
                claim_expires_at="2026-08-26T13:00:00Z",
            )
            conn.commit()
        finally:
            conn.close()
        db.init_db()
        db.init_db()
        conn = self._connect()
        try:
            rows = self._rows(conn)
        finally:
            conn.close()
        canonical = candidate_history.canonical_scope_key(
            "стоматология", city
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scope_key"], canonical)
        self.assertEqual(rows[0]["state"], "checked")
        self.assertEqual(rows[0]["last_outcome"], "legacy_persisted_lead")

    def test_kharkiv_and_zaporizhzhia_alias_scopes_migrate(self) -> None:
        cases = (("Харьков", 16), ("Запорожье", 17))
        for city, _ in cases:
            self._add_task(city)
        conn = self._connect()
        try:
            for city, suffix in cases:
                self._insert_history(
                    conn,
                    candidate_history.legacy_raw_city_scope_key(
                        "стоматология", city
                    ),
                    self._candidate(suffix, city),
                )
            conn.commit()
            candidate_history.reconcile_city_alias_scopes(conn, now=self.now)
            scopes = {row["scope_key"] for row in self._rows(conn)}
        finally:
            conn.close()
        self.assertEqual(
            scopes,
            {
                candidate_history.canonical_scope_key(
                    "стоматология", "Харків"
                ),
                candidate_history.canonical_scope_key(
                    "стоматология", "Запоріжжя"
                ),
            },
        )

    def test_identity_basis_corruption_rolls_back_whole_reconciliation(self) -> None:
        self._add_task("Львов")
        candidate = self._candidate(18)
        old_scope = candidate_history.legacy_raw_city_scope_key(
            "стоматология", "Львов"
        )
        canonical = candidate_history.canonical_scope_key(
            "стоматология", "Львов"
        )
        conn = self._connect()
        try:
            self._insert_history(conn, old_scope, candidate)
            self._insert_history(conn, canonical, candidate)
            conn.execute(
                "UPDATE candidate_history SET identity_basis='maps_url' "
                "WHERE scope_key=?",
                (old_scope,),
            )
            conn.commit()
            before = [tuple(row) for row in self._rows(conn)]
            with self.assertRaisesRegex(ValueError, "identity basis"):
                candidate_history.reconcile_city_alias_scopes(conn, now=self.now)
            after = [tuple(row) for row in self._rows(conn)]
        finally:
            conn.close()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()

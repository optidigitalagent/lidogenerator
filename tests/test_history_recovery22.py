"""Real SQLite/runtime regressions for qualified surplus and retryable history."""

import asyncio
import copy
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import candidate_history as h
import config
import db
import orchestrator
from models import Business
from query_planner import build_query_plan
from search_runtime import SearchRuntime


def candidate(i, instagram=True):
    return Business(
        name=f"Fixture clinic {i}",
        niche="стоматологія",
        city="Вінниця",
        address=f"Fixture street {i}",
        phone=f"+38050{i:07d}",
        google_place_id=f"r22-{i}",
        instagram_url=f"https://instagram.com/fixture_{i}/" if instagram else "",
    )


class HistoryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stack = ExitStack()
        for k, v in {
            "DB_PATH": str(Path(self.temp.name) / "history.sqlite"),
            "EXPORT_DIR": Path(self.temp.name) / "exports",
            "CANDIDATE_HISTORY_MODE": "apply",
            "DEEP_DISCOVERY_MODE": "off",
            "LEAD_WEBSITE_POLICY": "verified_no_site_only",
            "LEAD_CONTACTABILITY_MODE": "instagram_only",
            "WEBSITE_PRESENCE_VERIFICATION_MODE": "apply",
            "WEBSITE_RESOLVER_MODE": "off",
            "INSTAGRAM_FIRST_PARTY_MODE": "off",
            "RENDERED_SITE_AUDIT_MODE": "off",
            "OPTI_BRIDGE_ENABLED": False,
        }.items():
            self.stack.enter_context(patch.object(config, k, v))
        self.stack.enter_context(
            patch.object(
                orchestrator.social_checker, "check_instagram", new=AsyncMock()
            )
        )
        self.stack.enter_context(
            patch.object(orchestrator.ai_scorer, "score_businesses", new=AsyncMock())
        )
        db.init_db()
        self.scope = h.canonical_scope_key("стоматологія", "винница")

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def history(self):
        with sqlite3.connect(config.DB_PATH) as c:
            return c.execute(
                "select candidate_key,state,last_outcome from candidate_history order by candidate_key"
            ).fetchall()

    async def run_task(
        self, items, target=1, save_failure=False, stop_on=None, areas_only=False
    ):
        runtime = SearchRuntime(3)

        async def source(*args, progress_callback=None, **kwargs):
            batch = items
            if areas_only:
                batch = items if "Вишенька" in kwargs["query_text"] else [candidate(99)]
                batch = [
                    b
                    for b in batch
                    if kwargs["discovery_session"].claim_link(
                        "https://maps.google.com/?cid=" + b.google_place_id
                    )
                ]
            for _ in batch:
                kwargs["discovery_session"].record_card_opened()
            if progress_callback:
                await progress_callback(len(batch))
            yield copy.deepcopy(batch)

        class Provider:
            async def search(self, request):
                if request.business_name == stop_on:
                    runtime.stop(task)
                return ()

        async def runner(task_id, **kwargs):
            return await orchestrator.run_search(
                task_id,
                website_presence_search_provider=Provider(),
                progress_interval=0,
                **kwargs,
            )

        task = db.create_task("стоматологія", "винница", target, initiated_via="OPTI")
        with (
            patch.object(orchestrator.collector, "collect_stream", new=source),
            ExitStack() as s,
        ):
            if save_failure:
                s.enter_context(
                    patch.object(
                        db,
                        "save_businesses",
                        side_effect=RuntimeError("fixture persistence unavailable"),
                    )
                )
            runtime.launch(task, run_search=runner)
            for _ in range(1000):
                if runtime.active_count == 0:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(0, runtime.active_count)
            await runtime.shutdown()
        return task

    async def test_qualified_surplus_is_not_permanently_checked(self):
        items = [candidate(i) for i in [1, 2, 3]]
        first = await self.run_task(items, target=1)
        self.assertEqual(1, len(db.get_businesses(first)))
        self.assertEqual(
            1,
            len(self.history()),
            "Only committed selected lead may become terminal history",
        )
        second = await self.run_task(items, target=2)
        self.assertEqual(
            2,
            len(db.get_businesses(second)),
            "Later ordinary search must recover both surplus candidates",
        )
        self.assertEqual(3, len(self.history()))

    async def test_persistence_failure_releases_all_qualified_claims(self):
        task = await self.run_task([candidate(1), candidate(2)], save_failure=True)
        self.assertEqual("error", db.get_task(task)["status"])
        self.assertEqual([], db.get_businesses(task))
        self.assertEqual(
            [],
            self.history(),
            "Uncommitted lead must not become a permanent history veto",
        )
        recovered = await self.run_task([candidate(1)], target=1)
        self.assertEqual(1, len(db.get_businesses(recovered)))

    async def test_error_after_one_qualified_candidate_does_not_strand_it(self):
        calls = 0

        async def verify(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("controlled dependency failure")
            from website_presence import (
                WebsitePresenceResult,
                WebsitePresenceSource,
                WebsitePresenceStatus,
            )

            return WebsitePresenceResult(
                status=WebsitePresenceStatus.ABSENT_CONFIRMED,
                source=WebsitePresenceSource.WEB_SEARCH,
                evidence=("fixture_no_official_site",),
                requests_used=1,
            )

        with patch.object(
            orchestrator.website_presence_verifier,
            "verify_business_website_presence",
            side_effect=verify,
        ):
            task = await self.run_task([candidate(1), candidate(2)], target=3)
        self.assertEqual("error", db.get_task(task)["status"])
        self.assertEqual([], self.history())

    async def test_new_instagram_evidence_reopens_only_contact_exclusion_and_audits_prior(
        self,
    ):
        b = candidate(1)
        basis, key = h.candidate_fingerprint(b, "Вінниця")
        h.claim_candidate(self.scope, key, basis, 1)
        h.mark_candidate_checked(self.scope, key, 1, "no_instagram")
        task = await self.run_task([b], target=1)
        self.assertEqual(1, len(db.get_businesses(task)))
        with sqlite3.connect(config.DB_PATH) as c:
            prior = c.execute(
                "select prior_history_json,reason from candidate_history_rechecks"
            ).fetchone()
            self.assertEqual("no_instagram", json.loads(prior[0])["last_outcome"])
            self.assertEqual("instagram_now_available", prior[1])
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("delete from candidate_history_rechecks")

    async def test_uncertain_is_rechecked_but_valid_exclusions_stay_terminal(self):
        items = [candidate(i) for i in range(1, 6)]
        for b, outcome in zip(
            items,
            [
                "website_uncertain",
                "website_present_maps",
                "no_instagram",
                "lead",
                "legacy_persisted_lead",
            ],
        ):
            basis, key = h.candidate_fingerprint(b, "Вінниця")
            h.claim_candidate(self.scope, key, basis, 1)
            h.mark_candidate_checked(self.scope, key, 1, outcome)
        items[2].instagram_url = ""
        task = await self.run_task(items, target=3)
        self.assertEqual(
            ["Fixture clinic 1"], [b.name for b in db.get_businesses(task)]
        )

    async def test_recheck_stop_error_restores_previous_history(self):
        b = candidate(1)
        basis, key = h.candidate_fingerprint(b, "Вінниця")
        h.claim_candidate(self.scope, key, basis, 1)
        h.mark_candidate_checked(self.scope, key, 1, "no_instagram")
        self.assertEqual(
            h.CandidateClaimResult.CLAIMED,
            h.claim_candidate(self.scope, key, basis, 2, instagram_url=b.instagram_url),
        )
        self.assertTrue(h.release_candidate_claim(self.scope, key, 2))
        self.assertEqual([(key, "checked", "no_instagram")], self.history())
        self.assertEqual(
            h.CandidateClaimResult.ALREADY_CHECKED,
            h.claim_candidate(self.scope, key, basis, 2, instagram_url=b.instagram_url),
            "Same task cannot retry the recheck after release",
        )

    def test_city_continuation_is_distinct_bounded_and_in_selected_city(self):
        import city_catalog

        city = city_catalog.resolve_city("винница", city_catalog.CITY_DEFINITIONS)
        plan = build_query_plan(
            "стоматологія",
            city.canonical_name,
            niche_variants=("стоматологічна клініка",),
            local_areas=city.local_areas,
        )
        self.assertEqual(3, plan.deep_queries_total)
        self.assertTrue(all("Вінниця" in q.text for q in plan.deep_queries.queries))
        self.assertTrue(
            all(
                q.key not in {n.key for n in plan.normal_queries.queries}
                for q in plan.deep_queries.queries
            )
        )

    async def test_stop_preserves_capped_partial_and_releases_surplus(self):
        task = await self.run_task(
            [candidate(1), candidate(2)], target=1, stop_on="Fixture clinic 2"
        )
        self.assertEqual("stopped", db.get_task(task)["status"])
        self.assertEqual(1, len(db.get_businesses(task)))
        self.assertEqual(1, len(self.history()))
        later = await self.run_task([candidate(1), candidate(2)], target=1)
        self.assertEqual(
            ["Fixture clinic 2"], [b.name for b in db.get_businesses(later)]
        )

    async def test_stop_after_persistence_does_not_insert_partial_twice(self):
        b = candidate(1)
        b.phone = ""
        with patch.object(
            orchestrator.social_checker,
            "check_instagram",
            side_effect=orchestrator.SearchStopped(),
        ):
            task = await self.run_task([b], target=1)
        self.assertEqual("stopped", db.get_task(task)["status"])
        self.assertEqual(1, len(db.get_businesses(task)))
        self.assertEqual(1, len(self.history()))

    def test_business_and_history_rollback_together_if_claim_is_lost(self):
        task = db.create_task("стоматологія", "винница", 2)
        items = [candidate(1), candidate(2)]
        claims = {}
        for b in items:
            b.task_id = task
            basis, key = h.candidate_fingerprint(b, "Вінниця")
            h.claim_candidate(self.scope, key, basis, task)
            claims[id(b)] = key
        h.release_candidate_claim(self.scope, claims[id(items[1])], task)
        with self.assertRaises(RuntimeError):
            db.save_businesses(items, history_scope=self.scope, history_claims=claims)
        self.assertEqual([], db.get_businesses(task))
        self.assertTrue(all(b.id is None for b in items))
        self.assertEqual("claimed", self.history()[0][1])

    def test_crash_expiry_and_concurrent_recheck_keep_single_owner(self):
        b = candidate(1)
        basis, key = h.candidate_fingerprint(b, "Вінниця")
        now = datetime.now(timezone.utc)
        h.claim_candidate(self.scope, key, basis, 1, lease_seconds=60, now=now)
        self.assertEqual(
            h.CandidateClaimResult.CLAIMED_BY_OTHER_TASK,
            h.claim_candidate(self.scope, key, basis, 2, now=now),
        )
        self.assertEqual(
            h.CandidateClaimResult.CLAIMED,
            h.claim_candidate(
                self.scope, key, basis, 2, now=now + timedelta(seconds=61)
            ),
        )
        h.mark_candidate_checked(self.scope, key, 2, "website_uncertain")
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda task: h.claim_candidate(
                        self.scope, key, basis, task, instagram_url=b.instagram_url
                    ),
                    [3, 4],
                )
            )
        self.assertEqual(1, results.count(h.CandidateClaimResult.CLAIMED))
        self.assertEqual(1, results.count(h.CandidateClaimResult.CLAIMED_BY_OTHER_TASK))

    def test_additive_audit_schema_old_reader_and_snapshot_restore(self):
        b = candidate(1)
        basis, key = h.candidate_fingerprint(b, "Вінниця")
        h.claim_candidate(self.scope, key, basis, 1)
        h.mark_candidate_checked(self.scope, key, 1, "website_uncertain")
        with sqlite3.connect(config.DB_PATH) as c:
            c.execute("drop trigger candidate_history_rechecks_no_update")
            c.execute("drop trigger candidate_history_rechecks_no_delete")
            c.execute("drop table candidate_history_rechecks")
            before = c.execute("select * from candidate_history").fetchall()
        db.init_db()
        with (
            sqlite3.connect(config.DB_PATH) as c,
            sqlite3.connect(":memory:") as restored,
        ):
            self.assertEqual(
                before, c.execute("select * from candidate_history").fetchall()
            )
            # The unchanged old history reader sees its original columns/state.
            self.assertEqual(
                "checked",
                c.execute(
                    "select state from candidate_history where scope_key=? and candidate_key=?",
                    (self.scope, key),
                ).fetchone()[0],
            )
            c.backup(restored)
            self.assertEqual(
                "ok", restored.execute("pragma integrity_check").fetchone()[0]
            )
            self.assertEqual(
                0,
                restored.execute(
                    "select count(*) from candidate_history_rechecks"
                ).fetchone()[0],
            )

    def test_history_integrity_failure_rolls_back_business_insert(self):
        task = db.create_task("стоматологія", "винница", 1)
        b = candidate(1)
        b.task_id = task
        basis, key = h.candidate_fingerprint(b, "Вінниця")
        h.claim_candidate(self.scope, key, basis, task)
        with sqlite3.connect(config.DB_PATH) as c:
            c.execute(
                "create trigger fixture_history_failure before update on candidate_history begin select raise(ABORT,'controlled history failure'); end"
            )
        with self.assertRaises(sqlite3.IntegrityError):
            db.save_businesses(
                [b], history_scope=self.scope, history_claims={id(b): key}
            )
        self.assertEqual([], db.get_businesses(task))
        self.assertIsNone(b.id)

    async def test_real_runtime_continues_after_normal_exhaustion_to_new_area(self):
        b = candidate(99)
        basis, key = h.candidate_fingerprint(b, "Вінниця")
        h.claim_candidate(self.scope, key, basis, 1)
        h.mark_candidate_checked(self.scope, key, 1, "website_present_maps")
        with patch.object(config, "DEEP_DISCOVERY_MODE", "apply"):
            task = await self.run_task([candidate(1)], areas_only=True)
        saved = db.get_businesses(task)
        self.assertEqual(1, len(saved))
        self.assertEqual("absent_confirmed", saved[0].website_presence_status)
        progress = json.loads(db.get_task(task)["progress_json"])
        self.assertTrue(progress["deepPhaseActivated"])
        self.assertEqual(7, progress["normalQueriesCompleted"])
        self.assertEqual(3, progress["deepQueriesTotal"])
        self.assertEqual(1, progress["websitePresenceRequestsUsed"])
